# evaluate_steerability_custom.py
import os, json, argparse, gc, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from einops import einsum
from time import time

from model_helper.config import load_config
from model_helper.model_io import load_tokenizer_and_model

# ----------------- A/B token helpers -----------------
# def get_ab_token_ids(tokenizer):
#     """Return single-token ids for 'A' and 'B' under the current prompt formatting."""
#     a = tokenizer.encode("A", add_special_tokens=False)
#     b = tokenizer.encode("B", add_special_tokens=False)
#     if len(a) == 1 and len(b) == 1:
#         return a[0], b[0]
#     # Fallback: leading space form (if your prompt ends with a space)
#     a = tokenizer.encode(" A", add_special_tokens=False)
#     b = tokenizer.encode(" B", add_special_tokens=False)
#     assert len(a) == 1 and len(b) == 1, "Ensure the next token is a single 'A'/'B'. Adjust prompt tail if needed."
#     return a[0], b[0]

def get_ab_token_ids_or_fail(tokenizer):
    ids_A = tokenizer.encode("A", add_special_tokens=False)
    ids_B = tokenizer.encode("B", add_special_tokens=False)
    if len(ids_A) == 1 and len(ids_B) == 1:
        return ids_A[0], ids_B[0], "A"
    # fallback: leading-space variant
    ids_A = tokenizer.encode(" A", add_special_tokens=False)
    ids_B = tokenizer.encode(" B", add_special_tokens=False)
    assert len(ids_A) == 1 and len(ids_B) == 1, "A/B must be single tokens; adjust prompt tail."
    return ids_A[0], ids_B[0], " B"

def extract_pos_letter(rec: dict) -> str:
    """From the dataset row, return 'A' or 'B' that matches the target trait (y+)."""
    amb = rec.get("answer_matching_behavior", "")
    return "A" if "A" in amb else "B"

@torch.no_grad()
def verify_one_token_via_logits(ds_engine, tokenizer, device, rows, format_prompt_fn, k=24):
    # same subset on every rank
    subset = rows[:min(k, len(rows))]
    total = 0; ok = 0; bad_samples = []
    for ex in subset:
        prompt = format_prompt_fn(ex["question"])
        inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = ds_engine(**inp, return_dict=True)  # no autocast here
        next_logits = out.logits[0, -1, :]
        next_id = int(next_logits.argmax().item())
        text = tokenizer.decode([next_id], skip_special_tokens=True).strip()
        total += 1
        if text in ("A", "B"):
            ok += 1
        else:
            toks = tokenizer.convert_ids_to_tokens([next_id])
            bad_samples.append({"decoded": text, "toks": toks, "question_tail": ex["question"][-80:]})
    return ok, total, bad_samples

    

# --- helper: pick the A/B choice token index for one sequence of ids (used in SV building) ---
# def _choice_token_index_for_ids(tokenizer, ids):
#     toks = tokenizer.convert_ids_to_tokens(ids)
#     print("TOKENS" , toks)
#     # Search from the end for a clean "A" or "B" piece (strip leading-space markers)
#     for j in range(len(toks)-1, -1, -1):
#         t = toks[j]
#         plain = t.replace("Ġ", "").replace("▁", "").replace("Ċ", "")
#         if plain in ("A", "B"):
#             return j
#     # Fallback: if last token is punctuation/paren/newline, grab previous
#     if len(toks) >= 2:
#         last_plain = toks[-1].replace("Ġ","").replace("▁","").replace("Ċ","")
#         if last_plain in (")", "]", "}", ".", "!", "?", "", "\n"):
#             return len(toks) - 2
#     return max(len(toks) - 1, 0)

# --- helper: compute choice indices for a whole batch tensor [B, L] ---
# def _batch_choice_indices(tokenizer, ids_batch_cpu):
#     choice_idx = []
#     for row in ids_batch_cpu.tolist():
#         choice_idx.append(_choice_token_index_for_ids(tokenizer, row))
#     return choice_idx


def make_prompt_for_eval(q: str, tail: str = "Answer:") -> str:
    q = q.rstrip()
    return (
        f"{q}\n\n"
        "Reply with a single character ONLY: A or B.\n"
        "Do not include parentheses, words, punctuation, spaces, or newlines.\n"
        "Answer:"
    )

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)

    # model/runtime
    ap.add_argument("--config", required=True, help="Path to YAML for 70B")
    ap.add_argument("--layer", type=int, required=True, help="Layer index to intervene")
    ap.add_argument("--normalize", action="store_true", help="L2 normalize steering vector before use")
    ap.add_argument("--batch_size", type=int, default=4, help="batch size for SV build (prompt+completion pairs)")

    # SINGLE DATASET + split
    ap.add_argument("--dataset", required=True, help="jsonl with question/answer_matching_behavior/answer_not_matching_behavior")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.4)
    ap.add_argument("--val_frac", type=float, default=0.1)   # test is 1 - train - val

    # steering vector construction control
    ap.add_argument("--sv_max_items", type=int, default=1024)

    # evaluation control
    ap.add_argument("--multipliers", type=str, default="-1.5,-1.0,-0.5,0,0.5,1.0,1.5")
    ap.add_argument("--max_items", type=int, default=0)  # max eval items (on test split)
    ap.add_argument("--system_prompt", type=str, default="Answer:")
    ap.add_argument("--verify_outputs", action="store_true",help="Generate 1 token to verify outputs are strictly A/B during train & test.")
    ap.add_argument("--verify_k", type=int, default=24,help="Number of random examples to verify in each split.")
    return ap.parse_args()



# ----------------- Steering hook -----------------
def make_lastpos_pre_hook(steering_vec, scale=1.0, normalize=False, norm_scale=False):
    calls = {"n": 0}  # <-- assert hook is called
    def hook_fn(_module, inputs):
        calls["n"] += 1
        if steering_vec is None:
            return
        x = inputs[0]  # [B, L, H] input to MLP (post-attn layernorm output)
        sv = steering_vec
        if normalize:
            sv = sv / (sv.norm() + 1e-9)
        B, L, H = x.shape
        if L == 0:
            return
        xs = x[:, L-1, :]  # only the decision state
        alpha = scale * (xs.norm(dim=-1, keepdim=True).clamp_min(1e-6) if norm_scale else 1.0)
        xs += alpha * sv
        x[:, L-1, :] = xs
    hook_fn.calls = calls
    return hook_fn

# --- make an index-aware hook (works for both pre-MLP or residual hooks) ---
def make_indexed_add_hook(steer_vec, get_pos_idx, scale=1.0, normalize=False, norm_scale=False):
    calls = {"n": 0}
    def hook_fn(_module, inputs, output=None):  # works for pre or post hooks
        calls["n"] += 1
        # Select the tensor we are editing:
        x = inputs[0] if output is None else output          # [B, L, H]
        if steer_vec is None or x is None: 
            return None if output is None else output

        sv = steer_vec.to(x.dtype)
        if normalize:
            sv = sv / (sv.norm() + 1e-9)

        B, L, H = x.shape
        pos_idx = get_pos_idx(B, L, x)                        # LongTensor [B]
        # gather the slice [B, 1, H]
        xs = x[torch.arange(B, device=x.device), pos_idx, :].unsqueeze(1)
        alpha = scale * (xs.norm(dim=-1, keepdim=True).clamp_min(1e-6) if norm_scale else 1.0)
        xs = xs + alpha * sv.view(1, 1, -1)
        x[torch.arange(B, device=x.device), pos_idx, :] = xs.squeeze(1)

        return None if output is None else x  # for forward_hook return the modified output
    hook_fn.calls = calls
    return hook_fn


class SteeringContext:
    def __init__(self, layer_module, hook):
        self.layer_module = layer_module
        self.hook = hook
        self.handle = None
    def __enter__(self):
        self.handle = self.layer_module.register_forward_pre_hook(self.hook)
        return self
    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False

# ----------------- IO helpers -----------------
def load_jsonl(path, max_items=0, shuffle=False, seed=0):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except Exception:
                continue
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if max_items > 0:
        rows = rows[:max_items]
    return rows

def split_rows(rows: List[dict], train_frac=0.4, val_frac=0, seed=0) -> Tuple[List[dict], List[dict], List[dict]]:
    assert 0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1
    rng = random.Random(seed)
    rows = [r for r in rows if r.get("question") and r.get("answer_matching_behavior") and r.get("answer_not_matching_behavior")]
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(round(train_frac * n))
    train = rows[:n_train]
    test  = rows[n_train:]
    return train, test

def make_prompt(question_text: str) -> str:
    q = question_text.rstrip()
    return (
        f"{q}\n\n"
        "Reply with a single character ONLY: A or B.\n"
        "Do not include parentheses, words, punctuation, spaces, or newlines.\n"
        "Answer: "
    )


def compose_texts_for_sv_from_rows(rows: List[dict]) -> Tuple[List[str], List[str]]:
    """
    Returns (target_texts, base_texts) where each item is:
      prompt + single-token label  ('A' or 'B')
    target_texts = y+ (answer_matching_behavior)
    base_texts   = y- (opposite label)
    """
    pos, neg = [], []
    for r in rows:
        prompt = make_prompt(r["question"])
        y_pos = "A" if "A" in r.get("answer_matching_behavior", "") else "B"
        y_neg = "B" if y_pos == "A" else "A"
        pos.append(prompt + y_pos)  # teacher-forced single token
        neg.append(prompt + y_neg)
    return pos, neg

# ----------------- Steering vec builder -----------------


@torch.no_grad()
def compute_steering_vec(
    ds_engine, tokenizer, device,
    base_texts: List[str],    # each is: prompt + (y- single token)
    target_texts: List[str],  # each is: prompt + (y+ single token)
    batch_size: int = 4,
    target_layer: int = 0,
) -> torch.Tensor:
    """
    Build the steering vector at the decision state (last prompt token),
    using right padding and two clean passes (base, target) per microbatch.
    Also print human-readable output from logits.
    """
    from time import time
    start = time()

    # --- we assume you've built base_texts/target_texts from a common prompt "Answer: "
    # and appended exactly one A/B token (leading-space form).
    # We compute prompt lengths by tokenizing PROMPT-ONLY strings once.
    def split_prompt(txt: str) -> str:
        # txt = "<prompt>Answer: " + ("A" or "B")  (single char)
        return txt[:-1]  # remove the one A/B char

    # Temporarily force right padding here to make indices trivial.
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    layer = ds_engine.module.model.layers[target_layer]

    captured = None
    idx_holder = {"pos": None}  # (B,) decision indices

    def pre_hook(_m, inputs):
        nonlocal captured
        hs = inputs[0]                   # [B, L, H], pre-MLP input
        rows = hs[torch.arange(hs.size(0), device=hs.device), idx_holder["pos"], :]
        captured = rows.contiguous()     # [B, H]

    handle = layer.register_forward_pre_hook(pre_hook)

    def tok(lst):  # right padding, no specials
        t = tokenizer(lst, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        return t.input_ids.to(device), t.attention_mask.to(device)

    def logits_to_readable(logits, input_ids, tokenizer):
        """Convert logits to human-readable tokens and probabilities."""
        # Logits shape: [batch_size, sequence_length, vocab_size]
        # Take logits at the last non-padding token for each sequence
        batch_size, seq_len, vocab_size = logits.shape
        last_token_idx = input_ids.ne(tokenizer.pad_token_id).sum(dim=1) - 1  # Index of last real token
        last_logits = logits[torch.arange(batch_size), last_token_idx, :]  # [batch_size, vocab_size]
        
        # Apply softmax to get probabilities
        probs = torch.softmax(last_logits, dim=-1)
        # Get top predicted token IDs and their probabilities
        top_probs, top_ids = torch.topk(probs, k=3, dim=-1)  # Top 3 for readability
        top_tokens = [[tokenizer.decode([id_]) for id_ in ids] for ids in top_ids]
        
        # Decode input texts for context
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Print human-readable output
        for i in range(batch_size):
            print(f"\nInput text: {input_texts[i]}")
            print("Top 3 predicted tokens and probabilities:")
            for j in range(3):
                print(f"  Token: '{top_tokens[i][j]}', Probability: {top_probs[i][j]:.4f}")

    N = min(len(base_texts), len(target_texts))
    sum_delta = None
    count = 0

    for s in range(0, N, batch_size):
        e = min(s+batch_size, N)

        base_batch  = base_texts[s:e]
        targ_batch  = target_texts[s:e]
        prompts     = [split_prompt(t) for t in base_batch]  # same prompts for target

        # 1) prompt-only lengths (right padded → index is T-1)
        p_ids, p_mask = tok(prompts)             # [B, Tp]
        T = p_mask.sum(dim=1).to(torch.long)     # [B]
        pos_idx = T-1                        # last prompt token (decision state)
        idx_holder["pos"] = pos_idx

        # decision_token_id = p_ids[0, pos_idx].item()
        # Decode the token ID to get the human-readable token
        # decision_token_str = tokenizer.decode([decision_token_id])
        # print("decision_token_id" , decision_token_id)
        # 2) forward base; hook grabs decision states and capture logits
        b_ids, b_mask = tok(base_batch)
        base_output = ds_engine(input_ids=b_ids, attention_mask=b_mask)
        print("\n=== Base Batch Logits ===")
        logits_to_readable(base_output.logits, b_ids, tokenizer)
        base_vec = captured                     # [B, H]

        # # 3) forward target; hook grabs decision states and capture logits
        t_ids, t_mask = tok(targ_batch)
        targ_output = ds_engine(input_ids=t_ids, attention_mask=t_mask)
        print("\n=== Target Batch Logits ===")
        logits_to_readable(targ_output.logits, t_ids, tokenizer)
        targ_vec = captured                     # [B, H]

        delta = (targ_vec - base_vec)           # [B, H]
        sum_delta = delta.sum(dim=0) if sum_delta is None else (sum_delta + delta.sum(dim=0))
        count += delta.size(0)

        # small hygiene
        del p_ids, p_mask, b_ids, b_mask, t_ids, t_mask, base_vec, targ_vec, delta
        torch.cuda.empty_cache()

    handle.remove()
    tokenizer.padding_side = old_side

    sv = sum_delta / max(count, 1)
    print(f"[INFO] SV built at decision state in {time()-start:.2f}s over N={count} | ||sv||={float(sv.norm()):.4f}")
    return sv



# ----------------- Main -----------------
def main():
    args = parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    if dist.get_rank() == 0:
        print(f"[INFO] Loading model/tokenizer from config: {args.config}")
    cfg = load_config(args.config)
    tokenizer, base_model, _ = load_tokenizer_and_model(cfg)
    if cfg.get("pad_left", True):
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tp = torch.cuda.device_count() if str(cfg.get("tensor_parallel_size","auto")).lower() == "auto" else int(cfg["tensor_parallel_size"])
    ds_engine = deepspeed.init_inference(
        base_model, dtype=torch.bfloat16,
        tensor_parallel={"tp_size": tp},
        replace_method="none", replace_with_kernel_inject=False,
    )
    ds_engine.eval()

    if dist.get_rank() == 0:
        print("[INFO] Loading dataset and making 40/10/50 split...")
    all_rows = load_jsonl(args.dataset, shuffle=True, seed=args.split_seed)
    train_rows, test_rows = split_rows(all_rows, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed)

    # Build SV from TRAIN split: positive (target) vs negative (base)
    train_rows_for_sv = train_rows[:args.sv_max_items] if args.sv_max_items > 0 else train_rows
    pos_texts, neg_texts = compose_texts_for_sv_from_rows(train_rows_for_sv)

    sv = compute_steering_vec(
        ds_engine, tokenizer, device,
        base_texts=neg_texts, target_texts=pos_texts,
        batch_size=args.batch_size, target_layer=args.layer
    ).to(device)

    layer_module = ds_engine.module.model.layers[args.layer]

    # Prepare TEST split for evaluation
    eval_rows = test_rows[:args.max_items] if args.max_items > 0 else test_rows
    

    # Resolve A/B token IDs **once** for your prompt formatting
    id_A, id_B, ab_form = get_ab_token_ids_or_fail(tokenizer)
    print(f"[DEBUG] AB token form = {repr(ab_form)}, ids: A={id_A}, B={id_B}")

    if args.verify_outputs:
        dist.barrier()
        ok_tr, tot_tr, bad_tr = verify_one_token_via_logits(
            ds_engine, tokenizer, device, train_rows,
            lambda q: make_prompt_for_eval(q, args.system_prompt),
            k=args.verify_k
        )
        ok_te, tot_te, bad_te = verify_one_token_via_logits(
            ds_engine, tokenizer, device, eval_rows,
            lambda q: make_prompt_for_eval(q, args.system_prompt),
            k=args.verify_k
        )
        dist.barrier()
    if dist.get_rank() == 0:
        print(f"[VERIFY] TRAIN A/B strict: {ok_tr}/{tot_tr} ok")
        if bad_tr:
            print("[VERIFY][TRAIN] examples of violations:")
            for b in bad_tr[:5]:
                print("   ", b)
        print(f"[VERIFY] TEST  A/B strict: {ok_te}/{tot_te} ok")
        if bad_te:
            print("[VERIFY][TEST] examples of violations:")
            for b in bad_te[:5]:
                print("   ", b)
    # Multipliers
    multipliers = [float(x) for x in args.multipliers.split(",")]
    mean_vals = []
    valid_multipliers = []

    # Eval: single-pass, next-token logits, add λ·v at decision position
    if dist.get_rank() == 0:
        print("[INFO] Starting evaluation across multipliers (single-pass, next-token logits)...")

    for m in multipliers:
        diffs = []
        for idx, ex in enumerate(eval_rows):
            if idx % 50 == 0 and dist.get_rank() == 0:
                print(f"    Processed {idx}/{len(eval_rows)} for multiplier {m}...")

            # 1) Prompt ONLY (no answer appended)
            q = make_prompt_for_eval(ex["question"], args.system_prompt)
            base = tokenizer(q, return_tensors="pt", add_special_tokens=False)
            base = {k: v.to(device) for k, v in base.items()}
            prompt_len = base["input_ids"].shape[1]

            # 2) Map y+ and y- ids
            pos_letter = extract_pos_letter(ex)
            y_pos_id = id_A if pos_letter == "A" else id_B
            y_neg_id = id_B if pos_letter == "A" else id_A

            # 3) Inject λ·v at decision state (last prompt position)
            hook = None if m == 0.0 else make_lastpos_pre_hook(sv, scale=m, normalize=args.normalize)
            ctx = SteeringContext(layer_module, hook) if hook else type('', (), {'__enter__':lambda s:s,'__exit__':lambda s,*a:False})()
            with ctx:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = ds_engine(**base, return_dict=True)
                    next_logits = out.logits[0, -1, :]
            # (optional) check hook calls:
            if hook:
                assert hook.calls["n"] > 0, "MLP pre-hook never fired during eval forward"

            # 4) Propensity = raw logit difference (same pass!)
            mLD = float(next_logits[y_pos_id] - next_logits[y_neg_id])
            diffs.append(mLD)

        if len(diffs) == 0:
            continue
        mean_val = float(np.mean(diffs))
        mean_vals.append(mean_val)
        valid_multipliers.append(m)
        if dist.get_rank() == 0:
            print(f"    [STATS] m={m}: mean={mean_val:.6f} over N={len(diffs)} examples")

    # Fit slope, intercept on the multipliers we actually evaluated
    x = np.array(valid_multipliers, dtype=float)
    y = np.array(mean_vals, dtype=float)
    if len(x) >= 2:
        slope = ((x - x.mean()) @ (y - y.mean())) / np.maximum(((x - x.mean())**2).sum(), 1e-9)
        intercept = y.mean() - slope * x.mean()
    else:
        slope, intercept = 0.0, y[0] if len(y) == 1 else 0.0

    if dist.get_rank() == 0:
        out = {
            "layer": args.layer,
            "multipliers": valid_multipliers,
            "mean_propensity": [float(v) for v in mean_vals],
            "slope": float(slope),
            "intercept": float(intercept),
            "sv_counts": {"train_rows": len(train_rows_for_sv)},
            "splits": {"train": len(train_rows),  "test": len(test_rows)},
        }
        save_path = Path(f"steer_eval_layer{args.layer}_results.json")
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] Results saved to {save_path}")
        print(json.dumps(out, indent=2))

    torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
