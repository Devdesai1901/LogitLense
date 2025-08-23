# evaluate_steerability_custom.py
import os, json, argparse, gc, random, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.distributed as dist
import deepspeed
from einops import einsum
from time import time  # keep your existing usage inside compute_steering_vec
import time as pytime  # for perf_counter
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_helper.config import load_config
from model_helper.model_io import load_tokenizer_and_model

# ----------------- Timing helpers (rank-safe) -----------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def barrier_safe():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

def record_block(timings: dict, key: str, start_ts: str, end_ts: str, start_pc: float, end_pc: float):
    timings[key] = {"start": start_ts, "end": end_ts, "seconds": round(end_pc - start_pc, 6)}

# ----------------- A/B token helpers -----------------
def plot_propensity_curve(multipliers, mean_vals, layer, out_png):
    x = np.array(multipliers, dtype=float)
    y = np.array(mean_vals, dtype=float)

    xm = x.mean()
    den = max(((x - xm) ** 2).sum(), 1e-12)
    slope = float(((x - xm) @ (y - y.mean())) / den)
    intercept = float(y.mean() - slope * xm)

    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    plt.figure()
    plt.scatter(x, y, label="Mean propensity")
    plt.plot(x, yhat, label="Linear fit")
    plt.xlabel("Steering Multiplier")
    plt.ylabel("Logit difference")
    plt.title(f"Propensity curve (layer {layer})\nSlope={slope:.4f}, R²={r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return {"slope": slope, "intercept": float(intercept), "r2": r2}

def get_ab_token_ids_or_fail(tokenizer):
    ids_A = tokenizer.encode("A", add_special_tokens=False)
    ids_B = tokenizer.encode("B", add_special_tokens=False)
    if len(ids_A) == 1 and len(ids_B) == 1:
        return ids_A[0], ids_B[0], "A"
    ids_A = tokenizer.encode(" A", add_special_tokens=False)
    ids_B = tokenizer.encode(" B", add_special_tokens=False)
    assert len(ids_A) == 1 and len(ids_B) == 1, "A/B must be single tokens; adjust prompt tail."
    return ids_A[0], ids_B[0], " B"

def extract_pos_letter(rec: dict) -> str:
    amb = rec.get("answer_matching_behavior", "")
    return "A" if "A" in amb else "B"

@torch.no_grad()
def verify_one_token_via_logits(ds_engine, tokenizer, device, rows, format_prompt_fn, k=24):
    subset = rows[:min(k, len(rows))]
    total = 0; ok = 0; bad_samples = []
    for ex in subset:
        prompt = format_prompt_fn(ex["question"])
        inp = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inp = {k: v.to(device) for k, v in inp.items()}
        out = ds_engine(**inp, return_dict=True)
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
    # dataset
    ap.add_argument("--dataset", required=True, help="jsonl with question/answer_matching_behavior/answer_not_matching_behavior")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--sv_max_items", type=int, default=1024)
    # evaluation
    ap.add_argument("--multipliers", type=str, default="-1.5,-1.0,-0.5,0,0.5,1.0,1.5")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--system_prompt", type=str, default="Answer:")
    ap.add_argument("--verify_outputs", action="store_true")
    ap.add_argument("--verify_k", type=int, default=24)
    # timings
    ap.add_argument("--timing_json", default="timings_eval_sv.json", help="Write timing JSON here (rank 0 only)")
    return ap.parse_args()

# ----------------- Steering hook utils -----------------
def make_lastpos_pre_hook(steering_vec, scale=1.0, normalize=False, norm_scale=False):
    calls = {"n": 0}
    def hook_fn(_module, inputs):
        calls["n"] += 1
        if steering_vec is None:
            return
        x = inputs[0]  # [B, L, H]
        sv = steering_vec
        if normalize:
            sv = sv / (sv.norm() + 1e-9)
        B, L, H = x.shape
        if L == 0:
            return
        xs = x[:, L-1, :]
        alpha = scale * (xs.norm(dim=-1, keepdim=True).clamp_min(1e-6) if norm_scale else 1.0)
        xs += alpha * sv
        x[:, L-1, :] = xs
    hook_fn.calls = calls
    return hook_fn

def make_indexed_add_hook(steer_vec, get_pos_idx, scale=1.0, normalize=False, norm_scale=False):
    calls = {"n": 0}
    def hook_fn(_module, inputs, output=None):
        calls["n"] += 1
        x = inputs[0] if output is None else output
        if steer_vec is None or x is None:
            return None if output is None else output
        sv = steer_vec.to(x.dtype)
        if normalize:
            sv = sv / (sv.norm() + 1e-9)
        B, L, H = x.shape
        pos_idx = get_pos_idx(B, L, x)
        xs = x[torch.arange(B, device=x.device), pos_idx, :].unsqueeze(1)
        alpha = scale * (xs.norm(dim=-1, keepdim=True).clamp_min(1e-6) if norm_scale else 1.0)
        xs = xs + alpha * sv.view(1, 1, -1)
        x[torch.arange(B, device=x.device), pos_idx, :] = xs.squeeze(1)
        return None if output is None else x
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
    pos, neg = [], []
    for r in rows:
        prompt = make_prompt(r["question"])
        y_pos = "A" if "A" in r.get("answer_matching_behavior", "") else "B"
        y_neg = "B" if y_pos == "A" else "A"
        pos.append(prompt + y_pos)
        neg.append(prompt + y_neg)
    return pos, neg

# ----------------- Steering vec builder -----------------
@torch.no_grad()
def compute_steering_vec(
    ds_engine, tokenizer, device,
    base_texts: List[str],
    target_texts: List[str],
    batch_size: int = 4,
    target_layer: int = 0,
) -> torch.Tensor:
    from time import time
    start = time()

    def split_prompt(txt: str) -> str:
        return txt[:-1]

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    layer = ds_engine.module.model.layers[target_layer]
    captured = None
    idx_holder = {"pos": None}

    def pre_hook(_m, inputs):
        nonlocal captured
        hs = inputs[0]
        rows = hs[torch.arange(hs.size(0), device=hs.device), idx_holder["pos"], :]
        captured = rows.contiguous()

    handle = layer.register_forward_pre_hook(pre_hook)

    def tok(lst):
        t = tokenizer(lst, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
        return t.input_ids.to(device), t.attention_mask.to(device)

    def logits_to_readable(logits, input_ids, tokenizer):
        batch_size, seq_len, vocab_size = logits.shape
        last_token_idx = input_ids.ne(tokenizer.pad_token_id).sum(dim=1) - 1
        last_logits = logits[torch.arange(batch_size), last_token_idx, :]
        probs = torch.softmax(last_logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, k=3, dim=-1)
        top_tokens = [[tokenizer.decode([id_]) for id_ in ids] for ids in top_ids]
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
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
        prompts     = [split_prompt(t) for t in base_batch]

        p_ids, p_mask = tok(prompts)
        T = p_mask.sum(dim=1).to(torch.long)
        pos_idx = T-1
        idx_holder["pos"] = pos_idx

        b_ids, b_mask = tok(base_batch)
        base_output = ds_engine(input_ids=b_ids, attention_mask=b_mask)
        print("\n=== Base Batch Logits ===")
        logits_to_readable(base_output.logits, b_ids, tokenizer)
        base_vec = captured

        t_ids, t_mask = tok(targ_batch)
        targ_output = ds_engine(input_ids=t_ids, attention_mask=t_mask)
        print("\n=== Target Batch Logits ===")
        logits_to_readable(targ_output.logits, t_ids, tokenizer)
        targ_vec = captured

        delta = (targ_vec - base_vec)
        sum_delta = delta.sum(dim=0) if sum_delta is None else (sum_delta + delta.sum(dim=0))
        count += delta.size(0)

        del p_ids, p_mask, b_ids, b_mask, t_ids, t_mask, base_vec, targ_vec, delta
        torch.cuda.empty_cache()

    handle.remove()
    tokenizer.padding_side = old_side

    sv = sum_delta / max(count, 1)
    print(f"[INFO] SV built at decision state in {time()-start:.2f}s over N={count} | ||sv||={float(sv.norm()):.4f}")
    return sv

# ----------------- Eval helper (one multiplier) -----------------
@torch.no_grad()
def eval_diffs_for_multiplier(
    m: float,
    eval_rows: List[dict],
    ds_engine,
    tokenizer,
    device,
    layer_module,
    sv: torch.Tensor,
    id_A: int,
    id_B: int,
    system_prompt: str,
    normalize: bool,
):
    diffs = []
    for idx, ex in enumerate(eval_rows):
        if idx % 50 == 0 and is_rank0():
            print(f"    Processed {idx}/{len(eval_rows)} for multiplier {m}...")
        q = make_prompt_for_eval(ex["question"], system_prompt)
        base = tokenizer(q, return_tensors="pt", add_special_tokens=False)
        base = {k: v.to(device) for k, v in base.items()}

        pos_letter = extract_pos_letter(ex)
        y_pos_id = id_A if pos_letter == "A" else id_B
        y_neg_id = id_B if pos_letter == "A" else id_A

        hook = None if abs(m) < 1e-12 else make_lastpos_pre_hook(sv, scale=m, normalize=normalize)
        ctx = SteeringContext(layer_module, hook) if hook else type('', (), {'__enter__':lambda s:s,'__exit__':lambda s,*a:False})()
        with ctx:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = ds_engine(**base, return_dict=True)
                next_logits = out.logits[0, -1, :]

        if hook:
            assert hook.calls["n"] > 0, "MLP pre-hook never fired during eval forward"

        mLD = float(next_logits[y_pos_id] - next_logits[y_neg_id])
        diffs.append(mLD)
    return diffs

# ----------------- Main -----------------
def main():
    args = parse_args()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    # timings container
    timings = {
        "script": "evaluate_steerability_custom.py",
        "config": args.config,
        "env": {
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "layer": args.layer,
            "batch_size": args.batch_size,
            "sv_max_items": args.sv_max_items,
            "max_items": args.max_items,
        },
        "blocks": {}
    }

    if is_rank0():
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

    if is_rank0():
        print("[INFO] Loading dataset and making 40/10/50 split...")
    all_rows = load_jsonl(args.dataset, shuffle=True, seed=args.split_seed)
    train_rows, test_rows = split_rows(all_rows, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.split_seed)

    # Build SV
    train_rows_for_sv = train_rows[:args.sv_max_items] if args.sv_max_items > 0 else train_rows
    pos_texts, neg_texts = compose_texts_for_sv_from_rows(train_rows_for_sv)

    barrier_safe()
    sv_start_ts = now_iso()
    sv_start_pc = pytime.perf_counter()

    sv = compute_steering_vec(
        ds_engine, tokenizer, device,
        base_texts=neg_texts, target_texts=pos_texts,
        batch_size=args.batch_size, target_layer=args.layer
    ).to(device)

    barrier_safe()
    sv_end_pc = pytime.perf_counter()
    sv_end_ts = now_iso()
    if is_rank0():
        record_block(timings["blocks"], "steering_vector", sv_start_ts, sv_end_ts, sv_start_pc, sv_end_pc)

    layer_module = ds_engine.module.model.layers[args.layer]
    eval_rows = test_rows[:args.max_items] if args.max_items > 0 else test_rows

    # Resolve A/B token IDs
    id_A, id_B, ab_form = get_ab_token_ids_or_fail(tokenizer)
    if is_rank0():
        print(f"[DEBUG] AB token form = {repr(ab_form)}, ids: A={id_A}, B={id_B}")

    # Optional verification
    if args.verify_outputs:
        barrier_safe()
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
        barrier_safe()
        if is_rank0():
            print(f"[VERIFY] TRAIN A/B strict: {ok_tr}/{tot_tr} ok")
            if bad_tr:
                print("[VERIFY][TRAIN] examples of violations:")
                for b in bad_tr[:5]: print("   ", b)
            print(f"[VERIFY] TEST  A/B strict: {ok_te}/{tot_te} ok")
            if bad_te:
                print("[VERIFY][TEST] examples of violations:")
                for b in bad_te[:5]: print("   ", b)

    # Multipliers parsing
    multipliers = [float(x) for x in args.multipliers.split(",")]
    include_zero = any(abs(m) < 1e-12 for m in multipliers)

    mean_vals = []
    valid_multipliers = []

    # ----- Baseline (m=0) timing -----
    barrier_safe()
    base_start_ts = now_iso()
    base_start_pc = pytime.perf_counter()

    baseline_diffs = eval_diffs_for_multiplier(
        0.0, eval_rows, ds_engine, tokenizer, device, layer_module, sv, id_A, id_B,
        args.system_prompt, args.normalize
    )

    barrier_safe()
    base_end_pc = pytime.perf_counter()
    base_end_ts = now_iso()
    if is_rank0():
        record_block(timings["blocks"], "baseline_generation", base_start_ts, base_end_ts, base_start_pc, base_end_pc)

    # add to metrics only if user asked for 0 in multipliers
    if include_zero and len(baseline_diffs) > 0:
        mean_base = float(np.mean(baseline_diffs))
        mean_vals.append(mean_base)
        valid_multipliers.append(0.0)
        if is_rank0():
            print(f"    [STATS] m=0: mean={mean_base:.6f} over N={len(baseline_diffs)} examples")

    # ----- Steered overall timing (m != 0) -----
    if is_rank0():
        print("[INFO] Starting evaluation across multipliers (single-pass, next-token logits)...")
    barrier_safe()
    steer_start_ts = now_iso()
    steer_start_pc = pytime.perf_counter()

    for m in multipliers:
        if abs(m) < 1e-12:
            continue  # already did baseline
        diffs = eval_diffs_for_multiplier(
            m, eval_rows, ds_engine, tokenizer, device, layer_module, sv, id_A, id_B,
            args.system_prompt, args.normalize
        )
        if len(diffs) == 0:
            continue
        mean_val = float(np.mean(diffs))
        mean_vals.append(mean_val)
        valid_multipliers.append(m)
        if is_rank0():
            print(f"    [STATS] m={m}: mean={mean_val:.6f} over N={len(diffs)} examples")

    barrier_safe()
    steer_end_pc = pytime.perf_counter()
    steer_end_ts = now_iso()
    if is_rank0():
        record_block(timings["blocks"], "steered_generation_overall", steer_start_ts, steer_end_ts, steer_start_pc, steer_end_pc)

    # Fit slope, intercept
    x = np.array(valid_multipliers, dtype=float)
    y = np.array(mean_vals, dtype=float)
    if len(x) >= 2:
        slope = ((x - x.mean()) @ (y - y.mean())) / np.maximum(((x - x.mean())**2).sum(), 1e-9)
        intercept = y.mean() - slope * x.mean()
    else:
        slope, intercept = 0.0, y[0] if len(y) == 1 else 0.0

    if is_rank0():
        out = {
            "layer": args.layer,
            "multipliers": [float(v) for v in valid_multipliers],
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
        png_path = f"steer_eval_layer{args.layer}_propensity.png"
        metrics = plot_propensity_curve(valid_multipliers, mean_vals, args.layer, png_path)
        print("[PLOT]", metrics, "->", png_path)

        # Write timings JSON
        with open(args.timing_json, "w") as f:
            json.dump({"blocks": timings["blocks"], "env": timings["env"], "script": timings["script"], "config": timings["config"]}, f, indent=2)
        print(f"[timing] Wrote timing JSON to: {args.timing_json}")
        print(json.dumps(timings["blocks"], indent=2))

    torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
