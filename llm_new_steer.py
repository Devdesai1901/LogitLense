# steer_vector.py
import os, gc, argparse
from pathlib import Path
from typing import Dict
import torch
import torch.distributed as dist
import deepspeed
from einops import einsum
from tqdm import tqdm

from model_helper.config import load_config
from model_helper.model_io import load_tokenizer_and_model

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)  # deepspeed injects this
    ap.add_argument("--config", required=True, help="Path to YAML config for 70B")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--layer_stride", type=int, default=8, help="Intervene every N layers")
    ap.add_argument("--max_new_tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--save_dir", default="./explanation/steering")
    return ap.parse_known_args()[0]

# ----------------- Prompts -----------------
enthusiastic_prompts = [
    "Let’s make this as fun as possible!",
    "This is going to be a blast!",
    "I can’t stop laughing about this!",
]
unenthusiastic_prompts = [
    "We need to approach this with caution.",
    "This is a matter that requires full focus.",
    "Let’s stay on task and avoid distractions.",
]
test_prompts = [
    "Should we make this more lighthearted?",
    "Can we add some fun to this process?",
    "Do you think this needs to be taken more seriously?",
]

# ----------------- Core -----------------
def tokenize_prompts(tokenizer, device, prompt_list):
    toks = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True)
    return toks.input_ids.to(device), toks.attention_mask.to(device)

def pad_to_max_len(tokenizer, toks, mask, max_len):
    pad_len = max_len - toks.size(1)
    if pad_len > 0:
        pad_tok = torch.full((toks.size(0), pad_len), tokenizer.pad_token_id, device=toks.device)
        pad_mask = torch.zeros((mask.size(0), pad_len), device=mask.device)
        if tokenizer.padding_side == "left":
            toks = torch.cat([pad_tok, toks], dim=1)
            mask = torch.cat([pad_mask, mask], dim=1)
        else:
            toks = torch.cat([toks, pad_tok], dim=1)
            mask = torch.cat([mask, pad_mask], dim=1)
    return toks, mask

@torch.no_grad()
def find_steering_vecs(ds_engine, tokenizer, device, base_toks, target_toks, base_mask, target_mask, batch_size=2) -> Dict[int, torch.Tensor]:
    steering_vecs: Dict[int, torch.Tensor] = {}
    steps = len(range(0, base_toks.size(0), batch_size))
    captured_outputs = {}

    def capture_last_token(layer_idx):
        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured_outputs[layer_idx] = hidden_states[:, -1, :].detach()
        return hook_fn

    for i in tqdm(range(0, base_toks.size(0), batch_size), desc="Steering vec batches"):
        base_batch = base_toks[i:i+batch_size]
        target_batch = target_toks[i:i+batch_size]
        base_mask_batch = base_mask[i:i+batch_size]
        target_mask_batch = target_mask[i:i+batch_size]

        max_len = max(base_batch.size(1), target_batch.size(1))
        base_batch, base_mask_batch = pad_to_max_len(tokenizer, base_batch, base_mask_batch, max_len)
        target_batch, target_mask_batch = pad_to_max_len(tokenizer, target_batch, target_mask_batch, max_len)

        combined_toks = torch.cat([base_batch, target_batch], dim=0)
        combined_mask = torch.cat([base_mask_batch, target_mask_batch], dim=0)

        handles = []
        for idx, layer_module in enumerate(ds_engine.module.model.layers):
            handles.append(layer_module.register_forward_hook(capture_last_token(idx)))

        _ = ds_engine(input_ids=combined_toks, attention_mask=combined_mask)

        for h in handles:
            h.remove()

        for layer_idx, hidden_vecs in captured_outputs.items():
            base_vec = hidden_vecs[:base_batch.size(0)]
            target_vec = hidden_vecs[base_batch.size(0):]
            delta = (target_vec - base_vec).mean(dim=0).cpu() / steps
            steering_vecs[layer_idx] = steering_vecs.get(layer_idx, 0) + delta

        captured_outputs.clear()

    return steering_vecs

@torch.no_grad()
def do_steering(ds_engine, tokenizer, device, test_toks, test_mask, steering_vec=None, scale=1.0, normalise=True, layer=None, proj=True, batch_size=1, max_new_tokens=60, top_p=0.95, temperature=0.7):
    def hook_fn(module, input):
        if steering_vec is not None:
            sv = steering_vec / steering_vec.norm() if normalise else steering_vec
            if proj:
                sv = einsum(input[0], sv.view(-1, 1), 'b l h, h s -> b l s') * sv
            input[0][:, :, :] = input[0] - scale * sv

    handles = []
    if steering_vec is not None:
        for i, layer_module in enumerate(ds_engine.module.model.layers):
            if layer is None or i == layer:
                handles.append(layer_module.register_forward_pre_hook(hook_fn))

    outputs = []
    for i in range(0, test_toks.size(0), batch_size):
        out = ds_engine.generate(
            input_ids=test_toks[i:i+batch_size],
            attention_mask=test_mask[i:i+batch_size],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id
        )
        outputs.extend(out)

    for h in handles:
        h.remove()

    return outputs

def main():
    args = parse_args()

    # ----- Distributed -----
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ----- Load cfg/tokenizer/model -----
    cfg = load_config(args.config)
    tokenizer, base_model, source = load_tokenizer_and_model(cfg)
    if cfg.get("pad_left", True):
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    world_size = torch.cuda.device_count()
    tp = world_size if str(cfg.get("tensor_parallel_size", "auto")).lower() == "auto" else int(cfg["tensor_parallel_size"])

    ds_engine = deepspeed.init_inference(
        base_model,
        dtype=torch.bfloat16,
        tensor_parallel={"tp_size": tp},
        replace_method="none",
        replace_with_kernel_inject=False,
    )
    ds_engine.eval()

    # ----- Tokenize data -----
    base_toks, base_mask   = tokenize_prompts(tokenizer, device, unenthusiastic_prompts)
    target_toks, target_mask = tokenize_prompts(tokenizer, device, enthusiastic_prompts)
    test_toks, test_mask   = tokenize_prompts(tokenizer, device, test_prompts)

    # ----- Compute steering vecs -----
    steering_vecs = find_steering_vecs(ds_engine, tokenizer, device, base_toks, target_toks, base_mask, target_mask, batch_size=args.batch_size)

    # ----- Baseline -----
    BATCH_SIZE = len(test_prompts)
    baseline_outputs = do_steering(
        ds_engine, tokenizer, device,
        test_toks, test_mask,
        steering_vec=None,
        batch_size=BATCH_SIZE,
        proj=True,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature
    )

    # ----- Interventions -----
    if dist.get_rank() == 0:
        print("\n--- BASELINE OUTPUTS ---")
        for j, out in enumerate(baseline_outputs):
            print(f"Prompt: {test_prompts[j]}")
            print("BASELINE:", tokenizer.decode(out, skip_special_tokens=True))

    for layer in range(1, len(ds_engine.module.model.layers), args.layer_stride):
        steered_outputs = do_steering(
            ds_engine, tokenizer, device,
            test_toks, test_mask,
            steering_vec=steering_vecs[layer].to(device),
            layer=layer,
            batch_size=BATCH_SIZE,
            proj=True,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature
        )
        if dist.get_rank() == 0:
            print(f"\n--- LAYER {layer} INTERVENTION ---")
            for j in range(len(steered_outputs)):
                print(f"Prompt: {test_prompts[j]}")
                print("STEERED :", tokenizer.decode(steered_outputs[j], skip_special_tokens=True))

    torch.cuda.empty_cache(); gc.collect()

if __name__ == "__main__":
    main()
