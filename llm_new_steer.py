import torch
import torch.distributed as dist
import deepspeed
import os
import gc
from einops import einsum
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ========== Distributed Init ==========
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# ========== Load Tokenizer & Model ==========
model_name = "meta-llama/Meta-Llama-3.1-70B"
local_path = os.path.expanduser("~/LogitLens4LLMs/output/cache/models--meta-llama--Meta-Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac")

tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    local_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map=None,
)

model = deepspeed.init_inference(
    base_model,
    dtype=torch.bfloat16,
    tensor_parallel={"tp_size": torch.cuda.device_count()},
    replace_method="none",
    replace_with_kernel_inject=False,
)
model.eval()

torch.cuda.empty_cache()
gc.collect()

# ========== Prompts ==========
enthusiastic_prompts = [
    "I'm so excited about the new product launch!",
    "What a fantastic day we're having!",
    "I can't wait to try this amazing new feature!"
]

unenthusiastic_prompts = [
    "The new product launch happened.",
    "It's just another day.",
    "There's a new feature, I guess."
]

test_prompts = [
    "Tell me something exciting about today.",
    "Is this product launch important?",
    "Do you enjoy feature releases?"
]

# ========== Tokenize with Attention Mask ==========
def tokenize_prompts(prompt_list):
    toks = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True)
    return toks.input_ids.to(device), toks.attention_mask.to(device)

base_toks, base_mask = tokenize_prompts(unenthusiastic_prompts)
target_toks, target_mask = tokenize_prompts(enthusiastic_prompts)
test_toks, test_mask = tokenize_prompts(test_prompts)


def pad_to_max_len(toks, mask, max_len):
    pad_len = max_len - toks.size(1)
    if pad_len > 0:
        pad_tok = torch.full((toks.size(0), pad_len), tokenizer.pad_token_id, device=toks.device)
        pad_mask = torch.zeros((mask.size(0), pad_len), device=mask.device)
        toks = torch.cat([pad_tok, toks], dim=1) if tokenizer.padding_side == "left" else torch.cat([toks, pad_tok], dim=1)
        mask = torch.cat([pad_mask, mask], dim=1) if tokenizer.padding_side == "left" else torch.cat([mask, pad_mask], dim=1)
    return toks, mask
# ========== Compute Steering Vectors ==========
def find_steering_vecs(model, base_toks, target_toks, base_mask, target_mask, batch_size=2):
    steering_vecs = {}
    steps = len(range(0, base_toks.size(0), batch_size))

    captured_outputs = {}

    def capture_last_token(layer_idx):
        def hook_fn(module, input, output):
            # Handle tuple output from Hugging Face layer
            hidden_states = output[0] if isinstance(output, tuple) else output
            captured_outputs[layer_idx] = hidden_states[:, -1, :].detach()
        return hook_fn

    for i in tqdm(range(0, base_toks.size(0), batch_size)):
        base_batch = base_toks[i:i+batch_size]
        target_batch = target_toks[i:i+batch_size]
        base_mask_batch = base_mask[i:i+batch_size]
        target_mask_batch = target_mask[i:i+batch_size]

        # Pad both to same length for concatenation
        max_len = max(base_batch.size(1), target_batch.size(1))
        base_batch, base_mask_batch = pad_to_max_len(base_batch, base_mask_batch, max_len)
        target_batch, target_mask_batch = pad_to_max_len(target_batch, target_mask_batch, max_len)

        # Combine into a single forward pass
        combined_toks = torch.cat([base_batch, target_batch], dim=0)
        combined_mask = torch.cat([base_mask_batch, target_mask_batch], dim=0)

        # Register hooks for each transformer layer
        handles = []
        for idx, layer_module in enumerate(model.module.model.layers):
            handles.append(layer_module.register_forward_hook(capture_last_token(idx)))

        # Forward pass without output_hidden_states=True
        _ = model(
            input_ids=combined_toks,
            attention_mask=combined_mask
        )

        # Remove hooks
        for h in handles:
            h.remove()

        # Compute steering deltas
        for layer_idx, hidden_vecs in captured_outputs.items():
            base_vec = hidden_vecs[:base_batch.size(0)]
            target_vec = hidden_vecs[base_batch.size(0):]
            delta = (target_vec - base_vec).mean(dim=0).cpu() / steps
            steering_vecs[layer_idx] = steering_vecs.get(layer_idx, 0) + delta

        captured_outputs.clear()

    return steering_vecs

# ========== Apply Steering Vector ==========
def do_steering(model, test_toks, test_mask, steering_vec=None, scale=1.0, normalise=True, layer=None, proj=True, batch_size=1):
    def hook_fn(module, input):
        if steering_vec is not None:
            sv = steering_vec / steering_vec.norm() if normalise else steering_vec
            if proj:
                sv = einsum(input[0], sv.view(-1, 1), 'b l h, h s -> b l s') * sv
            input[0][:, :, :] = input[0] - scale * sv

    handles = []
    if steering_vec is not None:
        for i, layer_module in enumerate(model.module.model.layers):
            if layer is None or i == layer:
                handles.append(layer_module.register_forward_pre_hook(hook_fn))

    outputs = []
    for i in range(0, test_toks.size(0), batch_size):
        out = model.generate(
            input_ids=test_toks[i:i+batch_size],
            attention_mask=test_mask[i:i+batch_size],
            max_new_tokens=60,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
        outputs.extend(out)

    for h in handles:
        h.remove()

    return outputs

# ========== Run Layerwise Steering ==========
steering_vecs = find_steering_vecs(model, base_toks, target_toks, base_mask, target_mask)
BATCH_SIZE = len(test_prompts)

baseline_outputs = do_steering(
    model,
    test_toks.to(device),
    test_mask.to(device),
    steering_vec=None,
    batch_size=BATCH_SIZE,
    proj=False
)
# if dist.get_rank() == 0:
#     print("\n--- BASELINE OUTPUTS ---")
#     for j, out in enumerate(baseline_outputs):
#         print(f"Prompt: {test_prompts[j]}")
#         print("BASELINE:", tokenizer.decode(out, skip_special_tokens=True))
#         print()

for layer in range(1, len(model.module.model.layers), 8):  # every 8 layers
    steered_outputs = do_steering(
        model,
        test_toks.to(device),
        test_mask.to(device),
        steering_vec=steering_vecs[layer].to(device),
        layer=layer,
        batch_size=BATCH_SIZE,
        proj=False
    )
    if dist.get_rank() == 0:
            print(f"\n--- LAYER {layer} INTERVENTION ---")
            for j in range(len(steered_outputs)):
                print(f"Prompt: {test_prompts[j]}")
                print("BASELINE:", tokenizer.decode(baseline_outputs[j], skip_special_tokens=True))
                print("STEERED :", tokenizer.decode(steered_outputs[j], skip_special_tokens=True))
                print()