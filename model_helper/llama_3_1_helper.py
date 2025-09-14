# llama_3_1_8b_helper_optimized.py
import os
import gc
import argparse
from typing import List, Tuple, Dict

import torch
import numpy as np
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Memory-friendly CUDA allocator behavior ---
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------
# Wrappers (attention & MLP)
# ---------------------------
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None           # [B, T, H]
        self.add_tensor = None            # optional steering vector

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)  # tuple with hidden states first
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None


class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.activations = None           # [B, T, H]

    def forward(self, *args, **kwargs):
        out = self.mlp(*args, **kwargs)
        self.activations = out
        return out

    def reset(self):
        self.activations = None


# --------------------------------------
# Block wrapper (store *hidden* slices)
# --------------------------------------
class BlockOutputWrapper(torch.nn.Module):
    """
    Stores lightweight last-token hidden slices from attention, mlp, and block.
    We postpone lm_head projection until decode time (vectorized) to save memory.
    """
    def __init__(
        self,
        block,
        lm_head,
        norm,
        collect_attn_mech: bool = True,
        collect_mlp: bool = True,
        collect_block: bool = True
    ):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm

        self.collect_attn_mech = collect_attn_mech
        self.collect_mlp = collect_mlp
        self.collect_block = collect_block

        # Wrap submodules
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        # Buffers to accumulate per-step slices (list of [B, 1, H])
        self.attn_mech_output_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        output = self.block(
            x,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )
        hidden_states = output[0] if isinstance(output, tuple) else output  # [B, T, H]

        if self.collect_attn_mech:
            attn_out = self.block.self_attn.activations
            if attn_out is not None:
                self.attn_mech_output_unembedded.append(attn_out[:, -1:, :])  # last token slice

        if self.collect_mlp:
            mlp_out = self.block.mlp.activations
            if mlp_out is not None:
                self.mlp_output_unembedded.append(mlp_out[:, -1:, :])

        if self.collect_block:
            self.block_output_unembedded.append(hidden_states[:, -1:, :])

        return output

    def attn_add_tensor(self, tensor: torch.Tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        self.attn_mech_output_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []

    def get_attn_activations(self):
        return self.block.self_attn.activations


# -----------------------------
# Optimized 8B Helper (DeepSpeed)
# -----------------------------
class Llama3_1_8BHelper:
    """
    Optimized like your 70B helper:
      - bf16 (if available), otherwise fp16
      - DeepSpeed inference with TP across visible GPUs
      - selected_layers wrapping to cut overhead
      - vectorized top-k decode for all positions
      - minimal CPU transfers; deferred lm_head projection
      - safe distributed init (if not already initialized)
    """
    def __init__(
        self,
        token: str = None,
        collect_attn_mech: bool = True,
        collect_mlp: bool = True,
        collect_block: bool = True,
        selected_layers: List[int] = None,
        model_id: str = "meta-llama/Llama-3.1-8B"
    ):
        print("Initializing Llama-3.1-8B Helper (optimized)...")

        # Rank & device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank % world_size)
            self.device = torch.device(f"cuda:{local_rank % world_size}")
        else:
            self.device = torch.device("cpu")
        print(f"[Init] device={self.device}, local_rank={local_rank}, world_size={world_size}")

        # Init distributed if needed (safe to call once)
        try:
            if torch.cuda.is_available() and not torch.distributed.is_initialized():
                deepspeed.init_distributed(dist_backend="nccl")
                print("[Init] DeepSpeed distributed initialized")
        except Exception as e:
            print(f"[Warn] DeepSpeed distributed init failed (continuing single-process): {e}")

        # Tokenizer
        print("[Load] Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            use_fast=True,
            trust_remote_code=True
        )

        # Model (bf16 preferred on A6000; falls back to fp16)
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"[Load] Model dtype={dtype}")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=token,
            trust_remote_code=True
        )

        # DeepSpeed inference (TP across all visible GPUs if any)
        tp_size = world_size if world_size > 0 else 1
        print(f"[DeepSpeed] init_inference tp_size={tp_size}")
        self.model = deepspeed.init_inference(
            base_model,
            tensor_parallel={"tp_size": tp_size},
            dtype=dtype,
            replace_with_kernel_inject=False,
            enable_cuda_graph=False  # keep flexibility / lower memory
        )

        # Expose backbone pieces & wrap only selected layers
        full = self.model.module if hasattr(self.model, "module") else self.model
        backbone = full.model
        self.lm_head = full.lm_head
        self.norm = backbone.norm

        total_layers = len(backbone.layers)
        if selected_layers is None:
            # Sensible default: sparse sampling across depth
            # (early, mid, late) to reduce overhead yet keep signal
            sel = {2, 6, 12, 18, total_layers - 1}
        else:
            sel = set(int(i) for i in selected_layers if 0 <= i < total_layers)
            if not sel:
                sel = {total_layers - 1}
        self.selected_layers = sorted(sel)
        print(f"[Wrap] Selected layers: {self.selected_layers} / total {total_layers}")

        new_blocks = []
        for i, layer in enumerate(backbone.layers):
            if i in sel:
                new_blocks.append(
                    BlockOutputWrapper(
                        layer,
                        self.lm_head,
                        self.norm,
                        collect_attn_mech=collect_attn_mech,
                        collect_mlp=collect_mlp,
                        collect_block=collect_block
                    )
                )
            else:
                new_blocks.append(layer)
        backbone.layers = torch.nn.ModuleList(new_blocks)
        print("[Wrap] Layer wrapping complete")

        torch.cuda.empty_cache()
        gc.collect()

        import atexit
        def _cleanup():
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"[Cleanup] Error: {e}")
        atexit.register(_cleanup)

    # -----------------------
    # Basic convenience APIs
    # -----------------------
    def generate_text(self, prompt: str, max_length: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.batch_decode(
            out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def get_logits(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def set_add_attn_output(self, layer: int, add_output: torch.Tensor):
        self.model.module.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer: int):
        return self.model.module.model.layers[layer].get_attn_activations()

    def reset_all(self):
        # Reset only wrapped layers
        for i in self.selected_layers:
            blk = self.model.module.model.layers[i]
            if isinstance(blk, BlockOutputWrapper):
                blk.reset()

    # --------------------------------------
    # Vectorized decode (token-major output)
    # --------------------------------------
    def _vectorized_topk_decode(
        self,
        hidden_activations: torch.Tensor,
        topk: int = 10
    ) -> List[List[Tuple[str, int]]]:
        """
        hidden_activations: [B, T, H] or [T, H] or [1, T, H]
        Returns per-position top-k token strings with integer % probs.
        """
        with torch.inference_mode():
            if hidden_activations.dim() == 2:        # [T, H] -> [1, T, H]
                hidden_activations = hidden_activations.unsqueeze(0)
            if hidden_activations.dim() == 3 and hidden_activations.size(0) != 1:
                # assume batch=1 for generation case; squeeze otherwise
                hidden_activations = hidden_activations[:1]

            H = hidden_activations.contiguous()      # [1, T, H]
            target_dtype = self.lm_head.weight.dtype
            normed = self.norm(H).to(target_dtype)   # [1, T, H]
            logits = self.lm_head(normed)            # [1, T, V]
            top_vals, top_idx = torch.topk(logits, k=topk, dim=-1)   # [1, T, k]
            probs = torch.softmax(top_vals, dim=-1)                  # [1, T, k]

            top_idx_cpu = top_idx.squeeze(0).cpu()                   # [T, k]
            probs_cpu = (probs.squeeze(0) * 100).to(torch.int16).cpu()

            T, K = top_idx_cpu.shape
            flat_ids = top_idx_cpu.reshape(-1, 1).tolist()
            decoded = self.tokenizer.batch_decode(
                flat_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            decoded = np.array(decoded, dtype=object).reshape(T, K)

            return [
                list(zip(row_tokens, row_probs))
                for row_tokens, row_probs in zip(decoded.tolist(), probs_cpu.tolist())
            ]

    # --------------------------------------
    # Efficient one-pass generate + probing
    # --------------------------------------
    def generate_with_probing(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        topk: int = 3,
        threshold: int = 3,  # kept for API compatibility; not used here
        print_details: bool = False,
        collect_attn_mech: bool = True,
        collect_mlp: bool = True,
        collect_block: bool = True,
        selected_layers: List[int] = [10,15,25,35,79],
        **kwargs
    ):
        """
        Runs a single generate() call, stores last-token hidden slices per selected
        layer & position, then vector-decodes all positions at once.
        Returns a single-step record with token-major decoded activations.
        """
        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=None,
                use_cache=True
            )

        decoded_text = self.tokenizer.decode(
            gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Build token-major decodes for each selected layer/component
        all_layers_data: Dict[int, Dict[str, List[List[Tuple[str, int]]]]] = {}
        base = self.model.module.model
        for i in self.selected_layers:
            layer = base.layers[i]
            layer_data: Dict[str, List[List[Tuple[str, int]]]] = {}

            if collect_attn_mech and getattr(layer, "attn_mech_output_unembedded", None):
                H = torch.cat(layer.attn_mech_output_unembedded, dim=1)  # [1, T, H]
                layer_data["attention_mechanism"] = self._vectorized_topk_decode(H, topk=topk)

            if collect_mlp and getattr(layer, "mlp_output_unembedded", None):
                H = torch.cat(layer.mlp_output_unembedded, dim=1)
                layer_data["mlp_output"] = self._vectorized_topk_decode(H, topk=topk)

            if collect_block and getattr(layer, "block_output_unembedded", None):
                H = torch.cat(layer.block_output_unembedded, dim=1)
                layer_data["block_output"] = self._vectorized_topk_decode(H, topk=topk)

            all_layers_data[i] = layer_data

        if print_details:
            print("\n[generate_with_probing] Completed. Collected layers/components:")
            for li, comp in all_layers_data.items():
                print(f"  Layer {li}: {list(comp.keys())}")

        # Single PredictionStep for the full generation (token-major)
        prediction_step = {
            "step_idx": 0,
            "input_text": prompt,
            "predicted_token": decoded_text[len(prompt):],  # full continuation
            "all_layers_data": all_layers_data,
            "important_layers": {}
        }

        # Clean buffers for next call
        self.reset_all()
        torch.cuda.empty_cache()
        gc.collect()

        return [prediction_step]
