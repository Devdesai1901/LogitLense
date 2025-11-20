# model_helper/qwen3_14b_helper.py
import os, gc, torch, deepspeed, numpy as np
from typing import List, Tuple, Dict
from deepspeed.runtime.utils import see_memory_usage
from activation_analyzer import PredictionStep
from model_helper.model_io import load_tokenizer_and_model

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Reuse your wrappers with light generalization ----
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn): 
        super().__init__(); self.attn = attn; self.activations=None; self.add_tensor=None
    def forward(self, *args, **kwargs):
        out = self.attn(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        if self.add_tensor is not None: h = h + self.add_tensor
        self.activations = h
        return (h,) + out[1:] if isinstance(out, tuple) else h
    def reset(self): self.activations=None; self.add_tensor=None

class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp): 
        super().__init__(); self.mlp=mlp; self.activations=None
    def forward(self, *a, **k): 
        y = self.mlp(*a, **k); self.activations = y; return y
    def reset(self): self.activations=None

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, norm,
                 collect_attn_mech=True, collect_mlp=True, collect_block=True):
        super().__init__()
        # core refs
        self.block = block
        self.lm_head = lm_head
        self.norm = norm

        self.collect_attn_mech = collect_attn_mech
        self.collect_mlp = collect_mlp
        self.collect_block = collect_block

        # minimal attrs HF may read
        self.attention_type = getattr(block, "attention_type", None)
        self.config = getattr(block, "config", None)
        self.layer_idx = getattr(block, "layer_idx", None)

        # wrap submodules for capture
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = getattr(self.block, "post_attention_layernorm", None)

        # collectors
        self.attn_mech_output_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []

    def forward(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        if self.collect_attn_mech and self.block.self_attn.activations is not None:
            self.attn_mech_output_unembedded.append(self.block.self_attn.activations[:, -1:, :])
        if self.collect_mlp and self.block.mlp.activations is not None:
            self.mlp_output_unembedded.append(self.block.mlp.activations[:, -1:, :])
        if self.collect_block:
            self.block_output_unembedded.append(hidden[:, -1:, :])
        return out

    def attn_add_tensor(self, t): self.block.self_attn.add_tensor = t

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        self.attn_mech_output_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []

    def get_attn_activations(self):
        return self.block.self_attn.activations


# ---- Helper (14B dense text-only) ----
class Qwen_3_14B_Helper:
    """
    Works with dense text-only 14B variants, e.g.:
      - Qwen/Qwen3-14B-Instruct
      - Qwen/Qwen2-14B-Instruct
    """
    def __init__(self, cfg, collect_attn_mech=True, collect_mlp=True, collect_block=True, selected_layers: List[int] = None):
        print("Initializing Qwen 14B (dense) Helper...")
        see_memory_usage("Before init", force=True)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % world_size)
        self.device = torch.device(f"cuda:{local_rank % world_size}")

        if not torch.distributed.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")

        self.tokenizer, base_model, _source = load_tokenizer_and_model(cfg)
        self.selected_layers = set(selected_layers or [])

        tp = (world_size if str(cfg.get("tensor_parallel_size", "auto")).lower()=="auto"
              else int(cfg["tensor_parallel_size"]))
        self.model = deepspeed.init_inference(
            base_model,
            tensor_parallel={"tp_size": tp},
            dtype=torch.bfloat16,
            replace_with_kernel_inject=False,
            enable_cuda_graph=False,
        )
        see_memory_usage("After DS init", force=True)

        # ---- Locate the TEXT backbone and layers ----
        full = self.model.module if hasattr(self.model, "module") else self.model
        # Dense text-only Qwen typically exposes `full.model.layers` (+ optional aliases)
        candidates = [
            getattr(getattr(full, "model", full), "text_model", None),     # if present
            getattr(getattr(full, "model", full), "language_model", None),# if present
            getattr(full, "model", None),
            getattr(full, "language_model", None),
        ]
        text = next((c for c in candidates if c is not None and hasattr(c, "layers")), None)
        if text is None:
            raise RuntimeError("Could not locate Qwen 14B text backbone with a .layers ModuleList.")

        self.lm_head = getattr(full, "lm_head", None) or getattr(full, "language_model_head", None)
        self.norm = getattr(text, "norm", None) or getattr(getattr(full, "model", full), "norm", None)
        if self.lm_head is None or self.norm is None:
            raise RuntimeError("Expected lm_head and norm on Qwen 14B dense model.")

        # ---- Wrap only selected layers ----
        new_layers = []
        for i, layer in enumerate(text.layers):
            if i in self.selected_layers:
                new_layers.append(
                    BlockOutputWrapper(
                        layer, self.lm_head, self.norm,
                        collect_attn_mech=collect_attn_mech,
                        collect_mlp=collect_mlp,
                        collect_block=collect_block
                    )
                )
            else:
                new_layers.append(layer)
        text.layers = torch.nn.ModuleList(new_layers)
        print("Qwen 14B text layers wrapped.")

        torch.cuda.empty_cache(); gc.collect()

        import atexit
        @atexit.register
        def _cleanup():
            try:
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
                torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                print(f"Cleanup error: {e}")

    # ---- Decoding (same as your 32B path) ----
    def collect_decoded_activations(self, H: torch.Tensor, topk: int = 3):
        with torch.inference_mode():
            if H.dim()==2: H = H.unsqueeze(0)
            H = H.contiguous()
            target_dtype = self.lm_head.weight.dtype
            normed = self.norm(H).to(target_dtype)
            logits = self.lm_head(normed)
            topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            topk_idx_cpu = topk_idx.squeeze(0).cpu()
            probs_cpu = (probs.squeeze(0)*100).to(torch.int16).cpu()
            T, K = topk_idx_cpu.shape
            flat_ids = topk_idx_cpu.reshape(-1, 1).tolist()
            decoded = self.tokenizer.batch_decode(flat_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            decoded = np.array(decoded, dtype=object).reshape(T, K).tolist()
            probs_ll = probs_cpu.tolist()
            return [list(zip(tok_row, prob_row)) for tok_row, prob_row in zip(decoded, probs_ll)]

    def reset_all(self):
        full = self.model.module if hasattr(self.model, "module") else self.model
        text = (getattr(getattr(full, "model", full), "text_model", None) or
                getattr(getattr(full, "model", full), "language_model", None) or
                getattr(full, "model", None))
        if not text or not self.selected_layers: return
        for i in self.selected_layers:
            blk = text.layers[i]
            if hasattr(blk, "reset"): blk.reset()
            blk.attn_mech_output_unembedded=[]; blk.mlp_output_unembedded=[]; blk.block_output_unembedded=[]

    def generate_with_probing(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.3,
        top_p: float = 0.95,
        topk: int = 3,
        threshold: int = 3,
        collect_attn_mech: bool = True,
        collect_mlp: bool = True,
        collect_block: bool = True,
        selected_layers: List[int] = None
    ) -> List[PredictionStep]:
        sel = sorted(self.selected_layers) if self.selected_layers else sorted(set(selected_layers or []))

        tok = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = tok.input_ids.to(self.device)
        attn_mask = tok.attention_mask.to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=None,
                use_cache=True
            )
        decoded_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # collect per-layer unembedded streams
        full = self.model.module if hasattr(self.model, "module") else self.model
        text = (getattr(getattr(full, "model", full), "text_model", None) or
                getattr(getattr(full, "model", full), "language_model", None) or
                getattr(full, "model", None))
        all_layers_data: Dict[int, Dict[str, List[Tuple[str,int]]]] = {}
        for i in sel:
            L = text.layers[i]
            layer_data = {}
            if collect_attn_mech and getattr(L, "attn_mech_output_unembedded", None):
                A = torch.cat(L.attn_mech_output_unembedded, dim=1)
                layer_data["attention_mechanism"] = self.collect_decoded_activations(A, topk=topk)
            if collect_mlp and getattr(L, "mlp_output_unembedded", None):
                M = torch.cat(L.mlp_output_unembedded, dim=1)
                layer_data["mlp_output"] = self.collect_decoded_activations(M, topk=topk)
            if collect_block and getattr(L, "block_output_unembedded", None):
                B = torch.cat(L.block_output_unembedded, dim=1)
                layer_data["block_output"] = self.collect_decoded_activations(B, topk=topk)
            all_layers_data[i] = layer_data

        self.reset_all()
        return [{
            "step_idx": 0,
            "input_text": prompt,
            "predicted_token": decoded_text[len(prompt):],
            "all_layers_data": all_layers_data,
            "important_layers": {}
        }]
