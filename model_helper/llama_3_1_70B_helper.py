import torch
import argparse
import numpy as np
from model_helper.config import load_config
from model_helper.model_io import load_tokenizer_and_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Dict
from activation_analyzer import ActivationAnalyzer70B, PredictionStep
from deepspeed.runtime.utils import see_memory_usage
from pathlib import Path
import os
import deepspeed
import multiprocessing
import gc

# Wrapper for capturing or modifying attention outputs
class AttnWrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn     # Original attention layer
        self.activations = None     # Stores output activations
        self.add_tensor = None      # Optional tensor to add 

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        if self.add_tensor is not None:
            output = (output[0] + self.add_tensor,) + output[1:]
        self.activations = output[0]
        return output

    def reset(self):
        self.activations = None
        self.add_tensor = None

# Wrapper for capturing or modifying MLP outputs
class MLPWrapper(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.mlp(*args, **kwargs)
        self.activations = output
        return output

    def reset(self):
        self.activations = None


# Wrapper around transformer block to collect various intermediate outputs
class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, norm, collect_attn_mech: bool = True, collect_mlp: bool = True, collect_block: bool = True):
        super().__init__()

        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        
        self.collect_attn_mech = collect_attn_mech
        # self.collect_intermediate_res = collect_intermediate_res
        self.collect_mlp = collect_mlp
        self.collect_block = collect_block

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = []
        # self.intermediate_res_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        output = self.block(x, past_key_value=past_key_value, attention_mask=attention_mask, 
                            position_ids=position_ids, **kwargs)

        # Collect intermediate residual stream (after attention)

        # Get hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        
        if self.collect_attn_mech:
            attn_output = self.block.self_attn.activations
            self.attn_mech_output_unembedded.append(attn_output[:, -1:, :])
      
       
        if self.collect_mlp:
            mlp_output = self.block.mlp.activations
            self.mlp_output_unembedded.append(mlp_output[:, -1:, :])


        if self.collect_block:
            logits = hidden_states
            self.block_output_unembedded.append(logits[:, -1:, :])

        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        # Buffers to store unembedded outputs
        self.attn_mech_output_unembedded = []
        # self.intermediate_res_unembedded = []
        self.mlp_output_unembedded = []
        self.block_output_unembedded = []


    def get_attn_activations(self):
        return self.block.self_attn.activations

# Ensure multiprocessing safety
multiprocessing.set_start_method("spawn", force=True)
# Allow flexible memory growth in CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Helper class to load and manage LLaMA 3.1–70B model with DeepSpeed and logit lens
class Llama3_1_70BHelper:
    def __init__(self, cfg, collect_attn_mech=True, collect_mlp=True, collect_block=True, selected_layers: List[int] = [10,15,25,35,79]):
        print("Initializing Llama-3.1-70B Helper...")
        see_memory_usage("Before initialization", force=True)
        
        # Setup for distributed GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % world_size)
        self.device = torch.device(f"cuda:{local_rank % world_size}")
        print(f"Using device: {self.device}, Local rank: {local_rank}, World size: {world_size}")

        try:
            # Initialize DeepSpeed distributed training
            if not torch.distributed.is_initialized():
                deepspeed.init_distributed(dist_backend="nccl")
            print("DeepSpeed distributed initialized")
        except Exception as e:
            print(f"Error initializing DeepSpeed distributed: {e}")
            raise
        
        tokenizer, base_model, source = load_tokenizer_and_model(cfg)
        print(f"[ModelLoader] Loaded from: {source}")
        self.tokenizer = tokenizer
        self.selected_layers = set(selected_layers or [])  # save for later


        tp = (world_size 
        if str(cfg.get("tensor_parallel_size", "auto")).lower() == "auto" 
        else int(cfg["tensor_parallel_size"]))

        print("Initializing DeepSpeed inference...")
        try:
            self.model = deepspeed.init_inference(
                base_model,
                tensor_parallel={"tp_size": tp},
                dtype=torch.bfloat16,
                replace_with_kernel_inject=False,
                enable_cuda_graph=False  # Disable CUDA graph for memory savings
                  # Enable activation checkpointing for transformer blocks
            )
            see_memory_usage("After DeepSpeed inference initialization", force=True)
            print("DeepSpeed inference initialization complete")
        except Exception as e:
            print(f"Error initializing DeepSpeed inference: {e}")
            raise

        try:
            full_model = self.model.module if hasattr(self.model, 'module') else self.model
            base_model = full_model.model
            lm_head = full_model.lm_head
            norm = base_model.norm

            self.lm_head = lm_head
            self.norm = norm
            new_layers = []
            for i, layer in enumerate(base_model.layers):
                if i in self.selected_layers:
                    print("using BlockOut put wrappering ")
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
            base_model.layers = torch.nn.ModuleList(new_layers)
            print("All layers wrapped successfully")
        except Exception as e:
            print(f"Error wrapping layers: {e}")
            raise

        torch.cuda.empty_cache()
        gc.collect()

        import atexit
        def cleanup():
            try:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        atexit.register(cleanup)


    def generate_text(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generate_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits

    def set_add_attn_output(self, layer, add_output):
        self.model.module.model.layers[layer].attn_add_tensor(add_output)

    def get_attn_activations(self, layer):
        return self.model.module.model.layers[layer].get_attn_activations()

    def reset_all(self):
        if not self.selected_layers:
            return
        for i in self.selected_layers:
            layer = self.model.module.model.layers[i]
            layer.reset()
            layer.attn_mech_output_unembedded = []
            layer.mlp_output_unembedded = []
            layer.block_output_unembedded = []


    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))



    def collect_decoded_activations(
    self,
    hidden_activations: torch.Tensor,
    topk: int = 3
    ) -> List[List[Tuple[str, int]]]:
        """
        Vectorized logit-lens decode for ALL tokens at once:
        H -> norm -> lm_head -> topk (per position)
        Returns: List[length T] of [(token_str, prob_percent), ...] length topk
        """
        with torch.inference_mode():
            if hidden_activations.dim() == 2:   # [T, H] -> [1, T, H]
                hidden_activations = hidden_activations.unsqueeze(0)

            # Ensure shape [1, T, H] and dtype/device match lm_head weights
            H = hidden_activations.contiguous()
            target_dtype = self.lm_head.weight.dtype
            # LLaMA's final norm (RMSNorm) supports [B, T, H] directly
            normed = self.norm(H).to(target_dtype)          # [1, T, H]
            logits = self.lm_head(normed)                   # [1, T, V]
            # Top-k per token position (vectorized)
            topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)  # [1, T, k]
            probs = torch.softmax(topk_vals, dim=-1)                   # [1, T, k]

            # Move small results to CPU for token string conversion
            topk_idx_cpu = topk_idx.squeeze(0).cpu()               # [T, k]
            probs_cpu = (probs.squeeze(0) * 100).to(torch.int16).cpu()  # [T, k]

            # Convert IDs -> tokens efficiently
            T, K = topk_idx_cpu.shape
            flat_ids = topk_idx_cpu.reshape(-1, 1).tolist()
            decoded = self.tokenizer.batch_decode(
            flat_ids,
            skip_special_tokens=True,              # keep False if you want to see specials
            clean_up_tokenization_spaces=False     # don't collapse spaces
            )
            # Back to [T, k]
            decoded = np.array(decoded, dtype=object).reshape(T, K)

            decoded_ll = decoded.tolist()
            probs_ll = probs_cpu.tolist()

            # One comprehension (row-wise zip), no inner for-loop
            outputs = [list(zip(row_tokens, row_probs))
                    for row_tokens, row_probs in zip(decoded_ll, probs_ll)]

            return outputs

    def generate_with_probing(
    self,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.3,    
    topk: int = 3,
    threshold: int = 3,
    collect_attn_mech: bool = True,
    collect_mlp: bool = True,
    collect_block: bool = True,
    selected_layers: List[int] = [10,15,25,35,79]
) -> List[PredictionStep]:
        # Determine which layers to unembed (default to first 5 layers)
        total_layers = len(self.model.module.model.layers)
        if selected_layers is None:
            # Default: first 5 layers or all if fewer
            selected_layers = list(range(min(5, total_layers)))
        
        prediction_steps = []
        current_text = prompt

        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        sel = sorted(self.selected_layers) if self.selected_layers else sorted(set(selected_layers))
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=None,
                use_cache=True
            )

        # Decode the final text
        decoded_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        all_layers_data = {}
        # Loop over each layer and collect decoded activations only for selected layers
        for i in sel:
            layer = self.model.module.model.layers[i]
            layer_data = {}
            # Stack CPU-captured slices into [T, H] once, then decode
            if collect_attn_mech and layer.attn_mech_output_unembedded is not None:
                activations = torch.cat(layer.attn_mech_output_unembedded, dim=1)
                layer_data['attention_mechanism'] = self.collect_decoded_activations(
                    activations, topk=topk
                )
            if collect_mlp and layer.mlp_output_unembedded is not None:
                activations = torch.cat(layer.mlp_output_unembedded, dim=1)
                layer_data['mlp_output'] = self.collect_decoded_activations(
                     activations, topk=topk
                )
            if collect_block and layer.block_output_unembedded is not None:
                activations = torch.cat(layer.block_output_unembedded, dim=1)
                layer_data['block_output'] = self.collect_decoded_activations(
                    activations, topk=topk
                )
            all_layers_data[i] = layer_data

        # (The rest of the method remains unchanged: building PredictionStep, resetting, etc.)
        prediction_step: PredictionStep = {
            "step_idx": 0,
            "input_text": prompt,
            "predicted_token": decoded_text[len(prompt):],
            "all_layers_data": all_layers_data,
            "important_layers": {}
        }
        print("Inference Complete")
        self.reset_all()
        return [prediction_step]
