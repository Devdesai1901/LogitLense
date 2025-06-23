import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Dict
from LogitLens4LLMs.activation_analyzer import ActivationAnalyzer, PredictionStep
from deepspeed.runtime.utils import see_memory_usage
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

# Wrapper around transformer block to collect various intermediate outputs
class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, lm_head, norm, collect_attn_mech: bool = True, collect_intermediate_res: bool = True, collect_mlp: bool = True, collect_block: bool = True):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        self.collect_attn_mech = collect_attn_mech
        self.collect_intermediate_res = collect_intermediate_res
        self.collect_mlp = collect_mlp
        self.collect_block = collect_block
        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def forward(self, x, past_key_value=None, attention_mask=None, position_ids=None, **kwargs):
        output = self.block(x, past_key_value=past_key_value, attention_mask=attention_mask, 
                            position_ids=position_ids, **kwargs)

        # Collect intermediate residual stream (after attention)                    
        attn_output = output[0] + x
        if self.collect_intermediate_res:
            self.intermediate_res_unembedded = self.lm_head(self.norm(attn_output))
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        if self.collect_mlp:
            self.mlp_output_unembedded = self.lm_head(self.norm(mlp_output))
        if self.collect_block:
            self.block_output_unembedded = self.lm_head(self.norm(output[0]))
        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        # Buffers to store unembedded outputs
        self.attn_mech_output_unembedded = None
        self.intermediate_res_unembedded = None
        self.mlp_output_unembedded = None
        self.block_output_unembedded = None

    def get_attn_activations(self):
        return self.block.self_attn.activations

# Ensure multiprocessing safety
multiprocessing.set_start_method("spawn", force=True)
# Allow flexible memory growth in CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Helper class to load and manage LLaMA 3.1–70B model with DeepSpeed and logit lens
class Llama3_1_70BHelper:
    def __init__(self, use_local=True, local_path="~/LogitLens4LLMs/output/cache/models--meta-llama--Meta-Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac", token=None, collect_attn_mech=True, collect_intermediate_res=True, collect_mlp=True, collect_block=True):
        print("Initializing Llama-3.1-70B Helper...")
        see_memory_usage("Before initialization", force=True)
        
        # Setup for distributed GPU
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 4))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")
        print(f"Using device: {self.device}, Local rank: {local_rank}, World size: {world_size}")

        try:
            # Initialize DeepSpeed distributed training
            if not torch.distributed.is_initialized():
                deepspeed.init_distributed(dist_backend="nccl")
            print("DeepSpeed distributed initialized")
        except Exception as e:
            print(f"Error initializing DeepSpeed distributed: {e}")
            raise
        
        
        model_id = "meta-llama/Meta-Llama-3.1-70B"
        cache_dir = os.path.expanduser("~/LogitLens4LLMs/output/cache")
        os.makedirs(cache_dir, exist_ok=True)

        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path if use_local else model_id,
                use_fast=True,
                token=token,
                trust_remote_code=True,
                cache_dir=cache_dir,
                truncation=True
                # max_length=5
            )
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        print("Loading model with BF16...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_path if use_local else model_id,
                cache_dir=cache_dir,
                torch_dtype=torch.bfloat16
            )
            see_memory_usage("After loading pretrained weights", force=True)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        print("model", model)
        print("Initializing DeepSpeed inference...")
        try:
            self.model = deepspeed.init_inference(
                model,
                tensor_parallel={"tp_size": world_size},
                dtype=torch.bfloat16,
                replace_with_kernel_inject=False
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
            base_model.layers = torch.nn.ModuleList([
                BlockOutputWrapper(
                    layer,
                    lm_head,
                    norm,
                    collect_attn_mech=collect_attn_mech,
                    collect_intermediate_res=collect_intermediate_res,
                    collect_mlp=collect_mlp,
                    collect_block=collect_block
                )
                for layer in base_model.layers
            ])
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
        for layer in self.model.module.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))

    def collect_decoded_activations(self, decoded_activations: torch.Tensor, topk: int = 10) -> List[Tuple[str, int]]:
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent))

    def decode_all_layers_to_dict(
        self, 
        text: str, 
        topk: int = 10, 
        collect_attn_mech: bool = True,
        collect_intermediate_res: bool = True, 
        collect_mlp: bool = True,
        collect_block: bool = True
    ) -> Dict[int, Dict[str, List[Tuple[str, int]]]]:
        self.get_logits(text)
        all_layers_data = {}
        for i, layer in enumerate(self.model.module.model.layers):
            layer_data = {}
            if collect_attn_mech:
                layer_data['attention_mechanism'] = self.collect_decoded_activations(
                    layer.attn_mech_output_unembedded, topk=topk)
            if collect_intermediate_res:
                layer_data['intermediate_residual'] = self.collect_decoded_activations(
                    layer.intermediate_res_unembedded, topk=topk)
            if collect_mlp:
                layer_data['mlp_output'] = self.collect_decoded_activations(
                    layer.mlp_output_unembedded, topk=topk)
            if collect_block:
                layer_data['block_output'] = self.collect_decoded_activations(
                    layer.block_output_unembedded, topk=topk)
            all_layers_data[i] = layer_data
        return all_layers_data

    def decode_all_layers(self, text, topk=10, print_attn_mech=True, 
                         print_intermediate_res=True, print_mlp=True, print_block=True):
        self.get_logits(text)
        for i, layer in enumerate(self.model.module.model.layers):
            print(f'Layer {i}: Decoded intermediate outputs')
            if print_attn_mech:
                self.print_decoded_activations(layer.attn_mech_output_unembedded, 
                                            'Attention mechanism', topk=topk)
            if print_intermediate_res:
                self.print_decoded_activations(layer.intermediate_res_unembedded, 
                                            'Intermediate residual stream', topk=topk)
            if print_mlp:
                self.print_decoded_activations(layer.mlp_output_unembedded, 
                                            'MLP output', topk=topk)
            if print_block:
                self.print_decoded_activations(layer.block_output_unembedded, 
                                            'Block output', topk=topk)

    def generate_with_probing(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.3,    
        # 0. ascended_14,
        topk: int = 10,
        threshold: int = 3,
        print_details: bool = True,
        collect_attn_mech: bool = True,
        collect_intermediate_res: bool = True, 
        collect_mlp: bool = True,
        collect_block: bool = True
    ) -> List[PredictionStep]:
        prediction_steps = []
        current_text = prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        for step_idx in range(max_new_tokens):
            all_layers_data = self.decode_all_layers_to_dict(current_text, topk=topk, collect_attn_mech=collect_attn_mech, collect_intermediate_res=collect_intermediate_res, collect_mlp=collect_mlp, collect_block=collect_block)
            with torch.no_grad():
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumsum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
                next_token_id = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
            predicted_token = self.tokenizer.decode(next_token_id[0])
            important_layers = ActivationAnalyzer.filter_important_layers(all_layers_data, threshold=threshold)
            step_data = {
                "step_idx": step_idx,
                "input_text": current_text,
                "predicted_token": predicted_token,
                "all_layers_data": all_layers_data,
                "important_layers": important_layers
            }
            prediction_steps.append(step_data)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            current_text += predicted_token
            if print_details:
                print(f"\nStep {step_idx + 1}: Predicted token: {predicted_token}")
                print(f"Current text: {current_text}")
                print("\nImportant layers for this prediction:")
                for layer_idx, components in important_layers.items():
                    print(f"\nLayer {layer_idx}:")
                    for component_name, tokens_probs in components.items():
                        top_preds = ActivationAnalyzer.get_top_predictions(components, top_k=5)
                        print(f"  {component_name}: {top_preds[component_name]}")
            else:
                print(f"Step {step_idx + 1}: Generated '{predicted_token}'")
        torch.cuda.empty_cache()
        gc.collect()
        return prediction_steps