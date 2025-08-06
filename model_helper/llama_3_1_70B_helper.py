import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Tuple, Dict
from activation_analyzer import ActivationAnalyzer, PredictionStep
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
        self.block.mlp = MLPWrapper(self.block.mlp)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_mech_output_unembedded = []
        self.intermediate_res_unembedded = []
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

        
        if self.collect_intermediate_res:
            attn_output = self.block.self_attn.activations
            # last_attn = attn_output[:, -1:, :] if attn_output.ndim == 3 else attn_output.unsqueeze(1)
            # self.intermediate_resx_unembedded = attn_output
            self.attn_mech_output_unembedded.append(attn_output[:, -1:, :])
        # mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
       
        if self.collect_mlp is not None:
            mlp_output = self.block.mlp.activations
            # last_mlp = mlp_output[:, -1:, :] if mlp_output.ndim == 3 else mlp_output.unsqueeze(1)
            self.mlp_output_unembedded.append(mlp_output[:, -1:, :])


        if self.collect_block:
            # last_hidden = hidden_states[:, -1:, :] if hidden_states.ndim == 3 else hidden_states.unsqueeze(1)
            # self.block_output_unembedded = self.lm_head(self.norm(last_hidden))
            logits = hidden_states
            # logits = self.lm_head(self.norm(hidden_states))
            # self.block_output_unembedded.append(logits[:, -1:, :].detach().cpu())
            self.block_output_unembedded.append(logits[:, -1:, :])
            # self.block_output_unembedded = hidden_states

        return output

    def attn_add_tensor(self, tensor):
        self.block.self_attn.add_tensor = tensor

    def reset(self):
        self.block.self_attn.reset()
        self.block.mlp.reset()
        # Buffers to store unembedded outputs
        self.attn_mech_output_unembedded = []
        self.intermediate_res_unembedded = []
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
    def __init__(self, use_local=True, local_path="", token=None, collect_attn_mech=True, collect_intermediate_res=True, collect_mlp=True, collect_block=True):
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
        # Base dir = project root (two levels up from this file)
        base_dir = Path(__file__).resolve().parent.parent
        localPath = base_dir / "output" / "cache" / "models--meta-llama--Meta-Llama-3.1-70B" / "snapshots" / "349b2ddb53ce8f2849a6c168a81980ab25258dac"
        tokenizer, model = self.load_model_and_tokenizer(
            local_path= localPath,  
            cache_dir="output/cache",
            torch_dtype="bfloat16",
            token =  token
        )
        self.tokenizer = tokenizer

        print("Initializing DeepSpeed inference...")
        try:
            self.model = deepspeed.init_inference(
                model,
                tensor_parallel={"tp_size": world_size},
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

    def load_model_and_tokenizer(
    self,    
    model_name="meta-llama/Meta-Llama-3.1-70B",
    local_path=None,
    cache_dir="output/cache",
    torch_dtype="bfloat16",
    token = None,
    trust_remote_code=True
    ):
        """
        Loads a tokenizer and model from either:
        - Local path (if it exists)
        - Hugging Face Hub into a cache_dir (creates it if missing)

        Parameters:
        ----------
        model_name : str
            Hugging Face model ID for downloading.
        local_path : str or None
            Path to a locally stored model. If None, only cache_dir is used.
        cache_dir : str
            Directory for caching downloaded models.
        torch_dtype : str
            Data type for model weights ("bfloat16", "float16", etc.).
        trust_remote_code : bool
            Allow execution of model code from the hub (needed for custom models).
        """

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        if local_path and os.path.exists(local_path):
            print(f"✅ Loading model from local path: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=trust_remote_code, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype= torch.bfloat16,
                trust_remote_code=trust_remote_code,
                token=token
            )
        else:
            if local_path:
                print(f"⚠️ Local path not found: {local_path}. Downloading from Hugging Face Hub...")
            else:
                print(f"📥 Downloading model from Hugging Face Hub...")

            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype= torch.bfloat16,
                trust_remote_code=trust_remote_code,
                token=token
            )

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model    

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
            layer.attn_mech_output_unembedded = None
            layer.intermediate_res_unembedded = None
            layer.mlp_output_unembedded = None
            layer.block_output_unembedded = None

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        softmaxed = torch.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = torch.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        print(label, list(zip(tokens, probs_percent)))

    def collect_decoded_activations(
    self,
    hidden_activations: torch.Tensor,
    topk: int = 10
    ) -> List[List[Tuple[str, int]]]:
        """
        Return top-k decoded tokens for **each token** in the sequence.
        """
        token_outputs = []
        seq_len = hidden_activations.shape[1]
        for t in range(seq_len):
            # 1. Extract the hidden state for token position t
            hid_t = hidden_activations[:, t, :]  # shape [1, hidden_dim]
            # 2. Apply final layer normalization
            normed = self.norm(hid_t) 
            normed = normed.unsqueeze(1).contiguous()
            logits = self.lm_head(normed)  
            logits = logits.squeeze(1)
            # 4. Get top-K predictions
            values, indices = torch.topk(logits[0], topk, dim=-1)
            # 5. Calculate relative probabilities for top-K (optional)
            probs = torch.nn.functional.softmax(values, dim=-1)  # softmax over just top-K values
            probs_percent = [int(p.item() * 100) for p in probs]
            # 6. Decode token IDs to strings
            tokens = self.tokenizer.batch_decode(indices.unsqueeze(1))
            token_outputs.append(list(zip(tokens, probs_percent)))
        return token_outputs

    

    def generate_with_probing(
    self,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.3,    
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

        tokenized = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)



        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens = max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=None,
                use_cache=True
            )

        # decode final output
        decoded_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )


        all_layers_data = {}
        for i, layer in enumerate(self.model.module.model.layers):
            layer_data = {}
            if collect_attn_mech and layer.attn_mech_output_unembedded is not None:
                layer_data['attention_mechanism'] = self.collect_decoded_activations(
                    layer.attn_mech_output_unembedded, topk=topk
                )
            if collect_intermediate_res and layer.intermediate_res_unembedded is not None:
                layer_data['intermediate_residual'] = self.collect_decoded_activations(
                    layer.intermediate_res_unembedded, topk=topk
                )
            if collect_mlp and layer.mlp_output_unembedded is not None:
                layer_data['mlp_output'] = self.collect_decoded_activations(
                     layer.mlp_output_unembedded, topk=topk
                )
            if collect_block and layer.block_output_unembedded is not None:
                activations = torch.cat(layer.block_output_unembedded, dim=1)
                layer_data['block_output'] = self.collect_decoded_activations(
                    activations, topk=topk
                )
            all_layers_data[i] = layer_data
        # important_layers = ActivationAnalyzer.filter_important_layers(all_layers_data, threshold=threshold)
        important_layers = {}
    
        prediction_step: PredictionStep = {
            "step_idx": 0,
            "input_text": prompt,
            "predicted_token": decoded_text[len(prompt):],  # only new tokens
            "all_layers_data": all_layers_data,
            "important_layers": important_layers
        }


        if print_details:
            print(f"\nStep {step_idx + 1}: Predicted token: {predicted_token}")
            print(f"Current text: {current_text}")
            print("\nImportant layers for this prediction:")
            for layer_idx, components in important_layers.items():
                print(f"\nLayer {layer_idx}:")
                for component_name, tokens_probs in components.items():
                    top_preds = ActivationAnalyzer.get_top_predictions(components, top_k=5)
                    print(f"  {component_name}: {top_preds[component_name]}")            
        print("Inference Complete")
        self.reset_all()
        return [prediction_step]
