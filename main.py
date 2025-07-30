import sys
import os
import torch
import gc
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings  # Import warnings module
from enum import Enum
from LogitLens4LLMs.model_factory import ModelFactory, ModelType
from LogitLens4LLMs.activation_analyzer import ActivationAnalyzer, PredictionStep   
# from prompt_templates import TaskConfig, Concept, CrossDomainAnalogyConfig, PromptTemplate

class ModelPath(Enum):
    """Enumeration of local model paths"""
    DEEPSEEK_LLAMA_8B = "/root/autodl-fs/models_hf/DeepSeek-R1-Distill-Llama-8B"
    DEEPSEEK_QWEN_7B = "/root/autodl-fs/models_hf/DeepSeek-R1-Distill-Qwen-7B"
    LLAMA_2_7B = "/root/autodl-fs/models_hf/Llama-2-7b"
    LLAMA_3_1_8B = "/root/autodl-fs/models_hf/Llama-3.1-8b"
    QWEN_7B = "/root/autodl-fs/models_hf/Qwen2.5-7B-Instruct"
    
    @classmethod
    def get_path(cls, model_type: ModelType) -> str:
        """Get local path corresponding to ModelType"""
        path_mapping = {
            ModelType.LLAMA_7B: cls.LLAMA_2_7B.value,
            ModelType.LLAMA_3_1_8B: cls.LLAMA_3_1_8B.value,
            ModelType.QWEN_7B: cls.QWEN_7B.value
        }
        if model_type not in path_mapping:
            raise ValueError(f"No local path found for model type {model_type}")
        return path_mapping[model_type]

# HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-3.1-8b --local-dir /root/LACM/explanation/models_hf

# warnings.filterwarnings("ignore")

def run_analysis(
    model_type: ModelType,
    use_local: bool = False,
    token: str = None,
    prompt: str = "",
    num_trials: int = 5,
    extract_middle_token_num: int = 15,
    print_details: bool = False,
    max_output_new_tokens: int = 10,
    save_output: bool = True,
    output_base_path: str = "./explanation/logit_lens",
    collect_attn_mech: bool=True,
    collect_intermediate_res: bool=True, 
    collect_mlp: bool=True,
    collect_block: bool=True
):
    """
    Run logit lens analysis on large language models, generate prediction steps and save visualization results.
    
    Args:
        model_type: Type of model to analyze
        use_local: Whether to use locally saved model
        token: HuggingFace token for model access
        prompt: Input text prompt
        num_trials: Number of generation trials to run
        extract_middle_token_num: Number of intermediate tokens to extract
        print_details: Whether to print detailed prediction info
        max_output_new_tokens: Maximum number of new tokens to generate
        save_output: Whether to save visualization results
        output_base_path: Base directory for saving outputs
    """
    local_path = None
    if use_local:
        local_path = ModelPath.get_path(model_type)
        
    model = ModelFactory.create_model(
        model_type=model_type,
        use_local=use_local,
        local_path=local_path,
        token=token,
        collect_attn_mech = collect_attn_mech,
        collect_intermediate_res = collect_intermediate_res, 
        collect_mlp = collect_mlp,
        collect_block = collect_block
    )
    analyzer = ActivationAnalyzer()
    
    print(f"\nRunning logit lens analysis:")
    print(f"Model type: {model_type.value}")
    print(f"Prompt: {prompt}\n")

    for i in range(num_trials):
        print(f"Running trial {i+1}...")
        prediction_steps = model.generate_with_probing(
            prompt=prompt,
            max_new_tokens=max_output_new_tokens,
            temperature=0.3,
            top_p=0.95,
            topk=extract_middle_token_num,
            threshold=3,
            print_details = print_details,
            collect_attn_mech = collect_attn_mech,
            collect_intermediate_res = collect_intermediate_res, 
            collect_mlp = collect_mlp,
            collect_block = collect_block

        )
        
        print(f"Output: {[prediction_steps[-1]['input_text']]+[prediction_steps[-1]['predicted_token']]}\n")
        print("Inference Complete")
        print("Generating JSON file and Heatmaps")


        token_major_steps = ActivationAnalyzer.convert_to_token_major(prediction_steps)
        if save_output:
            trial_path = f"{output_base_path}/{model_type.value}/trial_{i}"
            
            #Save visualizations
            for step in token_major_steps:
                analyzer.visualize_per_token_heatmaps(
                    prediction_step=step,
                    output_dir=f"{trial_path}/visualizations/per_token",
                    step_idx=step["step_idx"],
                    component="block_output",
                    max_tokens=max_output_new_tokens,
                    log_scale=False,
                )


            # Save prediction step data
            analyzer.save_prediction_steps(
                prediction_steps=token_major_steps,
                output_dir=trial_path,
                save_all_data=True
            )

def main():
    # Simple test example
    token = "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"
    # Test 1: Basic logit lens functionality
    test_prompt = "Generate a long History of how independence story of india."
    print("\nRunning basic logit lens test...")
    run_analysis(
        model_type= ModelType.LLAMA_3_1_70B,
        token=token,
        prompt=test_prompt,
        extract_middle_token_num = 3,
        max_output_new_tokens=10,
        num_trials=1,
        print_details=False,
        save_output=True,
        collect_attn_mech=False,
        collect_intermediate_res=False, 
        collect_mlp=False,
        collect_block=True
    )
    torch.cuda.empty_cache()
    gc.collect()
 

# from LogitLens4LLMs.llm_steer.steer_vec_llama_3_1_8B import Steer

# # Initialize Steer with LLaMA 3.1 8B
# steer = Steer(device="cuda")

# # Add a steering vector to layer 20
# steer.add(
#     layer_idx=20,
#     coeff=0.5,
#     text="This is a positive response.",
#     try_keep_nr=1,
#     exclude_bos_token=False,
# )

# # Get all steering vectors
# print(steer.get_all())

# # Reset steering for layer 20
# steer.reset(20) 

if __name__ == "__main__":
    main()

