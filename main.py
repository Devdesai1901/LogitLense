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
    collect_attn_mech: bool=False,
    collect_intermediate_res: bool=False, 
    collect_mlp: bool=False,
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
        print("This functionality to introduce in later versions")
        sys.exit()
        
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
        print("Post Inference Generation Started")
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
    token = "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"
    test_prompt = "Tell me the sotry of Avengers!"
    print("\nRunning logit lens test...")
    run_analysis(
        model_type= ModelType.LLAMA_3_1_70B,
        token=token,
        prompt=test_prompt,
        extract_middle_token_num = 3,
        max_output_new_tokens=50,
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
 


if __name__ == "__main__":
    main()

