import sys, os, gc, argparse, torch
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_factory import ModelFactory, ModelType
from activation_analyzer import ActivationAnalyzer70B
from activation_analyzer_8B import ActivationAnalyzer8B
from model_helper.config import load_config

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[m.value for m in ModelType], default=ModelType.LLAMA_3_1_70B.value)
    ap.add_argument("--prompt", default="Burj Khalifa est la plus grande tour du monde")
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=15)
    ap.add_argument("--extract_topk", type=int, default=15)
    ap.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)

    # 70B config (YAML)
    ap.add_argument("--config", help="Path to YAML config for 70B")

    # 8B-only args (kept for backward compatibility)
    ap.add_argument("--token", default=None)

    # capture flags (shared)
    ap.add_argument("--collect_attn_mech", action="store_true")
    ap.add_argument("--collect_mlp", action="store_true")
    ap.add_argument("--collect_block", action="store_true")

    ap.add_argument("--output_base_path", default="./explanation/logit_lens")
    return ap.parse_args()

def run_analysis(
    model,
    analyzer,
    model_type: ModelType,
    prompt: str,
    num_trials: int,
    max_new_tokens: int,
    extract_topk: int,
    output_base_path: str,
    collect_attn_mech: bool,
    collect_mlp: bool,
    collect_block: bool
):
    print(f"\nRunning logit lens analysis:")
    print(f"Model type: {model_type.value}")
    print(f"Prompt: {prompt}\n")

    for i in range(num_trials):
        print(f"Running trial {i+1}...")
        prediction_steps = model.generate_with_probing(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.95,
            topk=extract_topk,
            threshold=3,
            collect_attn_mech=collect_attn_mech,
            collect_mlp=collect_mlp,
            collect_block=collect_block,
        )

        print(f"Output: {[prediction_steps[-1]['input_text'], prediction_steps[-1]['predicted_token']]}")
        print("Post Inference Generation Started")

        trial_path = f"{output_base_path}/{model_type.value}/trial_{i}"

        if model_type == ModelType.LLAMA_3_1_70B:
            token_major_steps = analyzer.convert_to_token_major(prediction_steps)
            for step in token_major_steps:
                analyzer.visualize_per_token_combined_heatmap(
                    prediction_step=step,
                    output_dir=f"{trial_path}/visualizations/per_token",
                    step_idx=step["step_idx"],
                    components=["attention_mechanism", "mlp_output", "block_output"],
                    max_tokens=max_new_tokens,
                    log_scale=False,
                )
            analyzer.save_prediction_steps(token_major_steps, trial_path, save_all_data=True)
        else:
            for step in prediction_steps:
                analyzer.visualize_layer_predictions(
                    prediction_step=step,
                    output_dir=f"{trial_path}/visualizations",
                    step_idx=step["step_idx"],
                    model_name=model_type.value,
                    threshold=3,
                    max_tokens=max_new_tokens,
                    log_scale=False,
                )
            analyzer.save_prediction_steps(prediction_steps, trial_path, save_all_data=True)

def main():
    args = parse_args()
    mt = ModelType.from_string(args.model)

    # Construct model per-arch
    if mt == ModelType.LLAMA_3_1_70B:
        if not args.config:
            raise SystemExit("Error: --config is required for 70B (YAML).")
        cfg = load_config(args.config)
        model = ModelFactory.create_model(
            model_type=mt,
            cfg=cfg,
            collect_attn_mech=args.collect_attn_mech,
            collect_mlp=args.collect_mlp,
            collect_block=args.collect_block,
        )
        analyzer = ActivationAnalyzer70B()
    else:
        # 8B unchanged signature
        model = ModelFactory.create_model(
            model_type=mt,
            token=args.token,
            collect_attn_mech=args.collect_attn_mech,
            collect_mlp=args.collect_mlp,
            collect_block=args.collect_block,
        )
        analyzer = ActivationAnalyzer8B()

    run_analysis(
        model=model,
        analyzer=analyzer,
        model_type=mt,
        prompt=args.prompt,
        num_trials=args.num_trials,
        max_new_tokens=args.max_new_tokens,
        extract_topk=args.extract_topk,
        output_base_path=args.output_base_path,
        collect_attn_mech=args.collect_attn_mech,
        collect_mlp=args.collect_mlp,
        collect_block=args.collect_block,
    )

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
