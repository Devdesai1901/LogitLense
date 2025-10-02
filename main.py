import sys, os, gc, argparse, torch, json
from datetime import datetime
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_factory import ModelFactory, ModelType
from activation_analyzer import ActivationAnalyzer70B
# from activation_analyzer_8B import ActivationAnalyzer8B
from model_helper.config import load_config
from time import time

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[m.value for m in ModelType], default=ModelType.LLAMA_3_1_70B.value)
    ap.add_argument("--prompt", default="Burj Khalifa est la plus grande tour du monde")
    ap.add_argument("--num_trials", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=15)
    ap.add_argument("--extract_topk", type=int, default=15)
    ap.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--layers",type=str,default=None,metavar="LIST",help=(
        "Layers to analyze (CSV/ranges). Examples: '0-4', '0,5,10-12', '3,-1', '5-'. "
        "Negative indices count from the end (-1 = last). "
        "Open-ended ranges allowed (e.g., '5-' = 5..last, '-10' = 0..10). "
        "Default: first 5 layers."
    ),
)
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


# ---------------- helper: paste this in a utils section ----------------
from typing import Optional, List, Set

def parse_layer_spec(spec: Optional[str], total_layers: int, default_k: int = 5) -> List[int]:
    """
    Convert a spec like '0,5,10-12,-1,5-, -10' into sorted unique layer indices.
    - Negative indices are supported (e.g., -1 is last layer).
    - Open-ended ranges supported: '5-' means 5..last, '-10' means 0..10.
    - If spec is None/empty -> first `default_k` layers (bounded by total_layers).
    """
    if not spec or not str(spec).strip():
        return list(range(min(default_k, total_layers)))

    layers: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a = a.strip()
            b = b.strip()
            # interpret open-ended bounds
            if a == "":  # '-X' -> 0..X
                start = 0
            else:
                start = int(a)
                if start < 0:
                    start += total_layers
            if b == "":  # 'X-' -> X..last
                end = total_layers - 1
            else:
                end = int(b)
                if end < 0:
                    end += total_layers
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if 0 <= i < total_layers:
                    layers.add(i)
        else:
            i = int(part)
            if i < 0:
                i += total_layers
            if 0 <= i < total_layers:
                layers.add(i)

    if not layers:  # safety fallback
        return list(range(min(default_k, total_layers)))
    return sorted(layers)

def _iso_now():
    return datetime.now().isoformat(timespec="seconds")

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
    collect_block: bool,
    selected_layers: List[int] = [10,15,25,35,79]
):
    print(f"\nRunning logit lens analysis:")
    print(f"Model type: {model_type.value}")
    print(f"Prompt: {prompt}\n")

    trial_records = []
    analysis_method_name = "Running Logit Lens"  # label requested for generate_with_probing

    for i in range(num_trials):
        trial_idx = i
        trial_path = f"{output_base_path}/{model_type.value}/trial_{trial_idx}"
        os.makedirs(trial_path, exist_ok=True)

        print(f"Running trial {trial_idx+1}...")
        t_trial_start = time()
        t_trial_start_iso = _iso_now()

        # --- generate_with_probing ---
        t_gen_start = time()
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
            selected_layers = selected_layers
        )
        t_gen_end = time()
        gen_duration = t_gen_end - t_gen_start

        print(f"Output: {[prediction_steps[-1]['input_text'], prediction_steps[-1]['predicted_token']]}")
        print("Post Inference Generation Started")

        # Common labels the user asked for
        generation_name = "heat map"

        convert_duration = None
        heatmap_duration = 0.0
        json_dump_duration = 0.0

       
        # --- convert_to_token_major (timed) ---
        t_conv_start = time()
        token_major_steps = analyzer.convert_to_token_major(prediction_steps)
        t_conv_end = time()
        convert_duration = t_conv_end - t_conv_start

        # --- visualize heatmaps (timed; total) ---
        t_viz_start = time()
        for step in token_major_steps:
            analyzer.visualize_per_token_combined_heatmap(
                prediction_step=step,
                output_dir=f"{trial_path}/visualizations/per_token",
                step_idx=step["step_idx"],
                components=["attention_mechanism", "mlp_output", "block_output"],
                max_tokens=max_new_tokens,
                log_scale=False,
            )
        t_viz_end = time()
        heatmap_duration = t_viz_end - t_viz_start

        # --- save JSON dump (timed) ---
        t_dump_start = time()
        analyzer.save_prediction_steps(token_major_steps, trial_path, save_all_data=True)
        t_dump_end = time()
        json_dump_duration = t_dump_end - t_dump_start

        
        t_trial_end = time()
        t_trial_end_iso = _iso_now()
        trial_total = t_trial_end - t_trial_start

        trial_records.append({
            "trial_index": trial_idx,
            "start_time_iso": t_trial_start_iso,
            "end_time_iso": t_trial_end_iso,
            "duration_seconds": trial_total,
            "analysis_method": analysis_method_name,
            "generate_with_probing_seconds": gen_duration,
            "generation_name": generation_name,
            # present regardless of model; None for 8B
            "convert_to_token_major_seconds": convert_duration,
            "heat_map_seconds": heatmap_duration,
            "json_dump_seconds": json_dump_duration,
            "paths": {
                "trial_dir": trial_path,
                "visualizations_dir": (
                    f"{trial_path}/visualizations/per_token"
                    if model_type == ModelType.LLAMA_3_1_70B else
                    f"{trial_path}/visualizations"
                )
            }
        })

    return trial_records

def main():
    args = parse_args()
    mt = ModelType.from_string(args.model)
    if ModelType.LLAMA_3_1_70B:
        total_layers = 80
    else:
         total_layers = 32    
    selected_layers = parse_layer_spec(args.layers, total_layers, default_k=5)
    print(f"[Layer Select] Analyzing layers: {selected_layers}")


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
            selected_layers = selected_layers
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
            selected_layers = selected_layers
        )
        analyzer = ActivationAnalyzer70B()

    # --- total run timing ---
    total_start_ts = time()
    total_start_iso = datetime.now().isoformat(timespec="seconds")
     
    

 
    trial_records = run_analysis(
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
        selected_layers = selected_layers

    )

    total_end_ts = time()
    total_end_iso = datetime.now().isoformat(timespec="seconds")
    total_duration = total_end_ts - total_start_ts

    # --- write timings summary JSON ---
    summary = {
        "run": {
            "model": mt.value,
            "prompt": args.prompt,
            "num_trials": args.num_trials,
            "start_time_iso": total_start_iso,
            "end_time_iso": total_end_iso,
            "duration_seconds": total_duration
        },
        "trials": trial_records
    }

    out_dir = os.path.join(args.output_base_path, mt.value)
    os.makedirs(out_dir, exist_ok=True)
    timings_path = os.path.join(out_dir, "timings.json")
    with open(timings_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Timing] Wrote summary to: {timings_path}")

    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
