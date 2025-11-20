from typing import Dict, List, Tuple, TypedDict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib


def _pretty_token(tok: str) -> str:
    # normalize common subword markers
    if tok.startswith("▁") or tok.startswith("Ġ"):
        tok = " " + tok[1:]
    return tok.replace("\n", "\\n").replace("\t", "\\t")

class ComponentData(TypedDict):
    attention_mechanism: List[Tuple[str, int]]
    mlp_output: List[Tuple[str, int]]
    block_output: List[Tuple[str, int]]

class LayerActivations(TypedDict):
    layer_idx: int
    components: ComponentData

class PredictionStep(TypedDict):
    step_idx: int
    input_text: str
    predicted_token: str
    all_layers_data: Dict[int, Dict[str, List[Tuple[str, int]]]]
    important_layers: Dict[int, Dict[str, List[Tuple[str, int]]]]

class ActivationAnalyzer70B:
    @staticmethod
    def convert_to_token_major(prediction_steps: List[PredictionStep]) -> List[Dict]:
        token_major_steps = []
        for step in prediction_steps:
            all_layers_data = step["all_layers_data"]

            # find true max tokens across all components/layers
            max_tokens = 0
            for comps in all_layers_data.values():
                for seq in comps.values():
                    if isinstance(seq, list):
                        max_tokens = max(max_tokens, len(seq))

            token_data = {}
            for tok_idx in range(max_tokens):
                per_layer = {}
                for layer_idx, comps in all_layers_data.items():
                    layer_comps = {}
                    for comp_name, seq in comps.items():
                        if seq and tok_idx < len(seq):         # <- only keep real data
                            layer_comps[comp_name] = seq[tok_idx]
                    if layer_comps:                             # <- skip empty layers
                        per_layer[str(layer_idx)] = layer_comps
                if per_layer:                                   # <- skip empty tokens
                    token_data[str(tok_idx)] = per_layer

            token_major_steps.append({
                "step_idx": step["step_idx"],
                "input_text": step["input_text"],
                "predicted_token": step["predicted_token"],
                "all_tokens_data": token_data
            })
        return token_major_steps

    @staticmethod
    def save_prediction_steps(
        prediction_steps: List[Dict],
        output_dir: str,
        save_all_data: bool = True
    ) -> None:
        import json
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        if save_all_data:
            all_data_path = os.path.join(output_dir, f"tokenwise_steps_{timestamp}.json")
            all_data = {
                f"step_{step['step_idx']}": {
                    "input_text": step["input_text"],
                    "predicted_token": step["predicted_token"],
                    "all_tokens_data": step["all_tokens_data"]
                }
                for step in prediction_steps
            }
            with open(all_data_path, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def visualize_per_token_combined_heatmap(
        prediction_step: Dict,
        output_dir: str,
        step_idx: int,
        components: List[str] = ["attention_mechanism", "mlp_output", "block_output"],
        max_layers: Optional[int] = None,
        max_tokens: int = 10,
        log_scale: bool = False,
        drop_empty: bool = True,   # <- new
    ):
        os.makedirs(output_dir, exist_ok=True)
        token_data = prediction_step["all_tokens_data"]

        for token_idx_str in list(token_data.keys())[:max_tokens]:
            token_idx = int(token_idx_str)
            layer_data = token_data[token_idx_str]
            layer_indices = sorted(int(k) for k in layer_data.keys())
            if max_layers:
                layer_indices = layer_indices[:max_layers]

            rows, annots, row_labels = [], [], []
            max_k = 0

            # build only rows that actually have data
            for layer_idx in layer_indices:
                ld = layer_data[str(layer_idx)]
                for comp in components:
                    topk = ld.get(comp)
                    if not topk:
                        if drop_empty:
                            continue            # <- skip empty component row entirely
                        else:
                            topk = []

                    if topk:
                        max_k = max(max_k, len(topk))
                        r_vals, r_anns = [], []
                        for tok, prob in topk:
                            val = np.log(prob + 1e-10) if log_scale else prob
                            r_vals.append(val)
                            t = _pretty_token(tok)
                            if len(t) > 12: t = t[:10] + "…"
                            r_anns.append(f"{t}\n({prob:.1f})")
                        rows.append(r_vals)
                        annots.append(r_anns)
                        row_labels.append(f"Layer {layer_idx} - {comp}")

            if not rows or max_k == 0:
                print(f"Skipping heatmap for token {token_idx}: No data available")
                continue

            # pad to rectangular shape with NaNs (don’t affect color scale)
            for i in range(len(rows)):
                pad = max_k - len(rows[i])
                if pad > 0:
                    rows[i].extend([np.nan] * pad)
                    annots[i].extend([""] * pad)

            plt.figure(figsize=(max_k * 3, len(rows) * 0.9 + 2))
            data = np.array(rows, dtype=float)
            mask = np.isnan(data)

            # seaborn is fine here; NaNs will render as empty cells without extra rows
            sns.heatmap(
                data,
                mask=mask,
                annot=np.array(annots, dtype=object),
                fmt="",
                cmap="YlOrRd",
                xticklabels=[f"Top-{i+1}" for i in range(max_k)],
                yticklabels=row_labels,
                cbar_kws={'label': 'Log Probability' if log_scale else 'Probability (%)'},
                annot_kws={'fontsize': 8}
            )
            plt.title(f"Token {token_idx} ({prediction_step['predicted_token']}) - All Components")
            plt.xlabel("Top-k Prediction Rank")
            plt.ylabel("Layer - Component")
            save_path = os.path.join(output_dir, f"token_{token_idx}_combined_step_{step_idx}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()