from typing import Dict, List, Tuple, TypedDict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

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
            num_tokens = max(
                len(layer_data[next(iter(layer_data))])
                for layer_data in all_layers_data.values()
                if layer_data
            )
            token_data = {}
            components_to_include = ['attention_mechanism', 'mlp_output', 'block_output']
            for token_idx in range(num_tokens):
                per_layer = {}
                for layer_idx, comps in all_layers_data.items():
                    layer_comps = {}
                    for comp_name in components_to_include:
                        if comp_name in comps and token_idx < len(comps[comp_name]):
                            layer_comps[comp_name] = comps[comp_name][token_idx]
                        else:
                            layer_comps[comp_name] = []  # Empty list if no data
                    if layer_comps:
                        per_layer[str(layer_idx)] = layer_comps
                token_data[str(token_idx)] = per_layer

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
        log_scale: bool = False
    ):
        os.makedirs(output_dir, exist_ok=True)
        token_data = prediction_step["all_tokens_data"]
        for token_idx_str in list(token_data.keys())[:max_tokens]:
            token_idx = int(token_idx_str)
            layer_data = token_data[token_idx_str]
            layer_indices = sorted(int(k) for k in layer_data.keys())
            if max_layers:
                layer_indices = layer_indices[:max_layers]

            values = []
            annotations = []
            row_labels = []
            max_k = 0

            for layer_idx in layer_indices:
                layer_str = str(layer_idx)
                for comp in components:
                    if comp in layer_data[layer_str] and layer_data[layer_str][comp]:
                        topk_preds = layer_data[layer_str][comp]
                        max_k = max(max_k, len(topk_preds))
                        row_values = []
                        row_annots = []
                        for tok, prob in topk_preds:
                            val = np.log(prob + 1e-10) if log_scale else prob
                            row_values.append(val)
                            t = tok.encode('unicode_escape').decode()
                            if len(t) > 12:
                                t = t[:10] + "…"
                            row_annots.append(f"{t}\n({prob:.1f})")
                        values.append(row_values)
                        annotations.append(row_annots)
                        row_labels.append(f"Layer {layer_idx} - {comp}")
                    else:
                        values.append([0] * max_k)
                        annotations.append([""] * max_k)
                        row_labels.append(f"Layer {layer_idx} - {comp}")

            if not values or max_k == 0:
                print(f"Skipping heatmap for token {token_idx}: No data available")
                continue

            for i in range(len(values)):
                pad_len = max_k - len(values[i])
                values[i].extend([0] * pad_len)
                annotations[i].extend([""] * pad_len)

            plt.figure(figsize=(max_k * 3, len(values) * 0.9 + 2))
            sns.heatmap(
                np.array(values),
                annot=np.array(annotations),
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