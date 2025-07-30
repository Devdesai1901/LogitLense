from typing import Dict, List, Tuple, TypedDict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class ComponentData(TypedDict):
    attention_mechanism: List[Tuple[str, int]]
    intermediate_residual: List[Tuple[str, int]]
    mlp_output: List[Tuple[str, int]]
    block_output: List[Tuple[str, int]]

class LayerActivations(TypedDict):
    layer_idx: int
    components: ComponentData

class PredictionStep(TypedDict):
    """Data structure for a single prediction step"""
    step_idx: int
    input_text: str
    predicted_token: str
    all_layers_data: Dict[int, Dict[str, List[Tuple[str, int]]]]
    important_layers: Dict[int, Dict[str, List[Tuple[str, int]]]]

class ActivationAnalyzer:
    """A utility class for analyzing and filtering model activation values"""



    @staticmethod
    def convert_to_token_major(prediction_steps: List[PredictionStep]) -> List[Dict]:
        """
        Convert a list of layer-major prediction steps to token-major format.
        Returns a list of updated prediction steps with 'all_tokens_data'.
        """
        token_major_steps = []

        for step in prediction_steps:
            all_layers_data = step["all_layers_data"]

            # Estimate token count from any component in first layer
            num_tokens = max(
                len(layer_data[next(iter(layer_data))])
                for layer_data in all_layers_data.values()
            )

            token_data = {}
            for token_idx in range(num_tokens):
                per_layer = {}
                for layer_idx, comps in all_layers_data.items():
                    layer_comps = {}
                    for comp_name, tokens_list in comps.items():
                        if token_idx < len(tokens_list):
                            layer_comps[comp_name] = tokens_list[token_idx]
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
    def filter_important_layers(
        all_layers_data: Dict[int, Dict[str, List[Tuple[str, int]]]], 
        threshold: int = 3
    ) -> Dict[int, Dict[str, List[Tuple[str, int]]]]:
        """
        Filter out important layers containing high-probability tokens.
        A layer is considered important if any of its components (attention_mechanism,
        mlp_output, or block_output) contains tokens with probability above the threshold.
        
        Args:
            all_layers_data: Activation data for all layers
                Format: {
                    layer_idx: {
                        component_name: [(token, probability), ...]
                    }
                }
            threshold: Probability threshold (percentage)
            
        Returns:
            Dict[int, Dict[str, List[Tuple[str, int]]]]: Filtered data of important layers
        """
        important_layers = {}
        components_to_save = ['attention_mechanism', 'mlp_output', 'block_output']
        
        for layer_idx, layer_data in all_layers_data.items():
            is_important_layer = False
            for component_name in components_to_save:
                if component_name in layer_data:
                    tokens_probs = layer_data[component_name]
                    flattened = [pair for token_list in tokens_probs for pair in token_list]
                    max_prob = max(prob for _, prob in flattened)
                    if max_prob > threshold:
                        is_important_layer = True
                        break
            
           
            if is_important_layer:
                important_components = {}
                for component_name in components_to_save:
                    if component_name in layer_data:
                        important_components[component_name] = layer_data[component_name]
                important_layers[layer_idx] = important_components
                
        return important_layers
    
    @staticmethod
    def get_top_predictions(
        layer_data: Dict[str, List[Tuple[str, int]]], 
        top_k: int = 1
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get top K predictions with highest probabilities for each component
        
        Args:
            layer_data: Activation data for a single layer
            top_k: Number of top predictions to return
            
        Returns:
            Dict[str, List[Tuple[str, int]]]: Top K predictions for each component
        """
        result = {}
        for component_name, tokens_probs in layer_data.items():
            sorted_preds = sorted(tokens_probs, key=lambda x: x[1], reverse=True)
            result[component_name] = sorted_preds[:top_k]
        return result 

    @staticmethod
    def save_prediction_steps(
        prediction_steps: List[Dict],
        output_dir: str,
        save_all_data: bool = True
    ) -> None:
        """
        Save token-major prediction steps to local files.
        Assumes each step has 'all_tokens_data'.
        """
        import json
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        # Skip saving 'important_layers' if not available
        important_path = os.path.join(output_dir, f"important_layers_{timestamp}.json")
        important_data = {}
        for step in prediction_steps:
            if "important_layers" in step:
                important_data[f"step_{step['step_idx']}"] = {
                    "input_text": step["input_text"],
                    "predicted_token": step["predicted_token"],
                    "important_layers": step["important_layers"]
                }

        if important_data:
            with open(important_path, 'w', encoding='utf-8') as f:
                json.dump(important_data, f, indent=2, ensure_ascii=False)

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
    def visualize_layer_predictions(
        prediction_step: PredictionStep,
        output_dir: str,
        step_idx: int,
        model_name: str = "Llama-3-7b",
        threshold: int = 3,
        max_tokens: int = 10,
        fig_width: int = 15,
        fig_height: int = None,
        log_scale: bool = False
    ) -> None:
        """
        Generate heatmaps for both important layers and all layers.
        
        Args:
            prediction_step: Data for a single prediction step
            output_dir: Output directory for saving visualizations
            step_idx: Step index
            model_name: Name of the model
            threshold: Threshold for important layers
            max_tokens: Maximum number of tokens to display per row
            fig_width: Width of the figure
            fig_height: Height of the figure (auto-calculated if None)
            log_scale: Whether to display probabilities in log scale
        """
        return


        def create_plot_data(layers_data):
                """Create data for a single plot"""
                plot_data = []
                token_data = []
                row_labels = []
                layer_boundaries = []
                current_row = 0
                
                for layer_idx in sorted(layers_data.keys()):
                    layer_data = layers_data[layer_idx]
                    layer_start = current_row
                    
                    for component in components_to_plot:
                        if component in layer_data:
                            tokens_probs = layer_data[component][:max_tokens]
                            tokens_probs.extend([("", 0)] * (max_tokens - len(tokens_probs)))
                            
                            probs = [prob for _, prob in tokens_probs]
                            if log_scale and any(probs):
                                probs = [np.log(p + 1e-10) if p > 0 else -np.inf for p in probs]
                            
                            tokens = []
                            for token, _ in tokens_probs:
                                if not token:
                                    tokens.append("")
                                else:
                                    printable_token = token.encode('unicode_escape').decode()
                                    printable_token = printable_token.replace('$', r'\$')  # 转义 $ 符号
                                    if len(printable_token) > 10:
                                        printable_token = printable_token[:10] + "..."
                                    tokens.append(printable_token)
                            
                            plot_data.append(probs)
                            token_data.append(tokens)
                            row_labels.append(f"Layer {layer_idx}\n{component}")
                            current_row += 1
                    
                    if current_row > layer_start:
                        layer_boundaries.append(current_row - 0.5)
                
                return plot_data, token_data, row_labels, layer_boundaries
    

    @staticmethod
    def visualize_per_token_heatmaps(
        prediction_step: Dict,
        output_dir: str,
        step_idx: int,
        component: str = "block_output",
        max_layers: Optional[int] = None,
        max_tokens: int = 10,
        log_scale: bool = False
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os

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

            max_k = 0  # how many top-k columns to draw

            for layer_idx in layer_indices:
                row_values = []
                row_annots = []

                layer_str = str(layer_idx)
                if component in layer_data[layer_str]:
                    topk_preds = layer_data[layer_str][component]
                    max_k = max(max_k, len(topk_preds))
                    for tok, prob in topk_preds:
                        val = np.log(prob + 1e-10) if log_scale else prob
                        row_values.append(val)
                        t = tok.encode('unicode_escape').decode()
                        if len(t) > 12:
                            t = t[:10] + "…"
                        row_annots.append(f"{t}\n({prob:.1f})")
                else:
                    row_values.append(0)
                    row_annots.append("")

                values.append(row_values)
                annotations.append(row_annots)

            # Pad rows to max_k width
            for i in range(len(values)):
                pad_len = max_k - len(values[i])
                values[i].extend([0] * pad_len)
                annotations[i].extend([""] * pad_len)

            plt.figure(figsize=(max_k * 1.8, len(values) * 0.6 + 2))

            sns.heatmap(
                np.array(values),
                annot=np.array(annotations),
                fmt="",
                cmap="YlOrRd",
                xticklabels=[f"Top-{i+1}" for i in range(max_k)],
                yticklabels=[f"Layer {i}" for i in layer_indices],
                cbar_kws={'label': 'Log Probability' if log_scale else 'Probability (%)'},
                annot_kws={'fontsize': 8}
            )

            plt.title(f"Token {token_idx} ({prediction_step['predicted_token']})")
            plt.xlabel("Top-k Prediction Rank")
            plt.ylabel("Layer")
            save_path = os.path.join(output_dir, f"token_{token_idx}_{component}_step_{step_idx}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()



        
        

    def create_heatmap(plot_data, token_data, row_labels, layer_boundaries, title_prefix, save_path):
            """Create and save heatmap visualization"""
            if not plot_data:
                print(f"No data to visualize for {title_prefix}")
                return
            
            plot_data = np.array(plot_data)
            
            
            actual_fig_height = fig_height if fig_height is not None else max(50, len(plot_data) * 0.5)
            
            plt.figure(figsize=(fig_width, actual_fig_height))
            
            
            ax = sns.heatmap(plot_data, 
                             cmap='YlOrRd',
                             annot=np.array(token_data),
                             fmt='',
                             cbar_kws={'label': 'Log Probability' if log_scale else 'Probability (%)'},
                             yticklabels=row_labels,
                             annot_kws={'size': 8})
            
            
            for boundary in layer_boundaries[:-1]:
                ax.axhline(y=boundary, color='black', linewidth=1)
            
            
            title = f"{title_prefix}\nModel: {model_name}\n"
            if "Important" in title_prefix:
                title += f"Layer Threshold: {threshold}%\n"
            if log_scale:
                title += "Probabilities shown in log scale"
            plt.title(title)
            plt.xlabel("Token Rank (by probability)")
            
            
            input_text = prediction_step['input_text'].encode('unicode_escape').decode()
            predicted_token = prediction_step['predicted_token'].encode('unicode_escape').decode()
            plt.figtext(0.05, -0.1, 
                        f"Input prompt: {input_text}\n"
                        f"Predicted token: {predicted_token}", 
                        wrap=True, 
                        horizontalalignment='left',
                        fontsize=10)
            
            plt.tight_layout()
            
           
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Main execution logic
            plt.rcParams['font.family'] = 'DejaVu Sans'
            components_to_plot = ['attention_mechanism', 'mlp_output', 'block_output']
            
            # Create heatmap for Important Layers
            important_data = create_plot_data(prediction_step['important_layers'])
            create_heatmap(
                *important_data,
                title_prefix="Important Layers",
                save_path=os.path.join(output_dir, f"important_layers_step_{step_idx}.png")
            )
            
            # Create heatmap for All Layers
            all_data = create_plot_data(prediction_step['all_layers_data'])
            create_heatmap(
                *all_data,
                title_prefix="All Layers",
                save_path=os.path.join(output_dir, f"all_layers_step_{step_idx}.png")
            )     