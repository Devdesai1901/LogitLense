


# LLM-Insight 

**LLM‑Insight** is an interpretability toolkit that reveals what’s happening inside large language models. Using **Logit Lens** and **Steering Vectors**, it performs **layer‑wise analysis** of hidden states, visualizes how token predictions evolve, and outputs insights through heatmaps and JSON files.

It also supports **hidden state steering**, letting you intentionally nudge model behavior to test ideas, reduce biases, or guide responses. This makes LLM‑Insight both a **diagnostic** and an **active** tool for improving the **explainability and transparency** of modern LLMs.



Currently supports:
- **Meta-LLaMA-3.1-70B** ✅ 
- **Meta-LLaMA-3.1-8B** ✅
- Designed to extend easily to other models.

---

## 🚀 Features

- **Layer-wise Analysis (Logit Lens)** – Analyze model hidden states & predictions layer-by-layer.
- **Steering Vectors** – Inject targeted changes into hidden states to influence outputs.
- **Multi-Model Support** – Tested with Meta-LLaMA-3.1-70B and 8B.
- **Visualization Outputs** – Generates heatmaps for each token at each layer.
- **Local Model Support** – Load models locally to reduce network dependency.
- **Distributed Inference with Model Parallelism** – Achieved via **DeepSpeed** + **PyTorch** + **Transformers**.
- Optimized for **4× NVIDIA RTX A6000 (48GB)** GPUs with **250GB CPU RAM**.

---

## 📂 Project Structure

```
LogitLens4LLMs/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry for Logit Lens analysis
├── llm_new_steer.py             # Main entry for Steering Vector experiments
├── activation_analyzer.py      # Utilities for processing & visualizing activations
├── model_factory.py            # Model loading factory
├── model_helper/               # Model-specific helper classes
│   ├── llama_3_1_70B_helper.py
│   ├── llama_2_helper.py
│   ├── qwen_helper.py
├── output/                     # Output folder for caches & results
│   ├── cache/                  # Model weights (downloaded on first run)
│   ├── explanations/           # Heatmaps + JSON outputs for Logit Lens
├── offload/                    # Disk offloading directory for large models
├── visualizations/             # Generated plots
├── myvenv/                     # (Optional) Virtual environment
```

---

## 🛠 Environment

Tested with:
- **Python**: 3.12.3
- **DeepSpeed**: 0.17.1
- **PyTorch**: 2.7.1
- **Transformers**: 4.52.4  
- GPUs: **4× NVIDIA RTX A6000 (48GB each)**
- CPU Memory: **250GB**

---
## 📦 Installation
```bash
git clone https://github.com/Devdesai1901/LogitLense.git
cd LogitLens4LLMs
pip install -r requirements.txt
```
## 🏃 Running the Code
> **Note:** If running inside a virtual environment, first activate it, then navigate **out of all folders** to your **home directory** before running the command.  
> Ensure you are **not** inside the `LogitLens4LLMs` folder — otherwise the module import will fail.


### **Logit Lens**
```bash
deepspeed --num_gpus=4 --module LogitLens4LLMs.main
```

### **Steering Vector**
```bash
deepspeed --num_gpus=4 --module LogitLens4LLMs.llm_new_steer
```

---

## ⚠️ First-Time Run Notes
- On first run, model checkpoints will be downloaded from HuggingFace into:
```bash
./output/cache
```
- This may take several minutes depending on network speed.
- Then model is loaded in **Tensor Parallelism** across GPUs.
- Second run onwards – startup is much faster due to local caching.

---

## 📊 Logit Lens Output
When running `main.py`, a new folder:
```bash
/explanations/
```
will be created containing:
- **heatmaps/** – PNG images showing per-token predictions at each layer.
- **predictions.json** – Detailed per-token, per-layer activations & predictions.

---

## ⚙️ Logit Lens Arguments (`main.py → run_analysis()`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| model_type | ModelType | Required | Model type (e.g., LLAMA_3_1_70B) |
| use_local | bool | False | Load from local cache instead of downloading |
| token | str | None | HuggingFace auth token |
| prompt | str | "" | Input text prompt |
| num_trials | int | 5 | Number of generation trials |
| extract_middle_token_num | int | 15 | Number of middle tokens to analyze |
| print_details | bool | False | Print verbose prediction details |
| max_output_new_tokens | int | 10 | Max tokens to generate |
| save_output | bool | True | Save heatmaps & JSON |
| output_base_path | str | "./explanations/logit_lens" | Output folder |
| collect_attn_mech | bool | False | Collect attention mechanism outputs |
| collect_intermediate_res | bool | False | Collect intermediate residuals |
| collect_mlp | bool | False | Collect MLP outputs |
| collect_block | bool | True | Collect block outputs |

💡 **Performance Tip**: For best speed, collect only one component (usually `block_output`). Collecting multiple granular details increases memory use and slows inference.

---

## 🎯 Steering Vector Parameters

### Prompt Arrays (inside `llm_new_steer.py`)

```python
enthusiastic_prompts = [ ... ]       # Target style prompts
unenthusiastic_prompts = [ ... ]     # Base style prompts
test_prompts = [ ... ]               # Prompts to test steering effect
```

## Parameters for `do_steering()`

| Parameter    | Default | Description |
|--------------|---------|-------------|
| **scale**    | `1.0`   | Multiplies the steering vector before applying it to hidden states. |
| **normalise**| `True`  | Normalizes the steering vector to unit length before scaling. |
| **layer**    | `None`  | Layer index to apply steering; `None` applies to all layers. |
| **proj**     | `True`  | Projects the steering vector onto the hidden state direction before applying. |
| **batch_size** | `1`   | Number of sequences processed per batch during inference. |

---

## Parameters for `find_steering_vecs()`

| Parameter      | Default | Description |
|----------------|---------|-------------|
| **base_toks**  | *Required* | Tokenized baseline prompts. |
| **target_toks**| *Required* | Tokenized target prompts. |
| **base_mask**  | *Required* | Attention mask for `base_toks`. |
| **target_mask**| *Required* | Attention mask for `target_toks`. |
| **batch_size** | `2`       | Prompt pairs processed per forward pass. |


### Main Functions
```python
find_steering_vecs(model, base_toks, target_toks, base_mask, target_mask, batch_size=2)
```
- Computes per-layer steering vectors from hidden state differences.

```python
do_steering(model, test_toks, test_mask, steering_vec=None, scale=1.0, normalise=True, layer=None, proj=True, batch_size=1)
```
- Injects steering vectors into hidden states before output.

---

## 📌 Example: Running Steering Vector Experiment
```bash
deepspeed --num_gpus=4 --module LogitLens4LLMs.llm_new_steer
```

**Expected Output**:
```
--- LAYER 33 INTERVENTION ---
Prompt: Tell me something exciting about today.
BASELINE: Tell me something exciting about today. Did you get a new book? Did you go to a new place? Did you make a new friend?
Tell me about the most beautiful thing you saw today.
Tell me about something funny that happened today.
Tell me about something you learned today.
Tell me about something that made you sad today.

STEERED : Tell me something exciting about today. What was the best thing that happened?
I'm going to go out on a limb and guess that it wasn't a meeting.
The only way to make a meeting interesting is to have it at a bar.
Which is exactly what I did this week.
I invited people to join me for a drink

Prompt: Is this product launch important?
BASELINE: Is this product launch important? Are the features worth talking about? Who cares?
If your product launch feels like it’s not getting the attention it deserves, you’re not alone. Many companies struggle with this, especially in today’s world of information overload.
But don’t worry, I’ve got you covered. In this article,
STEERED : Is this product launch important? You bet it is. As you may recall, I've been on a personal quest to bring back the original recipe for Coca-Cola with cane sugar. I've even developed a Facebook page devoted to the cause. I'm not alone. There is a groundswell of support for this effort. And

Prompt: Do you enjoy feature releases?
BASELINE: Do you enjoy feature releases? I do. And this one is a little more special than usual. This is the first major release of our new team. We are proud of the result and hope you will be too.
This is the first major release of our new team. We are proud of the result and hope you will be
STEERED : Do you enjoy feature releases? If you do, you’re in luck. A new release has just dropped. If you don’t, don’t worry, we’ve got a new release for you as well. It’s a brand-new feature release. We’re calling it v0.1.0. It’s got some cool


```

---
## 📌 Example: Running Logit Lens Analysis
```bash
deepspeed --num_gpus=4 --module LogitLens4LLMs.main

Prompt : Tell me the sotry of Avengers!
OutPut :  " (1 Viewer)\nThread starter TheBee\nTheBee\nI'm a new Marvel fan and I want to know more about them. Can you tell me their story?\nReactions: JtheDreamer, TheBee and Kage"
```
> **Note:**  
> The heatmap visualization below shows the top-k token probabilities across all layers for a single generated token.  
> This helps in understanding how the model’s predictions evolve layer-by-layer.

![Logit Lens Heatmap Example](assets/token_18_block_output_step_0.png)

## Acknowledgments

Thanks to the following projects for inspiration and support:
-  [zhenyu-02](https://github.com/zhenyu-02/LogitLens4LLMs)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- [Logit Lens for Llama-2](https://www.lesswrong.com/posts/fJE6tscjGRPnK8C2C/decoding-intermediate-activations-in-llama-2-7b)

## Contact

For any questions or suggestions, please contact me at [ddesai4@stevens.edu](mailto:ddesai4@stevens.edu).


---

Happy Coding! 🚀
