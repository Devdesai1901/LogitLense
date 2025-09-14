


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
.
├── explanation/ # Stores activation analysis outputs (JSON, heatmaps), runtime duration for model runs
│
├── model_helper/
│   ├── llama_3_1_70B_helper.py              # Helper functions & wrappers for running LLaMA 3.1–70B for Logit Lense 
│   ├── llama_3_1_helper.py                  # Helper functions & wrappers for running LLaMA 3.1–8B for Logit Lense 
│   └── model_io.py                          # Model weight loading & checkpoint management (I/O abstraction)
│                          
│
├── steering_vector/
│   ├── evaluate_steerability_custom.py     # Script to test steerability & measure vector influence 
│   │                                       # (For Research purpose: calculate steerability score & propensity curve)
|   |
│   ├── llm_new_steer.py                     # module for steering vector injection for LLaMA 3.1–70B
│   ├── steer_vec_llama_3_1_8b.py            # module for steering vector injection for LLaMA 3.1–8B
│   
│
├── activation_analyzer.py                   # Helper  File to genereate Heatmaps and JSON files output
│
├── config.yaml                              # YAML configuration for layer selection, capture flags, output paths
├── main.py                                  # Entry point for running logit-lens analysis (parsing args + launching pipeline)
│
├── model_factory.py                          # Factory for building models (8B/70B) with optional wrappers & probing
├── requirements.txt                          # Python dependencies 


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
cd LogitLense
pip install -r requirements.txt
```

## 🔧 Prerequisite (Required): Configure `config.yaml` for **LLaMA-3.1-70B**

Configure a `config.yaml` at the repo root before running 70B. This file controls model source, auth, and runtime.


## 🏃 Running the Code


### **Logit Lens for LLaMA 3.1–70B **
```bash
deepspeed --num_gpus=4 main.py \
  --model llama_3_1_70B \
  --config config.yaml \
  --prompt 'Tell me something about world' \
 --num_trials 1 \
  --max_new_tokens 10 \
  --extract_topk 3 \
  --collect_block \
  --collect_mlp \
  --collect_attn_mech \
  --layers 0-79

```

### **Logit Lens for LLaMA 3.1–8B **
```bash
deepspeed --num_gpus=4 main.py \
  --model llama_3_1_8b \
  --prompt 'Tell me something about our beautiful world' \
  --num_trials 1 \
  --max_new_tokens 30 \
  --token Hugging_Face Token \
  --extract_topk 3 \
  --collect_block \
  --collect_mlp \
  --collect_attn_mech \
  --layers 0-31
```

### **Steering Vector**
```bash
deepspeed --num_gpus=4 llm_new_steer.py \
  --config config.yaml \
  --batch_size 2 \
  --layer_stride 8 \
  --max_new_tokens 60 \
  --temperature 0.7 \
  --top_p 0.95 \
  --save_dir ./explanation/steering
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
- **time.json** - End-to-end runtime summary **and** granular timings for each major operation



## ⚙️ Run-Time Arguments — Description

| Argument | Description |
|---|---|
| `--model` | **Model selector** (choices from `ModelType`; default: `LLAMA_3_1_70B`). Use values like `llama_3_1_70B` or `llama_3_1_8b`. |
| `--prompt` | **Input text** for generation/analysis (default: `"Burj Khalifa est la plus grande tour du monde"`). |
| `--num_trials` | **Repeat runs** for the same prompt to average/compare results (int, default: `1`). |
| `--max_new_tokens` | **Generation length** in new tokens (int, default: `15`). |
| `--extract_topk` | **Top-K tokens** to keep per decoding step/layer for Logit Lens (int, default: `15`). |
| `--local_rank` | **Internal rank** used by DeepSpeed/torch.distributed (int, default: `0`). Typically set automatically—don’t pass manually. |
| `--layers` | **Layers to analyze**; CSV/range syntax (string). Examples: `0-4`, `0,5,10-12`, `3,-1`, `5-`, `-10`. Negative indices count from the end. Open-ended ranges allowed. *If omitted, the pipeline defaults to the first 5 layers.* |
| `--config` | **Path to YAML config** for **70B** runs (e.g., `config.yaml`). Controls model source, auth, dtype, tensor parallel size, etc. |
| `--token` | **Hugging Face token** (string). **8B-only runs** / backward compatibility|
| `--collect_attn_mech` | **Flag to capture attention mechanism**  |
| `--collect_mlp` | **Flag to capture MLP outputs** |
| `--collect_block` | **Flag to capture block outputs** |
| `--output_base_path` | **Output directory**  |

**Layer range syntax quick examples**
- `0-4` → layers 0,1,2,3,4  
- `5-` → from layer 5 to last  
- `-10` → layers 0..10  
- `3,-1` → layer 3 and the last layer  
- `0,5,10-12` → layers 0,5,10,11,12



## ⚙️ Logit Lens Arguments (`main.py → run_analysis()`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| model_type | ModelType | Required | Model type (e.g., LLAMA_3_1_70B) |
| prompt | str | "" | Input text prompt |
| num_trials | int | 5 | Number of generation trials |
| extract_top | int | 3 | Number of middle tokens to analyze |
| max_new_tokens | int | 10 | Max tokens to generate |
| output_base_path | str | "./explanations/logit_lens" | Output folder |
| collect_attn_mech | bool | False | Collect attention mechanism outputs |
| collect_mlp | bool | False | Collect MLP outputs |
| collect_block | bool | True | Collect block outputs |
| selected_layers | List[int]| [10,15,25,35,79] |  select the layers to analyze

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
- [Medium Blog](https://bobrupakroy.medium.com/steering-large-language-models-with-activation-vectors-a-practical-guide-45866b3697ac)
- 

## Contact

For any questions or suggestions, please contact me at [ddesai4@stevens.edu](mailto:ddesai4@stevens.edu).


---

Happy Coding! 🚀
