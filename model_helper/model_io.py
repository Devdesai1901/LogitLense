import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

def _resolve_token(cfg) -> Optional[str]:
    if cfg.get("hf_token"):
        return cfg["hf_token"]
    env_key = cfg.get("hf_token_env")
    return os.getenv(env_key) if env_key else None

def _resolve_dtype(s: str):
    s = (s or "bfloat16").lower()
    return {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16,  "float16": torch.float16,
        "fp32": torch.float32,  "float32": torch.float32
    }[s]

def load_tokenizer_and_model(cfg) -> Tuple[AutoTokenizer, torch.nn.Module, str]:
    """
    Returns (tokenizer, model, source_used), where source_used is 'local' or 'hub'.
    Loads from cfg['local_path'] if provided and exists; else from HF Hub.
    Falls back to AutoModel if AutoModelForCausalLM doesn't recognize the config
    (e.g., Qwen3-VL-MoE configs).
    """
    model_id   = cfg["model_id"]
    local_path = (cfg.get("local_path") or "").strip()
    cache_dir  = cfg.get("cache_dir")
    trust_rc   = bool(cfg.get("trust_remote_code", True))
    token      = _resolve_token(cfg)
    dtype      = _resolve_dtype(cfg.get("dtype", "bfloat16"))

    # prefer explicit local_path if it exists
    source_used = "hub"
    if local_path and Path(local_path).exists():
        model_id = local_path
        source_used = "local"

    kw = dict(trust_remote_code=trust_rc, token=token)
    if cache_dir:
        kw["cache_dir"] = cache_dir

    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    # optional padding side from YAML
    pad_left = cfg.get("pad_left", None)
    if pad_left is True:
        tok.padding_side = "left"
    elif pad_left is False:
        tok.padding_side = "right"

    # model (prefer CausalLM; fall back to AutoModel for configs not registered yet)
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, **kw
        )
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            mdl = AutoModel.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True, **kw
            )
        else:
            raise

    # sensible tokenizer defaults
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok, mdl, source_used
