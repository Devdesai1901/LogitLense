import os
from pathlib import Path
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _resolve_token(cfg) -> Optional[str]:
    if cfg.get("hf_token"):
        return cfg["hf_token"]
    env_key = cfg.get("hf_token_env")
    return os.getenv(env_key) if env_key else None

def _resolve_dtype(s: str):
    s = (s or "bfloat16").lower()
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[s]

def load_tokenizer_and_model(cfg) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    """
    Returns (tokenizer, model, source_used), where source_used is 'local' or 'hub'.
    Will download from HF Hub into cfg['cache_dir'] if local_path is missing.
    """
    model_id   = cfg["model_id"]
    local_path = (cfg.get("local_path") or "").strip()
    cache_dir  = cfg.get("cache_dir")
    trust_rc   = bool(cfg.get("trust_remote_code", True))
    token      = _resolve_token(cfg)
    dtype      = _resolve_dtype(cfg.get("dtype", "bfloat16"))

    # Prefer explicit local_path if it exists
    if local_path and Path(local_path).exists():
        src = local_path
        source_used = "local"
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=trust_rc, token=token)
        mdl = AutoModelForCausalLM.from_pretrained(
            src, torch_dtype=dtype, trust_remote_code=trust_rc, token=token, low_cpu_mem_usage=True
        )
    else:
        # Go to Hub; transformers will reuse cache if present or download otherwise
        source_used = "hub"
        kw = dict(trust_remote_code=trust_rc, token=token)
        if cache_dir:
            kw["cache_dir"] = cache_dir
        tok = AutoTokenizer.from_pretrained(model_id, **kw)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True, **kw)

    # sensible tokenizer defaults
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, mdl, source_used
