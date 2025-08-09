import yaml, os
from pathlib import Path

def load_config(path: str):
    with open(Path(path).expanduser(), "r") as f:
        cfg = yaml.safe_load(f) or {}
    # Expand ~ and $VARS in cache_dir/local_path
    for k in ("cache_dir", "local_path"):
        if cfg.get(k):
            cfg[k] = os.path.expandvars(os.path.expanduser(cfg[k]))
    return cfg
