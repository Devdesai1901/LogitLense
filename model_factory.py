from enum import Enum
from typing import Union, Dict, Type, Any, List
from model_helper.llama_3_1_helper import Llama3_1_8BHelper
from model_helper.llama_3_1_70B_helper import Llama3_1_70BHelper
from model_helper.qwen3_vl_30b_a3b_helper import Qwen3_VL_30B_A3B_Helper 

class ModelType(Enum):
    LLAMA_3_1_8B = "llama_3_1_8b"
    LLAMA_3_1_70B = "llama_3_1_70B"  # keep your existing value if other code depends on it
    QWEN3_VL_30B_A3B = "qwen3_vl_30b_a3b"

    @classmethod
    def from_string(cls, model_name: str) -> 'ModelType':
        # tolerate case differences & minor formatting
        key = (model_name or "").strip().lower()
        for mt in cls:
            if mt.value.lower() == key:
                return mt
        raise ValueError(f"Unsupported model type: {model_name}. Supported: {[m.value for m in cls]}")

class ModelFactory:
    _model_registry: Dict[ModelType, Type] = {
        ModelType.LLAMA_3_1_8B: Llama3_1_8BHelper,
        ModelType.LLAMA_3_1_70B: Llama3_1_70BHelper,
        ModelType.QWEN3_VL_30B_A3B: Qwen3_VL_30B_A3B_Helper,
    }

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type) -> None:
        cls._model_registry[model_type] = model_class

    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        *,
        # ---- 8B signature args (kept for backward compat) ----
        token: str = None,
        # ---- shared capture flags ----
        collect_attn_mech: bool = False,
        collect_mlp: bool = False,
        collect_block: bool = True,
        # ---- 70B signature arg ----
        cfg: dict | None = None,
        selected_layers: List[int] = [10,15,25,35,79],
        **kwargs: Any
    ) -> Union[Llama3_1_8BHelper, Llama3_1_70BHelper, Qwen3_VL_30B_A3B_Helper ]:
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = cls._model_registry[model_type]

        if model_type == ModelType.LLAMA_3_1_8B:
            # Call your existing 8B helper unchanged
            return model_class(
                token=token,
                collect_attn_mech=collect_attn_mech,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                selected_layers = selected_layers,
                **kwargs
            )

        if model_type == ModelType.QWEN3_VL_30B_A3B:
            if cfg is None:
                raise ValueError("Qwen3-VL-30B-A3B helper requires cfg (parsed YAML).")
            return Qwen3_VL_30B_A3B_Helper(
                cfg=cfg,
                collect_attn_mech=collect_attn_mech,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                selected_layers=selected_layers,
                **kwargs
            )    

        if model_type == ModelType.LLAMA_3_1_70B:
            if cfg is None:
                raise ValueError("70B helper now requires cfg (parsed YAML).")
            # Call the new 70B helper with cfg (no token/local_path here)
            return model_class(
                cfg=cfg,
                collect_attn_mech=collect_attn_mech,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                selected_layers = selected_layers,
                **kwargs
            )

        # fallback (shouldn’t hit)
        raise ValueError(f"No constructor path for model type: {model_type}")
