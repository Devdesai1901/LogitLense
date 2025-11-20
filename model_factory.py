from enum import Enum
from typing import Union, Dict, Type, Any, List
from model_helper.llama_3_1_helper import Llama3_1_8BHelper
from model_helper.llama_3_1_70B_helper import Llama3_1_70BHelper
from model_helper.qwen_3_32B_helper import Qwen_3_32B_Helper 
from model_helper.qwen_3_4B_helper import Qwen_3_4B_Helper
from model_helper.qwen_3_14B_helper import Qwen_3_14B_Helper

class ModelType(Enum):
    LLAMA_3_1_8B = "llama_3_1_8b"
    LLAMA_3_1_70B = "llama_3_1_70B"  # keep your existing value if other code depends on it
    QWEN_3_32B = "qwen_3_32B"
    QWEN_3_4B = "qwen_3_4B"
    QWEN_3_14B = "qwen_3_14B"

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
        ModelType.QWEN_3_32B: Qwen_3_32B_Helper,
        ModelType.QWEN_3_4B: Qwen_3_4B_Helper,
        ModelType.QWEN_3_14B: Qwen_3_14B_Helper
    }

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type) -> None:
        cls._model_registry[model_type] = model_class

    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        token: str = None,
        collect_attn_mech: bool = False,
        collect_mlp: bool = False,
        collect_block: bool = True,
        cfg: dict | None = None,
        selected_layers: List[int] = [10,15,25,35,79],
        **kwargs: Any
    ) -> Union[Llama3_1_8BHelper, Llama3_1_70BHelper, Qwen_3_32B_Helper, Qwen_3_4B_Helper ]:
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_class = cls._model_registry[model_type]

        if model_type == ModelType.LLAMA_3_1_8B :
            # Call your existing 8B helper unchanged
            return model_class(
                token=token,
                collect_attn_mech=collect_attn_mech,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                selected_layers = selected_layers,
                **kwargs
            )
        else:
            if cfg is None:
                raise ValueError("Qwen3-32B helper requires cfg (parsed YAML).")
            return model_class(
                cfg=cfg,
                collect_attn_mech=collect_attn_mech,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                selected_layers=selected_layers,
                **kwargs
            )    

        # fallback (shouldn’t hit)
        raise ValueError(f"No constructor path for model type: {model_type}")
