from enum import Enum
from typing import Union, Dict, Type
from model_helper.llama_3_1_helper import Llama3_1_8BHelper
from model_helper.llama_3_1_70B_helper import Llama3_1_70BHelper

class ModelType(Enum):
    """Enumeration of supported model types"""
    LLAMA_3_1_8B = "llama_3_1_8b"
    LLAMA_3_1_70B = "llama_3_1_70B"
   
    
    @classmethod
    def from_string(cls, model_name: str) -> 'ModelType':
        """Create ModelType enum value from string"""
        try:
            return cls(model_name.lower())
        except ValueError:
            raise ValueError(f"Unsupported model type: {model_name}")

class ModelFactory:
    """Factory class for creating and managing different types of model instances"""
    
    _model_registry: Dict[ModelType, Type] = {
        ModelType.LLAMA_3_1_8B: Llama3_1_8BHelper,
        ModelType.LLAMA_3_1_70B: Llama3_1_70BHelper,
    }
    
    @classmethod
    def register_model(cls, model_type: ModelType, model_class: Type) -> None:
        """Register a new model type
        
        Args:
            model_type: Model type enum value
            model_class: Model class
        """
        cls._model_registry[model_type] = model_class
    
    @classmethod
    def create_model(
        cls,
        model_type: ModelType,
        use_local: bool = False,
        local_path: str = "./explanation/models_hf",
        token: str = None,
        collect_attn_mech: bool = False,collect_intermediate_res: bool = False, collect_mlp:  bool = False, collect_block: bool = True,
        **kwargs
    ) -> Union[ Llama3_1_8BHelper,Llama3_1_70BHelper ]:
        """Create a model instance
        
        Args:
            model_type: Type of model to create
            use_local: Whether to use locally cached model
            local_path: Local model path
            token: HuggingFace token (only needed when use_local=False)
            **kwargs: Additional parameters passed to model constructor
            
        Returns:
            Created model instance
            
        Raises:
            ValueError: If model type is not supported
        """
        if model_type not in cls._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model_class = cls._model_registry[model_type]


         # Conditional logic for LLAMA_3_1_8B
        if model_type == ModelType.LLAMA_3_1_8B or ModelType.LLAMA_3_1_70B:
            return model_class(
                use_local=use_local,
                local_path=local_path,
                token=token,
                collect_attn_mech=collect_attn_mech,
                collect_intermediate_res=collect_intermediate_res,
                collect_mlp=collect_mlp,
                collect_block=collect_block,
                **kwargs
            )
        
        return model_class(use_local=use_local, local_path=local_path, token=token, **kwargs)
        
    @classmethod
    def get_supported_models(cls) -> list[str]:
        """Get names of all supported model types"""
        return [model_type.value for model_type in cls._model_registry.keys()] 