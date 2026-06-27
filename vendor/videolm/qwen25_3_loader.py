"""
Qwen2.5-VL and Qwen3-VL Model Loader Framework

This module provides a robust framework for loading Qwen2.5-VL and Qwen3-VL models
with proper architecture handling, error recovery, and fallback mechanisms.
"""

import torch
from typing import Optional, Dict, Any, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        Qwen2VLForConditionalGeneration,
        AutoConfig
    )
    # Try to import Qwen2.5-VL and Qwen3-VL specific classes if available
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN25_CLASS_AVAILABLE = True
    except ImportError:
        QWEN25_CLASS_AVAILABLE = False
        
    try:
        from transformers import Qwen3VLForConditionalGeneration
        QWEN3_CLASS_AVAILABLE = True
    except ImportError:
        QWEN3_CLASS_AVAILABLE = False
        
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    QWEN25_CLASS_AVAILABLE = False
    QWEN3_CLASS_AVAILABLE = False


class Qwen25_3ModelLoader:
    """
    Robust loader for Qwen2.5-VL and Qwen3-VL models with architecture compatibility handling.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model_name_lower = model_name.lower()
        self.is_qwen25 = "2.5" in model_name or "qwen2_5" in self.model_name_lower
        self.is_qwen3 = "qwen3" in self.model_name_lower or ("3." in model_name and "vl" in self.model_name_lower)
        
        # Determine model version
        if self.is_qwen25:
            self.model_version = "Qwen2.5-VL"
        elif self.is_qwen3:
            self.model_version = "Qwen3-VL"
        else:
            raise ValueError(f"Model {model_name} is not a Qwen2.5-VL or Qwen3-VL model")
    
    def detect_model_class(self) -> Tuple[Any, str]:
        """
        Detect the best model class to use for this model.
        
        Returns:
            Tuple of (model_class, class_name)
        """
        if self.is_qwen25:
            # Try Qwen2.5-VL specific class first
            if QWEN25_CLASS_AVAILABLE:
                logger.info(f"   Using Qwen2_5_VLForConditionalGeneration for {self.model_version}")
                return Qwen2_5_VLForConditionalGeneration, "Qwen2_5_VLForConditionalGeneration"
            
            # Fallback to Qwen2VLForConditionalGeneration (with ignore_mismatched_sizes)
            logger.info(f"   Using Qwen2VLForConditionalGeneration (fallback) for {self.model_version}")
            return Qwen2VLForConditionalGeneration, "Qwen2VLForConditionalGeneration"
        
        elif self.is_qwen3:
            # Try Qwen3-VL specific class first
            if QWEN3_CLASS_AVAILABLE:
                logger.info(f"   Using Qwen3VLForConditionalGeneration for {self.model_version}")
                return Qwen3VLForConditionalGeneration, "Qwen3VLForConditionalGeneration"
            
            # Fallback to Qwen2VLForConditionalGeneration (with ignore_mismatched_sizes)
            logger.info(f"   Using Qwen2VLForConditionalGeneration (fallback) for {self.model_version}")
            return Qwen2VLForConditionalGeneration, "Qwen2VLForConditionalGeneration"
        
        raise ValueError(f"Unknown model type: {self.model_name}")
    
    def load_processor(self, trust_remote_code: bool = True):
        """
        Load the processor for the model.
        
        Args:
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Processor instance
        """
        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code
            )
            logger.info(f"   ✓ Processor loaded successfully for {self.model_version}")
            return processor
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                if "2b" in self.model_name_lower and "qwen2.5" in self.model_name_lower:
                    raise ValueError(
                        f"❌ Model '{self.model_name}' does not exist!\n\n"
                        f"Qwen2.5-VL does not have a 2B model. Available models:\n"
                        f"  - Qwen/Qwen2.5-VL-3B-Instruct\n"
                        f"  - Qwen/Qwen2.5-VL-7B-Instruct\n"
                        f"  - Qwen/Qwen2.5-VL-72B-Instruct"
                    ) from e
                else:
                    raise ValueError(
                        f"❌ Model '{self.model_name}' not found on HuggingFace!\n"
                        f"Please check the model name and try again."
                    ) from e
            raise
    
    def get_model_kwargs(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """
        Get model loading kwargs with proper configuration for Qwen2.5-VL/Qwen3-VL.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            torch_dtype: Torch dtype to use
            
        Returns:
            Dictionary of model kwargs
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            # CRITICAL: Always ignore size mismatches for Qwen2.5-VL and Qwen3-VL
            "ignore_mismatched_sizes": True,
        }
        
        # Handle quantization
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if load_in_4bit:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif load_in_8bit:
                    # WARNING: 8-bit quantization with Qwen2.5-VL/Qwen3-VL often causes 'CB' errors
                    # The vision encoder (Conv3d layers) is incompatible with bitsandbytes quantization
                    # Even with skip_modules, quantization may still be applied during inference
                    logger.warning(
                        f"   ⚠️  WARNING: 8-bit quantization with {self.model_version} may cause 'CB' attribute errors!\n"
                        f"   The vision encoder is incompatible with bitsandbytes quantization.\n"
                        f"   If you encounter errors, use --no-quantization instead.\n"
                        f"   Attempting to exclude vision encoder from quantization..."
                    )
                    # Try to exclude all possible vision encoder module paths
                    skip_modules = [
                        "visual",
                        "model.visual",
                        "visual.patch_embed",
                        "model.visual.patch_embed",
                        "visual.blocks",
                        "model.visual.blocks",
                        "visual.merger",
                        "model.visual.merger",
                    ]
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=skip_modules
                    )
                    logger.warning(
                        f"   ⚠️  Note: Even with skip_modules, 'CB' errors may still occur during inference.\n"
                        f"   Most reliable solution: Use --no-quantization flag"
                    )
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install with: pip install bitsandbytes"
                )
        
        return kwargs
    
    def load_model(
        self,
        model_class: Any,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        **extra_kwargs
    ) -> Any:
        """
        Load the model with robust error handling and retry mechanisms.
        
        Args:
            model_class: The model class to use
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            device_map: Device map configuration
            **extra_kwargs: Additional kwargs to pass to from_pretrained
            
        Returns:
            Loaded model instance
        """
        if device_map is None:
            device_map = "auto" if self.device == "cuda" else None
        
        # Get base kwargs
        model_kwargs = self.get_model_kwargs(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        model_kwargs.update(extra_kwargs)
        model_kwargs["device_map"] = device_map
        
        # Strategy 1: Try loading with detected model class
        logger.info(f"   Loading {self.model_version} model: {self.model_name}")
        logger.info(f"   Using model class: {model_class.__name__}")
        
        try:
            model = model_class.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            logger.info(f"   ✓ {self.model_version} model loaded successfully!")
            return model
        except RuntimeError as e:
            error_str = str(e).lower()
            
            # Check if it's a size mismatch error
            if "ignore_mismatched_sizes" in error_str or "mismatched" in error_str:
                logger.warning(f"   ⚠️  Size mismatch detected, retrying with explicit ignore_mismatched_sizes=True")
                
                # Strategy 2: Retry with explicit ignore_mismatched_sizes
                model_kwargs["ignore_mismatched_sizes"] = True
                try:
                    model = model_class.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                    logger.info(f"   ✓ {self.model_version} model loaded with ignore_mismatched_sizes=True!")
                    return model
                except Exception as e2:
                    # Strategy 3: Try with AutoModelForCausalLM as last resort
                    if model_class != AutoModelForCausalLM:
                        logger.warning(f"   ⚠️  Retrying with AutoModelForCausalLM as fallback...")
                        try:
                            model_kwargs_auto = model_kwargs.copy()
                            model = AutoModelForCausalLM.from_pretrained(
                                self.model_name,
                                **model_kwargs_auto
                            )
                            logger.info(f"   ✓ {self.model_version} model loaded with AutoModelForCausalLM!")
                            return model
                        except Exception as e3:
                            raise RuntimeError(
                                f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                                f"   Tried: {model_class.__name__} and AutoModelForCausalLM\n"
                                f"   Error: {str(e3)[:500]}\n\n"
                                f"💡 Solutions:\n"
                                f"   1. Upgrade transformers: pip install --upgrade 'transformers>=4.50.0'\n"
                                f"   2. Check if model exists: https://huggingface.co/{self.model_name}\n"
                                f"   3. Try without quantization: --no-quantization"
                            ) from e3
                    raise RuntimeError(
                        f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                        f"   Error: {str(e2)[:500]}\n\n"
                        f"💡 Solutions:\n"
                        f"   1. Upgrade transformers: pip install --upgrade 'transformers>=4.50.0'\n"
                        f"   2. Try without quantization: --no-quantization"
                    ) from e2
            
            # Handle other RuntimeErrors
            raise RuntimeError(
                f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                f"   Error: {str(e)[:500]}\n\n"
                f"💡 Try: pip install --upgrade transformers"
            ) from e
        
        except AttributeError as e:
            # Handle 'CB' attribute errors (quantization issues)
            if "'Parameter' object has no attribute 'CB'" in str(e) or "has no attribute 'CB'" in str(e):
                raise RuntimeError(
                    f"❌ {self.model_version} quantization error: Vision encoder incompatibility!\n"
                    f"   Error: {str(e)}\n\n"
                    f"💡 Solutions:\n"
                    f"   1. Run WITHOUT quantization: Use --no-quantization flag\n"
                    f"   2. The vision encoder will use full precision (requires more GPU memory)\n\n"
                    f"   Example: python evaluate_sqa_qwen25.py --no-quantization"
                ) from e
            raise
        
        except AssertionError as e:
            # Handle bitsandbytes AssertionError
            if load_in_4bit or load_in_8bit:
                raise RuntimeError(
                    f"❌ {self.model_version} quantization error!\n"
                    f"   Error: {str(e)}\n\n"
                    f"💡 Solutions:\n"
                    f"   1. Use 8-bit instead of 4-bit: --load-in-8bit\n"
                    f"   2. Run without quantization: --no-quantization\n"
                    f"   3. Upgrade bitsandbytes: pip install --upgrade bitsandbytes"
                ) from e
            raise
    
    def load(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """
        Main loading method that returns both processor and model.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            device_map: Device map configuration
            
        Returns:
            Tuple of (processor, model)
        """
        # Load processor
        processor = self.load_processor()
        
        # Detect model class
        model_class, class_name = self.detect_model_class()
        
        # Load model
        model = self.load_model(
            model_class=model_class,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map
        )
        
        return processor, model


def create_qwen25_3_loader(model_name: str, device: str = "cuda") -> Qwen25_3ModelLoader:
    """
    Factory function to create a Qwen2.5-VL or Qwen3-VL loader.
    
    Args:
        model_name: Name of the model
        device: Device to load on
        
    Returns:
        Qwen25_3ModelLoader instance
    """
    return Qwen25_3ModelLoader(model_name, device)
