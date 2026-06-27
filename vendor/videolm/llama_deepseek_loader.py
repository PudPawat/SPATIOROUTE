"""
Llama Vision and DeepSeek-VL2 Model Loader Framework

This module provides a robust framework for loading Llama Vision and DeepSeek-VL2 models
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
        AutoConfig
    )
    # Try to import Llama-specific classes if available
    try:
        from transformers import LlamaForCausalLM
        LLAMA_CLASS_AVAILABLE = True
    except ImportError:
        LLAMA_CLASS_AVAILABLE = False
    
    # Try to import DeepSeek-VL2 specific classes if available
    try:
        from transformers import DeepSeekV2ForCausalLM
        DEEPSEEK_CLASS_AVAILABLE = True
    except ImportError:
        DEEPSEEK_CLASS_AVAILABLE = False
        
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    LLAMA_CLASS_AVAILABLE = False
    DEEPSEEK_CLASS_AVAILABLE = False


class LlamaDeepSeekModelLoader:
    """
    Robust loader for Llama Vision and DeepSeek-VL2 models with architecture compatibility handling.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model_name_lower = model_name.lower()
        self.is_llama = (
            ("llama" in self.model_name_lower and ("vision" in self.model_name_lower or "vl" in self.model_name_lower)) or
            ("llama-3.2" in self.model_name_lower and "vision" in self.model_name_lower) or
            ("llama-3.3" in self.model_name_lower and "vision" in self.model_name_lower) or
            ("llama-4" in self.model_name_lower and "vision" in self.model_name_lower) or
            ("meta-llama" in self.model_name_lower and ("vision" in self.model_name_lower or "vl" in self.model_name_lower))
        )
        self.is_deepseek = (
            ("deepseek" in self.model_name_lower and "vl" in self.model_name_lower) or
            ("deepseek-vl2" in self.model_name_lower) or
            ("deepseek_vl2" in self.model_name_lower) or
            ("deepseek-ai" in self.model_name_lower and "vl" in self.model_name_lower)
        )
        
        # Determine model version
        if self.is_llama:
            self.model_version = "Llama Vision"
        elif self.is_deepseek:
            self.model_version = "DeepSeek-VL2"
        else:
            raise ValueError(f"Model {model_name} is not a Llama Vision or DeepSeek-VL2 model")
    
    def detect_model_class(self) -> Tuple[Any, str]:
        """
        Detect the best model class to use for this model.
        
        Returns:
            Tuple of (model_class, class_name)
        """
        if self.is_llama:
            # Try Llama-specific class first
            if LLAMA_CLASS_AVAILABLE:
                logger.info(f"   Using LlamaForCausalLM for {self.model_version}")
                return LlamaForCausalLM, "LlamaForCausalLM"
            
            # Fallback to AutoModelForCausalLM
            logger.info(f"   Using AutoModelForCausalLM (fallback) for {self.model_version}")
            return AutoModelForCausalLM, "AutoModelForCausalLM"
        
        elif self.is_deepseek:
            # Try DeepSeek-VL2 specific class first
            if DEEPSEEK_CLASS_AVAILABLE:
                logger.info(f"   Using DeepSeekV2ForCausalLM for {self.model_version}")
                return DeepSeekV2ForCausalLM, "DeepSeekV2ForCausalLM"
            
            # Try Qwen2VLForConditionalGeneration (similar architecture)
            try:
                from transformers import Qwen2VLForConditionalGeneration
                logger.info(f"   Using Qwen2VLForConditionalGeneration (fallback) for {self.model_version}")
                return Qwen2VLForConditionalGeneration, "Qwen2VLForConditionalGeneration"
            except ImportError:
                # Final fallback to AutoModelForCausalLM
                logger.info(f"   Using AutoModelForCausalLM (fallback) for {self.model_version}")
                return AutoModelForCausalLM, "AutoModelForCausalLM"
        
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
                raise ValueError(
                    f"❌ Model '{self.model_name}' not found on HuggingFace!\n"
                    f"Please check the model name and try again.\n"
                    f"For Llama Vision models, check: https://huggingface.co/models?search=llama+vision\n"
                    f"For DeepSeek-VL2, check: https://huggingface.co/deepseek-ai/DeepSeek-VL2"
                ) from e
            raise
    
    def get_model_kwargs(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """
        Get model loading kwargs with proper configuration for Llama Vision/DeepSeek-VL2.
        
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
        }
        
        # Llama Vision and DeepSeek-VL2 may have architecture differences
        # Set ignore_mismatched_sizes if needed (similar to Qwen2.5-VL/Qwen3-VL)
        if self.is_llama or self.is_deepseek:
            kwargs["ignore_mismatched_sizes"] = True
        
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
                    # WARNING: 8-bit quantization may cause 'CB' errors with vision encoders
                    logger.warning(
                        f"   ⚠️  WARNING: 8-bit quantization with {self.model_version} may cause 'CB' attribute errors!\n"
                        f"   The vision encoder may be incompatible with bitsandbytes quantization.\n"
                        f"   If you encounter errors, use --no-quantization instead.\n"
                        f"   Attempting to exclude vision encoder from quantization..."
                    )
                    # Try to exclude vision encoder modules
                    skip_modules = [
                        "visual",
                        "model.visual",
                        "vision_model",
                        "model.vision_model",
                        "vision",
                        "model.vision",
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
            if "ignore_mismatched_sizes" in error_str or "mismatched" in error_str or "size mismatch" in error_str:
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
                    error_str2 = str(e2).lower()
                    if "ignore_mismatched_sizes" in error_str2 or "mismatched" in error_str2:
                        raise RuntimeError(
                            f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                            f"   {self.model_version} has architecture differences.\n"
                            f"   Error: {str(e2)[:500]}\n\n"
                            f"💡 Solutions:\n"
                            f"   1. Upgrade transformers: pip install --upgrade 'transformers>=4.50.0'\n"
                            f"   2. Check if model exists: https://huggingface.co/{self.model_name}\n"
                            f"   3. Try without quantization: --no-quantization"
                        ) from e2
                    else:
                        raise
            
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
                    f"   Example: python evaluate_sqa_llama4.py --no-quantization"
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


def create_llama_deepseek_loader(model_name: str, device: str = "cuda") -> LlamaDeepSeekModelLoader:
    """
    Factory function to create a Llama Vision or DeepSeek-VL2 loader.
    
    Args:
        model_name: Name of the model
        device: Device to load on
        
    Returns:
        LlamaDeepSeekModelLoader instance
    """
    return LlamaDeepSeekModelLoader(model_name, device)
