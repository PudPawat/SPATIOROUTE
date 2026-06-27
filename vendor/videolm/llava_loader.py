"""
LLaVA Model Loader Framework

This module provides a robust framework for loading LLaVA models
using LLaVA's standard transformers API.
"""

import torch
from typing import Optional, Dict, Any, Tuple, List, Union
from PIL import Image
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        LlavaProcessor,
        LlavaForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLaVALoader:
    """
    Robust loader for LLaVA models using transformers.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model_name_lower = model_name.lower()
        
        # Verify this is a LLaVA model
        is_llava = (
            "llava" in self.model_name_lower or
            "llava-hf" in self.model_name_lower or
            "liuhaotian/llava" in self.model_name_lower
        )
        
        if not is_llava:
            raise ValueError(
                f"Model {model_name} is not a LLaVA model. "
                f"LLaVA models should contain 'llava' in the name."
            )
        
        self.model_version = "LLaVA"
    
    def load_processor(self, trust_remote_code: bool = True):
        """
        Load the processor for the model.
        
        Args:
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Processor instance
        """
        try:
            # Try LlavaProcessor first, fallback to AutoProcessor
            try:
                processor = LlavaProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"   ✓ LlavaProcessor loaded successfully for {self.model_version}")
            except (ImportError, AttributeError):
                processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"   ✓ AutoProcessor loaded successfully for {self.model_version}")
            
            return processor
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                raise ValueError(
                    f"❌ Model '{self.model_name}' not found on HuggingFace!\n"
                    f"Please check the model name and try again.\n"
                    f"Available LLaVA models:\n"
                    f"  - llava-hf/llava-1.5-7b-hf\n"
                    f"  - llava-hf/llava-1.5-13b-hf\n"
                    f"  - llava-hf/llava-1.6-mistral-7b-hf\n"
                    f"  - llava-hf/llava-1.6-vicuna-7b-hf\n"
                    f"  - llava-hf/llava-1.6-vicuna-13b-hf\n"
                    f"  - microsoft/llava-1.5-7b\n"
                    f"  - microsoft/llava-1.5-13b"
                ) from e
            raise
    
    def get_model_kwargs(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """
        Get model loading kwargs with proper configuration for LLaVA.
        
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
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install with: pip install bitsandbytes"
                )
        
        return kwargs
    
    def load_model(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        **extra_kwargs
    ) -> Any:
        """
        Load the LLaVA model.
        
        Args:
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
        
        # Try LlavaForConditionalGeneration first, fallback to AutoModelForCausalLM
        logger.info(f"   Loading {self.model_version} model: {self.model_name}")
        
        try:
            # Try LlavaForConditionalGeneration first
            try:
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                logger.info(f"   Using LlavaForConditionalGeneration")
            except (ImportError, AttributeError, ValueError):
                # Fallback to AutoModelForCausalLM
                logger.info(f"   Using AutoModelForCausalLM (fallback)")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            
            # Set model to eval mode
            model.eval()
            
            # Move to device if needed (device_map="auto" handles this for CUDA)
            if self.device == "cpu" and device_map is None:
                model = model.to(self.device)
            
            logger.info(f"   ✓ {self.model_version} model loaded successfully!")
            return model
            
        except RuntimeError as e:
            error_str = str(e).lower()
            
            # Handle memory errors
            if "out of memory" in error_str or "cuda" in error_str:
                logger.warning(f"   ⚠️  Memory issue detected, retrying with different settings...")
                
                # Retry with 8-bit quantization if 4-bit was requested
                if load_in_4bit and not load_in_8bit:
                    logger.info(f"   Retrying with 8-bit quantization instead of 4-bit...")
                    retry_kwargs = self.get_model_kwargs(
                        load_in_4bit=False,
                        load_in_8bit=True
                    )
                    retry_kwargs["device_map"] = device_map
                    retry_kwargs.update(extra_kwargs)
                    
                    try:
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            **retry_kwargs
                        )
                        model.eval()
                        logger.info(f"   ✓ {self.model_version} model loaded with 8-bit quantization!")
                        return model
                    except Exception as e2:
                        raise RuntimeError(
                            f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                            f"   Error: {str(e2)[:500]}\n\n"
                            f"💡 Solutions:\n"
                            f"   1. Ensure you have enough GPU memory\n"
                            f"   2. Try loading on CPU: device='cpu'\n"
                            f"   3. Use a smaller LLaVA model"
                        ) from e2
            
            # Handle other RuntimeErrors
            raise RuntimeError(
                f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                f"   Error: {str(e)[:500]}\n\n"
                f"💡 Try: pip install --upgrade transformers"
            ) from e
        
        except AttributeError as e:
            # Handle quantization compatibility issues
            if "CB" in str(e) or "quantization" in str(e).lower():
                logger.warning(
                    f"   ⚠️  LLaVA quantization compatibility issue detected.\n"
                    f"   Retrying without quantization..."
                )
                
                # Retry without quantization
                retry_kwargs = self.get_model_kwargs(
                    load_in_4bit=False,
                    load_in_8bit=False
                )
                retry_kwargs["device_map"] = device_map
                retry_kwargs.update(extra_kwargs)
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **retry_kwargs
                    )
                    model.eval()
                    if self.device == "cpu" and device_map is None:
                        model = model.to(self.device)
                    
                    logger.info(f"   ✓ {self.model_version} model loaded without quantization!")
                    logger.warning(f"   ⚠️  Warning: Requires more GPU memory.")
                    return model
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                        f"   Error: {str(e2)[:500]}"
                    ) from e2
            raise
    
    def load(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """
        Main loading method that returns processor and model.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            device_map: Device map configuration
            
        Returns:
            Tuple of (processor, model)
        """
        # Load processor
        processor = self.load_processor()
        
        # Load model
        model = self.load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map
        )
        
        return processor, model
    
    def generate(
        self,
        model: Any,
        processor: Any,
        images: Union[Image.Image, List[Image.Image]],
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using LLaVA model.
        
        Args:
            model: The loaded LLaVA model
            processor: The processor
            images: Single PIL Image or list of PIL Images
            question: The question text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]
        
        # Prepare prompt - LLaVA uses a specific format
        # Format: "USER: <image>\n{question}\nASSISTANT:"
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        
        # Process inputs
        inputs = processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0
            )
        
        # Decode response
        # Remove input tokens from output
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response_text[0].strip()


def create_llava_loader(model_name: str, device: str = "cuda") -> LLaVALoader:
    """
    Factory function to create a LLaVA loader.
    
    Args:
        model_name: Name of the model
        device: Device to load on
        
    Returns:
        LLaVALoader instance
    """
    return LLaVALoader(model_name, device)
