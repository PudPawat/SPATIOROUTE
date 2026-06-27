"""
InternVL Model Loader Framework

This module provides a robust framework for loading InternVL models
using InternVL's own framework and API (model.chat() method).
"""

import torch
from typing import Optional, Dict, Any, Tuple, List, Union
from PIL import Image
import warnings
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        CLIPImageProcessor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class InternVLLoader:
    """
    Robust loader for InternVL models using InternVL's own framework.
    Uses AutoModel with model.chat() method for inference.
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model_name_lower = model_name.lower()
        
        # Verify this is an InternVL model
        is_internvl = (
            "internvl" in self.model_name_lower or
            "opengvlab/internvl" in self.model_name_lower or
            ("internlm" in self.model_name_lower and "vl" in self.model_name_lower)
        )
        
        if not is_internvl:
            raise ValueError(
                f"Model {model_name} is not an InternVL model. "
                f"InternVL models should contain 'internvl' in the name."
            )
        
        self.model_version = "InternVL"
    
    def load_tokenizer(self, trust_remote_code: bool = True):
        """
        Load the tokenizer for the model.
        
        Args:
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Tokenizer instance
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code
            )
            logger.info(f"   ✓ Tokenizer loaded successfully for {self.model_version}")
            return tokenizer
        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "404" in error_str:
                raise ValueError(
                    f"❌ Model '{self.model_name}' not found on HuggingFace!\n"
                    f"Please check the model name and try again.\n"
                    f"Available InternVL models:\n"
                    f"  - OpenGVLab/InternVL2-8B\n"
                    f"  - OpenGVLab/InternVL2-26B\n"
                    f"  - OpenGVLab/InternVL-Chat-Chinese-V1-1\n"
                    f"  - OpenGVLab/InternVL-Chat-V1-5\n"
                ) from e
            raise
    
    def load_image_processor(self, trust_remote_code: bool = True):
        """
        Load the image processor (CLIPImageProcessor) for the model.
        
        Args:
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Image processor instance
        """
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=trust_remote_code
            )
            logger.info(f"   ✓ Image processor loaded successfully for {self.model_version}")
            return image_processor
        except Exception as e:
            logger.warning(f"   ⚠️  Failed to load CLIPImageProcessor: {str(e)[:100]}")
            logger.warning(f"   Will try to use AutoProcessor as fallback...")
            try:
                from transformers import AutoProcessor
                image_processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trust_remote_code
                )
                logger.info(f"   ✓ AutoProcessor loaded as fallback for {self.model_version}")
                return image_processor
            except Exception as e2:
                raise RuntimeError(
                    f"❌ Failed to load image processor for {self.model_version}.\n"
                    f"   Tried CLIPImageProcessor and AutoProcessor.\n"
                    f"   Error: {str(e2)[:200]}"
                ) from e2
    
    def get_model_kwargs(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ) -> Dict[str, Any]:
        """
        Get model loading kwargs with proper configuration for InternVL.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization
            load_in_8bit: Whether to use 8-bit quantization
            torch_dtype: Torch dtype to use
            
        Returns:
            Dictionary of model kwargs
        """
        if torch_dtype is None:
            torch_dtype = torch.float32  # Load as float32 first to avoid meta tensor issues
        
        kwargs = {
            "low_cpu_mem_usage": False,  # CRITICAL: Disable low_cpu_mem_usage to avoid meta tensors
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        # InternVL has known compatibility issues with quantization
        # Disable quantization and warn user
        if load_in_4bit or load_in_8bit:
            logger.warning(
                f"   ⚠️  WARNING: InternVL has compatibility issues with bitsandbytes quantization!\n"
                f"   InternVL models will be loaded without quantization.\n"
                f"   This requires significant GPU memory (~16GB for InternVL2-8B).\n"
                f"   If you run out of memory, try:\n"
                f"     - Using a smaller model\n"
                f"     - Reducing --max-frames\n"
                f"     - Using CPU inference"
            )
            # Don't set quantization config for InternVL
        
        return kwargs
    
    def load_model(
        self,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: Optional[str] = None,
        **extra_kwargs
    ) -> Any:
        """
        Load the InternVL model using AutoModel.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization (ignored for InternVL)
            load_in_8bit: Whether to use 8-bit quantization (ignored for InternVL)
            device_map: Device map configuration
            **extra_kwargs: Additional kwargs to pass to from_pretrained
            
        Returns:
            Loaded model instance
        """
        if device_map is None:
            # InternVL works better without device_map="auto" due to meta tensor issues
            # We'll load to CPU first, then move to GPU manually
            device_map = None
        
        # Get base kwargs
        model_kwargs = self.get_model_kwargs(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )
        model_kwargs.update(extra_kwargs)
        model_kwargs["device_map"] = device_map
        
        # InternVL uses AutoModel, not AutoModelForCausalLM
        logger.info(f"   Loading {self.model_version} model: {self.model_name}")
        logger.info(f"   Using AutoModel (InternVL framework)")
        logger.info(f"   Loading to CPU first to avoid meta tensor issues...")
        
        try:
            # CRITICAL: Load to CPU first with low_cpu_mem_usage=False to avoid meta tensors
            # InternVL's initialization code calls .item() on tensors, which fails with meta tensors
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Load as float32 first
                device_map=None,  # No device_map to avoid meta tensors
                low_cpu_mem_usage=False,  # Disable to avoid meta tensors
                trust_remote_code=True
            )
            
            # Set model to eval mode
            model.eval()
            
            # Move to device and convert dtype after loading
            if self.device == "cuda":
                logger.info(f"   Moving {self.model_version} model to GPU and converting to bfloat16...")
                model = model.to(self.device)
                model = model.to(torch.bfloat16)
            else:
                logger.info(f"   Model loaded on CPU")
            
            logger.info(f"   ✓ {self.model_version} model loaded successfully!")
            return model
            
        except RuntimeError as e:
            error_str = str(e).lower()
            
            # Handle meta tensor issues (common with InternVL)
            if "meta" in error_str or "item()" in error_str or "Cannot copy out of meta tensor" in error_str:
                logger.warning(f"   ⚠️  Meta tensor issue detected, retrying with low_cpu_mem_usage=False...")
                
                try:
                    # Retry with low_cpu_mem_usage=False to completely disable meta tensors
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=False,  # Disable meta tensors
                        trust_remote_code=True
                    )
                    model.eval()
                    
                    # Move to device and convert
                    if self.device == "cuda":
                        logger.info(f"   Moving {self.model_version} model to GPU and converting to bfloat16...")
                        model = model.to(self.device)
                        model = model.to(torch.bfloat16)
                    
                    logger.info(f"   ✓ {self.model_version} model loaded with workaround!")
                    return model
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                        f"   Error: {str(e2)[:500]}\n\n"
                        f"💡 Solutions:\n"
                        f"   1. Ensure you have enough GPU memory (~16GB for InternVL2-8B)\n"
                        f"   2. Try loading on CPU first: device='cpu'\n"
                        f"   3. Check if model exists: https://huggingface.co/{self.model_name}\n"
                        f"   4. Upgrade transformers: pip install --upgrade transformers\n"
                        f"   5. Try a different InternVL model variant"
                    ) from e2
            
            # Handle other RuntimeErrors
            raise RuntimeError(
                f"❌ Failed to load {self.model_version} model '{self.model_name}'.\n"
                f"   Error: {str(e)[:500]}\n\n"
                f"💡 Try: pip install --upgrade transformers"
            ) from e
        
        except AttributeError as e:
            # Handle quantization compatibility issues
            if "all_tied_weights_keys" in str(e) or "CB" in str(e):
                logger.warning(
                    f"   ⚠️  InternVL quantization compatibility issue detected.\n"
                    f"   Retrying without quantization..."
                )
                
                # Retry without quantization and with low_cpu_mem_usage=False
                try:
                    model = AutoModel.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map=None,
                        low_cpu_mem_usage=False,  # Disable meta tensors
                        trust_remote_code=True
                    )
                    model.eval()
                    
                    if self.device == "cuda":
                        logger.info(f"   Moving {self.model_version} model to GPU and converting to bfloat16...")
                        model = model.to(self.device)
                        model = model.to(torch.bfloat16)
                    
                    logger.info(f"   ✓ {self.model_version} model loaded without quantization!")
                    logger.warning(f"   ⚠️  Warning: Requires ~16GB GPU memory.")
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
    ) -> Tuple[Any, Any, Any]:
        """
        Main loading method that returns tokenizer, image_processor, and model.
        
        Args:
            load_in_4bit: Whether to use 4-bit quantization (ignored for InternVL)
            load_in_8bit: Whether to use 8-bit quantization (ignored for InternVL)
            device_map: Device map configuration
            
        Returns:
            Tuple of (tokenizer, image_processor, model)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Load image processor
        image_processor = self.load_image_processor()
        
        # Load model
        model = self.load_model(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map
        )
        
        return tokenizer, image_processor, model
    
    def chat(
        self,
        model: Any,
        tokenizer: Any,
        image_processor: Any,
        images: Union[Image.Image, List[Image.Image]],
        question: str,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use InternVL's chat method for inference.
        
        Args:
            model: The loaded InternVL model
            tokenizer: The tokenizer
            image_processor: The image processor (CLIPImageProcessor)
            images: Single PIL Image or list of PIL Images
            question: The question text
            generation_config: Optional generation config dict
            
        Returns:
            Generated response text
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]
        
        # Process images with image processor
        # InternVL expects pixel_values in bfloat16
        pixel_values_list = []
        for img in images:
            pixel_values = image_processor(images=img, return_tensors='pt').pixel_values
            if self.device == "cuda":
                pixel_values = pixel_values.to(torch.bfloat16).to(self.device)
            else:
                pixel_values = pixel_values.to(self.device)
            pixel_values_list.append(pixel_values)
        
        # For multiple images, we need to handle them appropriately
        # InternVL chat typically handles one image at a time or concatenated images
        if len(pixel_values_list) == 1:
            pixel_values = pixel_values_list[0]
        else:
            # For multiple images, concatenate along batch dimension
            # Note: This may need adjustment based on InternVL version
            pixel_values = torch.cat(pixel_values_list, dim=0)
        
        # Default generation config
        if generation_config is None:
            generation_config = dict(
                num_beams=1,
                max_new_tokens=512,
                do_sample=False
            )
        
        # Use InternVL's chat method
        try:
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config
            )
            return response
        except Exception as e:
            logger.error(f"   ❌ InternVL chat error: {str(e)}")
            # Fallback: try with single image if multiple images failed
            if len(pixel_values_list) > 1:
                logger.warning(f"   ⚠️  Trying with first image only...")
                try:
                    response = model.chat(
                        tokenizer,
                        pixel_values_list[0],
                        question,
                        generation_config
                    )
                    return response
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ InternVL chat failed even with single image.\n"
                        f"   Error: {str(e2)[:500]}"
                    ) from e2
            raise


def create_internvl_loader(model_name: str, device: str = "cuda") -> InternVLLoader:
    """
    Factory function to create an InternVL loader.
    
    Args:
        model_name: Name of the model
        device: Device to load on
        
    Returns:
        InternVLLoader instance
    """
    return InternVLLoader(model_name, device)
