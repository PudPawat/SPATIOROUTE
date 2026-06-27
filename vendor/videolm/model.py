"""
VideoLM Model: Qwen2-VL based Video Question Answering
"""

import torch
from typing import List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
try:
    from importlib.metadata import PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import PackageNotFoundError
try:
    from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration
    # Qwen2.5-VL uses AutoProcessor but Qwen2VLForConditionalGeneration
    from transformers import AutoProcessor
    QWEN25_AVAILABLE = True
except ImportError:
    raise ImportError(
        "Qwen2-VL models require transformers>=4.37.0. "
        "Please install with: pip install transformers>=4.37.0"
    )

from .video_processor import VideoProcessor
from .multimodal_thinking import answer_from_images_with_thinking, split_qwen_thinking_string
from .prompt_utils import format_prompt_template

# Import the new Qwen2.5-VL/Qwen3-VL loader framework
try:
    from .qwen25_3_loader import Qwen25_3ModelLoader, create_qwen25_3_loader
    QWEN25_3_LOADER_AVAILABLE = True
except ImportError:
    QWEN25_3_LOADER_AVAILABLE = False
    Qwen25_3ModelLoader = None
    create_qwen25_3_loader = None

# Import the new Llama Vision/DeepSeek-VL2 loader framework
try:
    from .llama_deepseek_loader import LlamaDeepSeekModelLoader, create_llama_deepseek_loader
    LLAMA_DEEPSEEK_LOADER_AVAILABLE = True
except ImportError:
    LLAMA_DEEPSEEK_LOADER_AVAILABLE = False
    LlamaDeepSeekModelLoader = None
    create_llama_deepseek_loader = None

# Import the new InternVL loader framework
try:
    from .internvl_loader import InternVLLoader, create_internvl_loader
    INTERNVL_LOADER_AVAILABLE = True
except ImportError:
    INTERNVL_LOADER_AVAILABLE = False
    InternVLLoader = None
    create_internvl_loader = None

# Import the new LLaVA loader framework
try:
    from .llava_loader import LLaVALoader, create_llava_loader
    LLAVA_LOADER_AVAILABLE = True
except ImportError:
    LLAVA_LOADER_AVAILABLE = False
    LLaVALoader = None
    create_llava_loader = None


class VideoLM:
    """Qwen2-VL based Video Language Model for Question Answering"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
        max_frames: int = 8,
        frame_size: tuple = (448, 448),
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        prompt_template: Optional[str] = None,
        llm_model: Optional[str] = None
    ):
        """
        Initialize VideoLM model
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cuda', 'cpu', or None for auto)
            max_frames: Maximum number of frames to extract from video
            frame_size: Target size for frames (width, height)
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            prompt_template: Optional prompt template string with {question} placeholder
            llm_model: LLM model for text generation (required for CLIP models). 
                      Options: 'gpt-3.5-turbo', 'gpt-4', or local model path (e.g., 'Qwen/Qwen2-7B-Instruct')
        """
        self.model_name = model_name
        # Handle device specification: can be "cuda", "cpu", or "cuda:0", "cuda:1", etc.
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.prompt_template = prompt_template
        self.llm_model = llm_model
        self.llm_tokenizer = None
        self.llm_generator = None
        self.use_openai_api = False
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            max_frames=max_frames,
            frame_size=frame_size
        )
        
        # Load model and processor
        print(f"Loading model {model_name} on {self.device}...")
        
        # Validate that this is a VL model (not a text-only model)
        # Check for VL, Vision, or vision in model name (Llama uses "Vision", Qwen uses "VL")
        # Also accept CLIP models (image-text similarity models) and InternVL
        model_name_lower = model_name.lower()
        is_vl_model = (
            "vl" in model_name_lower or 
            "vision" in model_name_lower or
            "visual" in model_name_lower or
            ("clip" in model_name_lower and "vit" in model_name_lower) or
            "openai/clip" in model_name_lower or
            "laion/clip" in model_name_lower or
            "internvl" in model_name_lower or
            ("internlm" in model_name_lower and "vl" in model_name_lower) or
            "llava" in model_name_lower or
            "spatial-mllm" in model_name_lower
        )
        
        if not is_vl_model:
            raise ValueError(
                f"❌ Error: '{model_name}' is not a Vision-Language (VL) model!\n"
                f"This code requires Vision-Language models that can process images/videos.\n"
                f"\nAvailable Qwen2-VL models:\n"
                f"  - Qwen/Qwen2-VL-2B-Instruct (smallest, ~4-6 GB VRAM)\n"
                f"  - Qwen/Qwen2-VL-7B-Instruct (recommended, ~14-16 GB VRAM)\n"
                f"  - Qwen/Qwen2-VL-72B-Instruct (largest, ~140+ GB VRAM)\n"
                f"\nAvailable Qwen2.5-VL models:\n"
                f"  - Qwen/Qwen2.5-VL-3B-Instruct (smallest, ~6-8 GB VRAM)\n"
                f"  - Qwen/Qwen2.5-VL-7B-Instruct (recommended, ~14-16 GB VRAM)\n"
                f"  - Qwen/Qwen2.5-VL-72B-Instruct (largest, ~140+ GB VRAM)\n"
                f"\nAvailable Qwen3-VL models:\n"
                f"  - Qwen/Qwen3-VL-2B-Instruct (smallest, ~4-6 GB VRAM)\n"
                f"  - Qwen/Qwen3-VL-4B-Instruct (recommended, ~8-10 GB VRAM)\n"
                f"  - Qwen/Qwen3-VL-8B-Thinking (reasoning / long CoT, see evaluate_sqa_qwen3.py --thinking)\n"
                f"\nAvailable Llama Vision models:\n"
                f"  - meta-llama/Llama-3.2-11B-Vision-Instruct\n"
                f"  - meta-llama/Llama-3.3-70B-Vision-Instruct\n"
                f"\nAvailable InternVL models:\n"
                f"  - OpenGVLab/InternVL2-8B\n"
                f"  - OpenGVLab/InternVL2-26B\n"
                f"\nAvailable LLaVA models:\n"
                f"  - llava-hf/llava-1.5-7b-hf\n"
                f"  - llava-hf/llava-1.5-13b-hf\n"
                f"  - llava-hf/llava-1.6-mistral-7b-hf\n"
                f"  - microsoft/llava-1.5-7b\n"
                f"\nNote: Text-only models (e.g., Qwen2-1.5B-Instruct) cannot process videos.\n"
                f"Use Vision-Language models instead for video tasks."
            )
        
        # Detect model type
        model_name_lower = model_name.lower()
        is_qwen3 = "qwen3" in model_name_lower or ("3." in model_name and "vl" in model_name_lower)
        is_qwen25 = (
            "2.5" in model_name
            or "qwen2_5" in model_name_lower
            or "spatial-mllm" in model_name_lower
        )
        is_llama4 = (("llama" in model_name_lower and ("vision" in model_name_lower or "vl" in model_name_lower)) or
                     ("llama-3.2" in model_name_lower and "vision" in model_name_lower) or
                     ("llama-3.3" in model_name_lower and "vision" in model_name_lower) or
                     ("llama-4" in model_name_lower and "vision" in model_name_lower) or
                     ("meta-llama" in model_name_lower and ("vision" in model_name_lower or "vl" in model_name_lower)))
        is_deepseek_vl2 = (("deepseek" in model_name_lower and "vl" in model_name_lower) or
                           ("deepseek-vl2" in model_name_lower) or
                           ("deepseek_vl2" in model_name_lower) or
                           ("deepseek-ai" in model_name_lower and "vl" in model_name_lower))
        is_internvl = (("internvl" in model_name_lower) or
                       ("opengvlab/internvl" in model_name_lower) or
                       ("internlm" in model_name_lower and "vl" in model_name_lower))
        is_llava = (("llava" in model_name_lower) or
                    ("llava-hf" in model_name_lower) or
                    ("liuhaotian/llava" in model_name_lower) or
                    ("microsoft/llava" in model_name_lower))
        is_clip = (("clip" in model_name_lower and "vit" in model_name_lower) or
                   ("openai/clip" in model_name_lower) or
                   ("laion/clip" in model_name_lower) or
                   model_name_lower.startswith("clip-"))
        
        # Store model type flags as instance variables
        self.is_qwen3 = is_qwen3
        self.is_qwen25 = is_qwen25
        self.is_llama4 = is_llama4
        self.is_deepseek_vl2 = is_deepseek_vl2
        self.is_internvl = is_internvl
        self.is_llava = is_llava
        self.is_clip = is_clip
        
        # Use the new Qwen2.5-VL/Qwen3-VL loader framework if available
        if (is_qwen25 or is_qwen3) and QWEN25_3_LOADER_AVAILABLE:
            print(f"   🚀 Using Qwen2.5-VL/Qwen3-VL Framework for {model_name}")
            try:
                loader = create_qwen25_3_loader(model_name, self.device)
                self.processor, self.model = loader.load(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                # Set model to eval mode
                self.model.eval()
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                print("Model loaded successfully!")
                # Initialize video processor (needed for video processing)
                self.video_processor = VideoProcessor(
                    max_frames=max_frames,
                    frame_size=frame_size
                )
                # Set prompt template if provided
                if prompt_template:
                    self.prompt_template = prompt_template
                return  # Early return, model is already loaded
            except Exception as e:
                print(f"   ⚠️  Framework loader failed: {str(e)[:200]}")
                print("   Falling back to legacy loading method...")
                # Continue with legacy loading below
        
        # Use the new Llama Vision/DeepSeek-VL2 loader framework if available
        if (is_llama4 or is_deepseek_vl2) and LLAMA_DEEPSEEK_LOADER_AVAILABLE:
            print(f"   🚀 Using Llama Vision/DeepSeek-VL2 Framework for {model_name}")
            try:
                loader = create_llama_deepseek_loader(model_name, self.device)
                self.processor, self.model = loader.load(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                # Set model to eval mode
                self.model.eval()
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                print("Model loaded successfully!")
                # Initialize video processor (needed for video processing)
                self.video_processor = VideoProcessor(
                    max_frames=max_frames,
                    frame_size=frame_size
                )
                # Set prompt template if provided
                if prompt_template:
                    self.prompt_template = prompt_template
                return  # Early return, model is already loaded
            except Exception as e:
                print(f"   ⚠️  Framework loader failed: {str(e)[:200]}")
                print("   Falling back to legacy loading method...")
                # Continue with legacy loading below
        
        # Use the new InternVL loader framework if available
        if is_internvl and INTERNVL_LOADER_AVAILABLE:
            print(f"   🚀 Using InternVL Framework for {model_name}")
            try:
                loader = create_internvl_loader(model_name, self.device)
                self.tokenizer, self.image_processor, self.model = loader.load(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                # Store the loader for chat method
                self.internvl_loader = loader
                # Set model to eval mode
                self.model.eval()
                print("Model loaded successfully!")
                # Initialize video processor (needed for video processing)
                self.video_processor = VideoProcessor(
                    max_frames=max_frames,
                    frame_size=frame_size
                )
                # Set prompt template if provided
                if prompt_template:
                    self.prompt_template = prompt_template
                return  # Early return, model is already loaded
            except Exception as e:
                print(f"   ⚠️  InternVL Framework loader failed: {str(e)[:200]}")
                print("   Falling back to legacy loading method...")
                # Continue with legacy loading below
        
        # Use the new LLaVA loader framework if available
        if is_llava and LLAVA_LOADER_AVAILABLE:
            print(f"   🚀 Using LLaVA Framework for {model_name}")
            try:
                loader = create_llava_loader(model_name, self.device)
                self.processor, self.model = loader.load(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit
                )
                # Store the loader for generate method
                self.llava_loader = loader
                # Set model to eval mode
                self.model.eval()
                print("Model loaded successfully!")
                # Initialize video processor (needed for video processing)
                self.video_processor = VideoProcessor(
                    max_frames=max_frames,
                    frame_size=frame_size
                )
                # Set prompt template if provided
                if prompt_template:
                    self.prompt_template = prompt_template
                return  # Early return, model is already loaded
            except Exception as e:
                print(f"   ⚠️  LLaVA Framework loader failed: {str(e)[:200]}")
                print("   Falling back to legacy loading method...")
                # Continue with legacy loading below
        
        if is_qwen3:
            # Qwen3-VL uses AutoProcessor (similar to Qwen2.5-VL)
            print("   Using AutoProcessor for Qwen3-VL")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # Qwen3-VL likely uses Qwen2VLForConditionalGeneration class (same architecture)
            self.model_class = Qwen2VLForConditionalGeneration
        elif is_qwen25:
            # Qwen2.5-VL uses AutoProcessor but Qwen2VLForConditionalGeneration
            print("   Using AutoProcessor for Qwen2.5-VL")
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except (OSError, ValueError) as e:
                # Check if it's a model not found error
                error_str = str(e).lower()
                if "not found" in error_str or "404" in error_str or "repository not found" in error_str:
                    # Check if user tried to use non-existent 2B model
                    if "2b" in model_name.lower() and "qwen2.5" in model_name.lower():
                        raise ValueError(
                            f"❌ Model '{model_name}' does not exist!\n\n"
                            f"Qwen2.5-VL does not have a 2B model. Available Qwen2.5-VL models:\n"
                            f"  - Qwen/Qwen2.5-VL-3B-Instruct (smallest, ~6-8 GB VRAM)\n"
                            f"  - Qwen/Qwen2.5-VL-7B-Instruct (recommended, ~14-16 GB VRAM)\n"
                            f"  - Qwen/Qwen2.5-VL-72B-Instruct (largest, ~140+ GB VRAM)\n\n"
                            f"💡 Suggested fix: Use Qwen/Qwen2.5-VL-3B-Instruct instead\n"
                            f"   Example: python evaluate_sqa_qwen25.py --model-name Qwen/Qwen2.5-VL-3B-Instruct"
                        ) from e
                    else:
                        raise ValueError(
                            f"❌ Model '{model_name}' not found on HuggingFace!\n\n"
                            f"Available Qwen2.5-VL models:\n"
                            f"  - Qwen/Qwen2.5-VL-3B-Instruct (smallest)\n"
                            f"  - Qwen/Qwen2.5-VL-7B-Instruct (recommended)\n"
                            f"  - Qwen/Qwen2.5-VL-72B-Instruct (largest)\n\n"
                            f"Please check the model name and try again."
                        ) from e
                else:
                    raise
            # Qwen2.5-VL still uses Qwen2VLForConditionalGeneration class
            self.model_class = Qwen2VLForConditionalGeneration
        elif is_llama4:
            # Llama Vision models use AutoProcessor
            print("   Using AutoProcessor for Llama Vision")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # Llama Vision models (mllama) use AutoModelForCausalLM with trust_remote_code=True
            # Do NOT use LlamaForCausalLM - it's for standard Llama, not multimodal Llama
            from transformers import AutoModelForCausalLM
            self.model_class = AutoModelForCausalLM
        elif is_deepseek_vl2:
            # DeepSeek-VL2 uses AutoProcessor
            print("   Using AutoProcessor for DeepSeek-VL2")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # DeepSeek-VL2 may use Qwen2VLForConditionalGeneration or AutoModelForCausalLM
            # Try Qwen2VL first (similar architecture), fall back to AutoModel
            try:
                self.model_class = Qwen2VLForConditionalGeneration
            except:
                from transformers import AutoModelForCausalLM
                self.model_class = AutoModelForCausalLM
        elif is_internvl:
            # InternVL models use AutoProcessor
            print("   Using AutoProcessor for InternVL")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # InternVL uses AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            self.model_class = AutoModelForCausalLM
        elif is_llava:
            # LLaVA models use LlavaProcessor or AutoProcessor
            print("   Using AutoProcessor for LLaVA")
            try:
                from transformers import LlavaProcessor
                self.processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True)
            except (ImportError, AttributeError):
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # LLaVA uses LlavaForConditionalGeneration or AutoModelForCausalLM
            try:
                from transformers import LlavaForConditionalGeneration
                self.model_class = LlavaForConditionalGeneration
            except (ImportError, AttributeError):
                from transformers import AutoModelForCausalLM
                self.model_class = AutoModelForCausalLM
        elif is_clip:
            # CLIP models use CLIPProcessor and CLIPModel
            print("   Using CLIPProcessor for CLIP model")
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.processor = CLIPProcessor.from_pretrained(model_name)
                self.model_class = CLIPModel
                # Note: CLIP doesn't generate text, it computes image-text similarity
                # This will need special handling in answer_question method
                print("   ⚠️  Note: CLIP is for similarity, not generation. May need adaptation.")
            except ImportError:
                raise ImportError(
                    "CLIP models require transformers with CLIP support. "
                    "Please ensure transformers>=4.21.0 is installed."
                )
        else:
            # Use Qwen2-VL classes (default)
            self.processor = Qwen2VLProcessor.from_pretrained(model_name)
            self.model_class = Qwen2VLForConditionalGeneration
        
        # Model loading kwargs
        model_kwargs = {
            "low_cpu_mem_usage": not is_internvl,  # InternVL has issues with meta tensors, disable for it
            "trust_remote_code": True,  # Required for Qwen2-VL models and InternVL
        }
        
        # InternVL has a bug where it calls .item() on meta tensors during initialization
        # Disable fast init to prevent meta tensor usage
        if is_internvl:
            model_kwargs["_fast_init"] = False
        
        # Qwen2.5-VL and Qwen3-VL have architecture differences, need to ignore size mismatches
        if is_qwen25:
            model_kwargs["ignore_mismatched_sizes"] = True
            print("   ⚠️  Qwen2.5-VL detected: Using ignore_mismatched_sizes=True (architecture differences from Qwen2-VL)")
        elif is_qwen3:
            model_kwargs["ignore_mismatched_sizes"] = True
            print("   ⚠️  Qwen3-VL detected: Using ignore_mismatched_sizes=True (architecture differences from Qwen2-VL)")
        elif is_deepseek_vl2:
            model_kwargs["ignore_mismatched_sizes"] = True
            print("   ⚠️  DeepSeek-VL2 detected: Using ignore_mismatched_sizes=True (architecture differences - MoE, different layer counts)")
        elif is_internvl:
            model_kwargs["ignore_mismatched_sizes"] = True
            print("   ⚠️  InternVL detected: Using ignore_mismatched_sizes=True (architecture differences from Qwen2-VL)")
        
        # Handle device_map: use "auto" only if not using quantization
        # With quantization, device_map is handled by bitsandbytes
        use_device_map = False
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                # InternVL has severe compatibility issues with bitsandbytes quantization
                # Disable quantization for InternVL and load without it
                if is_internvl:
                    print("   ⚠️  InternVL: 4-bit quantization is not compatible with InternVL models.")
                    print("   ⚠️  Loading without quantization (requires ~16GB GPU memory).")
                    print("   ⚠️  If you run out of memory, try: --max-frames 4 or use a smaller model.")
                    # Don't set quantization_config for InternVL
                    load_in_4bit = False  # Disable quantization
                    load_in_8bit = False
                elif is_qwen25 or is_qwen3 or is_llama4 or is_deepseek_vl2:
                    # Qwen2.5-VL, Qwen3-VL, Llama4, DeepSeek-VL2 use more conservative settings
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                    )
                # With 4-bit quantization, device_map is handled automatically (except for InternVL)
                if not is_internvl:
                    if "cuda" in self.device:
                        if ":" in self.device:
                            use_device_map = self.device
                        else:
                            use_device_map = "auto"
                    else:
                        use_device_map = None
            except (ImportError, PackageNotFoundError) as e:
                raise ImportError(
                    "❌ bitsandbytes package is required for 4-bit quantization!\n"
                    "Please install it with: pip install bitsandbytes\n"
                    "Or install all requirements: pip install -r requirements.txt"
                ) from e
        elif load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                # InternVL has compatibility issues with 8-bit quantization too
                if is_internvl:
                    print("   ⚠️  InternVL: 8-bit quantization is not compatible with InternVL models.")
                    print("   ⚠️  Loading without quantization (requires ~16GB GPU memory).")
                    load_in_8bit = False  # Disable quantization
                # For Qwen2.5-VL, Qwen3-VL, Llama4, and DeepSeek-VL2, exclude vision encoder from quantization to avoid 'CB' attribute errors
                elif is_qwen25 or is_qwen3 or is_llama4 or is_deepseek_vl2:
                    # Exclude vision encoder modules from quantization
                    # The vision encoder (visual) has compatibility issues with bitsandbytes quantization
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_skip_modules=["visual", "model.visual"]  # Skip vision encoder quantization
                    )
                    if is_qwen25:
                        model_version = "Qwen2.5-VL"
                    elif is_qwen3:
                        model_version = "Qwen3-VL"
                    elif is_llama4:
                        model_version = "Llama4-VL"
                    elif is_deepseek_vl2:
                        model_version = "DeepSeek-VL2"
                    elif is_internvl:
                        model_version = "InternVL"
                    else:
                        model_version = "model"
                    print(f"   ⚠️  Excluding vision encoder from quantization ({model_version} compatibility)")
                else:
                    # Use BitsAndBytesConfig for 8-bit quantization (more reliable)
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                # With 8-bit quantization, device_map is handled automatically
                # If device is "cuda:X", use that specific device; otherwise use "auto"
                if "cuda" in self.device:
                    if ":" in self.device:
                        use_device_map = self.device
                    else:
                        use_device_map = "auto"
                else:
                    use_device_map = None
            except ImportError:
                raise ImportError(
                    "❌ bitsandbytes package is required for 8-bit quantization!\n"
                    "Please install it with: pip install bitsandbytes\n"
                    "Or install all requirements: pip install -r requirements.txt"
                )
        else:
            # No quantization: use device_map="auto" for CUDA, or specific device if specified
            if "cuda" in self.device:
                if ":" in self.device:
                    # Specific GPU specified (e.g., "cuda:1")
                    use_device_map = self.device
                else:
                    # Just "cuda" - let system decide
                    use_device_map = "auto"
            else:
                use_device_map = None
        
        # Workaround for transformers 4.46+ compatibility issue with Qwen2-VL
        # The error occurs in generation config initialization where decoder_config.to_dict() fails
        # because decoder_config is already a dict. We'll patch the problematic code.
        # Also handle Qwen2.5-VL and Qwen3-VL weight initialization issues
        # For InternVL: disable device_map and low_cpu_mem_usage to avoid meta tensors.
        # For Llama4 (transformers 5.2+), load AutoConfig explicitly to avoid dict/config issues.
        try:
            if is_internvl:
                # InternVL calls .item() on meta tensors during init without these settings.
                print("   ⚠️  InternVL: Disabling meta tensors (low_cpu_mem_usage=False)...")
                actual_device_map = None
                actual_torch_dtype = torch.float32
                model_kwargs["low_cpu_mem_usage"] = False
                try:
                    self.model = self.model_class.from_pretrained(
                        model_name,
                        torch_dtype=actual_torch_dtype,
                        device_map=actual_device_map,
                        **model_kwargs
                    )
                    if "cuda" in self.device:
                        print("   Moving InternVL model to GPU and converting to bfloat16...")
                        self.model = self.model.to(self.device)
                        self.model = self.model.to(torch.bfloat16)
                except AttributeError as e:
                    if "all_tied_weights_keys" in str(e):
                        print(f"\n   ⚠️  InternVL quantization compatibility issue detected: {str(e)[:100]}")
                        print("   Retrying without quantization (InternVL has compatibility issues with bitsandbytes)...")
                        retry_kwargs = model_kwargs.copy()
                        if "quantization_config" in retry_kwargs:
                            del retry_kwargs["quantization_config"]
                        retry_kwargs["low_cpu_mem_usage"] = False
                        self.model = self.model_class.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            device_map=None,
                            **retry_kwargs
                        )
                        if "cuda" in self.device:
                            print("   Moving InternVL model to GPU and converting to bfloat16...")
                            self.model = self.model.to(self.device)
                            self.model = self.model.to(torch.bfloat16)
                        print("   ⚠️  Warning: InternVL loaded without quantization. This requires ~16GB GPU memory.")
                        print("   If you run out of memory, consider using a smaller model or reducing --max-frames.")
                    else:
                        raise
            else:
                actual_device_map = use_device_map
                actual_torch_dtype = torch.bfloat16 if "cuda" in self.device else torch.float32
                if is_llama4:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    llama_kwargs = {k: v for k, v in model_kwargs.items() if k != "trust_remote_code"}
                    self.model = self.model_class.from_pretrained(
                        model_name,
                        config=config,
                        torch_dtype=actual_torch_dtype,
                        device_map=actual_device_map,
                        trust_remote_code=True,
                        **llama_kwargs
                    )
                else:
                    self.model = self.model_class.from_pretrained(
                        model_name,
                        torch_dtype=actual_torch_dtype,
                        device_map=actual_device_map,
                        **model_kwargs
                    )
        except RuntimeError as e:
            # Handle ignore_mismatched_sizes errors for Qwen2.5-VL and Qwen3-VL (must catch before AssertionError)
            error_str = str(e).lower()
            if ("ignore_mismatched_sizes" in error_str or "mismatched" in error_str or "size mismatch" in error_str) and (is_qwen25 or is_qwen3):
                model_version = "Qwen2.5-VL" if is_qwen25 else "Qwen3-VL"
                print(f"\n   ⚠️  {model_version} size mismatch error detected")
                print("   Retrying with ignore_mismatched_sizes=True (explicit)...")
                
                # Create new kwargs with explicit flag
                retry_kwargs = model_kwargs.copy()
                retry_kwargs["ignore_mismatched_sizes"] = True
                
                try:
                    self.model = self.model_class.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                        device_map=use_device_map,
                        ignore_mismatched_sizes=True,  # Explicitly pass as keyword argument
                        **{k: v for k, v in retry_kwargs.items() if k != "ignore_mismatched_sizes"}  # Include other kwargs
                    )
                    print(f"   ✓ {model_version} model loaded successfully with ignore_mismatched_sizes=True")
                except Exception as e2:
                    error_str2 = str(e2).lower()
                    if "ignore_mismatched_sizes" in error_str2 or "mismatched" in error_str2:
                        if is_qwen25:
                            raise RuntimeError(
                                f"❌ Failed to load Qwen2.5-VL model '{model_name}'.\n"
                                f"   Qwen2.5-VL has architecture differences from Qwen2-VL.\n"
                                f"   Error: {str(e2)[:500]}\n\n"
                                f"💡 Solutions:\n"
                                f"   1. Upgrade transformers: pip install --upgrade 'transformers>=4.50.0'\n"
                                f"   2. Try using Qwen2-VL instead: Qwen/Qwen2-VL-2B-Instruct\n"
                                f"   3. Check if the model checkpoint is corrupted\n\n"
                                f"   Note: Qwen2.5-VL models use Qwen2VLForConditionalGeneration class but have different architecture."
                            ) from e2
                        else:
                            raise RuntimeError(
                                f"❌ Failed to load Qwen3-VL model '{model_name}'.\n"
                                f"   Qwen3-VL has significant architecture differences from Qwen2-VL.\n"
                                f"   Error: {str(e2)[:500]}\n\n"
                                f"💡 Solutions:\n"
                                f"   1. Upgrade transformers: pip install --upgrade 'transformers>=4.50.0'\n"
                                f"   2. Qwen3-VL may require a different model class (not Qwen2VLForConditionalGeneration)\n"
                                f"   3. Use Qwen2-VL instead: Qwen/Qwen2-VL-2B-Instruct\n"
                                f"   4. Check Qwen3-VL documentation for correct model class\n\n"
                                f"   Note: Qwen3-VL models may not be fully compatible with this codebase yet."
                            ) from e2
                    else:
                        raise
            else:
                raise
        except AssertionError as e:
            # Handle bitsandbytes AssertionError (common with newer VL models and 4-bit quantization)
            if load_in_4bit and (is_qwen25 or is_qwen3 or is_llama4 or is_deepseek_vl2 or is_internvl):
                if is_qwen25:
                    model_version = "Qwen2.5-VL"
                elif is_qwen3:
                    model_version = "Qwen3-VL"
                elif is_llama4:
                    model_version = "Llama4-VL"
                elif is_deepseek_vl2:
                    model_version = "DeepSeek-VL2"
                elif is_internvl:
                    model_version = "InternVL"
                else:
                    model_version = "model"
                error_msg = (
                    f"❌ {model_version} models have compatibility issues with 4-bit quantization!\n"
                    f"   The error: {str(e) if str(e) else 'AssertionError in bitsandbytes'}\n\n"
                    f"💡 Solutions:\n"
                    f"   1. Use 8-bit quantization instead: --load-in-8bit\n"
                    f"   2. Run without quantization (may require more GPU memory)\n"
                    f"   3. Use Qwen2-VL models which support 4-bit: Qwen/Qwen2-VL-2B-Instruct"
                )
                raise RuntimeError(error_msg) from e
            else:
                raise
        except AttributeError as e:
            # Handle 'Parameter' object has no attribute 'CB' error
            # This occurs when bitsandbytes tries to quantize incompatible layers (e.g., vision encoder)
            if ("'Parameter' object has no attribute 'CB'" in str(e) or 
                "has no attribute 'CB'" in str(e)) and (is_qwen25 or is_qwen3 or is_llama4 or is_deepseek_vl2 or is_internvl):
                if is_qwen25:
                    model_version = "Qwen2.5-VL"
                    script_name = "evaluate_sqa_qwen25.py"
                elif is_qwen3:
                    model_version = "Qwen3-VL"
                    script_name = "evaluate_sqa_qwen3.py"
                elif is_llama4:
                    model_version = "Llama4-VL"
                    script_name = "evaluate_sqa_llama4.py"
                elif is_deepseek_vl2:
                    model_version = "DeepSeek-VL2"
                    script_name = "evaluate_sqa_deepseek_vl2.py"
                elif is_internvl:
                    model_version = "InternVL"
                    script_name = "evaluate_sqa.py"
                else:
                    model_version = "model"
                    script_name = "evaluate_sqa.py"
                error_msg = (
                    f"❌ {model_version} quantization error: Vision encoder incompatibility!\n"
                    f"   Error: {str(e)}\n\n"
                    f"💡 This error occurs because bitsandbytes cannot quantize the vision encoder.\n"
                    f"   Solutions:\n"
                    f"   1. Run WITHOUT quantization: Use --no-quantization flag\n"
                    f"   2. The vision encoder will use full precision (requires more GPU memory)\n"
                    f"   3. Alternatively, try a different quantization approach\n\n"
                    f"   Example: python {script_name} --no-quantization"
                )
                raise RuntimeError(error_msg) from e
            else:
                raise
        except RuntimeError as e:
            # Handle ignore_mismatched_sizes errors for Qwen3-VL
            if "ignore_mismatched_sizes" in str(e).lower() and is_qwen3:
                # If ignore_mismatched_sizes wasn't set, try again with it
                if "ignore_mismatched_sizes" not in model_kwargs or not model_kwargs.get("ignore_mismatched_sizes"):
                    print("   ⚠️  Retrying with ignore_mismatched_sizes=True for Qwen3-VL...")
                    model_kwargs["ignore_mismatched_sizes"] = True
                    try:
                        self.model = self.model_class.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                            device_map=use_device_map,
                            **model_kwargs
                        )
                        print("   ✓ Model loaded successfully with ignore_mismatched_sizes=True")
                    except Exception as e2:
                        raise RuntimeError(
                            f"❌ Failed to load Qwen3-VL model '{model_name}' even with ignore_mismatched_sizes=True.\n"
                            f"   Error: {str(e2)}\n\n"
                            f"💡 Qwen3-VL may require a different model class or transformers version.\n"
                            f"   Try upgrading transformers: pip install --upgrade transformers\n"
                            f"   Or use Qwen2-VL instead: Qwen/Qwen2-VL-2B-Instruct"
                        ) from e2
                else:
                    raise
            else:
                raise
        except (AttributeError, NotImplementedError) as e:
            if "'dict' object has no attribute 'to_dict'" in str(e):
                print("⚠️  Applying workaround for transformers 4.46+ compatibility issue...")
                print("   (This is a known issue with Qwen2-VL and transformers >= 4.46)")
                
                # Workaround: Patch the generation config utility function
                from transformers.generation import configuration_utils
                from transformers import GenerationConfig
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Save original function
                original_from_model_config = configuration_utils.GenerationConfig.from_model_config
                
                def patched_from_model_config(config):
                    """Patched version that handles dict decoder_config"""
                    try:
                        return original_from_model_config(config)
                    except AttributeError as err:
                        if "'dict' object has no attribute 'to_dict'" in str(err):
                            # The issue is decoder_config.to_dict() where decoder_config is a dict
                            # Create a minimal generation config
                            gen_config = GenerationConfig()
                            # Try to copy some basic attributes if available
                            if hasattr(config, 'vocab_size'):
                                gen_config.vocab_size = config.vocab_size
                            return gen_config
                        raise err
                
                # Temporarily replace the method (as a classmethod)
                configuration_utils.GenerationConfig.from_model_config = classmethod(patched_from_model_config)
                
                try:
                    # Try loading again with the patch
                    self.model = self.model_class.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                        device_map="auto" if self.device == "cuda" else None,
                        **model_kwargs
                    )
                finally:
                    # Restore original method
                    configuration_utils.GenerationConfig.from_model_config = original_from_model_config
                    # Restore original _init_weights if we patched it
                    if is_qwen25:
                        try:
                            qwen2_vl_module.Qwen2VLModel._init_weights = original_init_weights
                        except:
                            pass
            elif "normal_kernel_cpu" in str(e) and "Byte" in str(e):
                # Handle Qwen2.5-VL weight initialization error
                print("⚠️  Applying workaround for Qwen2.5-VL weight initialization issue...")
                print("   This is a known issue with Qwen2.5-VL and transformers 4.46.x")
                print("   Patching weight initialization to handle Byte tensors...")
                
                # Monkey patch the _init_weights method in modeling_qwen2_vl
                try:
                    import transformers.models.qwen2_vl.modeling_qwen2_vl as qwen2_vl_module
                    
                    # Save original _init_weights
                    original_init_weights = qwen2_vl_module.Qwen2VLModel._init_weights
                    
                    @staticmethod
                    def patched_init_weights(module):
                        """Patched _init_weights that checks dtype before initializing"""
                        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d)):
                            # Check if weight is Byte dtype - skip initialization
                            if hasattr(module, 'weight') and module.weight is not None:
                                if module.weight.dtype == torch.uint8:  # Byte
                                    return  # Skip initialization for Byte tensors
                        
                        # Call original for other cases
                        try:
                            return original_init_weights(module)
                        except NotImplementedError as err:
                            if "normal_kernel_cpu" in str(err) and "Byte" in str(err):
                                return  # Skip this initialization
                            raise err
                    
                    # Patch the method
                    qwen2_vl_module.Qwen2VLModel._init_weights = patched_init_weights
                    
                    try:
                        # Try loading again with the patch
                        print("   Attempting to load model with patched initialization...")
                        self.model = self.model_class.from_pretrained(
                            model_name,
                            torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                            device_map=use_device_map,
                            **model_kwargs
                        )
                        print("   ✓ Model loaded successfully with patch!")
                    finally:
                        # Restore original method
                        qwen2_vl_module.Qwen2VLModel._init_weights = original_init_weights
                        
                except Exception as e2:
                    raise RuntimeError(
                        f"❌ Failed to load Qwen2.5-VL model '{model_name}'.\n"
                        f"\nThis error typically requires:\n"
                        f"  1. Upgrading transformers: pip install --upgrade 'transformers>=4.47.0'\n"
                        f"  2. Or use Qwen2-VL instead: Qwen/Qwen2-VL-2B-Instruct\n"
                        f"\nOriginal error: {str(e)}\n"
                        f"Patch attempt error: {str(e2)}\n"
                        f"\n💡 Recommendation: Use Qwen/Qwen2-VL-2B-Instruct for a 2B-sized model"
                    ) from e2
            else:
                raise
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Model loaded successfully!")
        
        # Verify device placement for GPU models
        if "cuda" in self.device and torch.cuda.is_available():
            # Check if model is actually on GPU
            # For models with device_map="auto" or specific device, check the device of model parameters
            try:
                # Get the device of the first parameter
                first_param = next(self.model.parameters())
                model_device = first_param.device
                if model_device.type == "cuda":
                    device_id = model_device.index if model_device.index is not None else 0
                    gpu_name = torch.cuda.get_device_name(device_id)
                    print(f"✓ Model is on GPU {device_id}: {gpu_name}")
                    # Show GPU memory usage
                    allocated = torch.cuda.memory_allocated(device_id) / 1e9
                    reserved = torch.cuda.memory_reserved(device_id) / 1e9
                    total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
                    free = total - reserved
                    print(f"  GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free (of {total:.2f} GB)")
                    
                    # Verify if specific GPU was requested
                    if ":" in self.device:
                        requested_id = int(self.device.split(":")[1])
                        if device_id != requested_id:
                            print(f"⚠️  Warning: Requested GPU {requested_id} but model is on GPU {device_id}")
                        else:
                            print(f"✓ Confirmed: Model is on requested GPU {device_id}")
                else:
                    print(f"⚠️  Warning: Model is on {model_device}, expected CUDA!")
            except Exception as e:
                print(f"⚠️  Could not verify model device placement: {e}")
        
        # Initialize LLM for CLIP models if specified
        if self.is_clip and self.llm_model:
            self._initialize_llm_for_clip()
    
    def _initialize_llm_for_clip(self):
        """Initialize LLM for text generation with CLIP models"""
        llm_model_lower = self.llm_model.lower()
        
        # Check if using OpenAI API
        if llm_model_lower in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o']:
            self.use_openai_api = True
            try:
                import openai
                self.openai_client = openai.OpenAI()
                print(f"   ✓ Initialized OpenAI API client for {self.llm_model}")
            except ImportError:
                raise ImportError(
                    "OpenAI package is required for OpenAI API. "
                    "Install with: pip install openai"
                )
            except Exception as e:
                print(f"   ⚠️  Warning: Could not initialize OpenAI API: {e}")
                print("   Falling back to template-based responses")
                self.use_openai_api = False
        else:
            # Local LLM model
            self.use_openai_api = False
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                print(f"   Loading local LLM model: {self.llm_model}")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model,
                    trust_remote_code=True
                )
                self.llm_generator = AutoModelForCausalLM.from_pretrained(
                    self.llm_model,
                    torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.llm_generator.eval()
                print(f"   ✓ Local LLM model loaded successfully")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not load local LLM model: {e}")
                print("   Falling back to template-based responses")
                self.llm_tokenizer = None
                self.llm_generator = None
    
    def _generate_with_llm(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate text using LLM (OpenAI API or local model)"""
        if self.use_openai_api:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"   ⚠️  OpenAI API error: {e}")
                return "I cannot determine the answer from the scene."
        elif self.llm_generator and self.llm_tokenizer:
            try:
                # Format messages for chat template if available
                messages = [{"role": "user", "content": prompt}]
                
                if hasattr(self.llm_tokenizer, 'apply_chat_template'):
                    text = self.llm_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    text = prompt
                
                # Tokenize
                inputs = self.llm_tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True
                ).to(self.llm_generator.device)
                
                # Generate
                with torch.no_grad():
                    generated_ids = self.llm_generator.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                response = self.llm_tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                return response[0].strip()
            except Exception as e:
                print(f"   ⚠️  Local LLM generation error: {e}")
                return "I cannot determine the answer from the scene."
        else:
            # Fallback to template-based response
            return "I cannot determine the answer from the scene."
    
    def answer_question(
        self,
        video_path: Union[str, Path],
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        prompt_template: Optional[str] = None,
        situation: Optional[str] = None,
    ) -> str:
        """
        Answer a question about a video
        
        Args:
            video_path: Path to video file
            question: Question to ask about the video
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            prompt_template: Optional prompt template to override instance template (with {question} placeholder)
            
        Returns:
            Answer string
        """
        # Process video
        images = self.video_processor.process_video(video_path)
        
        # Format question with prompt template if provided (needed for both CLIP and generative models)
        # Method-level template overrides instance-level template
        template_to_use = prompt_template if prompt_template is not None else self.prompt_template
        if template_to_use:
            formatted_question = format_prompt_template(
                template_to_use, question=question, situation=situation
            )
        else:
            formatted_question = question
        
        # CLIP models don't support chat templates or text generation
        # Implement CLIP + simple text generation approach
        if self.is_clip:
            # CLIP computes image-text similarity, not generates text
            # Use CLIP to encode images and question, then generate answer using similarity
            # For SQA, we'll use CLIP to understand the scene and generate a simple answer
            try:
                # Encode images and question text with CLIP
                # Process images
                image_inputs = self.processor(
                    images=images,
                    return_tensors="pt"
                ).to(self.device)
                
                # Process question text
                text_inputs = self.processor(
                    text=[formatted_question],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    image_output = self.model.get_image_features(**image_inputs)
                    text_output = self.model.get_text_features(**text_inputs)
                    
                    # Extract tensor from BaseModelOutputWithPooling
                    # Use pooler_output which is the pooled/cls token representation
                    image_features = image_output.pooler_output  # Shape: [batch, hidden_dim]
                    text_features = text_output.pooler_output   # Shape: [batch, hidden_dim]
                    
                    # Normalize features for cosine similarity
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Average image features across frames if multiple frames
                    if image_features.shape[0] > 1:
                        image_features = image_features.mean(dim=0, keepdim=True)
                    
                    # Compute similarity (cosine similarity)
                    similarity = (image_features @ text_features.T).item()
                
                # Use LLM for text generation if available
                if self.llm_model and (self.use_openai_api or self.llm_generator):
                    # Create a prompt that includes image context based on similarity
                    if similarity > 0.2:
                        # High similarity: question is relevant to images
                        # The formatted_question may already include situation and CoT reasoning
                        # Add image context and ensure we're asking for a direct answer
                        image_context = "The video frames show a 3D indoor scene with various objects and spatial relationships."
                        
                        # Check if the question already contains situation/reasoning (from CoT evaluation)
                        # If so, use it directly; otherwise add image context
                        if "SITUATION" in formatted_question or "REASONING" in formatted_question:
                            # The prompt already includes situation/reasoning, just add image context
                            llm_prompt = f"""{formatted_question}

IMAGE CONTEXT: {image_context}

Based on the above information, provide a direct and concise answer. If the answer cannot be determined, say "I cannot determine the answer from the scene."

Answer:"""
                        else:
                            # Simple question, add full context
                            llm_prompt = f"""You are analyzing a 3D indoor scene from video frames. Based on the visual information:

{image_context}

QUESTION: {formatted_question}

Provide a direct and concise answer based on what you can infer from the scene description and question. If the answer cannot be determined from the available information, say "I cannot determine the answer from the scene."

Answer:"""
                        
                        return self._generate_with_llm(
                            llm_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature
                        )
                    else:
                        # Low similarity: question may not be relevant
                        return "I cannot determine the answer from the scene."
                else:
                    # Fallback to template-based responses if LLM not available
                    if similarity > 0.2:  # Reasonable similarity threshold
                        # Extract key words from question for answer
                        question_lower = formatted_question.lower()
                        if "what" in question_lower:
                            return "The scene shows the described situation."
                        elif "where" in question_lower:
                            return "In the scene."
                        elif "who" in question_lower:
                            return "The person in the scene."
                        elif "how" in question_lower:
                            return "As shown in the scene."
                        elif "why" in question_lower:
                            return "Because of the situation shown."
                        else:
                            return "Yes, based on the scene."
                    else:
                        return "I cannot determine the answer from the scene."
            except Exception as e:
                # Fallback if CLIP processing fails
                print(f"   ⚠️  CLIP processing error: {str(e)[:100]}")
                return f"[CLIP error: {str(e)[:50]}]"
        
        # InternVL models use their own chat API
        if self.is_internvl and hasattr(self, 'internvl_loader') and hasattr(self, 'tokenizer') and hasattr(self, 'image_processor'):
            try:
                # Use InternVL's chat method
                generation_config = dict(
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p if temperature > 0.0 else None
                )
                
                # InternVL chat can handle multiple images, but we'll process them one by one or use the first one
                # For video frames, we can either use the first frame or average them
                # Using the first frame for now (can be enhanced later)
                if len(images) > 0:
                    response = self.internvl_loader.chat(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        image_processor=self.image_processor,
                        images=images[0] if len(images) == 1 else images,  # Pass single image or list
                        question=formatted_question,
                        generation_config=generation_config
                    )
                    return response.strip()
                else:
                    return "[Error: No images extracted from video]"
            except Exception as e:
                print(f"   ⚠️  InternVL chat error: {str(e)[:200]}")
                # Fallback to standard processing if InternVL chat fails
                print("   Falling back to standard processing...")
        
        # Prepare messages (for generative models, not CLIP)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": formatted_question}
                ]
            }
        ]
        
        # Prepare inputs using the processor
        # The processor handles both text and images from messages
        # Check if processor supports apply_chat_template (CLIP doesn't)
        if hasattr(self.processor, 'apply_chat_template'):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: use formatted question directly
            text = formatted_question
        
        # Extract images from messages for processing
        image_list = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # For Llama models, handle vision inputs validation issue
        if self.is_llama4:
            # Llama's generate() validation rejects vision keys, but they're needed
            # Workaround: Temporarily patch _validate_model_kwargs to allow these keys
            import transformers.generation.utils as gen_utils
            original_validate = gen_utils.GenerationMixin._validate_model_kwargs
            
            def patched_validate(self, model_kwargs):
                # Filter out vision keys from validation but keep them for the model
                filtered_kwargs = {k: v for k, v in model_kwargs.items() 
                                 if k not in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']}
                return original_validate(self, filtered_kwargs)
            
            # Apply patch
            gen_utils.GenerationMixin._validate_model_kwargs = patched_validate
            
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,  # Pass all inputs including vision keys
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
            finally:
                # Restore original method
                gen_utils.GenerationMixin._validate_model_kwargs = original_validate
        else:
            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response_text[0].strip()
    
    def answer_question_with_thinking(
        self,
        video_path: Union[str, Path],
        question: str,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        prompt_template: Optional[str] = None,
        think_end_token_id: int = 151668,
        situation: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Video QA with Qwen3-VL-Thinking style outputs: split internal reasoning vs final answer.

        Uses the same multimodal path as ``answer_question``, then splits generated token ids on
        ``think_end_token_id`` (default 151668 = </think> per Qwen3 docs). Falls back to splitting on
        the literal ``</think>`` string if needed.

        Returns:
            (thinking_text, final_answer_text, full_decoded_text_skip_special_true)
        """
        images = self.video_processor.process_video(video_path)
        template_to_use = prompt_template if prompt_template is not None else self.prompt_template
        if template_to_use:
            formatted_question = format_prompt_template(
                template_to_use, question=question, situation=situation
            )
        else:
            formatted_question = question

        if self.is_clip:
            raise NotImplementedError(
                "answer_question_with_thinking is not supported for CLIP models."
            )

        if self.is_internvl and hasattr(self, "internvl_loader") and hasattr(self, "tokenizer") and hasattr(self, "image_processor"):
            try:
                generation_config = dict(
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p if temperature > 0.0 else None,
                )
                if len(images) > 0:
                    response = self.internvl_loader.chat(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        image_processor=self.image_processor,
                        images=images[0] if len(images) == 1 else images,
                        question=formatted_question,
                        generation_config=generation_config,
                    )
                    text = response.strip()
                    return ("", text, text)
                return ("", "[Error: No images extracted from video]", "")
            except Exception as e:
                print(f"   ⚠️  InternVL chat error: {str(e)[:200]}")
                print("   Falling back to standard multimodal processing...")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": formatted_question}],
            }
        ]

        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = formatted_question

        image_list = [
            item["image"] for item in messages[0]["content"] if item["type"] == "image"
        ]

        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        gen_kwargs = {"max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        in_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[0][in_len:].tolist()

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            trimmed = [generated_ids[0][in_len:]]
            full_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            thinking, answer = split_qwen_thinking_string(full_text)
            return thinking, answer or full_text, full_text

        try:
            split_at = len(new_tokens) - new_tokens[::-1].index(think_end_token_id)
        except ValueError:
            split_at = 0

        thinking = tokenizer.decode(new_tokens[:split_at], skip_special_tokens=True).strip()
        answer = tokenizer.decode(new_tokens[split_at:], skip_special_tokens=True).strip()
        full_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if not answer.strip():
            raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
            thinking_s, answer_s = split_qwen_thinking_string(raw)
            if answer_s.strip():
                return thinking_s, answer_s.strip(), full_text or answer_s.strip()

        return thinking, answer.strip() or full_text, full_text

    def answer_question_from_images(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        situation: Optional[str] = None,
    ) -> str:
        """
        Answer a question about a list of images (video frames)
        
        Args:
            images: List of PIL Images
            question: Question to ask about the images
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Answer string
        """
        # Format question with prompt template if provided (needed for both CLIP and generative models)
        if self.prompt_template:
            formatted_question = format_prompt_template(
                self.prompt_template, question=question, situation=situation
            )
        else:
            formatted_question = question
        
        # CLIP models don't support chat templates or text generation
        # Use same CLIP + similarity approach as answer_question
        if self.is_clip:
            try:
                # Encode images and question text with CLIP
                image_inputs = self.processor(
                    images=images,
                    return_tensors="pt"
                ).to(self.device)
                
                text_inputs = self.processor(
                    text=[formatted_question],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    image_output = self.model.get_image_features(**image_inputs)
                    text_output = self.model.get_text_features(**text_inputs)
                    
                    # Extract tensor from BaseModelOutputWithPooling
                    image_features = image_output.pooler_output
                    text_features = text_output.pooler_output
                    
                    # Normalize features for cosine similarity
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Average image features across frames if multiple frames
                    if image_features.shape[0] > 1:
                        image_features = image_features.mean(dim=0, keepdim=True)
                    
                    # Compute similarity (cosine similarity)
                    similarity = (image_features @ text_features.T).item()
                
                # Use LLM for text generation if available
                if self.llm_model and (self.use_openai_api or self.llm_generator):
                    # Create a prompt that includes image context based on similarity
                    if similarity > 0.2:
                        # High similarity: question is relevant to images
                        # The formatted_question may already include situation and CoT reasoning
                        # Add image context and ensure we're asking for a direct answer
                        image_context = f"The images show a 3D indoor scene with various objects and spatial relationships."
                        if len(images) > 1:
                            image_context += f" There are {len(images)} frames/views showing different perspectives of the scene."
                        
                        # Check if the question already contains situation/reasoning (from CoT evaluation)
                        # If so, use it directly; otherwise add image context
                        if "SITUATION" in formatted_question or "REASONING" in formatted_question:
                            # The prompt already includes situation/reasoning, just add image context
                            llm_prompt = f"""{formatted_question}

IMAGE CONTEXT: {image_context}

Based on the above information, provide a direct and concise answer. If the answer cannot be determined, say "I cannot determine the answer from the scene."

Answer:"""
                        else:
                            # Simple question, add full context
                            llm_prompt = f"""You are analyzing a 3D indoor scene. Based on the visual information:

{image_context}

QUESTION: {formatted_question}

Provide a direct and concise answer based on what you can infer from the scene description and question. If the answer cannot be determined from the available information, say "I cannot determine the answer from the scene."

Answer:"""
                        
                        return self._generate_with_llm(
                            llm_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature
                        )
                    else:
                        # Low similarity: question may not be relevant
                        return "I cannot determine the answer from the scene."
                else:
                    # Fallback to template-based responses if LLM not available
                    question_lower = formatted_question.lower()
                    if similarity > 0.2:
                        if "what" in question_lower:
                            return "The scene shows the described situation."
                        elif "where" in question_lower:
                            return "In the scene."
                        elif "who" in question_lower:
                            return "The person in the scene."
                        elif "how" in question_lower:
                            return "As shown in the scene."
                        elif "why" in question_lower:
                            return "Because of the situation shown."
                        else:
                            return "Yes, based on the scene."
                    else:
                        return "I cannot determine the answer from the scene."
            except Exception as e:
                print(f"   ⚠️  CLIP processing error: {str(e)[:100]}")
                return f"[CLIP error: {str(e)[:50]}]"
        
        # InternVL models use their own chat API
        if self.is_internvl and hasattr(self, 'internvl_loader') and hasattr(self, 'tokenizer') and hasattr(self, 'image_processor'):
            try:
                # Use InternVL's chat method
                generation_config = dict(
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=top_p if temperature > 0.0 else None
                )
                
                # InternVL chat can handle multiple images
                response = self.internvl_loader.chat(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    image_processor=self.image_processor,
                    images=images[0] if len(images) == 1 else images,  # Pass single image or list
                    question=formatted_question,
                    generation_config=generation_config
                )
                return response.strip()
            except Exception as e:
                print(f"   ⚠️  InternVL chat error: {str(e)[:200]}")
                # Fallback to standard processing if InternVL chat fails
                print("   Falling back to standard processing...")
        
        # LLaVA models use their own generate method
        if self.is_llava and hasattr(self, 'llava_loader') and hasattr(self, 'processor'):
            try:
                # Use LLaVA's generate method
                response = self.llava_loader.generate(
                    model=self.model,
                    processor=self.processor,
                    images=images[0] if len(images) == 1 else images,  # Pass single image or list
                    question=formatted_question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                return response.strip()
            except Exception as e:
                print(f"   ⚠️  LLaVA generate error: {str(e)[:200]}")
                # Fallback to standard processing if LLaVA generate fails
                print("   Falling back to standard processing...")
        
        # Prepare messages (for generative models, not CLIP)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img} for img in images
                ] + [
                    {"type": "text", "text": formatted_question}
                ]
            }
        ]
        
        # Prepare inputs using the processor
        # The processor handles both text and images from messages
        # Check if processor supports apply_chat_template (CLIP doesn't)
        if hasattr(self.processor, 'apply_chat_template'):
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: use formatted question directly
            text = formatted_question
        
        # Extract images from messages for processing
        image_list = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
        
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # For Llama models, separate vision inputs from text inputs
        # Llama's generate() validates model_kwargs and rejects vision-specific keys
        if self.is_llama4:
            # Extract vision inputs
            vision_inputs = {}
            text_inputs = {}
            for k, v in inputs.items():
                if k in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']:
                    vision_inputs[k] = v
                else:
                    text_inputs[k] = v
            
            # Generate answer - Llama's generate() validation rejects vision keys
            # These keys ARE needed by the model, but validation is too strict
            # Workaround: Temporarily patch _validate_model_kwargs to allow these keys
            import transformers.generation.utils as gen_utils
            original_validate = gen_utils.GenerationMixin._validate_model_kwargs
            
            def patched_validate(self, model_kwargs):
                # Filter out vision keys from validation but keep them for the model
                filtered_kwargs = {k: v for k, v in model_kwargs.items() 
                                 if k not in ['pixel_values', 'aspect_ratio_ids', 'aspect_ratio_mask']}
                return original_validate(self, filtered_kwargs)
            
            # Apply patch
            gen_utils.GenerationMixin._validate_model_kwargs = patched_validate
            
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,  # Pass all inputs including vision keys
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True
                    )
            finally:
                # Restore original method
                gen_utils.GenerationMixin._validate_model_kwargs = original_validate
        else:
            # Generate answer
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response_text[0].strip()

    def answer_question_from_images_with_thinking(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 8192,
        temperature: float = 1.0,
        top_p: float = 0.95,
        think_end_token_id: int = 151668,
        situation: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Like ``answer_question_from_images`` but split Qwen3-VL-Thinking style output into
        (thinking_text, answer_after_think_marker, full_decoded_skip_special_true).

        Implementation lives in ``videolm.multimodal_thinking`` so CoT eval can call the same
        function when an older ``VideoLM`` class is missing this method.
        """
        return answer_from_images_with_thinking(
            self,
            images,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            think_end_token_id=think_end_token_id,
            situation=situation,
        )

    def batch_answer(
        self,
        video_paths: List[Union[str, Path]],
        questions: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Answer multiple questions about multiple videos
        
        Args:
            video_paths: List of video file paths
            questions: List of questions (same length as video_paths)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of answer strings
        """
        if len(video_paths) != len(questions):
            raise ValueError("video_paths and questions must have the same length")
        
        answers = []
        for video_path, question in zip(video_paths, questions):
            answer = self.answer_question(
                video_path=video_path,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            answers.append(answer)
        
        return answers

