"""
VideoLM: Qwen-based Video Language Model for Question Answering
"""

from .model import VideoLM
from .video_processor import VideoProcessor

__version__ = "0.1.0"
__all__ = ["VideoLM", "VideoProcessor"]

