"""
Video processing utilities for VideoLM
"""

import cv2
import numpy as np
from typing import List, Union, Optional
from pathlib import Path
import torch
from PIL import Image


class VideoProcessor:
    """Process videos for VideoLM input"""
    
    def __init__(
        self,
        max_frames: int = 8,
        frame_size: tuple = (448, 448),
        fps: Optional[float] = None
    ):
        """
        Initialize video processor
        
        Args:
            max_frames: Maximum number of frames to extract
            frame_size: Target size for frames (width, height)
            fps: Target FPS for frame extraction (None = uniform sampling)
        """
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.fps = fps
    
    def load_video(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load video and extract frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame arrays
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices to sample
        if self.fps and video_fps > 0:
            # Sample at target FPS
            frame_interval = max(1, int(video_fps / self.fps))
            frame_indices = list(range(0, total_frames, frame_interval))[:self.max_frames]
        else:
            # Uniform sampling
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int).tolist()
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
        
        cap.release()
        
        # Pad frames if needed
        while len(frames) < self.max_frames:
            if frames:
                frames.append(frames[-1])  # Repeat last frame
            else:
                # Create black frame if no frames extracted
                frames.append(np.zeros((*self.frame_size[::-1], 3), dtype=np.uint8))
        
        return frames[:self.max_frames]
    
    def frames_to_images(self, frames: List[np.ndarray]) -> List[Image.Image]:
        """Convert frame arrays to PIL Images"""
        return [Image.fromarray(frame) for frame in frames]
    
    def process_video(self, video_path: Union[str, Path]) -> List[Image.Image]:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of PIL Images ready for model input
        """
        frames = self.load_video(video_path)
        return self.frames_to_images(frames)

