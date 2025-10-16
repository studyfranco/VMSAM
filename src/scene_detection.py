"""Scene detection module using ML models for automatic scene boundary detection.

This module integrates PySceneDetect library for CPU-based scene detection,
optimized for VMSAM video synchronization and delay adjustment workflows.

Author: studyfranco
Created: 2025-01-16
"""

import sys
import os
from typing import List, Tuple, Optional, Dict, Any
from decimal import Decimal
import tempfile
from threading import Thread, RLock
from time import strftime, gmtime
import gc

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    from scenedetect.video_splitter import split_video_ffmpeg
    from scenedetect.scene_manager import SceneManager
    from scenedetect import open_video
    PYSCENEDETECT_AVAILABLE = True
except ImportError:
    PYSCENEDETECT_AVAILABLE = False
    sys.stderr.write("Warning: PySceneDetect not available. Scene detection will be disabled.\n")

import tools
import video


class SceneDetectionError(Exception):
    """Custom exception for scene detection related errors."""
    pass


class SceneDetector:
    """CPU-optimized scene detector for video synchronization.
    
    This class provides ML-based scene detection capabilities using PySceneDetect
    library, optimized for CPU performance and integration with VMSAM delay
    calculation workflows.
    """
    
    def __init__(self, 
                 threshold: float = 27.0, 
                 min_scene_len: int = 15,
                 adaptive_detector: bool = True,
                 max_scenes: int = 10):
        """Initialize the scene detector.
        
        Args:
            threshold: Detection threshold for content-based scene detection
            min_scene_len: Minimum scene length in frames
            adaptive_detector: Use adaptive detector for better camera movement handling
            max_scenes: Maximum number of scenes to detect for delay calculation
        """
        if not PYSCENEDETECT_AVAILABLE:
            raise SceneDetectionError("PySceneDetect library is not available")
            
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.adaptive_detector = adaptive_detector
        self.max_scenes = max_scenes
        self.detection_lock = RLock()
        
    def detect_scenes(self, video_path: str, start_time: float = 0.0, 
                     duration: Optional[float] = None) -> List[Tuple[float, float]]:
        """Detect scenes in video using ML-based detection.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds for detection window
            duration: Duration in seconds for detection window (None for full video)
            
        Returns:
            List of tuples containing (start_time, end_time) for each scene
            
        Raises:
            SceneDetectionError: If scene detection fails
        """
        try:
            with self.detection_lock:
                if tools.dev:
                    sys.stderr.write(f"Starting scene detection for {video_path}\n")
                
                # Choose detector based on configuration
                if self.adaptive_detector:
                    detector = AdaptiveDetector(threshold=self.threshold,
                                              min_scene_len=self.min_scene_len)
                else:
                    detector = ContentDetector(threshold=self.threshold,
                                             min_scene_len=self.min_scene_len)
                
                # Detect scenes
                scene_list = detect(video_path, detector)
                
                # Filter scenes based on time window
                filtered_scenes = []
                for scene in scene_list[:self.max_scenes]:
                    scene_start = scene[0].get_seconds()
                    scene_end = scene[1].get_seconds()
                    
                    # Apply time window filtering
                    if duration is not None:
                        end_time = start_time + duration
                        if scene_start >= start_time and scene_end <= end_time:
                            filtered_scenes.append((scene_start - start_time, 
                                                  scene_end - start_time))
                    else:
                        if scene_start >= start_time:
                            filtered_scenes.append((scene_start - start_time,
                                                  scene_end - start_time))
                
                if tools.dev:
                    sys.stderr.write(f"Detected {len(filtered_scenes)} scenes\n")
                
                return filtered_scenes
                
        except Exception as e:
            raise SceneDetectionError(f"Scene detection failed: {str(e)}")
            
    def get_representative_scenes(self, scenes: List[Tuple[float, float]], 
                                num_scenes: int = 3) -> List[Tuple[float, float]]:
        """Get representative scenes for delay calculation.
        
        Selects scenes distributed across the video duration for better
        delay estimation accuracy.
        
        Args:
            scenes: List of detected scenes
            num_scenes: Number of representative scenes to select
            
        Returns:
            List of selected representative scenes
        """
        if len(scenes) <= num_scenes:
            return scenes
            
        # Distribute scenes evenly across the timeline
        selected_scenes = []
        step = len(scenes) / num_scenes
        
        for i in range(num_scenes):
            idx = int(i * step)
            if idx < len(scenes):
                selected_scenes.append(scenes[idx])
                
        return selected_scenes
        
    def validate_scene_consistency(self, video1_scenes: List[Tuple[float, float]], 
                                 video2_scenes: List[Tuple[float, float]], 
                                 tolerance: float = 2.0) -> bool:
        """Validate that detected scenes are consistent between two videos.
        
        Args:
            video1_scenes: Scenes detected in first video
            video2_scenes: Scenes detected in second video  
            tolerance: Maximum allowed difference in scene timing (seconds)
            
        Returns:
            True if scenes are consistent, False otherwise
        """
        if abs(len(video1_scenes) - len(video2_scenes)) > 1:
            return False
            
        min_scenes = min(len(video1_scenes), len(video2_scenes))
        
        for i in range(min_scenes):
            scene1_duration = video1_scenes[i][1] - video1_scenes[i][0]
            scene2_duration = video2_scenes[i][1] - video2_scenes[i][0]
            
            if abs(scene1_duration - scene2_duration) > tolerance:
                return False
                
        return True


class EnhancedDelayEstimator:
    """Enhanced delay estimator combining scene detection with frame comparison.
    
    This class improves the delay estimation accuracy by using ML-detected scenes
    and enhanced frame comparison for uncertainty quantification.
    """
    
    def __init__(self, scene_detector: SceneDetector):
        """Initialize the enhanced delay estimator.
        
        Args:
            scene_detector: Configured scene detector instance
        """
        self.scene_detector = scene_detector
        
    def estimate_delay_with_scenes(self, video_obj_1, video_obj_2, 
                                 audio_delay: Decimal,
                                 begin_in_second: float,
                                 length_time: float) -> Tuple[Decimal, float]:
        """Estimate delay using scene detection for improved accuracy.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            audio_delay: Initial delay estimate from audio correlation
            begin_in_second: Start time for scene detection window
            length_time: Duration of detection window
            
        Returns:
            Tuple of (adjusted_delay, confidence_score)
        """
        try:
            # Detect scenes in both videos
            scenes1 = self.scene_detector.detect_scenes(
                video_obj_1.filePath, begin_in_second, length_time
            )
            scenes2 = self.scene_detector.detect_scenes(
                video_obj_2.filePath, begin_in_second, length_time
            )
            
            # Validate scene consistency
            if not self.scene_detector.validate_scene_consistency(scenes1, scenes2):
                if tools.dev:
                    sys.stderr.write("Scene inconsistency detected, falling back to audio delay\n")
                return audio_delay, 0.5
            
            # Get representative scenes for comparison
            repr_scenes1 = self.scene_detector.get_representative_scenes(scenes1, 3)
            repr_scenes2 = self.scene_detector.get_representative_scenes(scenes2, 3)
            
            # Perform frame comparison on scene boundaries
            delay_estimates = []
            
            for i, (scene1, scene2) in enumerate(zip(repr_scenes1, repr_scenes2)):
                try:
                    # Import frame comparison module
                    from frame_compare import FrameComparer
                    
                    # Calculate scene boundary comparison window
                    scene1_start = begin_in_second + scene1[0]
                    scene1_end = begin_in_second + scene1[1]
                    scene2_start = begin_in_second + scene2[0] + float(audio_delay) / 1000.0
                    scene2_end = begin_in_second + scene2[1] + float(audio_delay) / 1000.0
                    
                    # Use frame comparison around scene boundaries
                    comparer = FrameComparer(
                        video_obj_1.filePath,
                        video_obj_2.filePath,
                        scene1_start,
                        min(scene1_end, scene1_start + 10.0),  # Limit to 10s windows
                        fps=10,
                        debug=tools.dev
                    )
                    
                    gap_info = comparer.find_scene_gap_requirements()
                    if gap_info:
                        # Convert frame-based delay to milliseconds
                        frame_delay = (gap_info["start_time"] - scene2_start) * 1000.0
                        delay_estimates.append(frame_delay)
                        
                except Exception as e:
                    if tools.dev:
                        sys.stderr.write(f"Frame comparison failed for scene {i}: {e}\n")
                    continue
                    
            # Calculate consensus delay and confidence
            if delay_estimates:
                # Remove outliers and calculate mean
                sorted_estimates = sorted(delay_estimates)
                if len(sorted_estimates) >= 3:
                    # Remove top and bottom estimates
                    consensus_estimates = sorted_estimates[1:-1]
                else:
                    consensus_estimates = sorted_estimates
                    
                if consensus_estimates:
                    scene_delay = sum(consensus_estimates) / len(consensus_estimates)
                    
                    # Calculate confidence based on estimate variance
                    variance = sum((est - scene_delay) ** 2 for est in consensus_estimates)
                    variance /= len(consensus_estimates)
                    confidence = max(0.1, 1.0 - min(1.0, variance / 1000.0))
                    
                    # Combine with audio delay
                    combined_delay = audio_delay + Decimal(scene_delay)
                    
                    if tools.dev:
                        sys.stderr.write(f"Scene-based delay adjustment: {scene_delay:.2f}ms\n")
                        sys.stderr.write(f"Combined delay: {combined_delay}ms, confidence: {confidence:.2f}\n")
                    
                    return combined_delay, confidence
                    
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Scene-based delay estimation failed: {e}\n")
                
        # Fallback to original audio delay
        return audio_delay, 0.3
        
    def check_framerate_compatibility(self, video_obj_1, video_obj_2) -> bool:
        """Check if videos have compatible framerates for frame comparison.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            
        Returns:
            True if framerates are compatible for frame comparison
        """
        try:
            fps1 = float(video_obj_1.video.get("FrameRate", 0))
            fps2 = float(video_obj_2.video.get("FrameRate", 0))
            
            if fps1 <= 0 or fps2 <= 0:
                return False
                
            # Consider framerates compatible if within 10% tolerance
            fps_ratio = min(fps1, fps2) / max(fps1, fps2)
            return fps_ratio >= 0.9
            
        except (ValueError, KeyError):
            return False


def integrate_scene_detection_with_delay_calculation(video_obj_1, video_obj_2, 
                                                   audio_delay: Decimal,
                                                   begin_in_second: float,
                                                   length_time: float) -> Decimal:
    """Integrate scene detection with existing delay calculation workflow.
    
    This function serves as the main integration point for the new ML-based
    scene detection with the existing VMSAM delay calculation system.
    
    Args:
        video_obj_1: First video object from VMSAM
        video_obj_2: Second video object from VMSAM
        audio_delay: Initial delay calculated from audio correlation
        begin_in_second: Start time for analysis window
        length_time: Duration of analysis window
        
    Returns:
        Adjusted delay value incorporating scene detection improvements
    """
    if not PYSCENEDETECT_AVAILABLE:
        if tools.dev:
            sys.stderr.write("PySceneDetect not available, using original delay\n")
        return audio_delay
        
    try:
        # Initialize scene detector with optimized parameters
        scene_detector = SceneDetector(
            threshold=30.0,  # Slightly higher threshold for better precision
            min_scene_len=30,  # Minimum 30 frames per scene
            adaptive_detector=True,  # Better handling of camera movement
            max_scenes=8  # Limit to 8 scenes for performance
        )
        
        # Initialize enhanced delay estimator
        delay_estimator = EnhancedDelayEstimator(scene_detector)
        
        # Check framerate compatibility
        framerate_compatible = delay_estimator.check_framerate_compatibility(
            video_obj_1, video_obj_2
        )
        
        if framerate_compatible:
            # Full scene detection with frame comparison
            adjusted_delay, confidence = delay_estimator.estimate_delay_with_scenes(
                video_obj_1, video_obj_2, audio_delay, begin_in_second, length_time
            )
            
            # Use adjusted delay if confidence is high enough
            if confidence >= 0.7:
                return adjusted_delay
            elif confidence >= 0.5:
                # Weighted average of audio and scene delays
                weight = (confidence - 0.5) / 0.5
                return audio_delay * (1 - weight) + adjusted_delay * weight
                
        else:
            # Framerate not compatible, only validate scene consistency
            scenes1 = scene_detector.detect_scenes(
                video_obj_1.filePath, begin_in_second, length_time
            )
            scenes2 = scene_detector.detect_scenes(
                video_obj_2.filePath, begin_in_second, length_time
            )
            
            if scene_detector.validate_scene_consistency(scenes1, scenes2, tolerance=1.0):
                if tools.dev:
                    sys.stderr.write("Scene structure validated, audio delay confirmed\n")
                return audio_delay
            else:
                if tools.dev:
                    sys.stderr.write("Scene structure inconsistent, potential sync issue\n")
                    
    except Exception as e:
        if tools.dev:
            sys.stderr.write(f"Scene detection integration failed: {e}\n")
            
    # Fallback to original audio delay
    return audio_delay
