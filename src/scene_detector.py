'''
Created on 16 Oct 2025

@author: studyfranco

ML Scene Detection Module using PySceneDetect for VMSAM
This module provides intelligent scene detection capabilities using machine learning
to improve delay detection and video synchronization accuracy.
'''

import sys
import gc
from threading import Thread
from decimal import Decimal
from time import gmtime, strftime
import numpy as np
from typing import List, Tuple, Optional, Dict

import tools
import video

try:
    from scenedetect import VideoManager, SceneManager, FrameTimecode
    from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
except ImportError:
    sys.stderr.write("PySceneDetect not available. Scene detection will be disabled.\n")
    SceneManager = None

class scene_detector:
    """
    Intelligent scene detection using PySceneDetect for enhanced delay calculation.
    
    This class provides machine learning-based scene detection to identify
    significant scene changes in videos, which can be used to improve
    delay detection accuracy between multiple video sources.
    """
    
    def __init__(self, video_path: str, detection_method: str = "content", 
                 threshold: float = 27.0, min_scene_len: float = 1.0):
        """
        Initialize the scene detector.
        
        Args:
            video_path (str): Path to the video file
            detection_method (str): Detection method ('content', 'adaptive', 'threshold')
            threshold (float): Detection threshold sensitivity
            min_scene_len (float): Minimum scene length in seconds
        """
        self.video_path = video_path
        self.detection_method = detection_method
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.scenes = []
        self.scene_times = []
        
        if SceneManager is None:
            raise Exception("PySceneDetect is not available. Please install it with: pip install scenedetect[opencv]")
    
    def detect_scenes(self, start_time: float = 0.0, end_time: Optional[float] = None,
                     downscale_factor: int = 2) -> List[Tuple[float, float]]:
        """
        Detect scenes in the video using ML-based analysis.
        
        Args:
            start_time (float): Start time for detection in seconds
            end_time (Optional[float]): End time for detection in seconds
            downscale_factor (int): Downscale factor for faster processing
            
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for each scene
        """
        try:
            video_manager = VideoManager([self.video_path])
            scene_manager = SceneManager()
            
            # Choose detection method based on configuration
            if self.detection_method == "content":
                detector = ContentDetector(threshold=self.threshold, 
                                         min_scene_len=FrameTimecode(timecode=self.min_scene_len, fps=25))
            elif self.detection_method == "adaptive":
                detector = AdaptiveDetector(adaptive_threshold=self.threshold,
                                         min_scene_len=FrameTimecode(timecode=self.min_scene_len, fps=25))
            elif self.detection_method == "threshold":
                detector = ThresholdDetector(threshold=self.threshold,
                                          min_scene_len=FrameTimecode(timecode=self.min_scene_len, fps=25))
            else:
                raise ValueError(f"Unknown detection method: {self.detection_method}")
            
            scene_manager.add_detector(detector)
            
            # Improve processing speed by downscaling
            video_manager.set_downscale_factor(downscale_factor)
            
            # Set time range if specified
            if start_time > 0 or end_time is not None:
                start_frame = FrameTimecode(timecode=start_time, fps=25)
                end_frame = FrameTimecode(timecode=end_time, fps=25) if end_time else None
                video_manager.set_duration(start_time=start_frame, end_time=end_frame)
            
            # Start detection
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list and convert to time tuples
            scene_list = scene_manager.get_scene_list()
            self.scenes = scene_list
            
            # Convert FrameTimecode to seconds
            scene_times = []
            for scene in scene_list:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                scene_times.append((start_sec, end_sec))
            
            self.scene_times = scene_times
            
            if tools.dev:
                sys.stderr.write(f"\t\tDetected {len(scene_times)} scenes in {self.video_path}\n")
            
            return scene_times
            
        except Exception as e:
            sys.stderr.write(f"Error during scene detection: {e}\n")
            return []
        finally:
            gc.collect()
    
    def get_scene_changes_in_range(self, start_sec: float, end_sec: float, 
                                  max_scenes: int = 3) -> List[float]:
        """
        Get scene change timestamps within a specific time range.
        
        Args:
            start_sec (float): Start time in seconds
            end_sec (float): End time in seconds
            max_scenes (int): Maximum number of scenes to return
            
        Returns:
            List[float]: List of scene change timestamps
        """
        scene_changes = []
        
        for start_time, end_time in self.scene_times:
            # Scene change occurs at the start of each scene (except the first)
            if start_sec <= start_time <= end_sec:
                scene_changes.append(start_time)
        
        # Remove first scene if it's at the beginning of our range
        if scene_changes and abs(scene_changes[0] - start_sec) < 0.5:
            scene_changes.pop(0)
        
        # Limit to max_scenes and return middle scenes for better accuracy
        if len(scene_changes) > max_scenes:
            # Take scenes from the middle of the range for better stability
            start_idx = (len(scene_changes) - max_scenes) // 2
            scene_changes = scene_changes[start_idx:start_idx + max_scenes]
        
        return scene_changes
    
    def calculate_uncertainty_score(self, scene_changes: List[float], 
                                  time_window: float = 2.0) -> float:
        """
        Calculate uncertainty score based on scene change density.
        
        Args:
            scene_changes (List[float]): List of scene change timestamps
            time_window (float): Time window in seconds to analyze
            
        Returns:
            float: Uncertainty score (0.0 = low uncertainty, 1.0 = high uncertainty)
        """
        if len(scene_changes) <= 1:
            return 0.0  # Low uncertainty with few scene changes
        
        # Calculate scene change density
        total_time = max(scene_changes) - min(scene_changes) if len(scene_changes) > 1 else time_window
        density = len(scene_changes) / total_time if total_time > 0 else 0
        
        # Normalize density to uncertainty score (more changes = higher uncertainty)
        # Assume 1 change per second is maximum uncertainty
        uncertainty = min(density, 1.0)
        
        return uncertainty

class scene_based_delay_detector(Thread):
    """
    Thread-based delay detector using scene detection for improved accuracy.
    
    This class combines scene detection with frame comparison to provide
    more accurate delay detection between video sources.
    """
    
    def __init__(self, video_obj_1, video_obj_2, begin_in_second: float,
                 length_time: float, scene_detector_1: scene_detector,
                 scene_detector_2: scene_detector):
        """
        Initialize the scene-based delay detector.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            begin_in_second (float): Start time for analysis
            length_time (float): Length of analysis window
            scene_detector_1: Scene detector for first video
            scene_detector_2: Scene detector for second video
        """
        Thread.__init__(self)
        self.video_obj_1 = video_obj_1
        self.video_obj_2 = video_obj_2
        self.begin_in_second = begin_in_second
        self.length_time = length_time
        self.scene_detector_1 = scene_detector_1
        self.scene_detector_2 = scene_detector_2
        self.detected_delay = None
        self.uncertainty_score = 1.0  # High uncertainty by default
        
    def run(self):
        """
        Run the scene-based delay detection analysis.
        """
        try:
            # Detect scenes in both videos
            end_time = self.begin_in_second + self.length_time
            
            scenes_1 = self.scene_detector_1.get_scene_changes_in_range(
                self.begin_in_second, end_time, max_scenes=3)
            scenes_2 = self.scene_detector_2.get_scene_changes_in_range(
                self.begin_in_second, end_time, max_scenes=3)
            
            if not scenes_1 or not scenes_2:
                if tools.dev:
                    sys.stderr.write("\t\tNo matching scenes found for delay detection\n")
                return
            
            # Find best matching scene changes
            delays = []
            for scene_1 in scenes_1:
                for scene_2 in scenes_2:
                    delay = (scene_2 - scene_1) * 1000  # Convert to milliseconds
                    delays.append(delay)
            
            if delays:
                # Use median delay for robustness
                delays.sort()
                median_delay = delays[len(delays) // 2]
                self.detected_delay = median_delay
                
                # Calculate uncertainty based on delay variance
                if len(delays) > 1:
                    delay_variance = np.var(delays)
                    # Normalize variance to uncertainty score
                    self.uncertainty_score = min(delay_variance / 10000.0, 1.0)
                else:
                    self.uncertainty_score = 0.5  # Medium uncertainty for single match
                
                if tools.dev:
                    sys.stderr.write(f"\t\tScene-based delay: {median_delay}ms, uncertainty: {self.uncertainty_score:.3f}\n")
            
        except Exception as e:
            sys.stderr.write(f"Error in scene-based delay detection: {e}\n")
            import traceback
            traceback.print_exc()

def create_scene_detectors(video_obj_1, video_obj_2, episode_duration: float) -> Tuple[Optional[scene_detector], Optional[scene_detector]]:
    """
    Create scene detectors for two video objects.
    
    Args:
        video_obj_1: First video object
        video_obj_2: Second video object
        episode_duration (float): Duration of the episode in seconds
        
    Returns:
        Tuple[Optional[scene_detector], Optional[scene_detector]]: Scene detectors for both videos
    """
    try:
        # Adaptive threshold based on video duration
        # Shorter videos need higher sensitivity
        if episode_duration < 1800:  # Less than 30 minutes
            threshold = 25.0
        elif episode_duration < 3600:  # Less than 1 hour
            threshold = 27.0
        else:
            threshold = 30.0
        
        detector_1 = scene_detector(video_obj_1.filePath, 
                                  detection_method="content",
                                  threshold=threshold,
                                  min_scene_len=2.0)
        
        detector_2 = scene_detector(video_obj_2.filePath,
                                  detection_method="content",
                                  threshold=threshold,
                                  min_scene_len=2.0)
        
        return detector_1, detector_2
        
    except Exception as e:
        sys.stderr.write(f"Failed to create scene detectors: {e}\n")
        return None, None

def detect_scenes_for_episode(video_obj, start_time: float = 0.0, 
                            end_time: Optional[float] = None) -> Optional[scene_detector]:
    """
    Detect scenes for a complete episode.
    
    Args:
        video_obj: Video object to analyze
        start_time (float): Start time in seconds
        end_time (Optional[float]): End time in seconds
        
    Returns:
        Optional[scene_detector]: Scene detector with detected scenes
    """
    try:
        duration = float(video_obj.video.get('Duration', 3600))
        detector = scene_detector(video_obj.filePath, 
                                detection_method="content",
                                threshold=27.0,
                                min_scene_len=1.5)
        
        # Detect scenes for the specified range
        detector.detect_scenes(start_time=start_time, end_time=end_time)
        
        return detector
        
    except Exception as e:
        sys.stderr.write(f"Error detecting scenes for episode: {e}\n")
        return None
