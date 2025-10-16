#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene Detection Module for VMSAM

This module provides advanced scene detection capabilities using PySceneDetect
with CPU-optimized performance. It integrates with the existing VMSAM workflow
to improve delay detection accuracy through scene-aware video analysis.

Author: VMSAM Enhancement Team
Compatible with: Python 3.7+
Dependencies: scenedetect, numpy, cv2
"""

import sys
import gc
from threading import Thread, RLock
from decimal import Decimal
from time import strftime, gmtime
import numpy as np
import cv2
import tools
from typing import List, Dict, Tuple, Optional, Union

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector
    from scenedetect.video_manager import VideoManager
    from scenedetect.scene_manager import SceneManager
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    if tools.dev:
        sys.stderr.write("Warning: PySceneDetect not available. Scene detection disabled.\n")

# Global configuration
scene_detection_lock = RLock()
scene_cache = {}

class SceneDetectionResult:
    """Container for scene detection results with timing information."""
    
    def __init__(self, scenes: List[Tuple[float, float]], confidence: float = 1.0):
        """
        Initialize scene detection result.
        
        Args:
            scenes: List of (start_time, end_time) tuples in seconds
            confidence: Overall confidence score [0.0-1.0]
        """
        self.scenes = scenes
        self.confidence = confidence
        self.scene_count = len(scenes)
        
    def get_scenes_in_range(self, start_time: float, end_time: float) -> List[Tuple[float, float]]:
        """
        Get scenes that overlap with the specified time range.
        
        Args:
            start_time: Range start in seconds
            end_time: Range end in seconds
            
        Returns:
            List of overlapping scenes
        """
        overlapping_scenes = []
        for scene_start, scene_end in self.scenes:
            if scene_start < end_time and scene_end > start_time:
                # Clip scene boundaries to the requested range
                clipped_start = max(scene_start, start_time)
                clipped_end = min(scene_end, end_time)
                overlapping_scenes.append((clipped_start, clipped_end))
        return overlapping_scenes
        
    def get_scene_boundaries_near_time(self, target_time: float, tolerance: float = 2.0) -> List[float]:
        """
        Get scene boundaries near a target time.
        
        Args:
            target_time: Target time in seconds
            tolerance: Time tolerance in seconds
            
        Returns:
            List of boundary times within tolerance
        """
        boundaries = []
        for scene_start, scene_end in self.scenes:
            if abs(scene_start - target_time) <= tolerance:
                boundaries.append(scene_start)
            if abs(scene_end - target_time) <= tolerance:
                boundaries.append(scene_end)
        return sorted(boundaries)

class CPUSceneDetector:
    """CPU-optimized scene detector using PySceneDetect."""
    
    def __init__(self, 
                 content_threshold: float = 27.0,
                 adaptive_threshold: float = 3.0,
                 min_scene_len: float = 0.6,
                 use_adaptive: bool = True):
        """
        Initialize the scene detector.
        
        Args:
            content_threshold: Threshold for ContentDetector
            adaptive_threshold: Threshold for AdaptiveDetector  
            min_scene_len: Minimum scene length in seconds
            use_adaptive: Whether to use AdaptiveDetector for better accuracy
        """
        if not SCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect is required but not available")
            
        self.content_threshold = content_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_scene_len = min_scene_len
        self.use_adaptive = use_adaptive
        
    def detect_scenes(self, video_path: str, 
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> SceneDetectionResult:
        """
        Detect scenes in video with optional time constraints.
        
        Args:
            video_path: Path to video file
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            
        Returns:
            SceneDetectionResult with detected scenes
        """
        with scene_detection_lock:
            # Check cache first
            cache_key = f"{video_path}_{start_time}_{end_time}_{self.content_threshold}_{self.use_adaptive}"
            if cache_key in scene_cache:
                if tools.dev:
                    sys.stderr.write(f"\t\tUsing cached scene detection for {video_path}\n")
                return scene_cache[cache_key]
                
            try:
                if tools.dev:
                    sys.stderr.write(f"\t\tDetecting scenes in {video_path}\n")
                
                # Use the high-level API for better performance
                if self.use_adaptive:
                    detector = AdaptiveDetector(adaptive_threshold=self.adaptive_threshold,
                                              min_scene_len=self.min_scene_len)
                else:
                    detector = ContentDetector(threshold=self.content_threshold,
                                             min_scene_len=self.min_scene_len)
                
                # Detect scenes with time constraints if provided
                scene_list = detect(video_path, detector, 
                                  start_time=start_time, 
                                  end_time=end_time,
                                  show_progress=tools.dev)
                
                # Convert scene list to our format
                scenes = []
                for scene in scene_list:
                    scene_start = scene[0].get_seconds()
                    scene_end = scene[1].get_seconds()
                    scenes.append((scene_start, scene_end))
                
                # Calculate confidence based on scene count and consistency
                confidence = self._calculate_confidence(scenes)
                
                result = SceneDetectionResult(scenes, confidence)
                
                # Cache result
                scene_cache[cache_key] = result
                
                if tools.dev:
                    sys.stderr.write(f"\t\tDetected {len(scenes)} scenes with confidence {confidence:.2f}\n")
                
                return result
                
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tScene detection failed for {video_path}: {e}\n")
                # Return empty result on failure
                return SceneDetectionResult([], 0.0)
                
    def _calculate_confidence(self, scenes: List[Tuple[float, float]]) -> float:
        """
        Calculate confidence score based on scene characteristics.
        
        Args:
            scenes: List of detected scenes
            
        Returns:
            Confidence score [0.0-1.0]
        """
        if not scenes:
            return 0.0
            
        # Base confidence on number of scenes and their length distribution
        scene_lengths = [end - start for start, end in scenes]
        
        if len(scene_lengths) < 2:
            return 0.5  # Low confidence for single scene
            
        # Calculate coefficient of variation for scene lengths
        mean_length = np.mean(scene_lengths)
        std_length = np.std(scene_lengths)
        
        if mean_length > 0:
            cv = std_length / mean_length
            # Lower CV indicates more consistent scenes = higher confidence
            confidence = max(0.1, min(1.0, 1.0 - cv / 2.0))
        else:
            confidence = 0.1
            
        return confidence

class SceneAwareDelayDetector:
    """Enhanced delay detector that uses scene information for improved accuracy."""
    
    def __init__(self, scene_detector: CPUSceneDetector):
        """
        Initialize the scene-aware delay detector.
        
        Args:
            scene_detector: Configured scene detector instance
        """
        self.scene_detector = scene_detector
        
    def get_optimal_comparison_points(self, video_path: str, 
                                     video_duration: float,
                                     num_points: int = 3) -> List[Tuple[float, float]]:
        """
        Get optimal time points for video comparison based on scene boundaries.
        
        Args:
            video_path: Path to video file
            video_duration: Total video duration in seconds
            num_points: Number of comparison points to generate
            
        Returns:
            List of (start_time, end_time) tuples for comparison
        """
        # Detect scenes in the video
        scenes = self.scene_detector.detect_scenes(video_path)
        
        if not scenes.scenes or len(scenes.scenes) < 2:
            # Fallback to traditional method if scene detection fails
            return self._get_fallback_comparison_points(video_duration, num_points)
            
        # Select scenes distributed across the video
        selected_scenes = self._select_distributed_scenes(scenes.scenes, num_points)
        
        # Generate comparison points within selected scenes
        comparison_points = []
        for scene_start, scene_end in selected_scenes:
            # Use middle portion of scene for stable comparison
            scene_duration = scene_end - scene_start
            if scene_duration > 2.0:  # Only use scenes longer than 2 seconds
                point_start = scene_start + scene_duration * 0.3
                point_end = scene_start + scene_duration * 0.7
                comparison_points.append((point_start, point_end))
                
        # Ensure we have at least the requested number of points
        while len(comparison_points) < num_points and len(comparison_points) > 0:
            # Duplicate the most stable point
            comparison_points.append(comparison_points[0])
            
        return comparison_points[:num_points]
        
    def _select_distributed_scenes(self, scenes: List[Tuple[float, float]], 
                                  count: int) -> List[Tuple[float, float]]:
        """
        Select scenes distributed evenly across the video timeline.
        
        Args:
            scenes: All detected scenes
            count: Number of scenes to select
            
        Returns:
            Selected scenes distributed across timeline
        """
        if len(scenes) <= count:
            return scenes
            
        # Sort scenes by start time
        sorted_scenes = sorted(scenes, key=lambda x: x[0])
        
        # Select scenes at regular intervals
        selected = []
        interval = len(sorted_scenes) / count
        
        for i in range(count):
            index = min(int(i * interval), len(sorted_scenes) - 1)
            selected.append(sorted_scenes[index])
            
        return selected
        
    def _get_fallback_comparison_points(self, duration: float, 
                                      num_points: int) -> List[Tuple[float, float]]:
        """
        Generate fallback comparison points when scene detection fails.
        
        Args:
            duration: Video duration in seconds
            num_points: Number of points to generate
            
        Returns:
            Evenly distributed comparison points
        """
        points = []
        segment_duration = min(5.0, duration / (num_points + 1))  # 5-second segments
        
        for i in range(num_points):
            start_time = (i + 1) * (duration / (num_points + 1))
            start_time = max(5.0, min(start_time, duration - segment_duration - 5.0))
            end_time = start_time + segment_duration
            points.append((start_time, end_time))
            
        return points
        
    def validate_delay_with_scenes(self, video1_path: str, video2_path: str,
                                  detected_delay: float,
                                  tolerance: float = 1.0) -> Tuple[bool, float]:
        """
        Validate detected delay using scene boundary alignment.
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video  
            detected_delay: Detected delay in seconds
            tolerance: Validation tolerance in seconds
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        try:
            # Detect scenes in both videos
            scenes1 = self.scene_detector.detect_scenes(video1_path)
            scenes2 = self.scene_detector.detect_scenes(video2_path)
            
            if not scenes1.scenes or not scenes2.scenes:
                # Cannot validate without scene information
                return True, 0.5
                
            # Check if scene boundaries align when delay is applied
            aligned_boundaries = 0
            total_boundaries = 0
            
            for scene1_start, scene1_end in scenes1.scenes[:10]:  # Check first 10 scenes
                # Apply delay to video1 times
                adjusted_start = scene1_start + detected_delay
                adjusted_end = scene1_end + detected_delay
                
                # Check if there are corresponding boundaries in video2
                for scene2_start, scene2_end in scenes2.scenes:
                    if (abs(adjusted_start - scene2_start) <= tolerance or
                        abs(adjusted_end - scene2_end) <= tolerance):
                        aligned_boundaries += 1
                        break
                        
                total_boundaries += 1
                
            if total_boundaries == 0:
                return True, 0.5
                
            alignment_ratio = aligned_boundaries / total_boundaries
            is_valid = alignment_ratio >= 0.3  # At least 30% alignment required
            
            if tools.dev:
                sys.stderr.write(f"\t\tScene validation: {aligned_boundaries}/{total_boundaries} boundaries aligned ({alignment_ratio:.2f})\n")
                
            return is_valid, alignment_ratio
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"\t\tScene validation failed: {e}\n")
            return True, 0.5  # Default to valid with low confidence

def clear_scene_cache():
    """Clear the scene detection cache to free memory."""
    global scene_cache
    with scene_detection_lock:
        scene_cache.clear()
        gc.collect()
        
def get_scene_detector(adaptive: bool = True) -> CPUSceneDetector:
    """
    Factory function to create a configured scene detector.
    
    Args:
        adaptive: Whether to use adaptive detection for better accuracy
        
    Returns:
        Configured CPUSceneDetector instance
    """
    return CPUSceneDetector(
        content_threshold=27.0,
        adaptive_threshold=3.0,
        min_scene_len=0.6,
        use_adaptive=adaptive
    )
