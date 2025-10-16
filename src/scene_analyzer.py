'''
Created on 16 Oct 2025

@author: studyfranco

ML-based scene detection and frame comparison module for VMSAM.
Provides advanced scene boundary detection using PySceneDetect and 
frame-level comparison for precise delay adjustment.

This software needs PySceneDetect, OpenCV
'''

import os
import sys
import traceback
from threading import Thread, RLock
from statistics import mean, median
from decimal import Decimal
import numpy as np
import cv2
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import tools

scene_detection_lock = RLock()

class SceneAnalyzer:
    """
    Analyzes video scenes using PySceneDetect and compares frames between videos.

    This class provides methods to detect scene cuts in a video and to find the best
    frame-based delay between two videos by comparing frames around scene changes.
    The implementation follows VMSAM's coding style with threading support and 
    comprehensive error handling.
    """

    def __init__(self, video_path, frame_rate=None):
        """
        Initialize the SceneAnalyzer.

        Args:
            video_path (str): Path to the video file
            frame_rate (float, optional): Video frame rate for calculations
        """
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.scenes_cache = None
        self.video_handle = None
        
    def get_video_handle(self):
        """
        Get or create video handle for PySceneDetect.
        
        Returns:
            VideoStream: PySceneDetect video handle
        """
        if self.video_handle is None:
            try:
                self.video_handle = open_video(self.video_path)
            except Exception as e:
                sys.stderr.write(f"Error opening video {self.video_path}: {e}\n")
                return None
        return self.video_handle

    def detect_scenes_with_threshold(self, threshold=30.0, min_scene_len=15):
        """
        Detect scene cuts in the video using content-based detection.

        Args:
            threshold (float): Sensitivity threshold for scene detection (lower = more sensitive)
            min_scene_len (int): Minimum scene length in frames
            
        Returns:
            list: List of scene tuples (start_frame, end_frame) or empty list if detection fails
        """
        try:
            with scene_detection_lock:
                video = self.get_video_handle()
                if video is None:
                    return []
                    
                scene_manager = SceneManager()
                scene_manager.add_detector(
                    ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
                )
                
                scene_manager.detect_scenes(video=video)
                scene_list = scene_manager.get_scene_list()
                
                # Convert to frame numbers for easier handling
                scenes = []
                for scene in scene_list:
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames()
                    scenes.append((start_frame, end_frame))
                    
                if tools.dev:
                    sys.stderr.write(f"\t\tDetected {len(scenes)} scenes in {self.video_path}\n")
                    
                self.scenes_cache = scenes
                return scenes
                
        except Exception as e:
            sys.stderr.write(f"Scene detection failed for {self.video_path}: {e}\n")
            if tools.dev:
                traceback.print_exc()
            return []

    def get_representative_scenes(self, num_scenes=3):
        """
        Get representative scenes from the middle portion of the video.
        
        Args:
            num_scenes (int): Number of scenes to return
            
        Returns:
            list: List of selected scene tuples
        """
        if self.scenes_cache is None:
            self.detect_scenes_with_threshold()
            
        if not self.scenes_cache or len(self.scenes_cache) < num_scenes:
            return self.scenes_cache or []
            
        # Take scenes from the middle portion for better representativity
        total_scenes = len(self.scenes_cache)
        start_idx = max(0, total_scenes // 4)
        end_idx = min(total_scenes, start_idx + num_scenes)
        
        selected = self.scenes_cache[start_idx:end_idx]
        if tools.dev:
            sys.stderr.write(f"\t\tSelected {len(selected)} representative scenes\n")
            
        return selected

class FrameComparisonAnalyzer:
    """
    Handles frame-by-frame comparison for precise delay calculation.
    
    This class provides methods to compare frames around scene boundaries
    to find the optimal delay between two videos.
    """
    
    @staticmethod
    def extract_frames_around_scene(video_path, scene_frame, frame_radius=5, resize_to=(64, 64)):
        """
        Extract frames around a scene boundary for comparison.
        
        Args:
            video_path (str): Path to the video file
            scene_frame (int): Frame number of the scene boundary
            frame_radius (int): Number of frames before and after to extract
            resize_to (tuple): Target size for frame resizing (width, height)
            
        Returns:
            list: List of grayscale frame arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            sys.stderr.write(f"Cannot open video {video_path}\n")
            return frames
            
        try:
            for i in range(scene_frame - frame_radius, scene_frame + frame_radius + 1):
                if i < 0:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Convert to grayscale and resize for faster comparison
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if resize_to:
                        gray_frame = cv2.resize(gray_frame, resize_to)
                    frames.append(gray_frame)
                    
        except Exception as e:
            sys.stderr.write(f"Error extracting frames from {video_path}: {e}\n")
        finally:
            cap.release()
            
        return frames
    
    @staticmethod
    def calculate_frame_similarity(frame1, frame2, method='mse'):
        """
        Calculate similarity between two frames.
        
        Args:
            frame1 (np.array): First frame
            frame2 (np.array): Second frame  
            method (str): Similarity calculation method ('mse', 'sad', 'ssim')
            
        Returns:
            float: Similarity score (lower is more similar for mse/sad)
        """
        if frame1.shape != frame2.shape:
            return float('inf')
            
        if method == 'mse':
            return np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
        elif method == 'sad':
            return np.sum(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
        else:
            # Default to MSE
            return np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
    
    @staticmethod
    def find_best_frame_offset_for_scene(ref_video_path, comp_video_path, scene_frame, 
                                       search_window=60, frame_radius=3):
        """
        Find the best frame offset by comparing frames around a scene boundary.
        
        Args:
            ref_video_path (str): Reference video path
            comp_video_path (str): Comparison video path
            scene_frame (int): Scene boundary frame in reference video
            search_window (int): Search window in frames (±search_window)
            frame_radius (int): Number of frames to compare around the scene
            
        Returns:
            int or None: Best frame offset, None if no good match found
        """
        try:
            # Extract reference frames
            ref_frames = FrameComparisonAnalyzer.extract_frames_around_scene(
                ref_video_path, scene_frame, frame_radius
            )
            
            if len(ref_frames) < frame_radius:
                return None
                
            best_offset = None
            min_total_error = float('inf')
            
            # Search for best matching offset
            for offset in range(-search_window, search_window + 1):
                comp_scene_frame = scene_frame + offset
                
                comp_frames = FrameComparisonAnalyzer.extract_frames_around_scene(
                    comp_video_path, comp_scene_frame, frame_radius
                )
                
                if len(comp_frames) != len(ref_frames):
                    continue
                    
                # Calculate total error for this offset
                total_error = 0
                for ref_frame, comp_frame in zip(ref_frames, comp_frames):
                    error = FrameComparisonAnalyzer.calculate_frame_similarity(ref_frame, comp_frame)
                    total_error += error
                    
                if total_error < min_total_error:
                    min_total_error = total_error
                    best_offset = offset
                    
            if tools.dev and best_offset is not None:
                sys.stderr.write(f"\t\tBest frame offset: {best_offset} (error: {min_total_error:.2f})\n")
                
            return best_offset
            
        except Exception as e:
            sys.stderr.write(f"Frame offset calculation failed: {e}\n")
            if tools.dev:
                traceback.print_exc()
            return None

class DelayOptimizer:
    """
    Optimizes delay calculation using scene detection and frame comparison.
    
    This class combines scene detection with frame-level analysis to provide
    more accurate delay calculations between videos.
    """
    
    def __init__(self, ref_video_obj, comp_video_obj):
        """
        Initialize the DelayOptimizer.
        
        Args:
            ref_video_obj: Reference video object
            comp_video_obj: Comparison video object
        """
        self.ref_video_obj = ref_video_obj
        self.comp_video_obj = comp_video_obj
        
    def calculate_enhanced_delay(self, initial_delay_ms, max_scenes=3):
        """
        Calculate enhanced delay using scene detection and frame comparison.
        
        Args:
            initial_delay_ms (float): Initial delay from audio correlation
            max_scenes (int): Maximum number of scenes to analyze
            
        Returns:
            float: Optimized delay in milliseconds
        """
        try:
            # Check if frame rates are compatible
            ref_fps = float(self.ref_video_obj.video.get("FrameRate", 0))
            comp_fps = float(self.comp_video_obj.video.get("FrameRate", 0))
            
            if abs(ref_fps - comp_fps) > 0.1:
                if tools.dev:
                    sys.stderr.write(f"\t\tFrame rates differ too much ({ref_fps} vs {comp_fps}), skipping frame analysis\n")
                return initial_delay_ms
                
            # Perform scene detection on reference video
            analyzer = SceneAnalyzer(self.ref_video_obj.filePath, ref_fps)
            representative_scenes = analyzer.get_representative_scenes(max_scenes)
            
            if not representative_scenes:
                if tools.dev:
                    sys.stderr.write("\t\tNo scenes detected, using initial delay\n")
                return initial_delay_ms
                
            # Calculate frame offsets for each scene
            frame_offsets = []
            search_window = int(ref_fps * 2)  # 2 seconds search window
            
            for scene_start, scene_end in representative_scenes:
                # Use scene start frame for analysis
                offset = FrameComparisonAnalyzer.find_best_frame_offset_for_scene(
                    self.ref_video_obj.filePath,
                    self.comp_video_obj.filePath, 
                    scene_start,
                    search_window=search_window
                )
                
                if offset is not None:
                    frame_offsets.append(offset)
                    
            if not frame_offsets:
                if tools.dev:
                    sys.stderr.write("\t\tNo valid frame offsets found, using initial delay\n")
                return initial_delay_ms
                
            # Use median offset for robustness
            median_offset = median(frame_offsets)
            
            # Convert frame offset to milliseconds
            frame_delay_ms = (median_offset * 1000.0) / ref_fps
            
            # Combine with initial audio-based delay
            final_delay = initial_delay_ms + frame_delay_ms
            
            if tools.dev:
                sys.stderr.write(f"\t\tFrame offsets: {frame_offsets}\n")
                sys.stderr.write(f"\t\tMedian offset: {median_offset} frames = {frame_delay_ms:.2f} ms\n")
                sys.stderr.write(f"\t\tFinal delay: {initial_delay_ms:.2f} + {frame_delay_ms:.2f} = {final_delay:.2f} ms\n")
                
            return final_delay
            
        except Exception as e:
            sys.stderr.write(f"Enhanced delay calculation failed: {e}\n")
            if tools.dev:
                traceback.print_exc()
            return initial_delay_ms
    
    def validate_delay_with_scenes(self, audio_delay_ms, visual_delay_ms, tolerance_ms=100):
        """
        Validate that audio and visual delays are consistent.
        
        Args:
            audio_delay_ms (float): Delay calculated from audio correlation
            visual_delay_ms (float): Delay calculated from visual analysis
            tolerance_ms (float): Acceptable difference in milliseconds
            
        Returns:
            bool: True if delays are consistent within tolerance
        """
        difference = abs(audio_delay_ms - visual_delay_ms)
        is_consistent = difference <= tolerance_ms
        
        if tools.dev:
            sys.stderr.write(f"\t\tDelay validation: audio={audio_delay_ms:.2f}ms, visual={visual_delay_ms:.2f}ms, diff={difference:.2f}ms\n")
            sys.stderr.write(f"\t\tConsistent: {is_consistent} (tolerance: {tolerance_ms}ms)\n")
            
        return is_consistent
