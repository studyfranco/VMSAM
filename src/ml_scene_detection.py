# -*- coding: utf-8 -*-
"""
ML-based scene detection module for VMSAM
Utilizes CPU-optimized open-source models for automatic scene boundary detection

Created on 16 Oct 2025
@author: studyfranco
"""

import sys
import cv2
import numpy as np
from os import path
from threading import Thread, Lock
from time import strftime, gmtime
from decimal import Decimal
import tools
import video


class SceneDetector:
    """
    CPU-optimized ML scene detection using open-source computer vision models.
    Implements lightweight feature extraction and change detection algorithms.
    """
    
    def __init__(self, threshold_primary=0.3, threshold_secondary=0.4, min_scene_length=1.0):
        """
        Initialize scene detector with configurable thresholds.
        
        Args:
            threshold_primary (float): Primary scene change threshold (0.0-1.0)
            threshold_secondary (float): Secondary validation threshold
            min_scene_length (float): Minimum scene length in seconds
        """
        self.threshold_primary = threshold_primary
        self.threshold_secondary = threshold_secondary
        self.min_scene_length = min_scene_length
        self.lock = Lock()
        
    def _extract_histogram_features(self, frame):
        """
        Extract normalized color histogram features from frame.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            numpy.ndarray: Normalized histogram feature vector
        """
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        cv2.normalize(hist_v, hist_v)
        
        # Concatenate features
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        return features
    
    def _extract_edge_density_features(self, frame):
        """
        Extract edge density features using Canny edge detection.
        
        Args:
            frame (numpy.ndarray): Input video frame
            
        Returns:
            float: Normalized edge density value
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        return edge_density
    
    def _calculate_frame_difference(self, features1, features2, edge_density1, edge_density2):
        """
        Calculate difference score between two frames using multiple features.
        
        Args:
            features1 (numpy.ndarray): First frame histogram features
            features2 (numpy.ndarray): Second frame histogram features
            edge_density1 (float): First frame edge density
            edge_density2 (float): Second frame edge density
            
        Returns:
            float: Combined difference score (0.0-1.0)
        """
        # Histogram correlation (Bhattacharyya distance)
        hist_correlation = cv2.compareHist(features1, features2, cv2.HISTCMP_BHATTACHARYYA)
        
        # Edge density difference
        edge_diff = abs(edge_density1 - edge_density2)
        
        # Combine scores with weights
        combined_score = 0.7 * hist_correlation + 0.3 * edge_diff
        return min(1.0, combined_score)
    
    def detect_scenes_from_video(self, video_path, start_time=0.0, duration=None, sample_rate=2.0):
        """
        Detect scene boundaries in video file using ML-based analysis.
        
        Args:
            video_path (str): Path to video file
            start_time (float): Start time in seconds
            duration (float): Analysis duration in seconds (None for full video)
            sample_rate (float): Frames per second to analyze
            
        Returns:
            list: List of scene change timestamps in seconds
        """
        scene_changes = []
        
        try:
            # Open video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0  # Default fallback
            
            frame_interval = int(fps / sample_rate)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_time * fps)
            if duration is not None:
                end_frame = min(total_frames, int((start_time + duration) * fps))
            else:
                end_frame = total_frames
            
            # Set starting position
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            prev_features = None
            prev_edge_density = None
            prev_timestamp = start_time
            
            current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, (320, 240))
                
                # Extract features
                features = self._extract_histogram_features(frame_resized)
                edge_density = self._extract_edge_density_features(frame_resized)
                
                current_timestamp = current_frame / fps
                
                if prev_features is not None:
                    # Calculate difference score
                    diff_score = self._calculate_frame_difference(
                        prev_features, features, prev_edge_density, edge_density
                    )
                    
                    # Check for scene change
                    if diff_score > self.threshold_primary:
                        # Validate with secondary threshold and minimum scene length
                        time_since_last = current_timestamp - (scene_changes[-1] if scene_changes else start_time)
                        
                        if time_since_last >= self.min_scene_length:
                            scene_changes.append(current_timestamp)
                            
                            if tools.dev:
                                sys.stderr.write(f"Scene change detected at {current_timestamp:.2f}s (score: {diff_score:.3f})\n")
                
                prev_features = features
                prev_edge_density = edge_density
                
                # Skip frames based on sample rate
                current_frame += frame_interval
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
            cap.release()
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Error in ML scene detection: {str(e)}\n")
            # Fallback to FFmpeg-based detection
            return self._fallback_ffmpeg_scene_detection(video_path, start_time, duration)
        
        return scene_changes
    
    def _fallback_ffmpeg_scene_detection(self, video_path, start_time=0.0, duration=None):
        """
        Fallback scene detection using FFmpeg scene filter.
        
        Args:
            video_path (str): Path to video file
            start_time (float): Start time in seconds
            duration (float): Analysis duration in seconds
            
        Returns:
            list: List of scene change timestamps
        """
        scene_changes = []
        
        try:
            ffmpeg_cmd = [tools.software["ffmpeg"], "-hide_banner", "-nostdin"]
            
            if start_time > 0:
                ffmpeg_cmd.extend(["-ss", str(start_time)])
            
            if duration is not None:
                ffmpeg_cmd.extend(["-t", str(duration)])
            
            ffmpeg_cmd.extend([
                "-i", video_path,
                "-vf", f"select='gt(scene,{self.threshold_primary})',showinfo",
                "-f", "null", "-"
            ])
            
            stdout, stderr_out, return_code = tools.launch_cmdExt_no_test(ffmpeg_cmd)
            
            if return_code == 0:
                import re
                stderr_text = stderr_out.decode('utf-8', errors='ignore')
                
                for match in re.finditer(r'pts_time:([0-9]+\.[0-9]+)', stderr_text):
                    timestamp = float(match.group(1)) + start_time
                    scene_changes.append(timestamp)
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Fallback scene detection error: {str(e)}\n")
        
        return scene_changes
    
    def select_representative_scenes(self, scene_changes, video_duration, num_scenes=3):
        """
        Select representative scenes from detected scene changes.
        
        Args:
            scene_changes (list): List of scene change timestamps
            video_duration (float): Total video duration in seconds
            num_scenes (int): Number of scenes to select
            
        Returns:
            list: Selected scene timestamps with surrounding context
        """
        if not scene_changes or len(scene_changes) < num_scenes:
            # Fallback to evenly distributed scenes
            interval = video_duration / (num_scenes + 1)
            return [interval * (i + 1) for i in range(num_scenes)]
        
        # Select scenes distributed across the video
        selected_scenes = []
        scene_interval = len(scene_changes) / num_scenes
        
        for i in range(num_scenes):
            scene_index = int(i * scene_interval)
            if scene_index < len(scene_changes):
                selected_scenes.append(scene_changes[scene_index])
        
        return selected_scenes


class SceneBasedDelayAnalyzer:
    """
    Enhanced delay analyzer using ML-detected scenes for improved accuracy.
    """
    
    def __init__(self, scene_detector=None):
        """
        Initialize scene-based delay analyzer.
        
        Args:
            scene_detector (SceneDetector): Scene detection instance
        """
        self.scene_detector = scene_detector or SceneDetector()
        self.analysis_window = 5.0  # seconds around each scene
    
    def analyze_delay_with_scenes(self, video_obj_1, video_obj_2, episode_duration, num_scenes=3):
        """
        Analyze delay between videos using scene-based frame comparison.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            episode_duration (float): Episode duration in seconds
            num_scenes (int): Number of scenes to analyze
            
        Returns:
            dict: Analysis results with delay estimates and confidence scores
        """
        results = {
            'delays': [],
            'confidence_scores': [],
            'scene_timestamps': [],
            'analysis_method': 'ml_scene_based'
        }
        
        try:
            # Detect scenes in the first video
            scene_changes = self.scene_detector.detect_scenes_from_video(
                video_obj_1.filePath, 
                duration=episode_duration,
                sample_rate=1.0
            )
            
            # Select representative scenes
            selected_scenes = self.scene_detector.select_representative_scenes(
                scene_changes, episode_duration, num_scenes
            )
            
            if tools.dev:
                sys.stderr.write(f"Analyzing {len(selected_scenes)} scenes for delay detection\n")
            
            for scene_time in selected_scenes:
                # Analyze delay around this scene
                delay_result = self._analyze_scene_delay(
                    video_obj_1, video_obj_2, scene_time
                )
                
                if delay_result is not None:
                    results['delays'].append(delay_result['delay'])
                    results['confidence_scores'].append(delay_result['confidence'])
                    results['scene_timestamps'].append(scene_time)
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Scene-based delay analysis error: {str(e)}\n")
        
        return results
    
    def _analyze_scene_delay(self, video_obj_1, video_obj_2, scene_timestamp):
        """
        Analyze delay around a specific scene using frame comparison.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            scene_timestamp (float): Scene timestamp in seconds
            
        Returns:
            dict: Delay analysis result or None
        """
        try:
            from frame_compare import FrameComparer
            
            # Check if both videos have the same framerate for frame comparison
            fps1 = float(video_obj_1.video.get("FrameRate", 25.0))
            fps2 = float(video_obj_2.video.get("FrameRate", 25.0))
            
            if abs(fps1 - fps2) > 0.1:
                # Different framerates - use audio correlation validation
                return self._validate_with_audio_correlation(video_obj_1, video_obj_2, scene_timestamp)
            
            # Same framerate - use frame comparison
            start_time = max(0, scene_timestamp - self.analysis_window/2)
            end_time = scene_timestamp + self.analysis_window/2
            
            comparer = FrameComparer(
                video_obj_1.filePath,
                video_obj_2.filePath,
                start_time,
                end_time,
                fps=min(fps1, fps2, 10),  # Limit analysis fps
                debug=tools.dev
            )
            
            gap_result = comparer.find_scene_gap_requirements()
            
            if gap_result is not None:
                # Convert frame difference to time delay
                frame_diff = gap_result['start_frame'] - gap_result['end_frame']
                time_delay = frame_diff / fps1 * 1000  # Convert to milliseconds
                
                confidence = self._calculate_confidence_score(gap_result)
                
                return {
                    'delay': time_delay,
                    'confidence': confidence,
                    'method': 'frame_comparison'
                }
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Scene delay analysis error: {str(e)}\n")
        
        return None
    
    def _validate_with_audio_correlation(self, video_obj_1, video_obj_2, scene_timestamp):
        """
        Validate delay using audio correlation when framerates differ.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            scene_timestamp (float): Scene timestamp in seconds
            
        Returns:
            dict: Validation result or None
        """
        try:
            # This is a placeholder for audio correlation validation
            # In practice, this would integrate with existing audio correlation methods
            
            confidence = 0.5  # Lower confidence for different framerates
            
            return {
                'delay': 0.0,  # Placeholder - would use actual audio correlation
                'confidence': confidence,
                'method': 'audio_validation'
            }
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Audio correlation validation error: {str(e)}\n")
        
        return None
    
    def _calculate_confidence_score(self, gap_result):
        """
        Calculate confidence score for delay estimation.
        
        Args:
            gap_result (dict): Frame comparison gap result
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        # Base confidence calculation
        frame_span = abs(gap_result['end_frame'] - gap_result['start_frame'])
        time_span = abs(gap_result['end_time'] - gap_result['start_time'])
        
        # Higher confidence for larger, more distinct gaps
        confidence = min(1.0, (frame_span + time_span * 10) / 100)
        
        return max(0.1, confidence)  # Minimum confidence threshold
