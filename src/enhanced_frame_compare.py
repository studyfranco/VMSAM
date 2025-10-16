'''
Created on 16 Oct 2025

@author: studyfranco

Enhanced Frame Comparison Module for VMSAM
This module provides advanced frame comparison capabilities with uncertainty estimation
and improved delay detection accuracy using multiple comparison methods.
'''

import sys
import math
import struct
import numpy as np
from typing import List, Tuple, Optional, Dict
from threading import Thread
from decimal import Decimal
from scipy.fft import dct
from scipy import ndimage
from scipy.stats import pearsonr
import cv2
import tools
import video

class enhanced_frame_comparer:
    """
    Enhanced frame comparison with multiple similarity metrics and uncertainty estimation.
    
    This class provides advanced frame comparison using multiple algorithms including
    perceptual hashing, structural similarity, and histogram comparison to improve
    delay detection accuracy and provide uncertainty estimates.
    """
    
    def __init__(self, ref_path: str, tgt_path: str, start_sec: float, end_sec: float,
                 fps: int = 10, comparison_methods: List[str] = None):
        """
        Initialize the enhanced frame comparer.
        
        Args:
            ref_path (str): Path to reference video
            tgt_path (str): Path to target video
            start_sec (float): Start time in seconds
            end_sec (float): End time in seconds
            fps (int): Frames per second for extraction
            comparison_methods (List[str]): Methods to use for comparison
        """
        self.ref_path = ref_path
        self.tgt_path = tgt_path
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.fps = int(max(1, fps))
        self.frame_size = 64  # Size for frame analysis
        
        if comparison_methods is None:
            self.comparison_methods = ["phash", "ssim", "histogram"]
        else:
            self.comparison_methods = comparison_methods
    
    @staticmethod
    def _extract_frames_opencv(video_path: str, start_sec: float, end_sec: float, 
                              fps: int, frame_size: int) -> List[np.ndarray]:
        """
        Extract frames using OpenCV for analysis.
        
        Args:
            video_path (str): Path to video file
            start_sec (float): Start time in seconds
            end_sec (float): End time in seconds
            fps (int): Frames per second for extraction
            frame_size (int): Size for resizing frames
            
        Returns:
            List[np.ndarray]: List of extracted frames
        """
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            # Set start position
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            
            frame_interval = 1.0 / fps
            current_time = start_sec
            
            while current_time <= end_sec:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Convert to grayscale and resize
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (frame_size, frame_size))
                frames.append(resized)
                
                current_time += frame_interval
            
            cap.release()
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Error extracting frames: {e}\n")
        
        return frames
    
    @staticmethod
    def _calculate_phash(frame: np.ndarray) -> int:
        """
        Calculate perceptual hash for a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            int: Perceptual hash value
        """
        try:
            # Resize to 8x8 for DCT
            small = cv2.resize(frame, (8, 8))
            
            # Apply DCT
            dct_frame = dct(dct(small.T, norm='ortho').T, norm='ortho')
            
            # Calculate median of top-left 8x8 excluding DC component
            dct_low = dct_frame[:8, :8]
            median = np.median(dct_low.flatten()[1:])  # Exclude DC component
            
            # Generate hash
            hash_bits = dct_low > median
            
            # Convert to integer
            phash = 0
            for i, bit in enumerate(hash_bits.flatten()):
                if bit:
                    phash |= 1 << i
            
            return phash
            
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two frames.
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            float: SSIM value (0-1, higher is more similar)
        """
        try:
            # Convert to float
            f1 = frame1.astype(np.float64)
            f2 = frame2.astype(np.float64)
            
            # Calculate means
            mu1 = np.mean(f1)
            mu2 = np.mean(f2)
            
            # Calculate variances and covariance
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = np.mean(f1 ** 2) - mu1_sq
            sigma2_sq = np.mean(f2 ** 2) - mu2_sq
            sigma12 = np.mean(f1 * f2) - mu1_mu2
            
            # SSIM constants
            k1, k2 = 0.01, 0.03
            L = 255  # Dynamic range
            c1 = (k1 * L) ** 2
            c2 = (k2 * L) ** 2
            
            # Calculate SSIM
            numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
            denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            
            if denominator == 0:
                return 0.0
            
            ssim = numerator / denominator
            return max(0.0, min(1.0, ssim))
            
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_histogram_correlation(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate histogram correlation between two frames.
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            float: Correlation coefficient (0-1, higher is more similar)
        """
        try:
            # Calculate histograms
            hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
            
            # Normalize histograms
            hist1 = hist1.flatten() / np.sum(hist1)
            hist2 = hist2.flatten() / np.sum(hist2)
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1.astype(np.float32), 
                                        hist2.astype(np.float32), 
                                        cv2.HISTCMP_CORREL)
            
            return max(0.0, min(1.0, correlation))
            
        except Exception:
            return 0.0
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Calculate Hamming distance between two hashes.
        
        Args:
            hash1 (int): First hash
            hash2 (int): Second hash
            
        Returns:
            int: Hamming distance
        """
        return bin(hash1 ^ hash2).count('1')
    
    def compare_frame_sequences(self, max_offset_frames: int = 30) -> Dict[str, any]:
        """
        Compare frame sequences using multiple methods and estimate uncertainty.
        
        Args:
            max_offset_frames (int): Maximum offset to search in frames
            
        Returns:
            Dict[str, any]: Comparison results with delay and uncertainty
        """
        try:
            # Extract frames from both videos
            ref_frames = self._extract_frames_opencv(self.ref_path, self.start_sec, 
                                                   self.end_sec, self.fps, self.frame_size)
            tgt_frames = self._extract_frames_opencv(self.tgt_path, self.start_sec, 
                                                   self.end_sec, self.fps, self.frame_size)
            
            if len(ref_frames) < 5 or len(tgt_frames) < 5:
                return {"delay": 0, "confidence": 0.0, "uncertainty": 1.0, "method": "insufficient_data"}
            
            # Limit search space for performance
            max_ref_frames = min(len(ref_frames), 60)
            max_tgt_frames = min(len(tgt_frames), 60)
            
            results = {}
            
            # Method 1: Perceptual Hash Comparison
            if "phash" in self.comparison_methods:
                results["phash"] = self._compare_phash(ref_frames[:max_ref_frames], 
                                                      tgt_frames[:max_tgt_frames], 
                                                      max_offset_frames)
            
            # Method 2: SSIM Comparison
            if "ssim" in self.comparison_methods:
                results["ssim"] = self._compare_ssim(ref_frames[:max_ref_frames], 
                                                    tgt_frames[:max_tgt_frames], 
                                                    max_offset_frames)
            
            # Method 3: Histogram Comparison
            if "histogram" in self.comparison_methods:
                results["histogram"] = self._compare_histogram(ref_frames[:max_ref_frames], 
                                                             tgt_frames[:max_tgt_frames], 
                                                             max_offset_frames)
            
            # Combine results and calculate final delay and uncertainty
            return self._combine_results(results)
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Error in frame comparison: {e}\n")
            return {"delay": 0, "confidence": 0.0, "uncertainty": 1.0, "method": "error"}
    
    def _compare_phash(self, ref_frames: List[np.ndarray], tgt_frames: List[np.ndarray], 
                      max_offset: int) -> Dict[str, any]:
        """
        Compare frames using perceptual hashing.
        
        Args:
            ref_frames (List[np.ndarray]): Reference frames
            tgt_frames (List[np.ndarray]): Target frames
            max_offset (int): Maximum offset to search
            
        Returns:
            Dict[str, any]: Comparison results
        """
        # Calculate hashes for all frames
        ref_hashes = [self._calculate_phash(frame) for frame in ref_frames]
        tgt_hashes = [self._calculate_phash(frame) for frame in tgt_frames]
        
        best_offset = 0
        best_score = float('inf')
        scores = []
        
        # Search for best alignment
        for offset in range(-max_offset, max_offset + 1):
            total_distance = 0
            comparisons = 0
            
            for i in range(len(ref_hashes)):
                j = i + offset
                if 0 <= j < len(tgt_hashes):
                    distance = self._hamming_distance(ref_hashes[i], tgt_hashes[j])
                    total_distance += distance
                    comparisons += 1
            
            if comparisons > 0:
                avg_distance = total_distance / comparisons
                scores.append(avg_distance)
                if avg_distance < best_score:
                    best_score = avg_distance
                    best_offset = offset
        
        # Calculate confidence (lower distance = higher confidence)
        confidence = max(0.0, 1.0 - (best_score / 32.0))  # 32 is max hamming distance for 64-bit hash
        
        return {
            "offset": best_offset,
            "score": best_score,
            "confidence": confidence,
            "scores": scores
        }
    
    def _compare_ssim(self, ref_frames: List[np.ndarray], tgt_frames: List[np.ndarray], 
                     max_offset: int) -> Dict[str, any]:
        """
        Compare frames using SSIM.
        
        Args:
            ref_frames (List[np.ndarray]): Reference frames
            tgt_frames (List[np.ndarray]): Target frames
            max_offset (int): Maximum offset to search
            
        Returns:
            Dict[str, any]: Comparison results
        """
        best_offset = 0
        best_score = 0.0
        scores = []
        
        # Search for best alignment
        for offset in range(-max_offset, max_offset + 1):
            total_ssim = 0.0
            comparisons = 0
            
            for i in range(len(ref_frames)):
                j = i + offset
                if 0 <= j < len(tgt_frames):
                    ssim = self._calculate_ssim(ref_frames[i], tgt_frames[j])
                    total_ssim += ssim
                    comparisons += 1
            
            if comparisons > 0:
                avg_ssim = total_ssim / comparisons
                scores.append(avg_ssim)
                if avg_ssim > best_score:
                    best_score = avg_ssim
                    best_offset = offset
        
        return {
            "offset": best_offset,
            "score": best_score,
            "confidence": best_score,
            "scores": scores
        }
    
    def _compare_histogram(self, ref_frames: List[np.ndarray], tgt_frames: List[np.ndarray], 
                          max_offset: int) -> Dict[str, any]:
        """
        Compare frames using histogram correlation.
        
        Args:
            ref_frames (List[np.ndarray]): Reference frames
            tgt_frames (List[np.ndarray]): Target frames
            max_offset (int): Maximum offset to search
            
        Returns:
            Dict[str, any]: Comparison results
        """
        best_offset = 0
        best_score = 0.0
        scores = []
        
        # Search for best alignment
        for offset in range(-max_offset, max_offset + 1):
            total_corr = 0.0
            comparisons = 0
            
            for i in range(len(ref_frames)):
                j = i + offset
                if 0 <= j < len(tgt_frames):
                    corr = self._calculate_histogram_correlation(ref_frames[i], tgt_frames[j])
                    total_corr += corr
                    comparisons += 1
            
            if comparisons > 0:
                avg_corr = total_corr / comparisons
                scores.append(avg_corr)
                if avg_corr > best_score:
                    best_score = avg_corr
                    best_offset = offset
        
        return {
            "offset": best_offset,
            "score": best_score,
            "confidence": best_score,
            "scores": scores
        }
    
    def _combine_results(self, results: Dict[str, Dict[str, any]]) -> Dict[str, any]:
        """
        Combine results from multiple comparison methods.
        
        Args:
            results (Dict[str, Dict[str, any]]): Results from different methods
            
        Returns:
            Dict[str, any]: Combined results
        """
        if not results:
            return {"delay": 0, "confidence": 0.0, "uncertainty": 1.0, "method": "no_results"}
        
        # Extract offsets and confidences
        offsets = []
        confidences = []
        
        for method, result in results.items():
            if result["confidence"] > 0.3:  # Only consider results with reasonable confidence
                offsets.append(result["offset"])
                confidences.append(result["confidence"])
        
        if not offsets:
            return {"delay": 0, "confidence": 0.0, "uncertainty": 1.0, "method": "low_confidence"}
        
        # Calculate weighted average offset
        if len(confidences) > 0:
            total_weight = sum(confidences)
            if total_weight > 0:
                weighted_offset = sum(o * c for o, c in zip(offsets, confidences)) / total_weight
            else:
                weighted_offset = np.median(offsets)
        else:
            weighted_offset = np.median(offsets)
        
        # Calculate uncertainty based on variance and confidence
        if len(offsets) > 1:
            offset_variance = np.var(offsets)
            uncertainty = min(1.0, offset_variance / 100.0)  # Normalize variance
        else:
            uncertainty = 0.3  # Medium uncertainty for single result
        
        # Overall confidence is average of method confidences
        overall_confidence = np.mean(confidences)
        
        # Convert frame offset to time delay (milliseconds)
        delay_ms = (weighted_offset / self.fps) * 1000
        
        return {
            "delay": delay_ms,
            "confidence": overall_confidence,
            "uncertainty": uncertainty,
            "frame_offset": weighted_offset,
            "method": "combined",
            "method_results": results
        }

class uncertainty_estimator:
    """
    Estimate uncertainty in delay detection based on multiple factors.
    
    This class combines various uncertainty sources to provide a comprehensive
    uncertainty estimate for delay detection results.
    """
    
    def __init__(self):
        self.factors = []
    
    def add_framerate_uncertainty(self, video1_fps: float, video2_fps: float) -> float:
        """
        Add uncertainty due to framerate differences.
        
        Args:
            video1_fps (float): First video framerate
            video2_fps (float): Second video framerate
            
        Returns:
            float: Framerate uncertainty factor
        """
        if abs(video1_fps - video2_fps) > 0.1:
            # Different framerates introduce uncertainty
            fps_diff = abs(video1_fps - video2_fps) / max(video1_fps, video2_fps)
            uncertainty = min(fps_diff * 2, 0.8)  # Cap at 0.8
        else:
            uncertainty = 0.05  # Minimal uncertainty for same framerate
        
        self.factors.append(("framerate", uncertainty))
        return uncertainty
    
    def add_scene_complexity_uncertainty(self, scene_changes_per_second: float) -> float:
        """
        Add uncertainty due to scene complexity.
        
        Args:
            scene_changes_per_second (float): Rate of scene changes
            
        Returns:
            float: Scene complexity uncertainty factor
        """
        # More scene changes can make detection more difficult
        if scene_changes_per_second > 0.5:
            uncertainty = min(scene_changes_per_second * 0.3, 0.6)
        else:
            uncertainty = 0.1  # Low uncertainty for stable scenes
        
        self.factors.append(("scene_complexity", uncertainty))
        return uncertainty
    
    def add_audio_quality_uncertainty(self, audio_similarity: float) -> float:
        """
        Add uncertainty based on audio quality/similarity.
        
        Args:
            audio_similarity (float): Audio similarity score (0-1)
            
        Returns:
            float: Audio quality uncertainty factor
        """
        # Lower audio similarity indicates potential issues
        uncertainty = 1.0 - audio_similarity
        uncertainty = min(uncertainty, 0.7)  # Cap uncertainty
        
        self.factors.append(("audio_quality", uncertainty))
        return uncertainty
    
    def calculate_combined_uncertainty(self) -> float:
        """
        Calculate combined uncertainty from all factors.
        
        Returns:
            float: Combined uncertainty score (0-1)
        """
        if not self.factors:
            return 0.5  # Default medium uncertainty
        
        # Use weighted combination of uncertainties
        weights = {"framerate": 0.3, "scene_complexity": 0.3, "audio_quality": 0.4}
        
        total_uncertainty = 0.0
        total_weight = 0.0
        
        for factor_name, uncertainty in self.factors:
            weight = weights.get(factor_name, 0.33)
            total_uncertainty += uncertainty * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(total_uncertainty / total_weight, 1.0)
        else:
            return 0.5
    
    def get_uncertainty_breakdown(self) -> Dict[str, float]:
        """
        Get breakdown of uncertainty factors.
        
        Returns:
            Dict[str, float]: Dictionary of uncertainty factors
        """
        return dict(self.factors)
