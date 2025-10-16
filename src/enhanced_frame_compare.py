#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Frame Comparison Module for VMSAM

This module provides advanced frame comparison with uncertainty detection
and multiple comparison methods. It extends the original frame_compare.py
with improved accuracy and reliability metrics.

Author: VMSAM Enhancement Team
Compatible with: Python 3.7+
Dependencies: numpy, opencv-python, scipy, scikit-image
"""

import sys
import os
import math
import struct
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from threading import Thread, RLock
from decimal import Decimal
import gc
import tools

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    
try:
    from scipy.fft import dct
    from scipy.spatial.distance import cosine, euclidean
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Global locks and caches
frame_compare_lock = RLock()
frame_cache = {}

class FrameComparisonResult:
    """Container for frame comparison results with uncertainty metrics."""
    
    def __init__(self, similarity: float, uncertainty: float, 
                 method: str, metadata: Optional[Dict] = None):
        """
        Initialize frame comparison result.
        
        Args:
            similarity: Similarity score [0.0-1.0]
            uncertainty: Uncertainty estimate [0.0-1.0]
            method: Comparison method used
            metadata: Additional metadata
        """
        self.similarity = max(0.0, min(1.0, similarity))
        self.uncertainty = max(0.0, min(1.0, uncertainty))
        self.method = method
        self.metadata = metadata or {}
        self.confidence = 1.0 - self.uncertainty
        
    def is_match(self, threshold: float = 0.8) -> bool:
        """Check if frames match based on similarity threshold."""
        return self.similarity >= threshold
        
    def is_reliable(self, min_confidence: float = 0.7) -> bool:
        """Check if result is reliable based on confidence."""
        return self.confidence >= min_confidence

class EnhancedFrameComparer:
    """Enhanced frame comparison with multiple methods and uncertainty estimation."""
    
    def __init__(self, 
                 primary_method: str = "phash",
                 fallback_method: str = "histogram",
                 enable_uncertainty: bool = True):
        """
        Initialize the enhanced frame comparer.
        
        Args:
            primary_method: Primary comparison method
            fallback_method: Fallback method if primary fails
            enable_uncertainty: Whether to calculate uncertainty metrics
        """
        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.enable_uncertainty = enable_uncertainty
        
        # Validate dependencies
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV is required for frame comparison")
            
        # Method availability check
        self.available_methods = ["phash", "histogram", "mse"]
        if SCIPY_AVAILABLE:
            self.available_methods.extend(["dct_hash", "correlation"])
        if SKIMAGE_AVAILABLE:
            self.available_methods.extend(["ssim", "lbp"])
            
    def compare_frames_from_video(self, 
                                video1_path: str, video2_path: str,
                                time_ranges: List[Tuple[float, float]],
                                fps: float = 5.0) -> List[FrameComparisonResult]:
        """
        Compare frames from two videos at specified time ranges.
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video
            time_ranges: List of (start_time, end_time) pairs
            fps: Frame sampling rate
            
        Returns:
            List of comparison results for each time range
        """
        results = []
        
        for start_time, end_time in time_ranges:
            try:
                # Extract frames from both videos
                frames1 = self._extract_frames(video1_path, start_time, end_time, fps)
                frames2 = self._extract_frames(video2_path, start_time, end_time, fps)
                
                if not frames1 or not frames2:
                    results.append(FrameComparisonResult(0.0, 1.0, "extraction_failed"))
                    continue
                    
                # Compare frame sequences
                result = self._compare_frame_sequences(frames1, frames2)
                results.append(result)
                
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tFrame comparison failed for range {start_time}-{end_time}: {e}\n")
                results.append(FrameComparisonResult(0.0, 1.0, "error", {"error": str(e)}))
                
        return results
        
    def _extract_frames(self, video_path: str, start_time: float, 
                       end_time: float, fps: float) -> List[np.ndarray]:
        """
        Extract frames from video using FFmpeg.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Frame extraction rate
            
        Returns:
            List of extracted frames as numpy arrays
        """
        duration = end_time - start_time
        if duration <= 0:
            return []
            
        # Use FFmpeg to extract frames
        ffmpeg_cmd = [
            tools.software["ffmpeg"],
            "-v", "error",
            "-ss", f"{start_time}",
            "-t", f"{duration}",
            "-i", video_path,
            "-vf", f"fps={fps},scale=64:64",  # Small size for fast comparison
            "-f", "rawvideo",
            "-pix_fmt", "gray",
            "pipe:1"
        ]
        
        try:
            stdout, stderr, returncode = tools.launch_cmdExt_no_test(ffmpeg_cmd)
            
            if returncode != 0:
                if tools.dev:
                    sys.stderr.write(f"\t\tFFmpeg extraction failed: {stderr}\n")
                return []
                
            # Parse frames from raw data
            frame_size = 64 * 64  # 64x64 grayscale
            frames = []
            
            for i in range(0, len(stdout), frame_size):
                if i + frame_size <= len(stdout):
                    frame_data = stdout[i:i + frame_size]
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((64, 64))
                    frames.append(frame)
                    
            return frames
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"\t\tFrame extraction error: {e}\n")
            return []
            
    def _compare_frame_sequences(self, frames1: List[np.ndarray], 
                               frames2: List[np.ndarray]) -> FrameComparisonResult:
        """
        Compare two sequences of frames with uncertainty estimation.
        
        Args:
            frames1: First frame sequence
            frames2: Second frame sequence
            
        Returns:
            Comparison result with uncertainty
        """
        if not frames1 or not frames2:
            return FrameComparisonResult(0.0, 1.0, "empty_sequences")
            
        # Align sequences and compare
        similarities = []
        uncertainties = []
        
        min_len = min(len(frames1), len(frames2))
        
        for i in range(min_len):
            # Primary comparison
            result = self._compare_single_frames(frames1[i], frames2[i], 
                                               self.primary_method)
            similarities.append(result.similarity)
            uncertainties.append(result.uncertainty)
            
        if not similarities:
            return FrameComparisonResult(0.0, 1.0, "no_comparisons")
            
        # Aggregate results
        mean_similarity = np.mean(similarities)
        mean_uncertainty = np.mean(uncertainties)
        
        # Add sequence-level uncertainty based on variance
        similarity_variance = np.var(similarities) if len(similarities) > 1 else 0.0
        sequence_uncertainty = min(1.0, similarity_variance * 2.0)  # Scale variance
        
        final_uncertainty = min(1.0, mean_uncertainty + sequence_uncertainty)
        
        metadata = {
            "frame_count": min_len,
            "similarity_variance": similarity_variance,
            "individual_similarities": similarities[:10]  # Store first 10 for analysis
        }
        
        return FrameComparisonResult(mean_similarity, final_uncertainty, 
                                   f"sequence_{self.primary_method}", metadata)
        
    def _compare_single_frames(self, frame1: np.ndarray, frame2: np.ndarray,
                              method: str) -> FrameComparisonResult:
        """
        Compare two individual frames using the specified method.
        
        Args:
            frame1: First frame
            frame2: Second frame
            method: Comparison method
            
        Returns:
            Comparison result
        """
        try:
            if method == "phash":
                return self._phash_compare(frame1, frame2)
            elif method == "histogram":
                return self._histogram_compare(frame1, frame2)
            elif method == "mse":
                return self._mse_compare(frame1, frame2)
            elif method == "dct_hash" and SCIPY_AVAILABLE:
                return self._dct_hash_compare(frame1, frame2)
            elif method == "ssim" and SKIMAGE_AVAILABLE:
                return self._ssim_compare(frame1, frame2)
            elif method == "correlation" and SCIPY_AVAILABLE:
                return self._correlation_compare(frame1, frame2)
            elif method == "lbp" and SKIMAGE_AVAILABLE:
                return self._lbp_compare(frame1, frame2)
            else:
                # Fallback to histogram
                return self._histogram_compare(frame1, frame2)
                
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"\t\tFrame comparison method {method} failed: {e}\n")
            return FrameComparisonResult(0.0, 1.0, f"{method}_error")
            
    def _phash_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using perceptual hash."""
        def compute_phash(frame):
            # Resize to 32x32 for DCT
            resized = cv2.resize(frame.astype(np.float32), (32, 32))
            
            # Compute DCT
            if SCIPY_AVAILABLE:
                dct_result = dct(dct(resized.T, norm='ortho').T, norm='ortho')
            else:
                # Fallback to simple method
                dct_result = resized
                
            # Extract 8x8 top-left
            dct_low = dct_result[:8, :8]
            
            # Compute hash based on median
            median = np.median(dct_low)
            hash_bits = (dct_low > median).astype(int)
            
            return hash_bits.flatten()
            
        hash1 = compute_phash(frame1)
        hash2 = compute_phash(frame2)
        
        # Hamming distance
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1.0 - (hamming_dist / len(hash1))
        
        # Uncertainty based on frame quality
        frame_variance = (np.var(frame1) + np.var(frame2)) / 2.0
        uncertainty = max(0.0, min(0.5, 1.0 - frame_variance / 1000.0))
        
        return FrameComparisonResult(similarity, uncertainty, "phash")
        
    def _histogram_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using histogram correlation."""
        hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist1 = hist1.flatten() / np.sum(hist1)
        hist2 = hist2.flatten() / np.sum(hist2)
        
        # Correlation coefficient
        correlation = cv2.compareHist(hist1.astype(np.float32), 
                                    hist2.astype(np.float32), 
                                    cv2.HISTCMP_CORREL)
        
        similarity = max(0.0, correlation)
        
        # Uncertainty based on histogram entropy
        entropy1 = -np.sum(hist1 * np.log(hist1 + 1e-10))
        entropy2 = -np.sum(hist2 * np.log(hist2 + 1e-10))
        avg_entropy = (entropy1 + entropy2) / 2.0
        
        # Lower entropy = higher uncertainty (less varied image)
        uncertainty = max(0.0, min(0.5, 1.0 - avg_entropy / 8.0))
        
        return FrameComparisonResult(similarity, uncertainty, "histogram")
        
    def _mse_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using mean squared error."""
        mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        
        # Convert MSE to similarity (0 MSE = 1.0 similarity)
        max_mse = 255.0 ** 2  # Maximum possible MSE for 8-bit images
        similarity = 1.0 - min(1.0, mse / max_mse)
        
        # Uncertainty based on image complexity
        complexity = (np.var(frame1) + np.var(frame2)) / 2.0
        uncertainty = max(0.0, min(0.3, 1.0 - complexity / 2000.0))
        
        return FrameComparisonResult(similarity, uncertainty, "mse")
        
    def _ssim_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using structural similarity index."""
        try:
            ssim_value, _ = ssim(frame1, frame2, full=True)
            similarity = (ssim_value + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            
            # SSIM is generally more reliable
            uncertainty = 0.1
            
            return FrameComparisonResult(similarity, uncertainty, "ssim")
        except:
            # Fallback to MSE if SSIM fails
            return self._mse_compare(frame1, frame2)
            
    def _correlation_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using Pearson correlation."""
        flat1 = frame1.flatten()
        flat2 = frame2.flatten()
        
        try:
            corr_coef, _ = pearsonr(flat1, flat2)
            similarity = (corr_coef + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
            
            # Uncertainty based on data variance
            combined_var = (np.var(flat1) + np.var(flat2)) / 2.0
            uncertainty = max(0.0, min(0.4, 1.0 - combined_var / 1500.0))
            
            return FrameComparisonResult(similarity, uncertainty, "correlation")
        except:
            return FrameComparisonResult(0.0, 1.0, "correlation_error")
            
    def _dct_hash_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using DCT-based hash."""
        def compute_dct_hash(frame):
            # Apply DCT to the frame
            dct_result = dct(dct(frame.astype(float).T, norm='ortho').T, norm='ortho')
            
            # Use low-frequency components
            low_freq = dct_result[:16, :16]
            
            # Compute hash based on sign of coefficients
            hash_vec = (low_freq > 0).astype(int).flatten()
            
            return hash_vec
            
        hash1 = compute_dct_hash(frame1)
        hash2 = compute_dct_hash(frame2)
        
        # Hamming distance
        hamming_dist = np.sum(hash1 != hash2)
        similarity = 1.0 - (hamming_dist / len(hash1))
        
        # DCT-based comparison is quite reliable
        uncertainty = 0.15
        
        return FrameComparisonResult(similarity, uncertainty, "dct_hash")
        
    def _lbp_compare(self, frame1: np.ndarray, frame2: np.ndarray) -> FrameComparisonResult:
        """Compare frames using Local Binary Patterns."""
        try:
            # Compute LBP for both frames
            lbp1 = local_binary_pattern(frame1, 8, 1, method='uniform')
            lbp2 = local_binary_pattern(frame2, 8, 1, method='uniform')
            
            # Compute histograms of LBP
            hist1, _ = np.histogram(lbp1.ravel(), bins=10, range=(0, 9))
            hist2, _ = np.histogram(lbp2.ravel(), bins=10, range=(0, 9))
            
            # Normalize histograms
            hist1 = hist1.astype(float) / np.sum(hist1)
            hist2 = hist2.astype(float) / np.sum(hist2)
            
            # Chi-square distance
            chi2_dist = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
            similarity = max(0.0, 1.0 - chi2_dist / 2.0)
            
            # LBP is texture-based, uncertainty depends on texture richness
            texture_var = (np.var(lbp1) + np.var(lbp2)) / 2.0
            uncertainty = max(0.1, min(0.4, 1.0 - texture_var / 10.0))
            
            return FrameComparisonResult(similarity, uncertainty, "lbp")
        except:
            return self._histogram_compare(frame1, frame2)

def create_frame_comparer(method: str = "phash") -> EnhancedFrameComparer:
    """
    Factory function to create a configured frame comparer.
    
    Args:
        method: Primary comparison method to use
        
    Returns:
        Configured EnhancedFrameComparer instance
    """
    return EnhancedFrameComparer(
        primary_method=method,
        fallback_method="histogram",
        enable_uncertainty=True
    )

def clear_frame_cache():
    """Clear the frame comparison cache to free memory."""
    global frame_cache
    with frame_compare_lock:
        frame_cache.clear()
        gc.collect()
