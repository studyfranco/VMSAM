"""Enhanced frame comparison module for improved delay uncertainty detection.

This module extends the existing frame_compare.py with additional algorithms
for more robust delay uncertainty quantification and multi-frame analysis.

Author: studyfranco
Created: 2025-01-16
"""

import sys
import os
import math
import io
import struct
import numpy as np
from scipy.fft import dct
from scipy.signal import correlate, find_peaks
from typing import List, Tuple, Optional, Dict, Any
from threading import Thread
from statistics import mean, variance
from decimal import Decimal

import tools
from frame_compare import FrameComparer


class EnhancedFrameComparer(FrameComparer):
    """Enhanced frame comparer with improved uncertainty quantification.
    
    Extends the base FrameComparer with additional algorithms for better
    delay uncertainty detection and multi-frame analysis capabilities.
    """
    
    def __init__(self, ref_path: str, tgt_path: str, start_sec: float, end_sec: float,
                 fps: int = 10, band_width: int = 20, max_search_frames: int = 50, 
                 debug: bool = False, scene_threshold: float = 0.30,
                 uncertainty_window: int = 5, confidence_threshold: float = 0.7):
        """Initialize enhanced frame comparer.
        
        Args:
            ref_path: Reference video path
            tgt_path: Target video path  
            start_sec: Start time in seconds
            end_sec: End time in seconds
            fps: Frames per second for extraction
            band_width: Search band width for alignment
            max_search_frames: Maximum frames to search
            debug: Enable debug output
            scene_threshold: Scene detection threshold
            uncertainty_window: Window size for uncertainty analysis
            confidence_threshold: Minimum confidence for reliable detection
        """
        super().__init__(ref_path, tgt_path, start_sec, end_sec, fps, 
                        band_width, max_search_frames, debug, scene_threshold)
        self.uncertainty_window = uncertainty_window
        self.confidence_threshold = confidence_threshold
        
    def _calculate_frame_similarity_matrix(self, ref_hashes: List[int], 
                                         tgt_hashes: List[int]) -> np.ndarray:
        """Calculate similarity matrix between reference and target frame hashes.
        
        Args:
            ref_hashes: Reference frame hashes
            tgt_hashes: Target frame hashes
            
        Returns:
            Similarity matrix (higher values = more similar)
        """
        n_ref = len(ref_hashes)
        n_tgt = len(tgt_hashes)
        
        similarity_matrix = np.zeros((n_ref, n_tgt))
        
        for i, ref_hash in enumerate(ref_hashes):
            for j, tgt_hash in enumerate(tgt_hashes):
                # Convert Hamming distance to similarity (64 - distance)
                hamming_dist = self._popcount64(ref_hash ^ tgt_hash)
                similarity_matrix[i, j] = 64 - hamming_dist
                
        return similarity_matrix
        
    def _find_optimal_alignment_with_uncertainty(self, ref_hashes: List[int], 
                                               tgt_hashes: List[int]) -> Dict[str, Any]:
        """Find optimal alignment with uncertainty quantification.
        
        Args:
            ref_hashes: Reference frame hashes
            tgt_hashes: Target frame hashes
            
        Returns:
            Dictionary containing alignment results and uncertainty metrics
        """
        similarity_matrix = self._calculate_frame_similarity_matrix(ref_hashes, tgt_hashes)
        
        n_ref, n_tgt = similarity_matrix.shape
        if n_ref == 0 or n_tgt == 0:
            return {"alignment": None, "confidence": 0.0, "uncertainty": 1.0}
            
        # Dynamic programming for optimal alignment
        dp = np.zeros((n_ref + 1, n_tgt + 1))
        path = np.zeros((n_ref + 1, n_tgt + 1, 2), dtype=int)
        
        for i in range(1, n_ref + 1):
            for j in range(1, n_tgt + 1):
                # Match
                match_score = dp[i-1, j-1] + similarity_matrix[i-1, j-1]
                # Gap in reference
                gap_ref_score = dp[i-1, j] - 1
                # Gap in target
                gap_tgt_score = dp[i, j-1] - 1
                
                scores = [match_score, gap_ref_score, gap_tgt_score]
                best_idx = np.argmax(scores)
                dp[i, j] = scores[best_idx]
                
                if best_idx == 0:
                    path[i, j] = [i-1, j-1]
                elif best_idx == 1:
                    path[i, j] = [i-1, j]
                else:
                    path[i, j] = [i, j-1]
                    
        # Backtrack to find alignment
        alignment = []
        i, j = n_ref, n_tgt
        while i > 0 or j > 0:
            if i > 0 and j > 0 and path[i, j][0] == i-1 and path[i, j][1] == j-1:
                # Match
                alignment.append((i-1, j-1))
                i, j = i-1, j-1
            elif i > 0 and path[i, j][0] == i-1:
                # Gap in target
                alignment.append((i-1, -1))
                i -= 1
            else:
                # Gap in reference
                alignment.append((-1, j-1))
                j -= 1
                
        alignment.reverse()
        
        # Calculate confidence and uncertainty metrics
        valid_matches = [(r, t) for r, t in alignment if r >= 0 and t >= 0]
        if not valid_matches:
            return {"alignment": alignment, "confidence": 0.0, "uncertainty": 1.0}
            
        match_scores = [similarity_matrix[r, t] for r, t in valid_matches]
        avg_similarity = mean(match_scores)
        similarity_variance = variance(match_scores) if len(match_scores) > 1 else 0.0
        
        # Normalize confidence (similarity out of 64 possible bits)
        base_confidence = avg_similarity / 64.0
        
        # Penalize high variance in matches
        variance_penalty = min(0.3, similarity_variance / 100.0)
        confidence = max(0.0, base_confidence - variance_penalty)
        
        # Calculate uncertainty as inverse of confidence
        uncertainty = 1.0 - confidence
        
        return {
            "alignment": alignment,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "match_count": len(valid_matches),
            "avg_similarity": avg_similarity,
            "similarity_variance": similarity_variance
        }
        
    def _analyze_temporal_consistency(self, alignment_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze temporal consistency across multiple alignment results.
        
        Args:
            alignment_results: List of alignment result dictionaries
            
        Returns:
            Dictionary containing temporal consistency metrics
        """
        if not alignment_results:
            return {"temporal_consistency": 0.0, "confidence_stability": 0.0}
            
        confidences = [result["confidence"] for result in alignment_results]
        uncertainties = [result["uncertainty"] for result in alignment_results]
        
        # Calculate temporal consistency
        confidence_variance = variance(confidences) if len(confidences) > 1 else 0.0
        temporal_consistency = max(0.0, 1.0 - confidence_variance)
        
        # Calculate confidence stability
        confidence_stability = mean(confidences)
        
        # Detect confidence drops (potential issues)
        confidence_drops = 0
        for i in range(1, len(confidences)):
            if confidences[i] < confidences[i-1] - 0.2:
                confidence_drops += 1
                
        stability_penalty = min(0.3, confidence_drops * 0.1)
        confidence_stability = max(0.0, confidence_stability - stability_penalty)
        
        return {
            "temporal_consistency": temporal_consistency,
            "confidence_stability": confidence_stability,
            "confidence_variance": confidence_variance,
            "confidence_drops": confidence_drops
        }
        
    def find_scene_gap_with_uncertainty(self, before_common: int = 2, 
                                      after_common: int = 3) -> Optional[Dict[str, Any]]:
        """Find scene gap with comprehensive uncertainty analysis.
        
        Args:
            before_common: Frames to include before scene boundary
            after_common: Frames to include after scene boundary
            
        Returns:
            Dictionary containing gap information and uncertainty metrics
        """
        # Get basic scene gap from parent class
        basic_gap = self.find_scene_gap_requirements(before_common, after_common)
        if not basic_gap:
            return None
            
        # Extract frames for detailed analysis
        dur = max(0.5, self.end_sec - self.start_sec)
        pad = min(2.0, dur / 4.0)
        start = max(0.0, self.start_sec - pad)
        dur2 = dur + 2 * pad
        
        ref_blob = self._ffmpeg_raw_frames(self.ref_path, start, dur2)
        tgt_blob = self._ffmpeg_raw_frames(self.tgt_path, start, dur2)
        ref_hashes = self._phash64_frames(ref_blob)
        tgt_hashes = self._phash64_frames(tgt_blob)
        
        if not ref_hashes or not tgt_hashes:
            return basic_gap
            
        # Perform detailed alignment analysis in windows
        window_size = self.uncertainty_window
        alignment_results = []
        
        # Analyze multiple overlapping windows
        for i in range(0, max(len(ref_hashes), len(tgt_hashes)) - window_size, window_size // 2):
            ref_window = ref_hashes[i:i+window_size]
            tgt_window = tgt_hashes[i:i+window_size]
            
            if len(ref_window) >= window_size // 2 and len(tgt_window) >= window_size // 2:
                result = self._find_optimal_alignment_with_uncertainty(ref_window, tgt_window)
                result["window_start"] = i
                alignment_results.append(result)
                
        # Analyze temporal consistency
        temporal_metrics = self._analyze_temporal_consistency(alignment_results)
        
        # Calculate overall confidence
        if alignment_results:
            overall_confidence = mean([r["confidence"] for r in alignment_results])
            overall_uncertainty = mean([r["uncertainty"] for r in alignment_results])
        else:
            overall_confidence = 0.0
            overall_uncertainty = 1.0
            
        # Adjust confidence based on temporal consistency
        temporal_weight = 0.3
        adjusted_confidence = (overall_confidence * (1 - temporal_weight) + 
                             temporal_metrics["confidence_stability"] * temporal_weight)
        
        # Create enhanced result
        enhanced_result = basic_gap.copy()
        enhanced_result.update({
            "confidence": adjusted_confidence,
            "uncertainty": 1.0 - adjusted_confidence,
            "temporal_consistency": temporal_metrics["temporal_consistency"],
            "confidence_stability": temporal_metrics["confidence_stability"],
            "alignment_results": alignment_results,
            "is_reliable": adjusted_confidence >= self.confidence_threshold
        })
        
        if self.debug:
            sys.stderr.write(f"[enhanced_frame_compare] Overall confidence: {adjusted_confidence:.3f}\n")
            sys.stderr.write(f"[enhanced_frame_compare] Temporal consistency: {temporal_metrics['temporal_consistency']:.3f}\n")
            sys.stderr.write(f"[enhanced_frame_compare] Reliable detection: {enhanced_result['is_reliable']}\n")
            
        return enhanced_result
        
    def compare_multiple_segments(self, segment_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Dict[str, Any]:
        """Compare multiple video segments for robust delay estimation.
        
        Args:
            segment_pairs: List of (ref_segment, tgt_segment) time pairs
            
        Returns:
            Dictionary containing aggregated comparison results
        """
        segment_results = []
        
        for i, ((ref_start, ref_end), (tgt_start, tgt_end)) in enumerate(segment_pairs):
            # Create temporary comparer for this segment
            segment_comparer = EnhancedFrameComparer(
                self.ref_path, self.tgt_path, ref_start, ref_end,
                self.fps, self.band_width, self.max_search_frames,
                self.debug, self.scene_threshold, 
                self.uncertainty_window, self.confidence_threshold
            )
            
            # Compare segment
            segment_result = segment_comparer.find_scene_gap_with_uncertainty()
            if segment_result:
                segment_result["segment_index"] = i
                segment_result["ref_segment"] = (ref_start, ref_end)
                segment_result["tgt_segment"] = (tgt_start, tgt_end)
                segment_results.append(segment_result)
                
        if not segment_results:
            return {"consensus_delay": None, "confidence": 0.0}
            
        # Calculate consensus delay
        delays = []
        confidences = []
        
        for result in segment_results:
            if result["is_reliable"]:
                # Calculate delay from gap information
                gap_delay = (result["start_time"] - result["tgt_segment"][0]) * 1000.0
                delays.append(gap_delay)
                confidences.append(result["confidence"])
                
        if not delays:
            return {"consensus_delay": None, "confidence": 0.0}
            
        # Weighted consensus based on confidence
        total_weight = sum(confidences)
        if total_weight > 0:
            consensus_delay = sum(d * c for d, c in zip(delays, confidences)) / total_weight
        else:
            consensus_delay = mean(delays)
            
        # Calculate consensus confidence
        consensus_confidence = mean(confidences)
        
        # Calculate delay variance for uncertainty assessment
        if len(delays) > 1:
            delay_variance = variance(delays)
            variance_penalty = min(0.3, delay_variance / 1000.0)  # Normalize by 1000ms
            consensus_confidence = max(0.0, consensus_confidence - variance_penalty)
            
        return {
            "consensus_delay": consensus_delay,
            "confidence": consensus_confidence,
            "individual_delays": delays,
            "individual_confidences": confidences,
            "delay_variance": delay_variance if len(delays) > 1 else 0.0,
            "segment_count": len(delays),
            "segment_results": segment_results
        }


def compare_video_segments_for_delay_uncertainty(video_obj_1, video_obj_2, 
                                               scene_boundaries: List[Tuple[float, float]],
                                               initial_delay: Decimal) -> Tuple[Decimal, float, Dict[str, Any]]:
    """Compare video segments around scene boundaries for delay uncertainty analysis.
    
    This function uses the enhanced frame comparison to analyze delay uncertainty
    by examining multiple scene boundaries and calculating consensus.
    
    Args:
        video_obj_1: First video object
        video_obj_2: Second video object
        scene_boundaries: List of (start_time, end_time) for scene boundaries
        initial_delay: Initial delay estimate in milliseconds
        
    Returns:
        Tuple of (refined_delay, confidence, analysis_details)
    """
    if not scene_boundaries:
        return initial_delay, 0.0, {"error": "No scene boundaries provided"}
        
    try:
        # Prepare segment pairs for comparison
        segment_pairs = []
        delay_seconds = float(initial_delay) / 1000.0
        
        for scene_start, scene_end in scene_boundaries[:3]:  # Limit to 3 scenes for performance
            # Create small windows around scene boundaries
            window_duration = min(5.0, (scene_end - scene_start) / 2.0)
            
            ref_segment = (scene_start, scene_start + window_duration)
            tgt_segment = (scene_start + delay_seconds, scene_start + delay_seconds + window_duration)
            
            segment_pairs.append((ref_segment, tgt_segment))
            
        if not segment_pairs:
            return initial_delay, 0.0, {"error": "No valid segments created"}
            
        # Create enhanced frame comparer
        comparer = EnhancedFrameComparer(
            video_obj_1.filePath, 
            video_obj_2.filePath,
            segment_pairs[0][0][0],  # Use first segment start time
            segment_pairs[-1][0][1],  # Use last segment end time
            fps=10,
            debug=tools.dev if hasattr(tools, 'dev') else False
        )
        
        # Perform multi-segment comparison
        comparison_result = comparer.compare_multiple_segments(segment_pairs)
        
        if comparison_result["consensus_delay"] is not None:
            # Refine delay based on frame comparison
            delay_adjustment = Decimal(str(comparison_result["consensus_delay"]))
            refined_delay = initial_delay + delay_adjustment
            confidence = comparison_result["confidence"]
            
            analysis_details = {
                "method": "enhanced_frame_comparison",
                "segments_analyzed": comparison_result["segment_count"],
                "delay_adjustment": float(delay_adjustment),
                "delay_variance": comparison_result["delay_variance"],
                "individual_results": comparison_result["segment_results"]
            }
            
            return refined_delay, confidence, analysis_details
        else:
            return initial_delay, 0.0, {"error": "Frame comparison failed"}
            
    except Exception as e:
        if hasattr(tools, 'dev') and tools.dev:
            sys.stderr.write(f"Enhanced frame comparison failed: {e}\n")
        return initial_delay, 0.0, {"error": str(e)}
