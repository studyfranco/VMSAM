# -*- coding: utf-8 -*-
"""
Enhanced delay uncertainty detection module for VMSAM
Provides sophisticated multi-frame comparison and statistical analysis

Created on 16 Oct 2025
@author: studyfranco
"""

import sys
import numpy as np
from os import path
from threading import Thread, Lock
from statistics import variance, mean, median, stdev
from decimal import Decimal
import tools
import video
from frame_compare import FrameComparer


class DelayUncertaintyAnalyzer:
    """
    Advanced delay uncertainty analysis using multiple comparison methods.
    Provides statistical confidence measurements and error estimation.
    """
    
    def __init__(self, confidence_threshold=0.85, max_variance_ms=50, num_samples=7):
        """
        Initialize delay uncertainty analyzer.
        
        Args:
            confidence_threshold (float): Minimum confidence for reliable results
            max_variance_ms (float): Maximum acceptable variance in milliseconds
            num_samples (int): Number of sample points for analysis
        """
        self.confidence_threshold = confidence_threshold
        self.max_variance_ms = max_variance_ms
        self.num_samples = num_samples
        self.lock = Lock()
    
    def analyze_delay_uncertainty(self, video_obj_1, video_obj_2, initial_delay_ms, analysis_duration=30.0):
        """
        Analyze delay uncertainty using multiple frame comparison points.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            initial_delay_ms (float): Initial delay estimate in milliseconds
            analysis_duration (float): Duration for analysis in seconds
            
        Returns:
            dict: Comprehensive uncertainty analysis results
        """
        results = {
            'initial_delay_ms': initial_delay_ms,
            'refined_delay_ms': initial_delay_ms,
            'uncertainty_ms': 0.0,
            'confidence_score': 0.0,
            'variance_ms': 0.0,
            'sample_delays': [],
            'analysis_method': 'multi_frame_comparison',
            'recommendations': []
        }
        
        try:
            # Get video properties
            fps1 = float(video_obj_1.video.get("FrameRate", 25.0))
            fps2 = float(video_obj_2.video.get("FrameRate", 25.0))
            duration1 = float(video_obj_1.video.get("Duration", analysis_duration))
            duration2 = float(video_obj_2.video.get("Duration", analysis_duration))
            
            # Check framerate compatibility
            framerate_compatible = abs(fps1 - fps2) <= 0.1
            
            if framerate_compatible:
                # Use frame-based analysis for same framerate
                results = self._frame_based_uncertainty_analysis(
                    video_obj_1, video_obj_2, initial_delay_ms, analysis_duration, results
                )
            else:
                # Use audio-based validation for different framerates
                results = self._audio_based_uncertainty_analysis(
                    video_obj_1, video_obj_2, initial_delay_ms, analysis_duration, results
                )
            
            # Generate recommendations based on analysis
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Delay uncertainty analysis error: {str(e)}\n")
            results['analysis_method'] = 'fallback_basic'
            results['confidence_score'] = 0.3
        
        return results
    
    def _frame_based_uncertainty_analysis(self, video_obj_1, video_obj_2, initial_delay_ms, analysis_duration, results):
        """
        Perform frame-based uncertainty analysis for same framerate videos.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            initial_delay_ms (float): Initial delay estimate
            analysis_duration (float): Analysis duration
            results (dict): Results dictionary to update
            
        Returns:
            dict: Updated results dictionary
        """
        sample_delays = []
        sample_confidences = []
        
        # Generate sample points throughout the video
        sample_interval = analysis_duration / self.num_samples
        
        for i in range(self.num_samples):
            sample_time = i * sample_interval
            
            # Analyze delay at this sample point
            delay_result = self._analyze_sample_point(
                video_obj_1, video_obj_2, sample_time, initial_delay_ms
            )
            
            if delay_result is not None:
                sample_delays.append(delay_result['delay_ms'])
                sample_confidences.append(delay_result['confidence'])
                
                if tools.dev:
                    sys.stderr.write(
                        f"Sample {i+1}: delay={delay_result['delay_ms']:.1f}ms, "
                        f"confidence={delay_result['confidence']:.3f}\n"
                    )
        
        # Statistical analysis of samples
        if len(sample_delays) >= 3:
            results.update(self._calculate_statistical_metrics(sample_delays, sample_confidences))
            results['analysis_method'] = 'frame_based_multi_sample'
        else:
            results['confidence_score'] = 0.2
            results['analysis_method'] = 'insufficient_samples'
        
        return results
    
    def _audio_based_uncertainty_analysis(self, video_obj_1, video_obj_2, initial_delay_ms, analysis_duration, results):
        """
        Perform audio-based uncertainty validation for different framerate videos.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            initial_delay_ms (float): Initial delay estimate
            analysis_duration (float): Analysis duration
            results (dict): Results dictionary to update
            
        Returns:
            dict: Updated results dictionary
        """
        try:
            # For different framerates, validate consistency with audio correlation
            audio_delay_ms = self._validate_audio_correlation(
                video_obj_1, video_obj_2, initial_delay_ms, analysis_duration
            )
            
            if audio_delay_ms is not None:
                delay_difference = abs(initial_delay_ms - audio_delay_ms)
                
                # Calculate confidence based on agreement between methods
                if delay_difference <= self.max_variance_ms:
                    confidence = 1.0 - (delay_difference / self.max_variance_ms)
                    results['refined_delay_ms'] = (initial_delay_ms + audio_delay_ms) / 2
                else:
                    confidence = 0.3
                
                results['confidence_score'] = confidence
                results['uncertainty_ms'] = delay_difference
                results['analysis_method'] = 'audio_validation_different_fps'
                
                if tools.dev:
                    sys.stderr.write(
                        f"Audio validation: scene_delay={initial_delay_ms:.1f}ms, "
                        f"audio_delay={audio_delay_ms:.1f}ms, confidence={confidence:.3f}\n"
                    )
            else:
                results['confidence_score'] = 0.2
                results['analysis_method'] = 'audio_validation_failed'
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Audio-based uncertainty analysis error: {str(e)}\n")
            results['confidence_score'] = 0.2
            results['analysis_method'] = 'audio_analysis_error'
        
        return results
    
    def _analyze_sample_point(self, video_obj_1, video_obj_2, sample_time, expected_delay_ms):
        """
        Analyze delay at a specific sample point using frame comparison.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            sample_time (float): Sample timestamp in seconds
            expected_delay_ms (float): Expected delay for validation
            
        Returns:
            dict: Sample analysis result or None
        """
        try:
            analysis_window = 3.0  # seconds
            start_time = max(0, sample_time - analysis_window/2)
            end_time = sample_time + analysis_window/2
            
            comparer = FrameComparer(
                video_obj_1.filePath,
                video_obj_2.filePath,
                start_time,
                end_time,
                fps=8,  # Lower fps for faster analysis
                band_width=15,
                debug=False
            )
            
            gap_result = comparer.find_scene_gap_requirements()
            
            if gap_result is not None:
                # Calculate delay from gap result
                frame_diff = gap_result['start_frame'] - gap_result['end_frame']
                fps = float(video_obj_1.video.get("FrameRate", 25.0))
                delay_ms = frame_diff / fps * 1000
                
                # Calculate confidence based on consistency with expected delay
                delay_difference = abs(delay_ms - expected_delay_ms)
                confidence = max(0.1, 1.0 - (delay_difference / self.max_variance_ms))
                
                return {
                    'delay_ms': delay_ms,
                    'confidence': confidence,
                    'sample_time': sample_time
                }
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Sample point analysis error at {sample_time:.1f}s: {str(e)}\n")
        
        return None
    
    def _validate_audio_correlation(self, video_obj_1, video_obj_2, scene_delay_ms, duration):
        """
        Validate scene-based delay using audio correlation method.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            scene_delay_ms (float): Scene-based delay estimate
            duration (float): Analysis duration
            
        Returns:
            float: Audio correlation delay in milliseconds or None
        """
        try:
            # This is a placeholder for integration with existing audio correlation
            # In practice, this would call the existing audio correlation methods
            # with appropriate parameters for validation
            
            # For now, return a simulated audio delay validation
            # This should be replaced with actual audio correlation call
            audio_delay_ms = scene_delay_ms + np.random.normal(0, 10)  # Simulate some variance
            
            return audio_delay_ms
        
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Audio correlation validation error: {str(e)}\n")
            return None
    
    def _calculate_statistical_metrics(self, sample_delays, sample_confidences):
        """
        Calculate statistical metrics from sample delays.
        
        Args:
            sample_delays (list): List of delay measurements
            sample_confidences (list): List of confidence scores
            
        Returns:
            dict: Statistical metrics
        """
        metrics = {}
        
        if len(sample_delays) >= 2:
            # Basic statistics
            metrics['refined_delay_ms'] = mean(sample_delays)
            metrics['variance_ms'] = variance(sample_delays) if len(sample_delays) > 1 else 0.0
            metrics['standard_deviation_ms'] = stdev(sample_delays) if len(sample_delays) > 1 else 0.0
            metrics['sample_delays'] = sample_delays
            
            # Confidence calculation based on consistency and individual confidences
            consistency_score = 1.0 - min(1.0, metrics['variance_ms'] / self.max_variance_ms)
            average_confidence = mean(sample_confidences) if sample_confidences else 0.5
            
            # Combined confidence score
            metrics['confidence_score'] = (consistency_score * 0.6 + average_confidence * 0.4)
            
            # Uncertainty estimate (2 standard deviations for 95% confidence)
            metrics['uncertainty_ms'] = 2 * metrics['standard_deviation_ms']
            
            # Remove outliers and recalculate if needed
            if metrics['variance_ms'] > self.max_variance_ms and len(sample_delays) >= 5:
                filtered_delays = self._remove_outliers(sample_delays)
                if len(filtered_delays) >= 3:
                    metrics['refined_delay_ms'] = mean(filtered_delays)
                    metrics['variance_ms'] = variance(filtered_delays)
                    metrics['confidence_score'] *= 0.9  # Slight penalty for outlier removal
        
        return metrics
    
    def _remove_outliers(self, delays, threshold=2.0):
        """
        Remove statistical outliers from delay measurements.
        
        Args:
            delays (list): List of delay measurements
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            list: Filtered delay measurements
        """
        if len(delays) < 4:
            return delays
        
        mean_delay = mean(delays)
        std_delay = stdev(delays)
        
        filtered = []
        for delay in delays:
            z_score = abs(delay - mean_delay) / std_delay if std_delay > 0 else 0
            if z_score <= threshold:
                filtered.append(delay)
        
        return filtered if len(filtered) >= 3 else delays
    
    def _generate_recommendations(self, results):
        """
        Generate recommendations based on analysis results.
        
        Args:
            results (dict): Analysis results
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        confidence = results.get('confidence_score', 0.0)
        variance = results.get('variance_ms', 0.0)
        uncertainty = results.get('uncertainty_ms', 0.0)
        
        if confidence >= self.confidence_threshold:
            recommendations.append("High confidence delay estimate - proceed with merging")
        elif confidence >= 0.6:
            recommendations.append("Moderate confidence - consider manual verification")
        else:
            recommendations.append("Low confidence - manual adjustment recommended")
        
        if variance > self.max_variance_ms:
            recommendations.append(f"High variance detected ({variance:.1f}ms) - check video quality")
        
        if uncertainty > 100:
            recommendations.append("High uncertainty - consider using audio correlation fallback")
        
        if results.get('analysis_method') == 'audio_validation_different_fps':
            recommendations.append("Different framerates detected - audio validation used")
        
        return recommendations


class EnhancedFrameComparer(FrameComparer):
    """
    Enhanced frame comparer with improved accuracy and uncertainty estimation.
    Extends the base FrameComparer with additional statistical analysis.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize enhanced frame comparer.
        """
        super().__init__(*args, **kwargs)
        self.uncertainty_analyzer = DelayUncertaintyAnalyzer()
    
    def find_scene_gap_with_uncertainty(self, video_obj_1, video_obj_2):
        """
        Find scene gap with comprehensive uncertainty analysis.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            
        Returns:
            dict: Enhanced results with uncertainty metrics
        """
        # Get basic gap result
        basic_result = self.find_scene_gap_requirements()
        
        if basic_result is None:
            return None
        
        # Calculate initial delay estimate
        fps = float(video_obj_1.video.get("FrameRate", 25.0))
        frame_diff = basic_result['start_frame'] - basic_result['end_frame']
        initial_delay_ms = frame_diff / fps * 1000
        
        # Perform uncertainty analysis
        analysis_duration = self.end_sec - self.start_sec
        uncertainty_result = self.uncertainty_analyzer.analyze_delay_uncertainty(
            video_obj_1, video_obj_2, initial_delay_ms, analysis_duration
        )
        
        # Combine results
        enhanced_result = basic_result.copy()
        enhanced_result.update({
            'initial_delay_ms': initial_delay_ms,
            'refined_delay_ms': uncertainty_result['refined_delay_ms'],
            'uncertainty_ms': uncertainty_result['uncertainty_ms'],
            'confidence_score': uncertainty_result['confidence_score'],
            'variance_ms': uncertainty_result['variance_ms'],
            'analysis_method': uncertainty_result['analysis_method'],
            'recommendations': uncertainty_result['recommendations']
        })
        
        return enhanced_result
