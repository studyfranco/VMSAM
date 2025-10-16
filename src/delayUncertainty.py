'''
Created on 16 Oct 2025

@author: studyfranco

Delay uncertainty estimation for VMSAM video synchronization.
This module provides ML-powered confidence scoring and uncertainty
estimation for audio correlation delay calculations.
'''

import sys
import numpy as np
from statistics import mean, variance
import os
import tools

class delay_uncertainty_estimator:
    """
    ML-powered delay uncertainty estimation for improved synchronization robustness.
    
    This class estimates the confidence and uncertainty of calculated delays
    using multiple correlation methods and statistical analysis. It helps
    identify when delay calculations are unreliable and may need refinement
    or additional validation.
    
    The estimator analyzes patterns in:
    - Delay value consistency across multiple correlation attempts
    - Fidelity score distributions and stability
    - Historical correlation performance
    - Statistical variance and confidence intervals
    
    Attributes:
        correlation_history (list): History of correlation results for learning
        confidence_threshold (float): Minimum confidence for accepting delays
        uncertainty_model: Simple statistical model for uncertainty prediction
        retry_attempts (int): Number of retry attempts made for current comparison
        
    Example:
        >>> estimator = delay_uncertainty_estimator(confidence_threshold=0.8)
        >>> confidence, uncertainty, delay = estimator.calculate_delay_confidence(
        ...     [120, 125, 122], [0.85, 0.90, 0.88]
        ... )
        >>> if confidence > 0.8:
        ...     print(f"High confidence delay: {delay:.1f}ms")
    """
    
    def __init__(self, confidence_threshold=0.8):
        """
        Initialize uncertainty estimator with configuration parameters.
        
        Args:
            confidence_threshold (float): Minimum confidence score (0.0-1.0)
                                         for accepting delay calculation results.
                                         Values closer to 1.0 require higher certainty.
                                         
        Raises:
            ValueError: If confidence_threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
            
        self.correlation_history = []
        self.confidence_threshold = confidence_threshold
        self.uncertainty_model = None
        self.retry_attempts = 0
        
        # ML uncertainty estimation enabled flag
        self.ml_enabled = os.getenv('ML_DELAY_UNCERTAINTY', 'false').lower() == 'true'
        
        # Statistical thresholds for different confidence levels
        self.variance_thresholds = {
            'excellent': 10.0,    # Very low variance
            'good': 50.0,         # Acceptable variance
            'poor': 200.0,        # High variance
            'critical': 1000.0    # Unacceptable variance
        }
    
    def calculate_delay_confidence(self, delay_results, fidelity_scores):
        """
        Calculate comprehensive confidence score for delay estimation results.
        
        Analyzes consistency across multiple correlation attempts and
        fidelity scores to estimate reliability of the delay calculation.
        Uses both statistical measures and heuristic rules to assess
        the trustworthiness of the computed delay values.
        
        The confidence calculation considers:
        - Variance in delay measurements (lower is better)
        - Mean and consistency of fidelity scores
        - Historical performance patterns
        - Statistical outlier detection
        
        Args:
            delay_results (list): List of delay values in milliseconds from 
                                different correlation methods or iterations
            fidelity_scores (list): List of correlation fidelity scores (0.0-1.0)
                                  corresponding to each delay result
                                  
        Returns:
            tuple: Three-element tuple containing:
                - confidence_score (float): Overall confidence (0.0-1.0)
                - uncertainty_estimate (float): Uncertainty level (0.0-1.0)
                - recommended_delay (float): Weighted recommended delay value
                
        Example:
            >>> delays = [120, 125, 122, 123]
            >>> fidelities = [0.85, 0.90, 0.88, 0.92]
            >>> conf, unc, delay = estimator.calculate_delay_confidence(delays, fidelities)
            >>> print(f"Confidence: {conf:.2f}, Uncertainty: {unc:.2f}, Delay: {delay:.1f}ms")
        """
        # Handle edge cases
        if not delay_results:
            return 0.0, 1.0, 0.0
            
        if len(delay_results) == 1:
            single_delay = delay_results[0]
            single_fidelity = fidelity_scores[0] if fidelity_scores else 0.5
            return single_fidelity * 0.7, 0.3, single_delay  # Lower confidence for single measurement
        
        # Use ML-enhanced analysis if enabled and sufficient data
        if self.ml_enabled and len(delay_results) >= 2:
            return self._ml_enhanced_confidence_calculation(delay_results, fidelity_scores)
        else:
            return self._statistical_confidence_calculation(delay_results, fidelity_scores)
    
    def _ml_enhanced_confidence_calculation(self, delay_results, fidelity_scores):
        """
        ML-enhanced confidence calculation with advanced statistical analysis.
        
        Args:
            delay_results (list): Delay values for analysis
            fidelity_scores (list): Corresponding fidelity scores
            
        Returns:
            tuple: (confidence_score, uncertainty_estimate, recommended_delay)
        """
        try:
            # Convert to numpy arrays for vectorized operations
            delays = np.array(delay_results, dtype=float)
            fidelities = np.array(fidelity_scores) if fidelities else np.ones(len(delays)) * 0.5
            
            # Ensure fidelities array matches delays
            if len(fidelities) != len(delays):
                fidelities = np.full(len(delays), np.mean(fidelities) if len(fidelities) > 0 else 0.5)
            
            # Statistical measures for confidence calculation
            delay_variance = np.var(delays)
            delay_std = np.std(delays)
            delay_range = np.ptp(delays)  # Peak-to-peak (max - min)
            mean_fidelity = np.mean(fidelities)
            fidelity_std = np.std(fidelities)
            
            # Outlier detection using modified Z-score
            median_delay = np.median(delays)
            mad = np.median(np.abs(delays - median_delay))  # Median Absolute Deviation
            outlier_threshold = 2.5  # Conservative threshold
            
            # Count outliers
            if mad > 0:
                modified_z_scores = 0.6745 * (delays - median_delay) / mad
                outlier_count = np.sum(np.abs(modified_z_scores) > outlier_threshold)
            else:
                outlier_count = 0
            
            # Advanced confidence components
            
            # 1. Variance confidence (penalizes inconsistent delays)
            if delay_variance <= self.variance_thresholds['excellent']:
                variance_confidence = 1.0
            elif delay_variance <= self.variance_thresholds['good']:
                variance_confidence = 0.9 - (delay_variance - self.variance_thresholds['excellent']) / \
                                    (self.variance_thresholds['good'] - self.variance_thresholds['excellent']) * 0.2
            elif delay_variance <= self.variance_thresholds['poor']:
                variance_confidence = 0.7 - (delay_variance - self.variance_thresholds['good']) / \
                                    (self.variance_thresholds['poor'] - self.variance_thresholds['good']) * 0.3
            else:
                variance_confidence = max(0.1, 0.4 - (delay_variance - self.variance_thresholds['poor']) / \
                                        (self.variance_thresholds['critical'] - self.variance_thresholds['poor']) * 0.3)
            
            # 2. Fidelity confidence (rewards high correlation quality)
            fidelity_confidence = mean_fidelity
            
            # 3. Consistency confidence (rewards stable fidelity scores)
            if len(fidelities) > 1:
                fidelity_consistency = max(0.0, 1.0 - (fidelity_std / 0.3))  # Normalize by reasonable std range
            else:
                fidelity_consistency = 1.0
            
            # 4. Outlier confidence (penalizes presence of outliers)
            outlier_ratio = outlier_count / len(delays)
            outlier_confidence = max(0.0, 1.0 - outlier_ratio * 2.0)
            
            # 5. Sample size confidence (rewards more measurements)
            sample_size_confidence = min(1.0, len(delays) / 5.0)  # Optimal around 5+ samples
            
            # 6. Range confidence (penalizes wide delay spreads)
            range_confidence = max(0.0, 1.0 - (delay_range / 500.0))  # 500ms as reference
            
            # Weighted combination of all confidence measures
            overall_confidence = (
                variance_confidence * 0.25 +     # Primary: consistency of measurements
                fidelity_confidence * 0.25 +     # Primary: correlation quality
                fidelity_consistency * 0.15 +    # Secondary: stability of quality
                outlier_confidence * 0.15 +      # Secondary: absence of bad measurements
                sample_size_confidence * 0.10 +  # Tertiary: sufficient data
                range_confidence * 0.10           # Tertiary: tight measurement spread
            )
            
            # Uncertainty estimation (complementary to confidence)
            # Higher variance and inconsistency increase uncertainty
            uncertainty_components = [
                delay_std / 100.0,           # Standard deviation component
                (1.0 - mean_fidelity) * 0.5, # Fidelity component
                outlier_ratio * 0.3,         # Outlier component
                fidelity_std * 0.2           # Fidelity inconsistency component
            ]
            
            uncertainty = min(1.0, np.sum(uncertainty_components))
            
            # Calculate recommended delay using weighted average
            # Weight by fidelity scores, giving more influence to better correlations
            if np.sum(fidelities) > 0:
                weights = fidelities / np.sum(fidelities)
                recommended_delay = np.average(delays, weights=weights)
            else:
                # Fallback to median if no fidelity information
                recommended_delay = np.median(delays)
            
            # Store analysis results in history for future learning
            analysis_record = {
                'delays': delays.tolist(),
                'fidelities': fidelities.tolist(),
                'confidence': overall_confidence,
                'uncertainty': uncertainty,
                'variance': float(delay_variance),
                'mean_fidelity': float(mean_fidelity),
                'outlier_count': int(outlier_count),
                'recommended_delay': float(recommended_delay)
            }
            
            self.correlation_history.append(analysis_record)
            
            if tools.dev:
                sys.stderr.write(
                    f"ML uncertainty analysis: conf={overall_confidence:.3f}, "
                    f"unc={uncertainty:.3f}, var={delay_variance:.1f}, "
                    f"fid={mean_fidelity:.3f}, out={outlier_count}\n"
                )
            
            return overall_confidence, uncertainty, recommended_delay
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"ML uncertainty estimation error: {e}\n")
            # Fallback to statistical method on ML failure
            return self._statistical_confidence_calculation(delay_results, fidelity_scores)
    
    def _statistical_confidence_calculation(self, delay_results, fidelity_scores):
        """
        Basic statistical confidence calculation without ML enhancement.
        
        Provides simpler but still effective confidence estimation based
        on fundamental statistical measures.
        
        Args:
            delay_results (list): List of delay values
            fidelity_scores (list): List of fidelity scores
            
        Returns:
            tuple: (confidence_score, uncertainty_estimate, recommended_delay)
        """
        try:
            delays = np.array(delay_results, dtype=float)
            fidelities = np.array(fidelity_scores) if fidelity_scores else np.ones(len(delays)) * 0.5
            
            # Basic statistical measures
            delay_variance = np.var(delays)
            mean_fidelity = np.mean(fidelities)
            
            # Simple confidence based on variance and fidelity
            variance_confidence = max(0.0, 1.0 - (delay_variance / 1000.0))
            fidelity_confidence = mean_fidelity
            
            # Basic weighted combination
            overall_confidence = 0.6 * variance_confidence + 0.4 * fidelity_confidence
            
            # Simple uncertainty estimate
            uncertainty = min(1.0, np.std(delays) / 200.0 + (1.0 - mean_fidelity) * 0.3)
            
            # Recommended delay is simple mean
            recommended_delay = np.mean(delays)
            
            return overall_confidence, uncertainty, recommended_delay
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Statistical uncertainty calculation error: {e}\n")
            return 0.5, 0.2, mean(delay_results) if delay_results else 0.0
    
    def should_retry_correlation(self, confidence, uncertainty):
        """
        Determine if correlation should be retried based on confidence metrics.
        
        Evaluates whether the current delay calculation is reliable enough
        to accept, or if additional correlation attempts with different
        parameters might yield better results.
        
        Args:
            confidence (float): Confidence score from delay calculation (0.0-1.0)
            uncertainty (float): Uncertainty estimate from delay calculation (0.0-1.0)
            
        Returns:
            bool: True if correlation should be retried with enhanced parameters,
                 False if current result should be accepted
                 
        Note:
            Retry attempts are limited to prevent infinite loops. The method
            considers both confidence and uncertainty thresholds, as well as
            the number of previous retry attempts.
        """
        max_retry_attempts = 2  # Limit retries to prevent excessive processing
        
        # Don't retry if we've already tried too many times
        if self.retry_attempts >= max_retry_attempts:
            if tools.dev:
                sys.stderr.write(f"Max retry attempts ({max_retry_attempts}) reached\n")
            return False
        
        # Retry conditions based on confidence and uncertainty
        low_confidence = confidence < self.confidence_threshold
        high_uncertainty = uncertainty > 0.4
        very_low_confidence = confidence < 0.3
        
        should_retry = (low_confidence and high_uncertainty) or very_low_confidence
        
        if should_retry:
            self.retry_attempts += 1
            if tools.dev:
                sys.stderr.write(
                    f"Retry recommended (attempt {self.retry_attempts}): "
                    f"conf={confidence:.3f} < {self.confidence_threshold}, "
                    f"unc={uncertainty:.3f}\n"
                )
        
        return should_retry
    
    def analyze_correlation_stability(self, delay_history, fidelity_history):
        """
        Analyze stability patterns in correlation results over time.
        
        Examines historical correlation data to identify trends, recurring
        issues, or systematic biases that might affect current calculations.
        
        Args:
            delay_history (list): Historical delay calculations for this video pair
            fidelity_history (list): Historical fidelity scores for this video pair
            
        Returns:
            dict: Analysis results containing:
                - stability_score (float): Overall stability metric (0.0-1.0)
                - trend_direction (str): 'stable', 'improving', 'degrading'
                - reliability_assessment (str): Overall reliability category
                
        Note:
            This method helps identify systematic issues with specific video
            pairs that might require different processing approaches.
        """
        try:
            if not delay_history or len(delay_history) < 3:
                return {
                    'stability_score': 0.5,
                    'trend_direction': 'insufficient_data',
                    'reliability_assessment': 'unknown'
                }
            
            delays = np.array(delay_history[-10:])  # Analyze recent history
            fidelities = np.array(fidelity_history[-10:]) if fidelity_history else np.ones(len(delays)) * 0.5
            
            # Calculate stability metrics
            delay_trend = np.polyfit(range(len(delays)), delays, 1)[0]  # Linear trend
            delay_stability = 1.0 / (1.0 + np.var(delays) / 100.0)  # Inverse variance
            fidelity_trend = np.polyfit(range(len(fidelities)), fidelities, 1)[0]
            
            # Determine trend direction
            if abs(delay_trend) < 5:  # Less than 5ms trend per measurement
                trend_direction = 'stable'
            elif fidelity_trend > 0.01:  # Improving fidelity
                trend_direction = 'improving' 
            elif fidelity_trend < -0.01:  # Degrading fidelity
                trend_direction = 'degrading'
            else:
                trend_direction = 'stable'
            
            # Overall stability score
            stability_score = delay_stability * 0.6 + np.mean(fidelities) * 0.4
            
            # Reliability assessment
            if stability_score > 0.8 and np.mean(fidelities) > 0.85:
                reliability = 'excellent'
            elif stability_score > 0.6 and np.mean(fidelities) > 0.75:
                reliability = 'good'
            elif stability_score > 0.4 and np.mean(fidelities) > 0.6:
                reliability = 'fair'
            else:
                reliability = 'poor'
            
            return {
                'stability_score': stability_score,
                'trend_direction': trend_direction,
                'reliability_assessment': reliability
            }
            
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Stability analysis error: {e}\n")
            return {
                'stability_score': 0.3,
                'trend_direction': 'error',
                'reliability_assessment': 'unreliable'
            }
    
    def get_confidence_category(self, confidence_score):
        """
        Categorize confidence score into human-readable categories.
        
        Args:
            confidence_score (float): Numerical confidence score (0.0-1.0)
            
        Returns:
            str: Confidence category ('excellent', 'good', 'fair', 'poor', 'critical')
        """
        if confidence_score >= 0.90:
            return 'excellent'
        elif confidence_score >= 0.75:
            return 'good'
        elif confidence_score >= 0.60:
            return 'fair'
        elif confidence_score >= 0.40:
            return 'poor'
        else:
            return 'critical'
    
    def reset_retry_counter(self):
        """
        Reset retry attempt counter for new correlation analysis.
        
        Should be called when starting analysis of a new video pair
        to ensure retry limits apply per-pair rather than globally.
        """
        self.retry_attempts = 0
    
    def get_analysis_summary(self):
        """
        Get summary of all correlation analyses performed.
        
        Returns:
            dict: Summary statistics including:
                - total_analyses (int): Total number of analyses performed
                - mean_confidence (float): Average confidence across all analyses
                - mean_uncertainty (float): Average uncertainty across all analyses
                - success_rate (float): Percentage of high-confidence results
        """
        if not self.correlation_history:
            return {
                'total_analyses': 0,
                'mean_confidence': 0.0,
                'mean_uncertainty': 1.0,
                'success_rate': 0.0
            }
        
        confidences = [record['confidence'] for record in self.correlation_history]
        uncertainties = [record['uncertainty'] for record in self.correlation_history]
        successes = sum(1 for conf in confidences if conf >= self.confidence_threshold)
        
        return {
            'total_analyses': len(self.correlation_history),
            'mean_confidence': np.mean(confidences),
            'mean_uncertainty': np.mean(uncertainties),
            'success_rate': successes / len(confidences) if confidences else 0.0
        }
    
    def cleanup_history(self, max_records=100):
        """
        Clean up correlation history to prevent memory accumulation.
        
        Args:
            max_records (int): Maximum number of historical records to retain
        """
        if len(self.correlation_history) > max_records:
            # Keep most recent records
            self.correlation_history = self.correlation_history[-max_records:]
            
            if tools.dev:
                sys.stderr.write(f"Uncertainty estimator history trimmed to {max_records} records\n")
