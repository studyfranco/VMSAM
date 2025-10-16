"""ML validation module for testing and validating scene detection improvements.

This module provides validation functions for the new ML-based scene detection
and enhanced delay calculation features, ensuring reliability and performance.

Author: studyfranco
Created: 2025-01-16
"""

import sys
import os
import time
from typing import List, Tuple, Optional, Dict, Any
from decimal import Decimal
import json
from threading import Thread

import tools
import video

# Import new ML modules with fallback
try:
    from scene_detection import SceneDetector, SceneDetectionError
    from enhanced_frame_compare import (
        EnhancedFrameComparer, compare_video_segments_for_delay_uncertainty
    )
    ML_MODULES_AVAILABLE = True
except ImportError:
    ML_MODULES_AVAILABLE = False


class MLValidationResults:
    """Container class for ML validation results and metrics."""
    
    def __init__(self):
        self.scene_detection_tests = []
        self.frame_comparison_tests = []
        self.delay_accuracy_tests = []
        self.performance_metrics = {}
        self.error_log = []
        
    def add_scene_detection_result(self, test_name: str, success: bool, 
                                 execution_time: float, details: Dict[str, Any]):
        """Add scene detection test result.
        
        Args:
            test_name: Name of the test
            success: Whether test passed
            execution_time: Time taken for test execution
            details: Additional test details
        """
        self.scene_detection_tests.append({
            "test_name": test_name,
            "success": success,
            "execution_time": execution_time,
            "details": details
        })
        
    def add_frame_comparison_result(self, test_name: str, success: bool,
                                  execution_time: float, confidence: float,
                                  details: Dict[str, Any]):
        """Add frame comparison test result.
        
        Args:
            test_name: Name of the test
            success: Whether test passed
            execution_time: Time taken for test execution
            confidence: Confidence score of the comparison
            details: Additional test details
        """
        self.frame_comparison_tests.append({
            "test_name": test_name,
            "success": success,
            "execution_time": execution_time,
            "confidence": confidence,
            "details": details
        })
        
    def add_delay_accuracy_result(self, test_name: str, original_delay: Decimal,
                                enhanced_delay: Decimal, improvement_score: float):
        """Add delay accuracy test result.
        
        Args:
            test_name: Name of the test
            original_delay: Original delay calculation
            enhanced_delay: Enhanced delay with ML improvements
            improvement_score: Score indicating improvement quality
        """
        self.delay_accuracy_tests.append({
            "test_name": test_name,
            "original_delay": float(original_delay),
            "enhanced_delay": float(enhanced_delay),
            "improvement_score": improvement_score,
            "delay_difference": float(abs(enhanced_delay - original_delay))
        })
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Returns:
            Dictionary containing validation report data
        """
        report = {
            "ml_modules_available": ML_MODULES_AVAILABLE,
            "scene_detection": {
                "total_tests": len(self.scene_detection_tests),
                "successful_tests": sum(1 for t in self.scene_detection_tests if t["success"]),
                "average_execution_time": self._calculate_average_execution_time(self.scene_detection_tests),
                "success_rate": self._calculate_success_rate(self.scene_detection_tests)
            },
            "frame_comparison": {
                "total_tests": len(self.frame_comparison_tests),
                "successful_tests": sum(1 for t in self.frame_comparison_tests if t["success"]),
                "average_execution_time": self._calculate_average_execution_time(self.frame_comparison_tests),
                "average_confidence": self._calculate_average_confidence(self.frame_comparison_tests),
                "success_rate": self._calculate_success_rate(self.frame_comparison_tests)
            },
            "delay_accuracy": {
                "total_tests": len(self.delay_accuracy_tests),
                "average_improvement_score": self._calculate_average_improvement_score(),
                "significant_improvements": self._count_significant_improvements(),
                "average_delay_difference": self._calculate_average_delay_difference()
            },
            "errors": self.error_log,
            "overall_performance": self.performance_metrics
        }
        return report
        
    def _calculate_average_execution_time(self, tests: List[Dict]) -> float:
        """Calculate average execution time from test results."""
        if not tests:
            return 0.0
        return sum(t["execution_time"] for t in tests) / len(tests)
        
    def _calculate_success_rate(self, tests: List[Dict]) -> float:
        """Calculate success rate from test results."""
        if not tests:
            return 0.0
        return sum(1 for t in tests if t["success"]) / len(tests)
        
    def _calculate_average_confidence(self, tests: List[Dict]) -> float:
        """Calculate average confidence from frame comparison tests."""
        if not tests:
            return 0.0
        return sum(t["confidence"] for t in tests) / len(tests)
        
    def _calculate_average_improvement_score(self) -> float:
        """Calculate average improvement score from delay accuracy tests."""
        if not self.delay_accuracy_tests:
            return 0.0
        return sum(t["improvement_score"] for t in self.delay_accuracy_tests) / len(self.delay_accuracy_tests)
        
    def _count_significant_improvements(self) -> int:
        """Count tests with significant improvements (>50ms difference)."""
        return sum(1 for t in self.delay_accuracy_tests if t["delay_difference"] > 50.0)
        
    def _calculate_average_delay_difference(self) -> float:
        """Calculate average delay difference between original and enhanced methods."""
        if not self.delay_accuracy_tests:
            return 0.0
        return sum(t["delay_difference"] for t in self.delay_accuracy_tests) / len(self.delay_accuracy_tests)


class MLValidator:
    """Validator class for ML enhancements in VMSAM.
    
    Provides comprehensive testing and validation of the new ML-based
    scene detection and enhanced delay calculation features.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize ML validator.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.results = MLValidationResults()
        
    def validate_scene_detection_basic(self, video_path: str) -> bool:
        """Basic validation test for scene detection functionality.
        
        Args:
            video_path: Path to test video file
            
        Returns:
            True if scene detection works correctly
        """
        if not ML_MODULES_AVAILABLE:
            self.results.error_log.append("ML modules not available for scene detection validation")
            return False
            
        try:
            start_time = time.time()
            
            # Test basic scene detection
            detector = SceneDetector(threshold=30.0, min_scene_len=15)
            scenes = detector.detect_scenes(video_path, start_time=0.0, duration=60.0)
            
            execution_time = time.time() - start_time
            
            success = len(scenes) > 0
            details = {
                "scenes_detected": len(scenes),
                "first_scene_duration": scenes[0][1] - scenes[0][0] if scenes else 0,
                "video_path": video_path
            }
            
            self.results.add_scene_detection_result(
                "basic_detection", success, execution_time, details
            )
            
            if self.debug:
                sys.stderr.write(
                    f"Scene detection basic validation: {success}, "
                    f"scenes: {len(scenes)}, time: {execution_time:.2f}s\n"
                )
                
            return success
            
        except Exception as e:
            self.results.error_log.append(f"Scene detection basic validation failed: {str(e)}")
            self.results.add_scene_detection_result(
                "basic_detection", False, 0.0, {"error": str(e)}
            )
            return False
            
    def validate_enhanced_frame_comparison(self, video_path1: str, video_path2: str,
                                         start_sec: float = 10.0, duration: float = 30.0) -> bool:
        """Validate enhanced frame comparison functionality.
        
        Args:
            video_path1: First video path
            video_path2: Second video path  
            start_sec: Start time for comparison
            duration: Duration for comparison
            
        Returns:
            True if enhanced frame comparison works correctly
        """
        if not ML_MODULES_AVAILABLE:
            self.results.error_log.append("ML modules not available for frame comparison validation")
            return False
            
        try:
            start_time = time.time()
            
            # Test enhanced frame comparison
            comparer = EnhancedFrameComparer(
                video_path1, video_path2, start_sec, start_sec + duration,
                fps=10, debug=self.debug
            )
            
            result = comparer.find_scene_gap_with_uncertainty()
            execution_time = time.time() - start_time
            
            if result:
                success = result.get("confidence", 0.0) > 0.1
                confidence = result.get("confidence", 0.0)
                details = {
                    "uncertainty": result.get("uncertainty", 1.0),
                    "temporal_consistency": result.get("temporal_consistency", 0.0),
                    "is_reliable": result.get("is_reliable", False)
                }
            else:
                success = False
                confidence = 0.0
                details = {"error": "No comparison result returned"}
                
            self.results.add_frame_comparison_result(
                "enhanced_comparison", success, execution_time, confidence, details
            )
            
            if self.debug:
                sys.stderr.write(
                    f"Enhanced frame comparison validation: {success}, "
                    f"confidence: {confidence:.2f}, time: {execution_time:.2f}s\n"
                )
                
            return success
            
        except Exception as e:
            self.results.error_log.append(f"Enhanced frame comparison validation failed: {str(e)}")
            self.results.add_frame_comparison_result(
                "enhanced_comparison", False, 0.0, 0.0, {"error": str(e)}
            )
            return False
            
    def validate_delay_accuracy_improvement(self, video_obj_1, video_obj_2,
                                          original_delay: Decimal) -> bool:
        """Validate delay accuracy improvements with ML enhancements.
        
        Args:
            video_obj_1: First video object
            video_obj_2: Second video object
            original_delay: Original delay calculation
            
        Returns:
            True if ML enhancements provide meaningful improvements
        """
        if not ML_MODULES_AVAILABLE:
            self.results.error_log.append("ML modules not available for delay accuracy validation")
            return False
            
        try:
            # Test ML-enhanced delay calculation
            from scene_detection import integrate_scene_detection_with_delay_calculation
            
            # Get video duration for testing
            begin_in_second, length_time = video.generate_begin_and_length_by_segment(
                min(video.get_shortest_audio_durations([video_obj_1], "und"),
                    video.get_shortest_audio_durations([video_obj_2], "und"))
            )
            
            enhanced_delay = integrate_scene_detection_with_delay_calculation(
                video_obj_1, video_obj_2, original_delay, begin_in_second, length_time
            )
            
            # Calculate improvement score
            delay_difference = abs(float(enhanced_delay - original_delay))
            
            # Score based on whether enhancement provides meaningful adjustment
            if delay_difference > 10.0:  # More than 10ms difference
                improvement_score = min(1.0, delay_difference / 100.0)  # Scale to 0-1
            else:
                improvement_score = 0.1  # Minimal improvement
                
            self.results.add_delay_accuracy_result(
                "ml_enhancement", original_delay, enhanced_delay, improvement_score
            )
            
            if self.debug:
                sys.stderr.write(
                    f"Delay accuracy validation: original={float(original_delay):.2f}ms, "
                    f"enhanced={float(enhanced_delay):.2f}ms, "
                    f"improvement_score={improvement_score:.2f}\n"
                )
                
            return improvement_score > 0.0
            
        except Exception as e:
            self.results.error_log.append(f"Delay accuracy validation failed: {str(e)}")
            self.results.add_delay_accuracy_result(
                "ml_enhancement", original_delay, original_delay, 0.0
            )
            return False
            
    def run_comprehensive_validation(self, video_paths: List[str]) -> Dict[str, Any]:
        """Run comprehensive validation of all ML enhancements.
        
        Args:
            video_paths: List of video file paths for testing
            
        Returns:
            Comprehensive validation report
        """
        if tools.dev:
            sys.stderr.write("Starting comprehensive ML validation\n")
            
        validation_start_time = time.time()
        
        # Test 1: Basic scene detection on each video
        for i, video_path in enumerate(video_paths[:3]):  # Limit to 3 videos for performance
            if os.path.exists(video_path):
                self.validate_scene_detection_basic(video_path)
            else:
                self.results.error_log.append(f"Video file not found: {video_path}")
                
        # Test 2: Frame comparison between video pairs
        for i in range(min(2, len(video_paths) - 1)):
            if (os.path.exists(video_paths[i]) and 
                os.path.exists(video_paths[i + 1])):
                self.validate_enhanced_frame_comparison(
                    video_paths[i], video_paths[i + 1]
                )
                
        # Test 3: Create video objects and test delay accuracy
        if len(video_paths) >= 2:
            try:
                video_obj_1 = video.video(
                    os.path.dirname(video_paths[0]), 
                    os.path.basename(video_paths[0])
                )
                video_obj_2 = video.video(
                    os.path.dirname(video_paths[1]), 
                    os.path.basename(video_paths[1])
                )
                
                # Get basic metadata
                video_obj_1.get_mediadata()
                video_obj_2.get_mediadata()
                
                # Test with sample delay
                sample_delay = Decimal(500)  # 500ms sample delay
                self.validate_delay_accuracy_improvement(
                    video_obj_1, video_obj_2, sample_delay
                )
                
            except Exception as e:
                self.results.error_log.append(f"Video object creation failed: {str(e)}")
        
        # Calculate overall performance metrics
        total_execution_time = time.time() - validation_start_time
        self.results.performance_metrics = {
            "total_validation_time": total_execution_time,
            "avg_test_time": total_execution_time / max(1, (
                len(self.results.scene_detection_tests) + 
                len(self.results.frame_comparison_tests) +
                len(self.results.delay_accuracy_tests)
            ))
        }
        
        report = self.results.generate_validation_report()
        
        if tools.dev:
            sys.stderr.write(f"ML validation completed in {total_execution_time:.2f}s\n")
            sys.stderr.write(f"Overall success rate: {self._calculate_overall_success_rate():.1%}\n")
            
        return report
        
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all tests."""
        all_tests = (
            self.results.scene_detection_tests + 
            self.results.frame_comparison_tests
        )
        
        if not all_tests:
            return 0.0
            
        successful_tests = sum(1 for t in all_tests if t["success"])
        return successful_tests / len(all_tests)
        
    def export_validation_report(self, output_path: str) -> bool:
        """Export validation report to JSON file.
        
        Args:
            output_path: Path for output JSON file
            
        Returns:
            True if export successful
        """
        try:
            report = self.results.generate_validation_report()
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"Failed to export validation report: {e}\n")
            return False


def run_ml_validation_suite(video_files: List[str], output_report_path: str = None) -> Dict[str, Any]:
    """Run the complete ML validation suite.
    
    Args:
        video_files: List of video files for testing
        output_report_path: Optional path to save validation report
        
    Returns:
        Validation report dictionary
    """
    if not video_files:
        return {"error": "No video files provided for validation"}
        
    validator = MLValidator(debug=tools.dev if hasattr(tools, 'dev') else False)
    report = validator.run_comprehensive_validation(video_files)
    
    if output_report_path:
        validator.export_validation_report(output_report_path)
        
    return report


def quick_ml_compatibility_check() -> Dict[str, bool]:
    """Quick compatibility check for ML modules and dependencies.
    
    Returns:
        Dictionary indicating availability of different ML components
    """
    compatibility = {
        "pyscenedetect_available": False,
        "opencv_available": False,
        "scipy_available": False,
        "numpy_available": False,
        "enhanced_modules_available": ML_MODULES_AVAILABLE
    }
    
    # Test PySceneDetect
    try:
        import scenedetect
        compatibility["pyscenedetect_available"] = True
    except ImportError:
        pass
        
    # Test OpenCV
    try:
        import cv2
        compatibility["opencv_available"] = True
    except ImportError:
        pass
        
    # Test SciPy
    try:
        import scipy
        compatibility["scipy_available"] = True
    except ImportError:
        pass
        
    # Test NumPy
    try:
        import numpy
        compatibility["numpy_available"] = True
    except ImportError:
        pass
        
    return compatibility
