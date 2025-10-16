#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Video Merger for VMSAM

This module extends the original mergeVideo.py with advanced scene detection
and improved delay detection capabilities. It integrates ML-based scene detection
and enhanced frame comparison for better synchronization accuracy.

Author: VMSAM Enhancement Team  
Compatible with: Python 3.7+
Dependencies: All original mergeVideo.py dependencies plus scene_detector, enhanced_frame_compare
"""

# Import all original functionality
from mergeVideo import *

# Import new enhanced modules
try:
    from scene_detector import (
        get_scene_detector, SceneAwareDelayDetector, 
        clear_scene_cache, SCENEDETECT_AVAILABLE
    )
    from enhanced_frame_compare import (
        create_frame_comparer, EnhancedFrameComparer,
        clear_frame_cache, CV2_AVAILABLE
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    if tools.dev:
        sys.stderr.write(f"Warning: Enhanced features not available: {e}\n")

class enhanced_compare_video(compare_video):
    """
    Enhanced video comparison class that extends the original compare_video
    with scene detection and improved frame comparison capabilities.
    """
    
    def __init__(self, video_obj_1, video_obj_2, begin_in_second, audioParam, 
                 language, lenghtTime, lenghtTimePrepare, list_cut_begin_length, 
                 time_by_test_best_quality_converted, process_to_get_best_video=True,
                 enable_scene_detection=True, enable_enhanced_frame_compare=True):
        """
        Initialize enhanced video comparison.
        
        Args:
            All original compare_video arguments plus:
            enable_scene_detection: Whether to use ML scene detection
            enable_enhanced_frame_compare: Whether to use enhanced frame comparison
        """
        # Initialize parent class
        super().__init__(video_obj_1, video_obj_2, begin_in_second, audioParam,
                        language, lenghtTime, lenghtTimePrepare, list_cut_begin_length,
                        time_by_test_best_quality_converted, process_to_get_best_video)
        
        # Enhanced features configuration
        self.enable_scene_detection = (enable_scene_detection and 
                                     ENHANCED_FEATURES_AVAILABLE and 
                                     SCENEDETECT_AVAILABLE)
        self.enable_enhanced_frame_compare = (enable_enhanced_frame_compare and 
                                            ENHANCED_FEATURES_AVAILABLE and 
                                            CV2_AVAILABLE)
        
        # Initialize enhanced components if available
        if self.enable_scene_detection:
            try:
                self.scene_detector = get_scene_detector(adaptive=True)
                self.scene_delay_detector = SceneAwareDelayDetector(self.scene_detector)
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tScene detection initialization failed: {e}\n")
                self.enable_scene_detection = False
                
        if self.enable_enhanced_frame_compare:
            try:
                self.frame_comparer = create_frame_comparer("phash")
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tEnhanced frame comparison initialization failed: {e}\n")
                self.enable_enhanced_frame_compare = False
                
        # Delay uncertainty tracking
        self.delay_uncertainty = 0.0
        self.delay_validation_results = []
        
    def adjuster_chroma_bugged_enhanced(self, list_delay, ignore_audio_couple):
        """
        Enhanced version of adjuster_chroma_bugged with scene detection validation.
        
        Args:
            list_delay: List of delay values to test
            ignore_audio_couple: Audio couples to ignore
            
        Returns:
            Validated delay value or None if validation fails
        """
        if tools.dev:
            sys.stderr.write(f"\t\tStarting enhanced chroma adjustment for delays {list_delay}\n")
            
        # Use original method as baseline
        original_result = self.adjuster_chroma_bugged(list_delay, ignore_audio_couple)
        
        if not self.enable_scene_detection or original_result is None:
            return original_result
            
        try:
            # Validate delay using scene detection
            is_valid, scene_confidence = self.scene_delay_detector.validate_delay_with_scenes(
                self.video_obj_1.filePath, 
                self.video_obj_2.filePath,
                original_result / 1000.0  # Convert to seconds
            )
            
            if is_valid:
                if tools.dev:
                    sys.stderr.write(f"\t\tScene validation passed with confidence {scene_confidence:.2f}\n")
                self.delay_uncertainty = max(0.0, 1.0 - scene_confidence)
                return original_result
            else:
                if tools.dev:
                    sys.stderr.write(f"\t\tScene validation failed, trying alternative delays\n")
                    
                # Try alternative delays based on scene boundaries
                for delay_candidate in list_delay:
                    is_valid, confidence = self.scene_delay_detector.validate_delay_with_scenes(
                        self.video_obj_1.filePath,
                        self.video_obj_2.filePath, 
                        delay_candidate / 1000.0
                    )
                    if is_valid and confidence > 0.5:
                        if tools.dev:
                            sys.stderr.write(f"\t\tAlternative delay {delay_candidate} validated\n")
                        self.delay_uncertainty = 1.0 - confidence
                        return delay_candidate
                        
                # If no scene validation passes, fall back to original result
                self.delay_uncertainty = 0.8  # High uncertainty
                return original_result
                
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"\t\tScene validation error: {e}\n")
            self.delay_uncertainty = 0.5  # Medium uncertainty
            return original_result
            
    def adjust_delay_to_frame_enhanced(self, delay):
        """
        Enhanced frame adjustment with scene-aware timing and frame comparison.
        
        Args:
            delay: Delay value in milliseconds
            
        Returns:
            Frame-adjusted delay value
        """
        if tools.dev:
            sys.stderr.write(f"\t\tStarting enhanced frame adjustment for delay {delay}ms\n")
            
        # Get frame rate compatibility
        video1_fps = None
        video2_fps = None
        
        try:
            if "FrameRate" in self.video_obj_1.video:
                video1_fps = float(self.video_obj_1.video["FrameRate"])
            if "FrameRate" in self.video_obj_2.video:
                video2_fps = float(self.video_obj_2.video["FrameRate"])
        except (KeyError, ValueError):
            pass
            
        # Check if frame rates are compatible for enhanced comparison
        fps_compatible = (video1_fps is not None and video2_fps is not None and 
                         abs(video1_fps - video2_fps) < 0.1)
        
        if fps_compatible and self.enable_enhanced_frame_compare:
            try:
                # Use enhanced frame comparison for validation
                validated_delay = self._validate_delay_with_frame_comparison(delay)
                if validated_delay is not None:
                    delay = validated_delay
                    if tools.dev:
                        sys.stderr.write(f"\t\tFrame comparison validated delay: {delay}ms\n")
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tFrame comparison validation failed: {e}\n")
                    
        # Apply original frame adjustment
        return self.adjust_delay_to_frame(delay)
        
    def _validate_delay_with_frame_comparison(self, delay_ms):
        """
        Validate delay using enhanced frame comparison at scene boundaries.
        
        Args:
            delay_ms: Delay in milliseconds
            
        Returns:
            Validated delay or None if validation fails
        """
        if not self.enable_scene_detection or not self.enable_enhanced_frame_compare:
            return None
            
        try:
            # Get optimal comparison points using scene detection
            video_duration = float(self.video_obj_1.video.get("Duration", 0))
            if video_duration <= 0:
                return None
                
            comparison_points = self.scene_delay_detector.get_optimal_comparison_points(
                self.video_obj_1.filePath, video_duration, num_points=3
            )
            
            if not comparison_points:
                return None
                
            # Perform frame comparison at each point
            similarities = []
            for start_time, end_time in comparison_points:
                # Adjust timing for video 2 based on delay
                adjusted_start = start_time + (delay_ms / 1000.0)
                adjusted_end = end_time + (delay_ms / 1000.0)
                
                # Skip if adjusted times are out of bounds
                if adjusted_start < 0 or adjusted_end > video_duration:
                    continue
                    
                # Compare frames
                time_ranges = [(start_time, end_time)]
                results = self.frame_comparer.compare_frames_from_video(
                    self.video_obj_1.filePath,
                    self.video_obj_2.filePath,
                    time_ranges,
                    fps=5.0
                )
                
                if results and results[0].is_reliable():
                    similarities.append(results[0].similarity)
                    
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                if avg_similarity >= 0.75:  # Good match threshold
                    if tools.dev:
                        sys.stderr.write(f"\t\tFrame comparison validation: avg similarity {avg_similarity:.2f}\n")
                    return delay_ms
                    
        except Exception as e:
            if tools.dev:
                sys.stderr.write(f"\t\tFrame comparison validation error: {e}\n")
                
        return None
        
    def estimate_delay_uncertainty(self):
        """
        Estimate uncertainty in the detected delay based on multiple factors.
        
        Returns:
            Uncertainty score [0.0-1.0] where 0 is most certain
        """
        uncertainty_factors = []
        
        # Base uncertainty from delay detection process
        if hasattr(self, 'delay_uncertainty'):
            uncertainty_factors.append(self.delay_uncertainty)
        else:
            uncertainty_factors.append(0.3)  # Default moderate uncertainty
            
        # Scene detection confidence
        if self.enable_scene_detection and self.delay_validation_results:
            scene_uncertainties = [1.0 - result[1] for result in self.delay_validation_results]
            if scene_uncertainties:
                uncertainty_factors.append(sum(scene_uncertainties) / len(scene_uncertainties))
                
        # Video quality factors
        try:
            video1_bitrate = float(self.video_obj_1.video.get("BitRate", 0))
            video2_bitrate = float(self.video_obj_2.video.get("BitRate", 0))
            
            # Lower bitrate videos have higher uncertainty
            min_bitrate = min(video1_bitrate, video2_bitrate)
            if min_bitrate > 0:
                # Normalize bitrate uncertainty (assume 1Mbps = low quality)
                bitrate_uncertainty = max(0.0, min(0.5, 1.0 - min_bitrate / 1000000))
                uncertainty_factors.append(bitrate_uncertainty)
        except (ValueError, KeyError):
            uncertainty_factors.append(0.2)  # Default bitrate uncertainty
            
        # Frame rate compatibility
        try:
            fps1 = float(self.video_obj_1.video.get("FrameRate", 0))
            fps2 = float(self.video_obj_2.video.get("FrameRate", 0))
            
            if fps1 > 0 and fps2 > 0:
                fps_diff = abs(fps1 - fps2)
                fps_uncertainty = min(0.4, fps_diff / max(fps1, fps2))
                uncertainty_factors.append(fps_uncertainty)
        except (ValueError, KeyError):
            uncertainty_factors.append(0.1)  # Default FPS uncertainty
            
        # Calculate weighted average uncertainty
        if uncertainty_factors:
            return min(1.0, sum(uncertainty_factors) / len(uncertainty_factors))
        else:
            return 0.5  # Default medium uncertainty
            
    def get_enhanced_comparison_report(self):
        """
        Generate a detailed report of the enhanced comparison process.
        
        Returns:
            Dictionary with comparison metrics and validation results
        """
        report = {
            "scene_detection_enabled": self.enable_scene_detection,
            "frame_comparison_enabled": self.enable_enhanced_frame_compare,
            "delay_uncertainty": self.estimate_delay_uncertainty(),
            "validation_results": self.delay_validation_results.copy() if self.delay_validation_results else [],
            "video1_info": {
                "path": self.video_obj_1.filePath,
                "framerate": self.video_obj_1.video.get("FrameRate", "unknown"),
                "duration": self.video_obj_1.video.get("Duration", "unknown")
            },
            "video2_info": {
                "path": self.video_obj_2.filePath,
                "framerate": self.video_obj_2.video.get("FrameRate", "unknown"), 
                "duration": self.video_obj_2.video.get("Duration", "unknown")
            }
        }
        
        return report
        
    def run(self):
        """
        Enhanced run method that includes scene detection and uncertainty estimation.
        """
        try:
            # Store original methods to allow fallback
            original_adjuster = self.adjuster_chroma_bugged
            original_frame_adjuster = self.adjust_delay_to_frame
            
            # Replace with enhanced methods if available
            if self.enable_scene_detection:
                self.adjuster_chroma_bugged = self.adjuster_chroma_bugged_enhanced
                
            if self.enable_enhanced_frame_compare:
                self.adjust_delay_to_frame = self.adjust_delay_to_frame_enhanced
                
            # Run the original comparison logic
            delay = self.test_if_constant_good_delay()
            
            if self.process_to_get_best_video:
                self.get_best_video(delay)
            else:
                self.video_obj_1.extract_audio_in_part(
                    self.language, self.audioParam, 
                    cutTime=self.list_cut_begin_length, asDefault=True
                )
                self.video_obj_2.remove_tmp_files(type_file="audio")
                self.video_obj_with_best_quality = self.video_obj_1
                delay = self.adjust_delay_to_frame(delay)
                self.video_obj_2.delays[self.language] += (delay * -Decimal(1.0))
                
            # Log enhanced comparison results
            if tools.dev and ENHANCED_FEATURES_AVAILABLE:
                uncertainty = self.estimate_delay_uncertainty()
                sys.stderr.write(
                    f"\t\tEnhanced comparison completed. "
                    f"Delay: {delay}ms, Uncertainty: {uncertainty:.2f}\n"
                )
                
        except Exception as e:
            # Restore original methods in case of error
            if hasattr(self, 'original_adjuster'):
                self.adjuster_chroma_bugged = original_adjuster
            if hasattr(self, 'original_frame_adjuster'):
                self.adjust_delay_to_frame = original_frame_adjuster
                
            traceback.print_exc()
            sys.stderr.write(str(e) + "\n")
            with errors_merge_lock:
                errors_merge.append(str(e))
                
def enhanced_get_delay_and_best_video(videosObj, language, audioRules, dict_file_path_obj):
    """
    Enhanced version of get_delay_and_best_video with scene detection support.
    
    Args:
        videosObj: List of video objects
        language: Common language for comparison
        audioRules: Audio comparison rules
        dict_file_path_obj: Dictionary mapping file paths to video objects
        
    Returns:
        Dictionary with comparison results
    """
    if tools.dev:
        sys.stderr.write("\t\tStarting enhanced delay and quality detection\n")
        
    # Use enhanced preparation if scene detection is available
    if ENHANCED_FEATURES_AVAILABLE and SCENEDETECT_AVAILABLE:
        begin_in_second, audio_param, length_time, length_time_converted, list_cut_begin_length = \
            prepare_get_delay_enhanced(videosObj, language, audioRules)
    else:
        begin_in_second, audio_param, length_time, length_time_converted, list_cut_begin_length = \
            prepare_get_delay(videosObj, language, audioRules)
    
    time_by_test_best_quality_converted = strftime('%H:%M:%S', gmtime(
        video.generate_time_compare_video_quality(length_time)
    ))
    
    compareObjs = videosObj.copy()
    already_compared = {}
    list_not_compatible_video = []
    
    while len(compareObjs) > 1:
        if len(compareObjs) % 2 != 0:
            new_compare_objs = [compareObjs.pop()]
        else:
            new_compare_objs = []
            
        list_in_compare_video = []
        
        for i in range(0, len(compareObjs), 2):
            if was_they_not_already_compared(compareObjs[i], compareObjs[i + 1], already_compared):
                # Use enhanced comparison class
                comparer = enhanced_compare_video(
                    compareObjs[i], compareObjs[i + 1], begin_in_second,
                    audio_param, language, length_time, length_time_converted,
                    list_cut_begin_length, time_by_test_best_quality_converted,
                    enable_scene_detection=True,
                    enable_enhanced_frame_compare=True
                )
                comparer.start()
                list_in_compare_video.append(comparer)
            elif len(new_compare_objs):
                # Handle remaining comparisons using enhanced logic
                handle_remaining_comparisons_enhanced(
                    compareObjs, i, new_compare_objs, already_compared,
                    begin_in_second, audio_param, language, length_time,
                    length_time_converted, list_cut_begin_length,
                    time_by_test_best_quality_converted, list_in_compare_video,
                    list_not_compatible_video
                )
            else:
                # Handle incompatible videos
                if tools.dev:
                    sys.stderr.write(
                        "You enter in a not working part. You have one last file "
                        "not compatible you may stop here the result will be random\n"
                    )
                list_not_compatible_video.append(compareObjs[i + 1].filePath)
                list_not_compatible_video.extend(
                    remove_not_compatible_audio(compareObjs[i + 1].filePath, already_compared)
                )
        
        # Process comparison results
        compareObjs = new_compare_objs
        for compare_video_obj in list_in_compare_video:
            nameInList = [compare_video_obj.video_obj_1.filePath, compare_video_obj.video_obj_2.filePath]
            sorted(nameInList)
            compare_video_obj.join()
            
            if compare_video_obj.video_obj_with_best_quality is not None:
                is_the_best_video = compare_video_obj.video_obj_with_best_quality.filePath == nameInList[0]
                compareObjs.append(compare_video_obj.video_obj_with_best_quality)
            else:
                is_the_best_video = None
                compareObjs.append(compare_video_obj.video_obj_1)
                compareObjs.append(compare_video_obj.video_obj_2)
                
            if nameInList[0] in already_compared:
                already_compared[nameInList[0]][nameInList[1]] = is_the_best_video
            else:
                already_compared[nameInList[0]] = {nameInList[1]: is_the_best_video}
                
        shuffle(compareObjs)
    
    remove_not_compatible_video(list_not_compatible_video, dict_file_path_obj)
    
    # Clean up caches to free memory
    if ENHANCED_FEATURES_AVAILABLE:
        clear_scene_cache()
        clear_frame_cache()
        
    return already_compared

def prepare_get_delay_enhanced(videos_obj, language, audioRules):
    """
    Enhanced preparation function that considers scene detection for optimal timing.
    
    Args:
        videos_obj: List of video objects
        language: Language for delay detection
        audioRules: Audio comparison rules
        
    Returns:
        Tuple of preparation parameters optimized for scene detection
    """
    if tools.dev:
        sys.stderr.write("\t\tUsing enhanced delay preparation with scene detection\n")
        
    # Get base parameters
    begin_in_second, audio_param, length_time, length_time_converted, list_cut_begin_length = \
        prepare_get_delay_sub(videos_obj, language)
    
    if not ENHANCED_FEATURES_AVAILABLE or not SCENEDETECT_AVAILABLE:
        return begin_in_second, audio_param, length_time, length_time_converted, list_cut_begin_length
        
    try:
        # Use scene detection to optimize timing parameters
        scene_detector = get_scene_detector(adaptive=True)
        scene_delay_detector = SceneAwareDelayDetector(scene_detector)
        
        # Analyze videos to find optimal comparison segments
        for videoObj in videos_obj:
            try:
                video_duration = float(videoObj.video.get("Duration", 0))
                if video_duration > 0:
                    # Get scene-aware comparison points
                    optimal_points = scene_delay_detector.get_optimal_comparison_points(
                        videoObj.filePath, video_duration, num_points=5
                    )
                    
                    if optimal_points:
                        # Adjust begin_in_second to start of first optimal scene
                        first_scene_start = optimal_points[0][0]
                        if first_scene_start > 5.0:  # Ensure we have some buffer
                            begin_in_second = max(begin_in_second, first_scene_start - 2.0)
                            
            except Exception as e:
                if tools.dev:
                    sys.stderr.write(f"\t\tScene optimization failed for {videoObj.filePath}: {e}\n")
                continue
                
        # Regenerate cut times with optimized parameters
        list_cut_begin_length = video.generate_cut_with_begin_length(
            begin_in_second, length_time, length_time_converted
        )
        
    except Exception as e:
        if tools.dev:
            sys.stderr.write(f"\t\tEnhanced preparation fallback due to error: {e}\n")
        # Fall back to original method
        pass
    
    # Apply enhanced parameters to video objects
    for videoObj in videos_obj:
        for language_obj, audios in videoObj.audios.items():
            for audio in audios:
                audio["keep"] = True
        videoObj.extract_audio_in_part(
            language, audio_param, cutTime=list_cut_begin_length, asDefault=True
        )
        videoObj.delays[language] = 0
        for language_obj, audios in videoObj.commentary.items():
            for audio in audios:
                audio["keep"] = not tools.special_params["remove_commentary"]
    
    return begin_in_second, audio_param, length_time, length_time_converted, list_cut_begin_length

def handle_remaining_comparisons_enhanced(compareObjs, i, new_compare_objs, already_compared,
                                        begin_in_second, audio_param, language, length_time,
                                        length_time_converted, list_cut_begin_length,
                                        time_by_test_best_quality_converted, list_in_compare_video,
                                        list_not_compatible_video):
    """
    Handle remaining video comparisons with enhanced logic.
    
    This function manages the complex logic for comparing remaining videos
    when the main comparison loop has unpaired videos.
    """
    compare_new_obj = None
    remove_i = False
    remove_i_1 = False
    
    if can_always_compare_it(compareObjs[i], compareObjs, new_compare_objs, already_compared):
        compare_new_obj = get_waiter_to_compare(compareObjs[i], new_compare_objs, already_compared)
        if compare_new_obj is not None:
            comparer = enhanced_compare_video(
                compareObjs[i], compare_new_obj, begin_in_second, audio_param,
                language, length_time, length_time_converted, list_cut_begin_length,
                time_by_test_best_quality_converted
            )
            comparer.start()
            list_in_compare_video.append(comparer)
        else:
            new_compare_objs.append(compareObjs[i])
    else:
        remove_i = True
        
    if can_always_compare_it(compareObjs[i + 1], compareObjs, new_compare_objs, already_compared):
        compare_new_obj = get_waiter_to_compare(compareObjs[i + 1], new_compare_objs, already_compared)
        if compare_new_obj is not None:
            comparer = enhanced_compare_video(
                compareObjs[i + 1], compare_new_obj, begin_in_second, audio_param,
                language, length_time, length_time_converted, list_cut_begin_length,
                time_by_test_best_quality_converted
            )
            comparer.start()
            list_in_compare_video.append(comparer)
        else:
            new_compare_objs.append(compareObjs[i + 1])
    elif compare_new_obj is not None and was_they_not_already_compared(
        compareObjs[i + 1], compare_new_obj, already_compared
    ):
        new_compare_objs.append(compareObjs[i + 1])
    else:
        remove_i_1 = True
    
    if remove_i and remove_i_1:
        remove_i = False  # Keep at least one video
        
    if remove_i:
        list_not_compatible_video.append(compareObjs[i].filePath)
        list_not_compatible_video.extend(
            remove_not_compatible_audio(compareObjs[i].filePath, already_compared)
        )
    elif remove_i_1:
        list_not_compatible_video.append(compareObjs[i + 1].filePath)
        list_not_compatible_video.extend(
            remove_not_compatible_audio(compareObjs[i + 1].filePath, already_compared)
        )

def enhanced_sync_merge_video(videosObj, audioRules, out_folder, dict_file_path_obj, forced_best_video):
    """
    Enhanced version of sync_merge_video with scene detection capabilities.
    
    Args:
        videosObj: List of video objects to merge
        audioRules: Audio merging rules
        out_folder: Output folder for merged video
        dict_file_path_obj: Dictionary mapping file paths to objects
        forced_best_video: Forced best video selection (optional)
    """
    if tools.dev:
        sys.stderr.write("\t\tStarting enhanced sync merge with scene detection\n")
        
    # Use original logic for language detection and preparation
    commonLanguages = video.get_common_audios_language(videosObj)
    try:
        commonLanguages.remove("und")
    except:
        pass
        
    if len(commonLanguages) == 0:
        # Handle case with no common languages
        audio_counts = {}
        for videoObj in videosObj:
            for language in videoObj.audios.keys():
                if language not in audio_counts:
                    audio_counts[language] = 0
                audio_counts[language] += 1
                
        most_frequent_language = max(audio_counts, key=audio_counts.get)
        if audio_counts[most_frequent_language] == 1:
            raise Exception(
                f"No common language between {[videoObj.filePath for videoObj in videosObj]}\n"
                f"The language we have {audio_counts}"
            )
        else:
            commonLanguages.add(most_frequent_language)
            # Remove incompatible videos
            list_video_not_compatible_name = []
            videosObj_copy = videosObj.copy()
            for videoObj in videosObj_copy:
                if most_frequent_language not in videoObj.audios:
                    list_video_not_compatible_name.append(videoObj.filePath)
                    videosObj.remove(videoObj)
                    if videoObj.filePath in dict_file_path_obj:
                        del dict_file_path_obj[videoObj.filePath]
                        
            if list_video_not_compatible_name:
                sys.stderr.write(
                    f"{list_video_not_compatible_name} do not have the language {most_frequent_language}\n"
                )
    
    # Select common language for delay generation
    if len(commonLanguages) > 1 and tools.special_params["original_language"] in commonLanguages:
        common_language_use_for_generate_delay = tools.special_params["original_language"]
        commonLanguages.remove(common_language_use_for_generate_delay)
    else:
        commonLanguages = list(commonLanguages)
        common_language_use_for_generate_delay = commonLanguages.pop()
    
    # Handle videos with same audio MD5 (duplicates)
    MD5AudioVideo = {}
    listVideoToNotCalculateOffset = []
    for videoObj in videosObj:
        MD5merged = "".join(set([audio['MD5'] for audio in videoObj.audios[common_language_use_for_generate_delay]]))
        if MD5merged in MD5AudioVideo:
            if forced_best_video == videoObj.filePath:
                videoObj.sameAudioMD5UseForCalculation.append(MD5AudioVideo[MD5merged])
                listVideoToNotCalculateOffset.append(MD5AudioVideo[MD5merged])
                MD5AudioVideo[MD5merged] = videoObj
            else:
                MD5AudioVideo[MD5merged].sameAudioMD5UseForCalculation.append(videoObj)
                listVideoToNotCalculateOffset.append(videoObj)
        else:
            MD5AudioVideo[MD5merged] = videoObj
    
    # Remove duplicate videos from processing
    for videoObj in listVideoToNotCalculateOffset:
        if videoObj in videosObj:
            videosObj.remove(videoObj)
        if videoObj.filePath in dict_file_path_obj:
            del dict_file_path_obj[videoObj.filePath]
    
    # Perform delay detection and quality comparison
    if forced_best_video is None:
        dict_with_video_quality_logic = enhanced_get_delay_and_best_video(
            videosObj, common_language_use_for_generate_delay, audioRules, dict_file_path_obj
        )
    else:
        print_forced_video(forced_best_video)
        dict_with_video_quality_logic = get_delay(
            videosObj, common_language_use_for_generate_delay, audioRules, 
            dict_file_path_obj, forced_best_video
        )
    
    # Process additional common languages (future enhancement)
    for language in commonLanguages:
        # TODO: Implement cross-validation using additional languages
        # This could use scene detection to verify delay consistency across languages
        pass
    
    # Generate final merged video
    generate_launch_merge_command(
        dict_with_video_quality_logic, dict_file_path_obj, out_folder,
        common_language_use_for_generate_delay, audioRules
    )
    
    # Clean up enhanced feature caches
    if ENHANCED_FEATURES_AVAILABLE:
        clear_scene_cache()
        clear_frame_cache()

def enhanced_merge_videos(files, out_folder, merge_sync, inFolder=None):
    """
    Enhanced version of merge_videos with scene detection and improved frame comparison.
    
    This is the main entry point that provides the same interface as the original
    merge_videos function but with enhanced capabilities when available.
    
    Args:
        files: List of video files to merge
        out_folder: Output folder for merged video
        merge_sync: Whether to use synchronization (True) or simple merge (False)
        inFolder: Input folder (optional)
    """
    if tools.dev and ENHANCED_FEATURES_AVAILABLE:
        sys.stderr.write("\t\tUsing enhanced video merge with ML scene detection\n")
    elif tools.dev:
        sys.stderr.write("\t\tEnhanced features not available, using original merge\n")
    
    # Use original video object creation and preparation
    videosObj = []
    name_file = {}
    files = list(files)
    files.sort()
    
    if inFolder is None:
        for file in files:
            videosObj.append(video.video(path.dirname(file), path.basename(file)))
            if videosObj[-1].fileBaseName in name_file:
                name_file[videosObj[-1].fileBaseName] += 1
                videosObj[-1].fileBaseName += "_" + str(name_file[videosObj[-1].fileBaseName])
            else:
                name_file[videosObj[-1].fileBaseName] = 0
    else:
        for file in files:
            videosObj.append(video.video(inFolder, file))
            if videosObj[-1].fileBaseName in name_file:
                name_file[videosObj[-1].fileBaseName] += 1
                videosObj[-1].fileBaseName += "_" + str(name_file[videosObj[-1].fileBaseName])
            else:
                name_file[videosObj[-1].fileBaseName] = 0
                
    name_file = None
    
    # Parse audio rules
    audioRules = decript_merge_rules(tools.mergeRules['audio'])
    
    # Create file path to object mapping
    dict_file_path_obj = {}
    forced_best_video = None
    md5_threads = []
    
    for videoObj in videosObj:
        process_mediadata_thread = Thread(target=videoObj.get_mediadata)
        process_mediadata_thread.start()
        dict_file_path_obj[videoObj.filePath] = videoObj
        
        # Check for forced best video
        if tools.special_params["forced_best_video"] != "":
            if tools.special_params["forced_best_video_contain"]:
                if tools.special_params["forced_best_video"] in videoObj.fileName:
                    forced_best_video = videoObj.filePath
            elif (videoObj.fileName == tools.special_params["forced_best_video"] or 
                  videoObj.filePath == tools.special_params["forced_best_video"]):
                forced_best_video = videoObj.filePath
                
        process_mediadata_thread.join()
        
        # Start MD5 calculation thread
        process_md5_thread = Thread(target=videoObj.calculate_md5_streams)
        process_md5_thread.start()
        md5_threads.append(process_md5_thread)
    
    # Wait for all MD5 calculations to complete
    for process_md5_thread in md5_threads:
        process_md5_thread.join()
    
    # Choose merge method based on sync requirement and feature availability
    if merge_sync:
        if ENHANCED_FEATURES_AVAILABLE:
            enhanced_sync_merge_video(videosObj, audioRules, out_folder, dict_file_path_obj, forced_best_video)
        else:
            sync_merge_video(videosObj, audioRules, out_folder, dict_file_path_obj, forced_best_video)
    else:
        # Simple merge doesn't benefit as much from scene detection
        simple_merge_video(videosObj, audioRules, out_folder, dict_file_path_obj, forced_best_video)
        
    # Final cleanup
    if ENHANCED_FEATURES_AVAILABLE:
        clear_scene_cache()
        clear_frame_cache()
        gc.collect()

# Export the enhanced function as the main interface
merge_videos_enhanced = enhanced_merge_videos
