import sys
import os
import math # Will be needed for _generate_conformed_segments
import copy # For deepcopy if needed later
from decimal import Decimal
import statistics # For _generate_conformed_segments

# To avoid conflicts, import module itself or alias to prevent name clashes
import tools as tools_module 
import video as video_module
import audioCorrelation # For correlate, get_delay_fidelity, second_correlation
import re # For path sanitization in _assemble_chimeric_mkv

# --- Local Helper Functions for Subtitle Handling ---
def _generate_blank_sub_segment(duration_s, output_srt_path):
    """Creates a blank SRT file of duration_s at output_srt_path."""
    try:
        formatted_duration = tools_module.seconds_to_srt_timeformat(duration_s)
        blank_srt_content = f"1\n00:00:00,000 --> {formatted_duration}\n\n"
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(blank_srt_content)
        return True
    except Exception as e:
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Error generating blank subtitle {output_srt_path}: {e}\n")
        return False

def _extract_specific_subtitle_stream_segment(video_file_path, stream_order_id, 
                                             start_s_float, duration_s_float, 
                                             output_srt_path):
    """Extracts a specific subtitle stream segment as SRT."""
    try:
        # For -ss and -t, ffmpeg accepts seconds directly as string.
        start_s_str = str(start_s_float)
        duration_s_str = str(duration_s_float)
        
        ffmpeg_cmd = [
            tools_module.software['ffmpeg'], '-y', '-nostdin',
            '-i', video_file_path,
            '-ss', start_s_str, 
            '-t', duration_s_str, 
            '-map', f"0:{stream_order_id}",
            '-c:s', 'srt', # Force SRT output
            output_srt_path
        ]
        _stdout, stderr_str, exit_code = tools_module.launch_cmdExt_no_test(ffmpeg_cmd)
        if exit_code != 0:
            # This can happen if no subtitles exist in the specified range. Not always a fatal error for the segment.
            # sys.stderr.write(f"CHIMERIC_PROCESSOR: FFmpeg non-zero exit ({exit_code}) extracting subtitle segment {output_srt_path}: {stderr_str.decode('utf-8', errors='ignore')}\n")
            return False 
        return True
    except Exception as e:
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Exception extracting subtitle segment {output_srt_path}: {e}\n")
        return False

# Local helper for audio extraction for correlation (distinct from final segment extraction)
def _extract_audio_segment_for_correlation(video_obj_path, 
                                           audio_track_stream_order, 
                                           start_time_str, # Expected HH:MM:SS.mmm
                                           duration_str, # Expected HH:MM:SS.mmm
                                           output_wav_path,
                                           target_sample_rate="48000", 
                                           target_channels="2"):
    """Extracts a specific audio track segment to output_wav_path as WAV PCM S16LE for correlation."""
    try:
        ffmpeg_cmd = [
            tools_module.software['ffmpeg'], '-y', '-nostdin',
            '-i', video_obj_path,
            '-ss', start_time_str,
            '-t', duration_str,
            '-map', f"0:{audio_track_stream_order}",
            '-vn', 
            '-acodec', 'pcm_s16le', 
            '-ar', target_sample_rate, 
            '-ac', target_channels,
            output_wav_path
        ]
        _stdout, stderr_str, exit_code = tools_module.launch_cmdExt(ffmpeg_cmd)
        if exit_code != 0:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Error extracting correlation audio segment {output_wav_path}: {stderr_str.decode('utf-8', errors='ignore')}\n")
            return False
        return True
    except Exception as e:
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Exception extracting correlation audio segment {output_wav_path}: {e}\n")
        return False

def _generate_conformed_segments(main_video_obj, other_video_obj, sync_language, 
                                audio_params_for_wav, # For main loop correlation (extract_audio_in_part)
                                length_time_initial_segment_chunk_duration, 
                                tools_tmpFolder):
    sys.stderr.write(f"CHIMERIC_PROCESSOR: _generate_conformed_segments for main: {main_video_obj.filePath}, other: {other_video_obj.filePath}\n")
    temp_local_files = [] 
    try:
        initial_offset_ms = None
        initial_offset_audio_params = {'Format': "WAV", 'codec': "pcm_s16le", 'Channels': "2"}
        min_duration_s = min(float(main_video_obj.video['Duration']), float(other_video_obj.video['Duration']))
        
        begin_s_for_initial_cuts = 5.0 
        if min_duration_s < (begin_s_for_initial_cuts + video_module.number_cut * length_time_initial_segment_chunk_duration):
            begin_s_for_initial_cuts = 0.0

        if min_duration_s < video_module.number_cut * length_time_initial_segment_chunk_duration :
             sys.stderr.write(f"CHIMERIC_PROCESSOR: Videos too short for robust initial offset. Min duration: {min_duration_s}s. Will proceed with fewer cuts.\n")

        list_cut_begin_length_initial = video_module.generate_cut_with_begin_length(
            begin_s_for_initial_cuts, 
            length_time_initial_segment_chunk_duration, 
            tools_module.format_time_ffmpeg_dot(length_time_initial_segment_chunk_duration) # Use tools
        )
        main_video_obj.extract_audio_in_part(sync_language, initial_offset_audio_params, cutTime=list_cut_begin_length_initial, asDefault=True)
        other_video_obj.extract_audio_in_part(sync_language, initial_offset_audio_params, cutTime=list_cut_begin_length_initial, asDefault=True)
        delay_fidelity_values = audioCorrelation.get_delay_fidelity(main_video_obj, other_video_obj, length_time_initial_segment_chunk_duration)
        main_video_obj.remove_tmp_files(type_file="audio")
        other_video_obj.remove_tmp_files(type_file="audio")

        if not delay_fidelity_values: return None, 'incompatible_initial_offset_no_results'
        all_collected_delays_ms = [df[2] for df_list in delay_fidelity_values.values() for df in df_list if df_list]
        if not all_collected_delays_ms: return None, 'incompatible_initial_offset_no_delays'
        if len(all_collected_delays_ms) > 1:
            delay_stdev = statistics.stdev(all_collected_delays_ms)
            if delay_stdev > 100: return None, 'incompatible_initial_offset_unstable'
        initial_offset_ms = Decimal(str(round(statistics.mean(all_collected_delays_ms))))
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Initial offset calculated: {initial_offset_ms}ms\n")

        FIDELITY_THRESHOLD = 0.80; SEGMENT_DURATION_S = 60.0; INCOMPATIBILITY_THRESHOLD_PERCENT = 50.0; OFFSET_CONSISTENCY_WINDOW_S = 0.5
        segment_assembly_plan = []; current_main_time_s = 0.0; total_bad_segments_duration_s = 0.0; total_processed_duration_s = 0.0
        current_other_offset_s = float(initial_offset_ms / Decimal(1000))
        main_total_duration_s = float(main_video_obj.video['Duration']); other_total_duration_s = float(other_video_obj.video['Duration'])
        main_corr_audio_stream_order = main_video_obj.audios[sync_language][0]['StreamOrder']
        other_corr_audio_stream_order = other_video_obj.audios[sync_language][0]['StreamOrder']
        if not main_corr_audio_stream_order or not other_corr_audio_stream_order: return None, 'incompatible_no_sync_audio_stream'

        idx = 0
        while current_main_time_s < main_total_duration_s:
            actual_segment_duration_s = min(SEGMENT_DURATION_S, main_total_duration_s - current_main_time_s)
            if actual_segment_duration_s < 1.0: break
            main_segment_start_str = tools_module.format_time_ffmpeg_dot(current_main_time_s)
            main_segment_duration_str = tools_module.format_time_ffmpeg_dot(actual_segment_duration_s)
            
            # Use local _extract_audio_segment_for_correlation for WAVs
            main_audio_segment_path = os.path.join(tools_tmpFolder, f"ch_main_corr_seg_{idx}.wav")
            temp_local_files.append(main_audio_segment_path)
            if not _extract_audio_segment_for_correlation(main_video_obj.filePath, main_corr_audio_stream_order, main_segment_start_str, main_segment_duration_str, main_audio_segment_path): # audio_params_for_wav not used by helper
                segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'main_extract_fail_corr_loop', 'audio_source': 'main'})
                total_bad_segments_duration_s += actual_segment_duration_s
            else:
                other_segment_start_abs_s = current_main_time_s - current_other_offset_s
                other_audio_segment_path = os.path.join(tools_tmpFolder, f"ch_other_corr_seg_{idx}.wav")
                temp_local_files.append(other_audio_segment_path)
                if (other_segment_start_abs_s + actual_segment_duration_s < 0.1) or (other_segment_start_abs_s >= other_total_duration_s - 0.1):
                    segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_out_of_bounds_corr_loop', 'audio_source': 'main'})
                    total_bad_segments_duration_s += actual_segment_duration_s
                else:
                    other_segment_start_clipped_s = max(0.0, other_segment_start_abs_s)
                    other_segment_duration_clipped_s = min(actual_segment_duration_s, other_total_duration_s - other_segment_start_clipped_s)
                    if other_segment_duration_clipped_s < 1.0:
                        segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_clip_too_short_corr_loop', 'audio_source': 'main'})
                        total_bad_segments_duration_s += actual_segment_duration_s
                    else:
                        other_segment_start_str = tools_module.format_time_ffmpeg_dot(other_segment_start_clipped_s)
                        other_segment_duration_str = tools_module.format_time_ffmpeg_dot(other_segment_duration_clipped_s)
                        if not _extract_audio_segment_for_correlation(other_video_obj.filePath, other_corr_audio_stream_order, other_segment_start_str, other_segment_duration_str, other_audio_segment_path): # audio_params_for_wav not used by helper
                            segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'other_extract_fail_corr_loop', 'audio_source': 'main'})
                            total_bad_segments_duration_s += actual_segment_duration_s
                        else:
                            try:
                                fidelity, _, _ = audioCorrelation.correlate(main_audio_segment_path, other_audio_segment_path, other_segment_duration_clipped_s)
                                shifted_file, precise_offset_s_fft_raw = audioCorrelation.second_correlation(main_audio_segment_path, other_audio_segment_path)
                                segment_correction_s = -precise_offset_s_fft_raw if shifted_file == main_audio_segment_path else precise_offset_s_fft_raw
                                if fidelity >= FIDELITY_THRESHOLD and abs(segment_correction_s) < OFFSET_CONSISTENCY_WINDOW_S:
                                    current_other_offset_s -= segment_correction_s
                                    segment_assembly_plan.append({'type': 'good', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'other_start_abs_s': other_segment_start_abs_s, 'other_duration_clipped_s': other_segment_duration_clipped_s, 'final_offset_for_other_s': current_other_offset_s, 'segment_correction_s': segment_correction_s, 'fidelity': fidelity, 'audio_source': 'other'})
                                else:
                                    reason = 'low_fidelity' if fidelity < FIDELITY_THRESHOLD else 'offset_drift_corr_loop'
                                    segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': reason, 'fidelity': fidelity, 'segment_correction_s': segment_correction_s, 'audio_source': 'main'})
                                    total_bad_segments_duration_s += actual_segment_duration_s
                            except Exception as e_corr:
                                sys.stderr.write(f"CHIMERIC_PROCESSOR: Error during correlation for main time {current_main_time_s:.2f}s: {e_corr}\n")
                                segment_assembly_plan.append({'type': 'bad', 'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s, 'reason': 'correlation_exception', 'audio_source': 'main'})
                                total_bad_segments_duration_s += actual_segment_duration_s
            total_processed_duration_s += actual_segment_duration_s
            current_main_time_s += actual_segment_duration_s
            idx += 1
        if main_total_duration_s > 0 and (total_bad_segments_duration_s / main_total_duration_s * 100.0) > INCOMPATIBILITY_THRESHOLD_PERCENT:
            return None, 'incompatible_high_bad_segments'
        return segment_assembly_plan, 'compatible'
    except Exception as e_outer:
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Outer exception in _generate_conformed_segments: {e_outer}\n")
        import traceback; traceback.print_exc(file=sys.stderr)
        return None, 'exception_outer_processing'
    finally:
        for f_path in temp_local_files:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError: pass

def _assemble_chimeric_mkv(other_video_obj_original_path_or_object, segment_assembly_plan, 
                           main_video_obj, # other_video_obj_conformed_info, # This param seems unused, other_video_obj_original_path_or_object is the one
                           sync_language, common_subtitle_languages_list, 
                           audio_params_for_final_encode, # New param
                           tools_tmpFolder, out_folder_for_chimeric_file):
    sys.stderr.write(f"CHIMERIC_PROCESSOR: _assemble_chimeric_mkv for original: {other_video_obj_original_path_or_object.filePath}, main_ref: {main_video_obj.filePath}\n")
    other_video_obj = other_video_obj_original_path_or_object 
    audio_segment_files_for_concat = []
    subtitle_segment_files_for_concat = {lang: [] for lang in common_subtitle_languages_list}
    local_temp_files = []
    main_audio_stream_order = main_video_obj.audios.get(sync_language, [{}])[0].get('StreamOrder')
    other_audio_stream_order = other_video_obj.audios.get(sync_language, [{}])[0].get('StreamOrder')
    if not main_audio_stream_order or not other_audio_stream_order: return None

    for idx, segment_info in enumerate(segment_assembly_plan):
        segment_duration_s = segment_info['duration_s']
        temp_audio_segment_path = tools_module.path.join(tools_tmpFolder, f"final_audio_seg_{other_video_obj.fileBaseName}_{idx}.wav")
        local_temp_files.append(temp_audio_segment_path)
        source_for_audio_obj = other_video_obj if segment_info['audio_source'] == 'other' else main_video_obj
        audio_extract_start_s = segment_info['other_start_abs_s'] if segment_info['audio_source'] == 'other' else segment_info['main_start_s']
        audio_extract_duration_s = segment_info['other_duration_clipped_s'] if segment_info['audio_source'] == 'other' else segment_duration_s
        current_audio_stream_order = other_audio_stream_order if segment_info['audio_source'] == 'other' else main_audio_stream_order
        audio_extract_start_s = max(0.0, audio_extract_start_s)
        if not _extract_audio_segment_for_correlation(source_for_audio_obj.filePath, current_audio_stream_order, 
                                                     tools_module.format_time_ffmpeg_dot(audio_extract_start_s), 
                                                     tools_module.format_time_ffmpeg_dot(audio_extract_duration_s), 
                                                     temp_audio_segment_path): # audio_params_for_wav removed
            cleanup_temp_files(local_temp_files); return None
        audio_segment_files_for_concat.append(temp_audio_segment_path)

        for sub_lang in common_subtitle_languages_list:
            temp_sub_segment_path = tools_module.path.join(tools_tmpFolder, f"final_sub_seg_{other_video_obj.fileBaseName}_{sub_lang}_{idx}.srt")
            local_temp_files.append(temp_sub_segment_path)
            extracted_sub_successfully = False
            source_for_subs_obj = other_video_obj if segment_info['audio_source'] == 'other' else main_video_obj
            sub_extract_start_s = segment_info['other_start_abs_s'] if segment_info['audio_source'] == 'other' else segment_info['main_start_s']
            sub_extract_duration_s = segment_info['other_duration_clipped_s'] if segment_info['audio_source'] == 'other' else segment_duration_s
            sub_extract_start_s = max(0.0, sub_extract_start_s)
            if sub_lang in source_for_subs_obj.subtitles and source_for_subs_obj.subtitles[sub_lang]:
                sub_track_info_to_extract = source_for_subs_obj.subtitles[sub_lang][0] 
                if _extract_specific_subtitle_stream_segment(source_for_subs_obj.filePath, sub_track_info_to_extract['StreamOrder'], sub_extract_start_s, sub_extract_duration_s, temp_sub_segment_path):
                    extracted_sub_successfully = True
            if not extracted_sub_successfully:
                if not _generate_blank_sub_segment(sub_extract_duration_s, temp_sub_segment_path):
                    cleanup_temp_files(local_temp_files); return None
            subtitle_segment_files_for_concat[sub_lang].append(temp_sub_segment_path)

    audio_concat_list_path = tools_module.path.join(tools_tmpFolder, f"audio_concat_{other_video_obj.fileBaseName}.txt")
    with open(audio_concat_list_path, 'w', encoding='utf-8') as f:
        for p in audio_segment_files_for_concat: f.write(f"file '{tools_module.path.basename(p)}'\n")
    local_temp_files.append(audio_concat_list_path)
    subtitle_concat_paths_map = {}
    for lang, file_list in subtitle_segment_files_for_concat.items():
        if file_list:
            sub_concat_path = tools_module.path.join(tools_tmpFolder, f"sub_{lang}_concat_{other_video_obj.fileBaseName}.txt")
            with open(sub_concat_path, 'w', encoding='utf-8') as f:
                for p in file_list: f.write(f"file '{tools_module.path.basename(p)}'\n")
            subtitle_concat_paths_map[lang] = sub_concat_path
            local_temp_files.append(sub_concat_path)

    safe_basename = re.sub(r'[^\w.-]', '_', other_video_obj.fileBaseName)
    chimeric_mkv_path = tools_module.path.join(out_folder_for_chimeric_file, f"{safe_basename}_chimeric_{sync_language}.mkv")
    ffmpeg_cmd = [tools_module.software['ffmpeg'], '-y', '-nostdin', '-i', other_video_obj.filePath, '-f', 'concat', '-safe', '0', '-i', audio_concat_list_path]
    input_idx_counter = 2; subtitle_map_commands = []; output_subtitle_stream_idx = 0
    for lang in common_subtitle_languages_list:
        if lang in subtitle_concat_paths_map:
            ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', subtitle_concat_paths_map[lang]])
            subtitle_map_commands.extend([f"-map", f"{input_idx_counter}:s:0"])
            subtitle_map_commands.extend([f"-metadata:s:s:{output_subtitle_stream_idx}", f"language={lang}"])
            input_idx_counter += 1; output_subtitle_stream_idx += 1
    ffmpeg_cmd.extend(['-map', f"0:{other_video_obj.video['StreamOrder']}", '-map', '1:a:0']); ffmpeg_cmd.extend(subtitle_map_commands)
    ffmpeg_cmd.extend(['-c:v', 'copy', '-c:a', audio_params_for_final_encode.get('codec', 'flac')])
    if audio_params_for_final_encode.get('params'): ffmpeg_cmd.extend(audio_params_for_final_encode['params'])
    ffmpeg_cmd.extend(['-c:s', 'ass', '-shortest', chimeric_mkv_path])
    final_mkv_generated_path = None
    try:
        sys.stdout.write(f"CHIMERIC_PROCESSOR: Assembling Chimeric MKV: {' '.join(ffmpeg_cmd)}\n")
        _stdout_ff, stderr_ff, exit_code_ff = tools_module.launch_cmdExt(ffmpeg_cmd, cwd=tools_tmpFolder)
        if exit_code_ff == 0: final_mkv_generated_path = chimeric_mkv_path
        else: sys.stderr.write(f"CHIMERIC_PROCESSOR: Error generating chimeric MKV: {stderr_ff.decode('utf-8', errors='ignore')}\n")
    except Exception as e_ff: sys.stderr.write(f"CHIMERIC_PROCESSOR: Exception during FFmpeg execution for chimeric MKV: {e_ff}\n")
    finally: cleanup_temp_files(local_temp_files)
    return final_mkv_generated_path

def process(list_not_compatible_video, already_compared,dict_file_path_obj, language,length_time_converted): # Added audio_params_for_final_encode
    if len(list_not_compatible_video) == 0:
        # If all files are ok, no chimeric process have to be do
        return
    sys.stderr.write("CHIMERIC_PROCESSOR: Chimeric processing started.\n")
    set_bad_video = set()
    dict_list_video_win = {}
    for video_path_file, dict_with_results in dict_with_video_quality_logic.items():
        for other_video_path_file, is_the_best_video in dict_with_results.items():
            if is_the_best_video:
                set_bad_video.add(other_video_path_file)
                if video_path_file in dict_list_video_win:
                    dict_list_video_win[video_path_file].append(other_video_path_file)
                else:
                    dict_list_video_win[video_path_file] = [other_video_path_file]
            else:
                set_bad_video.add(video_path_file)
                if other_video_path_file in dict_list_video_win:
                    dict_list_video_win[other_video_path_file].append(video_path_file)
                else:
                    dict_list_video_win[other_video_path_file] = [video_path_file]
    
    best_video = dict_file_path_obj[list(dict_file_path_obj.keys() - set_bad_video)[0]]

    if not V_abs_best:
        sys.stderr.write("CHIMERIC_PROCESSOR: CRITICAL ERROR - V_abs_best could not be identified.\n")
        for video_path in current_dict_file_path_obj.keys(): list_not_compatible_video_input_output.append(video_path)
        return current_dict_file_path_obj
    sys.stderr.write(f"CHIMERIC_PROCESSOR: V_abs_best identified as {V_abs_best.filePath}\n")

    if V_abs_best.chimeric_files is None:
        V_abs_best.chimeric_files = []

    updated_dict_file_path_obj = {V_abs_best.filePath: V_abs_best}
    common_subtitle_languages = list(set(V_abs_best.subtitles.keys())) # Start with V_abs_best's subs

    for original_path, video_to_conform in current_dict_file_path_obj.items():
        if video_to_conform.filePath == V_abs_best.filePath: continue
        common_subtitle_languages = list(set(common_subtitle_languages) | set(video_to_conform.subtitles.keys()))
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Attempting to conform {video_to_conform.filePath} to {V_abs_best.filePath}\n")
        segment_assembly_plan, status = _generate_conformed_segments(V_abs_best, video_to_conform, common_sync_language, audio_params_for_wav, length_time_for_initial_offset_calc, tools_tmpFolder)
        if status != 'compatible' or not segment_assembly_plan:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: {video_to_conform.filePath} plan generation failed or incompatible: {status}.\n")
            list_not_compatible_video_input_output.append(original_path)
        else:
            chimeric_mkv_path = _assemble_chimeric_mkv(video_to_conform, segment_assembly_plan, V_abs_best, common_sync_language, common_subtitle_languages, audio_params_for_final_encode, tools_tmpFolder, tools_tmpFolder)
            if chimeric_mkv_path and os.path.exists(chimeric_mkv_path):
                sys.stderr.write(f"CHIMERIC_PROCESSOR: Assembled chimeric MKV for {original_path} at {chimeric_mkv_path}\n")
                chimeric_video_obj = video_module.video(tools_tmpFolder, os.path.basename(chimeric_mkv_path)) # Chimeric file is in tools_tmpFolder
                chimeric_video_obj.get_mediadata(); chimeric_video_obj.calculate_md5_streams_split()
                chimeric_video_obj.delays[common_sync_language] = Decimal(0)
                updated_dict_file_path_obj[chimeric_mkv_path] = chimeric_video_obj
                if V_abs_best.chimeric_files is None: # Defensive check, should have been initialized in video.py
                    V_abs_best.chimeric_files = []
                V_abs_best.chimeric_files.append(chimeric_video_obj)
            else:
                sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to assemble chimeric MKV for {original_path}.\n")
                list_not_compatible_video_input_output.append(original_path)
    sys.stderr.write(f"CHIMERIC_PROCESSOR: Processing finished. Incompatible: {list_not_compatible_video_input_output}\n")
    return updated_dict_file_path_obj

def check_if_chimeric_tracks_needed(out_video_metadata_obj, V_abs_best_obj):
    """
    Checks if any audio or subtitle tracks from chimeric files are needed (i.e., languages not present in the main merged file).
    """
    needed_tracks_by_file = []
    if not V_abs_best_obj.chimeric_files: # Handles None or empty list
        return needed_tracks_by_file

    existing_audio_languages = set(out_video_metadata_obj.audios.keys())
    existing_subtitle_languages = set(out_video_metadata_obj.subtitles.keys())

    for chimeric_vid_obj in V_abs_best_obj.chimeric_files:
        if not chimeric_vid_obj: # Skip if a chimeric_vid_obj is None for some reason
            continue
        tracks_from_this_chimeric = []
        # Check audio tracks
        for lang, track_list in chimeric_vid_obj.audios.items():
            if lang not in existing_audio_languages:
                for track_info in track_list:
                    tracks_from_this_chimeric.append(track_info)
        
        # Check subtitle tracks
        for lang, track_list in chimeric_vid_obj.subtitles.items():
            if lang not in existing_subtitle_languages:
                for track_info in track_list:
                    tracks_from_this_chimeric.append(track_info)
        
        if tracks_from_this_chimeric:
            needed_tracks_by_file.append((chimeric_vid_obj, tracks_from_this_chimeric))
            # Add newly found languages to existing sets to prevent adding them again from other chimeric files if they offer the same new language
            for track in tracks_from_this_chimeric:
                if track['@type'] == 'Audio':
                    existing_audio_languages.add(track.get('Language', 'und'))
                elif track['@type'] == 'Text':
                    existing_subtitle_languages.add(track.get('Language', 'und'))

    return needed_tracks_by_file

def augment_merge_file_with_tracks(original_temp_merge_path, needed_tracks_info_list, out_path_tmp_file_name_split_augmented):
    """
    Augments the original_temp_merge_path with tracks specified in needed_tracks_info_list.
    Returns the path to the new augmented MKV file.
    """
    
    mkvmerge_cmd = [tools.software['mkvmerge'], '-o', out_path_tmp_file_name_split_augmented]

    # Add original_temp_merge_path as the first input group
    mkvmerge_cmd.extend(['(', original_temp_merge_path, ')'])

    for chimeric_vid_obj, tracks_to_add_list in needed_tracks_info_list:
        mkvmerge_cmd.append('(')
        mkvmerge_cmd.append(chimeric_vid_obj.filePath)

        audio_stream_orders_to_map = []
        subtitle_stream_orders_to_map = []
        track_details_for_setting_flags = {} # key: stream_order_str, value: track_info_dict

        for track_info_dict in tracks_to_add_list:
            stream_order_str = str(track_info_dict['StreamOrder'])
            track_details_for_setting_flags[stream_order_str] = track_info_dict
            if track_info_dict['@type'] == 'Audio':
                audio_stream_orders_to_map.append(stream_order_str)
            elif track_info_dict['@type'] == 'Text':
                subtitle_stream_orders_to_map.append(stream_order_str)

        if not audio_stream_orders_to_map and not subtitle_stream_orders_to_map:
            mkvmerge_cmd.append(')')
            continue # Skip if no tracks are actually selected from this chimeric file

        mkvmerge_cmd.append('--no-video')

        if audio_stream_orders_to_map:
            mkvmerge_cmd.extend(['--audio-tracks', ','.join(audio_stream_orders_to_map)])
        else:
            mkvmerge_cmd.append('--no-audio')

        if subtitle_stream_orders_to_map:
            mkvmerge_cmd.extend(['--subtitle-tracks', ','.join(subtitle_stream_orders_to_map)])
        else:
            mkvmerge_cmd.append('--no-subtitles')
        
        # Iterate through all tracks intended to be mapped from this source
        all_mapped_stream_orders = audio_stream_orders_to_map + subtitle_stream_orders_to_map
        for stream_order_str in all_mapped_stream_orders:
            track_lang = track_details_for_setting_flags[stream_order_str].get('Language', 'und')
            # Mkvmerge input file track selection is 0-indexed based on the order of tracks *selected* from that input,
            # not their original StreamOrder. However, mkvmerge allows specifying by original TID (Track ID)
            # which corresponds to StreamOrder. The format is `TID:new_TID` or just `TID`.
            # For setting flags, it's `TID:value` or `type:TID:value`.
            # When tracks are selected with --audio-tracks TID1,TID2, etc., their output order is fixed.
            # We need to refer to them by their original TIDs for setting flags.
            # The problem is that --language TID:lang assumes TID is the *output* track ID.
            # This gets complex. A simpler way is to map specific tracks and then rely on their relative order for subsequent flag settings,
            # or set flags for *all* tracks of a certain type from an input.
            # Let's try setting flags using the original track ID (StreamOrder) from the input file.
            # The format for --language is <TID>:<lang> or <type>:<TID>:<lang>
            # where TID is the track ID *in the input file*.
            
            # Corrected approach: mkvmerge options like --language, --track-name, --default-track-flag
            # take the *input track ID* (which is our StreamOrder) when used with a single input file context.
            # When multiple input files are used, track IDs can be prefixed with `file_idx:` (e.g., `1:TID`).
            # Within parenthesis, the file is treated as a single input, so `TID` should suffice.
            # The `0:` prefix was incorrect as it implies the first selected track from the input, not the file index.
            # The correct way is to just use the stream_order_str, as mkvmerge applies it to the current input file in parenthesis
            # when specific tracks are selected using --audio-tracks or --subtitle-tracks with their TIDs.
            
            mkvmerge_cmd.extend(['--language', f"{stream_order_str}:{track_lang}"])
            mkvmerge_cmd.extend(['--default-track-flag', f"{stream_order_str}:0"])
            mkvmerge_cmd.extend(['--forced-display-flag', f"{stream_order_str}:0"])
            if track_details_for_setting_flags[stream_order_str].get('Title'):
                 mkvmerge_cmd.extend(['--track-name', f"{stream_order_str}:{track_details_for_setting_flags[stream_order_str]['Title']}"])

        mkvmerge_cmd.extend(['--no-attachments', '--no-global-tags', '--no-chapters'])
        mkvmerge_cmd.append(')')

    sys.stderr.write(f"MERGEVIDEO_AUGMENT: Executing mkvmerge command: {' '.join(mkvmerge_cmd)}\n")
    tools.launch_cmdExt(mkvmerge_cmd)
    return augmented_mkv_path

def integrate_chimerics_files(out_video_metadata,out_path_tmp_file_name_split,best_video):
    # --- Chimeric Tracks Integration ---
    needed_chimeric_tracks = check_if_chimeric_tracks_needed(out_video_metadata, best_video)
    if needed_chimeric_tracks:
        sys.stderr.write(f"MERGEVIDEO: Found {len(needed_chimeric_tracks)} chimeric files with needed tracks.\n")
        out_path_tmp_file_name_split_augmented = path.join(tools.tmpFolder,f"{best_video.fileBaseName}_merged_split_augmented.mkv")
        current_tmp_merge_file_path = out_path_tmp_file_name_split
        augment_merge_file_with_tracks(
            out_path_tmp_file_name_split,
            needed_chimeric_tracks,
            out_path_tmp_file_name_split_augmented
        )
        
        # Re-initialize and load metadata for the new augmented file
        sys.stderr.write(f"MERGEVIDEO: Reloading metadata from augmented file: {out_path_tmp_file_name_split_augmented}\n")
        out_video_metadata = video.video(tools.tmpFolder, path.basename(out_path_tmp_file_name_split_augmented))
        out_video_metadata.get_mediadata()
        return out_video_metadata,out_path_tmp_file_name_split_augmented
    else:
        sys.stderr.write("MERGEVIDEO: No needed chimeric tracks found or V_abs_best.chimeric_files is empty.\n")
        return out_video_metadata,out_path_tmp_file_name_split
    # --- End Chimeric Tracks Integration ---

def chimeric_process():
    # --- Chimeric Processing Step ---
    if tools.special_params.get("chimeric_process_enabled", "false").lower() == "true":
        sys.stderr.write("INFO: Chimeric processing is enabled.\n")
        list_for_incompatible_chimera = [] # To be populated by chimeric_processor.process

        # Calculate a representative length_time for the initial coarse offset detection in chimeric_processor
        all_video_durations = [float(v.video['Duration']) for v in dict_file_path_obj.values() if v.video and 'Duration' in v.video and v.video['Duration']]
        min_overall_duration_s = min(all_video_durations) if all_video_durations else 300.0 # Default to 5 mins
        _unused_begin_s_overall, length_time_overall_param = video.generate_begin_and_length_by_segment(min_overall_duration_s)

        audio_params_for_wav_param = {'Format':"WAV", 'codec':"pcm_s16le", 'Channels': "2"}
        audio_params_for_final_encode_param = {'codec': 'flac', 'params': ['-compression_level', '12']}

        # current_dict_file_path_obj is dict_file_path_obj before this call
        # already_compared_dict is dict_with_video_quality_logic from initial comparisons
        updated_dict_file_path_obj_from_chimera = chimeric_processor.process(
            list_for_incompatible_chimera, # Populated by process
            dict_with_video_quality_logic, # Used to find V_abs_best
            dict_file_path_obj,            # Current videos
            common_language_use_for_generate_delay,
            length_time_overall_param,     # For initial coarse offset
            audio_params_for_wav_param,    # For correlation audio segments
            tools.tmpFolder,
            audio_params_for_final_encode_param # For final audio encoding in chimeric MKV
        )
        
        # Update dict_file_path_obj with results from chimeric processing
        dict_file_path_obj = updated_dict_file_path_obj_from_chimera

        # list_not_compatible_video might already exist from get_delay_and_best_video if some videos were incompatible there.
        # We need to make sure it's initialized if not already.
        if 'list_not_compatible_video' not in locals() and 'list_not_compatible_video' not in globals():
             list_not_compatible_video = [] # Initialize if it wasn't created before (e.g. if forced_best_video path was taken)
        
        # Add videos that failed chimeric process to the list of videos to be removed
        # The chimeric_processor.process function populates list_for_incompatible_chimera with original paths.
        for failed_original_path in list_for_incompatible_chimera:
            if failed_original_path not in list_not_compatible_video:
                 list_not_compatible_video.append(failed_original_path)
        
        # After chimeric processing, V_abs_best (identified within chimeric_processor.process) is the reference.
        # All other successfully processed videos in dict_file_path_obj are conformed to it (delay=0).
        # Rebuild dict_with_video_quality_logic to reflect this for generate_launch_merge_command.
        
        # Find V_abs_best again from the potentially updated dict_file_path_obj keys if its path changed (it shouldn't)
        # Or, more robustly, ensure V_abs_best determined by chimeric_processor is used.
        # For now, assume V_abs_best's path is stable and it's the first key if only one "best" found by it.
        # The chimeric_processor.process should ideally return V_abs_best_obj as well, or its path.
        # For now, we find it from the returned dict_file_path_obj assuming it contains V_abs_best and conformed files.
        
        # This V_abs_best identification logic here is a simplified assumption.
        # The one in chimeric_processor.process is more robust.
        # We need to ensure generate_launch_merge_command gets the correct reference.
        # The simplest is that dict_file_path_obj now contains V_abs_best and conformed videos.
        # And dict_with_video_quality_logic should reflect V_abs_best as the top.
        
        # V_abs_best_path_after_chimera = None
        # if dict_file_path_obj: # Check if not empty
        #    # This assumes V_abs_best is still identifiable or is the 'main' reference.
        #    # The actual V_abs_best path might be complex to re-identify if its original path was removed.
        #    # However, chimeric_processor starts its updated_dict with V_abs_best.
        #    V_abs_best_path_after_chimera = list(dict_file_path_obj.keys())[0] # Risky assumption
        # A better way: find the video object that is NOT in tools.tmpFolder (original V_abs_best)
        
        v_abs_best_ref_path = None
        for vid_path, vid_obj_iter in dict_file_path_obj.items():
            if tools.tmpFolder not in vid_obj_iter.fileFolder: # The original V_abs_best won't be in tmpFolder
                v_abs_best_ref_path = vid_path
                break
        
        if v_abs_best_ref_path:
            new_dict_with_video_quality_logic = {v_abs_best_ref_path: {}}
            for video_fpath_key in dict_file_path_obj.keys():
                if video_fpath_key != v_abs_best_ref_path:
                    new_dict_with_video_quality_logic[v_abs_best_ref_path][video_fpath_key] = True 
            dict_with_video_quality_logic = new_dict_with_video_quality_logic
        else:
            sys.stderr.write("ERROR: V_abs_best path could not be re-identified after chimeric processing. Merge may fail or be incorrect.\n")
            # If V_abs_best was itself a chimeric (e.g. if only two files), this logic fails.
            # The design implies V_abs_best is an original file.

        # remove_not_compatible_video was called earlier in get_delay_and_best_video.
        # Call it again if list_not_compatible_video has new entries from chimeric process.
        if list_for_incompatible_chimera: # If chimeric process added new incompatibles
             remove_not_compatible_video(list_for_incompatible_chimera, dict_file_path_obj)

    else:
        sys.stderr.write("INFO: Chimeric processing is disabled.\n")
    
    if not dict_file_path_obj:
        sys.stderr.write("No videos remaining after all processing. Aborting merge.\n")
        return

class get_cut_time(Thread):
    '''
    classdocs
    '''


    def __init__(self, main_video_obj,video_obj_to_cut,begin_in_second,audioParam,language,lenghtTime,lenghtTimePrepare,list_cut_begin_length,time_by_test_best_quality_converted):
        '''
        Constructor
        '''
        Thread.__init__(self)
        self.main_video_obj = main_video_obj
        self.video_obj_to_cut = video_obj_to_cut
        self.begin_in_second = begin_in_second
        self.audioParam = audioParam
        self.language = language
        self.lenghtTime = lenghtTime
        self.lenghtTimePrepare = lenghtTimePrepare
        self.list_cut_begin_length = list_cut_begin_length
        self.time_by_test_best_quality_converted = time_by_test_best_quality_converted

    def run(self):
        try:
            delay = self.get_first_delay_and_gap()
            if self.process_to_get_best_video:
                self.get_best_video(delay)
            else: # You must have the video you want process in video_obj_1
                self.video_obj_1.extract_audio_in_part(self.language,self.audioParam,cutTime=self.list_cut_begin_length,asDefault=True)
                self.video_obj_2.remove_tmp_files(type_file="audio")
                self.video_obj_with_best_quality = self.video_obj_1
                self.video_obj_2.delays[self.language] += (delay*-1.0) # Delay you need to give to mkvmerge to be good.
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(str(e)+"\n")
    
    def get_first_delay_and_gap(self):
        delay_Fidelity_Values = get_delay_fidelity(self.main_video_obj,self.video_obj_to_cut,self.lenghtTime)
        # Il va falloir verifier que nous avons bien les mêmes delays entre les différents audios
        keys_audio = list(delay_Fidelity_Values.keys())
        values_of_delay = delay_Fidelity_Values[keys_audio[0]]
        for key_audio, delay_fidelity_list in delay_Fidelity_Values.items():
            for i in range(len(values_of_delay)):
                if values_of_delay[i] != delay_fidelity_list[i]:
                    raise Exception(f"{delay_Fidelity_Values} Impossible to find a way to cut {self.video_obj_to_cut.filePath} who have differents audio not compatible with {self.main_video_obj.filePath}")