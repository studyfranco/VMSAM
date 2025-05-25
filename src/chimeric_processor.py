import sys
import os
import math # Will be needed for _generate_conformed_segments
import copy # For deepcopy if needed later
from decimal import Decimal
import statistics # For _generate_conformed_segments

# To avoid conflicts, import module itself or alias to prevent name clashes
from src import tools as tools_module 
from src import video as video_module
from src import audioCorrelation # For correlate, get_delay_fidelity, second_correlation
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

def process(list_not_compatible_video_input_output, already_compared_dict, 
            current_dict_file_path_obj, common_sync_language, 
            length_time_for_initial_offset_calc, 
            audio_params_for_wav, tools_tmpFolder, 
            audio_params_for_final_encode): # Added audio_params_for_final_encode
    sys.stderr.write("CHIMERIC_PROCESSOR: Chimeric processing started.\n")
    if not current_dict_file_path_obj: return {}

    V_abs_best = None; set_bad_video_paths = set()
    for video_path_1, comparisons in already_compared_dict.items():
        for video_path_2, path1_is_better in comparisons.items():
            if path1_is_better is True: set_bad_video_paths.add(video_path_2)
            elif path1_is_better is False: set_bad_video_paths.add(video_path_1)
    possible_best_paths = list(set(current_dict_file_path_obj.keys()) - set_bad_video_paths)
    if not possible_best_paths:
        if len(current_dict_file_path_obj) == 1: V_abs_best = list(current_dict_file_path_obj.values())[0]
        elif current_dict_file_path_obj: V_abs_best = list(current_dict_file_path_obj.values())[0]; sys.stderr.write(f"CHIMERIC_PROCESSOR: WARNING - No clear best. Using {V_abs_best.filePath} as fallback.\n")
        else: return current_dict_file_path_obj
    elif len(possible_best_paths) > 1:
        possible_best_paths.sort(); V_abs_best = current_dict_file_path_obj[possible_best_paths[0]]; sys.stderr.write(f"CHIMERIC_PROCESSOR: WARNING - Multiple best. Using {V_abs_best.filePath}.\n")
    else: V_abs_best = current_dict_file_path_obj[possible_best_paths[0]]

    if not V_abs_best:
        sys.stderr.write("CHIMERIC_PROCESSOR: CRITICAL ERROR - V_abs_best could not be identified.\n")
        for video_path in current_dict_file_path_obj.keys(): list_not_compatible_video_input_output.append(video_path)
        return current_dict_file_path_obj
    sys.stderr.write(f"CHIMERIC_PROCESSOR: V_abs_best identified as {V_abs_best.filePath}\n")

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
            else:
                sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to assemble chimeric MKV for {original_path}.\n")
                list_not_compatible_video_input_output.append(original_path)
    sys.stderr.write(f"CHIMERIC_PROCESSOR: Processing finished. Incompatible: {list_not_compatible_video_input_output}\n")
    return updated_dict_file_path_obj
