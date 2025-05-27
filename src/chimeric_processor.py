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
from threading import Thread # Added for ChimericFileProcessorThread

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


# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

# At module level:
# from threading import Thread # This is already at the top of the file.
# ... other imports ...

def _cleanup_local_temp_files(files_list): # Defined helper (moved from _assemble_chimeric_mkv for broader use if needed)
    for f_path in files_list:
        if os.path.exists(f_path):
            try: os.remove(f_path)
            except OSError as e: sys.stderr.write(f"CHIMERIC_PROCESSOR: Error removing temp file {f_path}: {e}\n")

# Make sure this function is defined before ChimericFileProcessorThread if it's called directly,
# or ensure it's correctly defined at module level.

def _assemble_chimeric_mkv_from_plan(video_to_conform_obj, # For video stream
                                     segment_assembly_plan, 
                                     best_video_obj, # Reference, though plan dictates sources
                                     sync_language, # For naming output file primarily
                                     all_subtitle_langs, # List of language codes to assemble for
                                     audio_final_encode_params, 
                                     tmp_folder, 
                                     out_folder_for_chimeric_file):
    sys.stderr.write(f"CHIMERIC_PROCESSOR: Assembling MKV from plan for {video_to_conform_obj.filePath}\n")
    
    local_temp_files = []
    audio_segment_files_for_concat = []
    # Initialize subtitle_segment_files_for_concat for all languages in the plan
    # This ensures that even if a language has only blank subs, its concat list is created.
    subtitle_segment_files_for_concat = {lang: [] for lang in all_subtitle_langs}

    try:
        for idx, segment_info in enumerate(segment_assembly_plan):
            segment_duration_s = segment_info['duration_s']
            
            # Audio Segment
            # Make temp audio file name unique using video names and segment index
            temp_audio_segment_path = os.path.join(tmp_folder, 
                f"plan_audio_{os.path.basename(segment_info['audio_source_object_path']).replace('.','_')}_{idx}.wav")
            local_temp_files.append(temp_audio_segment_path)

            if not _extract_audio_segment_for_correlation(
                segment_info['audio_source_object_path'], 
                segment_info['audio_source_stream_order'], 
                tools_module.format_time_ffmpeg_dot(segment_info['audio_source_start_s']), 
                tools_module.format_time_ffmpeg_dot(segment_duration_s), # Duration of the segment
                temp_audio_segment_path
            ):
                sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to extract audio segment {idx} from {segment_info['audio_source_object_path']}\n")
                _cleanup_local_temp_files(local_temp_files)
                return None
            audio_segment_files_for_concat.append(temp_audio_segment_path)

            # Subtitle Segments
            for sub_segment_detail in segment_info['subtitle_segments_info']:
                lang = sub_segment_detail['lang']
                # Make temp sub file name unique
                sub_source_basename = os.path.basename(sub_segment_detail.get('sub_source_object_path', 'blank_sub')).replace('.', '_')
                temp_sub_segment_path = os.path.join(tmp_folder, 
                    f"plan_sub_{sub_source_basename}_{lang}_{idx}.srt")
                local_temp_files.append(temp_sub_segment_path)

                if sub_segment_detail['sub_generate_blank']:
                    if not _generate_blank_sub_segment(sub_segment_detail['sub_source_duration_s'], temp_sub_segment_path):
                        sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to generate blank sub for lang {lang}, seg {idx}\n")
                        _cleanup_local_temp_files(local_temp_files)
                        return None
                else:
                    if not _extract_specific_subtitle_stream_segment(
                        sub_segment_detail['sub_source_object_path'],
                        sub_segment_detail['sub_source_stream_order'],
                        sub_segment_detail['sub_source_start_s'],
                        sub_segment_detail['sub_source_duration_s'],
                        temp_sub_segment_path
                    ):
                        # If extraction fails, generate blank as fallback
                        sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to extract sub lang {lang}, seg {idx}. Generating blank instead.\n")
                        if not _generate_blank_sub_segment(sub_segment_detail['sub_source_duration_s'], temp_sub_segment_path):
                            _cleanup_local_temp_files(local_temp_files)
                            return None
                
                if lang not in subtitle_segment_files_for_concat: # Should have been initialized
                     subtitle_segment_files_for_concat[lang] = []
                subtitle_segment_files_for_concat[lang].append(temp_sub_segment_path)
        
        # Create Concat List Files
        audio_concat_list_path = os.path.join(tmp_folder, f"plan_audio_concat_{video_to_conform_obj.fileBaseName}.txt")
        with open(audio_concat_list_path, 'w', encoding='utf-8') as f:
            for p in audio_segment_files_for_concat: f.write(f"file '{os.path.basename(p)}'\n")
        local_temp_files.append(audio_concat_list_path)

        subtitle_concat_paths_map = {}
        sorted_langs_for_ffmpeg = sorted(list(all_subtitle_langs)) # For deterministic ffmpeg map order

        for lang in sorted_langs_for_ffmpeg:
            if lang in subtitle_segment_files_for_concat and subtitle_segment_files_for_concat[lang]:
                sub_concat_path = os.path.join(tmp_folder, f"plan_sub_{lang}_concat_{video_to_conform_obj.fileBaseName}.txt")
                with open(sub_concat_path, 'w', encoding='utf-8') as f:
                    for p in subtitle_segment_files_for_concat[lang]: f.write(f"file '{os.path.basename(p)}'\n")
                subtitle_concat_paths_map[lang] = sub_concat_path
                local_temp_files.append(sub_concat_path)
            else: # Ensure all languages are considered even if they end up being all blank segments
                sys.stderr.write(f"CHIMERIC_PROCESSOR: No subtitle segments generated for language {lang} for {video_to_conform_obj.fileBaseName}. It might result in an empty track or no track.\n")


        # Construct FFmpeg Command
        safe_basename = re.sub(r'[^\w.-]', '_', video_to_conform_obj.fileBaseName) # Sanitize for output filename
        chimeric_mkv_path = os.path.join(out_folder_for_chimeric_file, f"{safe_basename}_chimeric_plan_{sync_language}.mkv")

        ffmpeg_cmd = [tools_module.software['ffmpeg'], '-y', '-nostdin',
                      '-i', video_to_conform_obj.filePath, # Input 0: Original video for video stream
                      '-f', 'concat', '-safe', '0', '-i', audio_concat_list_path] # Input 1: Concatenated audio

        input_idx_counter = 2 # Starts from 2 (0 is video, 1 is audio)
        subtitle_map_commands = []
        output_subtitle_stream_idx = 0

        # Video stream from original video_to_conform_obj
        # Ensure 'video' key and 'StreamOrder' exist
        if not video_to_conform_obj.video or 'StreamOrder' not in video_to_conform_obj.video:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Video stream information missing for {video_to_conform_obj.filePath}\n")
            _cleanup_local_temp_files(local_temp_files)
            return None
        video_stream_order = video_to_conform_obj.video['StreamOrder']
        
        ffmpeg_cmd.extend(['-map', f"0:{video_stream_order}"]) # Map video from Input 0
        ffmpeg_cmd.extend(['-map', '1:a:0']) # Map audio from Input 1
        ffmpeg_cmd.extend(['-metadata:s:a:0', f"language={sync_language}"]) # Set language for the main audio track

        for lang in sorted_langs_for_ffmpeg:
            if lang in subtitle_concat_paths_map:
                ffmpeg_cmd.extend(['-f', 'concat', '-safe', '0', '-i', subtitle_concat_paths_map[lang]])
                subtitle_map_commands.extend(['-map', f"{input_idx_counter}:s:0"])
                subtitle_map_commands.extend([f"-metadata:s:s:{output_subtitle_stream_idx}", f"language={lang}"])
                input_idx_counter += 1
                output_subtitle_stream_idx += 1
        
        ffmpeg_cmd.extend(subtitle_map_commands)
        
        ffmpeg_cmd.extend(['-c:v', 'copy'])
        ffmpeg_cmd.extend(['-c:a', audio_final_encode_params.get('codec', 'flac')])
        if audio_final_encode_params.get('params'):
            ffmpeg_cmd.extend(audio_final_encode_params['params'])
        
        if output_subtitle_stream_idx > 0 : # Only add subtitle codec if there are subtitles
            ffmpeg_cmd.extend(['-c:s', 'ass'])

        ffmpeg_cmd.extend(['-shortest', chimeric_mkv_path])

        sys.stdout.write(f"CHIMERIC_PROCESSOR: Assembling Chimeric MKV (from plan): {' '.join(ffmpeg_cmd)}\n")
        _stdout_ff, stderr_ff, exit_code_ff = tools_module.launch_cmdExt(ffmpeg_cmd, cwd=tmp_folder)

        if exit_code_ff == 0:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Successfully generated chimeric MKV (from plan): {chimeric_mkv_path}\n")
            return chimeric_mkv_path
        else:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Error generating chimeric MKV (from plan): {stderr_ff.decode('utf-8', errors='ignore')}\n")
            _cleanup_local_temp_files(local_temp_files) # Also cleanup if ffmpeg fails
            return None

    except Exception as e:
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Exception in _assemble_chimeric_mkv_from_plan for {video_to_conform_obj.filePath}: {e}\n{traceback.format_exc()}\n")
        _cleanup_local_temp_files(local_temp_files)
        return None
    finally:
        # Ensure cleanup happens, though specific failures above might have already cleaned.
        # This is a catch-all.
        _cleanup_local_temp_files(local_temp_files)

class ChimericFileProcessorThread(Thread):

class ChimericFileProcessorThread(Thread):
    def __init__(self, best_video_obj, video_to_conform_obj, sync_lang, 
                 segment_proc_duration_s, all_vids_dict, audio_wav_prms, 
                 tmp_fldr, audio_final_enc_prms, all_sub_langs_available_list,
                 out_folder_for_chimeric_file, length_time_initial_offset_calc_param): # Added length_time_initial_offset_calc_param
        Thread.__init__(self)
        self.best_video_obj = best_video_obj
        self.video_to_conform_obj = video_to_conform_obj
        self.sync_lang = sync_lang
        self.segment_proc_duration_s = segment_proc_duration_s # This is SEGMENT_DURATION_S from the plan
        self.all_vids_dict = all_vids_dict
        self.audio_wav_prms = audio_wav_prms # These are audio_params_for_wav in process()
        self.tmp_fldr = tmp_fldr
        self.audio_final_enc_prms = audio_final_enc_prms
        self.all_sub_langs_available_list = all_sub_langs_available_list
        self.out_folder_for_chimeric_file = out_folder_for_chimeric_file
        self.length_time_initial_offset_calc = length_time_initial_offset_calc_param # Stored
        self.result = None
        self.original_video_path = video_to_conform_obj.filePath 
        self.temp_files_to_cleanup = []

    def run(self):
        sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Started.\n")
        try:
            # --- a. Initial Coarse Offset Calculation ---
            initial_offset_audio_params = {'Format': "WAV", 'codec': "pcm_s16le", 'Channels': "2"}
            
            best_duration_str = self.best_video_obj.video.get('Duration')
            conform_duration_str = self.video_to_conform_obj.video.get('Duration')

            if not best_duration_str or not conform_duration_str:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Missing duration for one or both videos.\n")
                self.result = None; return
            try:
                best_duration = float(best_duration_str)
                conform_duration = float(conform_duration_str)
            except ValueError:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Invalid duration format.\n")
                self.result = None; return

            min_duration_s = min(best_duration, conform_duration)
            begin_s_for_initial_cuts = 5.0
            if min_duration_s < (begin_s_for_initial_cuts + video_module.number_cut * self.length_time_initial_offset_calc):
                begin_s_for_initial_cuts = 0.0
            
            list_cut_begin_length_initial = video_module.generate_cut_with_begin_length(
                begin_s_for_initial_cuts, 
                self.length_time_initial_offset_calc, 
                tools_module.format_time_ffmpeg_dot(self.length_time_initial_offset_calc)
            )
            
            initial_cut_temp_files_best_paths = self.best_video_obj.extract_audio_in_part(self.sync_lang, initial_offset_audio_params, cutTime=list_cut_begin_length_initial, asDefault=False) 
            if initial_cut_temp_files_best_paths: self.temp_files_to_cleanup.extend(initial_cut_temp_files_best_paths)
            
            initial_cut_temp_files_conform_paths = self.video_to_conform_obj.extract_audio_in_part(self.sync_lang, initial_offset_audio_params, cutTime=list_cut_begin_length_initial, asDefault=False)
            if initial_cut_temp_files_conform_paths: self.temp_files_to_cleanup.extend(initial_cut_temp_files_conform_paths)

            if not initial_cut_temp_files_best_paths or not initial_cut_temp_files_conform_paths:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Failed to extract initial audio cuts for offset calculation.\n")
                self.result = None; return

            delay_fidelity_values = audioCorrelation.get_delay_fidelity(self.best_video_obj, self.video_to_conform_obj, self.length_time_initial_offset_calc, initial_cut_temp_files_best_paths, initial_cut_temp_files_conform_paths) 
            
            # Cleanup initial cut files immediately after use
            _cleanup_local_temp_files(initial_cut_temp_files_best_paths)
            _cleanup_local_temp_files(initial_cut_temp_files_conform_paths)
            self.temp_files_to_cleanup = [f for f in self.temp_files_to_cleanup if f not in initial_cut_temp_files_best_paths and f not in initial_cut_temp_files_conform_paths]

            if not delay_fidelity_values:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Initial offset: No results from get_delay_fidelity.\n")
                self.result = None; return
            all_collected_delays_ms = [df[2] for df_list in delay_fidelity_values.values() for df in df_list if df_list]
            if not all_collected_delays_ms:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Initial offset: No delays collected.\n")
                self.result = None; return
            if len(all_collected_delays_ms) > 1:
                try:
                    delay_stdev = statistics.stdev(all_collected_delays_ms)
                    if delay_stdev > 100: 
                        sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Initial offset: Unstable delays (stdev > 100ms).\n")
                        self.result = None; return
                except statistics.StatisticsError: pass 
            
            initial_offset_ms = Decimal(str(round(statistics.mean(all_collected_delays_ms))))
            sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Initial offset calculated: {initial_offset_ms}ms\n")
            current_dynamic_offset_s = float(initial_offset_ms / Decimal(1000))

            # --- b. Sequential Segment-by-Segment Analysis ---
            FIDELITY_THRESHOLD = 0.80
            INCOMPATIBILITY_THRESHOLD_PERCENT = 50.0
            OFFSET_CONSISTENCY_WINDOW_S = 0.5 

            segment_assembly_plan = []
            total_bad_audio_duration_s = 0.0
            current_main_time_s = 0.0

            best_audio_tracks = self.best_video_obj.audios.get(self.sync_lang)
            conform_audio_tracks = self.video_to_conform_obj.audios.get(self.sync_lang)
            if not best_audio_tracks or not conform_audio_tracks:
                 sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Sync language '{self.sync_lang}' audio tracks not found.\n")
                 self.result = None; return
            best_corr_audio_stream_order = best_audio_tracks[0]['StreamOrder']
            conform_corr_audio_stream_order = conform_audio_tracks[0]['StreamOrder']
            
            idx = 0
            while current_main_time_s < best_duration:
                actual_segment_duration_s = min(self.segment_proc_duration_s, best_duration - current_main_time_s)
                if actual_segment_duration_s < 1.0: break 

                segment_info = {
                    'main_start_s': current_main_time_s, 'duration_s': actual_segment_duration_s,
                    'audio_source_object_path': None, 'audio_source_stream_order': None,
                    'audio_extract_start_s': None, 'audio_extract_duration_s': None,
                    'subtitles_plan': {} 
                }
                main_segment_start_str = tools_module.format_time_ffmpeg_dot(current_main_time_s)
                main_segment_duration_str = tools_module.format_time_ffmpeg_dot(actual_segment_duration_s)
                
                main_audio_segment_path = os.path.join(self.tmp_fldr, f"ch_main_corr_seg_{self.best_video_obj.fileBaseName}_{idx}.wav")
                self.temp_files_to_cleanup.append(main_audio_segment_path)
                other_segment_start_abs_s = current_main_time_s - current_dynamic_offset_s
                other_audio_segment_path = os.path.join(self.tmp_fldr, f"ch_other_corr_seg_{self.video_to_conform_obj.fileBaseName}_{idx}.wav")
                self.temp_files_to_cleanup.append(other_audio_segment_path)

                audio_segment_good = False; reason_for_bad = "unknown"

                if not _extract_audio_segment_for_correlation(self.best_video_obj.filePath, best_corr_audio_stream_order, main_segment_start_str, main_segment_duration_str, main_audio_segment_path):
                    reason_for_bad = "main_audio_extract_fail"
                elif (other_segment_start_abs_s + actual_segment_duration_s < 0.1) or (other_segment_start_abs_s >= conform_duration - 0.1):
                    reason_for_bad = "other_audio_out_of_bounds"
                else:
                    other_segment_start_clipped_s = max(0.0, other_segment_start_abs_s)
                    other_segment_duration_clipped_s = min(actual_segment_duration_s, conform_duration - other_segment_start_clipped_s)
                    if other_segment_duration_clipped_s < 1.0: reason_for_bad = "other_audio_clip_too_short"
                    elif not _extract_audio_segment_for_correlation(self.video_to_conform_obj.filePath, conform_corr_audio_stream_order, tools_module.format_time_ffmpeg_dot(other_segment_start_clipped_s), tools_module.format_time_ffmpeg_dot(other_segment_duration_clipped_s), other_audio_segment_path):
                        reason_for_bad = "other_audio_extract_fail"
                    else:
                        try:
                            fidelity, _, _ = audioCorrelation.correlate(main_audio_segment_path, other_audio_segment_path, other_segment_duration_clipped_s)
                            _shifted_file, precise_offset_s_fft_raw = audioCorrelation.second_correlation(main_audio_segment_path, other_audio_segment_path)
                            segment_correction_s = -precise_offset_s_fft_raw if _shifted_file == main_audio_segment_path else precise_offset_s_fft_raw
                            if fidelity >= FIDELITY_THRESHOLD and abs(segment_correction_s) < OFFSET_CONSISTENCY_WINDOW_S:
                                current_dynamic_offset_s -= segment_correction_s
                                segment_info.update({'audio_source_object_path': self.video_to_conform_obj.filePath, 'audio_source_stream_order': conform_corr_audio_stream_order, 'audio_extract_start_s': other_segment_start_clipped_s, 'audio_extract_duration_s': other_segment_duration_clipped_s, 'type': 'good', 'reason': 'good_match', 'fidelity': fidelity, 'segment_correction_s': segment_correction_s, 'other_start_abs_s': other_segment_start_abs_s, 'other_duration_clipped_s': other_segment_duration_clipped_s}) 
                                audio_segment_good = True
                            else: reason_for_bad = 'low_fidelity' if fidelity < FIDELITY_THRESHOLD else 'offset_drift_corr_loop'; segment_info.update({'fidelity': fidelity, 'segment_correction_s': segment_correction_s})
                        except Exception as e_corr: sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Correlation exception: {e_corr}\n"); reason_for_bad = "correlation_exception"
                
                if not audio_segment_good:
                    segment_info.update({'audio_source_object_path': self.best_video_obj.filePath, 'audio_source_stream_order': best_corr_audio_stream_order, 'audio_extract_start_s': current_main_time_s, 'audio_extract_duration_s': actual_segment_duration_s, 'type': 'bad', 'reason': reason_for_bad})
                    total_bad_audio_duration_s += actual_segment_duration_s
                
                for sub_lang in self.all_sub_langs_available_list:
                    sub_plan = {'generate_blank': True, 'sub_source_object_path': None, 'sub_source_stream_order': None, 'sub_source_start_s': None, 'sub_source_duration_s': actual_segment_duration_s}
                    potential_sub_sources = []
                    if audio_segment_good: 
                        potential_sub_sources.append({'obj': self.video_to_conform_obj, 'start_s': segment_info['audio_extract_start_s'], 'dur_s': segment_info['audio_extract_duration_s']})
                        potential_sub_sources.append({'obj': self.best_video_obj, 'start_s': current_main_time_s, 'dur_s': actual_segment_duration_s})
                    else: 
                        potential_sub_sources.append({'obj': self.best_video_obj, 'start_s': current_main_time_s, 'dur_s': actual_segment_duration_s})
                    for vid_path_key, vid_obj_candidate in self.all_vids_dict.items():
                        if vid_obj_candidate.filePath not in [s['obj'].filePath for s in potential_sub_sources]:
                            potential_sub_sources.append({'obj': vid_obj_candidate, 'start_s': current_main_time_s, 'dur_s': actual_segment_duration_s}) 
                    
                    for source_candidate in potential_sub_sources:
                        candidate_obj = source_candidate['obj']
                        if hasattr(candidate_obj, 'subtitles') and candidate_obj.subtitles and sub_lang in candidate_obj.subtitles and candidate_obj.subtitles[sub_lang]:
                            sub_track_info = candidate_obj.subtitles[sub_lang][0]
                            sub_plan.update({'generate_blank': False, 'sub_source_object_path': candidate_obj.filePath, 'sub_source_stream_order': sub_track_info['StreamOrder'], 'sub_source_start_s': source_candidate['start_s'], 'sub_source_duration_s': source_candidate['dur_s']})
                            break 
                    segment_info['subtitles_plan'][sub_lang] = sub_plan
                
                segment_assembly_plan.append(segment_info)
                current_main_time_s += actual_segment_duration_s
                idx += 1
            
            # Cleanup per-segment WAVs after loop
            _cleanup_local_temp_files([f for f in self.temp_files_to_cleanup if "ch_main_corr_seg" in f or "ch_other_corr_seg" in f])
            self.temp_files_to_cleanup = [f for f in self.temp_files_to_cleanup if "ch_main_corr_seg" not in f and "ch_other_corr_seg" not in f]


            if best_duration > 0 and (total_bad_audio_duration_s / best_duration * 100.0) > INCOMPATIBILITY_THRESHOLD_PERCENT:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Incompatible due to high bad audio duration: {total_bad_audio_duration_s / best_duration * 100.0:.2f}%\n")
                self.result = None; return

            sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Proceeding to assemble MKV. Plan has {len(segment_assembly_plan)} segments.\n")
            mkv_path = _assemble_chimeric_mkv_from_plan(
                self.video_to_conform_obj, segment_assembly_plan, self.best_video_obj, self.sync_lang,
                self.all_sub_langs_available_list, self.audio_final_enc_prms, self.tmp_fldr, self.out_folder_for_chimeric_file
            )
            if mkv_path and os.path.exists(mkv_path):
                chimeric_video_obj = video_module.video(self.out_folder_for_chimeric_file, os.path.basename(mkv_path))
                chimeric_video_obj.get_mediadata()
                chimeric_video_obj.calculate_md5_streams_split()
                chimeric_video_obj.delays[self.sync_lang] = Decimal(0) 
                self.result = chimeric_video_obj
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Successfully created chimeric MKV: {mkv_path}\n")
            else:
                sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: MKV assembly failed or file not found.\n")
                self.result = None
        except Exception as e:
            sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Exception in run: {e}\n")
            import traceback; traceback.print_exc(file=sys.stderr)
            self.result = None
        finally:
            _cleanup_local_temp_files(self.temp_files_to_cleanup) 
            sys.stderr.write(f"CHIMERIC_THREAD [{self.original_video_path}]: Finished run method.\n")


def process(list_not_compatible_video_input_output, dict_with_video_quality_logic, 
            current_dict_file_path_obj, common_sync_language, 
            length_time_for_initial_offset_calc, 
            audio_params_for_wav, tools_tmpFolder, 
            audio_params_for_final_encode):

    sys.stderr.write("CHIMERIC_PROCESSOR: Chimeric processing started (new threaded version).\n")

    if not current_dict_file_path_obj:
        sys.stderr.write("CHIMERIC_PROCESSOR: No video files to process.\n")
        return {} 

    set_bad_video = set()
    if dict_with_video_quality_logic: 
        for video_path_file_key, dict_with_results_val in dict_with_video_quality_logic.items():
            if dict_with_results_val: 
                for other_video_path_file_key2, is_video_path_file_key_better in dict_with_results_val.items():
                    if is_video_path_file_key_better: 
                        set_bad_video.add(other_video_path_file_key2)
                    else: 
                        set_bad_video.add(video_path_file_key)
    
    possible_best_paths = list(set(current_dict_file_path_obj.keys()) - set_bad_video)
    
    V_abs_best_path = None
    if not possible_best_paths: 
        sys.stderr.write("CHIMERIC_PROCESSOR: No best video candidates after exclusion. Checking all original files.\n")
        if not dict_with_video_quality_logic and len(current_dict_file_path_obj) == 1:
            V_abs_best_path = list(current_dict_file_path_obj.keys())[0]
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Only one video, selecting {V_abs_best_path} as best by default.\n")
        elif current_dict_file_path_obj: 
            V_abs_best_path = list(current_dict_file_path_obj.keys())[0]
            sys.stderr.write(f"CHIMERIC_PROCESSOR: WARNING - No clear best video from quality logic. Defaulting to first video: {V_abs_best_path}\n")
        else: 
            sys.stderr.write("CHIMERIC_PROCESSOR: CRITICAL ERROR - No videos available to select a best one.\n")
            for vid_path in current_dict_file_path_obj.keys(): 
                if vid_path not in list_not_compatible_video_input_output:
                    list_not_compatible_video_input_output.append(vid_path)
            return current_dict_file_path_obj 
    else: 
        for p in possible_best_paths:
            if p in current_dict_file_path_obj: 
                V_abs_best_path = p
                break 
    
    if not V_abs_best_path: 
        sys.stderr.write("CHIMERIC_PROCESSOR: CRITICAL ERROR - Could not determine V_abs_best_path.\n")
        for vid_path in current_dict_file_path_obj.keys():
            if vid_path not in list_not_compatible_video_input_output:
                list_not_compatible_video_input_output.append(vid_path)
        return current_dict_file_path_obj

    V_abs_best = current_dict_file_path_obj[V_abs_best_path]
    sys.stderr.write(f"CHIMERIC_PROCESSOR: V_abs_best identified as {V_abs_best.filePath}\n")

    if not hasattr(V_abs_best, 'chimeric_files') or V_abs_best.chimeric_files is None:
        V_abs_best.chimeric_files = []

    updated_dict_file_path_obj = {V_abs_best.filePath: V_abs_best}

    all_subtitle_languages_available = set()
    for video_obj_iter in current_dict_file_path_obj.values():
        if hasattr(video_obj_iter, 'subtitles') and video_obj_iter.subtitles:
            all_subtitle_languages_available.update(video_obj_iter.subtitles.keys())
    
    threads = []
    segment_processing_duration_s = 60.0 

    for original_path, video_to_conform in current_dict_file_path_obj.items():
        if video_to_conform.filePath == V_abs_best.filePath:
            continue

        sys.stderr.write(f"CHIMERIC_PROCESSOR: Preparing thread for {video_to_conform.filePath}\n")
        thread = ChimericFileProcessorThread(
            best_video_obj=V_abs_best,
            video_to_conform_obj=video_to_conform,
            sync_lang=common_sync_language,
            segment_proc_duration_s=segment_processing_duration_s,
            all_vids_dict=current_dict_file_path_obj, # Pass the whole dict for subtitle search
            audio_wav_prms=audio_params_for_wav,
            tmp_fldr=tools_tmpFolder,
            audio_final_enc_prms=audio_params_for_final_encode,
            all_sub_langs_available_list=list(all_subtitle_languages_available),
            out_folder_for_chimeric_file=tools_tmpFolder # Chimeric MKVs are temporary
        )
        threads.append(thread)
        thread.start()

    # Join threads and collect results
    sys.stderr.write(f"CHIMERIC_PROCESSOR: Launched {len(threads)} threads. Waiting for completion...\n")
    for thread_idx, thread_item in enumerate(threads):
        thread_item.join()
        sys.stderr.write(f"CHIMERIC_PROCESSOR: Thread {thread_idx} for {thread_item.original_video_path} finished.\n")
        if thread_item.result and isinstance(thread_item.result, video_module.video): # video_module.video is the type of video objects
            chimeric_video_obj = thread_item.result
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Successfully processed {thread_item.original_video_path} into {chimeric_video_obj.filePath}\n")
            
            # Ensure chimeric_files list exists
            if V_abs_best.chimeric_files is None: V_abs_best.chimeric_files = []
            V_abs_best.chimeric_files.append(chimeric_video_obj)
            updated_dict_file_path_obj[chimeric_video_obj.filePath] = chimeric_video_obj
            
            # Remove from incompatible list if it was there
            if thread_item.original_video_path in list_not_compatible_video_input_output:
                list_not_compatible_video_input_output.remove(thread_item.original_video_path)
        else:
            sys.stderr.write(f"CHIMERIC_PROCESSOR: Failed to process {thread_item.original_video_path}. Marking as incompatible.\n")
            if thread_item.original_video_path not in list_not_compatible_video_input_output:
                list_not_compatible_video_input_output.append(thread_item.original_video_path)

    sys.stderr.write(f"CHIMERIC_PROCESSOR: All threads finished. Incompatible list: {list_not_compatible_video_input_output}\n")
    sys.stderr.write(f"CHIMERIC_PROCESSOR: Final V_abs_best.chimeric_files: {len(V_abs_best.chimeric_files) if V_abs_best.chimeric_files else 0} files.\n")
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
    # This class is largely superseded by ChimericFileProcessorThread and its planned internal logic.
    # It can be removed in a future step once ChimericFileProcessorThread fully implements
    # the synchronization and segment generation logic.


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