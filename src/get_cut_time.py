import os
import tempfile
import shutil
import math
import subprocess
from statistics import mean, stdev

from frame_compare import FrameComparer

class get_cut_time(Thread):
    """
    New get_cut_time:
      - params same as before plus output_path (path to the final .mkv chimera)
      - finds precise cut (frame-level when fps equal using phash-based frame_compare,
        else falls back to audio second_correlation)
      - builds chimera audio tracks per target-track policy described by the user
      - produces an mkv (no video) at output_path containing audio chimera tracks and all subs from the target
      - does not include the audio track used for fidelity calculation inside the chimera mkv
    """
    def __init__(self, main_video_obj, video_obj_to_cut, begin_in_second, audioParam, language, lenghtTime, lenghtTimePrepare, list_cut_begin_length, time_by_test_best_quality_converted, output_path, debug=False):
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
        self.output_path = output_path
        self.debug = debug

        # Will be filled on success
        self.result = {
            "cut": None,                 # dict with 'type': 'add'|'delete', 'start_frame','end_frame', 'start_time','end_time'
            "audio_chimeras": [],        # list of generated audio files (path + metadata)
            "subtitle_files": [],        # extracted subtitle files
            "mkv_path": None
        }

    def run(self):
        try:
            cut_info = self.get_first_delay_and_gap()  # high-level: find interval to inspect
            if cut_info is None:
                raise Exception("No cutoff zone found")

            # choose best audio pair used for fidelity (we exclude it from chimera)
            fidelity_pair = self._select_best_audio_pair_for_fidelity()
            # build precise zone and attempt frame-level detection
            precise = self._refine_cut_with_frames_or_audio(cut_info, fidelity_pair)

            # Build chimera audios for each target audio track except the one used for fidelity
            audio_outputs = self._build_chimeric_audios(precise, fidelity_pair)

            # Extract subtitle tracks from target and create final mkv
            subs = self._extract_subtitles_from_target()
            mkv_path = self._build_final_mkv(audio_outputs, subs)

            # fill result
            self.result["cut"] = precise
            self.result["audio_chimeras"] = audio_outputs
            self.result["subtitle_files"] = subs
            self.result["mkv_path"] = mkv_path

        except Exception as e:
            # don't crash the whole program — rethrow/collect where you want
            traceback.print_exc()
            raise

    # ---------- helper methods ----------

    def get_first_delay_and_gap(self):
        """
        Use existing audio fidelity scans to detect a coarse zone where fidelity drops or delays change.
        Returns dict with approximate start_time,end_time (seconds), estimated_gap_frames (signed),
        and 'type' in {'add','delete'} relative to target (positive gap => added frames in target).
        """
        delay_Fidelity_Values = get_delay_fidelity(self.main_video_obj, self.video_obj_to_cut, self.lenghtTime)
        # sanity check: all audios must match per earlier contract
        keys = list(delay_Fidelity_Values.keys())
        if not keys:
            return None
        # evaluate per-segment fidelity (mean over number_cut segments)
        per_segment = []  # list of (segment_index, mean_fidelity, delays_set)
        segments = len(delay_Fidelity_Values[keys[0]])
        for seg_i in range(segments):
            fidelities = []
            delays = set()
            for key in keys:
                fv = delay_Fidelity_Values[key][seg_i]
                fidelities.append(fv[0])
                delays.add(fv[2])
            per_segment.append((seg_i, mean(fidelities), delays))

        # adaptive threshold: base_mean - alpha*std, but never below 0.85; if data small fallback to 0.90
        fid_values = [p[1] for p in per_segment]
        if len(fid_values) >= 2:
            mu = mean(fid_values)
            try:
                sigma = stdev(fid_values)
            except:
                sigma = 0.0
        else:
            mu = mean(fid_values) if fid_values else 1.0
            sigma = 0.0
        threshold = max(0.85, min(0.95, mu - 0.75 * sigma))
        # also ensure not higher than 0.95
        threshold = min(threshold, 0.95)

        # find contiguous ranges where fidelity < threshold OR delays vary (indicates a gap)
        bad_ranges = []
        cur = None
        for seg_i, fid, delays in per_segment:
            bad = fid < threshold or (len(delays) > 1)
            if bad:
                if cur is None:
                    cur = [seg_i, seg_i]
                else:
                    cur[1] = seg_i
            else:
                if cur is not None:
                    bad_ranges.append(tuple(cur))
                    cur = None
        if cur is not None:
            bad_ranges.append(tuple(cur))

        if not bad_ranges:
            # nothing suspicious
            return None

        # choose the biggest bad range (most likely the real gap)
        bad_range = max(bad_ranges, key=lambda r: r[1] - r[0])
        seg_start, seg_end = bad_range

        # convert segment positions to seconds via video.generate_cut_with_begin_length info
        # reuse list_cut_begin_length which is a list of [start,end] per generate_cut...
        # Our cut segments probably correspond to video.number_cut slices in video.generate_cut...
        # Build time window that covers seg_start-1 .. seg_end+1 segments (safety margin)
        seg0 = max(0, seg_start - 1)
        seg1 = min(len(self.list_cut_begin_length) - 1, seg_end + 1)
        start_time_str = self.list_cut_begin_length[seg0][0]
        end_time_str = self.list_cut_begin_length[seg1][1]

        # Convert HH:MM:SS(.ms) => seconds (helper)
        def hhmmss_to_seconds(s):
            if "." in s:
                base, ms = s.split(".")
            else:
                base, ms = s, "0"
            h, m, sec = [int(x) for x in base.split(":")]
            return h * 3600 + m * 60 + sec + float("0." + ms) if ms != "0" else h * 3600 + m * 60 + sec

        start_sec = hhmmss_to_seconds(start_time_str)
        end_sec = hhmmss_to_seconds(end_time_str)

        # rough gap estimate via delays on the segment centers:
        # compute mean delay over the bad segments
        delays_collected = []
        for key in keys:
            for seg_i in range(seg_start, seg_end + 1):
                delays_collected.append(delay_Fidelity_Values[key][seg_i][2])
        # we'll use the difference between majority delay values to estimate how many ms/frames
        if delays_collected:
            # choose majority value
            from collections import Counter
            cnt = Counter(delays_collected)
            major_delay = cnt.most_common(1)[0][0]
        else:
            major_delay = 0

        # approximate type: if target is ahead (positive) => it likely has added frames (add), else delete
        # we can't be 100% — keep 'tentative' label
        estimated_type = "add" if major_delay > 0 else "delete"

        # Return coarse window
        return {
            "seg_start": seg_start,
            "seg_end": seg_end,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "threshold": threshold,
            "estimated_type": estimated_type,
            "major_delay_ms": major_delay
        }

    def _select_best_audio_pair_for_fidelity(self):
        """
        Select the audio pair (key 'i-j') used for fidelity: choose the pair with highest mean fidelity across its slices.
        This pair will be excluded from chimera mkv.
        """
        df = get_delay_fidelity(self.main_video_obj, self.video_obj_to_cut, self.lenghtTime)
        if not df:
            return None
        best_pair = None
        best_score = -1.0
        for key, values in df.items():
            vals = [v[0] for v in values]
            if vals:
                s = mean(vals)
                if s > best_score:
                    best_score = s
                    best_pair = key
        # return "i-j" string and indices
        if best_pair is None:
            return None
        i_str, j_str = best_pair.split("-")
        return {"pair": best_pair, "i": int(i_str), "j": int(j_str), "score": best_score}

    def _refine_cut_with_frames_or_audio(self, coarse, fidelity_pair):
        """
        If FPS identical: frame-level phash search around coarse window to find exact frames.
        If FPS differs or frame method fails: use second_correlation (audio) fallback.

        Returns dict:
          {
           'type': 'add'/'delete',
           'start_time','end_time' (seconds),
           'start_frame','end_frame', 'fps'
          }
        """
        # determine fps
        fps_main = self.main_video_obj.get_fps()
        fps_cut = self.video_obj_to_cut.get_fps()

        # compute search band width from coarse interval length and delay
        coarse_len = max(0.5, coarse["end_sec"] - coarse["start_sec"])
        # estimate number of frames in coarse window for target video framerate
        fps_use = fps_cut if fps_cut is not None else (fps_main if fps_main is not None else 25.0)
        estimated_frames = int(math.ceil(coarse_len * fps_use))
        # adapt BAND_WIDTH and MAX_SEARCH_FRAMES as requested: +10% of gap possible (we use coarse_len)
        BAND_WIDTH = max(1, int(estimated_frames * 0.20) )  # 20% of region
        MAX_SEARCH_FRAMES = max(8, int(estimated_frames * 0.30) )

        # Attempt frame-based refine only if both have fps and almost equal:
        if fps_main is not None and fps_cut is not None and abs(fps_main - fps_cut) < 1e-6:
            # frame-based approach
            fc = FrameComparer(self.main_video_obj.filePath, self.video_obj_to_cut.filePath,
                               coarse["start_sec"], coarse["end_sec"],
                               fps=int(fps_use),
                               band_width=BAND_WIDTH,
                               max_search_frames=MAX_SEARCH_FRAMES,
                               debug=self.debug)
            frame_result = fc.find_scene_gap_requirements(before_common=2, after_common=3)
            if frame_result is not None:
                # frame_result contains start_frame,end_frame (in target coordinates) and times
                return {
                    "type": coarse["estimated_type"],
                    "start_frame": frame_result["start_frame"],
                    "end_frame": frame_result["end_frame"],
                    "start_time": frame_result["start_time"],
                    "end_time": frame_result["end_time"],
                    "fps": fps_use,
                    "method": "phash_frames"
                }
            # else fall through to audio fallback

        # Fallback to audio refinement: use second_correlation around coarse area to find precise ms offset
        # We'll extract small slices around coarse and run second_correlation already available
        tmp_dir = tempfile.mkdtemp(prefix="cut_audio_refine_")
        try:
            # prepare cut lengths
            pad = min(4, int(coarse_len)) + 1  # seconds padding
            # main slice start and cut start using get_begin_time_with_millisecond helper from video.py
            # For simplicity use ffmpeg to extract WAVs of the coarse range + pad
            start_main = max(0, coarse["start_sec"] - pad)
            dur = coarse["end_sec"] - coarse["start_sec"] + 2 * pad
            main_wav = os.path.join(tmp_dir, "main.wav")
            cut_wav = os.path.join(tmp_dir, "cut.wav")
            ffmpeg = tools.software["ffmpeg"]
            cmd_main = [ffmpeg, "-y", "-ss", f"{start_main}", "-t", f"{dur}", "-i", self.main_video_obj.filePath,
                        "-vn", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", main_wav]
            tools.launch_cmdExt(cmd_main)
            cmd_cut = [ffmpeg, "-y", "-ss", f"{start_main}", "-t", f"{dur}", "-i", self.video_obj_to_cut.filePath,
                       "-vn", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", cut_wav]
            tools.launch_cmdExt(cmd_cut)
            # second_correlation expects temp files like earlier use:
            # It returns (value, frame_shift) tuples; we call second_correlation(main_wav, cut_wav)
            val = second_correlation(main_wav, cut_wav)
            # val = (correlation_value, offset_frames_or_ms?) -> in earlier code second_correlation returns a tuple (score, offset_frames)
            # We assume offset in frames or ms. The existing code uses mean(delay_second_method)*1000 previously.
            # Here try to interpret: many existing callers do round(delay_second_method*1000)
            # So if val[1] is offset frames (?) handle cautiously. We'll place offset_ms variable:
            try:
                offset_frames = val[1]
                # If offset_frames is in frames, convert to seconds using fps_use
                offset_seconds = float(offset_frames) / float(fps_use) if fps_use else 0.0
            except Exception:
                offset_seconds = 0.0

            # Rough locate start and end times in target using offset_seconds
            start_time = coarse["start_sec"] + offset_seconds
            end_time = coarse["end_sec"] + offset_seconds
            # Map to frames
            start_frame = int(round(start_time * fps_use))
            end_frame = int(round(end_time * fps_use))
            return {
                "type": coarse["estimated_type"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_time,
                "end_time": end_time,
                "fps": fps_use,
                "method": "audio_second_correlation"
            }

        finally:
            shutil.rmtree(tmp_dir)

    def _build_chimeric_audios(self, precise, fidelity_pair):
        """
        For each audio track in the target:
          - check if main has identical track for this language via first audio correlation method;
          - if identical -> do not create chimera for that language (we will reuse main's)
          - else -> build chimera audio by cutting around precise start/end time:
               - if 'delete' -> remove region from target audio (concatenate before+after)
               - if 'add' -> remove the extra frames from target (same logic)
               - If need to fill with main audio: use main's same-language track to fill.
          - re-encode to the original codec/bitrate of the target audio track
        Returns list of dicts: {"track_index":..., "file":path, "language":..., "codec":..., "original_streaminfo":...}
        """
        tmp_dir = tempfile.mkdtemp(prefix="chimeras_")
        outputs = []
        try:
            # time boundaries in seconds
            s_time = precise["start_time"]
            e_time = precise["end_time"]
            fps = precise["fps"]
            # for each audio in target file (assuming data structure video_obj_to_cut.audios[language] exists)
            # If language group missing, fallback to enumerating all audio tracks in video_obj_to_cut.audios
            target_audios = []
            if (self.language in self.video_obj_to_cut.audios):
                target_audios = self.video_obj_to_cut.audios[self.language]
            else:
                # pick all audios irrespective of language
                for lang, auds in self.video_obj_to_cut.audios.items():
                    target_audios.extend(auds)

            # find which main audio is same using correlate mean fidelity test (reuse get_delay_fidelity per-audio)
            # We'll compare each target audio track to all main audio tracks of same language and see if any pair is "identical" (mean fidelity >= 0.90)
            for aud_meta in target_audios:
                track_file_index = aud_meta.get("StreamOrder", None)
                track_lang = aud_meta.get("Language", self.language)
                # Skip the pair used for fidelity calculation (fidelity_pair indicates an audio pair i-j)
                # we only have fidelity_pair["j"] as target index if its 'j' matches current audio's audio_pos_file maybe.
                # As heuristic: if aud_meta['audio_pos_file']==fidelity_pair['j'] then skip chimera (user asked fidelity track not included)
                if fidelity_pair is not None and "j" in fidelity_pair and aud_meta.get("audio_pos_file", None) == fidelity_pair["j"]:
                    # do not produce chimera for the fidelity audio (we will not include it in chimera mkv)
                    continue

                # check if main has identical audio for this language
                identical_found = False
                if track_lang in self.main_video_obj.audios:
                    # compare each main track with this target track using first correlation (correlate)
                    for main_aud in self.main_video_obj.audios[track_lang]:
                        # Prepare temp WAV extracts for quick compare (short samples)
                        # Use audioParam passed earlier to generate WAVs
                        tmp_main = os.path.join(tmp_dir, f"main_{track_lang}_{main_aud.get('audio_pos_file',0)}.wav")
                        tmp_target = os.path.join(tmp_dir, f"target_{track_lang}_{aud_meta.get('audio_pos_file',0)}.wav")
                        # Extract short clips around begin_in_second for comparison
                        t_start = max(0, self.begin_in_second - 1)
                        t_dur = max(3, self.lenghtTime)
                        # Ask video object to extract audio in part if it exposes such method — else ffmpeg directly
                        # Use ffmpeg to be independent
                        ffmpeg = tools.software["ffmpeg"]
                        cmd_m = [ffmpeg, "-y", "-ss", str(t_start), "-t", str(t_dur), "-i", self.main_video_obj.filePath,
                                 "-map", f"0:{main_aud['StreamOrder']}", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", tmp_main]
                        cmd_t = [ffmpeg, "-y", "-ss", str(t_start), "-t", str(t_dur), "-i", self.video_obj_to_cut.filePath,
                                 "-map", f"0:{aud_meta['StreamOrder']}", "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", tmp_target]
                        tools.launch_cmdExt(cmd_m)
                        tools.launch_cmdExt(cmd_t)
                        # correlate() returns (score, other info...) per your audioCorrelation module
                        try:
                            corr = correlate(tmp_main, tmp_target, self.lenghtTime)
                            # correlate returns a tuple (score, ?, offset) for slices (used earlier). We consider mean of returned values
                            if isinstance(corr, (list, tuple)):
                                # support both forms
                                scores = [c[0] for c in corr] if isinstance(corr[0], (list, tuple)) else [corr[0]]
                                avg_score = mean(scores)
                            else:
                                avg_score = float(corr)
                        except Exception:
                            avg_score = 0.0
                        if avg_score >= 0.90:
                            identical_found = True
                            break

                if identical_found:
                    # no chimera required
                    continue

                # else: create chimera audio for this track
                # Strategy: cut target audio into before-gap and after-gap, optionally fill gap by main audio if available
                # We'll produce PCM WAV by concatenation then re-encode to original codec & bitrate from aud_meta.
                before_file = os.path.join(tmp_dir, f"before_{aud_meta['StreamOrder']}.wav")
                after_file = os.path.join(tmp_dir, f"after_{aud_meta['StreamOrder']}.wav")
                parts = []

                ffmpeg = tools.software["ffmpeg"]
                # times in seconds
                # We add a tiny slop (0.05s) to avoid frame rounding issues
                slop = 0.05
                # Choose safe before & after ranges: [start_of_file .. s_time) and (e_time .. end_of_file]
                start_before = max(0.0, s_time - 0.5)
                dur_before = max(0.0, (s_time - start_before))
                start_after = e_time + 0.01
                # For after duration, take up to 5s after gap as minimal sample (or up to end)
                dur_after = 5.0

                # extract before part
                if dur_before > 0.01:
                    cmd_b = [ffmpeg, "-y", "-ss", str(start_before), "-t", str(dur_before + slop),
                             "-i", self.video_obj_to_cut.filePath, "-map", f"0:{aud_meta['StreamOrder']}",
                             "-acodec", "pcm_s16le", "-ac", str(aud_meta.get("Channels", 2)), "-ar", str(aud_meta.get("SamplingRate", 44100)),
                             before_file]
                    tools.launch_cmdExt(cmd_b)
                    parts.append(before_file)
                # decide filler: try to use same-language main audio if available (first match)
                filler_file = None
                if self.language in self.main_video_obj.audios:
                    # pick first main audio track of that language
                    m_aud = self.main_video_obj.audios[self.language][0]
                    filler_file = os.path.join(tmp_dir, f"filler_{m_aud['StreamOrder']}.wav")
                    # extract slice corresponding to the gap region mapped to main: compute times using delays if available
                    # For simplicity, extract the same time region from main around same begin_in_second
                    main_start = max(0.0, s_time - 0.1)
                    cmd_f = [ffmpeg, "-y", "-ss", str(main_start), "-t", str(max(0.1, (e_time - s_time) + 0.5)),
                             "-i", self.main_video_obj.filePath, "-map", f"0:{m_aud['StreamOrder']}",
                             "-acodec", "pcm_s16le", "-ac", str(m_aud.get("Channels", 2)), "-ar", str(m_aud.get("SamplingRate", 44100)),
                             filler_file]
                    tools.launch_cmdExt(cmd_f)

                # extract after part
                cmd_a = [ffmpeg, "-y", "-ss", str(start_after), "-t", str(dur_after),
                         "-i", self.video_obj_to_cut.filePath, "-map", f"0:{aud_meta['StreamOrder']}",
                         "-acodec", "pcm_s16le", "-ac", str(aud_meta.get("Channels", 2)), "-ar", str(aud_meta.get("SamplingRate", 44100)),
                         after_file]
                tools.launch_cmdExt(cmd_a)
                parts.append(after_file)

                # create concatenation list for ffmpeg concat demuxer
                concat_txt = os.path.join(tmp_dir, f"concat_{aud_meta['StreamOrder']}.txt")
                with open(concat_txt, "w") as fh:
                    # If deletion: parts are before + after (we removed the gap)
                    # If addition: we should also remove the added frames (equal procedure)
                    for p in parts:
                        fh.write(f"file '{p}'\n")

                concatenated_wav = os.path.join(tmp_dir, f"concat_{aud_meta['StreamOrder']}.wav")
                cmd_concat = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", concatenated_wav]
                tools.launch_cmdExt(cmd_concat)

                # Now re-encode the concatenated WAV into the original codec/bitrate of the target audio
                # aud_meta likely contains Format, codec, SamplingRate, Channels, and bitrate info accessible via video.get_bitrate
                target_codec = aud_meta.get("codec", aud_meta.get("Format", "aac")).lower()
                # mapping common codecs; more complex mapping may be needed in your environment
                out_ch = str(aud_meta.get("Channels", 2))
                out_ar = str(aud_meta.get("SamplingRate", 44100))
                # try to preserve bitrate
                try:
                    orig_bitrate = str(video.get_bitrate(aud_meta))
                except Exception:
                    orig_bitrate = None

                # target encoded file path:
                ext = "mka" if target_codec in ("dts","ac3","aac","mp3") else "mka"
                encoded_out = os.path.join(tmp_dir, f"chimera_{aud_meta['StreamOrder']}.{ext}")

                ffmpeg_enc = [ffmpeg, "-y", "-i", concatenated_wav]
                # map channels and sampling rate
                ffmpeg_enc += ["-ac", out_ch, "-ar", out_ar]
                # choose codec options
                if target_codec in ("aac", "aac_latm"):
                    ffmpeg_enc += ["-c:a", "aac"]
                    if orig_bitrate:
                        ffmpeg_enc += ["-b:a", orig_bitrate]
                elif target_codec == "ac3":
                    ffmpeg_enc += ["-c:a", "ac3"]
                    if orig_bitrate:
                        ffmpeg_enc += ["-b:a", orig_bitrate]
                elif target_codec in ("dts",):
                    ffmpeg_enc += ["-c:a", "dts"]
                    if orig_bitrate:
                        ffmpeg_enc += ["-b:a", orig_bitrate]
                elif target_codec in ("mp3",):
                    ffmpeg_enc += ["-c:a", "libmp3lame"]
                    if orig_bitrate:
                        ffmpeg_enc += ["-b:a", orig_bitrate]
                else:
                    # fallback: keep it as AAC or FLAC
                    ffmpeg_enc += ["-c:a", "flac"]

                ffmpeg_enc.append(encoded_out)
                tools.launch_cmdExt(ffmpeg_enc)

                outputs.append({
                    "track_index": aud_meta.get("StreamOrder"),
                    "file": encoded_out,
                    "language": aud_meta.get("Language", self.language),
                    "codec": target_codec,
                    "original": aud_meta
                })

            return outputs

        finally:
            # keep tmp_dir for debug; if not debug remove it
            if not self.debug:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

    def _extract_subtitles_from_target(self):
        """
        Extract all subtitle tracks from target video into temporary files (one-by-one),
        return list of (stream_order, file_path, codec).
        Uses ffmpeg -map to extract. If ffmpeg cannot extract a particular codec, we still try to copy.
        """
        tmp_dir = tempfile.mkdtemp(prefix="subs_")
        out_subs = []
        try:
            # need to know subtitle stream indexes; try to use ffprobe via tools.launch_cmdExt
            ffprobe = tools.software.get("ffprobe", "ffprobe")
            cmd_probe = [ffprobe, "-v", "error", "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language", "-of", "json", self.video_obj_to_cut.filePath]
            rc, out = tools.launch_cmdExt(cmd_probe, want_stdout=True)
            import json as _json
            streams = []
            try:
                info = _json.loads(out[1].decode("utf-8"))
                streams = info.get("streams", [])
            except Exception:
                streams = []

            sub_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
            for s in sub_streams:
                idx = s.get("index")
                codec = s.get("codec_name")
                lang = (s.get("tags") or {}).get("language", "und")
                # choose extension
                ext = "ass" if codec and codec.lower() != "srt" else "srt"
                outp = os.path.join(tmp_dir, f"sub_{idx}.{ext}")
                ffmpeg = tools.software["ffmpeg"]
                cmd = [ffmpeg, "-y", "-i", self.video_obj_to_cut.filePath, "-map", f"0:{idx}", "-c", "copy", outp]
                # ffmpeg will produce the right container
                tools.launch_cmdExt(cmd)
                out_subs.append({"stream_index": idx, "file": outp, "codec": codec, "language": lang})

            return out_subs
        except Exception:
            # on error, return []
            return []
        finally:
            # if not debug we keep tmp_dir removal to mkv building cleanup
            pass

    def _build_final_mkv(self, audio_outputs, subtitle_files):
        """
        Build the final MKV without video:
          - include each audio_outputs entry with language flags
          - include subtitle_files (copied)
        Creates mkv at self.output_path (overwrites).
        """
        ffmpeg = tools.software["ffmpeg"]
        mkvmerge = tools.software.get("mkvmerge", "mkvmerge")
        tmp_dir = tempfile.mkdtemp(prefix="mkv_build_")
        try:
            # Strategy:
            # 1) Start from target file but ask mkvmerge to exclude its audio (we want to replace) but keep subtitles.
            #    mkvmerge global option --no-audio will exclude audio tracks from all input files, so we'll add the target file first with --no-audio to keep video+subs (but we will then remove video using --no-video on the final combine)
            # Simpler and robust approach:
            #   - Create an intermediate file that contains ONLY subtitles from the original target (no audio, no video).
            sub_only = os.path.join(tmp_dir, "subs_only.mkv")
            cmd_subs = [mkvmerge, "-o", sub_only, "--no-audio", "--no-video", self.video_obj_to_cut.filePath]
            tools.launch_cmdExt(cmd_subs)

            # 2) Build arguments to add each audio file produced (with language flags if present)
            mkv_args = [mkvmerge, "-o", self.output_path]
            # Add subtitle-only container first (it contains only subtitles)
            mkv_args.append(sub_only)
            # Then add chimera audio files, with language tags when available
            for ao in audio_outputs:
                lang = ao.get("language", "und")
                # --language 0:eng sets language for track 0 of this file
                mkv_args += ["--language", f"0:{lang}", ao["file"]]

            # Run mkvmerge to produce the final file
            tools.launch_cmdExt(mkv_args)
            return self.output_path
        finally:
            # cleanup tmp_dir only if not debug
            if not self.debug:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass