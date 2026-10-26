'''One ffmpeg job per kept track of a candidate, run on the shared ffmpeg pool (decision: audio-reencode).'''

import json
import re
import sys
from threading import Thread
from decimal import Decimal
from os import path,rename

import pysubs2

import tools
import video
import audio_reencode
import integrity
import audio_integrity

ORIGINAL_TAG="ORIGINAL_FILE_STREAM"
PREVIOUS_STREAM_ID="VMSAM_PREVIOUS_STREAM_ID"

class TrackJobError(Exception):
    pass

class finalTransformations(Thread):
    def __init__(self, video_obj, tmp_file, audio_track_to_convert_or_keep, sub_track_to_convert_or_keep, delay_to_put, duration_best_video, is_master):
        Thread.__init__(self)
        self.video_obj = video_obj
        self.tmp_file = tmp_file
        self.audio_track_to_convert_or_keep = audio_track_to_convert_or_keep
        self.sub_track_to_convert_or_keep = sub_track_to_convert_or_keep
        self.delay_to_put = delay_to_put
        self.duration_best_video = duration_best_video
        self.is_master = is_master
        self.new_video = None
        self.error = None

    def run(self):
        file_base_name = path.splitext(path.basename(self.tmp_file))[0]
        list_convert_process = []
        list_file_to_merge = []
        try:
            for sub in self.sub_track_to_convert_or_keep:
                list_convert_process.append(video.ffmpeg_pool_audio_convert.apply_async(subtitle_track_job, (self.video_obj, file_base_name, sub, self.delay_to_put, self.duration_best_video, self.is_master)))
                list_file_to_merge.append(path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp.mkv"))

            for audio in self.audio_track_to_convert_or_keep:
                list_convert_process.append(video.ffmpeg_pool_audio_convert.apply_async(audio_track_job, (self.video_obj, file_base_name, audio, self.delay_to_put, self.duration_best_video, self.is_master)))
                list_file_to_merge.append(path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp.mkv"))

            first_error = None
            for result in list_convert_process:
                try:
                    result.get()
                except Exception as e:
                    if first_error is None:
                        first_error = e
            if first_error is not None:
                raise first_error

            cmd_merge = [tools.software["mkvmerge"], "-o", self.tmp_file]
            cmd_merge.extend(list_file_to_merge)
            try:
                tools.launch_cmdExt_with_timeout_reload(cmd_merge, 2, 2400)
            except Exception as e:
                tools.mkv_merge_error_detections(e,self.video_obj.filePath)

            remove_files(list_file_to_merge)

            self.new_video = video.video(path.dirname(self.tmp_file), path.basename(self.tmp_file))
            self.new_video.need_one_audio_track = False
            self.new_video.get_mediadata()

            list_new_track = [new_track for new_track in self.new_video.mediadata['media']['track'] if 'StreamOrder' in new_track]
            list_track = self.audio_track_to_convert_or_keep + self.sub_track_to_convert_or_keep
            if len(list_new_track) != len(list_track):
                raise TrackJobError(f"{self.tmp_file} of {self.video_obj.filePath} has {len(list_new_track)} tracks, {len(list_track)} expected")

            for language,audios in self.new_video.audios.items():
                for audio in audios:
                    check_track(audio, self.video_obj.filePath, list_track, self.delay_to_put)

            for language,subtitles in self.new_video.subtitles.items():
                for subtitle in subtitles:
                    check_track(subtitle, self.video_obj.filePath, list_track, self.delay_to_put)

            for language,audios in self.new_video.commentary.items():
                for audio in audios:
                    check_track(audio, self.video_obj.filePath, list_track, self.delay_to_put)
        
            for language,audios in self.new_video.audiodesc.items():
                for audio in audios:
                    check_track(audio, self.video_obj.filePath, list_track, self.delay_to_put)
        
        except Exception as e:
            remove_files(list_file_to_merge + [self.tmp_file])
            self.new_video = None
            self.error = e

    def get(self):
        self.join()
        if self.error is not None:
            raise self.error

def check_track(new_track, original_video_path, track_to_convert_or_keep, delay_to_put):
    old_track = track_to_convert_or_keep[int(new_track['StreamOrder'])]
    label = f"{new_track['@type'].lower()} track {old_track['StreamOrder']} of {original_video_path}"

    if not get_original_tag(new_track):
        raise TrackJobError(f"Final check failed for {label}: the produced track has no {ORIGINAL_TAG} tag")

    if new_track['@type'] != old_track['@type'] or new_track.get('Language', 'und') != old_track.get('Language', 'und'):
        raise TrackJobError(f"Final check failed for {label}: produced {new_track['@type']} in {new_track.get('Language', 'und')}, expected {old_track['@type']} in {old_track.get('Language', 'und')}")

    if not (new_track.get("extra") or {}).get(tools.NOT_MY_FIRST_TAG):
        raise TrackJobError(f"Final check failed for {label}: the produced track has no {tools.NOT_MY_FIRST_TAG} tag")

    if tools.dev:
        integrity.check_delay_retention_sub(new_track, old_track, original_video_path, delay_to_put, "Track jobs")

def get_original_tag(track):
    global ORIGINAL_TAG
    return track.get("extra", {}).get(ORIGINAL_TAG, {})

def extract_stream(video_obj, type_stream, id_stream, out_file):
    cmd_extract = [tools.software["mkvmerge"], "-o", out_file]
    if type_stream == "audio":
        cmd_extract.extend(["-D","-S","--no-global-tags", "-M", "-B", "--no-chapters",
                            "--audio-tracks", f"{id_stream}"])
    elif type_stream == "subtitle":
        cmd_extract.extend(["-D","-A","--no-global-tags", "-M", "-B", "--no-chapters",
                            "--subtitle-tracks", f"{id_stream}"])
    elif type_stream == "video":
        cmd_extract.extend(["-S","-A","--no-global-tags", "-M", "-B", "--no-chapters",
                            "--video-tracks", f"{id_stream}"])
    cmd_extract.extend([video_obj.filePath])
    try:
        tools.launch_cmdExt_with_timeout_reload(cmd_extract, 2, 2400)
    except Exception as e:
        tools.mkv_merge_error_detections(e,video_obj.filePath)

def print_ffmpeg_message(stderror, video_obj, cmd):
    stderr_text = stderror.decode("utf-8", errors="ignore")

    keywords = ["error", "invalid", "corrupt", "pts", "non monoton", "discarding", "out of order"]
    any_error = []
    for line in stderr_text.splitlines():
        if (not line.startswith("Input #")) and (not re.search(r"attachment:\s*none", line, re.IGNORECASE)) and any(kw in line.lower() for kw in keywords):
            any_error.append(line)

    if any_error:
        sys.stderr.write(f"[FFmpeg WARN] Message of a stream of {video_obj.filePath}\n")
        sys.stderr.write(f"{cmd}\n")
        for line in any_error:
            sys.stderr.write(f"  {line}\n")

def delay_input_args(delay_to_put, file_path):
    if delay_to_put > 0:
        return ["-itsoffset", f"{delay_to_put/Decimal('1000'):.6f}", "-i", file_path]
    elif delay_to_put < 0:
        return ["-i", file_path, "-ss", f"{delay_to_put/Decimal('1000')*Decimal(-1):.6f}"]
    else:
        return ["-i", file_path]

def remove_files(paths):
    for file_path in paths:
        if file_path and path.exists(file_path):
            tools.file_remove(file_path)

def is_already_converted(track, delay_to_put, on_master):
    """True when a master track with no delay to apply already carries NOT_MY_FIRST_TAG: no ffmpeg runs on it."""
    if delay_to_put != 0 or not on_master:
        return False
    return bool((track.get("extra") or {}).get(tools.NOT_MY_FIRST_TAG))

def read_subtitle_events(label, file_path, stream):
    stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload([tools.software["ffmpeg"], "-v", "error", "-nostdin", "-i", file_path,
                                                                          "-map", f"0:{stream}", "-c:s", "srt", "-f", "srt", "pipe:1"], 2, 3600)
    text = stdout.decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    try:
        return pysubs2.SSAFile.from_string(text).events
    except pysubs2.exceptions.FormatAutodetectionError as e:
        raise TrackJobError(f"Track job failed for {label}: stream {stream} of {file_path} is not readable by pysubs2: {e}") from e


def audio_track_job(video_obj, file_base_name, audio, delay_to_put, duration_best_video, on_master):
    tmp_file_extract = path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp_extr.mkv")
    tmp_file_convert = path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp.mkv")

    label = f"audio track {audio['StreamOrder']} of {video_obj.filePath}"
    try:
        extract_stream(video_obj, "audio", audio['StreamOrder'], tmp_file_extract)
        integrity.test_video_file_have_no_big_error(tmp_file_extract)
        if is_already_converted(audio, delay_to_put, on_master):
            rename(tmp_file_extract,tmp_file_convert)
            audio_integrity.check_reencoded_track(video_obj.filePath, int(audio["StreamOrder"]), tmp_file_convert, delay_to_put)
        else:
            cmd_convert = [tools.software["ffmpeg"], "-y", "-nostdin", "-err_detect", "crccheck+bitstream+buffer", "-fflags", "+genpts+igndts",
                            "-analyzeduration", "1000M", "-probesize", "1000M",
                            "-threads", "1", "-vn"]

            cmd_convert.extend(delay_input_args(delay_to_put, tmp_file_extract))
            cmd_convert.extend(["-map", "0:a?", "-map_metadata", "0", "-copy_unknown",
                    "-movflags", "use_metadata_tags", "-c", "copy", "-copyts"])

            if audio_reencode.is_lossless(audio) and audio_reencode.all_lossless_in_flac():
                cmd_convert.extend(audio_reencode.get_parameters_lossless(audio, "0"))
                if delay_to_put < 0:
                    cmd_convert.extend([f"-filter:a:0", audio_reencode.NEGATIVE_DELAY_FILTER])
            # elif audio["offset"] + delay_to_put > 0: # A possibility is to remove a part of the offset. To explore.
            elif delay_to_put < 0:
                cmd_convert.extend(audio_reencode.get_parameters_in_reencode(audio, "0", video_obj.filePath, filter_param_to_add=audio_reencode.NEGATIVE_DELAY_FILTER))
            
            cmd_convert.extend(["-metadata:s:a:0", f"{tools.NOT_MY_FIRST_TAG}={duration_best_video}"])
            if not get_original_tag(audio):
                cmd_convert.extend(["-metadata:s:a:0", add_original_informations(audio,video_obj.filePath)])

            cmd_convert.extend(["-strict", "-2", "-t", str(duration_best_video), "-max_interleave_delta", "0", "-max_muxing_queue_size", "16384", tmp_file_convert])
            stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(cmd_convert, 2, 3600)

            print_ffmpeg_message(stderror, video_obj, cmd_convert)

            tools.file_remove(tmp_file_extract)

            new_encode = video.video(path.dirname(tmp_file_convert), path.basename(tmp_file_convert))
            new_encode.get_mediadata()

            new_track = [track for track in new_encode.mediadata['media']['track'] if track['@type'] == 'Audio'][0]
            if Decimal(str(new_track.get('Duration', '0'))) > Decimal(str(duration_best_video)) + Decimal("1"):
                raise TrackJobError(f"Final check failed for {label}: produced duration {new_track.get('Duration')} s, longest allowed {duration_best_video} s")

            audio_integrity.check_reencoded_track(video_obj.filePath, int(audio["StreamOrder"]), tmp_file_convert, delay_to_put)
    except Exception as e:
        remove_files([tmp_file_extract, tmp_file_convert])
        if isinstance(e, TrackJobError):
            raise
        raise TrackJobError(f"Track job failed for {label}: {e}") from e


def subtitle_track_job(video_obj, file_base_name, sub, delay_to_put, duration_best_video, on_master):
    tmp_file_extract = path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp_extr.mkv")
    tmp_file_convert = path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp.mkv")

    label = f"subtitle track {sub['StreamOrder']} of {video_obj.filePath}"
    try:
        extract_stream(video_obj, "subtitle", sub['StreamOrder'], tmp_file_extract)

        if is_already_converted(sub, delay_to_put, on_master):
            rename(tmp_file_extract,tmp_file_convert)
            stdout, stderror, exitCode = tools.launch_cmdExt_no_test([tools.software["ffprobe"], "-v", "error", "-err_detect", "crccheck+bitstream+buffer",
                                                                      "-show_entries", "frame=pts_time", "-of", "csv=p=0", tmp_file_convert])
            if exitCode != 0 or stderror.strip():
                raise TrackJobError(f"Track job failed for {label}: the already converted track does not decode cleanly (exit {exitCode}): {stderror.decode('utf-8', errors='ignore').strip()}")

        else:
            cmd_convert = [tools.software["ffmpeg"], "-y", "-nostdin", "-err_detect", "crccheck+bitstream+buffer", "-fflags", "+genpts+igndts",
                            "-analyzeduration", "1000M", "-probesize", "1000M",
                            "-threads", "1", "-vn"]

            if 'ffmpeg_to_convert' in sub:
                cmd_convert.extend(["-c:s:0", sub["ffmpeg_to_convert"][0]])

            cmd_convert.extend(delay_input_args(delay_to_put, tmp_file_extract))
            cmd_convert.extend(["-map", "0:a?", "-map", "0:s?", "-map_metadata", "0", "-copy_unknown",
                    "-movflags", "use_metadata_tags", "-c", "copy"])

            codec = video.get_codec(sub)
            sub_encoder = None
            if 'ffmpeg_to_convert' in sub:
                cmd_convert.extend(["-c:s:0", sub["ffmpeg_to_convert"][1]])
            elif codec in tools.sub_type_not_encodable:
                cmd_convert.extend(["-copyts", "-avoid_negative_ts", "disabled"])
                cmd_convert.remove("-fflags")
                cmd_convert.remove("+genpts+igndts")
            elif codec in tools.sub_type_near_srt:
                cmd_convert.extend(["-c:s:0", "srt"])
            else:
                cmd_convert.extend(["-c:s:0", "ass"])

            if codec not in tools.sub_type_not_encodable and delay_to_put < 0:
                cmd_convert.extend(["-output_ts_offset", f"{delay_to_put/Decimal('1000'):.6f}"])

            cmd_convert.extend([f"-metadata:s:s:0", f"{tools.NOT_MY_FIRST_TAG}={duration_best_video}"])
            if not get_original_tag(sub):
                cmd_convert.extend([f"-metadata:s:s:0", add_original_informations(sub,video_obj.filePath)])

            cmd_convert.extend(["-t", str(duration_best_video), "-max_interleave_delta", "0", "-max_muxing_queue_size", "16384", tmp_file_convert])
            stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(cmd_convert, 2, 3600)

            print_ffmpeg_message(stderror, video_obj, cmd_convert)

            tools.file_remove(tmp_file_extract)

            if codec not in tools.sub_type_not_encodable:
                shift_ms = float(delay_to_put)
                duration_ms = float(duration_best_video) * 1000
                expected = [(event.start + shift_ms, event.end + shift_ms, event.text)
                            for event in read_subtitle_events(label, video_obj.filePath, sub["StreamOrder"])
                            if 0 <= event.start + shift_ms < duration_ms]
                produced = [(event.start, event.end, event.text) for event in read_subtitle_events(label, tmp_file_convert, 0)]
                matches = len(expected) == len(produced)
                if matches and expected:
                    matches = all(want[2] == got[2] and abs(want[0] - got[0]) <= 1 and abs(want[1] - got[1]) <= 1
                                for want, got in ((expected[0], produced[0]), (expected[-1], produced[-1])))
                if not matches:
                    raise TrackJobError(f"Track job failed for {label}: {len(expected)} cues expected, {len(produced)} produced, first/last cues expected "
                                        f"{expected[:1] + expected[-1:]}, produced {produced[:1] + produced[-1:]}")
    except Exception as e:
        remove_files([tmp_file_extract, tmp_file_convert])
        if isinstance(e, TrackJobError):
            raise
        raise TrackJobError(f"Track job failed for {label}: {e}") from e


def wait_track_jobs(convert_processes):
    """Wait for every candidate's finalTransformations; log a TrackJobError in tools.logs, in this thread, then raise it."""
    for convert_process in convert_processes:
        try:
            convert_process.get()
        except TrackJobError as e:
            tools.log_always(f"[FFmpeg ERROR] {e}\n")
            raise

def add_original_informations(data,file_path):
    global ORIGINAL_TAG
    data_to_save = {
        "original_position": int(data.get('StreamOrder', '0')),
        "original_file": path.basename(file_path)
    }
    data_to_save_string = json.dumps(data_to_save)
    return f"{ORIGINAL_TAG}={data_to_save_string}"