'''One ffmpeg job per kept track of a candidate, run on the shared ffmpeg pool (decision: audio-reencode).'''

import re
import sys
from threading import Thread
from decimal import Decimal
from os import path,rename

import pysubs2

import tools
import video
import audio_reencode
import audio_integrity

TRACK_JOB_TIMEOUT = 3600
WARNING_KEYWORDS = ["error", "invalid", "corrupt", "dts", "pts", "non monoton", "discarding", "out of order"]


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

    def run(self):
        try:
            list_convert_process = []
            list_file_to_merge = []
            for sub in self.sub_track_to_convert_or_keep:
                list_convert_process.append(video.ffmpeg_pool_audio_convert.apply_async(subtitle_track_job (self.video_obj, path.basename(self.tmp_file), sub, self.delay_to_put, self.duration_best_video, self.is_master)))
                list_file_to_merge.append(path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp.mkv"))

            for audio in self.audio_track_to_convert_or_keep:
                list_convert_process.append(video.ffmpeg_pool_audio_convert.apply_async(audio_track_job (self.video_obj, path.basename(self.tmp_file), audio, self.delay_to_put, self.duration_best_video, self.is_master)))
                list_file_to_merge.append(path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp.mkv"))

            for result in list_convert_process:
                result.get()
            try:
                tools.launch_cmdExt_with_timeout_reload(merge_cmd, 2, 1200)
            except Exception as e:
                if not str(e).rstrip().endswith("Return code: 1"):
                    raise TrackJobError(f"mkvmerge of the track files of {file_path} failed: {e}") from e
                sys.stderr.write(f"[mkvmerge WARN] track files of {file_path}: {e}\n")
        except TrackJobError:
            remove_files(list_file_to_merge)
            raise

        try:
            merge_cmd = [tools.software["mkvmerge"], "-o", tmp_file]
            merge_cmd.extend(list_file_to_merge)
            stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(merge_cmd, 2, 3600)
            remove_files(list_file_to_merge)
            self.new_video = video.video(path.dirname(tmp_file), path.basename(tmp_file))
            self.new_video.need_one_audio_track = False
            self.new_video.get_mediadata()

            #Add some final checkup
        except Exception as e:
            self.error = e

    def get(self):
        self.join()
        if self.error is not None:
            raise self.error

def check_delay_retention_sub(new_track, original_tracks, original_video_path, delay_to_put, sentence):
    sys.stderr.write(f"[FFmpeg DEBUG] {sentence} {original_video_path} track {original_tracks['StreamOrder']}: delay {original_tracks.get('Delay', '0')} -> {new_track.get('Delay', '0')}\n")
    if Decimal(str(original_tracks.get('Delay', '0'))) != Decimal(str(new_track.get('Delay', '0'))):
        sys.stderr.write(f"[FFmpeg WARN] {sentence} Delay retention for {original_video_path} track {original_tracks['StreamOrder']}: original delay {original_tracks.get('Delay', '0')}, new delay {new_track.get('Delay', '0')}\n")


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
    tools.launch_cmdExt(cmd_extract)

def print_ffmpeg_message(stderror, video_obj, cmd):
    stderr_text = stderror.decode("utf-8", errors="ignore")

    keywords = ["error", "invalid", "corrupt", "dts", "pts", "non monoton", "discarding", "out of order"]
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
    return ["-i", file_path]


def ffmpeg_job(genpts, input_args, output_args, avoid_negative_ts, duration, out_path):
    cmd = [tools.software["ffmpeg"], "-y", "-nostdin", "-err_detect", "crccheck+bitstream+buffer"]
    if genpts:
        cmd.extend(["-fflags", "+genpts+igndts"])
    cmd.extend(["-analyzeduration", "1000M", "-probesize", "1000M", "-threads", "5", "-vn"] + input_args + output_args)
    if avoid_negative_ts:
        cmd.extend(["-avoid_negative_ts", "disabled"])
    return cmd + ["-strict", "-2", "-t", str(duration), "-max_interleave_delta", "0",
                  "-max_muxing_queue_size", "16384", "-f", "matroska", out_path]


def run_track_cmd(label, cmd):
    try:
        stdout, stderror, exit_code = tools.launch_cmdExt_with_timeout_reload(cmd, 2, TRACK_JOB_TIMEOUT)
    except Exception as e:
        message = f"Track job failed for {label}: {e}"
        sys.stderr.write(f"[FFmpeg ERROR] {message}\n")
        raise TrackJobError(message) from e
    warnings = [line for line in stderror.decode("utf-8", errors="ignore").splitlines()
                if (not line.startswith("Input #")) and (not re.search(r"attachment:\s*none", line, re.IGNORECASE))
                and any(kw in line.lower() for kw in WARNING_KEYWORDS)]
    if warnings:
        sys.stderr.write(f"[FFmpeg WARN] Track job for {label}\n{cmd}\n")
        for line in warnings:
            sys.stderr.write(f"  {line}\n")
    return stdout


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
    text = run_track_cmd(label, [tools.software["ffmpeg"], "-v", "error", "-nostdin", "-i", file_path,
                                 "-map", f"0:{stream}", "-c:s", "srt", "-f", "srt", "pipe:1"]).decode("utf-8", errors="ignore")
    if not text.strip():
        return []
    try:
        return pysubs2.SSAFile.from_string(text).events
    except pysubs2.exceptions.FormatAutodetectionError as e:
        raise TrackJobError(f"Track job failed for {label}: stream {stream} of {file_path} is not readable by pysubs2: {e}") from e


def audio_track_job(video_obj, file_base_name, audio, delay_to_put, duration_best_video, on_master):
    tmp_file_extract = path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp_extr.mkv")
    extract_stream(video_obj, "audio", audio['StreamOrder'], tmp_file_extract)
    tmp_file_convert = path.join(tools.tmpFolder,f"{file_base_name}_{audio['StreamOrder']}_tmp.mkv")

    if is_already_converted(audio, delay_to_put, on_master):
        rename(tmp_file_extract,tmp_file_convert)



    label = f"audio track {audio['StreamOrder']} of {file_path}"
    output_args = ["-map", f"0:{int(audio['StreamOrder'])}", "-map_metadata", "0", "-copy_unknown",
                   "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"]
    reencoded = audio_reencode.generate_audio_ffmpeg_conversion_cmd(output_args, audio, delay_to_put, stream=":a:0",
                                                                    path=file_path)
    output_args.extend(["-metadata:s:a:0", f"{tools.MASTER_PASS_TAG}={duration_best_video}"])
    run_track_cmd(label, ffmpeg_job(not have_pgs, delay_input_args(delay_to_put, file_path), output_args, have_pgs,
                                    duration_best_video, out_path))
    if reencoded:
        try:
            audio_integrity.check_reencoded_track(file_path, int(audio["StreamOrder"]), out_path, delay_to_put)
        except audio_integrity.AudioReencodeIntegrityError as e:
            raise TrackJobError(str(e)) from e


def subtitle_track_job(video_obj, file_base_name, sub, delay_to_put, duration_best_video, on_master):
    tmp_file_extract = path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp_extr.mkv")
    extract_stream(video_obj, "subtitle", sub['StreamOrder'], tmp_file_extract)
    tmp_file_convert = path.join(tools.tmpFolder,f"{file_base_name}_{sub['StreamOrder']}_tmp.mkv")

    if is_already_converted(sub, delay_to_put, on_master):
        rename(tmp_file_extract,tmp_file_convert)
        # Maybe add a test on the file with a "-err_detect", "crccheck+bitstream+buffer" ?

    else:
        cmd_convert = [tools.software["ffmpeg"], "-err_detect", "crccheck+bitstream+buffer", "-fflags", "+genpts+igndts",
                        "-analyzeduration", "1000M", "-probesize", "1000M",
                        "-threads", "1", "-vn"]

        cmd_convert.extend(delay_input_args(delay_to_put, tmp_file_extract))
        cmd_convert.extend(["-map", "0:a?", "-map", "0:s?", "-map_metadata", "0", "-copy_unknown",
                "-movflags", "use_metadata_tags", "-c", "copy"])
        
        codec = sub["Format"].lower()
        if 'ffmpeg_to_convert' in sub:
            cmd_convert.append(f"-c:s")
            cmd_convert.append(sub['ffmpeg_to_convert'][1])
        elif codec in tools.sub_type_not_encodable:
            cmd_convert.extend(["-copyts", "-avoid_negative_ts", "disabled"])
            cmd_convert.remove("-fflags")
            cmd_convert.remove("+genpts+igndts")
        elif codec in tools.sub_type_near_srt:
            cmd_convert.extend(["-c:s", "srt"])
        else:
            cmd_convert.extend(["-c:s", "ass"])


        cmd_first_pass.extend([f"-metadata:s:s:0", f"{tools.NOT_MY_FIRST_TAG}={duration_best_video}"])
        
        cmd_convert.extend(["-t", duration_best_video, "-max_interleave_delta", "0", "-max_muxing_queue_size", "16384", tmp_file_convert])
        stdout, stderror, exitCode = tools.launch_cmdExt_with_timeout_reload(cmd_convert, 2, 3600)

        print_ffmpeg_message(stderror, video_obj, cmd)

        tools.file_remove(tmp_file_extract)

        label = f"subtitle track {sub['StreamOrder']} of {video_obj.filePath}"
        shift_ms = float(delay_to_put)
        duration_ms = float(duration_best_video) * 1000
        expected = [(event.start + shift_ms, event.end + shift_ms, event.text)
                    for event in read_subtitle_events(label, video_obj.filePath, sub["StreamOrder"])
                    if 0 <= event.start + shift_ms < duration_ms]
        produced = [(event.start, event.end, event.text) for event in read_subtitle_events(label, out_path, 0)]
        if len(expected) == 0 or len(produced) == 0:
            matches = len(expected) == len(produced)
        else:
            matches = all(want[2] == got[2] and abs(want[0] - got[0]) <= 1 and abs(want[1] - got[1]) <= 1
                        for want, got in ((expected[0], produced[0]), (expected[-1], produced[-1])))
        if not matches:
            raise TrackJobError(f"Track job failed for {label}: first/last cues expected "
                                f"{expected[:1] + expected[-1:]}, produced {produced[:1] + produced[-1:]}")

        #Find a efficient way to use check_delay_retention_sub(new_track, original_tracks, original_video_path, delay_to_put, sentence)

def wait_track_jobs(convert_processes):
    """Wait for every candidate's ResultThread; log a TrackJobError in tools.logs, in this thread, then raise it."""
    for convert_process in convert_processes:
        try:
            convert_process.get()
        except TrackJobError as e:
            tools.log_always(f"[FFmpeg ERROR] {e}\n")
            raise
