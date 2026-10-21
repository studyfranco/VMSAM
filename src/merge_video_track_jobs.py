'''One ffmpeg job per kept track of a candidate, run on the shared ffmpeg pool.'''

import json
import re
import subprocess
import sys
import threading
from decimal import Decimal
from os import path

import tools
import video
import audio_reencode
import audio_integrity

TRACK_JOB_TIMEOUT = 3600
WARNING_KEYWORDS = ["error", "invalid", "corrupt", "dts", "pts", "non monoton", "discarding", "out of order"]


class TrackJobError(Exception):
    pass


class ResultThread(threading.Thread):
    '''Run target(*args) in a thread whose get() re-raises its exception, like an AsyncResult.'''
    def __init__(self, target, args):
        super().__init__(daemon=True)
        self.job_target, self.job_args, self.error = target, args, None
        self.start()

    def run(self):
        try:
            self.job_target(*self.job_args)
        except Exception as e:
            self.error = e

    def get(self):
        self.join()
        if self.error is not None:
            raise self.error


def delay_input_args(delay_to_put, file_path):
    if delay_to_put > 0:
        return ["-itsoffset", f"{delay_to_put/Decimal('1000'):.6f}", "-i", file_path]
    elif delay_to_put < 0:
        return ["-i", file_path, "-ss", f"{delay_to_put/Decimal('1000')*Decimal(-1):.6f}"]
    return ["-i", file_path]


def vmsam_tmp_metadata(track, stream):
    position = int(track.get('StreamOrder', '0'))
    delay = float(track.get('Delay', '0'))
    return [f"-metadata:s:{stream}", "VMSAM_TMP=" + json.dumps({"original_position": position, "original_delay": delay})]


def ffmpeg_job(genpts, input_args, output_args, avoid_negative_ts, duration, out_path):
    cmd = [tools.software["ffmpeg"], "-y", "-nostdin", "-err_detect", "crccheck+bitstream+buffer"]
    if genpts:
        cmd.extend(["-fflags", "+genpts+igndts"])
    cmd.extend(["-analyzeduration", "1000M", "-probesize", "1000M", "-threads", "5", "-vn"] + input_args + output_args)
    if avoid_negative_ts:
        cmd.extend(["-avoid_negative_ts", "disabled"])
    return cmd + ["-strict", "-2", "-t", str(duration), "-max_interleave_delta", "0",
                  "-max_muxing_queue_size", "16384", "-f", "matroska", out_path]


def track_file_paths(audios, subs, tmp_prefix):
    """Return the per-track files of one candidate in mkvmerge order: audio tracks by StreamOrder, then the subtitles."""
    paths = [f"{tmp_prefix}_track_{int(audio['StreamOrder'])}.mka"
             for audio in sorted(audios, key=lambda x: int(x["StreamOrder"]))]
    if len(subs):
        paths.append(f"{tmp_prefix}_subtitles.mks")
    return paths


def merge_cmd_for_track_files(out_paths):
    """Return the mkvmerge arguments adding the per-track files of one candidate, chapters from the first only."""
    args = []
    for index, out_path in enumerate(out_paths):
        args.extend(["--no-global-tags", "-M", "-B"])
        if index:
            args.append("--no-chapters")
        args.append(out_path)
    return args


def run_track_job(label, cmds):
    for cmd in cmds:
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=TRACK_JOB_TIMEOUT)
        except subprocess.TimeoutExpired:
            message = f"Track job failed for {label}: timeout after {TRACK_JOB_TIMEOUT} s\n{' '.join(cmd)}\n"
            sys.stderr.write(f"[FFmpeg ERROR] {message}")
            raise TrackJobError(message)
        stderr_text = result.stderr.decode("utf-8", errors="ignore")
        if result.returncode != 0:
            message = (f"Track job failed for {label}: return code {result.returncode}\n{' '.join(cmd)}\n"
                       + "\n".join(stderr_text.splitlines()[-20:]) + "\n")
            sys.stderr.write(f"[FFmpeg ERROR] {message}")
            raise TrackJobError(message)
        warnings = [line for line in stderr_text.splitlines()
                    if (not line.startswith("Input #")) and (not re.search(r"attachment:\s*none", line, re.IGNORECASE))
                    and any(kw in line.lower() for kw in WARNING_KEYWORDS)]
        if warnings:
            sys.stderr.write(f"[FFmpeg WARN] Track job for {label}\n{cmd}\n")
            for line in warnings:
                sys.stderr.write(f"  {line}\n")


def remove_files(paths):
    for file_path in paths:
        if path.exists(file_path):
            tools.file_remove(file_path)


def generate_track_files(video_obj, audios, subs, delay_to_put, duration_best_video, tmp_prefix):
    """Run one ffmpeg job per kept track on the shared pool and return the per-track files; raise TrackJobError on the first failure."""
    have_pgs = any(sub["Format"].lower() in tools.sub_type_not_encodable for sub in subs)
    out_paths = track_file_paths(audios, subs, tmp_prefix)
    jobs = []
    for audio, out_path in zip(sorted(audios, key=lambda x: int(x["StreamOrder"])), out_paths):
        output_args = ["-map", f"0:{int(audio['StreamOrder'])}", "-map_metadata", "0", "-copy_unknown",
                       "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"]
        reencoded = audio_reencode.generate_audio_ffmpeg_conversion_cmd(output_args, audio, delay_to_put, stream=":a:0",
                                                                        path=video_obj.filePath)
        output_args.extend(vmsam_tmp_metadata(audio, "a:0"))
        cmd = ffmpeg_job(not have_pgs, delay_input_args(delay_to_put, video_obj.filePath), output_args, have_pgs,
                         duration_best_video, out_path)
        jobs.append((f"audio track {audio['StreamOrder']} of {video_obj.filePath}", [cmd], out_path,
                     audio if reencoded else None))
    if len(subs):
        subs = sorted(subs, key=lambda x: int(x["StreamOrder"]))
        convert_path = out_paths[-1] + ".convert"
        output_args = []
        for sub in subs:
            output_args.extend(["-map", f"0:{int(sub['StreamOrder'])}"])
        output_args.extend(["-map_metadata", "0", "-copy_unknown", "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"])
        for sub_relative_index, sub in enumerate(subs):
            codec = sub["Format"].lower()
            if codec in tools.sub_type_near_srt:
                output_args.extend([f"-c:s:{sub_relative_index}", "srt"])
            elif codec not in tools.sub_type_not_encodable:
                output_args.extend([f"-c:s:{sub_relative_index}", "ass"])
            output_args.extend(vmsam_tmp_metadata(sub, f"s:{sub_relative_index}"))
        convert = ffmpeg_job(not have_pgs, ["-i", video_obj.filePath], output_args, have_pgs,
                             Decimal(duration_best_video)-(Decimal(delay_to_put)/Decimal(1000)), convert_path)
        # Measured on ffmpeg 9.0.2: an output -ss under -copyts moves a stream copy but not a re-encoded
        # text subtitle, so the conversion and the delay need two commands.
        shift = ffmpeg_job(False, delay_input_args(delay_to_put, convert_path),
                           ["-map", "0", "-map_metadata", "0", "-copy_unknown", "-movflags", "use_metadata_tags",
                            "-c", "copy", "-copyts"], False, duration_best_video, out_paths[-1])
        jobs.append((f"subtitle tracks {[int(sub['StreamOrder']) for sub in subs]} of {video_obj.filePath}",
                     [convert, shift], out_paths[-1], None))
    try:
        submitted = [(out_path, audio, video.ffmpeg_pool_audio_convert.apply_async(run_track_job, (label, cmds)))
                     for label, cmds, out_path, audio in jobs]
        for out_path, audio, result in submitted:
            result.get()
            if audio is not None:
                try:
                    audio_integrity.check_reencoded_track(video_obj.filePath, int(audio["StreamOrder"]), out_path, delay_to_put)
                except audio_integrity.AudioReencodeIntegrityError as e:
                    raise TrackJobError(str(e)) from e
    except TrackJobError:
        remove_files(out_paths)
        raise
    finally:
        remove_files([out_path + ".convert" for out_path in out_paths])
    return out_paths


def wait_track_jobs(convert_processes):
    """Wait for every candidate's ResultThread; log a TrackJobError in tools.logs, in this thread, then raise it."""
    for convert_process in convert_processes:
        try:
            convert_process.get()
        except TrackJobError as e:
            tools.logs.append(f"[FFmpeg ERROR] {e}\n")
            raise
