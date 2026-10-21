'''One ffmpeg job per kept track of a candidate, run on the shared ffmpeg pool (decision: audio-reencode).'''

import re
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


def remove_files(paths):
    for file_path in paths:
        if path.exists(file_path):
            tools.file_remove(file_path)


def audio_track_job(file_path, audio, delay_to_put, duration_best_video, have_pgs, is_master, out_path):
    label = f"audio track {audio['StreamOrder']} of {file_path}"
    output_args = ["-map", f"0:{int(audio['StreamOrder'])}", "-map_metadata", "0", "-copy_unknown",
                   "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"]
    already_passed = is_master and (audio.get("extra") or {}).get(tools.MASTER_PASS_TAG)
    reencoded = (not already_passed) and audio_reencode.generate_audio_ffmpeg_conversion_cmd(
        output_args, audio, delay_to_put, stream=":a:0", path=file_path)
    if reencoded and is_master:
        output_args.extend(["-metadata:s:a:0", f"{tools.MASTER_PASS_TAG}={duration_best_video}"])
    run_track_cmd(label, ffmpeg_job(not have_pgs, delay_input_args(delay_to_put, file_path), output_args, have_pgs,
                                    duration_best_video, out_path))
    if reencoded:
        try:
            audio_integrity.check_reencoded_track(file_path, int(audio["StreamOrder"]), out_path, delay_to_put)
        except audio_integrity.AudioReencodeIntegrityError as e:
            raise TrackJobError(str(e)) from e


def subtitle_track_job(file_path, sub, delay_to_put, duration_best_video, have_pgs, out_path):
    label = f"subtitle track {sub['StreamOrder']} of {file_path}"
    convert_path = out_path + ".convert"
    output_args = ["-map", f"0:{int(sub['StreamOrder'])}", "-map_metadata", "0", "-copy_unknown", "-copyts",
                   "-movflags", "use_metadata_tags", "-c", "copy"]
    codec = sub["Format"].lower()
    if codec in tools.sub_type_near_srt:
        output_args.extend(["-c:s:0", "srt"])
    elif codec not in tools.sub_type_not_encodable:
        output_args.extend(["-c:s:0", "ass"])
    # Measured on ffmpeg 9.0.2: an output -ss under -copyts moves a stream copy but not a re-encoded
    # text subtitle, so the conversion and the delay need two commands.
    try:
        run_track_cmd(label, ffmpeg_job(not have_pgs, ["-i", file_path], output_args, have_pgs,
                                        Decimal(duration_best_video)-(Decimal(delay_to_put)/Decimal(1000)), convert_path))
        run_track_cmd(label, ffmpeg_job(False, delay_input_args(delay_to_put, convert_path),
                                        ["-map", "0", "-map_metadata", "0", "-copy_unknown", "-movflags",
                                         "use_metadata_tags", "-c", "copy", "-copyts"],
                                        False, duration_best_video, out_path))
    finally:
        remove_files([convert_path])
    if codec not in tools.sub_type_not_encodable:
        source_md5 = video.subtitle_text_srt_md5(file_path, sub["StreamOrder"])[1]
        produced_md5 = video.subtitle_text_srt_md5(out_path, 0)[1]
        if source_md5 != produced_md5:
            raise TrackJobError(f"Track job failed for {label}: text md5 {source_md5} became {produced_md5}")


def generate_track_files(video_obj, audios, subs, delay_to_put, duration_best_video, tmp_file, is_master=False):
    """Run one pool job per kept track, then mkvmerge their files into tmp_file; raise TrackJobError on the first failure."""
    have_pgs = any(sub["Format"].lower() in tools.sub_type_not_encodable for sub in subs)
    jobs = []
    for audio in sorted(audios, key=lambda x: int(x["StreamOrder"])):
        jobs.append((audio_track_job, (video_obj.filePath, audio, delay_to_put, duration_best_video, have_pgs, is_master,
                                       f"{tmp_file}_track_{int(audio['StreamOrder'])}.mka")))
    for sub in sorted(subs, key=lambda x: int(x["StreamOrder"])):
        jobs.append((subtitle_track_job, (video_obj.filePath, sub, delay_to_put, duration_best_video, have_pgs,
                                          f"{tmp_file}_sub_{int(sub['StreamOrder'])}.mks")))
    if not jobs:
        return
    out_paths = [args[-1] for target, args in jobs]
    merge_cmd = [tools.software["mkvmerge"], "-o", tmp_file]
    for index, out_path in enumerate(out_paths):
        merge_cmd.extend(["--no-global-tags", "-M", "-B"] + (["--no-chapters"] if index else []) + [out_path])
    try:
        submitted = [video.ffmpeg_pool_audio_convert.apply_async(target, args) for target, args in jobs]
        for result in submitted:
            result.get()
        try:
            tools.launch_cmdExt_with_timeout_reload(merge_cmd, 2, 1200)
        except Exception as e:
            if not str(e).rstrip().endswith("Return code: 1"):
                raise TrackJobError(f"mkvmerge of the track files of {video_obj.filePath} failed: {e}") from e
            sys.stderr.write(f"[mkvmerge WARN] track files of {video_obj.filePath}: {e}\n")
    except TrackJobError:
        remove_files([tmp_file])
        raise
    finally:
        remove_files(out_paths)


def wait_track_jobs(convert_processes):
    """Wait for every candidate's ResultThread; log a TrackJobError in tools.logs, in this thread, then raise it."""
    for convert_process in convert_processes:
        try:
            convert_process.get()
        except TrackJobError as e:
            tools.log_always(f"[FFmpeg ERROR] {e}\n")
            raise
