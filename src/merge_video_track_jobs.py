'''
One ffmpeg job per kept track of a candidate file, for generate_new_file.

Each kept audio track gets its own ffmpeg command, read straight from the
original file: the delay block (-itsoffset before -i for a positive delay,
-ss after -i for a negative one), the audio_reencode conversion, the
VMSAM_TMP tag and -t duration_best_video, written to its own .mka.

The kept subtitles get one job of their own, in two ffmpeg commands: the
text conversion (srt or ass) first, then the delay block on a stream copy.
They cannot share one command: measured on ffmpeg 9.0.2, a re-encoded text
subtitle is NOT moved by an output -ss under -copyts (a stream copy is), so
a single command left every cue of a negative-delay candidate at its
original time, 1.5 s late on the measured case.

PGS/VobSub among the kept subtitles drops -fflags +genpts+igndts and adds
-avoid_negative_ts disabled, on every job of the file, as the former
whole-file first pass did.

At most MAX_PARALLEL_TRACK_JOBS jobs run at once. The per-track files then
go, in track_file_paths order (audio tracks in StreamOrder, then the
subtitles: the layout of the former -map 0:a? -map 0:s? command), into the
one mkvmerge that generate_launch_merge_command already runs.

A job that fails raises TrackJobError naming the track and ffmpeg's error:
the fusion stops, a track is never dropped silently.
'''

import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from decimal import Decimal
from os import path

import tools
import audio_reencode

MAX_PARALLEL_TRACK_JOBS = 4
TRACK_JOB_TIMEOUT = 3600
WARNING_KEYWORDS = ["error", "invalid", "corrupt", "dts", "pts", "non monoton", "discarding", "out of order"]


class TrackJobError(Exception):
    '''A per-track ffmpeg job failed.'''


def delay_input_args(delay_to_put, file_path):
    '''Return the input part of a job: the delay done by ffmpeg.

    delay > 0: -itsoffset <s> -i file; delay < 0: -i file -ss <s>; 0: -i file.
    '''
    if delay_to_put > 0:
        return ["-itsoffset", f"{delay_to_put/Decimal('1000'):.6f}", "-i", file_path]
    elif delay_to_put < 0:
        return ["-i", file_path, "-ss", f"{delay_to_put/Decimal('1000')*Decimal(-1):.6f}"]
    return ["-i", file_path]


def vmsam_tmp_value(track):
    '''Return the VMSAM_TMP tag value: the track's original position and delay.'''
    return json.dumps({
        "original_position": int(track.get('StreamOrder', '0')),
        "original_delay": float(track.get('Delay', '0'))
    })


def have_image_subtitle(subs):
    '''Return True when a kept subtitle is PGS/VobSub (copied, never encoded).'''
    return any(sub["Format"].lower() in tools.sub_type_not_encodable for sub in subs)


def job_head(have_pgs):
    '''Return the start of an ffmpeg command, before its input.

    -fflags +genpts+igndts is left out when the file keeps a PGS/VobSub track.
    '''
    cmd = [tools.software["ffmpeg"], "-y", "-nostdin", "-err_detect", "crccheck+bitstream+buffer"]
    if not have_pgs:
        cmd.extend(["-fflags", "+genpts+igndts"])
    cmd.extend(["-analyzeduration", "1000M", "-probesize", "1000M", "-threads", "5", "-vn"])
    return cmd


def job_tail(duration, out_path, have_pgs):
    '''Return the end of a command: the PGS rule, the cut at `duration`, the output.'''
    tail = ["-avoid_negative_ts", "disabled"] if have_pgs else []
    return tail + ["-strict", "-2", "-t", str(duration), "-max_interleave_delta", "0",
                   "-max_muxing_queue_size", "16384", "-f", "matroska", out_path]


def build_audio_job(file_path, audio, delay_to_put, duration_best_video, out_path, have_pgs):
    '''Return the ffmpeg command for one audio track (StreamOrder of file_path).'''
    cmd = job_head(have_pgs) + delay_input_args(delay_to_put, file_path)
    cmd.extend(["-map", f"0:{int(audio['StreamOrder'])}", "-map_metadata", "0", "-copy_unknown",
                "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"])
    audio_reencode.generate_audio_ffmpeg_conversion_cmd(cmd, audio, delay_to_put, stream=":a:0", path=file_path)
    cmd.extend(["-metadata:s:a:0", f"VMSAM_TMP={vmsam_tmp_value(audio)}"])
    cmd.extend(job_tail(duration_best_video, out_path, have_pgs))
    return cmd


def build_subtitle_job(file_path, subs, delay_to_put, duration_best_video, out_path, have_pgs):
    '''Return the two ffmpeg commands of the subtitle job: the conversion of
    every kept subtitle of file_path into out_path's ".convert" file (text to
    srt or ass, PGS/VobSub copied, cut at duration - delay), then the delay
    block on a stream copy of it into out_path, cut at duration_best_video.'''
    subs = sorted(subs, key=lambda x: int(x["StreamOrder"]))
    convert_path = out_path + ".convert"
    convert = job_head(have_pgs) + ["-i", file_path]
    for sub in subs:
        convert.extend(["-map", f"0:{int(sub['StreamOrder'])}"])
    convert.extend(["-map_metadata", "0", "-copy_unknown", "-copyts", "-movflags", "use_metadata_tags", "-c", "copy"])
    for sub_relative_index, sub in enumerate(subs):
        codec = sub["Format"].lower()
        if codec in tools.sub_type_near_srt:
            convert.extend([f"-c:s:{sub_relative_index}", "srt"])
        elif codec not in tools.sub_type_not_encodable:
            convert.extend([f"-c:s:{sub_relative_index}", "ass"])
        convert.extend([f"-metadata:s:s:{sub_relative_index}", f"VMSAM_TMP={vmsam_tmp_value(sub)}"])
    convert.extend(job_tail(Decimal(duration_best_video)-(Decimal(delay_to_put)/Decimal(1000)), convert_path, have_pgs))

    shift = job_head(True) + delay_input_args(delay_to_put, convert_path)
    shift.extend(["-map", "0", "-map_metadata", "0", "-copy_unknown", "-movflags", "use_metadata_tags", "-c", "copy", "-copyts"])
    shift.extend(job_tail(duration_best_video, out_path, False))
    return [convert, shift]


def track_file_paths(audios, subs, tmp_prefix):
    '''Return the per-track output files, in the order mkvmerge takes them:
    audio tracks in StreamOrder, then the subtitle file when a subtitle is kept.'''
    paths = [f"{tmp_prefix}_track_{int(audio['StreamOrder'])}.mka"
             for audio in sorted(audios, key=lambda x: int(x["StreamOrder"]))]
    if len(subs):
        paths.append(f"{tmp_prefix}_subtitles.mks")
    return paths


def build_track_jobs(video_obj, audios, subs, delay_to_put, duration_best_video, tmp_prefix):
    '''Return the jobs of one candidate file: [(label, [cmd, ...], out_path)],
    in track_file_paths order.'''
    have_pgs = have_image_subtitle(subs)
    out_paths = track_file_paths(audios, subs, tmp_prefix)
    jobs = []
    for audio, out_path in zip(sorted(audios, key=lambda x: int(x["StreamOrder"])), out_paths):
        label = f"audio track {audio['StreamOrder']} of {video_obj.filePath}"
        jobs.append((label, [build_audio_job(video_obj.filePath, audio, delay_to_put, duration_best_video, out_path, have_pgs)], out_path))
    if len(subs):
        label = f"subtitle tracks {sorted(int(sub['StreamOrder']) for sub in subs)} of {video_obj.filePath}"
        jobs.append((label, build_subtitle_job(video_obj.filePath, subs, delay_to_put, duration_best_video, out_paths[-1], have_pgs), out_paths[-1]))
    return jobs


def warning_lines(stderr_text):
    '''Return the stderr lines the former passes reported as warnings.'''
    return [line for line in stderr_text.splitlines()
            if (not line.startswith("Input #")) and (not re.search(r"attachment:\s*none", line, re.IGNORECASE))
            and any(kw in line.lower() for kw in WARNING_KEYWORDS)]


def run_track_job(label, cmds):
    '''Run the commands of one job in order; raise TrackJobError with the
    track and ffmpeg's error when one fails. Warning lines of a successful
    run go to stderr, as the former passes did.'''
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
        lines = warning_lines(stderr_text)
        if lines:
            sys.stderr.write(f"[FFmpeg WARN] Track job for {label}\n{cmd}\n")
            for line in lines:
                sys.stderr.write(f"  {line}\n")


def run_track_jobs(jobs):
    '''Run the jobs, at most MAX_PARALLEL_TRACK_JOBS at once. On the first
    failure no pending job starts, the running ones finish, and the
    TrackJobError is raised. The stop flag is set by the failing job itself:
    a worker freed by that failure would otherwise pick the next pending job
    before the cancel below runs (measured by the unit test).'''
    stop = threading.Event()

    def guarded(label, cmds):
        if stop.is_set():
            return
        try:
            run_track_job(label, cmds)
        except TrackJobError:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TRACK_JOBS) as executor:
        futures = [executor.submit(guarded, label, cmds) for label, cmds, out_path in jobs]
        done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        failed = [future for future in futures if future.done() and not future.cancelled() and future.exception() is not None]
        if failed:
            for future in not_done:
                future.cancel()
            wait(futures)
            raise failed[0].exception()


def remove_files(paths):
    for file_path in paths:
        if path.exists(file_path):
            tools.file_remove(file_path)


def generate_track_files(video_obj, audios, subs, delay_to_put, duration_best_video, tmp_prefix):
    '''Build and run the jobs of one candidate file; return the per-track
    files (track_file_paths order). On a failure every file of the candidate
    is removed and the TrackJobError is raised.'''
    jobs = build_track_jobs(video_obj, audios, subs, delay_to_put, duration_best_video, tmp_prefix)
    try:
        run_track_jobs(jobs)
    except TrackJobError:
        remove_files([out_path for label, cmds, out_path in jobs])
        raise
    finally:
        remove_files([out_path + ".convert" for label, cmds, out_path in jobs])
    return [out_path for label, cmds, out_path in jobs]


def merge_cmd_for_track_files(out_paths):
    '''Return the mkvmerge arguments adding the per-track files of one
    candidate: no global tags, attachments or buttons (as the former single
    temporary file), chapters from the first file only.'''
    args = []
    for index, out_path in enumerate(out_paths):
        args.extend(["--no-global-tags", "-M", "-B"])
        if index:
            args.append("--no-chapters")
        args.append(out_path)
    return args


def wait_track_jobs(convert_processes):
    '''Wait for every candidate's jobs (the Pool results); a TrackJobError is
    logged in tools.logs, in the parent process, then raised again.'''
    for convert_process in convert_processes:
        try:
            convert_process.get()
        except TrackJobError as e:
            tools.logs.append(f"[FFmpeg ERROR] {e}\n")
            raise
