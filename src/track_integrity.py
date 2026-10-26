"""Integrity of the audio and subtitle tracks of a produced file, against the tracks they come from.

A produced track must end where its original, shifted by the delay put on it, ends (within one
codec frame, never past the reference video), hold no silence its original plays through, and
decode without error. Any finding raises track_integrity_error.
"""

import concurrent.futures
import json
import os
import subprocess

import file_conformity as fc
import integrity

# Without the config.json table, every codec gets the longest frame of the table.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")) as _config:
    SAMPLES_PER_FRAME = json.load(_config).get("integrity_samples_per_frame", {"default": 4608})


class track_integrity_error(Exception):
    """A produced audio or subtitle track is truncated, too long, broken or gained a silence."""


def expected_end_s(original_end_s, delay_ms, duration_best_video_s):
    """Return where a track ending at original_end_s, delayed by delay_ms, ends in the produced file.

    The rebuild keeps each track's own start time, so its end moves by the delay; a negative
    delay drops the head; nothing runs past the video.
    """
    return min(max(original_end_s + float(delay_ms) / 1000.0, 0.0), float(duration_best_video_s))


def frame_s(stream, last_packet_s):
    """Return one frame in seconds: samples per frame / sample rate, or a subtitle's last cue.

    A cue that starts before the video ends is kept whole, so it may end past the video.
    """
    rate = float(stream.get("sample_rate") or 0)
    if not rate:
        return max(last_packet_s, 0.001)
    return SAMPLES_PER_FRAME.get(stream.get("codec_name"), SAMPLES_PER_FRAME["default"]) / rate


def decode_errors(path, index):
    """Return the error lines of a full decode of one stream (ffprobe decodes subtitles too)."""
    p = subprocess.run([fc._bin("ffprobe"), "-v", "error", "-err_detect", "crccheck+bitstream+buffer",
                        "-select_streams", str(index), "-show_entries", "frame=pts_time:subtitle=pts_time",
                        "-of", "csv=p=0", path], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return p.stderr.strip().splitlines() or ([f"ffprobe exit {p.returncode}"] if p.returncode else [])


def check_track(original_path, original_index, original_span, produced_path, stream, produced_span,
                delay_ms, duration_best_video_s):
    """Return the findings (strings) on one produced track; empty when it is sound."""
    index = stream["index"]
    label = (f"{stream['codec_type']} track {index} ({fc._tag(stream, 'LANGUAGE') or 'und'}) of {produced_path}, "
             f"from track {original_index} of {original_path} at {float(delay_ms):.0f} ms")
    findings = []
    expected = expected_end_s(original_span[1], delay_ms, duration_best_video_s)
    if produced_span is None:
        findings.append(f"{label}: no packet, expected an end at {expected:.3f} s")
    elif abs(produced_span[1] - expected) > frame_s(stream, produced_span[2]):
        findings.append(f"{label}: ends at {produced_span[1]:.3f} s, expected {expected:.3f} s "
                        f"(+/- {frame_s(stream, produced_span[2]):.3f} s)")
    errors = decode_errors(produced_path, index)
    if errors:
        findings.append(f"{label}: decode errors: {' | '.join(errors[:3])}")
    if stream["codec_type"] == "audio":
        for s in integrity.silence_report(original_path, original_index, produced_path, index, delay_ms)["b_only"]:
            findings.append(f"{label}: silence {s['start_s']}-{s['end_s']} s that the original plays")
    return findings


def check_tracks(produced_path, origins, duration_best_video_s):
    """Check produced tracks in parallel and raise track_integrity_error naming every finding.

    origins: {produced stream index: (original path, original stream index, delay_ms)}.
    """
    streams = {s["index"]: s for s in fc.probe(produced_path)["streams"]}
    paths = sorted({o[0] for o in origins.values()} | {produced_path})
    with concurrent.futures.ThreadPoolExecutor(max(len(paths), len(origins), 1)) as pool:
        measured = dict(zip(paths, pool.map(fc.packet_spans, paths)))
        jobs = [pool.submit(check_track, o_path, o_index, measured[o_path][o_index], produced_path,
                            streams[index], measured[produced_path].get(index), delay_ms, duration_best_video_s)
                for index, (o_path, o_index, delay_ms) in sorted(origins.items())]
        findings = [f for job in jobs for f in job.result()]
    if findings:
        raise track_integrity_error("\n".join(findings))


