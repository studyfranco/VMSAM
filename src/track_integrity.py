"""Integrity of the audio and subtitle tracks of a produced file, against the tracks they come from.

A produced track must end where its original, shifted by the delay put on it, ends (within one
codec frame, never past the reference video). Any finding raises TrackIntegrityError.
"""

import concurrent.futures
import json
import os
import subprocess

import file_conformity as fc

# Without the config.json table, every codec gets the longest frame of the table.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")) as _config:
    SAMPLES_PER_FRAME = json.load(_config).get("integrity_samples_per_frame", {"default": 4608})


class TrackIntegrityError(Exception):
    """A produced audio or subtitle track is truncated or too long."""


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
    return findings


def check_tracks(produced_path, origins, duration_best_video_s):
    """Check produced tracks in parallel and raise TrackIntegrityError naming every finding.

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
        raise TrackIntegrityError("\n".join(findings))


def _sound_streams(path):
    return [s for s in fc.probe(path)["streams"] if s.get("codec_type") in ("audio", "subtitle")]


def _original_index(stream):
    return json.loads(fc._tag(stream, "VMSAM_TMP"))["original_position"]


def check_rebuilt(original_path, produced_path, delay_ms, duration_best_video_s):
    """Check a file rebuilt from original_path with delay_ms; its tracks carry VMSAM_TMP tags."""
    check_tracks(produced_path, {s["index"]: (original_path, _original_index(s), delay_ms)
                                 for s in _sound_streams(produced_path)}, duration_best_video_s)


def check_merged(final_path, sources, track_order, duration_best_video_s):
    """Check the final file of a merge.

    sources: [(rebuilt file, original path, delay_ms)] in the order mkvmerge joined them; the
    joined file holds every audio track, then every subtitle track, each in that order.
    track_order: the joined file's track ids, in the order the final file carries its
    audio and subtitle tracks. Each final track's VMSAM_TMP tag must agree with its source.
    """
    joined = sorted([(s["codec_type"], _original_index(s), original_path, delay_ms)
                     for rebuilt, original_path, delay_ms in sources for s in _sound_streams(rebuilt)],
                    key=lambda t: t[0] != "audio")
    final = _sound_streams(final_path)
    pinned = [(s, joined[int(t)]) for s, t in zip(final, track_order)]
    if len(final) != len(track_order) or any((s["codec_type"], _original_index(s)) != j[:2] for s, j in pinned):
        raise TrackIntegrityError(f"{final_path}: its audio and subtitle tracks do not match the merged "
                                  f"sources in order {track_order}")
    check_tracks(final_path, {s["index"]: (j[2], j[1], j[3]) for s, j in pinned}, duration_best_video_s)
