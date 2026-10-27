"""Check each retimed candidate subtitle track against the master's and the other candidate tracks' cues (decision: candidate-subtitle-sync)."""
import bisect
import difflib
import re
import statistics
from os import path

import progress_watchdog
import tools
import merge_video_chimeric

HISTOGRAM_BIN_MS = 50
MIN_PEAK_CUES = 20
MIN_THIRD_CUES = 8
PEAK_HALF_WIDTH_BINS = 3
RIVAL_DISTANCE_MS = 500
PAIR_WINDOW_MS = 2500
DUPLICATE_TEXT_SHARE = 0.95
# Measured on the candidate subtitles of 13 replayed files: an in-sync track authored apart from
# its reference sits up to 130 ms from the other candidate tracks and up to 260 ms from the
# master's same-language text, while the desynchronised tracks found sit 981 to 2019 ms off.
LOOSE_TOLERANCE_MS = 500
DRAWING = re.compile(r"\\p[1-9]")


def read_cues(subtitle_path):
    """Return the dialogue cues of a text subtitle file as sorted `(start_ms, end_ms, text)`."""
    import pysubs2
    cues = []
    for event in pysubs2.load(subtitle_path):
        text = merge_video_chimeric.normalized_cue_text(event)
        if text and not event.is_comment and not DRAWING.search(event.text):
            cues.append((event.start, event.end, text))
    return sorted(cues)


def read_text_tracks(file_path, subtitles, work_dir):
    """Read the text tracks of `subtitles` [(language, dict)] in one ffmpeg pass: {StreamOrder: (language, cues)}."""
    command = [tools.software["ffmpeg"], "-y", "-nostdin", "-analyzeduration", "1000M",
               "-probesize", "1000M", "-i", file_path]
    outputs = {}
    for language, subtitle in subtitles:
        target = merge_video_chimeric.classify_subtitle(
            (subtitle.get("ffprobe") or {}).get("codec_name", ""))
        order = int(subtitle["StreamOrder"])
        if target == "bitmap" or order in outputs:
            continue
        outputs[order] = (language, path.join(work_dir, f"sync_reference_{order}.{target}"))
        command.extend(["-map", f"0:{order}", "-c:s", target, outputs[order][1]])
    if not outputs:
        return {}
    try:
        progress_watchdog.run_checked(command, "subtitle_reference_extract")
    except Exception as error:
        tools.log_always(f"chimeric: subtitle_sync reference_unread file={file_path} "
                         f"error={type(error).__name__}\n")
        return {}
    tracks = {}
    for order, (language, out_path) in outputs.items():
        if path.exists(out_path) and path.getsize(out_path):
            try:
                tracks[order] = (language, read_cues(out_path))
            except Exception as error:
                tools.log_always(f"chimeric: subtitle_sync reference_unread file={file_path} "
                                 f"stream={order} error={type(error).__name__}: {error}\n")
    return tracks


def histogram_peak(pairs, min_cues):
    if len({start for start, difference in pairs}) < min_cues:
        return None
    counts = {}
    for start, difference in pairs:
        counts[difference // HISTOGRAM_BIN_MS] = counts.get(difference // HISTOGRAM_BIN_MS, 0) + 1
    windows = {bin_index: sum(counts.get(bin_index + k, 0)
                              for k in range(-PEAK_HALF_WIDTH_BINS, PEAK_HALF_WIDTH_BINS + 1))
               for bin_index in counts}
    peak_bin = max(windows, key=lambda bin_index: (windows[bin_index], -abs(bin_index)))
    rival = max((count for bin_index, count in windows.items()
                 if abs(bin_index - peak_bin) * HISTOGRAM_BIN_MS >= RIVAL_DISTANCE_MS), default=0)
    inside = [(start, difference) for start, difference in pairs
              if abs(difference // HISTOGRAM_BIN_MS - peak_bin) <= PEAK_HALF_WIDTH_BINS]
    if len({start for start, difference in inside}) < min_cues or windows[peak_bin] < 2 * rival:
        return None
    offset = statistics.median(difference for start, difference in inside)
    spread = 1.4826 * statistics.median(abs(difference - offset) for start, difference in inside)
    return offset, spread


def sync_decisions(candidate_tracks, master_tracks, zones, frame_ms):
    """Decide keep, shift, drop or duplicate for every candidate track {key: (language, cues)}."""
    frame_tolerance_ms = float(frame_ms) + 1
    decisions = {}
    for key, (language, cues) in candidate_tracks.items():
        decision = {"action": "keep", "reason": "in_sync", "zones": [], "shifts": []}
        decisions[key] = decision
        texts = [cue[2] for cue in cues]
        references = []
        for master_key, (master_language, master_cues) in master_tracks.items():
            if master_language != language or not master_cues:
                continue
            matcher = difflib.SequenceMatcher(None, texts, [cue[2] for cue in master_cues],
                                              autojunk=False)
            pairs = [(cues[a + k][0], cues[a + k][0] - master_cues[b + k][0])
                     for a, b, size in matcher.get_matching_blocks() for k in range(size)]
            if len(pairs) >= DUPLICATE_TEXT_SHARE * max(len(cues), len(master_cues)):
                decision.update(action="duplicate", reason=f"text_of_master_stream_{master_key}")
                break
            references.append((f"master_text_{master_key}", pairs, frame_tolerance_ms))
        if decision["action"] == "duplicate":
            continue
        consensus_pairs = []
        for other_key, (other_language, other_cues) in candidate_tracks.items():
            if other_key == key:
                continue
            starts = sorted({cue[0] for cue in other_cues})
            for start in sorted({cue[0] for cue in cues}):
                low = bisect.bisect_left(starts, start - PAIR_WINDOW_MS)
                high = bisect.bisect_right(starts, start + PAIR_WINDOW_MS)
                consensus_pairs.extend((start, start - other) for other in starts[low:high])
        references.append(("candidate_consensus", consensus_pairs, LOOSE_TOLERANCE_MS))
        widest_tolerance = frame_tolerance_ms
        for zone_start, zone_end in zones:
            measured = []
            for reference, pairs, tolerance in references:
                in_zone = [(start, difference) for start, difference in pairs
                           if zone_start <= start < zone_end]
                peak = histogram_peak(in_zone, MIN_PEAK_CUES)
                if peak is None:
                    continue
                offset, spread = peak
                if spread > tolerance / 2:
                    tolerance = max(LOOSE_TOLERANCE_MS, 2 * spread)
                third = (zone_end - zone_start) / 3
                constant = True
                for rank in range(3):
                    low = zone_start + rank * third
                    local = histogram_peak([(start, difference) for start, difference in in_zone
                                            if low <= start < low + third], MIN_THIRD_CUES)
                    if local is not None and abs(local[0] - offset) > tolerance:
                        constant = False
                measured.append((abs(offset) > tolerance, tolerance, abs(offset), reference, offset,
                                 len({start for start, difference in in_zone}), constant))
            if not measured:
                decision["zones"].append((zone_start, zone_end, "none", None, 0, None, None))
                decision["shifts"].append(None)
                continue
            is_off, tolerance, distance, reference, offset, cue_count, constant = min(measured)
            decision["zones"].append((zone_start, zone_end, reference, offset, cue_count,
                                      round(tolerance, 1), constant))
            widest_tolerance = max(widest_tolerance, tolerance)
            if not is_off:
                decision["shifts"].append(0)
            elif constant:
                decision["shifts"].append(-offset)
            else:
                decision.update(action="drop", reason="offset_not_constant_in_zone")
        if decision["action"] == "drop":
            continue
        measured_shifts = [shift for shift in decision["shifts"] if shift is not None]
        if not any(measured_shifts):
            decision["shifts"] = [0] * len(zones)
            if not measured_shifts:
                decision["reason"] = "no_evidence"
        elif max(measured_shifts) - min(measured_shifts) <= widest_tolerance:
            common = statistics.median(measured_shifts)
            decision["shifts"] = [common if shift is None else shift
                                  for shift in decision["shifts"]]
            decision.update(action="shift", reason="constant_offset")
        elif None in decision["shifts"]:
            decision.update(action="drop", reason="unmeasured_zone_between_differing_offsets")
        else:
            decision.update(action="shift", reason="offset_per_zone")
    return decisions


def shift_cues(subtitle_path, zones, shifts, timeline_end_ms):
    """Move every cue of `subtitle_path` by its zone's shift (zone found by cue start), in place."""
    import pysubs2
    subtitles = pysubs2.load(subtitle_path)
    kept = []
    for event in subtitles.events:
        shift = next((shift for (start, end), shift in zip(zones, shifts)
                      if start <= event.start < end), 0)
        event.start = int(round(max(0, min(event.start + shift, timeline_end_ms))))
        event.end = int(round(max(0, min(event.end + shift, timeline_end_ms))))
        if event.end > event.start:
            kept.append(event)
    subtitles.events = kept
    subtitles.save(subtitle_path)
    return len(kept)
