# -*- coding: utf-8 -*-
"""Picture lag of each zone, read over the scene changes of both whole files.

The anchors resolving a hole read 12 frames on each side of it. On a candidate whose picture
lag jitters by one frame from shot to shot (an IVTC-type cadence difference) or a shot drawn on
twos (each drawing held two frames, so lags L and L+1 tie), that one window can read the zone's
lag one frame off. The zone's own lag is the majority over its scene changes: both files are
decoded once (`video_offset_plan.decode_scenes_and_hashes`, the visual route's own decode and
disk cache), each master scene change is matched to the candidate (`video_offset_plan.
match_changes`, two-sided), and every zone counts the lags of the changes inside it.

The orchestrator runs it once the holes are resolved, so it never competes with their
time-bounded local decodes; `ZoneLagJob.pairs` waits for the decode and matches.
"""
from collections import Counter
from fractions import Fraction
import os
import threading

import tools

# Changes nearer than this to a zone bound are left out: they belong to the cut itself.
ZONE_BOUND_MARGIN_S = 2.0
# A zone majority needs at least this many scene changes at its lag.
ZONE_LAG_MIN_EVENTS = 3
# A second lag holding more than this share of a zone's changes is logged as jitter.
ZONE_LAG_JITTER_SHARE = 0.15


class ZoneLagJob:
    """Decode both files in a background thread; match their scene changes on demand."""

    def __init__(self, master_obj, candidate_obj, domain):
        self.master_path = master_obj.filePath
        self.candidate_path = candidate_obj.filePath
        self.domain = domain
        self.decoded = None
        self.reason = None
        self._thread = threading.Thread(target=self._decode, daemon=True)
        self._thread.start()

    def _decode(self):
        import video_offset_plan
        try:
            if self.domain.get("time_scale") is not None:
                self.reason = "time_scaled_pair"
                return
            m_info, why = video_offset_plan.probe_video(self.master_path)
            if m_info is None:
                self.reason = f"master_probe:{why}"
                return
            c_info, why = video_offset_plan.probe_video(self.candidate_path)
            if c_info is None:
                self.reason = f"candidate_probe:{why}"
                return
            fps, why = video_offset_plan.check_fps(m_info, c_info)
            if fps is None:
                self.reason = f"fps:{why}"
                return
            cache_root = os.path.join(tools.tmpFolder, "repair")
            tools.make_dirs(cache_root)
            crops = video_offset_plan.geometry_crop_filters(self.master_path, self.candidate_path)
            decoded = {}
            for side, path, info in (("master", self.master_path, m_info),
                                     ("candidate", self.candidate_path, c_info)):
                cuts, hashes, coloured, _cached = video_offset_plan.decode_scenes_and_hashes(
                    path, fps, info["duration_s"], cache_root, crops.get(path))
                decoded[side] = (cuts, hashes, coloured)
            self.decoded = decoded
        except Exception as error:                                       # noqa: BLE001
            self.reason = f"decode:{type(error).__name__}:{str(error)[:160]}"

    def pairs(self, hint_shifts):
        """Matched scene changes `[(master_frame, lag_frames), ...]`, or (None, reason).

        The lag search covers every hinted shift (anchor shifts, zone audio offsets in frames)
        widened by the visual route's own reach.
        """
        import video_offset_plan
        self._thread.join()
        if self.decoded is None:
            return None, self.reason or "not_decoded"
        hints = [int(h) for h in hint_shifts if h is not None]
        if not hints:
            return None, "no_hint"
        low = min(hints) - video_offset_plan.LAG_REACH
        high = max(hints) + video_offset_plan.LAG_REACH
        m_cuts, m_hashes, m_coloured = self.decoded["master"]
        c_cuts, c_hashes, c_coloured = self.decoded["candidate"]
        try:
            matched, _ambiguous, _total = video_offset_plan.match_changes(
                m_hashes, m_cuts, c_hashes, c_cuts, low, high,
                m_coloured=m_coloured, c_coloured=c_coloured)
        except Exception as error:                                       # noqa: BLE001
            return None, f"match:{type(error).__name__}:{str(error)[:160]}"
        return [(int(m), int(d)) for m, d, _dist in matched], None


def zone_majority(pairs, start_ms, end_ms, frame_ms):
    """(lag, counts) of the scene changes inside master [start_ms, end_ms).

    Changes within ZONE_BOUND_MARGIN_S of a bound are left out. `lag` is None when no lag
    reaches ZONE_LAG_MIN_EVENTS or two lags tie.
    """
    margin = Fraction(str(ZONE_BOUND_MARGIN_S * 1000.0)) / Fraction(frame_ms)
    first = Fraction(str(float(start_ms))) / Fraction(frame_ms) + margin
    last = Fraction(str(float(end_ms))) / Fraction(frame_ms) - margin
    counts = Counter(lag for frame, lag in pairs or [] if first <= frame < last)
    if not counts:
        return None, counts
    ranked = counts.most_common()
    best, count = ranked[0]
    if count < ZONE_LAG_MIN_EVENTS or (len(ranked) > 1 and ranked[1][1] == count):
        return None, counts
    return best, counts


def jitter(counts):
    """The lags each holding more than ZONE_LAG_JITTER_SHARE of the changes, when two or more."""
    total = sum(counts.values())
    shares = [lag for lag, count in counts.most_common()
              if total and count > ZONE_LAG_JITTER_SHARE * total]
    return shares if len(shares) > 1 else None


def video_start_delta_ms(master_obj, candidate_obj):
    """Candidate minus master video stream start time, in ms (exact Fraction)."""
    import frame_count_check
    return (frame_count_check._video_start_ms(candidate_obj)
            - frame_count_check._video_start_ms(master_obj))
