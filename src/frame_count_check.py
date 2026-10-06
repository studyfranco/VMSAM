# -*- coding: utf-8 -*-
"""Whole-file frame-count check of a repair plan, and its picture-only fallback.

A plan reads candidate zones at an offset and fills the rest from the master. Counted in
master frames over the whole file it must give back the master:

    candidate frames - removed frames + filled frames == master frames

- candidate frames: the candidate's video length (stream frame count over its exact rate,
  else its declared duration) at the plan's time scale, converted to master frames;
- removed frames: what the plan leaves out of the candidate -- before the first zone, between
  two zones, after the last zone. Each boundary is read with the offset MEASURED at that edge
  of the zone (the audio walk's first / last windows inside it), not the zone's single
  applied offset, so a cut the plan missed (an offset change inside one zone) or a cut sized
  wrong leaves the sum off by its size;
- filled frames: every master fill of the plan.

A plan reading before the candidate's first frame or past its last one is short by that many
frames (removed frames are never negative at the two ends).

When the audio plan fails the count, `visual_fallback` rebuilds the plan from the pictures
alone (`video_offset_plan.measure_video_offset`, its zones and the divergent-span carve that
`merge_video_visual_fallback` uses), the master forced as the reference, counts it the same
way, and applies it through `repair_orchestrator.apply_plan` so the repair returns its usual
chimeric file or its usual decline. Nothing here raises on a runtime path.
"""
from decimal import Decimal
from fractions import Fraction
from os import path
import statistics

import tools

FRAME_COUNT_MISMATCH = "frame_count_mismatch"

# The plan's timeline is the master's declared video duration while the expected count is its
# stream frame count; the two may differ by the last partial frame.
TOLERANCE_FRAMES = Fraction(1)

# A zone edge's offset is the median of this many `ok` walk windows at that edge: one window
# straddling a cut placed a frame or two off its audio edit cannot move it.
EDGE_WINDOWS = 5
EDGE_MIN_WINDOWS = 3


def _fraction(value):
    """Exact Fraction of a Decimal/Fraction/number (None passes through)."""
    if value is None:
        return None
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _video_start_ms(video_obj):
    """The video stream's container start time in ms (0 when unknown)."""
    try:
        start = ((getattr(video_obj, "video", None) or {}).get("ffprobe") or {}).get("start_time")
        return Fraction(str(start)) * 1000 if start not in (None, "") else Fraction(0)
    except (TypeError, ValueError, ZeroDivisionError):
        return Fraction(0)


def video_length(video_obj, rate):
    """Return (frames, ms, source) of the file's video stream; (None, None, reason) if unread.

    Uses the stream's frame count over the exact rate, else the declared duration times the
    rate; nothing is decoded.
    """
    video = getattr(video_obj, "video", None) or {}
    try:
        count = video.get("FrameCount")
        if count not in (None, "") and rate:
            frames = int(str(count))
            if frames > 0:
                return Fraction(frames), Fraction(frames) * 1000 / rate, "frame_count"
    except (TypeError, ValueError):
        pass
    try:
        duration_ms = Fraction(str(video["Duration"])) * 1000
        return duration_ms * rate / 1000, duration_ms, "duration_x_rate"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None, None, "video_length_unread"


def zone_edge_offsets(zones, walk):
    """Measure each zone's offset at its start and at its end from the walk's windows.

    Returns one dict per zone: `start_ms`, `end_ms`, `n_ok` and `source` ("walk" or
    "zone_offset" when fewer than EDGE_MIN_WINDOWS windows lie wholly inside the zone).
    """
    import audio_walk
    rows = [row for row in (walk or {}).get("rows") or [] if row.get("status") == "ok"]
    edges = []
    for zone in zones:
        low = float(zone["master_start_ms"]) / 1000.0
        high = float(zone["master_end_ms"]) / 1000.0
        inside = [row["off"] for row in rows
                  if row["t"] >= low and row["t"] + audio_walk.WALK_WINDOW_S <= high]
        if len(inside) < EDGE_MIN_WINDOWS:
            offset = _fraction(zone["offset_ms"])
            edges.append({"start_ms": offset, "end_ms": offset, "n_ok": len(inside),
                          "source": "zone_offset"})
            continue
        edges.append({"start_ms": _fraction(round(statistics.median(inside[:EDGE_WINDOWS]), 3)),
                      "end_ms": _fraction(round(statistics.median(inside[-EDGE_WINDOWS:]), 3)),
                      "n_ok": len(inside), "source": "walk"})
    return edges


def count_plan(zones, fills, edges, master_rate, master_frames, candidate_start_ms,
               candidate_ms):
    """Count a plan in master frames.

    Args:
        zones, fills: the plan on the master timeline (ms), zones sorted.
        edges: `zone_edge_offsets` rows, one per zone.
        master_rate: exact master frame rate.
        master_frames: the expected count.
        candidate_start_ms, candidate_ms: the candidate's video start and length on the plan's
            clock (already at the plan's time scale).

    Returns:
        dict with `candidate`, `removed`, `filled`, `obtained`, `expected`, `delta` (Fractions
        of master frames), `ok`, and the per-term detail.
    """
    def frames(ms):
        return Fraction(ms) * master_rate / 1000

    candidate_end_ms = candidate_start_ms + candidate_ms
    head = _fraction(zones[0]["master_start_ms"]) + edges[0]["start_ms"] - candidate_start_ms
    tail = candidate_end_ms - (_fraction(zones[-1]["master_end_ms"]) + edges[-1]["end_ms"])
    interior = [(_fraction(zones[k + 1]["master_start_ms"]) + edges[k + 1]["start_ms"])
                - (_fraction(zones[k]["master_end_ms"]) + edges[k]["end_ms"])
                for k in range(len(zones) - 1)]
    drifts = [edge["end_ms"] - edge["start_ms"] for edge in edges]
    filled_ms = sum((_fraction(fill["master_end_ms"]) - _fraction(fill["master_start_ms"])
                     for fill in fills), Fraction(0))
    removed_ms = max(head, Fraction(0)) + sum(interior, Fraction(0)) + max(tail, Fraction(0))
    candidate = frames(candidate_ms)
    obtained = candidate - frames(removed_ms) + frames(filled_ms)
    delta = obtained - master_frames
    return {
        "candidate": candidate, "removed": frames(removed_ms), "filled": frames(filled_ms),
        "obtained": obtained, "expected": Fraction(master_frames), "delta": delta,
        "ok": abs(delta) <= TOLERANCE_FRAMES,
        "head_removed": frames(head), "tail_removed": frames(tail),
        "interior_removed": [frames(value) for value in interior],
        "zone_drift": [frames(value) for value in drifts],
    }


def _num(value):
    return None if value is None else round(float(value), 3)


def log_count(candidate_path, route, count, extra=""):
    """Log the check's numbers on one line (pass or fail)."""
    if count.get("unmeasured"):
        tools.log_always(f"repair: frame_count route={route} checked=no "
                         f"reason={count['unmeasured']} {extra}for {candidate_path}\n")
        return
    worst = max(range(len(count["zone_drift"])), key=lambda k: abs(count["zone_drift"][k]))
    tools.log_always(
        f"repair: frame_count route={route} ok={'yes' if count['ok'] else 'no'} "
        f"expected={_num(count['expected'])} obtained={_num(count['obtained'])} "
        f"delta={_num(count['delta'])} tolerance={_num(TOLERANCE_FRAMES)} "
        f"candidate={_num(count['candidate'])} removed={_num(count['removed'])} "
        f"filled={_num(count['filled'])} head_removed={_num(count['head_removed'])} "
        f"tail_removed={_num(count['tail_removed'])} "
        f"interior_removed={[_num(v) for v in count['interior_removed']]} "
        f"zone_drift={[_num(v) for v in count['zone_drift']]} worst_zone={worst} "
        f"master_source={count.get('master_source')} "
        f"candidate_source={count.get('candidate_source')} {extra}for {candidate_path}\n")


def check(zones, fills, edges, domain, scale, master_obj, candidate_obj):
    """Count a plan against the master; returns the `count_plan` dict or {"unmeasured": reason}.

    `scale` is the plan's time scale (candidate ms x scale = plan ms), 1 without a rate.
    """
    if not zones:
        return {"unmeasured": "no_zone"}
    master_rate, candidate_rate = domain["master_rate"], domain["candidate_rate"]
    master_frames, _, master_source = video_length(master_obj, master_rate)
    _, candidate_ms, candidate_source = video_length(candidate_obj, candidate_rate)
    if master_frames is None or candidate_ms is None:
        return {"unmeasured": f"master:{master_source},candidate:{candidate_source}"}
    scale = Fraction(scale) if scale not in (None, 1) else Fraction(1)
    count = count_plan(zones, fills, edges, master_rate, master_frames,
                       _video_start_ms(candidate_obj) * scale, candidate_ms * scale)
    count["master_source"], count["candidate_source"] = master_source, candidate_source
    return count


def check_audio_plan(zones, fills, walk, domain, factor, master_obj, candidate_obj,
                     candidate_path):
    """Count the audio-led plan and log it; returns the count dict (`ok` False on a mismatch).

    Never raises: an error is logged and reported as unmeasured.
    """
    try:
        count = check(zones, fills, zone_edge_offsets(zones, walk), domain, factor,
                      master_obj, candidate_obj)
    except Exception as error:                                           # noqa: BLE001
        count = {"unmeasured": f"raised:{type(error).__name__}:{error}"}
    log_count(candidate_path, "audio", count)
    return count


def check_constant_plan(master_obj, candidate_obj, offset_ms, candidate_path,
                        route="video_anchored"):
    """Count the one-zone plan of the picture-anchored route (no fill) and log it.

    Returns the count dict, or {"unmeasured": reason}; never raises.
    """
    import merge_video_chimeric
    import repair_orchestrator as orch
    try:
        domain, reason = orch.frame_domain(master_obj, candidate_obj, None)
        if domain is None:
            count = {"unmeasured": reason}
        else:
            offset = _fraction(offset_ms)
            zones = [{"master_start_ms": Decimal(0), "offset_ms": offset_ms, "zone": 0,
                      "master_end_ms": merge_video_chimeric.get_master_timeline_length_ms(
                          master_obj)}]
            count = check(zones, [], [{"start_ms": offset, "end_ms": offset, "n_ok": 0,
                                       "source": "picture"}], domain, 1, master_obj,
                          candidate_obj)
    except Exception as error:                                           # noqa: BLE001
        count = {"unmeasured": f"raised:{type(error).__name__}:{error}"}
    log_count(candidate_path, route, count)
    return count


def _refine_with_audio(zones, walk, candidate_path):
    """Replace each picture zone's offset by the comparison track's own offset near it.

    The audio is searched only around the picture's offset (`audio_walk.zone_offset`), so the
    picture keeps the zone; a zone the audio cannot measure keeps the picture offset.
    """
    import audio_walk
    if not walk or walk.get("master") is None or walk.get("candidate") is None:
        return zones
    refined = []
    for zone in zones:
        low = float(zone["master_start_ms"]) / 1000.0
        high = float(zone["master_end_ms"]) / 1000.0
        picture = zone["offset_ms"]
        measured = None
        if high - low >= audio_walk.WALK_WINDOW_S:
            measured = audio_walk.zone_offset(walk["master"], walk["candidate"], low, high,
                                              float(picture))["offset_ms"]
        tools.log_line(f"repair: frame_count_visual_zone zone={zone['zone']} "
                       f"master_ms=[{_num(zone['master_start_ms'])},"
                       f"{_num(zone['master_end_ms'])}] picture_ms={_num(picture)} "
                       f"audio_ms={_num(measured)} for {candidate_path}\n")
        offset = picture if measured is None else Decimal(str(round(float(measured), 3)))
        refined.append(dict(zone, offset_ms=offset, picture_offset_ms=picture,
                            n_windows=zone.get("n_windows", 0)))
    return refined


def visual_plan(master_obj, candidate_obj, walk, work_dir, repair_deadline, candidate_path):
    """Build the plan from the pictures alone, the master as the reference.

    Returns:
        (zones, fills, None) or (None, None, cause).
    """
    import merge_video_chimeric
    import merge_video_visual_fallback as visual
    import video_offset_plan
    work_root = path.join(work_dir, "frame_count_visual")
    tools.make_dirs(path.join(work_root, "measure"))
    result = video_offset_plan.measure_video_offset(
        master_obj.filePath, candidate_obj.filePath, path.join(work_root, "measure"),
        deadline=repair_deadline)
    timeline_ms = merge_video_chimeric.get_master_timeline_length_ms(master_obj)
    if result.status == video_offset_plan.STATUS_OK:
        zones = [{"master_start_ms": Decimal(0), "master_end_ms": timeline_ms,
                  "offset_ms": Decimal(str(round(float(-result.candidate_track_delay_ms), 3))),
                  "offset_frames": result.offset_frames, "n_windows": 0, "zone": 0}]
        fills = []
    elif result.status in (video_offset_plan.STATUS_NOT_CONSTANT,
                           video_offset_plan.STATUS_COVERAGE):
        groups = video_offset_plan.group_video_zones(result.pairs or [], float(result.frame_ms))
        cause = video_offset_plan.check_zone_compatibility(groups, result.master_frames)
        if cause is not None:
            return None, None, f"{result.status}:{cause}"
        zones, fills = video_offset_plan.video_zone_plan(
            groups, result.frame_ms, timeline_ms,
            (result.candidate_start_s - result.master_start_s) * 1000)
    else:
        return None, None, f"{result.status}:{result.reason}"
    zones, fills = visual._carve_divergence(master_obj, candidate_obj, result.fps, zones, fills,
                                            work_root, repair_deadline)
    for index, zone in enumerate(zones):
        zone["zone"] = index
    return _refine_with_audio(zones, walk, candidate_path), fills, None


def visual_fallback(audio_count, master_obj, candidate_obj, walk, domain, factor, context,
                    candidate_path):
    """Replace a plan that failed the frame count by the picture-only plan, when it counts.

    Args:
        audio_count: the failed count of the audio-led plan.
        context: the `apply_plan` context of the audio-led plan (language, streams, domain...).

    Returns:
        (ok, cause, reason, detail) as `repair_orchestrator.chimeric` returns it; every failure
        is `frame_count_mismatch` naming the route.
    """
    import repair_orchestrator as orch

    def decline(route, why, count=None):
        numbers = (f"audio plan expected {_num(audio_count['expected'])} frames, obtained "
                   f"{_num(audio_count['obtained'])}")
        if count is not None and not count.get("unmeasured"):
            numbers += (f"; picture plan expected {_num(count['expected'])}, obtained "
                        f"{_num(count['obtained'])}")
        tools.log_always(f"repair: frame_count_fallback route={route} outcome=declined "
                         f"why={why} for {candidate_path}\n")
        return False, FRAME_COUNT_MISMATCH, (
            f"the plan does not give back the master's frame count ({numbers}): a cut missed "
            f"or sized wrong; the picture-only route ({route}) {why}"), None

    tools.log_always(f"repair: frame_count_fallback route=visual outcome=started for "
                     f"{candidate_path}\n")
    if factor not in (None, 1):
        return decline("visual", "does_not_measure_a_rate_pair")
    try:
        zones, fills, cause = visual_plan(master_obj, candidate_obj, walk, context["work_dir"],
                                          domain.get("repair_deadline"), candidate_path)
        if zones is None:
            return decline("visual", f"failed:{cause}")
        edges = [{"start_ms": _fraction(zone["offset_ms"]), "end_ms": _fraction(zone["offset_ms"]),
                  "n_ok": 0, "source": "picture"} for zone in zones]
        count = check(zones, fills, edges, domain, 1, master_obj, candidate_obj)
        log_count(candidate_path, "visual", count)
        if count.get("unmeasured") or not count["ok"]:
            return decline("visual", "fails_the_same_count", count)
        head_s, tail_s = orch.written_edge_seconds(fills, walk["master_audio_end_s"])
        tagged, tag_reason = orch.tag_decision(len(zones) - 1, head_s + tail_s)
        ok, cause, reason = orch.apply_plan(candidate_path, {
            "zones": zones, "fills": fills, "walk": walk, "head_written_s": head_s,
            "tail_written_s": tail_s, "zone_picture_offsets_ms": {
                zone["zone"]: {"picture_ms": _fraction(zone["picture_offset_ms"]),
                               "cuts": "whole_file_picture"} for zone in zones}},
            factor, master_obj, candidate_obj,
            dict(context, tagged=tagged, tag_reason=tag_reason))
    except Exception as error:                                           # noqa: BLE001
        if getattr(error, "cause", None) == "repair_budget_exceeded":
            return False, "repair_budget_exceeded", (
                f"the repair's budget ran out during the picture-only route ({error})"), None
        return decline("visual", f"raised:{type(error).__name__}:{str(error)[:200]}")
    if not ok:
        return decline("visual", f"build_declined:{cause}")
    tools.log_always(f"repair: frame_count_fallback route=visual outcome=repaired "
                     f"n_zones={len(zones)} n_fills={len(fills)} for {candidate_path}\n")
    return True, None, reason, None
