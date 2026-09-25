'''Tests for the ADDENDUM 27.8 routing -- run: python3 src/test_video_anchored_route.py.

No media: the triggers (i master contradicts itself, ii candidate contradicts itself, iii the
pictures sit more than one frame from the audio) and the three declines of the video route
(`video_offset_not_constant`, `video_content_mismatch`, `video_fps_mismatch`) are driven with
stubbed measurements; the numbers are the lab's (Erai AAC 0 / E-AC-3 -84.4 ms, 23.976 fps).'''

import os
import sys
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
if not getattr(tools, "tmpFolder", None):
    tools.tmpFolder = "/tmp"

import frame_snap  # noqa: E402
import merge_video_repair  # noqa: E402
import repair_orchestrator as ro  # noqa: E402
import video_offset_plan as vop  # noqa: E402

FRAME_MS = 1001 / 24.0          # 41.708 ms


class Obj:
    def __init__(self, path, audios=None):
        self.filePath = path
        self.video = {"Duration": "1500.0", "FrameRate_Mode": "CFR", "FrameRate_Num": "24000",
                      "FrameRate_Den": "1001", "FrameRate": "23.976", "StreamOrder": "0"}
        self.audios = audios or {}
        self.commentary, self.audiodesc, self.subtitles = {}, {}, {}


def _row(m, c, delay, corr=0.98):
    return {"couple": f"{m}x{c}", "master_stream": m, "candidate_stream": c, "coarse_ms": delay,
            "delay_ms": delay, "correlation": corr, "at_s": 750.0, "reason": None}


class Patch:
    '''Set attributes for the duration of a block, restore them after.'''

    def __init__(self, *triples):
        self.triples, self.saved = triples, []

    def __enter__(self):
        for owner, name, value in self.triples:
            self.saved.append((owner, name, getattr(owner, name)))
            setattr(owner, name, value)
        return self

    def __exit__(self, *exc):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)


def _terminal_causes(since):
    return [line.split("cause=")[1].split()[0] for line in tools.logs[since:]
            if line.startswith("repair: orchestrator cause=")]


# -- the class map --------------------------------------------------------------------------

def test_master_intertrack_desync_is_a_route_not_a_decline():
    assert "master_intertrack_desync" not in ro.DECLINE_CAUSES
    assert ro.DECLINE_CAUSES["video_content_mismatch"] == ro.CLASS_CONCLUSIVE
    for cause in ("video_fps_mismatch", "video_offset_not_constant",
                  "video_offset_coverage_incomplete", "video_probe_failed",
                  "video_decode_failed", "decoder_timeout", "repair_budget_exceeded"):
        assert ro.DECLINE_CAUSES[cause] == ro.CLASS_COULD_NOT_RUN, cause


# -- trigger (ii) and (i) from the couples; agreement is no trigger --------------------------

def test_trigger_ii_candidate_tracks_disagree():
    rows = [_row(1, 1, 0.3), _row(1, 2, -84.4)]
    trigger, evidence = vop.contradiction_among_couples(rows, Fraction(1001, 24))
    assert trigger == vop.TRIGGER_CANDIDATE_DESYNC, trigger
    assert evidence["spread_ms"] == 84.7 and evidence["delays_ms"] == {"1x1": 0.3, "1x2": -84.4}


def test_trigger_i_by_couples_master_tracks_disagree():
    rows = [_row(1, 1, 0.0), _row(2, 1, 131.7)]
    trigger, _ = vop.contradiction_among_couples(rows, Fraction(1001, 24))
    assert trigger == vop.TRIGGER_MASTER_DESYNC, trigger


def test_couples_within_one_frame_are_one_story():
    rows = [_row(1, 1, -84.4), _row(1, 2, -80.6), _row(2, 1, -84.0, corr=0.5)]
    assert vop.contradiction_among_couples(rows, Fraction(1001, 24)) == (None, None)
    unmeasured = [_row(1, 1, 0.0), dict(_row(1, 2, -84.4), delay_ms=None)]
    assert vop.contradiction_among_couples(unmeasured, Fraction(1001, 24)) == (None, None)
    assert vop.contradiction_among_couples([_row(1, 1, 0), _row(1, 2, -84)], None) == (None, None)


def test_trigger_iii_pictures_two_frames_from_the_audio():
    master, candidate = Obj("/m.mkv"), Obj("/c.mkv")
    primed = {"couples": [(1, 1)], "alignments": {}}
    signal = {"signal": "audio_video_offset_disagree", "audio_frames": -2, "picture_frames": 0}
    with Patch((vop, "coarse_offsets_from_prime", lambda *a: {"1x1": -84.4}),
               (vop, "couple_fine_delays", lambda *a: [_row(1, 1, -84.4)]),
               (frame_snap, "disagreement", lambda a, b: None),
               (frame_snap, "probe_disagreement", lambda m, c, d: (signal, "audio_video_offset_disagree"))):
        trigger, evidence, rows = vop.detect_audio_contradiction(master, candidate, "ja", primed,
                                                                 "/c.mkv")
    assert trigger == vop.TRIGGER_PICTURE_DISAGREE and evidence["picture_frames"] == 0
    assert rows[0]["delay_ms"] == -84.4


def test_no_trigger_when_audio_and_picture_agree():
    master, candidate = Obj("/m.mkv"), Obj("/c.mkv")
    with Patch((vop, "coarse_offsets_from_prime", lambda *a: {"1x1": 0.0}),
               (vop, "couple_fine_delays", lambda *a: [_row(1, 1, 0.3)]),
               (frame_snap, "disagreement", lambda a, b: None),
               (frame_snap, "probe_disagreement", lambda m, c, d: (None, "frame_snap_chosen"))):
        trigger, evidence, _ = vop.detect_audio_contradiction(master, candidate, "ja",
                                                              {"couples": [(1, 1)]}, "/c.mkv")
    assert trigger is None and evidence == {"picture_probe": "frame_snap_chosen"}


# -- the video's three declines, and the fallback ---------------------------------------------

def _route_with(status, trigger, **fields):
    result = vop.VideoOffsetResult(status, fps=Fraction(24000, 1001), master_scenes=300,
                                   candidate_scenes=298, paired=fields.pop("paired", 0),
                                   total=280, **fields)
    with Patch((vop, "measure_video_offset", lambda *a, **k: result)):
        return vop.video_anchored_route(trigger, {}, Obj("/m.mkv"), Obj("/c.mkv"), "ja",
                                        [_row(1, 1, 0.0), _row(1, 2, -84.4)], None)


def test_decline_video_offset_not_constant_for_i_fallback_for_ii():
    status, cause, reason = _route_with(vop.STATUS_NOT_CONSTANT, vop.TRIGGER_MASTER_DESYNC,
                                        regimes=[(-24, 100, 900, 5)])
    assert (status, cause) == ("declined", "video_offset_not_constant"), (status, cause)
    assert "1x2:-84.4" in reason and "1x1:0.0" in reason       # every couple, never a mean
    status, cause, _ = _route_with(vop.STATUS_NOT_CONSTANT, vop.TRIGGER_CANDIDATE_DESYNC)
    assert (status, cause) == ("fallback", "video_offset_not_constant")


def test_decline_video_content_mismatch_with_numbers():
    status, cause, reason = _route_with(vop.STATUS_UNMATCHED, vop.TRIGGER_PICTURE_DISAGREE,
                                        paired=1)
    assert (status, cause) == ("declined", "video_content_mismatch")
    assert "paired=1/280" in reason and "master=300" in reason


def test_decline_video_fps_mismatch():
    status, cause, _ = _route_with(vop.STATUS_FPS_MISMATCH, vop.TRIGGER_MASTER_DESYNC)
    assert (status, cause) == ("declined", "video_fps_mismatch")


# -- the seam in repair(): (i) at STEP 1, (ii)/(iii) after the prime --------------------------

def test_repair_routes_master_intertrack_desync_to_the_video():
    calls = []

    def route(trigger, evidence, *a):
        calls.append((trigger, evidence))
        return "declined", "video_content_mismatch", "stub"

    verdict = {"verdict": "master_intertrack_desync", "reason": "stub", "worst": {"lag_ms": 131.7},
               "pairs": [{"stream_a": 1, "stream_b": 2, "lag_ms": 131.7, "correlation": 0.97}]}
    since = len(tools.logs)
    with Patch((merge_video_repair, "master_intertrack_verdict", lambda *a: verdict),
               (vop, "video_anchored_route", route)):
        ok = ro.repair(Obj("/m.mkv"), Obj("/c.mkv"), "ja", work_root="/tmp/t278")
    assert ok is False and calls[0][0] == "master_intertrack_desync"
    assert calls[0][1]["worst_lag_ms"] == 131.7
    assert _terminal_causes(since) == ["video_content_mismatch"], _terminal_causes(since)
    with Patch((merge_video_repair, "master_intertrack_verdict", lambda *a: verdict),
               (vop, "video_anchored_route", lambda *a: ("repaired", None, "stub"))):
        assert ro.repair(Obj("/m.mkv"), Obj("/c.mkv"), "ja", work_root="/tmp/t278") is True


class Continued(Exception):
    pass


def _after_prime(trigger, route_status):
    def prime(master, candidate, language, work_dir, primed, resample_routing=None):
        primed["couples"] = [("1", "1"), ("1", "2")]
        return True, None, None

    def gate(*a):
        raise Continued()

    master = Obj("/m.mkv", {"ja": [{"StreamOrder": "1"}]})
    master.video.pop("Duration")                 # no tail check: its duration is unread
    candidate = Obj("/c.mkv", {"ja": [{"StreamOrder": "1"}, {"StreamOrder": "2"}]})
    with Patch((merge_video_repair, "master_intertrack_verdict", lambda *a: {"verdict": None}),
               (ro, "prime_couples", prime), (ro, "ensemble_similarity_gate", gate),
               (vop, "detect_audio_contradiction",
                lambda *a: (trigger, {"stub": 1}, [_row(1, 1, 0.3), _row(1, 2, -84.4)])),
               (vop, "video_anchored_route", lambda *a: route_status)):
        return ro.repair(master, candidate, "ja", work_root="/tmp/t278")


def test_repair_routes_candidate_contradiction_and_picture_disagreement():
    assert _after_prime(vop.TRIGGER_CANDIDATE_DESYNC, ("repaired", None, "stub")) is True
    assert _after_prime(vop.TRIGGER_PICTURE_DISAGREE, ("repaired", None, "stub")) is True
    since = len(tools.logs)
    assert _after_prime(vop.TRIGGER_PICTURE_DISAGREE,
                        ("declined", "video_fps_mismatch", "stub")) is False
    assert _terminal_causes(since) == ["video_fps_mismatch"]


def test_fallback_and_no_trigger_continue_the_ordinary_path():
    for trigger, status in ((vop.TRIGGER_CANDIDATE_DESYNC,
                             ("fallback", "video_offset_not_constant", "stub")),
                            (None, ("repaired", None, "never called"))):
        try:
            _after_prime(trigger, status)
        except Continued:
            continue
        raise AssertionError(f"the ordinary path did not continue for {trigger}")


# -- the plan: one zone, head/tail additions only ---------------------------------------------

def test_pieces_move_the_whole_track_and_add_only_at_the_edges():
    timeline = Decimal("1500000")
    pieces, adjustments, _ = vop.video_anchored_pieces(Decimal("-83.417"), Decimal("1499000"),
                                                       timeline)
    kinds = [(p["source"], p["reason"]) for p in pieces]
    assert kinds == [("master", "head_gap"), ("candidate", "zone"), ("master", "tail_gap")], kinds
    assert pieces[1]["source_start_ms"] == Decimal(0)
    assert pieces[0]["master_end_ms"] == Decimal("83.417")
    assert pieces[2]["master_end_ms"] == timeline
    # the candidate reads master + offset everywhere: one relation, the track's own
    assert pieces[1]["source_start_ms"] - pieces[1]["master_start_ms"] == Decimal("-83.417")
    pieces, _, _ = vop.video_anchored_pieces(Decimal("0"), None, timeline)
    assert [(p["source"], p["master_start_ms"], p["master_end_ms"]) for p in pieces] == [
        ("candidate", Decimal(0), timeline)]


def test_route_success_moves_every_track_by_the_picture_and_marks_it():
    result = vop.VideoOffsetResult(vop.STATUS_OK, fps=Fraction(24000, 1001), offset_frames=0,
                                   master_frames=35964, candidate_frames=35964,
                                   master_scenes=300, candidate_scenes=300, paired=250,
                                   total=280, matched=260, mean_hamming=1.2,
                                   covered_frames=(100, 35800), thirds=[(0, 80, 80)] * 3,
                                   regimes=[], trend_frames=0.0)
    built = {}

    class Built:
        filePath = "/tmp/t278_repaired.mkv"

    def build(candidate_obj, master_obj, plan, work_root, job_start_utc):
        built["plan"] = plan
        return Built(), {"audios": [{"stream_order": 1, "marker": plan["marker"]},
                                    {"stream_order": 2, "marker": plan["marker"]}],
                         "subtitles": [], "marker": plan["marker"], "verification": []}

    import merge_video_chimeric as mvc
    candidate = Obj("/c.mkv", {"ja": [{"StreamOrder": "1"}, {"StreamOrder": "2"}]})
    master = Obj("/m.mkv", {"ja": [{"StreamOrder": "1"}]})
    with Patch((vop, "measure_video_offset", lambda *a, **k: result),
               (ro, "_track_timing", lambda *a: (Decimal(0), Decimal("1499000"), "stub")),
               (mvc, "build_delivered_chapters", lambda *a: (None, [])),
               (mvc, "probe_delivered_durations",
                lambda p: {"container_ms": 1, "streams": [], "max_cue_end_ms": None}),
               (merge_video_repair, "build_repaired_video_object", build),
               (os.path, "exists", lambda p: True)):
        status, cause, reason = vop.video_anchored_route(
            vop.TRIGGER_CANDIDATE_DESYNC, {}, master, candidate, "ja",
            [_row(1, 1, 0.3), _row(1, 2, -84.4)], None)
    assert (status, cause) == ("repaired", None), (status, cause, reason)
    plan = built["plan"]
    assert plan["kind"] == "orchestrator_video_anchored" and plan["marker"] == "video_anchored:+0"
    assert sorted(plan["track_plans"]) == [1, 2]
    offsets = {o: [p["source_start_ms"] - p["master_start_ms"] for p in tp["pieces"]
                   if p["source"] == "candidate"] for o, tp in plan["track_plans"].items()}
    assert offsets == {1: [Decimal(0)], 2: [Decimal(0)]}, offsets   # never "corrected" to -84.4
    assert plan["video_anchored"]["offset_frames"] == 0
    assert "1x2:-84.4" in reason


if __name__ == "__main__":
    tools.logs = getattr(tools, "logs", [])
    failures = 0
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
            print(f"ok   {name}")
        except Exception as error:                                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(names) - failures}/{len(names)} passed")
    sys.exit(1 if failures else 0)
