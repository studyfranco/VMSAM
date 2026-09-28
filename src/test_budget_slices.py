'''Tests for the per-slice budgets of ADDENDUM 32.10 (a, f) and the stage summary lines of 32.10 g
-- run: python3 src/test_budget_slices.py (or pytest). No media: synthetic master objects carry
only the video duration the budgets read.'''

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402
import repair_orchestrator as ro  # noqa: E402


class _Master:
    def __init__(self, seconds):
        self.filePath = "/synthetic/master.mkv"
        self.video = {} if seconds is None else {"Duration": str(seconds)}


def test_repair_budget_is_40_min_per_started_slice_capped_at_4_h():
    cases = [(None, 2400.0), (60.0, 2400.0), (1800.0, 2400.0), (1800.001, 4800.0),
             (3600.0, 4800.0), (3747.0, 7200.0), (7200.0, 9600.0), (10800.0, 14400.0),
             (10800.5, 14400.0), (30000.0, 14400.0)]
    for seconds, expected in cases:
        budget, video_s = ro.repair_budget_seconds(_Master(seconds))
        assert budget == expected, (seconds, budget)
        assert video_s == (None if seconds is None else seconds)
    assert ro.REPAIR_BUDGET_CAP_S == 4 * 3600.0


def test_hole_budget_is_45_per_started_slice():
    cases = [(None, 45), (1420.0, 45), (1800.0, 45), (1801.0, 90), (3747.0, 135),
             (7200.0, 180), (7300.0, 225)]
    for seconds, expected in cases:
        assert ro.max_holes_per_couple(_Master(seconds)) == expected, seconds


def test_stage_summaries_are_unconditional_and_detail_is_dev_only():
    lines = []
    saved = (tools.dev, tools.log_always, tools.logs)
    tools.dev, tools.logs = False, []
    tools.log_always = lambda message: lines.append(message)
    try:
        transitions = [{"change_point": 0, "decision": "video_cut", "at_s": 12.5, "fill_s": 0.25,
                        "a_ms": 100.0, "b_ms": -150.0},
                       {"change_point": 1, "decision": "slip_applied", "at_s": 99.0,
                        "fill_s": 0.0, "a_ms": -150.0, "b_ms": -90.0}]
        ro.log_transitions_summary(transitions, "/c.mkv")
        report = {"agree": True, "n_couples": 2, "n_events": 2, "disagreements": [],
                  "clusters": [{"cluster_index": 0, "verdict": "agree",
                                "master_position_seconds": 12.4, "spread_ms": 3.0,
                                "tolerance_ms": 186.0, "could_not_see": [],
                                "below_floor_excluded": [], "members": []}]}
        ro.log_cross_verification("/c.mkv", report)
    finally:
        tools.dev, tools.log_always, tools.logs = saved[0], saved[1], saved[2]
    assert lines[0] == ("repair: audio_transitions n=2 transitions=0:video_cut:12.5:250.0:-250.0,"
                        "1:slip_applied:99.0:0.0:60.0 for /c.mkv\n"), lines[0]
    assert lines[1].startswith("repair: cross_verify_summary agree=True n_couples=2 n_events=2 "
                               "n_clusters=1 n_disagreements=0 clusters=0:agree:12.4:3.0 "), lines[1]
    assert len(lines) == 2, lines


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
