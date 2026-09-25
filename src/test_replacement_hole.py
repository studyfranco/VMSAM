'''Tests for ADDENDUM 25.9 -- run: python3 src/test_replacement_hole.py (or pytest).

1. the replacement hole (`repair_orchestrator.replacement_hole`), on id 111's measured numbers;
2. the edge probes' fallback to the level's LOCAL offset (`audio_walk.change_points`), on
   synthetic audio at WALK_RATE: a level whose first seconds sit 4 ms off its median.'''

import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_walk as aw  # noqa: E402
import repair_orchestrator as ro  # noqa: E402

R = aw.WALK_RATE
RATE = Fraction(24000, 1001)
DOMAIN = {"master_rate": RATE}
FRAME_S = float(1 / RATE)
SLACK_S = 0.1238095 + 2 * FRAME_S       # one quantum + two frames, as audio_transitions passes
# id 111 CP3 (CASE_audio_step_unlocalised_id111_20260925): audio edges and step at the local
# offsets, video cuts at master 674.841 / 748.957 s
EDGES = {"edge_A": 673.52, "edge_B": 748.865, "extra_s": 73.9905}


def _frame(seconds):
    return round(seconds * RATE)


def _outcome(start_s, end_s, status=ro.HOLE_RESOLVED):
    return {"status": status, "master_start_frame": _frame(start_s),
            "master_end_frame": _frame(end_s)}


def test_id_111_is_a_replacement_hole():
    hole = ro.replacement_hole(_outcome(674.841, 748.957), DOMAIN, EDGES, SLACK_S)
    assert hole is not None, hole
    assert abs(hole["fill_s"] - 74.116) < 2 * FRAME_S, hole
    assert 100.0 < hole["cut_ms"] < 170.0, hole          # 126 ms measured, to the frame


def test_no_replacement_without_both_video_edges_or_past_one_second():
    # the video pinned only one side / did not resolve
    assert ro.replacement_hole(_outcome(674.841, 748.957, ro.HOLE_DECLINED), DOMAIN, EDGES,
                               SLACK_S) is None
    # a video cut outside the audio edges by more than the slack
    assert ro.replacement_hole(_outcome(673.2, 748.957), DOMAIN, EDGES, SLACK_S) is None
    assert ro.replacement_hole(_outcome(674.841, 749.2), DOMAIN, EDGES, SLACK_S) is None
    # an excess of 1 s or more
    assert ro.replacement_hole(_outcome(673.6, 748.8), DOMAIN,
                               dict(EDGES, extra_s=73.0), SLACK_S) is None
    # a span narrower than the step is not a replacement either
    assert ro.replacement_hole(_outcome(680.0, 748.8), DOMAIN, EDGES, SLACK_S) is None


def _content(seconds, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * R))
    x = np.convolve(x, np.ones(4) / 4, "same")
    return (0.1 * x / np.sqrt(np.mean(x * x))).astype(np.float32)


def test_edge_probe_falls_back_to_the_local_offset():
    # master = A + X (1.2 s master-only) + near + far; the candidate lacks X and carries 4 ms of
    # its own material before `near`, dropped again before `far`: the after-level's median is
    # -1200 ms, its first seconds read -1196 ms -- 4 ms, inside the level tolerance
    a_part, x, near, far = (_content(s, 51 + i) for i, s in enumerate((20.0, 1.2, 6.0, 40.0)))
    master = np.concatenate([a_part, x, near, far])
    lag = np.zeros(int(0.004 * R), np.float32)
    cand = np.concatenate([a_part, lag, near, far[len(lag):]])
    rows = aw.walk(master, cand, [0.0, -1200.0])
    found, _outliers = aw.levels(rows)
    before, after = found[0], found[1]
    assert round(after["off_ms"]) == -1200 and round(after["off_first_ms"]) == -1196, found
    # at the medians the edge is unreadable (what the probes did before 25.9) ...
    assert aw.coarse_edges(master, cand, before["off_ms"], after["off_ms"],
                           before["t_last"] - 1.0, after["t_first"] + 3.0) is None
    # ... and the fallback reads it at the local offsets; step and fill stay the medians'
    edges = aw.change_points(master, cand, found)[0]["edges"]
    assert edges["status"] == "ok" and edges["feasible"], edges
    assert abs(edges["edge_A"] - 20.0) <= 0.015 and abs(edges["edge_B"] - 21.2) <= 0.015, edges
    assert edges["extra_s"] == 1.2 and edges["probe_ms"] == [0.0, after["off_first_ms"]], edges


if __name__ == "__main__":
    import tools
    tools.log_always = lambda message: None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
