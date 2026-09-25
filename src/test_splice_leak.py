'''Tests for the splice kept out of the candidate's own speech (`audio_walk.leak_bounds`) and for
the 10 ms crossfade at an audible candidate-to-candidate join (`merge_video_chimeric`) -- run:
python3 src/test_splice_leak.py (or pytest).

CASE_id695_splice_in_dialogue_20260925 (Bleach TYBW S17E41, change point +10005.456 ->
+22142.581 ms): the master is digital silence over the whole audio interval, the candidate carries
its own speech 0.3 s past edge_A under the before-offset, and the master's "quietest instant"
(dither) fell 380 ms into that speech. Built here on synthetic audio at WALK_RATE.'''

import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_walk as aw  # noqa: E402

R = aw.WALK_RATE


def _content(seconds, seed, level=0.1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(round(seconds * R)))
    x = np.convolve(x, np.ones(4) / 4, "same")
    return (level * x / np.sqrt(np.mean(x * x))).astype(np.float32)


def _dither(seconds, seed, start_amp, end_amp):
    """Master 'silence': dither at ~ -110 dB whose amplitude falls across the span, so the
    master's quietest instant is at its END -- where the candidate talks (id 695's shape)."""
    n = int(round(seconds * R))
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * np.linspace(start_amp, end_amp, n)).astype(np.float32)


def _addition(speech_s=2.5, lead_s=0.3, tail_s=1.4, hole_s=0.9, seed=5):
    """Master = c1 (8 s) + dither (hole_s) + c2 (8 s). Candidate = c1 + silence (lead_s) + its
    OWN speech (speech_s) + silence (tail_s) + c2: an addition, a = 0, b = lead+speech+tail-hole.
    Under a, the candidate's speech starts lead_s after edge_A."""
    c1, c2 = _content(8.0, seed), _content(8.0, seed + 1)
    master = np.concatenate([c1, _dither(hole_s, seed + 2, 3e-6, 1e-6), c2])
    own = np.concatenate([np.zeros(int(round(lead_s * R)), np.float32),
                          _content(speech_s, seed + 3),
                          np.zeros(int(round(tail_s * R)), np.float32)])
    cand = np.concatenate([c1, own, c2])
    b_ms = (lead_s + speech_s + tail_s - hole_s) * 1000.0
    return master, cand, b_ms


def _edges(master, cand, a_ms, b_ms, near_s):
    coarse = aw.coarse_edges(master, cand, a_ms, b_ms, near_s - 3.0, near_s + 3.0)
    assert coarse is not None
    edges = aw.fine_edges(master, cand, a_ms, b_ms, *coarse)
    assert edges["status"] == "ok", edges
    return edges


def test_splice_stays_out_of_the_candidates_own_speech():
    master, cand, b_ms = _addition()
    edges = _edges(master, cand, 0.0, b_ms, 8.4)
    assert edges["kind"] == "addition", edges
    lo, hi = edges["interval"]
    # the defect: the master alone ranks the dithered span, its quietest instant is late
    old = aw.quietest_instant(master, lo, hi)
    assert old > 8.3, (old, edges)                   # inside the candidate's speech under a
    leak = aw.leak_bounds(master, cand, 0.0, b_ms, edges["edge_A"], edges["edge_B"], lo, hi)
    assert leak["clean"] and leak["narrowed"], leak
    assert abs(leak["leak_a_s"] - 8.3) <= aw.FINE_WIN_S, leak
    clo, chi = leak["interval"]
    assert clo >= lo - 1e-6 and chi <= 8.3 - aw.LEAK_GUARD_S + 1e-6, leak
    at = aw.quietest_instant(master, clo, chi)
    assert 8.0 - 0.015 <= at <= 8.3 - aw.LEAK_GUARD_S + 1e-6, at


def test_speech_under_the_after_offset_bounds_from_below():
    # the mirror: the candidate's own speech ends 0.3 s before edge_B under b
    master, cand, b_ms = _addition(speech_s=2.5, lead_s=1.4, tail_s=0.3)
    edges = _edges(master, cand, 0.0, b_ms, 8.4)
    lo, hi = edges["interval"]
    leak = aw.leak_bounds(master, cand, 0.0, b_ms, edges["edge_A"], edges["edge_B"], lo, hi)
    assert leak["clean"] and leak["narrowed"], leak
    assert abs(leak["leak_b_s"] - (8.9 - 0.3)) <= aw.FINE_WIN_S, leak
    assert leak["interval"][0] >= 8.6 + aw.LEAK_GUARD_S - 1e-6, leak


def test_silent_own_material_changes_nothing():
    # the candidate's own material is silence: every F is the same splice, the bounds stand
    master, cand, b_ms = _addition()
    lead = int(round(8.0 * R))
    cand[lead:lead + int(round(4.2 * R))] = 0.0
    edges = _edges(master, cand, 0.0, b_ms, 8.4)
    lo, hi = edges["interval"]
    leak = aw.leak_bounds(master, cand, 0.0, b_ms, edges["edge_A"], edges["edge_B"], lo, hi)
    assert leak["clean"] and not leak["narrowed"], leak
    assert leak["interval"] == [lo, hi], leak


def test_no_silent_boundary_takes_the_least_leak():
    # speech right at both edges: no F is clean; the least-leak instant is where the least
    # of it plays (the shorter side)
    master, cand, b_ms = _addition(speech_s=3.5, lead_s=0.0, tail_s=0.0, hole_s=0.9)
    edges = _edges(master, cand, 0.0, b_ms, 8.4)
    lo, hi = edges["interval"]
    leak = aw.leak_bounds(master, cand, 0.0, b_ms, edges["edge_A"], edges["edge_B"], lo, hi)
    assert not leak["clean"], leak
    assert lo <= leak["min_leak_s"] <= hi, leak
    assert leak["interval"] == [lo, hi], leak


def test_candidate_join_is_crossfaded_with_its_margin():
    import merge_video_chimeric as chi
    pieces = [{"source": "candidate", "master_start_ms": Decimal(0),
               "master_end_ms": Decimal(8000), "source_start_ms": Decimal(0)},
              {"source": "candidate", "master_start_ms": Decimal(8000),
               "master_end_ms": Decimal(16000), "source_start_ms": Decimal(11300)}]
    graph, _pad, _heads = chi.build_audio_filtergraph(pieces, 1, None, 48000, "stereo",
                                                      splices={0: {"fade_right": True}})
    assert "acrossfade=d=0.010" in graph, graph
    assert "atrim=start=0.000000:end=8.010000" in graph, graph
    plain, _pad, _heads = chi.build_audio_filtergraph(pieces, 1, None, 48000, "stereo",
                                                      splices={})
    assert "acrossfade" not in plain and "end=8.000000" in plain, plain


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} passed")
