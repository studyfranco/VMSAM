'''Tests for the fine edges of a walk change point (`audio_walk.fine_edges`), and for the walk
finding every level they bound (CASE_audio_step_unlocalised_id111_20260925) -- run:
python3 src/test_audio_walk_edges.py (or pytest).

CASE_hole_width_contradicts_audio_step_20260925: on ids 152/278/686 the levels' step was exact
(1 s xcorr to 0.001 ms up to both edges) and the 20 ms edges were not -- a gain fit straddling
the splice, a stray window setting edge_A, a master fade no 0.4 s gain follows. Each shape is
built here on synthetic audio at WALK_RATE, with the guard that must survive: master-only sound
wider than the step is still a contradiction.'''

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_walk as aw  # noqa: E402

R = aw.WALK_RATE
TOLERANCE_MS = 15.0          # ADDENDUM 25.3, the delivery tolerance


def _content(seconds, seed):
    """Band-limited noise, speech-like level (~ -20 dB)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * R))
    x = np.convolve(x, np.ones(4) / 4, "same")
    return (0.1 * x / np.sqrt(np.mean(x * x))).astype(np.float32)


def _deletion(before_s=8.0, hole_s=1.2, after_s=8.0, seed=1):
    """Master = content[:A] + master-only X (hole_s) + content[A:]; the candidate is the content
    without X. Offsets: a = 0 before, b = -hole_s after."""
    content = _content(before_s + after_s, seed)
    x = _content(hole_s, seed + 100)
    a_i = int(before_s * R)
    master = np.concatenate([content[:a_i], x, content[a_i:]])
    return master, content.copy(), a_i


def _edges(master, cand, a_ms, b_ms, near_s):
    coarse = aw.coarse_edges(master, cand, a_ms, b_ms, near_s - 3.0, near_s + 3.0)
    assert coarse is not None
    return aw.fine_edges(master, cand, a_ms, b_ms, *coarse)


def _gap_minus_step_ms(edges):
    return (edges["edge_B"] - edges["edge_A"]) * 1000.0 - edges["extra_s"] * 1000.0


def test_gain_read_across_the_splice_does_not_move_the_edge():
    # id 152's shape: the candidate is loud just before the splice and the master quiet just
    # after it, so a gain centred on the first windows after edge_B mixes in the loud side.
    master, cand, a_i = _deletion()
    cand[a_i - int(0.2 * R):a_i] *= 8.0
    master[:a_i][-int(0.2 * R):] *= 8.0
    b_i = a_i + int(1.2 * R)
    master[b_i:b_i + int(0.3 * R)] *= 0.1
    cand[a_i:a_i + int(0.3 * R)] *= 0.1
    edges = _edges(master, cand, 0.0, -1200.0, 8.6)
    assert edges["status"] == "ok" and edges["feasible"], edges
    # the centred gain put edge_B 195 ms late here (9.395 s) and the hole did not fit
    assert abs(edges["edge_B"] - 9.2) <= TOLERANCE_MS / 1000.0, edges


def test_isolated_matching_window_is_not_edge_a():
    # id 278's shape: 25 ms inside the master-only span happen to read at offset a, 12 dB
    # under the candidate (the gain clip's floor, as a fading master reads).
    master, cand, a_i = _deletion()
    at = a_i + int(0.5 * R)
    master[at:at + int(0.025 * R)] = 0.25 * cand[at:at + int(0.025 * R)]
    edges = _edges(master, cand, 0.0, -1200.0, 8.6)
    assert edges["status"] == "ok" and edges["feasible"], edges
    assert abs(edges["edge_A"] - 8.0) <= 0.015, edges
    assert abs(_gap_minus_step_ms(edges)) <= TOLERANCE_MS, edges


def test_master_fade_to_silence_is_one_hole():
    # id 686's shape: the master fades out (-180 dB/s) to 1 s of digital silence and fades back
    # in; the candidate runs on, unfaded, and lacks exactly that second.
    content = _content(16.0, 7)
    a_i, hole = int(8.0 * R), int(1.0 * R)
    fade = int(0.5 * R)
    ramp = (10.0 ** (-180.0 * (np.arange(fade) / R) / 20.0)).astype(np.float32)
    master = np.concatenate([content[:a_i], np.zeros(hole, np.float32), content[a_i:]])
    master[a_i - fade:a_i] *= ramp
    master[a_i + hole:a_i + hole + fade] *= ramp[::-1]
    edges = _edges(master, content.copy(), 0.0, -1000.0, 8.5)
    assert edges["status"] == "ok" and edges["feasible"], edges
    assert edges["master_only_audible"] is None, edges
    # every edge sits inside the fades' last audible 0.1 s, not 0.2-0.3 s inward
    assert 7.9 <= edges["edge_A"] <= 8.0 + 0.015 and 9.0 - 0.015 <= edges["edge_B"] <= 9.1, edges


def test_master_only_sound_wider_than_the_step_still_contradicts():
    # THE GUARD: 0.2 s of candidate content matching neither side stands where the master
    # carries sound of its own -- the master-only span is the step + 0.2 s; no fill of the
    # step's width covers it.
    master, cand, a_i = _deletion()
    cand[a_i:a_i + int(0.2 * R)] = _content(0.2, 555)
    edges = _edges(master, cand, 0.0, -1200.0, 8.6)
    assert edges["status"] == "ok", edges
    assert not edges["feasible"], edges
    assert _gap_minus_step_ms(edges) > 150.0, edges


def test_clean_splice_edges_match_the_step():
    master, cand, _a_i = _deletion()
    edges = _edges(master, cand, 0.0, -1200.0, 8.6)
    assert edges["status"] == "ok" and edges["feasible"], edges
    assert abs(_gap_minus_step_ms(edges)) <= TOLERANCE_MS, edges


def _walk_points(master, cand, seeds, probe_gaps=True):
    rows = aw.walk(master, cand, seeds, probe_gaps=probe_gaps)
    found, _outliers = aw.levels(rows)
    return found, aw.change_points(master, cand, found)


def test_walk_finds_the_level_between_two_seeds():
    # id 111's shape (CASE_audio_step_unlocalised_id111_20260925): the candidate lacks two master
    # spans, and the 6 s it carries between them read at an offset no seed searches -- without
    # the gap probe the walk fuses both deletions into one step the edges cannot read.
    a, x, b, y, c = (_content(s, 20 + i) for i, s in enumerate((20.0, 8.0, 6.0, 10.0, 20.0)))
    master = np.concatenate([a, x, b, y, c])
    cand = np.concatenate([a, b, c])
    fused, _points = _walk_points(master, cand, [0.0, -18000.0], probe_gaps=False)
    assert [round(level["off_ms"]) for level in fused] == [0, -18000], fused
    found, points = _walk_points(master, cand, [0.0, -18000.0])
    assert [round(level["off_ms"]) for level in found] == [0, -8000, -18000], found
    assert [round(p["jump_ms"]) for p in points] == [-8000, -10000], points
    for point, (edge_a, edge_b) in zip(points, ((20.0, 28.0), (34.0, 44.0))):
        edges = point["edges"]
        assert edges["status"] == "ok" and edges["feasible"], edges
        assert abs(edges["edge_A"] - edge_a) <= 0.015 and abs(edges["edge_B"] - edge_b) <= 0.015, \
            edges


def test_gap_probe_invents_no_level_in_unrelated_content():
    # THE GUARD: 6 s of the candidate's own content stands in for 14 s of the master's -- the
    # probe searches that span and must find nothing there.
    a, x, c = (_content(s, 40 + i) for i, s in enumerate((20.0, 14.0, 20.0)))
    master = np.concatenate([a, x, c])
    cand = np.concatenate([a, _content(6.0, 99), c])
    found, points = _walk_points(master, cand, [0.0, -8000.0])
    assert [round(level["off_ms"]) for level in found] == [0, -8000], found
    assert len(points) == 1, points


def test_coarse_edges_reach_one_hop_into_each_level():
    # id 111's +623 ms addition: the after-level's first windows match at a 2 s NCC of ~0.65
    # (a remixed candidate), so no 0.4 s window inside the two bounding walk windows claims it;
    # one hop further in, the after-level's content is clean.
    master = _content(44.0, 60)
    a_i = int(20.0 * R)
    tail = master[a_i:].copy()
    noisy = int(2.3 * R)
    tail[:noisy] += 1.2 * _content(2.3, 61)
    cand = np.concatenate([master[:a_i], _content(1.0, 62), tail])
    found, points = _walk_points(master, cand, [0.0, 1000.0])
    assert [round(level["off_ms"]) for level in found] == [0, 1000], found
    before, after = found
    assert aw.coarse_edges(master, cand, 0.0, 1000.0, before["t_last"],
                           after["t_first"] + aw.WALK_WINDOW_S) is None
    edges = points[0]["edges"]
    assert edges["status"] == "ok" and edges["kind"] == "addition", edges
    assert edges["interval"][0] - 0.015 <= 20.0 <= edges["interval"][1] + 0.015, edges


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
