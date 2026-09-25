'''Tests for frame_snap.py -- run: python3 src/test_frame_snap.py (or pytest).

No media is read: the frame groups are synthetic, and the call-site test runs
`adjust_delay_to_frame` itself with `frame_snap.snap_for_merge` replaced.'''

import ast
import os
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import frame_snap as fs  # noqa: E402

RATE = Fraction(24000, 1001)
FRAME_S = float(1 / RATE)


def _clip(n, seed):
    '''n frames of a textured picture panning one pixel per frame.'''
    rng = np.random.default_rng(seed)
    wide = rng.uniform(0, 255, size=(fs.HEIGHT, fs.WIDTH + n + 8)).astype(np.float32)
    wide = (wide + np.roll(wide, 1, axis=1) + np.roll(wide, 1, axis=0)) / 3.0  # some spatial coherence
    return [wide[:, i:i + fs.WIDTH].copy() for i in range(n)]


def _position(clip, start, true_shift, base, noise_seed):
    '''One position: the first video's group at clip[start..], and the
    second's G + 2*REACH frames decoded where `base` says, while its content
    really sits `true_shift` frames after the first's.'''
    rng = np.random.default_rng(noise_seed)
    g, reach = fs.GROUP_FRAMES, fs.REACH
    ref = clip[start:start + g]
    ref_times = np.array([(start + i) * FRAME_S for i in range(g)])
    # the second video's frame at index k (its own timeline) shows clip[k - true_shift]
    first_other = start + base - reach
    other, other_times = [], []
    for k in range(first_other, first_other + g + 2 * reach):
        picture = clip[k - true_shift] + rng.normal(0, 2.0, size=clip[0].shape).astype(np.float32)
        other.append(np.clip(picture, 0, 255))
        other_times.append(k * FRAME_S)
    return fs.score_offsets([fs.phash(f) for f in ref], ref_times,
                            [fs.phash(f) for f in other], np.array(other_times), base, FRAME_S)


def test_known_plus_one_shift_is_found():
    clip = _clip(400, 1)
    # the audio said 10 frames; the pictures sit 11 frames later
    rows = []
    for i, start in enumerate(range(20, 340, 40)):
        scores = _position(clip, start, true_shift=11, base=10, noise_seed=i)
        scores["t"] = start * FRAME_S
        assert min((1, 0, -1), key=lambda o: scores[o]) == 1, scores
        rows.append(scores)
    offset, reason, detail = fs.decide(rows)
    assert (offset, reason) == (1, "frame_snap_chosen"), (offset, reason, detail)
    assert detail["winners"] == [1] * 8


def test_exact_base_keeps_zero():
    clip = _clip(400, 2)
    rows = []
    for i, start in enumerate(range(20, 340, 40)):
        scores = _position(clip, start, true_shift=-7, base=-7, noise_seed=100 + i)
        scores["t"] = start * FRAME_S
        rows.append(scores)
    assert fs.decide(rows)[:2] == (0, "frame_snap_chosen")


def _row(winner, t):
    row = {o: 10.0 for o in fs.VOTE_OFFSETS + fs.GUARD_OFFSETS}
    row[winner] = 1.0
    row["t"] = t
    return row


def test_drift_is_refused():
    winners = [-1, -1, -1, 0, 0, 1, 1, 1]
    offset, reason, detail = fs.decide([_row(w, 100.0 * i) for i, w in enumerate(winners)])
    assert offset is None and reason == "frame_snap_drift", (offset, reason)
    assert detail["winners"] == winners
    # a drift that the pictures show on real groups: the shift grows along the file
    clip = _clip(400, 3)
    rows = []
    for i, (start, shift) in enumerate(zip(range(20, 340, 40), (9, 9, 9, 10, 10, 11, 11, 11))):
        scores = _position(clip, start, true_shift=shift, base=10, noise_seed=200 + i)
        scores["t"] = start * FRAME_S
        rows.append(scores)
    assert fs.decide(rows)[:2] == (None, "frame_snap_drift")


def test_single_outlier_is_not_drift_and_split_is_no_consensus():
    assert fs.decide([_row(w, i) for i, w in enumerate([0, 0, 0, 0, 0, 0, 0, 1])])[:2] == (0, "frame_snap_chosen")
    assert fs.decide([_row(w, i) for i, w in enumerate([0, 1, 0, 1, 0, 1, 0, 1])])[:2] == (None, "frame_snap_no_consensus")
    assert fs.decide([_row(0, i) for i in range(4)])[:2] == (None, "frame_snap_too_few_positions")


def test_guard_never_selected():
    rows = []
    for i in range(8):
        row = {-2: 1.0, -1: 5.0, 0: 9.0, 1: 12.0, 2: 14.0, "t": i}
        rows.append(row)
    offset, reason, _ = fs.decide(rows)
    assert offset is None and reason == "audio_video_offset_disagree"


def test_two_frame_audio_picture_disagreement_is_signalled():
    '''Addendum 27.7: the audio says 10 frames, the pictures 12. The rounding
    is kept, and the disagreement is logged and returned to the caller.'''
    clip = _clip(400, 4)
    rows = []
    for i, start in enumerate(range(20, 340, 40)):
        scores = _position(clip, start, true_shift=12, base=10, noise_seed=300 + i)
        scores["t"] = start * FRAME_S
        rows.append(scores)
    offset, reason, detail = fs.decide(rows)
    assert (offset, reason) == (None, "audio_video_offset_disagree"), (offset, reason, detail)
    assert detail["picture_offset"] == 2 and detail["picture_votes"] == 8

    frame_ms = Decimal('1000.0') / Decimal("23.976")
    audio_delay = Decimal(10) * frame_ms + Decimal("3.0")
    real_measure, lines = fs.measure, []
    real_log = fs.tools.log_always
    fs.measure = lambda *a, **k: (10, rows, [], 1.0)
    fs.tools.log_always = lines.append
    try:
        v1, v2 = _Video("a.mkv"), _Video("b.mkv")
        out = fs.snap_for_merge(v1, v2, v1, audio_delay)
    finally:
        fs.measure, fs.tools.log_always = real_measure, real_log
    assert out == audio_delay                                   # default behaviour: audio rounding
    signal = fs.disagreement("a.mkv", "b.mkv")
    assert signal["audio_frames"] == 10 and signal["picture_frames"] == 12 and signal["votes"] == 8, signal
    assert any(l.startswith("frame_snap audio_video_offset_disagree:") for l in lines), lines


def test_exact_rate_reads_rationals_never_the_decimal():
    assert fs.exact_rate({"FrameRate_Num": "24000", "FrameRate_Den": "1001", "FrameRate": "23.976"})[0] == RATE
    assert fs.exact_rate({"ffprobe": {"r_frame_rate": "24000/1001"}, "FrameRate": "23.976"})[0] == RATE
    assert fs.exact_rate({"FrameRate": "23.976"})[0] is None


class _Video:
    def __init__(self, path, mode="CFR", rate=("24000", "1001", "23.976")):
        self.filePath = path
        self.video = {"FrameRate_Mode": mode, "FrameRate_Num": rate[0], "FrameRate_Den": rate[1],
                      "FrameRate": rate[2], "StreamOrder": "0", "Duration": "1400.0"}


def test_call_site_is_pinned_in_adjust_delay_to_frame():
    '''`adjust_delay_to_frame` calls frame_snap.snap_for_merge BEFORE its
    rounding, on the CFR branch, and the rounding lands on what it returns.'''
    src = open(os.path.join(HERE, "mergeVideo.py")).read()
    func = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "adjust_delay_to_frame")
    cfr_branch = next(n for n in func.body if isinstance(n, ast.If)).body
    calls = [i for i, st in enumerate(cfr_branch)
             if "frame_snap.snap_for_merge" in ast.unparse(st)]
    rounding = [i for i, st in enumerate(cfr_branch) if "round(delay / framerate_duration_ms)" in ast.unparse(st)]
    first_return = next(i for i, st in enumerate(cfr_branch) if isinstance(st, ast.Return))
    assert calls and rounding and calls[0] < rounding[0] < first_return, (calls, rounding, first_return)

    # and it really runs: the method, with the snap replaced by a recorder
    import mergeVideo
    seen = []

    def fake_snap(v1, v2, best, delay):
        seen.append((v1, v2, best, Decimal(delay)))
        return Decimal(2) * Decimal('1000.0') / Decimal("23.976")   # "the pictures chose frame 2"

    real = fs.snap_for_merge
    fs.snap_for_merge = fake_snap
    try:
        obj = mergeVideo.compare_video.__new__(mergeVideo.compare_video)
        obj.video_obj_1, obj.video_obj_2 = _Video("a.mkv"), _Video("b.mkv")
        obj.video_obj_with_best_quality = obj.video_obj_1
        out = obj.adjust_delay_to_frame(Decimal("62.0"))   # rounds to frame 1 on its own
    finally:
        fs.snap_for_merge = real
    assert len(seen) == 1 and seen[0][0] is obj.video_obj_1 and seen[0][1] is obj.video_obj_2
    assert round(out / (Decimal('1000.0') / Decimal("23.976"))) == 2, out


def test_same_family_rates_match_and_1001_ratios_do_not():
    assert fs.same_rate(Fraction(18965, 791), RATE)          # ToonsHub vs 24000/1001: 1.8e-6
    assert not fs.same_rate(Fraction(24, 1), RATE)           # 1001/1000 apart
    assert not fs.same_rate(Fraction(25, 1), Fraction(25000, 1001))
    assert not fs.same_rate(None, RATE)


def test_different_rates_and_vfr_decline_untouched():
    for v2 in (_Video("b.mkv", rate=("25", "1", "25.000")), _Video("b.mkv", mode="VFR")):
        assert fs.snap_for_merge(_Video("a.mkv"), v2, _Video("a.mkv"), Decimal("62.5")) == Decimal("62.5")


if __name__ == "__main__":
    import tools
    tools.log_always = lambda message: None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
