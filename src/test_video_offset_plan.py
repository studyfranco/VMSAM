'''Tests for video_offset_plan.py -- run: python3 src/test_video_offset_plan.py (or pytest).

No media is read: scene lists and per-frame pHashes are synthetic (a random 64-bit picture per
scene, a few bits of motion per frame, a few more bits of "re-encode" noise on the candidate).'''

import os
import sys
import unittest
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import video_offset_plan as vop  # noqa: E402

RATE = Fraction(24000, 1001)
N_FRAMES = 30000


def _flip(rng, hashes, bits):
    out = hashes.copy()
    for _ in range(bits):
        out ^= np.uint64(1) << rng.integers(0, 64, size=len(out)).astype(np.uint64)
    return out


def _timeline(seed, n=N_FRAMES):
    '''(scene_starts, hashes) of a synthetic episode: scenes of 30-150 frames.'''
    rng = np.random.default_rng(seed)
    starts, pos = [], 0
    while pos < n:
        starts.append(pos)
        pos += int(rng.integers(30, 150))
    hashes = np.empty(n, dtype=np.uint64)
    bounds = starts + [n]
    for a, b in zip(bounds[:-1], bounds[1:]):
        hashes[a:b] = np.uint64(int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2)))
    return starts[1:], _flip(rng, hashes, 2)


def _cuts_of(hashes):
    '''Scene starts read back from the hashes themselves (a cut = a jump of > 16 bits).'''
    jumps = np.bitwise_count(hashes[1:] ^ hashes[:-1]) > 16
    return [int(i) + 1 for i in np.nonzero(jumps)[0]]


def _run(m_hashes, c_hashes, lo=-2878, hi=2878):
    matched, ambiguous, total = vop.match_changes(m_hashes, _cuts_of(m_hashes),
                                                  c_hashes, _cuts_of(c_hashes), lo, hi)
    return vop.prove_constant(matched, len(m_hashes)), matched, total


class ConstantShift(unittest.TestCase):

    def _shifted(self, d, head_extra=0):
        rng = np.random.default_rng(7)
        _, master = _timeline(1)
        if d >= 0:   # candidate carries d frames of other content first
            _, other = _timeline(99, n=d + 1)
            cand = np.concatenate([other[:d], master])
        else:        # candidate starts -d frames into the master
            cand = master[-d:].copy()
        return master, _flip(rng, cand, 2)

    def test_positive_shift_found(self):
        master, cand = self._shifted(37)
        proof, matched, total = _run(master, cand)
        self.assertEqual(proof["status"], vop.STATUS_OK)
        self.assertEqual(proof["d"], 37)
        self.assertGreater(proof["paired"], 0.9 * total)
        self.assertTrue(all(mode == 37 for mode, _, _ in proof["thirds"]))
        self.assertLess(proof["mean_hamming"], vop.PAIR_MEAN_HAMMING_MAX)

    def test_negative_shift_found_and_head_added(self):
        master, cand = self._shifted(-240)
        proof, _, _ = _run(master, cand)
        self.assertEqual(proof["status"], vop.STATUS_OK)
        self.assertEqual(proof["d"], -240)
        result = vop.VideoOffsetResult(vop.STATUS_OK, fps=RATE, offset_frames=-240,
                                       master_frames=len(master), candidate_frames=len(cand))
        head_add, tail_add, head_trim, tail_trim = result.head_tail()
        self.assertEqual((head_add, tail_add, head_trim, tail_trim), (240, 0, 0, 0))
        # exact ms: the candidate's tracks move 240 frames LATER, 240 x 1001/24 ms
        self.assertEqual(result.candidate_track_delay_ms, Fraction(240 * 1001, 24))


class ShiftChangingMidFile(unittest.TestCase):

    def test_interior_insert_refused(self):
        '''The Tougen shape: the master carries 120 frames the candidate lacks, mid-file.'''
        rng = np.random.default_rng(3)
        _, master = _timeline(2)
        cut = 17000
        cand = _flip(rng, np.concatenate([master[:cut], master[cut + 120:]]), 2)
        proof, _, _ = _run(master, cand)
        self.assertEqual(proof["status"], vop.STATUS_NOT_CONSTANT)
        modes = [mode for mode, _, _ in proof["thirds"]]
        self.assertEqual(modes[0], 0)
        self.assertEqual(modes[2], -120)

    def test_short_regime_inside_one_third_refused(self):
        '''A shift that holds for a stretch inside one third only is still a second regime.'''
        rng = np.random.default_rng(4)
        _, master = _timeline(5)
        cand = np.concatenate([master[:2000], master[2048:4000], master[3952:]])
        proof, _, _ = _run(master, _flip(rng, cand, 2))
        self.assertEqual(proof["status"], vop.STATUS_NOT_CONSTANT)
        self.assertTrue(proof["regimes"])

    def test_drift_refused(self):
        '''One frame dropped every 2 000: the offset walks across the file.'''
        rng = np.random.default_rng(6)
        _, master = _timeline(8)
        cand = _flip(rng, np.delete(master, np.arange(1000, N_FRAMES, 2000)), 2)
        proof, _, _ = _run(master, cand)
        self.assertEqual(proof["status"], vop.STATUS_NOT_CONSTANT)

    def test_unrelated_pictures_unmatched(self):
        _, master = _timeline(10)
        _, cand = _timeline(11)
        proof, _, _ = _run(master, cand)
        self.assertEqual(proof["status"], vop.STATUS_UNMATCHED)


class FpsPrecondition(unittest.TestCase):

    def _info(self, r, avg):
        return {"r_rate": Fraction(r), "avg_rate": Fraction(avg), "start_s": Fraction(0),
                "duration_s": 1400.0}

    def test_check_fps(self):
        ntsc = "24000/1001"
        self.assertEqual(vop.check_fps(self._info(ntsc, ntsc), self._info(ntsc, ntsc)),
                         (RATE, None))
        rate, why = vop.check_fps(self._info(ntsc, ntsc), self._info("24/1", "24/1"))
        self.assertIsNone(rate)
        self.assertTrue(why.startswith("rates_differ"))
        rate, why = vop.check_fps(self._info(ntsc, ntsc), self._info(ntsc, "1000/42"))
        self.assertIsNone(rate)
        self.assertTrue(why.startswith("candidate_not_cfr"))

    def test_fps_mismatch_declines_without_decoding(self):
        infos = {"m": self._info("24000/1001", "24000/1001"), "c": self._info("25/1", "25/1")}
        saved = (vop.probe_video, vop.decode_scenes_and_hashes)
        decoded, lines = [], []
        try:
            vop.probe_video = lambda path: (infos[path], None)
            vop.decode_scenes_and_hashes = lambda *a, **k: decoded.append(a)
            result = vop.measure_video_offset("m", "c", None, lines.append)
        finally:
            vop.probe_video, vop.decode_scenes_and_hashes = saved
        self.assertEqual(result.status, vop.STATUS_FPS_MISMATCH)
        self.assertEqual(decoded, [])
        self.assertEqual(len(lines), 1)
        self.assertIn("status=fps_mismatch", lines[0])
        self.assertEqual(vop.build_candidate_plan(result, {}, {})["cause"],
                         vop.STATUS_FPS_MISMATCH)


class Plan(unittest.TestCase):

    def test_plan_shifts_every_candidate_track_and_adds_only_at_edges(self):
        result = vop.VideoOffsetResult(vop.STATUS_OK, fps=RATE, offset_frames=3,
                                       master_frames=36000, candidate_frames=35990,
                                       master_start_s=Fraction(0), candidate_start_s=Fraction(0))
        master = {"filePath": "/m.mkv", "audios": {"ja": [{"StreamOrder": "9"}]}}
        cand = {"filePath": "/c.mkv", "audios": {"ja": [{"StreamOrder": "1"}],
                                                  "en": [{"StreamOrder": "2"}]},
                "subtitles": {"en": [{"StreamOrder": "3"}]}}
        plan = vop.build_candidate_plan(result, master, cand, comparison_language="ja")
        self.assertEqual(plan["marker"], "video_anchored:-3")
        self.assertEqual(plan["master"]["tracks"], "untouched")
        self.assertEqual(len(plan["candidate"]["tracks"]), 3)
        self.assertTrue(all(t["shift_frames"] == -3 for t in plan["candidate"]["tracks"]))
        self.assertEqual(plan["candidate"]["delay_ms_exact"], str(Fraction(-3 * 1001, 24)))
        self.assertEqual(plan["head"]["add_frames"], 0)
        self.assertEqual(plan["head"]["trim_frames"], 3)
        self.assertEqual(plan["tail"]["add_frames"], 13)
        self.assertEqual(plan["tail"]["fill"]["source"], "master_audio")
        self.assertEqual(plan["interior"], "none")

    def test_phash_matches_scipy_reference(self):
        from scipy.fft import dctn
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, size=(16, 36, 64)).astype(np.float64)
        ref = dctn(frames, type=2, norm="ortho", axes=(1, 2))[:, :8, :8].reshape(16, 64)
        bits = (ref > np.median(ref[:, 1:], axis=1, keepdims=True)).astype(np.uint64)
        expected = (bits * (np.uint64(1) << np.arange(64, dtype=np.uint64))).sum(
            axis=1, dtype=np.uint64)
        self.assertTrue(np.array_equal(vop.phash64(frames), expected))


if __name__ == "__main__":
    unittest.main()
