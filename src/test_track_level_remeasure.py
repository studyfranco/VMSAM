'''Tests for `repair_orchestrator.remeasure_at_other_levels` -- run:
python3 src/test_track_level_remeasure.py (or pytest).

CASE_delivery_offset_id126_20260925: a candidate whose dubs do not take the reference track's
steps (id 126: jpn +538/-464/-1464/-2465 ms, en/es/pt -501 ms end to end). Each dub zone is first
searched +/-150 ms around the reference's offset, as production does; only the zone whose
reference level lies near -501 ms measures, and the others must be measured again at the track's
own level instead of being derived through the reference's steps.'''

import os
import sys
from decimal import Decimal

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_walk as aw  # noqa: E402
import repair_orchestrator as ro  # noqa: E402

R = aw.WALK_RATE
TOLERANCE_MS = 15.0          # ADDENDUM 25.3, the delivery tolerance
REFERENCE_MS = [540.0, -460.0, -1460.0, -2460.0]
ZONE_S = 20.0


def _content(seconds, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * R))
    x = np.convolve(x, np.ones(4) / 4, "same")
    return (0.1 * x / np.sqrt(np.mean(x * x))).astype(np.float32)


def _zones():
    return [{"zone": i, "offset_ms": Decimal(str(off)),
             "master_start_ms": Decimal(str(int(i * ZONE_S * 1000))),
             "master_end_ms": Decimal(str(int((i + 1) * ZONE_S * 1000)))}
            for i, off in enumerate(REFERENCE_MS)]


def _first_pass(master, cand, zones):
    '''What `track_offsets` does before the retry: each zone at its reference seed.'''
    readings = []
    for zone in zones:
        low = float(zone["master_start_ms"]) / 1000.0 + ro.ZONE_EDGE_MARGIN_S
        high = float(zone["master_end_ms"]) / 1000.0 - ro.ZONE_EDGE_MARGIN_S
        measured = aw.zone_offset(master, cand, low, high, float(zone["offset_ms"]))
        readings.append({"zone": zone["zone"], "reason": None, "windows": measured["n_ok"],
                         "offset_ms": (None if measured["offset_ms"] is None
                                       else Decimal(str(measured["offset_ms"])))})
    return readings


def _measure(master, cand):
    return lambda low, high, seed: aw.zone_offset(master, cand, low, high, seed)


def _constant_dub(offset_ms):
    '''cand[t + offset] = master[t] over the whole file (offset < 0: the dub starts late).'''
    master = _content(4 * ZONE_S + 5.0, 1)
    shift = int(round(-offset_ms / 1000.0 * R))
    return master, master[shift:].copy()


def test_a_dub_without_the_reference_steps_is_measured_in_every_zone():
    master, cand = _constant_dub(-500.0)
    zones = _zones()
    readings = _first_pass(master, cand, zones)
    assert [r["offset_ms"] is not None for r in readings] == [False, True, False, False], readings
    retries = ro.remeasure_at_other_levels(zones, readings, _measure(master, cand),
                                           aw.WALK_SEARCH_MS)
    assert sorted(r["zone"] for r in retries) == [0, 2, 3], retries
    for reading in readings:
        assert abs(float(reading["offset_ms"]) + 500.0) <= TOLERANCE_MS, readings
        assert reading["reason"] is None


def test_a_track_that_takes_the_reference_steps_is_left_alone():
    master = _content(4 * ZONE_S + 5.0, 2)
    cand = np.zeros(int((4 * ZONE_S + 10.0) * R), dtype=np.float32)
    for i, off in enumerate(REFERENCE_MS):
        m0, m1 = int(i * ZONE_S * R), int((i + 1) * ZONE_S * R)
        c0 = m0 + int(round(off / 1000.0 * R)) + int(3.0 * R)
        cand[c0:c0 + (m1 - m0)] = master[m0:m1]
    cand = cand[int(3.0 * R):]
    zones = _zones()
    readings = _first_pass(master, cand, zones)
    assert all(r["offset_ms"] is not None for r in readings), readings
    before = [r["offset_ms"] for r in readings]
    assert ro.remeasure_at_other_levels(zones, readings, _measure(master, cand),
                                        aw.WALK_SEARCH_MS) == []
    assert [r["offset_ms"] for r in readings] == before


def test_a_zone_no_level_explains_stays_unmeasured():
    master, cand = _constant_dub(-500.0)
    z3 = int(3 * ZONE_S * R)
    cand[z3 - int(1.0 * R):] = _content(len(cand[z3 - int(1.0 * R):]) / R, 77)
    zones = _zones()
    readings = _first_pass(master, cand, zones)
    ro.remeasure_at_other_levels(zones, readings, _measure(master, cand), aw.WALK_SEARCH_MS)
    assert readings[3]["offset_ms"] is None, readings[3]
    assert all(abs(float(r["offset_ms"]) + 500.0) <= TOLERANCE_MS for r in readings[:3]), readings


def test_nothing_measured_means_nothing_to_retry():
    zones = _zones()
    readings = [{"zone": z["zone"], "offset_ms": None, "reason": "x", "windows": 0}
                for z in zones]
    calls = []
    assert ro.remeasure_at_other_levels(
        zones, readings, lambda *a: calls.append(a) or {"offset_ms": None, "n_ok": 0},
        aw.WALK_SEARCH_MS) == []
    assert calls == []


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
