'''Tests for the rate arm's pure half (ADDENDUM 30/30.5) -- run: python3 src/test_rate_arm.py (or
pytest). No media: the finalist rows carry the numbers measured on the ids named.'''

import os
import sys
from decimal import Decimal
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import merge_video_chimeric as mvc  # noqa: E402
import merge_video_resample as mvr  # noqa: E402
import rate_direction as rd  # noqa: E402
import repair_orchestrator as ro  # noqa: E402


def _row(ratio, engine, span, zones, ladder=False):
    return {"ratio": Fraction(ratio), "engine": engine, "span_coverage": span, "zones": zones,
            "offset_spread_ms": 0.0, "ladder": ladder}


def test_id_101_asetrate_wins():
    rows = [_row(1, None, 0.007, 1), _row("1001/960", "asetrate", 0.9997, 1),
            _row("1001/960", "atempo", 0.02, 2), _row("25/24", "asetrate", 0.97, 15, ladder=True)]
    winner = rd.choose_winner(rows)
    assert winner["ratio"] == Fraction(1001, 960) and winner["engine"] == "asetrate", winner


def test_id_57_atempo_wins_on_span_not_points():
    # atempo's fingerprints are sparser (points 0.533) but its one zone spans 0.97 of the master
    rows = [_row(1, None, 0.75, 141, ladder=True), _row("1001/960", "asetrate", 0.003, 1),
            _row("1001/960", "atempo", 0.97, 3)]
    winner = rd.choose_winner(rows)
    assert winner["engine"] == "atempo", winner


def test_unity_wins_and_nothing_qualifies():
    assert rd.choose_winner([_row(1, None, 0.99, 4), _row("1001/1000", "asetrate", 0.95, 12,
                                                        ladder=True)])["ratio"] == 1
    assert rd.choose_winner([_row(1, None, 0.2, 3), _row("25/24", "atempo", 0.4, 9)]) is None


def test_tie_prefers_asetrate_then_fewer_zones():
    rows = [_row("1001/1000", "asetrate", 0.995, 2), _row("1001/1000", "atempo", 0.996, 2)]
    assert rd.choose_winner(rows)["engine"] == "asetrate"
    # Fallout S01E03, measured: the engines tie at NTSC -- asetrate, whatever the zone count
    rows = [_row("1001/1000", "asetrate", 0.9972, 6), _row("1001/1000", "atempo", 0.9965, 4)]
    assert rd.choose_winner(rows)["engine"] == "asetrate"
    rows = [_row("25/24", "asetrate", 0.95, 9), _row("1001/960", "asetrate", 0.955, 1)]
    assert rd.choose_winner(rows)["ratio"] == Fraction(1001, 960)


def test_first_finalists_carry_both_directions():
    firsts = rd.first_finalists(Fraction(1001, 960), [Fraction(1001, 960), Fraction(25, 24)])
    assert firsts == [Fraction(1001, 960), Fraction(960, 1001), Fraction(25, 24),
                      Fraction(24, 25)], firsts
    assert rd.first_finalists(None, []) == []


def test_quantum_flicker_merged():
    detail = [{"offset_points": 3, "master_ms": [0, 100]}, {"offset_points": 4, "master_ms": [100, 110]},
              {"offset_points": 3, "master_ms": [110, 300]}, {"offset_points": 9, "master_ms": [300, 400]}]
    zones = [[[0, 1], [0, 1]]] * 4
    _z, out, merged = ro.merge_quantum_flicker(zones, detail)
    assert [d["offset_points"] for d in out] == [3, 3, 3, 9], out
    assert merged == [([100, 110], 4, 3)], merged
    assert detail[1]["offset_points"] == 4        # the aligner's own list is untouched


def test_engines_and_markers():
    chain, effective = mvr.build_transform_chain(48000, Fraction(1001, 960), "atempo")
    assert chain.startswith("atempo=") and effective == Decimal(1001) / Decimal(960)
    chain, effective = mvr.build_transform_chain(48000, Fraction(1001, 960), "asetrate")
    assert "asetrate=" in chain and abs(effective - Decimal(1001) / Decimal(960)) < Decimal("1e-6")
    assert mvc.compose_marker("chimeric", Decimal(1001) / Decimal(960), "atempo") == \
        "chimeric+atempo:1001/960"
    assert mvc.compose_marker("", Decimal(1001) / Decimal(1000)) == "resampled:1001/1000"


if __name__ == "__main__":
    import tools
    tools.log_always = lambda message: None
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
