'''Tests for `merge_video_chimeric.retime_subtitle_file` at a cut
(CASE_subtitle_cue_dropped_at_cut_id714_20260928) -- run: python3 src/test_subtitle_retime.py.

The rule: a cue that straddles a cut is never dropped; it is mapped through every candidate
piece it overlaps and keeps only its kept span, retimed by the plan. Pieces that meet on the
master timeline (a cut: candidate content removed, nothing inserted) give ONE cue; only a cue
entirely inside a removed candidate span is dropped, and counted.'''

import os
import sys
import tempfile
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pysubs2  # noqa: E402

import merge_video_chimeric as mvc  # noqa: E402


def _piece(source, master_start, master_end, source_start=None, reason=None):
    piece = {"source": source, "master_start_ms": Decimal(master_start),
             "master_end_ms": Decimal(master_end), "reason": reason}
    if source == "candidate":
        piece["source_start_ms"] = Decimal(source_start)
    return piece


def _retime(cues, pieces):
    subs = pysubs2.SSAFile()
    for start, end, text in cues:
        subs.events.append(pysubs2.SSAEvent(start=start, end=end, text=text))
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "t.srt")
        subs.save(path)
        kept, dropped, applied, decisions = mvc.retime_subtitle_file(path, pieces)
        out = [(e.start, e.end, e.plaintext) for e in pysubs2.load(path).events]
    return kept, dropped, applied, decisions, out


# A CUT: the candidate's [11.0, 11.5) s is removed. Candidate [0, 11.0) -> master [0, 11.0),
# candidate [11.5, 30.5) -> master [11.0, 30.0) (shift -500 ms).
CUT = [_piece("candidate", 0, 11000, 0), _piece("candidate", 11000, 30000, 11500)]


def test_cue_straddling_a_cut_is_shortened_not_dropped():
    kept, dropped, _applied, decisions, out = _retime([(10000, 12000, "straddles")], CUT)
    assert (kept, dropped) == (1, 0), (kept, dropped, decisions)
    assert out == [(10000, 11500, "straddles")], out
    assert [d["outcome"] for d in decisions] == ["clipped_to_kept_span"], decisions
    assert decisions[0]["gap_ms"] == "500", decisions


def test_cue_entirely_inside_the_removed_span_is_dropped_and_counted():
    kept, dropped, _applied, decisions, out = _retime(
        [(1000, 2000, "kept"), (11100, 11400, "removed")], CUT)
    assert (kept, dropped) == (1, 1), (kept, dropped)
    assert out == [(1000, 2000, "kept")], out
    assert decisions[0]["outcome"] == "dropped_master_filled_span" or \
        decisions[0]["outcome"].startswith("dropped"), decisions


def test_id_714_sign_starting_10_ms_inside_the_removed_span_is_clipped_to_the_piece_start():
    # MEASURED id 714 (DSNP ger/eng/spa/fre/por): piece 3 master [652526.9, 1076460) at
    # +3003.12 ms, piece 4 master [1076460, 1429972) at +4004.125 ms, so the candidate's
    # [1079463.12, 1080464.125) is removed. The sign cue reads candidate 1080454-1082707 ms: its
    # start is 10 ms inside the removed span, and the retimer dropped it on every track.
    pieces = [_piece("candidate", Decimal("652526.9"), 1076460, Decimal("655530.02")),
              _piece("candidate", 1076460, 1429972, Decimal("1080464.125"))]
    kept, dropped, _applied, decisions, out = _retime(
        [(1080454, 1082707, "SEVERAL DAYS LATER")], pieces)
    assert (kept, dropped) == (1, 0), (kept, dropped, decisions)
    assert out == [(1076460, 1078702, "SEVERAL DAYS LATER")], out


def test_cue_across_a_master_only_insert_is_split_into_its_two_kept_spans():
    # AN INSERT: the master carries 1.0 s the candidate lacks, filled from the master. The cue's
    # two halves play on either side of it; stretching one cue over the master's own content
    # would show the line over a scene that does not carry it.
    pieces = [_piece("candidate", 0, 11000, 0), _piece("master", 11000, 12000, reason="interior_bracket"),
              _piece("candidate", 12000, 30000, 11000)]
    kept, dropped, _applied, decisions, out = _retime([(10000, 12000, "split")], pieces)
    assert (kept, dropped) == (2, 0), (kept, dropped, decisions)
    assert out == [(10000, 11000, "split"), (12000, 13000, "split")], out
    assert [d["outcome"] for d in decisions] == ["split_across_master_span"], decisions


def test_cue_inside_one_piece_is_untouched():
    kept, dropped, applied, decisions, out = _retime([(20000, 21000, "plain")], CUT)
    assert (kept, dropped, decisions) == (1, 0, []), (kept, dropped, decisions)
    assert out == [(19500, 20500, "plain")], out
    assert applied == {"-500": 1}, applied


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
