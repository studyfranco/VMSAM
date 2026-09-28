'''Tests for the owner's integrity policy in the repair path (2026-09-25 23:4x) -- run:
python3 src/test_comparison_track_corrupt.py.

  * the free strict decode: `audio_extract.extract_audio_window` decodes with -xerror
    -err_detect crccheck+bitstream+buffer+explode and raises `StrictDecodeFailed` on the
    source's own corruption (real: a 200 s stream-copy cut of the Chainsaw VARYG AMZN DUAL
    around the corrupt E-AC-3 frame at 4 007.968 s, and the ToonsHub cut at the same place);
  * the prime turns it into `comparison_track_corrupt`, and `repair()` routes the pair to the
    ADDENDUM 27.8 video-anchored route when both pictures are sound, else declines
    `comparison_track_corrupt` (ran_conclusive_negative) with the decoder's line;
  * the build drops a rebuilt track whose source fails its strict decode
    (`repair: track_dropped_corrupt`), and `iterate_candidate_audios` no longer yields it;
  * the build's silence gate (owner 2026-09-26 01:3x, e352): a delivered track's silence the
    master's comparison track lacks drops it (`repair: track_dropped_silence`) unless it runs to
    the master's end or every other delivered stream shares it (`repair: track_kept_silence`).
Stubs stand for the measurements where the media is not the point.'''

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "fpcalc": "fpcalc"}
if not getattr(tools, "tmpFolder", None) or tools.tmpFolder == "/tmp":
    tools.tmpFolder = tempfile.mkdtemp(prefix="test_ctc_")

import audio_extract  # noqa: E402
import integrity  # noqa: E402
import merge_video_chimeric  # noqa: E402
import merge_video_repair  # noqa: E402
import repair_orchestrator as ro  # noqa: E402
import video_offset_plan as vop  # noqa: E402

CSM = "/config/example/ChainSaw_Movie"
BAD_SRC = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.AMZN.WEB-DL.DUAL.DDP5.1.H.264-VARYG.mkv"
GOOD_SRC = f"{CSM}/Chainsaw.Man.The.Movie.Reze.Arc.2025.1080p.AMZN.WEB-DL.DUAL.DDP5.1.H.264.MSubs-ToonsHub.mkv"


class Obj:
    def __init__(self, path, audios=None):
        self.filePath = path
        self.video = {"Duration": "1500.0", "FrameRate_Mode": "CFR", "FrameRate_Num": "24000",
                      "FrameRate_Den": "1001", "FrameRate": "23.976", "StreamOrder": "0"}
        self.audios = audios or {}
        self.commentary, self.audiodesc, self.subtitles = {}, {}, {}


class Patch:
    def __init__(self, *triples):
        self.triples, self.saved = triples, []

    def __enter__(self):
        for owner, name, value in self.triples:
            self.saved.append((owner, name, getattr(owner, name, None)))
            setattr(owner, name, value)
        return self

    def __exit__(self, *exc):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)


def _causes(since):
    return [line.split("cause=")[1].split()[0] for line in tools.logs[since:]
            if line.startswith("repair: orchestrator cause=")]


# -- the verdict of an extraction's stderr ---------------------------------------------------

def test_strict_decode_verdict():
    crc = "[eac3 @ 0x55] frame CRC mismatch\n[aist#0:1/eac3 @ 0x55] [dec:eac3 @ 0x5] Error submitting packet to decoder: Invalid data found when processing input\n"
    assert audio_extract.strict_decode_verdict(183, crc)[0].endswith("frame CRC mismatch")
    assert audio_extract.strict_decode_verdict(0, "") is None
    # an rc != 0 without a decoder line keeps its old refusal (not a verdict on the media)
    assert audio_extract.strict_decode_verdict(1, "") is None
    # the seek's complaint on a TrueHD file whose index marks no keyframe is not a verdict
    seek = "[in#0/matroska,webm @ 0x55] File is broken, keyframes not correctly marked!\n"
    assert audio_extract.strict_decode_verdict(0, seek) is None
    assert audio_extract.strict_decode_verdict(1, seek) is None
    assert audio_extract.strict_decode_verdict(
        0, "[af#0:0 @ 0x5] Reconfiguring filter graph because audio parameters changed\n")
    assert audio_extract.STRICT_DECODE_FLAGS == ["-xerror", "-reinit_filter", "0", "-err_detect",
                                                 "crccheck+bitstream+buffer+explode"]


def test_extraction_is_the_strict_decode_on_real_cuts():
    if not (os.path.exists(BAD_SRC) and os.path.exists(GOOD_SRC)):
        print("  skipped: Chainsaw files absent")
        return
    tmp = tempfile.mkdtemp(prefix="test_ctc_cut_")
    try:
        cuts = {}
        for name, src in (("bad", BAD_SRC), ("good", GOOD_SRC)):
            cuts[name] = os.path.join(tmp, f"{name}.mka")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "3900", "-i", src,
                            "-t", "200", "-map", "0:1", "-c", "copy", cuts[name]], check=True)
        out = os.path.join(tmp, "w.wav")
        audio_extract.extract_audio_window(cuts["good"], 0, 0.0, 199.0, out, 16000)
        assert os.path.getsize(out) > 16000 * 2 * 190
        try:
            audio_extract.extract_audio_window(cuts["bad"], 0, 0.0, 199.0, out, 16000)
        except audio_extract.StrictDecodeFailed as error:
            assert error.rc != 0 and error.lines, error
            assert any("CRC" in l or "coupling" in l for l in error.lines), error.lines
            print(f"  corrupt cut refused: rc={error.rc} first={error.lines[0][:90]}")
        else:
            raise AssertionError("the corrupt cut was extracted without a refusal")
        # a window that does not reach the corrupt frame extracts fine
        audio_extract.extract_audio_window(cuts["bad"], 0, 0.0, 100.0, out, 16000)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- the prime -------------------------------------------------------------------------------

def test_prime_turns_a_strict_failure_into_comparison_track_corrupt():
    master = Obj("/m.mkv", {"ja": [{"StreamOrder": "1"}]})
    candidate = Obj("/c.mkv", {"ja": [{"StreamOrder": "2"}]})

    def fp(video_obj, language, stream, side, *a, **k):
        if side == "candidate":
            raise audio_extract.StrictDecodeFailed(video_obj.filePath, stream, 183,
                                                   ["[eac3 @ 0x1] frame CRC mismatch"])
        return [1, 2, 3], ro.CHROMAPRINT_HOP_MS
    primed = {"couples": None, "fingerprints": {}, "alignments": {}, "factor_label": "1",
              "sample_rate": None}
    with Patch((ro, "enumerate_couples", lambda m, c, l: [("1", "2")]),
               (ro, "comparison_sample_rate", lambda m, c, l: 16000),
               (ro, "_track_duration_seconds", lambda v, l, s: 1400.0),
               (ro, "_video_duration_ms", lambda v: 1500000),
               (ro, "fingerprint_track", fp)):
        ok, cause, reason = ro.prime_couples(master, candidate, "ja", "/tmp", primed)
    assert (ok, cause) == (False, ro.COMPARISON_TRACK_CORRUPT), (ok, cause, reason)
    assert primed["corrupt_track"]["side"] == "candidate"
    assert primed["corrupt_track"]["stream"] == "2"
    assert "frame CRC mismatch" in reason
    assert ro.DECLINE_CAUSES[ro.COMPARISON_TRACK_CORRUPT] == ro.CLASS_CONCLUSIVE


def test_fingerprint_track_lets_the_refusal_through():
    def boom(*a, **k):
        raise audio_extract.StrictDecodeFailed("/c.mkv", 2, 183, ["x"])
    with Patch((audio_extract, "extract_audio_window", boom)):
        try:
            ro.fingerprint_track(Obj("/c.mkv"), "ja", 2, "candidate", tools.tmpFolder, 16000,
                                 1400.0)
        except audio_extract.StrictDecodeFailed:
            pass
        else:
            raise AssertionError("fingerprint_track swallowed StrictDecodeFailed")


# -- the route -------------------------------------------------------------------------------

def _run_repair(unreliable, route_status):
    master = Obj("/m.mkv", {"ja": [{"StreamOrder": "1"}]})
    candidate = Obj("/c.mkv", {"ja": [{"StreamOrder": "2"}]})
    calls = []

    def prime(m, c, l, w, primed, resample_routing=None):
        primed["couples"] = [("1", "2")]
        primed["corrupt_track"] = {"side": "candidate", "stream": "2", "rc": 183,
                                   "lines": ["[eac3 @ 0x1] frame CRC mismatch"]}
        return False, ro.COMPARISON_TRACK_CORRUPT, "the candidate ja comparison stream 2 fails"

    def route(trigger, evidence, m, c, language, rows, deadline):
        calls.append((trigger, evidence))
        return route_status, (None if route_status == "repaired" else "video_offset_not_constant"), "r"
    cache = {("conformity", master.filePath): {"verdict": None, "failed": [], "seconds": 0,
                                               "warnings": []}}
    since = len(tools.logs)
    with Patch((ro, "prime_couples", prime),
               (ro, "_video_unreliable", lambda obj: (unreliable.get(obj.filePath, False), "tag")),
               (vop, "video_anchored_route", route),
               (merge_video_repair, "master_intertrack_verdict", lambda *a: None)):
        result = ro.repair(master, candidate, "ja", work_root=tools.tmpFolder,
                           master_intertrack_cache=cache)
    return result, calls, _causes(since), tools.logs[since:]


def test_corrupt_comparison_track_routes_to_the_video_when_both_pictures_are_sound():
    result, calls, causes, logs = _run_repair({}, "repaired")
    assert result is True
    assert calls and calls[0][0] == ro.COMPARISON_TRACK_CORRUPT
    assert calls[0][1]["first"].endswith("frame CRC mismatch")
    assert any("repair: comparison_track_corrupt route=video_anchored" in l for l in logs)


def test_corrupt_comparison_track_declines_when_a_picture_is_unreliable():
    result, calls, causes, logs = _run_repair({"/c.mkv": True}, "repaired")
    assert result is False and not calls
    assert causes == [ro.COMPARISON_TRACK_CORRUPT], causes
    assert any("frame CRC mismatch" in l and "candidate picture is unreliable" in l
               for l in logs), logs[-3:]


def test_no_audio_fallback_once_the_comparison_track_is_corrupt():
    result, calls, causes, logs = _run_repair({}, "fallback")
    assert result is False and calls
    assert causes == ["video_offset_not_constant"], causes


# -- the build's gate ------------------------------------------------------------------------

def test_build_drops_a_rebuilt_track_whose_source_is_corrupt():
    candidate = Obj("/c.mkv", {"ja": [{"StreamOrder": "1"}], "en": [{"StreamOrder": "2"}],
                               "fr": [{"StreamOrder": "3"}], "de": [{"StreamOrder": "4"}]})
    verdicts = {"1": "sound", "2": "corrupt", "3": "decoder_timeout"}
    seen = []

    def check(obj, stream, *a, **k):
        seen.append(str(stream))
        v = verdicts[str(stream)]
        return {"verdict": v, "sound": {"sound": True, "corrupt": False}.get(v), "rc": 183,
                "error_lines": ["[eac3 @ 0x1] frame CRC mismatch"], "codec": "eac3",
                "cost_s": 1.0}
    plan = {"track_plans": {1: {}, 2: {}, 3: {}, 4: {}}, "strictly_decoded_streams": [4]}
    since = len(tools.logs)
    with Patch((integrity, "track_check", check)):
        dropped = merge_video_repair.drop_corrupt_candidate_tracks(candidate, plan)
    assert [d["stream_order"] for d in dropped] == [2]
    assert sorted(seen) == ["1", "2", "3"]              # 4 was decoded strictly at the prime
    assert [a["StreamOrder"] for _, a in merge_video_chimeric.iterate_candidate_audios(candidate)] \
        == ["1", "3", "4"]
    logs = tools.logs[since:]
    assert any(l.startswith("repair: track_dropped_corrupt stream=2 language=en") for l in logs)
    assert any(l.startswith("repair: track_integrity unmeasured stream=3") for l in logs)


def test_corrupt_track_gate_respects_the_repair_deadline():
    # 345a28b0's shape: the deadline is read before each batch of DROP_CORRUPT_JOBS tracks;
    # past it the repair declines repair_budget_exceeded naming the tracks checked so far
    import time
    streams = [str(i) for i in range(1, 8)]
    candidate = Obj("/c.mkv", {f"l{i}": [{"StreamOrder": i}] for i in streams})
    plan = {"track_plans": {int(i): {} for i in streams}, "strictly_decoded_streams": []}
    clock = {"now": 100.0}
    seen = []

    def check(obj, stream, *a, **k):
        seen.append(str(stream))
        clock["now"] += 10.0                                 # each decode costs 10 s
        return {"verdict": "corrupt" if str(stream) == "2" else "sound", "rc": 183,
                "error_lines": ["[eac3 @ 0x1] frame CRC mismatch"], "codec": "eac3",
                "cost_s": 10.0}
    since = len(tools.logs)
    with Patch((integrity, "track_check", check), (time, "monotonic", lambda: clock["now"])):
        try:
            merge_video_repair.drop_corrupt_candidate_tracks(candidate, plan, deadline=125.0)
        except merge_video_chimeric.chimeric_error as error:
            assert getattr(error, "cause", None) == "repair_budget_exceeded", error
            assert "l1:1" in str(error) and "l4:4" in str(error), error
        else:
            raise AssertionError("the gate ran past the repair's deadline without declining")
    assert sorted(seen) == ["1", "2", "3"], seen          # the second batch never started
    logs = tools.logs[since:]
    assert any("repair: partial_plan cause=repair_budget_exceeded stage=corrupt_track_gate" in l
               and "checked_so_far=['l1:1', 'l2:2', 'l3:3']" in l
               and "dropped_so_far=['l2:2']" in l for l in logs), logs[-2:]
    # within the budget, or without a deadline, every track is checked
    for i in streams:
        candidate.audios[f"l{i}"][0].pop("dropped_corrupt", None)
    seen.clear()
    with Patch((integrity, "track_check", check), (time, "monotonic", lambda: clock["now"])):
        merge_video_repair.drop_corrupt_candidate_tracks(candidate, plan, deadline=None)
    assert sorted(seen) == streams


def test_silence_gate_keeps_the_legitimate_and_drops_the_rest():
    # owner 2026-09-26 01:3x (e352 / Netflix credits) -- the measurement is stubbed here; the
    # real one is `integrity.delivered_silence_report` (test_integrity.py, synthetic + e352)
    repaired = Obj("/r.mkv", {"ja": [{"StreamOrder": "1"}], "fr": [{"StreamOrder": "2"}],
                              "en": [{"StreamOrder": "3"}], "de": [{"StreamOrder": "4",
                                                                    "keep": False}],
                              "it": [{"StreamOrder": "5"}]})
    master = Obj("/m.mkv", {"ja": [{"StreamOrder": "1"}]})
    silence = {"start_s": 1419.22, "end_s": 1451.12}
    verdicts = {"1": ("agree", []), "2": ("kept", [dict(silence, rule="outside_master_length")]),
                "3": ("dropped", [dict(silence, start_s=300.0, end_s=330.0, rule=None)])}
    calls = []

    def report(ref, ref_stream, obj, order, delay, master_end_s=None, other_streams=None):
        calls.append((ref_stream, order, delay, master_end_s, tuple(other_streams)))
        if order == "5":
            raise tools.decoder_timeout("silence_map", 120.0, "stub")
        verdict, spans = verdicts[order]
        return {"verdict": verdict, "keep": verdict != "dropped", "silences": spans,
                "reference_only": [], "master_end_s": 1451.117}
    since = len(tools.logs)
    with Patch((integrity, "video_end_s", lambda obj: 1451.117)):
        dropped = merge_video_repair.gate_delivered_silences(repaired, master, "1",
                                                             report=report)
    assert [d["stream_order"] for d in dropped] == [3], dropped
    assert repaired.audios["en"][0]["keep"] is False
    assert repaired.audios["fr"][0].get("keep", True) is True
    assert [c[1] for c in calls] == ["1", "2", "3", "5"]          # 4 already dropped: not judged
    assert calls[0][:4] == ("1", "1", 0, 1451.117) and calls[0][4] == ("1", "2", "3", "5")
    logs = tools.logs[since:]
    assert any(l.startswith("repair: track_kept_silence stream=2 language=fr") and
               "outside_master_length" in l for l in logs), logs
    assert any(l.startswith("repair: track_dropped_silence stream=3 language=en") for l in logs)
    assert any(l.startswith("repair: track_silence unmeasured stream=5") for l in logs)
    # no reference stream: nothing measured, nothing dropped, said once
    assert merge_video_repair.gate_delivered_silences(repaired, master, None, report=report) == []


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as error:                                       # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {error}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
