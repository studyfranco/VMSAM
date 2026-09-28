'''Tests for ADDENDUM 28 in the repair path (`language_content_check`, the orchestrator's
`_language_content_route`) -- run: python3 src/test_language_content_check.py.

Synthetic chromaprint lists stand for the fingerprints (the instrument is the prime's fpcalc and
file_conformity's Hamming similarity; the media is not the point here):
  * `pair_similarity`: the same content at an offset reads ~1.0, other content ~0.5, a silent
    window is skipped;
  * the id-110 shape (master ja; candidate en + « ja » = the English duplicated) declines
    `no_common_language_after_tag_check` with the best similarity, and logs the candidate's
    same-file `audio_tag_conflict`;
  * a lying tag (candidate « en » = the master's Japanese) routes `audio_tag_conflict`: the track
    is re-tagged ja with `VMSAM_tag_corrected`, the old « ja » track leaves the comparison
    language, only the matched track's WAV is kept, and the repair re-runs ONCE inside the same
    deadline with the prime's fingerprints;
  * truthful tags -> nothing to say; the class map (conclusive / routing).'''

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "fpcalc": "fpcalc"}
if not getattr(tools, "tmpFolder", None) or tools.tmpFolder == "/tmp":
    tools.tmpFolder = tempfile.mkdtemp(prefix="test_lcc_")

import language_content_check as lcc  # noqa: E402
import repair_orchestrator as ro  # noqa: E402

HOP = ro.CHROMAPRINT_HOP_MS
N = int(1400 * 1000 / HOP)                       # a 1 400 s track


def content(seed, n=N):
    return np.random.default_rng(seed).integers(0, 2 ** 32, size=n, dtype=np.uint64) \
        .astype(np.uint32).tolist()


def shifted(points, items):
    """The same content, `items` later (a container delay)."""
    pad = content(999, items)
    return (pad + points)[:len(points)]


class Obj:
    def __init__(self, path, audios):
        self.filePath = path
        self.video = {"Duration": "1400.0", "StreamOrder": "0"}
        self.audios = audios
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


def a(order):
    return {"StreamOrder": order, "Duration": "1400.0"}


JA, EN = content(1), content(2)


def stub_fingerprint(table, made):
    def fingerprint(video_obj, language, order, side, work_dir, sample_rate, duration,
                    measures=None, keep_wav=False):
        made.append((side, str(order)))
        if keep_wav and measures is not None:
            wav = os.path.join(work_dir, f"lcc_{side}_{order}.wav")
            open(wav, "wb").write(b"RIFF")
            measures["wav"] = wav
        return table[(side, str(order))], HOP
    return fingerprint


def test_pair_similarity():
    s, w = lcc.pair_similarity(JA, shifted(JA, int(7000 / HOP)), HOP)
    assert s is not None and s > 0.99 and len(w) == 3, (s, w)
    assert abs(w[0][2] - 7.0) < 0.2, w
    s, _ = lcc.pair_similarity(JA, EN, HOP)
    assert s is not None and s < 0.6, s
    silent = [7] * N
    s, w = lcc.pair_similarity(silent, silent, HOP)
    assert s is None and w == [], (s, w)


def _run(master, candidate, table, couples, language="ja"):
    made = []
    primed = {"couples": couples, "sample_rate": 16000,
              "fingerprints": {(side, o): (table[(side, o)], HOP, 1400.0)
                               for side, o in [("master", m) for m, _ in couples]
                               + [("candidate", c) for _, c in couples]}}
    since = len(tools.logs)
    check = lcc.content_check(master, candidate, language, primed, tools.tmpFolder,
                              fingerprint=stub_fingerprint(table, made))
    return check, primed, made, tools.logs[since:]


def test_id110_shape_declines_no_common_language():
    master = Obj("/m.mkv", {"ja": [a("1")]})
    candidate = Obj("/c.mkv", {"en": [a("1")], "ja": [a("2")]})
    table = {("master", "1"): JA, ("candidate", "1"): EN, ("candidate", "2"): list(EN)}
    check, primed, made, logs = _run(master, candidate, table, [("1", "2")])
    assert check["verdict"] == lcc.NO_COMMON_LANGUAGE, check["verdict"]
    assert made == [("candidate", "1")], made            # the prime's fingerprints reused
    assert check["best"]["similarity"] < 0.6
    assert "best similarity" in check["reason"]
    assert any(l.startswith("repair: audio_tag_conflict within=candidate") and "en/ja" in l
               for l in logs), logs
    assert "tag_check_wav" not in primed
    assert not [f for f in os.listdir(tools.tmpFolder) if f.startswith("lcc_")]


def test_lying_tag_routes_and_retags():
    master = Obj("/m.mkv", {"ja": [a("1")], "en": [a("2")]})
    candidate = Obj("/c.mkv", {"en": [a("1")], "ja": [a("2")], "fr": [a("3")]})
    table = {("master", "1"): JA, ("master", "2"): EN,
             ("candidate", "1"): shifted(JA, 40), ("candidate", "2"): list(EN),
             ("candidate", "3"): content(3)}
    check, primed, made, logs = _run(master, candidate, table, [("1", "2")])
    assert check["verdict"] == lcc.AUDIO_TAG_CONFLICT, check
    assert (check["track"], check["tag"], check["master_stream"]) == ("1", "en", "1")
    assert check["similarity"] > 0.99
    assert primed["tag_check_wav"]["stream"] == "1"      # only the matched track's WAV kept
    assert [f for f in os.listdir(tools.tmpFolder) if f.startswith("lcc_")] == \
        ["lcc_candidate_1.wav"]
    moves = {m["stream"]: (m["from"], m["to"]) for m in check["moves"]}
    assert moves == {"1": ("en", "ja"), "2": ("ja", "en")}, moves   # 2 = the master's en
    lcc.apply_correction(candidate, check)
    assert [x["StreamOrder"] for x in candidate.audios["ja"]] == ["1"]
    assert candidate.audios["ja"][0]["VMSAM_tag_corrected"].startswith("en->ja similarity=")
    assert [x["StreamOrder"] for x in candidate.audios["en"]] == ["2"]
    assert candidate.tag_checked is True
    lcc.drop_kept_wav(primed)


def test_truthful_tags_say_nothing():
    master = Obj("/m.mkv", {"ja": [a("1")]})
    candidate = Obj("/c.mkv", {"ja": [a("2")]})
    table = {("master", "1"): JA, ("candidate", "2"): shifted(JA, 10)}
    check, primed, made, logs = _run(master, candidate, table, [("1", "2")])
    assert check["verdict"] == "tags_consistent", check["verdict"]
    assert made == []


def test_route_in_repair():
    master = Obj("/m.mkv", {"ja": [a("1")]})
    candidate = Obj("/c.mkv", {"en": [a("1")], "ja": [a("2")]})
    calls = []

    def fake_check(m, c, language, primed, work_dir, deadline=None):
        calls.append(deadline)
        return {"verdict": lcc.AUDIO_TAG_CONFLICT, "track": "1", "tag": "en",
                "master_stream": "1", "similarity": 0.99, "best": {"similarity": 0.99},
                "moves": [{"stream": "1", "from": "en", "to": "ja", "holder": "audios",
                           "similarity": 0.99, "matched": "master#1"}], "cost_s": 0.1}
    reruns = []

    def fake_repair(m, c, language, work_root=None, master_intertrack_cache=None,
                    _carried_deadline=None, _carried_prime=None):
        reruns.append((_carried_deadline, _carried_prime))
        return True
    primed = {"fingerprints": {("master", "1"): ([1], HOP, 1.0)}, "sample_rate": 16000,
              "content_end": {}}
    since = len(tools.logs)
    with Patch((lcc, "content_check", fake_check), (ro, "repair", fake_repair)):
        out = ro._language_content_route(master, candidate, "ja", primed, "/w", None, 1234.5,
                                         {}, 1200.0)
        assert out is True and calls == [1234.5]
        assert reruns[0][0] == 1234.5
        assert reruns[0][1]["fingerprints"] == primed["fingerprints"]
        assert "1" in [x["StreamOrder"] for x in candidate.audios["ja"]]
        assert "en" not in candidate.audios
        # once: the corrected candidate is never checked again
        assert ro._language_content_route(master, candidate, "ja", primed, "/w", None, 1.0,
                                          {}, 1200.0) is None
    logs = tools.logs[since:]
    assert any(l.startswith("repair: audio_tag_conflict route=retag_and_rerun track=1 tag=en "
                            "matched_language=ja") for l in logs), logs

    def fake_no_common(*args, **kwargs):
        return {"verdict": lcc.NO_COMMON_LANGUAGE, "reason": "best similarity 0.1500",
                "best": {"similarity": 0.15}, "references": ["1"], "cost_s": 0.1}
    candidate2 = Obj("/c2.mkv", {"en": [a("1")], "ja": [a("2")]})
    since = len(tools.logs)
    with Patch((lcc, "content_check", fake_no_common)):
        out = ro._language_content_route(master, candidate2, "ja", {}, "/w", None, 1.0, {}, 1.0)
    assert out is False
    logs = tools.logs[since:]
    assert any("cause=no_common_language_after_tag_check" in l and "best similarity 0.1500" in l
               for l in logs), logs


def test_no_comparison_language_track_is_decided_before_the_prime():
    # CASE_id684: master jpn only, candidate eng only -- the repair decides by content, by name,
    # before the prime (whose ValueError "the caller guarantees the language" is never reached)
    import merge_video_repair
    master = Obj("/m684.mkv", {"ja": [a("1")]})
    candidate = Obj("/c684.mkv", {"en": [a("1")]})
    table = {("master", "1"): JA, ("candidate", "1"): EN}
    made, primes = [], []
    cache = {("conformity", master.filePath): {"verdict": None, "failed": [], "seconds": 0,
                                               "warnings": []}}
    real = lcc.content_check

    def check(m, c, language, primed, work_dir, deadline=None):
        return real(m, c, language, primed, work_dir, deadline,
                    fingerprint=stub_fingerprint(table, made))
    since = len(tools.logs)
    with Patch((lcc, "content_check", check),
               (ro, "prime_couples", lambda *x, **k: primes.append(1)),
               (merge_video_repair, "master_intertrack_verdict", lambda *x: None)):
        out = ro.repair(master, candidate, "ja", work_root=tools.tmpFolder,
                        master_intertrack_cache=cache)
    assert out is False and not primes
    assert sorted(made) == [("candidate", "1"), ("master", "1")], made
    logs = tools.logs[since:]
    assert any("declined cause=no_common_language_after_tag_check" in l for l in logs), logs[-3:]
    err = ro.VmsamDecline("No common language", lcc.NO_COMMON_LANGUAGE)
    assert err.cause == lcc.NO_COMMON_LANGUAGE and isinstance(err, Exception)


def test_class_map():
    assert ro.DECLINE_CAUSES[lcc.NO_COMMON_LANGUAGE] == ro.CLASS_CONCLUSIVE
    assert lcc.AUDIO_TAG_CONFLICT not in ro.DECLINE_CAUSES
    assert ro.ROUTING_SIGNALS[lcc.AUDIO_TAG_CONFLICT] == "routing"


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
