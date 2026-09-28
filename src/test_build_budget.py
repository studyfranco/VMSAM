'''Tests for the repair budget over the track build (ADDENDUM 26.3 / 26.8) -- run:
python3 src/test_build_budget.py (or pytest). The real `assemble_on_master_timeline` loop over a
twelve-track candidate, each track's build replaced by a one-second stand-in: with a budget of
2.5 s the build must stop by name within the budget + one build, name the tracks built, and
never reach the mux.'''

import os
import sys
import tempfile
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402
import merge_video_chimeric as mvc  # noqa: E402

BUILD_S = 1.0
TRACKS = 12


class _Video:
    def __init__(self, n_audio):
        self.filePath = "/synthetic/candidate.mkv"
        self.audios = {"ja": [{"StreamOrder": i + 1} for i in range(n_audio)]}
        self.audiodesc, self.commentary, self.subtitles = {}, {}, {}
        self.video = {"Duration": "60.0", "FrameRate": "23.976", "FrameRate_Mode": "CFR"}


def _run(budget_s, logs):
    piece = [{"source": "candidate", "master_start_ms": Decimal(0),
              "master_end_ms": Decimal(60000), "source_start_ms": Decimal(0)}]
    plans = {i + 1: {"pieces": piece, "extent_ms": Decimal(60000), "extent_source": "test",
                     "offset_measured": True} for i in range(TRACKS)}
    built, muxed = [], []
    real_build, real_mux = mvc.build_one_audio_track, mvc.mux_repaired_file

    def fake_build(candidate_obj, master_obj, audio, *args, **kwargs):
        time.sleep(BUILD_S)
        built.append(audio["StreamOrder"])
        return {"stream_order": audio["StreamOrder"], "speed_ratio_applied": None}

    mvc.build_one_audio_track = fake_build
    mvc.mux_repaired_file = lambda *a, **k: muxed.append(True)
    tools.log_always = lambda message: logs.append(message)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory() as work:
            mvc.assemble_on_master_timeline(
                _Video(TRACKS), _Video(1), plans, piece, work, os.path.join(work, "o.mkv"), "",
                "test", deadline=started + budget_s)
        return None, built, muxed, time.monotonic() - started
    except mvc.chimeric_error as error:
        return error, built, muxed, time.monotonic() - started
    finally:
        mvc.build_one_audio_track, mvc.mux_repaired_file = real_build, real_mux


def test_track_build_declines_by_the_budget_within_one_build():
    logs = []
    error, built, muxed, elapsed = _run(2.5, logs)
    assert error is not None and getattr(error, "cause", None) == "repair_budget_exceeded", error
    assert elapsed <= 2.5 + BUILD_S + 0.5, elapsed                    # budget + one build
    assert 2 <= len(built) <= 4 and not muxed, (built, muxed)          # stopped, never muxed
    assert any("partial_plan cause=repair_budget_exceeded" in m and "audio:1" in m for m in logs), logs



# ---- OWNER 2026-09-28: the plan on EVERY non-video candidate track, the ordinary selection after ----

def _track(order, language, rate="48000", **extra):
    track = {"StreamOrder": str(order), "keep": True, "Format": "AAC", "Channels": "2",
             "SamplingRate": rate, "BitRate": "128000", "Language": language}
    track.update(extra)
    return track


class _Candidate:
    def __init__(self):
        self.filePath = "/synthetic/candidate.mkv"
        ja = [_track(1, "ja")]
        self.audios = {"ja": ja, "en": [_track(2, "en")], "und": [_track(3, "und")]}
        self.audios["fr"] = self.audios["und"]        # video.py's 'und' alias: one dict, two keys
        self.audiodesc, self.commentary = {}, {}
        self.subtitles = {"en": [_track(4, "en", Format="ASS")],
                          "ja": [_track(5, "ja", Format="PGS")]}


class _Master:
    def __init__(self):
        self.filePath = "/synthetic/master.mkv"
        self.audios = {"ja": [_track(1, "ja", "44100")], "en": [_track(2, "en", "44100")],
                       "fr": [_track(3, "fr", "44100")]}
        self.audiodesc, self.commentary, self.subtitles = {}, {}, {}


class _Repaired:
    """The temporary file as `video.video` re-probes it: the rebuilt audios, then the subtitle."""
    def __init__(self, *args):
        self.filePath = "/synthetic/repair/key_repaired.mkv"
        marked = {"VMSAM_FABRICATED": "chimeric"}
        self.audios = {"ja": [_track(0, "ja", extra=marked)], "en": [_track(1, "en", extra=marked)],
                       "fr": [_track(2, "fr", extra=marked)]}
        self.audiodesc, self.commentary = {}, {}
        self.subtitles = {"en": [_track(3, "en", Format="ASS")]}

    def get_mediadata(self):
        pass


def _plan():
    return {"track_plans": {1: {"offset_measured": True}, 2: {"offset_measured": True},
                            3: {"offset_measured": False,
                                "borrow_reason": "inherited(stream_1)"}},
            "reference_pieces": [], "marker": "chimeric", "reference_stream": "1",
            "language": "ja"}


def _assembly():
    return {"path": "/synthetic/repair/key_repaired.mkv", "marker": "chimeric",
            "audios": [{"stream_order": 1, "language": "ja"}, {"stream_order": 2, "language": "en"},
                       {"stream_order": 3, "language": "fr"}],
            "subtitles": [{"stream_order": 4, "language": "en", "kept_cues": 300,
                           "dropped_cues": 2}],
            "declined": [{"kind": "subtitle", "stream_order": 5, "language": "ja",
                          "reason": "codec hdmv_pgs_subtitle is a bitmap subtitle: its "
                                    "timestamps live inside binary segments"}],
            "failed": []}


def _build(assemble):
    import merge_video_repair as mvr
    logs = []
    saved = (mvr.drop_corrupt_candidate_tracks, mvr.assemble_or_log_the_decline,
             mvr.log_assembly, mvr.video.video, mvr._prefetch_same_content,
             mvr.measure_same_content, mvr.gate_delivered_silences, tools.logs, tools.dev,
             sys.stderr)
    built = []
    mvr.drop_corrupt_candidate_tracks = lambda *a, **k: []

    def fake_assemble(logged, plan, unverified, candidate_obj, *args, **kwargs):
        # what the real build loop is handed: `iterate_candidate_audios`, then every subtitle
        import merge_video_chimeric as mvc_
        built.extend(str(a["StreamOrder"]) for _, a in mvc_.iterate_candidate_audios(candidate_obj))
        built.extend(str(t["StreamOrder"]) for ts in candidate_obj.subtitles.values() for t in ts)
        return assemble()
    mvr.assemble_or_log_the_decline = fake_assemble
    mvr.log_assembly = lambda *a, **k: None
    mvr.video.video = _Repaired
    mvr._prefetch_same_content = lambda *a, **k: None
    mvr.measure_same_content = lambda *a, **k: (True, "synthetic_same_content")
    mvr.gate_delivered_silences = lambda *a, **k: []
    tools.logs, tools.dev = logs, True
    sys.stderr = open(os.devnull, "w")
    try:
        with tempfile.TemporaryDirectory() as work:
            try:
                repaired, assembly = mvr.build_repaired_video_object(
                    _Candidate(), _Master(), _plan(), work, "test")
                return repaired, assembly, logs, built, None
            except Exception as error:                               # noqa: BLE001
                return None, None, logs, built, error
    finally:
        sys.stderr.close()
        (mvr.drop_corrupt_candidate_tracks, mvr.assemble_or_log_the_decline,
         mvr.log_assembly, mvr.video.video, mvr._prefetch_same_content,
         mvr.measure_same_content, mvr.gate_delivered_silences, tools.logs, tools.dev,
         sys.stderr) = saved


def _keep_lines(logs):
    return [line for line in logs if line.startswith("repair: chimeric_keep ")
            or " repair: chimeric_keep " in line]


def test_every_track_built_the_gate_drops_every_audio_the_product_is_master_plus_added():
    repaired, assembly, logs, built, error = _build(_assembly)
    assert error is None, error
    assert set(built) == {"1", "2", "3", "4", "5"}, built           # every non-video track
    # the gate dropped every rebuilt audio: the master's intact tracks stay the product's audio
    assert [d["cause"] for d in assembly["fabricated_dropped"]] == ["intact_same_language_wins"] * 3
    delivered = [(holder, t["StreamOrder"]) for holder in ("audios", "subtitles")
                 for ts in getattr(repaired, holder).values() for t in ts
                 if t.get("keep", True) is not False]
    assert delivered == [("subtitles", "3")], delivered            # only the added stream
    lines = _keep_lines(logs)
    body = [line.split("repair: chimeric_keep ", 1)[1] for line in lines]
    assert len(body) == 5, lines                                    # the 'und' alias once
    assert body[0].startswith("track=1 lang=ja kept=no reason=intact_same_language_wins"
                              "(master_stream=1) kind=audio"), body[0]
    assert body[1].startswith("track=2 lang=en kept=no reason=intact_same_language_wins"
                              "(master_stream=2)"), body[1]
    assert body[2].startswith("track=3 lang=und kept=no reason=intact_same_language_wins"
                              "(master_stream=3)"), body[2]
    assert "plan=borrowed_offset(inherited(stream_1))" in body[2], body[2]
    assert body[3].startswith("track=4 lang=en kept=yes reason=retimed(cues_kept=300,"
                              "cues_dropped=2) kind=subtitle"), body[3]
    assert body[4].startswith("track=5 lang=ja kept=no reason=bitmap_subtitle_not_retimable"), body[4]
    assert any("repair: chimeric_keep_summary tracks=5 kept=1 dropped=4" in l for l in logs), logs


def test_a_refused_build_names_every_track_not_kept():
    import merge_video_chimeric as mvc_

    def refuse():
        raise mvc_.chimeric_error("synthetic", cause="repair_budget_exceeded")
    _, _, logs, built, error = _build(refuse)
    assert getattr(error, "cause", None) == "repair_budget_exceeded", error
    body = [line.split("repair: chimeric_keep ", 1)[1] for line in _keep_lines(logs)]
    assert len(body) == 5 and all("kept=no reason=repair_refused(repair_budget_exceeded)" in b
                                  for b in body), body


def test_track_dedup_builds_the_aliased_track_once():
    """OWNER 2026-09-28 (CASE_plan_all_tracks_20260928.md, "Found, not fixed"): the frozen
    `video.py` (line ~163) aliases the 'und' audio list under the default language when that
    language is itself absent -- ONE dict under TWO keys. `_Candidate` already carries this
    shape (`self.audios["fr"] = self.audios["und"]`, stream 3). Before the dedup, `built` (a
    LIST, unlike the set the older assertion collapses duplicates into) held stream 3 twice --
    the build loop built, and delivered, the same track under two language keys. Deduped by
    StreamOrder it is built once, under the FIRST language it is listed under ('und'), with one
    `repair: track_dedup` line naming both languages."""
    import merge_video_chimeric as mvc_
    pairs = [(lang, a["StreamOrder"]) for lang, a in mvc_.iterate_candidate_audios(_Candidate())]
    assert pairs == [("ja", "1"), ("en", "2"), ("und", "3")], pairs   # stream 3 once, under 'und'

    repaired, assembly, logs, built, error = _build(_assembly)
    assert error is None, error
    # one build per candidate StreamOrder (3 audio + 2 subtitle) -- not six
    assert built == ["1", "2", "3", "4", "5"], built
    assert any("repair: track_dedup stream=3 langs=und,fr" in line for line in logs), logs


def test_the_keep_lines_are_dev_only():
    import merge_video_repair as mvr
    saved, tools.dev = tools.dev, False
    real_logs, tools.logs = tools.logs, []
    try:
        assert mvr.log_chimeric_keep(_Candidate(), _plan(), _assembly(), _Repaired()) == []
        assert tools.logs == []
    finally:
        tools.dev, tools.logs = saved, real_logs


if __name__ == "__main__":
    tools.dev = False
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
