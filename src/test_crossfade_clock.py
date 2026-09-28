'''Tests for the crossfade's clock -- run: python3 src/test_crossfade_clock.py

A splice crossfaded by `acrossfade` (ADDENDUM 23.3) hands the encoder timestamps that wander
from its sample count: MEASURED 2026-09-28 on Fallout S01E03 (candidate E-AC-3 through the 1001/1000
asetrate chain, tail filled from the BD master's TrueHD), the delivered track read -1 ms at 30 s
and +24 ms at 3222 s against the master -- 8 ppm, refused `alignment_contradicts_plan` by the
delivery gate -- while the same pieces joined by `concat` read -1.1 ms flat. The fold now
re-states each crossfade's clock from its samples (`asetpts=N/SR/TB`).

Synthetic media (no /srv): a sine candidate in E-AC-3, a sine master in TrueHD (40-sample
frames on a 1 ms container clock -- what makes the drift), the graph from
`build_audio_filtergraph` itself, the product's packet times against its packet count.'''

import os
import subprocess
import sys
import tempfile
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools  # noqa: E402

if not getattr(tools, "software", None):
    tools.software = {"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}
tools.logs = getattr(tools, "logs", [])

import merge_video_chimeric as mvc  # noqa: E402

CHAIN = "aresample=384000:resampler=soxr,asetrate=383616,aresample=48000:resampler=soxr"
FRAME_S = 1536 / 48000.0            # one E-AC-3 packet


def _pieces():
    return [{"source": "master", "master_start_ms": Decimal(0), "master_end_ms": Decimal(1000),
             "source_start_ms": Decimal(0)},
            {"source": "candidate", "master_start_ms": Decimal(1000),
             "master_end_ms": Decimal(301000), "source_start_ms": Decimal("5385.996")},
            {"source": "master", "master_start_ms": Decimal(301000),
             "master_end_ms": Decimal(309000), "source_start_ms": Decimal(301000)}]


def _splices():
    return {2: {"fade_left": True, "fade_right": False, "gain": None}}


def test_every_crossfade_restates_its_clock():
    graph, _, _ = mvc.build_audio_filtergraph(_pieces(), 0, 0, 48000, "stereo", CHAIN,
                                              Decimal(0), Decimal(0), _splices())
    folds = [c for c in graph.split(";") if "acrossfade" in c]
    assert folds and all(",asetpts=N/SR/TB[" in c for c in folds), graph
    plain, _, _ = mvc.build_audio_filtergraph(_pieces(), 0, 0, 48000, "stereo", CHAIN,
                                              Decimal(0), Decimal(0), None)
    assert "acrossfade" not in plain and "asetpts=N/SR/TB" not in plain, plain


def test_crossfaded_product_keeps_its_sample_clock():
    graph, _, _ = mvc.build_audio_filtergraph(_pieces(), 0, 0, 48000, "stereo", CHAIN,
                                              Decimal(0), Decimal(0), _splices())
    with tempfile.TemporaryDirectory() as tmp:
        cand, master, out = (os.path.join(tmp, n) for n in ("c.mka", "m.mka", "o.mka"))
        run = lambda *a: subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin"] + list(a),
                                        check=True)
        run("-f", "lavfi", "-i", "sine=f=440:r=48000:d=320", "-ac", "2", "-c:a", "eac3", cand)
        run("-f", "lavfi", "-i", "sine=f=660:r=48000:d=320", "-ac", "2", "-c:a", "truehd",
            "-strict", "-2", master)
        run("-i", cand, "-i", master, "-filter_complex", graph, "-map", "[aout]",
            "-c:a", "eac3", out)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "packet=pts_time",
                                "-of", "csv=p=0", out], capture_output=True, text=True,
                               check=True)
        times = [float(l.strip(", ")) for l in probe.stdout.splitlines() if l.strip(", ")]
        deviation = [t - i * FRAME_S for i, t in enumerate(times)]
        spread_ms = (max(deviation) - min(deviation)) * 1000.0
        # the container clock is 1 ms: a product on its sample clock wanders by at most that
        assert len(times) > 9000 and spread_ms <= 1.0 + 1e-6, (len(times), spread_ms)


if __name__ == "__main__":
    failures = 0
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
            print(f"ok   {name}")
        except Exception as error:                                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"{len(names) - failures}/{len(names)} passed")
    sys.exit(1 if failures else 0)
