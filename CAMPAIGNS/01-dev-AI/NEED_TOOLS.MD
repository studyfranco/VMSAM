# VMSAM — Missing Tooling Inventory

Utilities and libraries the merge/alignment work needs that are **not present in
the runtime image**. The `Dockerfile` is read-only per `AGENT.MD`, so nothing
here can be added by the agent — each entry is a request for the lead developer.

Every entry states the concrete failure case it unblocks and why the tools
already in the image are insufficient. Entries are ordered by the number of
recorded failures they would help resolve.

**Do not add speculative entries.** If `ffmpeg`, `ffprobe`, `mediainfo`,
`fpcalc`, `mkvmerge`/`mkvextract`, `numpy`, `scipy` or `resampy` can do the job,
it does not belong here.

---

## What the image already provides

Verified present, 2026-08-28:

| Category | Available |
| :--- | :--- |
| Binaries | `ffmpeg`, `ffprobe`, `mediainfo`, `fpcalc`, `mkvmerge`, `mkvextract` |
| Python | `numpy` 2.4.6, `scipy` 1.17.1, `resampy` 0.4.3, `matplotlib` 3.10.9 |
| Rust | full toolchain; the project's own `audio_sync` binary |
| ffmpeg filters | `atempo`, `asetrate`, `aresample`, **`rubberband`**, `astats`, `silencedetect`, `scdet` |

> **`rubberband` is available as an ffmpeg filter** even though the standalone
> `rubberband` CLI is not installed. High-quality time-stretch and pitch-shift
> are therefore already reachable through `ffmpeg -af rubberband=…`. Likewise
> `sox` is absent, but every `sox` operation this project needs has an ffmpeg
> equivalent. **Neither is requested.**

### Explicitly NOT needed

**PAL speedup correction (13 recorded failures) requires no new tooling.**
The 25 → 23.976 fps case is a pure linear rescale, measured at a duration ratio
of 0.9589 against a theoretical 0.9590. `asetrate` + `aresample` (exact,
resamples pitch along with speed, which is what a PAL telecine actually did) or
`atempo` (tempo only, preserves pitch) both handle it with what is installed.
`resampy` covers the same ground on the Python side.

---

## 1. A dynamic time warping implementation

| | |
| :--- | :--- |
| **Package** | `pip install dtw-python` (or `fastdtw` for the linear-memory approximation) |
| **Unblocks** | 238 failures — 180 structural cut differences (> 5 s) and 58 near-miss trims (1–5 s) |
| **Why existing tools fall short** | The whole pipeline assumes **one constant offset** per pair. `audioCorrelation.correlate` and the Rust `audio_sync` FFT pass both return a single scalar delay, and `compare_video` rejects the pair when the per-segment delays disagree. That is correct behaviour for a shifted timeline and structurally unable to describe an *edited* one, where the offset changes at each insertion or deletion. Recovering a piecewise-constant offset function needs a monotonic alignment path, which is what DTW produces and cross-correlation cannot. |
| **Note** | Implementable directly on `numpy`/`scipy` — a banded DTW over per-segment correlation scores is a few hundred lines. Requesting the library is a convenience and a correctness hedge, not a hard blocker. **Lower priority than it looks.** |

## 2. `librosa` + `soundfile`

| | |
| :--- | :--- |
| **Package** | `pip install librosa soundfile` (`soundfile` needs `libsndfile1` via apt) |
| **Unblocks** | Diagnosis across all 309 failures; segmentation quality for the cut-difference work |
| **Why existing tools fall short** | `fpcalc` emits a Chromaprint fingerprint tuned for whole-track identification, not for locating *where* two timelines diverge. Onset envelopes, chroma/CQT features and beat tracking give a frame-level similarity curve, which is the natural input to the alignment path in entry 1 and to detecting the exact boundary of an inserted recap. Reimplementing onset detection and CQT on bare `scipy` is possible but is a research project rather than a feature. |
| **Note** | `soundfile` only matters if audio is read directly; the pipeline currently pipes PCM through ffmpeg, which already works. `librosa` is the substantive half of this request. |

## 3. `libchromaprint-tools` beyond `fpcalc` — *not required*

Recorded here to close the question: `fpcalc` is present and is the only
Chromaprint entry point the pipeline uses (`audioCorrelation` shells out to it).
No additional Chromaprint tooling is needed.

---

## Open questions that may generate future entries

These are under investigation and have **not** yet produced a tooling request.
Listed so the next session does not re-derive them.

1. **VFR / non-linear drift.** No recorded failure has yet been *proven* to be
   variable-frame-rate drift as opposed to a structural edit. `ffprobe
   -show_frames` can measure per-frame timestamp deltas to settle it; that is an
   analysis task with existing tools, not a missing utility.

2. **Container frame rate is unreliable.** Among the 35 frame-rate mismatches,
   the 30 fps and 29.97 fps groups show duration ratios (0.9591, 1.0025) that
   contradict their declared rates, and the 24.00 fps group is really a cut
   difference wearing a misleading header. Any fps-based logic must derive the
   rate from timestamps rather than trust `avg_frame_rate`. Again an analysis
   problem, not a tooling one.

3. **Chimeric track assembly** (patching a candidate's missing segments from the
   reference track) needs no new tooling to *build* — `ffmpeg` concat plus
   `mkvmerge` cover the assembly. It needs entry 1 to know *where* to cut.
