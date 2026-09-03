# 006 — Locate the change points inside a staircase

**Agent:** `vmsam-dev-1` (measurement). **Pairs with 007.** **State:** open.
**Campaign objective 2, the measurement half.**

## The signal

A window sequence reading

```
(0.92, -1, 125)  (0.93, 0, 0)  (0.90, -1, 125)
```

is **one content with a small insertion or deletion**: correlation stays high
throughout while the offset steps by whole points. One point = **125 ms**, the
chromaprint frame — every delay in this corpus is a multiple of it.

That is the shape. `first_delay_test` sees the disagreement across windows and
refuses the pair, which is correct: no single constant offset exists. It records
what it saw in `video_obj.delayFirstMethodAbort[best_video.filePath]` →
`[1|2, delays]`, filled at 12 points. **You do not need to re-run the correlation
that just failed** (`SPEC_ZONE_A.MD` §1).

## Goal

Build the method that turns that staircase into **change points**: where, in
master-timeline seconds, does the offset step, and by how much. Windows are 12,
their length is twice their spacing so consecutive ones overlap by half — a step
is bracketed to roughly one spacing by the window data alone. Getting from there
to a frame is the part that needs an instrument, 005's.

## Target-first — binding, `AGENT.MD` and `CAMPAIGN.MD` both

1. **Build one correct answer by hand first**, on one real error file, with
   whatever tools you like. The located change point, in seconds, and how you
   know. This exists **before** the code.
2. **State what result would refute you, before the run.** A prediction written
   afterwards is not a prediction.
3. Only then write code, and check it reproduces the hand answer.

## The rule that decides whether this counts

**Your work counts when dev-2 calls it, not when it passes a test.** Five pieces
of campaign-1 code were written, reviewed and unit-tested green and never called
by anything. Agree the interface with dev-2 **before** you build behind it, and
pair this with 007 before either is declared done.

## Ask the Lead for

- a tagged call site, if you need one — one agent at a time, one edit
- a container run, through `vmsam-ci` — never directly

## What changed / measured / dropped / blocked

**State: calibration passed on constructed ground truth. Not yet run on a real
error file — that is what closes this task, and it is pending an exemplar.**

### Interface — agreed with dev-2 before building, both directions confirmed

`VMSAM_HELP_AI/dev-2/INTERFACE_dev1_dev2.md` v1 plus my four corrections. dev-2
has confirmed the signature, the `None` contract and `quantum_ms`:

```python
# src/change_point_locator.py  -- one caller, and it is dev-2's zone-A path
def locate_change_points(best_video, candidate_video, language): ...
```

Returns a partition of the master's timeline under
`candidate_time_ms = master_time_ms + candidate_offset_ms`, or **`None`** meaning
*I could not measure* — never *the files are compatible*. dev-2 maps `None` to its
`no_plan` outcome and the refusal stands untouched. It reports five outcomes
(`disabled / no_plan / declined / repaired / failed`), so the third answer is not
lost at its end either.

### Measured — the sign, which was the one that could have destroyed the output

Constructed pair: one real audio track from `/config/example/ore1`, and a copy with
**exactly 1000.000 ms removed at exactly 700.000 s** (sample-exact via `atrim` +
`concat`, verified by byte count: 88200 bytes = 44100 samples). Instrument:
`audioCorrelation.correlate` **imported from `src/`**, not reimplemented.

Ten windows at the pipeline's own geometry (`number_cut=10`, spacing 107 s,
window 214 s, first at 120 s):

| h | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | **0.8724** | **0.8966** | 0.9859 | 0.9865 | 0.9861 | 0.9866 |
| delay_ms | 0 | 0 | 0 | 0 | 0 | **-1000** | -1000 | -1000 | -1000 | -1000 |

**Negative control** (master against itself): all ten windows 1.0000 / 0 ms.

Every prediction in `VMSAM_HELP_AI/dev-1/lab/M2-change-point-prediction.md` held
and **no refutation fired**. The one that mattered was R1, the sign: the post-cut
plateau is **negative**, matching the derivation from the source. dev-2 confirmed
the same convention independently by numpy cross-correlation from the opposite
direction, having caught its own sign error with a control first.

The straddling windows behave as the task describes: correlation stays high while
the offset steps by whole points. The bracket from the window data alone is the
disagreement region **[655, 762]** — one spacing, 107 s — and it contains the true
700, so `BRIEF_COMMON.md`'s one-spacing bracket claim is confirmed rather than
assumed.

### Measured — a constraint on refinement that was not predicted

**The same physical 1000 ms cut reads -1000 ms at a 214 s window and -1080 ms at a
30 s window.** The quantum is 125 ms at L=214 and ~135 ms at L=30, and 8 x 135 =
1080. So: **the offset must be read at the pipeline's own window length. Short
windows may locate a boundary; they may never re-measure a step.** This is the
`AGENT_NOTES.MD` quantum finding appearing as a wrong number in a real measurement.

### Measured — chromaprint is blind to the tail of every window

`fpcalc` emits ~20.6 fewer items than `W/0.12384`, and the shortfall is at the
**end**: ~2.55 s of every window is fingerprinted weakly or not at all. Measured
consequence: a probe window whose *tail* is contaminated stays at correlation
1.00000 until 500-800 ms of wrong content has accumulated.

So boundary probes must be **head-anchored**. The gap start needs the audio
**reversed** so that its edge becomes a head:

| edge | probe | measured | truth | error |
| :--- | :--- | ---: | ---: | ---: |
| gap END | head-anchored, forward | 700.886 | 701.000 | **-115 ms** |
| gap START | tail-anchored, forward | 700.526 | 700.000 | +526 ms — **rejected** |
| gap START | head-anchored, **reversed** | 700.046 | 700.000 | **+46 ms** |

Both accepted errors are within one fingerprint item (124 ms). **Both biases point
inward**, so the recovered gap is systematically narrow — 840 ms against a true
1000. dev-2 has been told: a narrow gap makes the splice take more from the
candidate than it should, so the gap wants widening by one item per side, or
pinning by 005's video instrument.

### Dropped

- **Tail-anchored boundary probes.** Blind for ~2.5 s, for the measured reason
  above. Replaced by reversed head-anchored probes.
- **Refining the step with a short window.** It changes the quantum and therefore
  the answer.

### Blocked

- **A real error file.** This task closes on one, not on constructed ground truth,
  and I will not report the calibration as more than it is. Asked
  `vmsam-forensic` for an exemplar and `vmsam-ci` whether a given exemplar is
  misfiled content first — its error id 237 is a **different series** from its
  folder master, with 26 of that folder's 49 entries named the same way, which is
  a refusal that is correct and must not be "repaired". I offered to take 237 as a
  **negative control** instead: a pair my locator must return `None` for.

### Measured — three window signatures, and why fidelity must not be the refusal rule

Ten windows, pipeline geometry, `correlate` imported from `src/`. Details and
per-window numbers in `VMSAM_HELP_AI/dev-1/lab/M3-three-signatures.md`.

| pair | distinct delays | sign flips | fidelity |
| :--- | ---: | ---: | :--- |
| `ore1`, real different-source pair (BDRip vs WEB-DL) | **1** (constant +1000 ms) | 0 | 0.915-0.945 |
| `ore1`, constructed 1000 ms deletion | **2** (0 then -1000) | 0 | 0.872-1.000 |
| **error id 237**, hard negative | **10 of 10** | **8** | **0.556-0.578** |

Error id 237 is a real refused pair that `vmsam-ci` established, by extracting
frames and looking at them, to be a **different television series** from its
folder master. It is therefore a pair my locator must return `None` for — a
confident answer on it is wrong by construction. ci predicted the shape before I
ran it (delays spread ~421 s with sign flips, fidelity 0.56-0.58); I measured
0.5562-0.5776 and 421.1 s from a different code path, with six of its ten delay
values reproduced exactly.

**Two of my three "real" `ore1` pairs were not independent tests**, and I am
recording that rather than counting them: `md5sum` shows two releases carry the
byte-identical jpn track, and a third correlates at 1.0000/0 ms on every window.
Only the BDRip is a genuinely different source.

**The refusal rule cannot be a fidelity threshold.** `docs/AUDIO_SPEED_POLICY.MD`
measured uncorrected PAL-against-film pairs at **0.558-0.562** — the same floor as
unrelated content, and for the same geometric reason (across a 214 s window a
4.27 % rate difference drifts the signals ~9 s apart, so no single shift aligns
them). A rule *"fidelity < 0.7 -> None"* would refuse precisely the family
`TASKS/008` exists to repair. What separates them is the **pattern** of the delays:
a speed relation gives low-fidelity delays that are **monotone and grow along the
file**; no-shared-content gives low-fidelity delays that are **scattered and flip
sign**. That discriminator is inferred from two measured signatures plus one
frozen document's numbers, and is **not yet established** on a real speed pair.

### A duration fact worth carrying

On error id 237 the container durations (1438.04 / 1501.25 s) differ by 63 s, but
the master's container duration is set by its **video**. Per stream, both files'
**audio** is 1436.977 / 1438.040 s — **1.06 s apart**. `get_shortest_audio_durations`
drives the window geometry from audio, so the geometry is 108 s spacing / 216 s
window. The 63 s is a video-vs-audio artefact of that master, not a content delta.
This does not touch ci's different-series conclusion, which was established by eye.

### Not yet done

`src/change_point_locator.py` is not written. The method is calibrated and the
interface agreed; the module follows, and it is deliberately not written before a
real exemplar has shown the method survives real content.
