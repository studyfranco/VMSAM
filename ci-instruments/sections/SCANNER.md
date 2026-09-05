# The whole-file offset scanner — `scan_hidden.py`

**Author: `vmsam-ci`.** Written for `reports/ALGORITHMS_INDEX.md`. All excerpts are
ids only, passed through `redact.py`, which **refuses to emit** if a path, a
catalogue id, a media extension or an `SxxExx` survives.

---

## 1. What it does

Given a refused pair — a candidate file and the master it failed to merge with — it
produces an **offset profile across the whole file**: a sequence of `(time, lag,
correlation)` readings, and a label naming the *shape* of that sequence.

It exists to answer one question: **how often does a refused file carry structure the
pipeline's stage-1 record could not see?** Not to repair anything, and not to decide
repairability.

## 2. The method, in enough detail to reimplement

**Stage A — baseline.** Four probes of **120 s** at 10 / 25 / 50 / 75 % of the pair's
shorter audio duration, each cross-correlated over a **±60 s** lag search. Keep the
strongest; stop early once one clears **0.40**. If the best is below **0.25**, emit
`baseline-not-found` and **probe no further**.

**Stage B — profile.** **30 probes of 20 s**, evenly spaced from `t = 2 s` to
`duration − probe − 2 s`. **No privileged regions.** Each probe:

1. decode master audio at `t`, and candidate audio at `t − baseline/1000`, both to
   mono PCM at **8000 Hz**;
2. mean-subtract, cross-correlate by FFT, normalise by the product of the norms;
3. take the argmax within `baseline ± fine`, where `fine` is sized per file.

Lag resolution is `1/8000 s = 0.125 ms`.

**Reading acceptance.** A probe counts only if its correlation clears **both** a floor
of 0.25 **and half that file's own median correlation**.

**Labelling.** Deviant probes are those more than `max(5 ms, 6 × MAD)` from the
sequence median, where MAD is the file's own within-plateau scatter. Consecutive
deviant probes group into runs, and the *shape* names the label:

| runs | touches an end | label |
| :--- | :--- | :--- |
| 0 | — | `flat` |
| 1 | no | **`island`** — departs **and returns** |
| 1 | yes | `step` |
| ≥2 | — | `scatter` |

Depth below half a quantum appends `-subquantum`. Coverage below 70 % of probes or of
span gives **`partial-coverage`**.

## 3. Why this method rather than the obvious alternative

**The obvious alternative is to reuse the pipeline's own chromaprint correlation.** It
is faster, already written, and already trusted by the code under test. It is the
wrong instrument here for one reason:

> **The thing under test is what the pipeline's geometry cannot see. An instrument
> sharing that geometry cannot see it either, by construction.**

Two properties follow from decoding PCM instead:

- **No quantisation floor.** The pipeline's quantum is computed *per call* as
  `int(lengthFile / len(fingerprint) * 1000)` and ranges **124–142 ms**. At 0.125 ms
  resolution, half a quantum is ~496 samples of lag to this scanner — a step of 24.7 ms
  is a first-class measurement, not a rounding artefact.
- **No shared blind spot.** The pipeline starts its first window at 120 s. This scanner
  starts at 2 s.

**The rule, and it was violated in this very file before it was obeyed:**

> **An independent instrument aimed by a dependent one is not independent.**

The per-file window is sized from *the record's* step span — so on a file where the
record understates the true displacement, the window inherits exactly the record's
blindness. Fixed by **escalating on the observed symptom** (poor coverage), which owes
the record nothing.

`vmsam-dev-2`'s stronger form, reached independently and worth more than mine:
**independence of instruments is not independence of assumptions.** Four instruments —
a pHash probe, an envelope method, a stretch search and a slope fit — all reported a
rate that was really a step.

## 4. What it cannot do

- **It cannot see below its own deviance threshold.** That was a fixed 25 ms; it is now
  `max(5 ms, 6 × MAD)`. **Still a constant chosen by its author, not physics.**
- **It bounds nothing about the video modality.** Structure perturbing no audio at all
  — frames removed with audio intact, an edit inside a silent passage, a video-only
  splice — is invisible to it and is `frame_compare.py`'s question.
- **It emits one label per file.** A file carrying both a visible step and an invisible
  one is labelled by the visible one. See §6.
- **It measures offsets. It does not decide repairability.**

## 5. A log excerpt showing it WORKING

Error id 352, jpn pair. The record for that pair holds exactly two delays —
`{32875, 33875}`, one 1000 ms transition, nothing else. The profile:

```
t=   2.0  lag= -33521.5  corr=0.952
t=  50.8  lag= -33521.4  corr=0.946
t=  99.7  lag= -33522.4  corr=0.885
t= 148.5  lag= -33520.5  corr=0.901
t= 197.3  lag= -33521.8  corr=0.876     <- plateau 1, spread 1.9 ms
t= 246.1  lag= -32529.8  corr=0.988
t= 295.0  lag= -32529.6  corr=0.986
...
t= 734.4  lag= -32528.8  corr=0.944     <- plateau 2, spread 2.1 ms
t= 783.3  lag= -32553.2  corr=0.985
t= 832.1  lag= -32554.4  corr=0.977
...
t=1369.2  lag= -32553.9  corr=0.924     <- plateau 3, spread 1.4 ms
```

**Three plateaus.** The 992.5 ms step at ~220 s is the one the record shows. **The
24.75 ms step at ~758 s is not in the record at any position**, being far below half a
quantum. Re-measured at a geometry sharing nothing with the scan — 120 s probes,
±40 s absolute search, no baseline subtraction — it reads **−24.75 ms**, and a peer's
fingerprint instrument in index space agrees with the absolute offset to **57 ms,
under half a fingerprint item**.

## 6. A log excerpt showing it DECLINING — and why this matters as much

Error id 266, the ~30 s boundary, probing the head:

```
p1 t=  1.5s dur=27s  lag  990.4 ms  corr 0.0475   <- below floor, NOT a measurement
p1 t=  2.0s dur=26s  lag  991.0 ms  corr 0.0481   <- below floor, NOT a measurement
p1 t=  2.5s dur=26s  lag  989.6 ms  corr 0.0472   <- below floor, NOT a measurement
p2 t= 32.0s dur=26s  lag  991.6 ms  corr 0.8849
p2 t= 40.0s dur=26s  lag  991.2 ms  corr 0.9334
p2 t=1000.0s          corr 0.9420

CANNOT SETTLE IT: p1 n=0, p2 n=4 -- instrument did not run on a plateau
```

**Identical probe geometry, correlations of 0.047 against 0.94 — a factor of twenty.**
A peer's locator reports a −168.32 ms step at that boundary. This scanner **can neither
confirm nor refute it**, and says so.

> **`partial-coverage`, `unreadable`, `baseline-not-found` and `instrument-failed` are
> a THIRD OUTCOME. They mean *I could not measure*, never *there is nothing here*.**
> A file readable over 40 % of its length is not a file in which nothing was found, and
> such files must never enter another agent's "clean" sampling frame.

## 7. What the scanner has been used to measure

**A bound on the miss rate.** 98 files, seeded random sample of the refused corpus,
manifest written **before the first probe**. Uniform 30-probe instrument. Estimand and
a 10 % refutation condition frozen in writing beforehand and scored without editing.

```
instrument RAN on 88, DID NOT RUN on 10
MISSES 2 of 88 = 2.3 %   Wilson 95 % [1.3 %, 7.2 %], fpc 0.810
frozen refutation threshold 10 %  ->  NOT REFUTED
```

**Two of three frozen predictions were wrong and are recorded as wrong.**

**The sentence that must travel with the number:**

> **2.3 % is the rate of FILES the record was wholly blind to. It is NOT the rate of
> STRUCTURE the record missed.** id 352 is the worked instance: labelled `scatter`,
> record `within-step`, scored as *agreement* — while carrying a step the record cannot
> see. **The second number is larger and this sample does not bound it.**

## 8. Three defects in this scanner, and how each was found

| defect | consequence | found by |
| :--- | :--- | :--- |
| fine window ±3000 ms around **zero** | 64 of 100 files carry offsets beyond it; correlation **saturates at the window edge and returns the same wrong value every probe**, so a constant sequence read as `flat`. That run reported **28 % flat**; corrected, **zero eligible-flat** | a peer, from the garbage output |
| a **single** baseline probe | one file's baseline landed at 0.14 correlation, and the file was labelled `partial-coverage` — an **instrument** property wearing a **media** label | inspecting a positive instead of counting it |
| fine window fixed at **±3 s** while the baseline search was ±60 s | 37 of 100 have a step span exceeding it; six carried depths **understated by 2000–3100 ms** | another agent's **positive controls** |

**All three sat in the same file and were found one at a time. Not one was found by
inspecting my own output.**

> **A broken instrument does not produce noise. It produces false cleanliness.** Noise
> is discarded in a minute; saturation yields stable, repeatable, confident agreement
> that no summary statistic separates from a clean file — and *clean* is the set
> everyone else samples **from**. That defect damaged another agent's plan, silently,
> through a list.

**And a fourth, about method rather than code.** A sensitivity sweep over eleven chosen
constants showed almost nothing moving, which read as robustness. Run on the 62 rows
**already known to be wrong**, it called them just as stable.

> **A sensitivity analysis re-judges measurements; it cannot re-measure.** Every
> sequence it reclassifies was produced through the same defective window.

The split this forces, and it is the spine of the defect catalogue:

- **A threshold decides how you judge what you measured.** Sweeps test these.
- **A span decides whether you measured at all.** *Nothing internal tests these.*

All three defects above were spans. All eleven swept constants were thresholds, and all
eleven were harmless.
