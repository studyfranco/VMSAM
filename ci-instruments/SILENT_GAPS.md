# Repaired tracks that ship carry silence where the master cannot fill them

**PRIVATE.** Measured on the delivered artefact for error id 150, build `94cac4a`.
Ids only.

## The measurement

The fabricated `eng` track in the delivered file, sampled every 60 s and compared
against the candidate's own `eng` mapped through the local offset:

```
      t  output  cand(mapped)  verdict
     60   627.8        632.3   ok
    120  1811.4       1828.2   ok
    ...
    360     0.0        802.8   LOST -- source had audio
    ...
    900     0.0        983.4   LOST
    960     0.0       1171.5   LOST
    ...
   1320     0.0       1295.8   LOST
   1380     0.0       1775.7   LOST

sampled 23 points at 60 s spacing
LOST (output silent, source had audio):  5  = 21.7 % of the track
source silent anyway:                    0
```

**Zero points where the source was silent too.** Where the output is silent, the
candidate had audio every time. One gap was bounded directly: continuous silence from
before 320 s to between **410 and 420 s**, and it sits **exactly at the change point**
between the −10011 ms and −21815 ms plateaus.

The non-silent points match closely (627.8 vs 632.3, 1811.4 vs 1828.2), which is what
makes the silent ones trustworthy: **the offset mapping is right where it can be
checked.**

## The structural result, which follows from two measured facts

1. **`keep_best_audio` drops a repaired track whose language the master also carries.**
   Measured: id 150 delivered five fabricated tracks, all candidate-only languages;
   `fre` and `jpn` in the artefact are the master's at r = 1.0000.
2. **Master fill requires a master track in that language.** `vmsam-dev-2` reports the
   fill piece is cut from the master's same-language stream.

> **Therefore every repaired track that survives to a delivered artefact is, by
> construction, one the master CANNOT fill — so its gaps are silence.**

The two conditions are complementary: the same property that lets a repaired track
survive the merge is the property that denies it any fill material. **A track that
could be filled is dropped; a track that ships cannot be filled.**

## What this does and does not say

**It does not say the repair is wrong.** Inserting silence rather than possibly
misaligned audio is a defensible choice, and the campaign has a commit describing
exactly that — *an unverified offset becomes a gap*. The gap here sits at a change
point, which is where verification is hardest.

**It does say the magnitude is large and nobody has stated it.** 21.7 % of a delivered
track. dev-2's master-fill median of ~200 s per repaired track (~14 %) is measured on
the **intermediate**, where a same-language master track still exists. For the tracks
that actually ship, that fraction is **not master audio — it is silence.**

## Limits

- **One file, one track, 23 sample points at 60 s spacing.** The 21.7 % is a coarse
  fraction, not a duration: gaps shorter than the spacing are missed and the boundaries
  of the counted ones are known only to ±60 s except the one I bisected.
- The offset mapping used three measured plateau values. It is validated by the
  matching non-silent RMS, but a mis-mapped point would read as *ok*, not as *lost* —
  so this **understates** rather than overstates.
- I have not established that all five gaps are at change points. One is.
- **Nothing here is about audio quality.** Silence is measurable; degradation is not,
  by this method.

---

## Corpus magnitude, and an independent check of it

`vmsam-dev-2` confirmed the mechanism in code — `find_master_audio_for_language`
returns `None` when the master lacks the language, and `build_one_audio_track` then
sets fill = silence, **with no fallback to another language, deliberately**, because
filling a French gap with Japanese passes every structural check and is indefensible
on listening.

Their sweep over 109 repaired tracks:

| | |
| :--- | ---: |
| tracks with any silence fill | **14** |
| median silence fraction | **12.9 %** |
| largest | **24.5 %** (id 695, 350.1 s of 1430 s) |

> **On all 14, `silence_ms` equals `gap_ms` EXACTLY.** Where the master lacks the
> language, *all* the fill is silence; where it has it, *none* is. No intermediate
> case in 109 tracks — which is the structural result confirmed by their accounting
> rather than by my sampling.

### The cross-check

id 695 is the corpus worst case **and** the file I already ran a container control on,
so its delivered artefact was on disk. Sampling it directly at 30 s spacing:

```
47 points on the delivered fabricated track
silent: 11  = 23.4 %
dev-2's assembler accounting: 24.5 %
difference: 1.1 percentage points
```

**An accounting method and a sampling method that share no code, no stage and no
input path, agreeing to one point.** This is the first of dev-2's figures confirmed
from outside its own stage.

**The two numbers are not the same kind and must not be averaged:** theirs is exact
from the assembler's own bookkeeping; mine is a coarse fraction from 30 s sampling
that understates, since a mis-mapped point reads as *ok* rather than *lost*.

## A third inert control, found by dev-2 in its own zone

```
assemble_on_master_timeline(..., max_silence_fraction=None, ...)
    if max_silence_fraction != None:
        if silence_fraction > max_silence_fraction:  -> DECLINE
```

**The repair never passes a value.** The guard has never fired and cannot. A quarter of
a track can be silence and nothing objects. That is the third inert control in that
zone today — the `fabricated` rule reading a key nothing sets, this silence budget, and
the verifier's reference read that could not see the head.

**dev-2 is not inventing a threshold now**, having refused one this morning for want of
a distribution — and it now has the distribution, so the refusal no longer holds and
the number goes to the owner with data in front of him.

---

## The ceiling on both numbers, and it points the wrong way for reassurance

`vmsam-dev-sandbox` asked whether *"a defect's best test cases are systematically absent
from any sample collected while the defect was live"* reaches the silence distribution.
**It does.**

The 14 tracks come from **109 repaired tracks measured on the intermediate.** A file
whose repair *declined* produces no track, contributes no silence measurement, and
cannot enter the distribution. Release 32 declined.

> **Median 12.9 % and max 24.5 % are a median and a maximum OVER SURVIVORS.**

**The DIRECTION of the bias is WITHDRAWN.** What follows was my argument; the
premise it rests on was never checked, and `vmsam-dev-2` broke it. Kept rather than
deleted, because a retracted argument with its refutation beside it is worth more than
a silently corrected one.

~~**And the bias points the unfavourable way**~~ — stated as a mechanism, not a measurement.
A track needs a verified offset per segment, and *an unverified offset becomes a gap*.
A file with many unverifiable segments therefore accumulates **both** more silence
**and** more chances to trip whatever makes a repair decline. If declining correlates
with gappiness at all, **the gappiest files are systematically absent and 24.5 % is a
floor on the true maximum, not a ceiling.**

**THE REFUTATION.** The chain above requires that a gap can *cause* a verification
failure. dev-2's counter: **the gaps may be irrelevant to probes that only land inside
candidate pieces.** If probe positions are chosen inside candidate content — or a gap
position is skipped the way a position earlier than the master's start is already
recorded as `no_reference` — then a gap is invisible to the verifier, gappiness does
not raise the decline rate, **and my mechanism does not run at all.**

I built a directional argument on a property of another agent's verifier that I had not
read and could not see. **I inferred a mechanism from a distribution rather than reading
the code or asking** — the distribution was consistent with my mechanism *and* with
dev-2's, and consistency was all I checked. That is the same error I spent the day
warning about in my own instrument: every defect in the scanner was found by an external
answer, never by inspecting output.

**What survives is the qualifier, not the direction:** *median 12.9 %, max 24.5 %, over
tracks that were produced.* dev-2's finding, independent of which mechanism is true —
the harness records fill columns only on a repaired row, so a declining file builds its
tracks and throws the counts away with the exception.

**And the disagreement is being made moot rather than adjudicated.** dev-2 changed the
decline path to carry the reports it built rather than only the probes it measured —
*a decline was reporting what it MEASURED and not what it PRODUCED* — so re-running the
declines yields their fill fractions and the distribution can be computed over
everything **built** rather than everything **passed**. **A measurement replacing two
arguments about its shape.**

**My own 23.4 % carries that ceiling and a second one.** It is measured on a
**delivered** artefact, so it is a survivor twice over — the repair produced it *and*
`keep_best_audio` did not drop it — and the 30 s sampling understates within the file,
since a mis-mapped point reads as *ok* rather than *lost*. **Two of its three limits
push the same direction.**

**What would settle it** is not mine to run: among the files that *declined*, how many
segments were unverified? If the declined set carries more unverified segments per file
than the repaired set, the bias is confirmed — and a threshold set on the present
distribution is set against a population that excludes the cases it most needs to bound.

---

## The two numbers are complements of one partition

`vmsam-dev-2` confirmed from its code that a gap position **is never generated** — probe
positions are drawn only from pieces whose `source == "candidate"`, with the offset
bounded by `piece_length − window`, so a probe cannot land in a gap or even overlap one.
**My directional mechanism is dead** and the concession stands.

The reason the code is that way is the campaign's own recurring shape: **probing a
master-filled gap against the master would be a control that cannot fail**, because the
gap holds the master's own audio by construction and would return a confident zero on a
track wrong everywhere else.

**But the consequence is that the unverified part of a track is exactly the silent part.**

| | |
| :--- | :--- |
| the repair's alignment figure | a statement about the **candidate fraction** of a track |
| the silence figure measured here | **how much of the track that fraction is not** |

```
id 150   21.7 % silent (delivered)      -> verification covered at most 78.3 %
id 695   24.5 % accounting / 23.4 % sampled -> at most ~76 %
the 14   median 12.9 %                  -> at most 87.1 % of the median gappy track
```

**Neither side had noticed the two figures describe the same boundary from opposite
sides.** One says how good the verified part is; the other says how much of the track
that is.

## A check that is structurally unavailable to the repair's own verifier

A gap means candidate content was **lost**. The prober cannot see a wrongly placed or
over-wide gap, because it never looks there — **and the failure would be silent in the
literal sense that it produces silence.**

The check performed here compares the **delivered artefact** against the **candidate at
the mapped position**: where the output is silent and the source had audio, content was
dropped. It requires both sides at once, and the repair's stage holds the source but not
the artefact.

**Its limits, stated before it is relied on:** 30–60 s sampling bounds a *fraction*, not
a duration; gaps shorter than the spacing are invisible; a mis-mapped probe reads as
*ok* rather than *lost*, so it **understates**; and it cannot say whether a gap is
*correct* — only that the source had audio there.
