# 011 — the corpus generator, and the mailbox watcher under it

**`vmsam-ci-build`.** Opened 2026-09-05 at `70be985`. **OPEN** — the generator produces
artefact 1 of a case and not artefact 2; see *what stayed blocked*.

Live state is in `VMSAM_HELP_AI/ci-build/STATE.md`, not here. This file is the report.

## Context

`VMSAM_HELP_AI/corpus/SELECTION.md` is 489 lines of forensic's selection and nothing had
been built from it. forensic owns WHICH files; this task owns turning that into artefacts.
Three developers had each named a case whose absence makes their own results
unfalsifiable, and the shared CI mailbox had no picker on this side.

## What I changed

**Nothing in this repository except this file.** The generator and the watcher are
harnesses, so `WRITE_ZONES.MD` §4 puts them beside the repository:

    VMSAM_HELP_AI/ci-build/corpus_build.py    the generator
    VMSAM_HELP_AI/ci-build/watch_inbox.py     the mailbox watcher
    VMSAM_HELP_AI/corpus/cases/               the artefacts
    VMSAM_HELP_AI/corpus/INDEX.jsonl          the index

`git config user.name vmsam-ci-build` in this clone. **It was unset**, so every commit
would have landed under the owner's identity.

## What I measured — every number here has the command behind it in `STATE.md`

### The four measurements that changed the design before it was written

**THE CONVERTER NULL HOLDS.** `CAMPAIGN.MD` requires it before the staleness check can be
trusted and nobody had run it. Same converter, same input, twice: **bit-identical samples**.
So a difference between two conversions is attributable to drift rather than to
nondeterminism. *Scope, because a null that has not failed is not a null that has been
swept: one source, one 20 s window, ffmpeg 9.0.1 / libavcodec 63.1.101.*

**A DERIVED PROFILE IS NOT THE PROFILE, AND THIS KILLED THE DESIGN I STARTED WITH.** I
intended to store one native archive and derive the other profiles from it.

    source -> mono 44100                  against
    source -> native PCM -> mono 44100    DIFFER: 25.1 % of samples, max 1 LSB,
                                          69.8 dB below signal
    the same at 32000:                    13.6 % of samples, max 1 LSB, 72.5 dB below

*The difference is inaudible and that is not the point.* **The converter check compares
SAMPLES, so a derived artefact would report drift on every case, someone would bolt on a
tolerance, and the tolerance would hide the real drift.** Every profile is now produced
from the source in one pass and no check compares across profiles.

**FLAC AT ITS DEFAULTS SILENTLY STORES 24-BIT.** On this build `-c:a flac` from an s16
source produced `bits_per_raw_sample=24`, a file **larger than the raw PCM**, and a decode
that differed from the original. `-sample_fmt s16 -frame_size 4096` round-trips
bit-identical at 0.55x. **Every artefact's round-trip is now asserted against the direct
PCM before publication and a failure is a refusal.** *The default would have produced a
corpus that was silently wrong and 5 % bigger while looking compressed.*

**`/tmp` IS A DIFFERENT FILESYSTEM FROM THE CORPUS** — device 2097216 against 69. Staging
lives inside the case directory so `EXDEV` cannot arise; the guard fires on a cross-device
publish and is exercised in the self-test.

### The three conversion profiles, read out of the engine, and they DISAGREE

    STAGE 1   prepare_get_delay_sub    pcm_s16le · pair-minimum channels · NO `-ar`,
                                       so each side keeps ITS OWN rate · loudnorm
    STAGE 2   second_delay_test        mono · `-ar` = min(pair's lowest, 44100) · loudnorm
    LOCATOR   change_point_locator     mono · `-ar 44100` HARDCODED · no loudnorm
              _extract -- WHAT THE REPAIR CONSUMES

**The locator pins 44100 where the pipeline derives `min(pair, 44100)`. On a pair whose
grid falls below 44100 the two disagree and the locator upsamples.** Filed as a finding
below; `change_point_locator.py` is another agent's module.

### Pairing 315 error ids to their masters

    314 of 315   resolve to EXACTLY ONE master by the SxxExx token in the season directory
      1 (id 110) UNRESOLVED -- two files carry that episode. NOT guessed.

*Positive control on the finder before believing it: the token pattern was run over all
5915 names in every season directory involved and matched 5915. The no-match branch never
fired, so a pattern matching nothing could not have reported a confident zero.*
**Cross-checked against forensic's independently-derived masters for ids 108, 109, 111 and
134: four masters, durations agreeing to 0.000 s.** *That validates the PAIRING and not the
durations — forensic and I both use ffprobe, so the durations are ONE instrument. What is
independent is how each of us decided which file is the master.*

### dev-3's sub-44100 case exists in the wild, and forensic's statement is still correct

    swept all 314 resolved pairs
    id 307  jpn  candidate 32000 against master 48000   pair grid 32000
    id 316  fre  candidate 32000 against master 48000   pair grid 32000

**forensic wrote "~2 streams in 3482 sampled read 32 kHz, and NO PAIR exists", asking for a
pair with BOTH SIDES below 44100. That is still true: every master here is 48000.** *But
the grid is `min` across both sides — `video.get_less_sampling_rate` takes the minimum over
the two audio lists and `mergeVideo` then clamps — so **ONE side below 44100 is enough for
the clamp to stop firing**, which is the property dev-3's tripwire needs.* Read from the
code, not inferred. **So the case does not have to be wholly synthetic.**

### The two cut constructions do not remove the same amount

Built from one shared PCM base, 300.000 s exactly, cut nominally 5.000 s at t=120:

    nocut                   removed  -0.0018 s   (AAC priming, 80 samples)
    cut_reencode_across     removed   4.9905 s   delta  -0.0095
    cut_concat_at_join      removed   4.9673 s   delta  -0.0327

> **THE TWO CONSTRUCTIONS DIFFER BY 23.2 ms ON A NOMINALLY IDENTICAL 5 s CUT — more than
> half a video frame at 23.976 fps.** *That is forensic's rule earning itself: two agents
> who each built one construction were comparing cuts that differ by more than a frame.*

Measured by **counting decoded samples**, never by reading a container duration.

### `-t` is eaten by the stream's container start_time, silently

    -t 300 on a stream stamped start_time=1.165 s  ->  298.848 s of audio
    the SAME command on the SAME file, on a stream stamped 0.000  ->  exact

**So one code path in my own program was exact and another was 1.15 s short, on the same
file, because they happened to pick different streams.** *This is the quantity
`change_point_locator._start_times_ms` was written for.* Fixture bases now raise `-t` by the
start_time and then trim the payload to an exact sample count; a short base is a refusal.

## What I dropped

- **Native-archive-plus-derivation.** Measured above; a derived profile is not the profile.
- **Deduplicating profiles by resemblance.** Stage 2 and the locator are both "mono at some
  rate" and look mergeable. They answer different questions and their rates disagree on
  exactly the pairs that matter — `WRITE_ZONES.MD` §9's counter-rule.

## What stayed blocked

- **ARTEFACT 2 OF A CASE IS NOT BUILT.** `CAMPAIGN.MD` defines a case as the comparison
  audios **and** the in-memory state captured immediately before
  `remove_not_compatible_video`, the produced files, and a written record of what exists at
  that instant. Only artefact 1 exists. **A case directory today is half a case and the
  record says so in its own field.**
- **dev-1's null cannot be built in this tier.** They asked for same content, no cut,
  **different frame rate**; the owner's construction constraint is **audio only, no video**,
  so a frame rate cannot be varied at all. A no-cut arm at the same rate is built instead
  and the record names it as a **weaker substitute**: it controls for *the instrument always
  finds something* and not for a rate relation being mistaken for a cut. *dev-1's null
  produced the decisive result of that evening and this is not it.*
- **dev-2's head-gap PAIR is not built.** forensic established it is a case to SELECT rather
  than build, and the missing half is narrow: a within-job pair whose A side is
  **substantive**. Selecting it needs the job logs, which is measurement and belongs to ci.
- **The curated tier has no ground truth**, and that is a property of the design rather than
  of this program. Every verdict over it is *consistent with its source*, which is agreement
  and never correctness. Carried in every curated record as its own field.
- **`errors_20260903.json` is two days old** and forensic says errors keep arriving. Every
  count here is against that dump, named in the record.

## Findings for the architect

1. **The locator's hardcoded 44100 against the pipeline's derived `min(pair, 44100)`** —
   `change_point_locator._extract`, reading of the code, not observed in a run. Triggering
   condition: any pair whose lowest source rate is below 44100. Two exist in the corpus
   today (ids 307, 316).
2. **A citation crossing a staleness boundary** — a section real at the citer's reference
   and absent at the reader's produces a clean, correct negative on both sides. Surfaced
   when `WRITE_ZONES.MD` §15 was cited to a clone whose `origin` served a tree without it.
