# 009 — Apply a measured speed relation to audio and subtitles

**Agent:** `vmsam-dev-2` (repair). **Pairs with 008.** **State:** open.
**Campaign objective 3, the repair half.**

## Goal

Take the coefficient 008 measured and produce a track on the master's timeline:
audio resampled, **and subtitles re-timed by the same coefficient**. `CAMPAIGN.MD`
is explicit that both are taken.

Method: `docs/AUDIO_SPEED_POLICY.MD`, which is a ruling and not a suggestion —
`asetrate`, not `rubberband`, not `atempo`, and read the inverting case before you
assume it carries over to a coefficient it does not name.

Subtitles: `docs/SUBTITLE_CODECS.MD` says which formats can be re-timed at all.
An image format is not one of them; say so rather than shipping a plausible file.

## Marking

`fabricated = "resampled:<factor>"`, or `"chimeric+resampled:<factor>"` in that
order when a staircase sat on top of the slope (`SPEC_ZONE_A.MD` §4). The same
string as a `VMSAM_FABRICATED` Matroska track tag. Consumers test truthiness.

Write the factor you actually applied, at the precision you actually applied it.
A tag reading `resampled:1.042709` on a track resampled by something else is worse
than no tag: campaign 1 spent real work dating 67 outputs by mtime because nothing
in them said what produced them.

## Same structure as 007

Own module, called from zone A, behind an **off-by-default `config.ini`
`[features]` key**. Closes on `SPEC_ZONE_A.MD` §2 — real error file, repaired
track present in the final mkv, compared against a target you built yourself.

## What changed / measured / dropped / blocked

**State: the primitive is built and measured. It is NOT yet wired into the
repair path, and no real coefficient has been applied** — 008 has not delivered
one. What exists is the transformation, verified in both directions.

### What changed — commit `3747488`

`src/merge_video_resample.py`, a new module. It applies, it does not re-judge:
`docs/AUDIO_SPEED_POLICY.MD` is a ruling, so `asetrate`, never `rubberband`,
never `atempo`.

### What was measured — T7, target computed by hand before the run

Input: a pure 1000.000 Hz tone, 60.000 s, 48 kHz. Requested ratio
r = 1.042709, defined as `master_timespan / candidate_timespan`, so the
candidate runs **fast** and must be **slowed**.

| arm | duration | pitch |
| :--- | ---: | ---: |
| predicted, right direction | 62.562624 s | 959.039 Hz |
| **measured, `asetrate` right direction** | **62.562625 s** | **959.039 Hz** |
| **measured, WRONG direction** | **57.542458 s** | **1042.709 Hz** |

The wrong-direction arm is carried as its own column because
`AUDIO_SPEED_POLICY.MD` weakness 3 records a sweep that applied `× r` instead of
`÷ r`, landed twice as wrong, and produced `rubberband` winning 33 files of 33 —
consistent sign, plausible magnitudes, and **nothing inside the run able to catch
it**. Here the two arms are 83.7 Hz apart, so a sign error cannot hide.

### The factor written into the tag is not the factor requested

`asetrate` takes an **integer** sample rate, so the factor actually obtained is
`intermediate / round(intermediate / r)` and not `r`. `SPEC_ZONE_A.MD` §4 is
explicit that the tag must carry what was applied, at the precision applied, so:

| | |
| :--- | ---: |
| requested | 1.042709 |
| **effective, exact** | 1.042710395333870980880927361 |
| **written to `VMSAM_FABRICATED`** | `resampled:1.042710` |
| residual drift over a 1420 s episode | **1.981 ms** |
| one video frame at 23.976 fps | 41.708 ms |

Setting the rate directly to `48000/r` would quantise at 0.5 Hz, i.e. 1.1 × 10⁻⁵
relative — about 15 ms over the same episode. Going up to 8× first divides that
by eight, which is where the 1.981 ms comes from. The residual is stated before
the run, not discovered after it.

### What was dropped

- **Trusting the requested ratio as the tag value.** It is wrong in the sixth
  decimal, and a tag reading `resampled:1.042709` on a track stretched by
  something else is worse than no tag (`SPEC_ZONE_A.MD` §4).
- **Applying the speed relation as a separate pre-pass over the whole file.**
  The chain is a filter prefix instead, so a speed relation with a staircase on
  top of it is one pass, not two, and the order is fixed: **resample first, then
  splice** — the slope is underneath and the staircase sits on it.

### What stays blocked

1. **A measured coefficient, and its verdict.** Agreed with `vmsam-dev-1`:
   I get `verdict ∈ {asetrate, rubberband, leave_alone, decline}`, not a bare
   number. `AUDIO_SPEED_POLICY.MD` requires three outcomes and a decline, records
   that the destructive mistake — resampling a file that needed nothing — was
   **twice as common** as the inverting case the detector exists to find, and
   fixes the match bar at **0.85** from measured spread. A bare coefficient would
   make me apply `asetrate` by default and let an inverting case through
   undetected.
2. **`rubberband` is not implemented.** If dev-1 returns that verdict I decline
   the track and say so rather than apply the wrong transformation. Deliberate:
   the policy names it as the correct repair for the inverting case, and
   `AUDIO_SPEED_POLICY.MD` says the inverting case **exists** (1 file of 33), so
   this is a real gap and not a hypothetical one.
3. **Subtitles.** Re-timing by the same coefficient is implemented for the text
   class; `docs/SUBTITLE_CODECS.MD` rules that bitmap cannot be re-timed by cue
   rewriting, so those tracks are declined **by name** and counted apart from a
   failed extract. Not yet exercised against a real PAL pair.
4. Everything blocking `TASKS/007` blocks this too — the `config.ini` flag, the
   locked module, and the absence of a real error file.
