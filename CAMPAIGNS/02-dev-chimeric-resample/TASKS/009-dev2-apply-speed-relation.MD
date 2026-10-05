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

### The speed path now runs end to end — T15

`vmsam-forensic` corrected its own earlier finding: there are **19 PAL files** in
the corpus, roughly twice the NTSC population, so the `asetrate` path is exercised
at 1.0427 and not only at 1.001. It also warned that those 19 carry **no usable
delay measurement** — a 4.27 % relation destroys correlation, so stage 1, its own
prominence measure and `vmsam-dev-1`'s locator all find nothing, and the duration
ratio is the evidence. That reshaped the orchestrator: a **speed-only plan carries
no segments**, and demanding one would have made the intended case look like a
failure. One full-coverage segment is now synthesised instead.

Fixture built as a real naive PAL transfer would be — sped up **with the pitch
rising with it**, which `docs/AUDIO_SPEED_POLICY.MD` says is industry practice and
is exactly what makes `asetrate` the exact inverse rather than merely the
best-scoring option. Master 600.023 s of real audio, candidate 575.458 s.

| | |
| :--- | :--- |
| outcome | **repaired**, 2 audio tracks |
| marker | **`resampled:1.042698`** — and correctly **not** `chimeric`, because there is no splice |
| requested ratio | 1.042698412698412698… |
| **applied ratio** | **1.042698010657508578…** |
| verification | **aligned, worst 0.9 ms** |

Probes at 72.5 / 217.5 / 362.5 / 507.5 s read **0.00 / −0.88 / −0.25 / −0.75 ms**.
Flat across the file is the point: a residual coefficient error shows as a lag
that *grows* with position, so four spread probes reading flat is what says the
speed relation was removed rather than merely shifted.

**The probes had to be spread for that to be true.** With a speed-only plan there
is one segment, and the original probe rule took a single sample at its midpoint —
which is where a residual drift is *smallest*. That was measuring the error where
it is least visible. Probes are now distributed across a single long segment.

### The wrong direction, as its own column

`docs/AUDIO_SPEED_POLICY.MD` weakness 3 requires it: a sweep that applied `× r`
instead of `÷ r` produced `rubberband` winning 33 files of 33, and nothing inside
that run could have caught it.

Run: **declined**, and by the *bounds* check rather than the verifier —
`segment reads the candidate at [0, 600016) ms, outside its 551893 ms`. Inverting
the ratio makes the resampled candidate 551.9 s against a 600 s master, so the
plan cannot be executed at all and is refused before any work is done.

So the sign error is covered **twice, at different stages**: the bounds check when
the wrong direction makes the candidate too short, and the alignment verifier when
it does not — the latter demonstrated on `TASKS/007`'s fixture 2, where a wrong
sign that *did* fit was caught at 16 145.9 ms.

### A guard fired where it should not have, which is why they are fired

The first attempt at T15 was **declined** by the constant-offset refusal written
for objective 2: a speed-only plan is one full-coverage segment at offset zero,
which is indistinguishable from a pure container shift *unless you know a resample
is being applied*. The refusal now exempts that case. Caught on the first real
run of the speed path, by a guard doing its job in the wrong place — which is the
argument for firing guards rather than reading them.

### What `vmsam-dev-1` delivered, and what is carried rather than restated

Verdict `asetrate` on ids 33 and 31, with all four arms:

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **`asetrate` correct** | **0.9588** | **0.9551** |
| `asetrate` wrong sign | 0.6569 | 0.6562 |
| `rubberband` | 0.9484 | 0.9453 |

The wrong-direction arm lands **below the uncorrected control** on both files, so
the direction is demonstrated rather than asserted.

`vmsam-dev-1` instructs using **1001/1000 exactly** rather than its point estimate
of 1.000993: the exact rational lies inside its interval and is more precise than
the measurement, so propagating measurement noise into a file would be strictly
worse. Its interval `[1.000953, 1.001034]` and `ratio_precision: 3` are carried in
the **repair report**, never in the tag — the tag states the transformation
applied, the report states the confidence, and merging them would make the tag
mean two things.

**Its warning, carried with its framing, not hardened.** `AUDIO_SPEED_POLICY.MD`
measures `rubberband` losing to `asetrate` by 0.278–0.313; here the gap is
**0.0104 and 0.0098**, an order of magnitude smaller. That is consistent with the
document rather than against it — the document already records that against a real
master `asetrate` reaches only 0.94–0.98, and at 0.099 % the losing arm has far
less room to fall. The ruling stands, and its justification is physical rather
than statistical. But **the margin that would detect an inverting case in this
corpus is ~0.010, not ~0.109**, so a detector built to the document's expected
separation could miss one. `vmsam-dev-1` will send the four scores with every
future verdict so the separation is visible rather than taken on trust.

`vmsam-dev-1` also measured that the residual scatter about the linear fit is
38.3 and 41.0 ms against 35.8 ms predicted by pure quantisation of a 124 ms step,
so **no staircase hides under the slope** on those two files: plain
`resampled:<factor>` is correct for them, not `chimeric+resampled`.

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
