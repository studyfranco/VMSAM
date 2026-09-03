# Agent notes — logbook

One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-03 — a measurement taken where there is no signal is not a measurement

Comparing the repaired track of error id 266 against a hand-built target, the head
region reported **−170.69 ms with r = 0.82** — a confident-looking disagreement.
It was noise correlated against noise: the first seconds of that programme are
near-silent, RMS **1 × 10⁻⁵** of full scale. With signal in the window the same
comparison reads **−0.19 ms at r = 0.974**.

**This is the most insidious member of the "control that cannot fail" family** the
campaign keeps meeting, because there is no construction to inspect. The window
was real, the instrument ran, a number came back with a plausible correlation.
Only the *content* was missing — and a high correlation is no defence, because
silence correlates with silence.

**Decision:** every verification probe now measures the RMS of both windows and
reports `no_signal` instead of a lag when either is below **1 × 10⁻⁴** of full
scale. If *no* probe on a track carries signal the track's outcome is **`skipped`
— unverified, not verified.**

*Where the threshold comes from:* the measured gap on this file, silence at
1 × 10⁻⁵ against content at 2.4 × 10⁻³ to 2.9 × 10⁻². It sits an order of
magnitude above the silence and an order below the quietest content. It is a
measured spread, not a curve (`docs/AUDIO_SPEED_POLICY.MD`: "every threshold must
come from measured spread").

*Why not use the correlation instead:* it does not work. The silent window scored
0.82–0.93, higher than several windows that were genuinely aligned. Loudness is
the discriminator here, not agreement.

*Why a per-probe outcome rather than dropping the probe silently:* dropping it
would leave a track verified on fewer probes than it appears to be. The count of
windows without signal is reported alongside the count measured.

`vmsam-dev-1` reports the same defect in its own locator — its narrowing scan
probes at fixed positions and would bisect across a silent stretch with nothing
checking for content.

## 2026-09-03 — a hand-built target needs verifying as much as the code does

`AGENT.MD` § Target-first says to build the expected file by hand, first, then
compare. On error id 266 the target was **wrong**, and the code was right.

Every region of the comparison reported an identical −170.69 ms. A constant lag
across gaps *and* plateaus means a **uniform shift**, not misplaced pieces — so
the question became which of the two objects was moving, and the way to settle it
was to correlate the *target* against the master. The target read +162 ms; the
produced track did not. Its first piece measured 1.829 s instead of 2.000, because
`-ss 0` on an AAC track drops the decoder's priming samples, and every later piece
inherited the shift.

**Decision:** build targets in exact sample counts rather than trusting ffmpeg's
`-t`, and correlate the target itself against the reference before believing a
disagreement with the code.

*Why this is a finding and not just a mistake:* `AGENT.MD` requires the target and
requires the comparison, and says nothing about verifying the target. On this
evidence it should, because a wrong target produces a **confident false failure** —
and, in the symmetric case, would produce a confident false pass. The whole weight
of § Target-first rests on an artefact that is currently assumed correct because a
human made it. `vmsam-dev-1` independently proposed the same addition after seeing
this. A finding for the owner (`AGENT.MD` § Documents), never an edit.

## 2026-09-03 — a fixture built from one file cannot test what depends on two files differing

`vmsam-ci` said it before I hit it: a control whose two arms are one file and a
copy of itself proves the gate opens and closes, and proves nothing about media
whose two sides were encoded independently — which is every real case.

Acted on. A second fixture was built from two genuinely independent releases of
the same episode: a Blu-ray master against a streaming master, FLAC 48 kHz
against AAC 44.1 kHz, natural offset constant at −955 ms and correlation
0.910–0.973 — high, but not the 1.000 that two releases sharing one audio source
read. **It found a defect on its first run.** The plan's base offset carried the
wrong sign, and the repaired track came out uniformly twice the base offset away
from the master, with the staircase correctly removed — uniform, so invisible to
any per-plateau check. The first fixture had a base offset of zero and could not
have exposed it.

The sign was written down as an equation, agreed in writing with `vmsam-dev-1`,
and verified once on the first fixture. **Stating a convention does not protect
you from misapplying it.** `vmsam-dev-1` withdrew a calibrated edge estimator on
the same day for the same underlying reason — its threshold tested for a
correlation of 1.00000, which only a file against its own copy ever reaches. Three
of us have now paid this tax.

**Decision:** the repair verifies its own output. After muxing, each rebuilt track
is correlated against the master at up to four positions and the repair is
**refused** above 100 ms. It is the automatic form of `SPEC_ZONE_A.MD` §2 point 3.

*Why not simply be more careful with the sign:* because the failure is uniform and
silent, and the whole campaign exists because five pieces of careful, reviewed,
green-tested code did nothing. A convention that can only be enforced by attention
is not enforced.

*Why the probes never sit in a gap:* a gap is filled from the master by
construction, so it matches the master at any offset. Probing there is the control
that cannot fail which `docs/AUDIO_SPEED_POLICY.MD` devotes a section to.

*Why it verifies a zero rather than measuring a step:* `vmsam-dev-1` measured that
the same physical step reads different millisecond values at different window
lengths, because the chromaprint quantum is computed per call. Re-measuring a step
would disagree with the measurement half and neither would be wrong. A zero is
quantum-independent.

*Why `repair_verify` defaults ON inside a feature that is off by default:*
`WRITE_ZONES.MD` §4's rule is that new capability must not run unless enabled, and
that is held by `repair_chimeric`. The verification is not a further capability; it
is what stops the repair shipping a silently wrong track. A safety you have to
remember to arm is not a safety.

*Why a verification refusal is `declined` and not `failed`:* the module ran,
inspected what it had produced, and said no. Collapsing the two would drop the
outcome that carries a reason — the same error as collapsing "could not measure"
into "unrepairable".

Fired both ways on real runs rather than in a unit test: the bad plan declines
with "track 1 (ja) off by 16145.9 ms", and the merge then correctly falls through
to "Only ... file left" because nothing was repaired; the corrected plan reports
"aligned, worst lag 0.5 ms". A track whose language the master lacks reports
**skipped**, never aligned.

## 2026-09-03 — the repair is reachable only because one guard in zone A was moved

`remove_not_compatible_video` ended with `if len(dict_file_path_obj) < 2: raise`.
`fusion.py:286` calls `merge_videos` with exactly **two** files — the master and
the file in error. A refusal removes one, one remains, and that line raised —
one statement *after* the repair had attached its object to
`best_video.sameAudioMD5UseForCalculation`.

So on the fusion path, which is the path this campaign exists to fix, a repair
would have been killed every single time, and so would anything built on the
measurement half. Written, reviewed, tested green, reached by nothing: the
campaign-1 shape, in the guard structure itself.

**Decision:** the guard now stands unless something was actually repaired. A
repaired object makes the merge useful on its own — verified by reading the
downstream path with a single-entry `dict_file_path_obj`:
`dict_with_video_quality_logic` is `{master:{}}`, `set_bad_video` is empty,
`best_video` resolves to the master, and the `sameAudioMD5UseForCalculation`
loop at `mergeVideo.py:1780` adds the repaired track. Then verified by running
it: a merged mkv is produced and carries the repaired tracks.

**Why not widen it further** — e.g. drop the guard altogether: with nothing
repaired, one file left really is useless to merge, and the exception is the
correct outcome. The control run with the flag shut still dies there, which is
what makes the new branch a gate rather than a deletion.

## 2026-09-03 — finding: `keep_best_audio`'s `fabricated` rule cannot fire, and the harm it prevents is now measured

`SPEC_ZONE_A.MD` §4 says the branch *"activates by itself the day the repair sets
the key"*. It does not, as written.

Both call sites — `mergeVideo.py:1853`, and `:998` reached through `:1861` —
iterate `out_video_metadata.audios`, and `out_video_metadata` is a **fresh**
`video` object built from the merged split file at `:1822` and populated by
`get_mediadata()` at `:1823`. `get_mediadata` never writes a `fabricated` key.
The key the repair sets in memory is on a different dict object, which nothing
downstream reads.

The marker does survive to that point, but under another name. Measured: a
`VMSAM_FABRICATED` track tag survives ffmpeg pass 1 (`-c copy -map_metadata 0`),
pass 2, `mkvmerge --no-global-tags` and the final `mkvmerge`, and `get_mediadata`
returns it as `audio['extra']['VMSAM_FABRICATED']` — never as
`audio['fabricated']`.

**The consequence is not hypothetical.** In a real merge the split file carried
four audio tracks — two intact from the master, two fabricated by the repair —
and the final file kept **only the fabricated ones**. A fabricated track evicted
an intact one of the same language, which is the exact sentence §4 was written
to prevent.

**What was done instead.** Nothing, because both repairs are frozen:
`keep_best_audio` carries no marker in the tree (see the entry below, 2026-09-03)
and `get_mediadata` lives in `video.py` outside its two tagged functions. The
repair sets the in-memory key anyway — it is the contract — and states plainly
that no consumer reads it. Recorded rather than resolved: `AGENT.MD` § Documents,
being right about a rule does not make it ours to change. The fix is one line in
either place and both are the owner's.

**A tempting non-fix, rejected.** The eviction could be dodged by encoding the
rebuilt track at a lower bitrate so it loses the codec comparison. That games a
broken rule at the cost of the track, and it would hide the finding. Measured
that it would not even work reliably: with the source bitrate resolved and
passed to the encoder, AAC still overshoots its target by ~4 % and the fabricated
track still compares higher.

## 2026-09-03 — finding: `main.py` is broken under Python 3.14, and the fix already exists ten files away

A plain two-file merge through `main.py` dies with `KeyError: 'ffmpeg'` raised
inside a `Pool` worker, before any campaign-2 code runs.

Cause, and `AGENT.MD` § Runtime environment already names the mechanism: Python
3.14 defaults to `forkserver`, so a pool child sees only module defaults and
`tools.software` is `{}` there. `main_gestionar_show.py:432` calls
`set_start_method('fork', force=True)` and `internal_api.py:45` calls
`tools.load_merge_runtime_from_env()`; **`main.py` calls neither.** `run.sh`
reaches `main.py` through `software=main` and `infinite_wrapper.py`.

`main.py` is frozen (`WRITE_ZONES.MD` §1), so this is a finding. Practical cost
to us: the CLI path cannot validate anything, and any agent reproducing locally
through `main.py` will hit this first and may read it as a defect of their own
work. The campaign-2 harness drives `merge_videos` through its own entry point
that forces `fork`, and says in its docstring that it is a workaround and not a
fix.

## 2026-09-03 — `check_write_zones.py`: the rules get an instrument, and it lives outside the repository

`WRITE_ZONES.MD` §2 rule 2 says a zone is real only if it is checked, and lists
four shell snippets. Three of them are read by eye. Campaign 1 pushed a commit
with a failing test by gating on the wrong thing and reading a suite's output
instead of its exit status (`CAMPAIGNS/01-dev-AI/RESTART.MD` §8), so the checks
are now one command with one exit status.

**Where it lives: beside the repository, not in it** —
`/home/vmsam/src/VMSAM_HELP_AI/tools/check_write_zones.py`. §4's test is *would
this still matter to someone running VMSAM in production?* A harness that reads
`git diff` and enforces a campaign's write rules would not. Campaign 1 left 148 773
lines of exactly this kind of scaffolding under `src/`, nineteen times the program.
One copy outside, called by absolute path from every clone, also means the four
agents cannot drift onto four versions of the rule.

**Why the naive form of the zone check is wrong.** `git diff --cached -U0` read by
eye compares changed lines against the zones of the *new* file. A pure deletion has
no new-side line at all, so a `-` hunk removing a frozen line above a zone reads as
"nothing added outside a zone" and passes. The checker maps removals onto the *old*
blob's zones and additions onto the *new* blob's, separately. The self-test case
`outside zone: pure deletion below the zone` is that defect, pinned.

**Two more holes the eye version leaves open**, both now refused: moving an `END`
marker down to swallow a line into a zone (the diff lands inside the new zone and
looks legal), and creating a fresh marker pair — §2 says adding a zone is an owner
decision, so any change in marker count fails.

**It answers three things, not two.** Exit 2 means the checker could not run and is
*not* a pass. `RESTART.MD` §3.4 found the same shape five times in one morning: a
reader with two outcomes describing a world with three, and the missing outcome is
always *the instrument did not run*.

**Fired, both answers, before being believed.** `--self-test` builds throwaway
repositories and drives 27 cases; each refusal has a matching case that must be
accepted. On the real tree at `f0e7119`: clean; a one-line insert at
`mergeVideo.py:1` refused; the same insert at `mergeVideo.py:795`, inside zone A,
accepted. An armed guard that has never fired is indistinguishable from a working
one, and that includes this one.

**What it deliberately does not check**, so that a pass is never over-read: that a
new module is called by anything; that a file belongs in the repository at all
(§4 is a judgement, not a path rule); privacy; `config.ini` semantics.

## 2026-09-03 — finding: `keep_best_audio` is listed as a tagged zone and carries no tag

`WRITE_ZONES.MD` §2 lists `mergeVideo.py` / `keep_best_audio` among the existing
zones, and the paragraph above the table cites it as the reason the comment form of
the marker exists: *"`keep_best_audio` uses the comment form for exactly this
reason."* In the tree at `f0e7119` there is no marker of either form in that
function:

```sh
grep -n "AGENT modification" src/mergeVideo.py
# 663, 667, 674, 678, 682, 690, 789, 800 — all in adjust_delay_to_frame and
# remove_not_compatible_video. Nothing near keep_best_audio (1011).
```

The `fabricated` branch the entry refers to is present and correct
(`mergeVideo.py:1036-1039`), so what is missing is the permission, not the code.
By rule 1 — a path that is not open is closed — the function is frozen, and
`check_write_zones.py` enforces that reading.

**Consequence, and what was done instead.** Nothing is blocked today: the branch is
already written, and `SPEC_ZONE_A.MD` §4 says it activates by itself the day the
repair sets the key on the audio dict elsewhere. So the campaign proceeds without
the zone. It becomes blocking only if the `fabricated` comparison itself has to
change — a case we do not expect and will report if we meet it.
**This is a finding for the owner, not an edit** (`AGENT.MD` § Documents): the fix
is either two marker lines in `mergeVideo.py` or one line struck from the §2 table,
and both are owner decisions.

## 2026-09-03 — finding: the Rust sources are frozen by rule and editable by instruction

`src/lib.rs`, `src/main.rs` and `Cargo.toml` appear in neither `WRITE_ZONES.MD` §1
(frozen) nor §4 (open). Rule 1 settles it — *a path that is not listed as open is
closed* — and `check_write_zones.py` treats them as frozen.

`AGENT.MD` § Coding Directives points the other way: *"Rust (`audio_sync`) must
pass `cargo check`, `cargo clippy`, `cargo fmt` cleanly before submitting any
changes"*, and § Testing says `cargo build` / `cargo check` are "allowed and
recommended". A file nobody may change cannot have changes submitted, so the two
documents disagree.

**What was done instead:** the strict reading. Nothing in the first objectives needs
the Rust core — stage 2 is the instrument we measure *with*, not the one under
repair — so no work is blocked. Recorded rather than resolved, because
`AGENT.MD` § Documents is explicit that being right about a rule does not make it
ours to change.

## 2026-09-02 — master prepared for campaign 2

`gestionar_show` imported from `dev-AI` and corrected: the dual-uvicorn split, a
fixed internal port (`tools.internal_api_port = 42085`), and `fusion.py` rewritten
from 1334 to 381 lines — the queue and the merge kept, the campaign instrumentation
dropped.

Twelve defects found and fixed while wiring it, none of which the compiler could
see: `get_git_commit` raising `NameError` on first call, three `tools.*` attributes
referenced but never defined, `_env_flag` used for a path (it returns a boolean),
nine `tools.logs` writes leaking into thread contexts through cross-file
`Thread(target=…)` declarations, `.get(0)` on a list, a `Decimal`/`float` mix that
made the no-forced-best-video path drop every candidate *silently*, a lost-master
window in the production swap, a pool never shut down, and a production failure
that left no trace anywhere.

Two hazards were closed structurally rather than by a fix: the episode lock, because
the integration loop and the fusion worker are separate processes that can write the
same episode; and `tools.load_merge_runtime_from_env()`, because Python 3.14
defaults to `forkserver` and a forkserver child sees only module defaults — which
would have put the two processes' locks in different trees, silently, while
`software={}` failed every ffmpeg call.

Write zones were declared and the first tags placed. The rule that governs them —
*a unit test cannot tell you that nothing calls the unit* — comes from campaign 1
shipping five pieces of code that were reviewed, tested green, and never called.
