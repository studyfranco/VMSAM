# 017 — dev-subs: id 12's four missing subtitle languages (Task 1+2 of BRIEF.md)

Seat: `vmsam-dev-subs`. Status: **CLOSED on the "four languages" question —
Lead confirmed independently. `mergeVideo.py:1744-1746` (FROZEN, inside
`generate_new_file`), driven by `config.json`'s `language_to_completely_remove`
(`['ar','tr','id','fil','ms','he','hi']`). Configured policy, executed
correctly, silently. NOT content loss — the "four languages silently lost"
framing does not survive. What survives: the removal leaves no record, and
the repair wastes effort rebuilding tracks config has already condemned.
Task 3 (verify shipped file) and the policy question (should ar/hi/id/ms be
on that list) both routed to the owner by the Lead. Task 4 (head-span cue
drop, the one remaining PRODUCT-correctness defect on this surface) stays
held pending `vmsam-dev-frame`'s retention patch landing. No code changed.**

## CLOSURE — Lead's independent confirmation, and the corrected framing

Lead re-ran the same check independently (`src/config.json` vs the deployed
copy beside the container, identical; `mergeVideo.py:1743-1746`'s removal
branch carries no `tools.logs.append` anywhere, unlike its sibling). **Lead
had escalated this to the owner as the campaign's most serious product
defect and is now correcting that** — configured policy, executed correctly,
silently, is not the same thing as content loss. What actually survives as
real, product-facing findings:

1. **A decision with no record** — four tracks removed, nothing anywhere
   says why. Same class as `SPEC_ZONE_A.MD` §4d one stage further on: a skip
   is a decision and gets a line. This is what cost the investigation time —
   had the branch logged, nobody would have needed to chase it.
2. **100% wasted work** — the repair fully rebuilds four tracks (retiming,
   238/239/238/238 cues each) whose fate config already decided before the
   repair ran.
3. **`17/17` is honest, not disconnected-by-lying** — it correctly answers
   "did the repair build what it attempted," which is a different question
   than "what shipped." Both need answering; only one currently is.

**What does NOT survive:** "four subtitle languages were silently lost."
They were deliberately removed by standing configuration. The delivered
file is correct on that axis.

**What remains the only product-correctness defect on this surface:** the
13 tracks that DID ship are each short their opening 3 m 41 s of real
dialogue (Task 4's head-span drop) — unrelated to the removal list, and
possibly already fixed by `vmsam-dev-frame`'s retention patch. Holding as
instructed; will re-measure once that patch is gated.

## The methodological lesson (Lead's instruction to record it here, because
it is the day's lesson and this ticket demonstrated it twice)

**A measurement of an intermediate is not a measurement of the product.**
Two escalations today were built on real, correctly-taken measurements whose
*meaning* was wrong because the measurement stopped at an intermediate
artefact instead of the delivered file: 43% master fill that never actually
shipped, and — on this exact ticket — four "silently lost" languages that
were in fact policy-removed downstream, correctly, by design. **Both
corrections came from actually going and reading the delivered file/the
frozen removal branch instead of trusting the repair's own log.** This is
the whole argument for the five-leg validation and for Task 3's premise
("the verifier counts the shipped file, not the track it built") — not as
an abstract principle, but because it is the exact thing that would have
caught both errors before they reached the owner.

## UPDATE — the exact line, found by instrumented bisection (Lead's unlock:
copy the frozen file into the lab, run the copy, don't reconstruct)

`config.json`'s `language_to_completely_remove` = `['ar','tr','id','fil',
'ms','he','hi']` (checked both the repo's `src/config.json` and the deployed
copy beside the container — identical). `generate_new_file`
(`mergeVideo.py:1743-1746`), called on **every** video object fed into the
first merge pass — including `repaired_obj`, the chimeric repair's own
output — does this **before** any MD5/keep/empty check runs, and with **no
`tools.logs.append` on this branch at all** (its sibling MD5-dedup branch
three lines down does log, gated on `tools.dev`; this one is silent
unconditionally):

```python
for language,subs in video_obj.subtitles.items():
    if language in tools.language_to_completely_remove:
        for sub in subs:
            sub_track_to_remove.append(sub)   # <- no log line, ever
```

**Measured directly**, running this exact logic against the real
`full_repaired.mkv` (the byte-exact rebuilt intermediate) with the real
`config.json` loaded:

    KEPT:    en(1) pt(3) pt(4) es(5) es(6) fr(7) it(10) ja(11) ko(12)
             th(14) vi(15) zh(16) zh(17)          -- 13 tracks
    REMOVED: ar(2) hi(8) id(9) ms(13)              -- 4 tracks, all reason
             LANGUAGE_TO_COMPLETELY_REMOVE

**Exact match to the delivered file's 13 chimeric tracks and the 4 missing
languages, language-for-language.** This is the mechanism, not a correlate
of it.

**What this actually is, and why it is not the "adaptation" defect the
owner's standing objective names:** `language_to_completely_remove` reads as
a deliberate, hand-curated operator policy (it also names `tr`, `fil`, `he`
— not present in this candidate's language set, so not observed here, but
consistent with a curated list rather than an accident). **The repair did
not fail to adapt the candidate onto the master — it correctly rebuilt all
17 tracks, and a completely separate, later, frozen filter then discards
four of them by language, unconditionally, as configured.** Whether `ar`/
`hi`/`id`/`ms` *should* ship is an owner policy question, not a robustness
defect in difference-detection or candidate-adaptation.

**What IS the defect, precisely:** the repair's own success line
(`subtitles 17/17`, `0 declined, 0 failed`) is not a false claim about what
*it* built, but it is presented with no visibility that 4 of those 17 are
already known, by the very config the merge itself reads seconds later, to
be discarded before delivery. **Two costs, both real:** (1) the repair does
100% wasted work — extraction, retiming, muxing — on 4 tracks whose fate is
decided in advance; (2) nothing between the repair and the delivered file
says so, which is exactly `SPEC_ZONE_A.MD` §4e's "a skip is a decision and
gets a line too," except this skip isn't even a code-level skip inside the
zone-A contract — it is a *config-driven policy applied after the contract's
own accounting closes the book at 17/17*.

**This reframes Task 3.** "The verifier counts the shipped file" is exactly
right, and now for a *known, named* reason rather than an open question: a
verifier reading the delivered file's actual streams would catch this
specific case (and any other config-driven or frozen-chain drop) without
needing to know the mechanism in advance — which is the structural argument
for building it at the file boundary rather than chasing every individual
frozen-chain cause. My structural finding stands: there is still no open
call site between the repair and the file landing on disk, so where this
verifier can legally run is still the owner's fork to resolve, not mine.

---

## Original Task 1+2 report (kept for the record; superseded above on "where"
but the elimination work below is what let the language filter stand out
as the only remaining candidate)

## What I measured (all commands re-runnable; extracts stay in my lab, never
here — `VMSAM_HELP_AI/dev-subs/lab/`)

**1. Confirmed which test-output artefact is id 12.** The `.log` under
the shared output volume whose `[change_point_locator]` line carries
`pair=e39737967605 build=9ba571ad5307` matches `dev-frame`'s
`REPORT_id12_dropped_head_defect.md` exactly. Its `repair:` block: 17
subtitle tracks built (`en,ar,pt,pt,es,es,fr,hi,id,it,ja,ko,ms,th,vi,zh,zh`),
**0 declined, 0 failed**, line 78 `subtitles 17/17`. `ffprobe` on the
delivered `.mkv`: **23 total subtitle streams, 13 tagged
`VMSAM_FABRICATED=chimeric`**. The 13 shipped chimeric languages are exactly
the 17 built **minus `ar`, `hi`, `id`, `ms`** — matches the brief's
measurement exactly.

**2. Rebuilt the true intermediate, byte-for-byte.** The original intermediate
doesn't survive (checked the shared output volume and scratch storage; the
logged `manifest=` tmp dir is gone). Both source files (master + candidate)
still exist on the
read-only library, so I called `merge_video_chimeric.assemble_on_master_timeline`
**directly**, in my lab, with the exact two segments the log records
(`master=220000.0-480000.0` and `620000.0-1040000.0`, `candidate_offset_ms=
-1412.34` both). **Target, stated before running:** the report's
`kept_cues`/`dropped_cues` per subtitle track should match the production log
line-for-line, or the reproduction is not trustworthy. **Result: exact match
on all 17 tracks** (e.g. `ar 238/101`, `hi 239/101`, `id 238/100`, `ms
238/101` — identical to the production log). This is the real repair, not an
approximation.

**3. Probed the rebuilt intermediate (`full_repaired.mkv`) with `ffprobe`:
all 17 subtitle streams present, correctly tagged, including `ar`, `hi`,
`id`, `ms`.** `mux_repaired_file` (`merge_video_chimeric.py:1513`, OPEN zone)
never filters `subtitle_reports` — read the full body of
`assemble_on_master_timeline` (:1648-1933): the only post-build filter
(`max_silence_fraction`) touches `audio_reports` only. **Verdict: World 1 is
FALSE. The OPEN-zone repair is exonerated — it delivers all 17 subtitle
streams to its own output.** This is Task 1's answer, measured, not inferred.

**4. Traced the FROZEN chain one step at a time, on the real intermediate:**
   - `video.video.get_mediadata()` (`video.py`, FROZEN) reprobed
     `full_repaired.mkv`: all 17 languages resolve correctly via
     `iso639.Lang(...).pt1`, `keep=True` for all, no exception.
   - `is_language()`/`Lang(...).pt1` tested directly on `ar`, `hi`, `id`,
     `ms`, `en`, `fr`: **all resolve identically, no special-casing, no
     failure for the four "awkward" codes.**
   - `video.video.calculate_md5_streams_split()` (FROZEN) run on the real
     intermediate: **17 distinct content-MD5s, zero collisions** (measured
     both on the raw un-retimed candidate text and on the actual retimed
     output — same conclusion both times).
   - `mergeVideo.generate_merge_command_insert_ID_sub_track_set_not_default`
     (FROZEN, the final per-track keep/drop decision) run directly against
     the reprobed+MD5'd data: **kept all 17, dropped 0.**
   - `mergeVideo.generate_new_file` (FROZEN, the function that actually
     consumes `repaired_obj` on the path into the first merge pass): reading
     it closely, `sub['MD5']` is **never computed** for `repaired_obj` before
     this function runs — `generate_merge_command_common_md5` calls
     `generate_new_file` directly, no `calculate_md5_streams_split()` in
     between. So every one of the 17 tracks carries `MD5=''`, and the code's
     own guard (`if sub['MD5'] != '': md5_sub_already_added.add(...)`) never
     adds `''` to the dedup set — so the MD5 dedup branch cannot be what
     drops these four either, by construction. The other guard in the same
     function, `int(sub.get("StreamSize",1))==0 or
     float(sub.get("Duration",1))==0` (this is the exact mechanism
     `SPEC_ZONE_A.MD` §4e's "one observed run skipped an element as 'empty'"
     describes) — measured `Duration` on all 17 tracks of the real
     intermediate: **all non-zero (~1039 s), `StreamSize` absent (defaults to
     1) for all.** Does not fire here either.

## Task 1's answer

**World 2. The repaired intermediate carries all 17 subtitle streams,
including `ar`/`hi`/`id`/`ms`. The loss happens downstream, inside the FROZEN
merge chain (`mergeVideo.py`), and I have not yet pinned the exact line** —
every specific mechanism I could name and test (MD5 cross-language dedup,
the `StreamSize`/`Duration==0` skip-guard, `iso639` resolution failure, the
`max_stream=80` cap) fires clean on this real data and does **not** explain a
selective 4-language drop. This is a **finding against a frozen file**, per
`BRIEF.md`'s own framing — not mine to edit
(`src/mergeVideo.py` is frozen except two named tagged zones, neither of
which is any of the functions above).

**Also structural, and worth flagging on its own:** every call site of
`generate_launch_merge_command` (`mergeVideo.py:2078`, `:2169`) sits inside
`mergeVideo.py` itself, and its own callers (`simple_merge_video`/
`sync_merge_video`, `merge_videos`) are called from `src/gestionar_show/
fusion.py` — explicitly frozen, "do not touch it"
(`WRITE_ZONES.MD` §1). **There is no open call site between the repair and
the file actually landing on disk.** This matters directly for Task 3: a
"verify the file that shipped" check cannot be inserted anywhere in that
chain from an open zone as it stands today — it would need either a new
owner-granted tagged zone, or a genuinely separate post-hoc pass (something
that reads the already-written delivered file after the fact, e.g. from CI
or a new open module triggered after `fusion.py` returns). I have not yet
identified whether such a post-hoc entry point exists; that is the next
question before I can act on Task 3.

## Task 2's answer: hypothesis REFUTED, with the adversarial tests the brief asked for

- **Counter-test passed as predicted, but it doesn't discriminate**: `vi`,
  `th`, `ko`, `ja` (also non-ASCII/non-trivial-script languages) resolved and
  shipped fine — same as the brief expected.
- **The actual awkward-ISO-639 mechanism never manifests anywhere I could
  find it**: `is_language()` and `Lang(...).pt1` return the exact same shape
  of answer for `ar`/`hi`/`id`/`ms` as for `en`/`fr` (tested directly, and
  via a real `mediainfo` reprobe of a synthetic mkv carrying 2-letter
  metadata language tags — mediainfo passes the tag through unchanged for
  all six, no special case for the "awkward" four).
- **Rival explored and refuted with real data, not reasoning**: cross-
  language content-MD5 collision (the natural "silent dedup" mechanism given
  `SPEC_ZONE_A.MD`'s own language about the shape of this defect family) —
  measured **zero collisions among the real 17 tracks**, both before and
  after retiming.
- **`max_stream=80` cap**: 23 total subtitle streams in the delivered file,
  nowhere near the cap (`max_stream-1-number_track_audio` ≈ 78). Ruled out.

**Conclusion: the "four awkward ISO codes" pattern in this one file is very
likely coincidence, not the mechanism** — Task 2's own instruction was to
treat a well-fitted hypothesis adversarially, and it did not survive contact
with the real content and the real frozen-code paths.

## What I dropped

- Did not pursue an ffmpeg/first-merge-pass integration run of the full
  `generate_launch_merge_command` (needs `audioRules`, `dict_with_video_
  quality_logic`, several `tools.*` globals I have not reconstructed) — it
  is the next concrete step to pin the exact line, but is a bigger lift than
  the per-function tracing above and I stopped to report this checkpoint
  first, since it changes what Task 3 even means.

## A separate, latent finding (Lead asked this be kept distinct from the
mechanism above — it is NOT the cause of id 12's loss, confirmed)

`generate_merge_command_common_md5` (`mergeVideo.py:1549`) calls
`generate_new_file` on the repaired object **without ever calling
`calculate_md5_streams_split()` first**. Every one of `repaired_obj`'s
subtitle tracks therefore carries `MD5=''` at the point `generate_new_file`
reads `sub['MD5']`. The code's own guard (`if sub['MD5'] != '':
md5_sub_already_added.add(...)`) never adds `''` to the dedup set, so
`'' not in md5_sub_already_added` is always true and the MD5-dedup branch
never blocks a repaired track on this path — **confirmed not the cause
here**. But it means the dedup set is running against a **uniformly-empty
hash for an entire class of input** (every repaired track, on every repair,
always) rather than being genuinely inert — one future change to either the
guard (`!= ''`) or to where `calculate_md5_streams_split` gets called would
silently activate cross-track collision on empty string for every repaired
subtitle in the fleet. Flagging as a latent defect, not touching it (frozen).

## Blocked / needs the Lead

1. ~~The exact frozen-code line~~ — **RESOLVED, see UPDATE above:
   `mergeVideo.py:1744-1746`, `config.json`'s `language_to_completely_remove`.
   Measured, exact match, language-for-language.**
2. **Task 3 still needs a place to stand.** No open call site exists between
   the repair and the delivered file landing on disk. Before I can build
   "the verifier counts the shipped file," I need to know whether a post-hoc
   check (reading the already-landed file, from outside the merge call
   chain) is an acceptable shape, or whether this needs an owner-granted
   zone. Lead is routing this to the owner directly (per their reply) — I am
   not designing it.
3. **New, owner-shaped question the line reveals**: is `ar`/`hi`/`id`/`ms`
   in `language_to_completely_remove` intentional policy (in which case the
   fix is purely "make the 17/17 line honest and stop rebuilding tracks that
   are already known to be discarded"), or should the repair itself consult
   that same config **before** doing 4 tracks' worth of wasted extraction/
   retiming/mux work? I have not assumed an answer either way — this is the
   owner's call, not mine.

Nothing committed, nothing edited in the repository. All reproduction lives
in `VMSAM_HELP_AI/dev-subs/lab/` (scripts +
`probe_task1_full_repair.py`'s output `full_repaired.mkv`); subtitle text
extracts stay there, never referenced by content here.
