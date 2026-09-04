# VMSAM — Write Zones

**What an agent may change, where, and how compliance is checked.**

This file is the authority. A campaign that violates it is not validated, whatever
its results. Read it before the first edit, not after a review.

This file says **what may be changed and why**. It says nothing about what to
build — that is `SPEC_ZONE_A.MD` and the campaign's own task documents.

Two rules govern everything below:

1. **Default is frozen.** A path that is not listed as open is closed. Silence is
   not permission.
2. **A zone is real only if it is checked.** Every rule here names the command
   that verifies it. A rule nobody can verify is a wish.

---

## 1. Frozen — no modification, ever

| path | why |
| :--- | :--- |
| `src/gestionar_show/` (all of it) | the API and the fusion worker: destructive paths, database writes, atomic file replacement |
| `src/main_gestionar_show.py` | the daemon, its loop and its two uvicorn instances |
| `src/audioCorrelation.py` | the delay measurement the whole pipeline trusts |
| `src/main.py`, `src/infinite_wrapper.py` | the other entry points |
| `src/tools/` | the database module |
| `src/video.py` | **except the two tagged functions below** — media handling, ffmpeg pools, MD5 threads: the part that touches disk and processes |
| `init.sh`, `run.sh` | production entry points |
| `Dockerfile` | the runtime image |
| `.github/`, `.forgejo/` | CI/CD |
| `AGENT.MD`, `WRITE_ZONES.MD`, `CAMPAIGN.MD`, `FILE_MAP.MD`, `SPEC_ZONE_A.MD`, `COMING_BACK.MD`, `COMMITTING.MD`, `TALKING.MD`, `MEASURING.MD`, `LEAD.MD`, `docs/`, `CAMPAIGNS/` | the rules that bind you, and the campaign archives — history is even less yours to rewrite than a rule. Not yours to edit — a wrong rule is a finding, see §6. **`AGENT_NOTES.MD` and `TASKS/` are yours**, and no other document is frozen |

`src/gestionar_show/fusion.py` deserves its own line: it replaces the master file
in the library and deletes the source. It has been reviewed and corrected — atomic
`os.replace`, episode lock, pool lifecycle, silent-failure trace. **Do not touch it.**

Need something from a frozen file? Do not edit it: open a question for the owner
and keep working elsewhere.

## 2. Tagged zones — the only writable places in shared engine files

`src/mergeVideo.py`, `src/video.py` and `src/tools.py` are frozen **except** between these markers:

```python
"""
BEGIN: AGENT modification ok
"""
    ...your code...
"""
END: AGENT modification
"""
```

**One modification outside a tagged zone invalidates the campaign.** Not the
commit — the campaign. This is deliberate: the cost has to be visible before the
edit, not after.

Existing zones:

| file | function | purpose |
| :--- | :--- | :--- |
| `mergeVideo.py` | `remove_not_compatible_video` | **zone A** — repair of rejected files |
| `mergeVideo.py` | `adjust_delay_to_frame` | **the whole function is open**, not only its inner tags: frame snapping, and the VFR case that is still a TODO |
| `tools.py` | module level | shared helpers for the above |
| `video.py` | `get_best_quality_video` | the quality vote that names the reference — verify it really returns the best video |
| `video.py` | `get_good_frame` | PSNR frame scan; already carries a `float()` coercion fix, see below |
| `mergeVideo.py` | `keep_best_audio` | the `fabricated` rule — inert until the repair sets the key (`SPEC_ZONE_A.MD` §4) |

Adding a new zone is an owner decision, not an agent one.

### Two tag forms — the string form is not always legal

The `"""BEGIN: AGENT modification ok"""` block is a **statement**, so it is illegal
wherever a statement is: between two `elif` branches, between `except` clauses,
between a `try` and its `finally`. Use the comment form there:

```python
            # BEGIN: AGENT modification ok
            elif ...
            # END: AGENT modification
```

Both forms carry the same text, so the checks below find either. `keep_best_audio`
uses the comment form for exactly this reason.

### A tag placed after a `return` is a dead zone

`adjust_delay_to_frame` currently contains one: the `return` on the CFR branch
short-circuits the tag that follows it, so anything written there never runs.
That is the campaign-1 defect — code that nothing calls — built into the tag
structure itself. It is harmless here only because the whole function is open,
so restructure it rather than writing inside the dead block.

Check before writing in any zone:

```sh
python3 - <<'PY'
import ast
src = open('src/mergeVideo.py').read()
for n in ast.walk(ast.parse(src)):
    body = getattr(n, 'body', None)
    if not isinstance(body, list):      # lambda/IfExp carry an expression here
        continue
    for i, st in enumerate(body[:-1]):
        if isinstance(st, ast.Return):
            rest = [x for x in body[i+1:]
                    if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant))]
            if rest:
                print("unreachable after return line", st.lineno, "->", [x.lineno for x in rest])
PY
```

### Two lines inside open zones that are not free to remove

`get_good_frame` opens with `calculated_delay = float(calculated_delay)`, and the
two frozen fallbacks in `video.py` re-raise `KeyboardInterrupt` / `SystemExit`
before returning their fabricated value. Both fix defects that produced **silent**
wrong behaviour. Removing either is allowed — understanding what it protected
first is not optional. See `SPEC_ZONE_A.MD`.

## 3. Configuration

**`config.ini` — additions only.** New lines may be added and are then freely
editable by the agent that added them. **Existing variables are frozen**: not
their value, not their name, not their position. The merge rules encode measured
codec relationships; changing one silently changes every merge.

**`*.json` (`config.json`, `param_template.json`, `titles_subs_group.json`)** —
frozen unless there is an imperative need, which means an owner decision recorded
before the edit.

## 4. Open

`src/frame_compare.py`, `src/get_cut_time.py`, `src/merge_video_unused.py`,
`src/folder_generator.py`, `src/gestionar_movie/`, and **new runtime modules** —
code that ships and runs in production.

**`src/merge_video_chimeric.py` is fully editable, and its author is its master.** The
owner's ruling, and it is written here because three agents spent a day treating a
defect in it as a blocked need. `src/merge_video_repair.py` and
`src/merge_video_resample.py` are structurally identical — campaign-2 modules, absent
from `master`, authored by the same agent — and are treated the same. **The owner named
the first explicitly; the other two follow by the same reasoning and not by his word.**

**A file becomes shared by being useful and imported, not by anyone deciding it is.**
So the only defence against backing into a standard is **the file saying what it is
not.** An agent asked to build a fleet harness drew the line itself and drew it better
than the request: a lab instrument in its own tree is inside its charter; *a thing three
other agents import is shared code in function even where it is not in location, and I
do not own fleet tooling.* It shipped the tool with the disclaimer in the first
paragraph of its docstring — not fleet infrastructure, not a standard, anyone may read
it, **nobody should import it expecting it to be maintained for them.** That is a guard
in the file rather than a rule in a document, which is the only kind that fires at the
moment it is needed.

**A module is yours; the schema it emits stops being yours the moment someone reads
it.** Open means you may add, change and remove freely — and that freedom ends at the
fields another agent's code consumes. Adding a field and removing it twenty minutes
later were both within one agent's rights, and both would have broken its consumer
(`vmsam-dev-1`, which reported it rather than discovering it downstream). **Say what
you are changing in an emitted schema, to whoever reads it, before you change it.**
That pair handled it by talking; a successor pair may not have the conversation.

### What does not go in the repository — the owner's line

> **Anything that belongs specifically to the test environment stays in the test
> environment. The repository is for the code, and for what relates directly to the
> code.**

**This closes a grievance that was open all day, and it closes it against the asker.**
An agent held 57 KB of privacy-clean measurement tables — no path, no title, no
catalogue id — outside the repository under this section, and reported that their only
copy and its backup sit on one physical device, so **what survives is a snapshot and not
a history**: the sequence of corrections exists only where an author narrated it in
prose. The archive is not unprotected — six snapshots, two real recoveries — and the
grievance was never that. **The ruling stands regardless: measurement tables are test
environment, and versioning them is not a reason to move them.**

**The test for this section is therefore not "is it clean" or "is it valuable".** Those
tables are both. It is *does this relate directly to the code* — and a table describing
a corpus does not.

### What does not go in the repository

This repository holds **VMSAM's code**. Nothing else. Batch scripts, verification
harnesses, corpus sweeps, forensic reports, per-agent state files, task archives,
scratch analysis: `VMSAM_HELP_AI/`, beside the repository, never inside it.

Campaign 1 is the measurement. It left behind, under `src/`:

| directory | files | lines |
| :--- | ---: | ---: |
| `diagnostics/` | 624 | 88 394 |
| `tools_batch/` | 173 | 47 349 |
| `task_archive/` | 101 | 13 030 |
| **total** | **898** | **148 773** |

VMSAM itself is 7 225 lines of Python and 569 of Rust. The scaffolding outweighed
the program **nineteen to one**, and someone opening the repository could no longer
tell what VMSAM *is*.

The test, applied to every file you are about to create: **would this still matter
to someone running VMSAM in production?** If it exists to help *you* work — to
measure, to sweep, to remember, to report — it goes beside the repository.

### Files with no importer

Open like the rest, listed because their state is not obvious from reading them.

| file | state |
| :--- | :--- |
| `frame_compare.py` | pHash/DCT frame comparison. Compiles; the `da1c191` breakage was repaired in campaign 2. |
| `get_cut_time.py` | multi-range cut detection, drift, audio chimera. No importer; imports `frame_compare`. Homonym: `mergeVideo` used to define a *class* of the same name, now in `merge_video_unused.py`. |
| `merge_video_unused.py` | `find_a_cut_for_not_compatible` and the old `get_cut_time` class, extracted from `mergeVideo.py`. No importer. |

New capability goes in **its own module**, called from a tagged zone, with a test
that pins the call site.

**It must not be conditioned on a parameter.** VMSAM runs with nobody watching —
that is the premise the whole design rests on, and it is why a wrong automated
decision is destructive. A repair that waits for a human to switch it on is not a
repair: it is a feature that will never run, on a service where nobody is there to
run it. Campaign 2 measured its own version of this without naming it — *"zone A is
measurably inert with its flag off"* is a true statement about work that reaches no
library.

**The gate is the measurement, not a setting.** A capability decides for itself, from
what it measured, whether it should act — and when it cannot establish that it
should, it **declines**, which is already a correct outcome here. That is the
difference between a decision procedure and a deferral to somebody who is not
present. The campaign has the example: error id 237 was refused on a median fidelity
of 0.576 with 18 sign flips, not on a flag.

**Reimplement an idea; do not copy the code.** Comparable projects exist and reading
them is encouraged — the search strategies are worth having. But an idea described in
a README and implemented yourself is unencumbered, and copied lines are not.

**If you do copy anything, say so in a comment at the copy**, naming the project, the
file, the revision, and its licence. That is required and it is not sufficient: a
comment records provenance, it does not grant permission.

Licences as declared by the projects surveyed:

| project | licence |
| :--- | :--- |
| `ffsubsync` | MIT |
| `tympanix/subsync` | Apache-2.0 |
| `alass`, `sc0ty/subsync` | **GPL-3.0** |
| `VideoDiff` | **GPL-2.0** |
| `video-offset-finder` | **none declared** |

The last row is the strictest, not the most permissive: **no licence means no
permission to copy at all** by default. The two permissive ones still require their
notice to travel with the code. And VMSAM carries no `LICENSE` file of its own, so
what copied copyleft code would oblige is an owner question, not an agent's — which
is the whole reason to reimplement instead and never reach it.

**A sub-option of an unconditional capability is a condition on it.** The ruling
above names a capability and its flag; it says nothing about the options *inside* a
capability whose flag is already gone, and every key removed after the two headline
ones was of that kind. Anything that can make the capability **not act, or act
unchecked**, is the same defect one level down — a plan supplied by a human is the
deferral in different clothes, a switch disabling the only source makes the capability
inert, and a verifier you can turn off is not a safety. Found by `vmsam-dev-2`, which
applied the ruling's reason one level down three times while the document merely
failed to contradict it.

**A flag may exist while a capability is being built**, so that half-finished
behaviour does not reach a library — and **removing it is part of finishing**. It
belongs in `config.ini`, never in the environment: `run.sh` passes no `--config` and
the image bakes its variables into the frozen `Dockerfile`, so a new `VMSAM_<NAME>`
is absent at runtime and cannot be turned on at all. Campaign 1 measured the
consequence — **the code read 18 `VMSAM_` names and the image set two** — so every
feature built to a ruling that day was unreachable by construction. `config.ini`
additions are allowed by §3; a scaffolding flag left behind at the end is not.

**A removed `config.ini` line is judged against `origin/master`**: a line master
carries stays frozen, a line this campaign added is this campaign's to take back out.
That satisfies §3 and §4 both rather than overriding one — found while removing the
flags, and it is why the removal needed no ruling.

## 5. Verification

**Run the write-zone checker** — `VMSAM_HELP_AI/tools/check_write_zones.py`, one
command with one exit status, and **exit 2 means it could not run, which is not a
pass**. It is the authority; the snippets below are what it replaced, kept because
knowing what they missed is the point.

**Why reading a diff by eye is not enough.** `git diff --cached -U0` shows changed
lines, and a reader maps them onto the zones of the file **as it now is**. A pure
deletion has no new-side line at all — so removing a frozen line *above* a zone reads
as "nothing added outside a zone" and passes. Removals have to be mapped onto the
**old** blob's zones and additions onto the new one's, separately. Two further holes
close the same way: moving an `END` marker down swallows a line into a zone while the
diff lands inside it and looks legal, and a fresh marker pair grants a zone that §2
reserves to the owner — so any change in marker count fails.

Found by the campaign-2 lead against this section, which its author wrote.

**What the checker covers, in prose, because a code block gets obeyed as a
procedure.** This section used to print the superseded snippets under a paragraph
saying nobody runs them — and `vmsam-ci` reported the obvious: *an agent under time
pressure reads a code block as a procedure, and the paragraph explaining that it is
not one is above it.* So the reasoning stays and the block is gone. The checker
verifies that nothing is written outside the open paths, that no governing document
is rewritten, that every changed line in a tagged file sits inside a zone with the
marker count unchanged, that `config.ini` only gains lines, and that the tree
compiles.

**The tree must compile cleanly — there is no allowance, and this paragraph used to
say the opposite of itself.** It granted `frame_compare.py` an expected failure in
its last sentence while stating in its first that campaign 2 had repaired it. Two
other documents carried the same stale claim. Measured on this revision:
`python3 -m compileall -q src -x '__pycache__'` exits **0**, clean. `vmsam-dev-1`'s
fix is what made the old allowance false, and it reported that as a finding rather
than editing it out. **A document that excuses a failure the code no longer has will
excuse the next real one** — `vmsam-ci` read the allowance and left a compile error
alone on its authority.

**Run Rust's checks when you touched Rust**: `cargo check && cargo clippy &&
cargo fmt --check`.

## 6. Reporting a blocked need

A frozen file that stands in your way is a finding, not an obstacle to route
around. Record what you wanted to change, why, and what you did instead. A
campaign that ends with three well-argued blocked needs is worth more than one
that ends with three undeclared edits.

### A defect in code you may not write is a finding, and it has a route

**Report it. Do not fix it, and do not keep it.** You read frozen code all day and
you cannot write it, so a defect you find there has no natural outlet — it goes into
your own notes and reaches nobody who can act. Three agents named that independently
on the same day, unprompted, and one had been carrying a reconciliation-window
finding since that morning. Until it was asked for, **the campaign had no channel for
the defects only its agents could see.**

Send it to the architect, who carries it to the owner. Each one carries:

- the **file and the function**, verified against the revision you are on, not
  remembered. **Name the function; give the line as a convenience, not as the
  reference.** A line number goes stale within a day on a moving branch — one report's
  `:388`, correct when written, was `:436` by the time it was read, because the file
  grew with that evening's promotion.
- **no permission label. None.** Do not write *"in frozen code"*, *"blocked"*, or
  *"not mine to touch"*. **A permission is a property of (file, master state, actor),
  and the reporter is usually none of the three** (`vmsam-forensic`). The report
  carries the **address**; the actor derives the permission at the moment of acting.
  Three agents in one day filed editable defects as blocked — the label was supplied by
  the way the question was asked, and they stopped deriving one. **A wrong address
  announces itself the moment someone opens it; a correct address with an inherited
  label reads as fully verified and sits waiting for a ruling it never needed.**
  Removing the label also dissolves the other variant: a wrong address can only acquire
  the wrong file's permissions *because reports carry permissions.*
- **and be careful what you derive it from, because there are two masters.** The hub
  serves agents its **local** `master`, which lags its own view of the true upstream.
  Measured: agents reach `fe0c34e`; the upstream tip was `71bb165`, nine hours later,
  and **not in any agent's object store**. It cost nothing that day — the divergence was
  documentation-only, 6 files, 560 insertions, no `src/` — but **the first time a
  campaign-2 module is merged upstream it freezes, and no agent clone can see the freeze
  while the hub's local master lags.** Four agents would go on editing an
  editable-looking file, deriving the answer correctly from the only reference they
  have. The fix is the hub fast-forwarding its local master, not a rule that writes a
  known lag into itself.
- **what goes wrong and under which input** — a defect with no triggering condition is
  a worry, not a finding;
- **whether this is an observation from a run or a reading of the code.** Say it
  plainly — **and note what the label does *not* cover.** It marks the epistemic status
  of the **claim**, never of the **address**, and one report's items marked *reading*
  were the ones its author had verified least: the label that was supposed to warn of
  uncertainty sat on the item checked last (`vmsam-dev-2`, on its own list). **Verify
  the address as separately as the content.** That report had a true statement about
  real code at a false file — the right line number in the wrong module, which is the
  one combination that survives a careless check, because both files carried plausible
  code at that line. The author had written the name of the module it had been tracing
  *through* for an hour rather than the one it was reading. *A labelled hypothesis beats silence; a hypothesis wearing an
  observation's label is worse than silence* (`vmsam-ci`) — it spends credibility the
  next reader has no way to audit. The campaign lead mislabelled one of its own that
  day, ci checked before agreeing, and the label was corrected within minutes. That
  is the mechanism working.
- **observations consistent with a defect are not evidence of it.** Repaired tracks
  were seen losing to the master's own track, which is the correct outcome because the
  master is the timeline. Same observation, and it settles nothing about the defect
  next to it.

**When one of your findings turns out to have been invalidated by a property of your
METHOD, re-check every other finding that used it.** Not the ones that share a subject —
the ones that share the instrument. A grep that cannot see a syntax tree, a query over
one record structure, a harness reading a different metadata authority than production:
each produces a FAMILY of small wrongnesses, and the family is invisible until someone
sweeps it. One such sweep this campaign found a published band excluding one of its own
members by 0.003 — nobody would have gone looking for that, and a successor would have
found it first.

**Report the ones you decided were not your problem, and the ones you are unsure of.**
Both categories produced real findings here. And ask the question this way round:
**not "what is wrong in `src/`" but "what happens to what I produce after I hand it
over"** — `vmsam-dev-2` located three defects in frozen code only by tracing where a
repaired track goes, and none of them by inspecting the module. **The defects in
frozen code an agent can see are the ones on the path its own output takes.**

## 7. Artefacts another agent reads

**A file one agent writes and another parses is a seam, and a seam is nobody's.** Three
defects turned up at one seam in one hour, **every one a disagreement about what a name
means rather than about what a value is, and none visible from either side:**

    reader's spec    "<hash>\t<status>"                    -- two columns
    writer produced  file size_bytes duration_s mtime_utc status reason
                     and FOUR statuses instead of two
    reader's code    took $2 -- WHICH IS size_bytes -- as the status

**It matched no class, would have resolved every file to `NOT_EXAMINED`, and would have
evicted a 2.7 GB refuted artefact under the ordering built to protect it.** Both halves
passed their own tests: the reader parsed correctly by its own spec, and the writer's
file was correct and *better* than the spec asked for. **The failure was entirely in the
join, and a third party running one artefact against the other is the only position from
which "column 2 is the size" is visible.**

The entry point was not the reader. It was the spec:

> ***My spec was the defect. I asked for two columns, the other agent wrote six with
> more information in them, and my positional read broke on the improvement***
> (`vmsam-ci`).

**Nobody guards against improvement**, and a spec that under-asks is how an improvement
becomes a hazard. Third instance of that class in this campaign, first with an artefact
at stake.

**Two rules, and they are here rather than in a checklist because a rule that depends on
someone noticing a seam exists is the control-that-cannot-fail wearing a virtue.** The
first proposal for this section was *an obligation on whoever reads both sides* — which
is a check that happens when one agent feels like it, i.e. the defect restated as its
own cure.

**1. No positional reads across an agent boundary.** A cross-agent artefact carries a
header line, and readers resolve columns **by name**. With a named read, six columns
where two were specified is a **silent improvement** instead of a near-eviction. This
removes the mechanism; it does not add a watcher.

**2. The consumer runs its reader against the producer's real bytes before either side
calls the integration done.** Not against the spec — **against the file.** The consumer
owns the join because *the consumer is the one making a claim about what the producer's
bytes mean*, and placing it there means it fires automatically at the next seam instead
of waiting for someone to notice there is one.

**Two worked failures of rule 2, both by agents who believed they had tested the join.**
A lead's cross-check returned **24 of 24 correct** because both sides returned `None` on
abbreviated keys — *a join returning zero is an empty search with two vocabularies, not
"nothing matched" but "nothing matched when I compared THESE keys to THOSE"*. And
`vmsam-dev-2` built a consumer against a contract's **shape**, correctly, and only saw
the truth on running the real producer: *the shape was right and I had not seen a byte
of it.* **Do not let anyone substitute a shaped dict for the bytes, including the person
who wrote the contract.**

**A CONSUMER DOES NOT AVOID A SEAM. THE CONSUMER *IS* THE SEAM.** The claim that a
pure-consumer role adds no join was put forward by the architect and refuted the same
hour, with counts:

    a retainer      consuming a status file      took column 2 (size_bytes) as the status
    a validator     consuming repair logs        hit the wrong-key defect TWICE
    a lead          consuming locator artefacts  24 of 24 correct, comparing two Nones
    the same validator, consuming a sweep directory that exists only because those files
                    were DECLINED, and reading a rate off it

**Four consumer joins, four defects, one day.** Reading rather than writing does not put
you outside the join — **it puts you on the side that makes a claim about what the other's
bytes mean**, which is the whole reason rule 2 sits with the consumer.

**And a renderer is the sharpest case, because its failure mode is a FALSE DEFECT
REPORT.** *A renderer handed a field it does not understand draws a blank cell and calls
it "the producer did not emit this."* On a log whose format changed seven times in one
day, that is a defect filed against an innocent producer.

> **A field your reader cannot find must be distinguishable from a field the producer
> genuinely did not emit. Today those are the same blank cell.**

**So: confirm the field is absent from the EMITTED BYTES, not merely absent from your
reader, before filing anything** (`vmsam-dev-sandbox`, naming its own error of four hours
earlier as the one a new consumer will repeat on day one unless told).

## 8. Quoting a log line is the fleet's widest privacy exit

**A CONTROL THAT DEPENDS ON A PATTERN MATCHING ADVERSARIAL TEXT IS NOT A CONTROL.**

An agent building the merge-plan report collected non-`repair:` lines deliberately —
*a skip is a decision and an omission reads as "there weren't any"* — and wrote a pattern
redactor to strip the paths out of them. **It did not hold, and the reason is one
character:**

    the line:   ['/srv/.../<two words>/<three words> {tvdb-...}/...'] not compatible
    THE PATH CONTAINS SPACES, so the pattern terminated at the first space and
    the second half of a series title walked through intact.

**Measured, not reasoned.** *And the fix is not a better pattern.*

> **WHERE THE INPUT IS FREE TEXT YOU DO NOT OWN, WITHHOLD IT RATHER THAN SANITISE IT.**

The row now carries **the FACT** — a digest, a character count, `carries_path`, an
alphabetic tail phrase — **and never the content.** Enough to locate the line in a log
that does not travel. **Three layers, and only the first is a guarantee:**

    1  NEVER EMIT UNVOCABULARIED FREE TEXT              <- the actual control
    2  opaque-token redaction on every other value      <- defence in depth
    3  a final re-read of the finished document that RAISES

**Layer 3 raises rather than corrects, deliberately: a silent fix hides the next leak.**
*And it caught something immediately — itself: `</title>` matched a single-segment path
pattern, which would also have mutilated `verified=1/1` and `master/ja`. A path now
requires two segments* (`vmsam-dev-4`).

**AND THE HAZARD IS NOT THAT AGENT'S. IT IS EVERYONE'S.**

**Every agent here quotes log lines into reports and into messages, and frozen
`mergeVideo.py` writes real library paths into the log in twelve places by design.**
*Anyone pasting a non-`repair:` line is one space-containing path away from the same leak,
and a pattern redactor will not save them.*

    QUOTE A LOG LINE ONLY IF EVERY FIELD IN IT COMES FROM A VOCABULARY YOU OWN.
    Otherwise report the FACT of the line -- its shape, its length, where to find it --
    and leave the line where it is.

*This is `satisfy a rule by construction` applied to redaction, and it runs the opposite
way to the usual case: elsewhere, construction lets you EMIT something and guarantee a
property. Here the only construction that guarantees anything is to NOT emit.*

**A CROSS-CHECK DELETED THE MOMENT IT FIRST AGREES IS A CROSS-CHECK THAT NEVER RUNS
AGAIN.** A report derives each region's offset from cut bounds *and* reads the value its
producer emits; **the emitted value takes precedence and the derivation still runs beside
it, printing both on a disagreement.** *Most people delete the old computation the moment
the new one agrees — which is exactly when its value begins, because from then on it is
the only thing that can catch a producer changing underneath.*

**THE EXAMPLE THIS SECTION FIRST USED FOR THAT RULE WAS WRONG, AND ITS AUTHOR CORRECTED
IT AGAINST HIMSELF.** The two computations were reported as independent — *twenty offsets
across three files and nine tracks agreeing to the digit.* **They are not independent:**

    cut_regions and used_regions are appended IN THE SAME `for piece in pieces:` LOOP,
    FROM THE SAME `source_start`.

> ***Two instruments that cannot disagree are one instrument reported twice.***

**So the twenty-offset agreement measured nothing about the offsets. It measured that the
consumer's parser reads the producer's format correctly** — *which is worth having, and is
a different claim* (`vmsam-dev-2`, checking its own code after the paragraph praising it
had been promoted; `vmsam-forensic` for the rule). **The practice stands; this pair is not
an instance of it.**

## 9. Two copies drift, and the drift is invisible from inside either file

**Owner instruction:** *réutiliser au maximum les mêmes fonctions. Un code de plus de
cinq lignes, présent deux fois à deux endroits différents et qui a le même but, devient
une fonction.*

**The trigger is mechanical — more than five lines, twice, same purpose — and the reason
is not tidiness.** *Two copies do not stay identical. They drift, and the drift is
invisible from inside either file, because each one is correct where it sits.*

=== THE WORKED EXAMPLE IS IN THIS CAMPAIGN'S OWN HISTORY, ON BOTH SIDES ===

**Applied, and the motive was measured:** subtitle kinds were classified in more than one
place with more than one vocabulary, **so two call sites could disagree about the same
track.** One classifier now answers and the callers ask it.

**Violated — and this is the instance, verified line by line rather than characterised:**

    mergeVideo.py:1486-1495   generate_merge_command_common_md5
    mergeVideo.py:1502-1511   generate_merge_command_other_part

**Normalise the delay variable name and ONE LINE DIFFERS — the one computing the delay.
NINE ARE IDENTICAL:** the same `generate_new_file` call, the same `if number_track`
branch, the same `["-A","-S","-D","--no-chapters"]`, the same `if delay != 0` guard, the
same `--sync -1:{int(round(delay))}`, the same fallback `extend`, the same `stdout` line
at the same `.3f`.

**And the drift hazard is named rather than hypothetical: one copy is the path A REPAIRED
OBJECT REJOINS THE MERGE THROUGH, the other is the ordinary MD5-group path.** *A fix
applied to the path someone happens to be working on, in a campaign where one agent owns
the repair, is exactly how two correct-in-place copies come apart.* **They agree today,
which is why it is worth reporting before they do not** (`vmsam-dev-sandbox`; §6 shape,
frozen zone, reading plus a mechanical diff, not observed causing harm).

**AND THE EXAMPLE THIS SECTION FIRST CARRIED WAS AN INSTANCE OF THE COUNTER-RULE, NOT OF
THE RULE.** The architect offered the two frame snaps — `get_good_frame` and
`adjust_delay_to_frame` — and both agents who measured them refused it:

    the quantise arithmetic   3 lines against 4.  THE TRIGGER DOES NOT FIRE.
    the 11-line block         chooses frame_rate_use TO MATCH the fps= filter
                              applied to both streams -- the rate choice and the
                              filter are ONE decision, and it belongs to the
                              COMPARISON rather than to the delay

**Run the counter-rule properly and merging them is wrong in BOTH directions:** *the PSNR
search would be wrong to put into a function that has no search; the rate SOURCE would be
wrong to leave out of either.* **They share one micro-step — turning a declared rate into
a frame period — and diverge above it.**

**And the real defect is better stated without duplication at all:** *two different rate
policies are applied IN SEQUENCE to one value and neither knows about the other, and the
remedy is that the second should not run after the first* (`vmsam-dev-3`,
`vmsam-dev-sandbox`). **Shipping the snaps as the headline example would have taught the
fleet to merge precisely the pairs the counter-rule forbids, using this section as
licence.**

=== THE COUNTER-RULE, WHICH MATTERS AS MUCH ===

> **RESEMBLANCE IS NOT PURPOSE. Two blocks that look alike and answer DIFFERENT questions
> must stay apart.**

**Merging by resemblance is how a name comes to mean two things, and this campaign has
paid for that repeatedly in one day:** *two dicts both carrying a field called `outcome`;
a `MOVIE/ENCODER` tag inherited from a source file wearing the output's field name;
`codec_delay` and `start_time` both called "encoder priming".* **Every one of those is a
merge that happened in someone's head rather than in the code, and it produced the same
damage.**

*So the test is not "do these look the same". It is: **would a change to one of them be
wrong to leave out of the other?** If yes, they are one function. If no, they are two
things that resemble each other.*

=== WHEN IT FIRES, AND WHERE IT DOES NOT ===

**It fires at the moment you write the SECOND copy** — *if you are about to write it a
second time, it was a function the first time.* **That is cheaper than any detector and it
is the only moment both copies are in one person's head.**

**And it does not authorise touching frozen code.** *Where this rule points at a duplicate
inside a frozen zone, the output is a defect report under §6 — file, function, triggering
condition, epistemic status — and not an edit.* **Two of the clearest instances in this
tree are frozen, which is exactly why they survived to be found.**

=== TWO ARGUMENTS STRONGER THAN ANY EXAMPLE, BOTH FROM AGENTS APPLYING IT TO THEMSELVES ===

**FOUR COPIES OF ONE STATISTIC, ALL AGREEING NUMERICALLY, AND THE DRIFT WAS SOMEWHERE A
DETECTOR CANNOT LOOK.** An agent found four copies in its own lab and measured before
unifying: **15 of 15 hypotheses, every copy agreeing.** *The drift was not in the numbers.
It was in what they KNEW* — **the statistic that SELECTS a winning hypothesis and the
statistic that VALIDATES whether a split is real were two copies, and only one of them
knew a split can be spurious.** *A hypothesis could have won on a split that explained
nothing, and a drift detector looking for numerical disagreement finds nothing there.*

> ***The copies agree today, and the rule is not about today. The reason it has not bitten
> is luck, not design*** (`vmsam-dev-1`).

**AND THE COUNTER-RULE FIRING TOWARD INACTION IS WHAT PROVES IT IS LOAD-BEARING.** Another
agent applied the same test to three label-placement blocks in its own code **and concluded
AGAINST refactoring**: the axis version must never drop a tick, the amber version may fall
back to a shorter string, the salmon version clamps to the plot edge. *Three resemblances,
three different questions.*

> ***A rule that only ever fires toward action is not being applied, it is being obeyed***
> (`vmsam-dev-4`).

**Two agents ran the same test on their own code within one hour and reached opposite
answers, both correctly.** *That pair is the best evidence this section has that the
counter-rule is doing work rather than softening the rule.*
