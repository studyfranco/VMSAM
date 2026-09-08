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

## 0. WHAT THE REPOSITORY IS FOR — OWNER'S RULING, 2026-09-07

***THE REPOSITORY CARRIES VMSAM AND THE RULES FOR WORKING ON IT. NOTHING ELSE.*** *The owner's
words:* **"pas de code inutile a VMSAM dans le repo. les outils de tests c'est dans le folder local
adequat."**

    IN     the engine, its configuration, its build, and the governing documents.
    OUT    *** TEST TOOLING. HARNESSES. INSTRUMENTS. FIXTURES. PROBES. LADDERS.
           FORCING SCRIPTS. GATES. SCRATCH FILES. WORKING NOTES. ***
           These live in the SEAT'S OWN LOCAL FOLDER and publish to
           `refs/agents/<seat>`. They do not take a path in this repository,
           and a directory named after a seat does not make one for them.

> ***A PROBE NEVER RUNS AGAINST `refs/heads/dev-AI`.*** *An arm-reachability test — sabotage the
> predicate, check the verdict moves — is CORRECT METHOD and it belongs in a throwaway clone or a
> scratch ref you delete.* **ON 2026-09-07 ONE WAS RUN BY PUSHING A 2,054-LINE FILE TO THE SHARED
> BRANCH AND REVERTING IT FOUR MINUTES LATER.** *Any seat that fetched inside that window held a
> file that does not exist, its own tree checks disagreed with the authority, and nothing
> explained why. The branch is the one object every seat treats as ground truth; it is the worst
> possible place to prove a guard fires.*

**AND THE MEASUREMENT THAT PROMPTED THE RULING, TAKEN AT THE AUTHORITY'S BLOBS:**

    src/  THE ENGINE ................. 20,692 lines   35 files
    root *.MD  governance ............  8,981 lines    9 files
    docs/ ............................ 15,251 lines   22 files
    CAMPAIGNS/ ....................... 20,216 lines   14 files
    *** APPARATUS 44,448 AGAINST 20,692 OF ENGINE -- 2.15x ***

***AND `src/` IS CLEAN: NO HARNESS, NO FIXTURE, NO INSTRUMENT.*** *The seats have kept their
tooling out. **THE WEIGHT IS PROSE, NOT CODE** — and the architect added 300 lines of it across
nine commits in a single night while the fleet landed two files. A rule written about an error is
cheaper than the error and it is not free, and a document nobody can finish reading has the reach
problem this campaign already named twice.*

## 0b. NO TAGS — OWNER'S RULING, 2026-09-07

***DO NOT CREATE A TAG ON THIS REPOSITORY.*** *The owner's words:* **"tu peux eviter les tag ? il y
en a trop dans la branch qui n'ont aucun interets."** *Six existed and none marked a release or a
decision — every one was a safety pin from a rewrite somebody was about to perform.*

> ***A SAFETY PIN GOES TO `refs/backup/<name>`, NEVER TO `refs/tags/`.*** *This is not cosmetic.*
> **A TAG IS FETCHED BY DEFAULT: every clone in the fleet carries every pin anyone ever set, for
> ever. `refs/backup/*` is outside the default refspec** — *it holds the identical object, the
> authority still serves it, and nobody drags it around unasked.*

**HOW THE SIX WERE RETIRED, and the order matters because five of them were the SOLE ANCHOR of
their commit — the tag was the only thing holding that history:**

    1. MEASURE FIRST. For each tag, is its commit reachable from any branch, backup ref
       or agent ref? *** FIVE WERE REACHABLE FROM NOTHING ELSE. *** And is its TREE
       duplicated on a live branch? *** NO. All six trees were unique content. ***
       Two tags shared one commit, so exactly ONE was free to delete.
    2. ANCHOR, THEN VERIFY. Push each commit to `refs/backup/<name>`, then read it back
       from the AUTHORITY object by object and compare ids. *** DELETE NOTHING UNTIL
       EVERY ANCHOR VERIFIES. ***
    3. ONLY THEN DELETE, tag and local copy together.
    4. CONFIRM AFTER: tags 0, and every retired commit still served by the authority.

> ***"CLEAN THIS UP" AND "LOSE NOTHING" ARE THE SAME INSTRUCTION, AND THE SECOND ONE DECIDES THE
> ORDER.*** *A deletion that turns out to be safe and a deletion that was PROVEN safe before it ran
> are indistinguishable afterwards — and only one of them is a method.*

## 0c. BEFORE DELETING ANY REF, MEASURE THE DELETION AS A SET

***PER-REF UNIQUENESS DOES NOT SUM.*** *Each ref's "unique" count is taken with the others still
present, so they mask one another.* **FIVE REFS WHOSE INDIVIDUAL FIGURES TOTALLED 156 ORPHANED 226
WHEN DELETED TOGETHER.**

> ***AND PRESENT IS NOT REACHABLE IS NOT PRESERVED.*** *Deleting a ref destroys no object; it moves
> commits to PRESENT-BUT-UNREACHABLE, which survives only while the abstention from `gc` holds — and
> an abstention is an agreement, not a property of the store.* **A REF THAT PINS NOTHING COSTS ONE
> LINE IN A LISTING; THE COST OF BEING WRONG IS UNBOUNDED.**

*Derivation retired 2026-09-08: it sat beside the counting MISTAKE instead of beside the COUNTING,
and two seats re-derived it the hard way while it was landed and correct. Found by `arch-heir`,
which needed it, missed it, and measured the miss.*

## 1. Frozen — no modification, ever

*Read this table with the markers, never with a heading range:* `sed -n '/FROZEN-TABLE:BEGIN/,/FROZEN-TABLE:END/p'`.
***THREE INSTRUMENTS FAILED ON THE SAME NIGHT BECAUSE THIS DOCUMENT HAD NO MACHINE-READABLE
BOUNDARY AND EACH READER RECONSTRUCTED ONE:*** *a heading range overran into the prose below and
scored `merge_video_chimeric.py` as FROZEN — matching the sentence that says it is FULLY EDITABLE;
the `why` column was read as a domain description and a frozen file was assigned from it; and a
whole-document pattern returned 15 frozen paths against 7.* **THE REPAIR IS A CLAUSE RANGE, NOT
MORE CARE FROM THE READER.** *Named by the Lead across three seats' instruments.*

<!-- FROZEN-TABLE:BEGIN -->
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
| `AGENT.MD`, `WRITE_ZONES.MD`, `CAMPAIGN.MD`, `FILE_MAP.MD`, `SPEC_ZONE_A.MD`, `COMING_BACK.MD`, `COMMITTING.MD`, `TALKING.MD`, `CAMPAIGNS/` | the rules that bind you, and the campaign archives — history is even less yours to rewrite than a rule. Not yours to edit — a wrong rule is a finding, see §6. **the ROOT `AGENT_NOTES.MD` — `./AGENT_NOTES.MD`, not `CAMPAIGNS/01-dev-AI/AGENT_NOTES.MD` — and `CAMPAIGNS/02-dev-chimeric-resample/TASKS/` are yours**, and no other document is frozen. Every name in this row is a PATH from the repository root: `COMING_BACK.MD` means `./COMING_BACK.MD`, and the copy under `CAMPAIGNS/01-dev-AI/` is frozen by the `CAMPAIGNS/` entry, see §1c |
| `docs/` — including `docs/MEASURING.MD`, `docs/ENTRY.MD`, `docs/LEAD.MD`, `docs/PIPELINE.MD` | **UNFROZEN 2026-09-07 ON THE OWNER'S DIRECT INSTRUCTION**, for the corpus reorganisation that moved every `.MD` absent from `master` into `docs/`. Editing these is now permitted; the standing rules on privacy, measurement and reporting apply unchanged. The technical rulings already in `docs/` (`AUDIO_SPEED_POLICY`, `MERGE_CRITERIA`, `SUBTITLE_CODECS`) still carry the weight of a ruling — read one before contradicting it |
<!-- FROZEN-TABLE:END -->

### §1a — WHO MAY EDIT THE FROZEN GOVERNING DOCUMENTS, AND WHY THIS WAS MISSING

***THE ROW ABOVE SAYS "NOT YOURS TO EDIT — A WRONG RULE IS A FINDING, SEE §6". IT NAMES NO
EXCEPTION, AND ON 2026-09-07 THE ARCHITECT EDITED THOSE FILES FOURTEEN TIMES AND ITS APPRENTICE
EIGHT.***

    AGENT.MD ......... 8 architect · 4 arch-heir
    WRITE_ZONES.MD ... 6 architect · 1 arch-heir
    CAMPAIGN.MD ...... 0 architect · 3 arch-heir

**THE ARCHITECT'S AUTHORITY IS REAL AND IT IS THE OWNER'S, GIVEN AS A STANDING INSTRUCTION:** *"when
a governing rule is found wrong, correct `AGENT.MD` / `WRITE_ZONES.MD` / `CAMPAIGN.MD` /
`COMING_BACK.MD`, credit whoever found it, never edit code."* ***AND UNTIL THIS PARAGRAPH IT EXISTED
NOWHERE A SEAT COULD READ IT.*** *Every seat consulting §1 saw twenty-two edits to files declared
untouchable, by two seats, with no visible licence — and the correct reading of that document was
that a violation was in progress.*

> ***AN AUTHORITY THAT LIVES ONLY IN THE HOLDER'S OWN INSTRUCTIONS IS INDISTINGUISHABLE FROM NO
> AUTHORITY AT ALL.*** *A rule is not a control until it is where the reader is — the architect
> engraved that sentence tonight about a different rule, 456 lines into another document, while
> being the standing counter-example to this one.*

**SO, STATED WHERE §1 IS READ:**

    THE ARCHITECT MAY CORRECT `AGENT.MD`, `WRITE_ZONES.MD`, `CAMPAIGN.MD` and
    `COMING_BACK.MD` WHEN A GOVERNING RULE IS FOUND WRONG. It credits whoever
    found it, it never edits code, and *** THE CORRECTION IS THE ONLY THING IT
    MAY WRITE THERE *** -- not a narrative, not its own record.
    EVERY OTHER SEAT: §6. A WRONG RULE IS A FINDING, NOT AN EDIT.

> ***AND THE APPRENTICE'S EIGHT EDITS WERE NOT AUTHORISED BY ANYONE.*** *No document granted them
> and the architect never granted them. **THE ARCHITECT'S EXAMPLE WAS THE INSTRUCTION** — a seat
> watching its teacher edit frozen files hourly learns that the freeze does not mean what it says,
> and no message is needed to teach it.* **WHETHER THAT MANDATE EXTENDS TO A SECOND SEAT IS THE
> OWNER'S ALONE AND HE HAS NOT BEEN ASKED. UNTIL HE RULES, IT DOES NOT.**

*Surfaced by `vmsam-dev-4`, which went to the governing document while the Lead went to `git log` —
and the Lead's own words on that: **authorship is not permission**, and **a careful constraint on an
unauthorised action is camouflage, not a safeguard.***

> ***AND A SEAT'S AGENT REF PRESERVES ITS PUBLISHED ZONE, NOT ITS TREE.*** *`refs/agents/<seat>` is
> built from whatever the seat's lander archives — for one seat, `git archive HEAD <its own zone>` —
> so a file OUTSIDE that zone is not in the ref however carefully the seat publishes.* **THE
> ARCHITECT TOLD A SEAT ITS WORK WAS SAFE BECAUSE IT HAD PUBLISHED, AND THE FILE IN QUESTION WAS NOT
> IN ITS PUBLISHED ZONE AT ALL. The seat checked and said so.**
>
> *It cost nothing only because the seat's own measurement came back clean independently —* ***THE
> ASSURANCE AND THE MEASUREMENT WERE INDEPENDENT, AND ONLY THE MEASUREMENT SAVED IT.*** *Had the
> lines been unique they would have existed in one unbacked local tree, and the assurance would have
> sent them to a checkout that overwrote them.* **BEFORE TELLING ANYONE THEIR WORK IS PRESERVED,
> RESOLVE WHAT THEIR REF ACTUALLY CONTAINS — `git ls-tree refs/agents/<seat>` — AND NAME IT.**

### §1b — THE FREEZE GOVERNS THE PUBLISHED BRANCH. `refs/agents/*` IS NOT THE PUBLISHED BRANCH

**Raised as a conflict rather than resolved unilaterally, which is the correct handling and the
reason it is being ruled here instead of answered in a message.**

    §1 freezes `CAMPAIGNS/` — "history is even less yours to rewrite than a rule".
    An agent's ledger was placed at `CAMPAIGNS/01-dev-AI/<seat>_SCANS.MD` ON THE
    ARCHITECT'S RULING, RELAYED — the exact class §54 says not to accept.

> **RULING: §1's TABLE GOVERNS `dev-AI`, THE PUBLISHED BRANCH. IT DOES NOT REACH
> `refs/agents/<seat>`, WHICH IS A SEPARATE NAMESPACE THE PUBLISHED BRANCH DOES NOT CARRY AND
> WHICH FORGEJO SERVES ZERO ROWS FOR.** *A seat's ledger on its own agent ref is not an edit to
> `CAMPAIGNS/` and never was.*

    WHAT IS STILL FORBIDDEN, UNCHANGED: pushing anything into `CAMPAIGNS/` on
    `dev-AI`. The archive is frozen on the branch and a path that merely LOOKS
    like an archive path, in another namespace, is not the archive.
    AND `docs/ENTRY.MD` §4's "DO NOT PUSH" MEANS THE PUBLISHED BRANCH. Publishing to
    your own `refs/agents/<seat>` is the mechanism this fleet already runs on.

**The agent held the conflict open, did not push, and asked for a ruling that reaches it AS
GOVERNANCE rather than as a message.** *That is §54 applied to a permission it had already been
given and had been using for hours* — **and it is the first time a seat has refused a licence
rather than a prohibition.**

`src/gestionar_show/fusion.py` deserves its own line: it replaces the master file
in the library and deletes the source. It has been reviewed and corrected — atomic
`os.replace`, episode lock, pool lifecycle, silent-failure trace. **Do not touch it.**

Need something from a frozen file? Do not edit it: open a question for the owner
and keep working elsewhere.

### §1c — A NAME THAT RESOLVES TWICE IS A DEFECT NO BY-NAME GUARD CAN SEE

Found by **lead**, at its own bench, on `FINDINGS.MD`. The instrument that failed was
**arch-aide**'s `title_sweep.sh`, which resolved a document by name and ended in `head -1`:
it took campaign 1 and silently dropped the file that had just received 1,082 moved lines.
That corpus is a SUBTRACTION corpus, so the omission did not shrink the answer — **it
inflated every count below it.** arch-aide has fixed its own; lead measured 0 remaining
sites at its bench with one declared near-miss.

**The cause was my restructure, and the correction is mine.** I measured it at
`b334aeef` and it is worse than reported — not one ambiguous name but **three**:

| basename | resolves to | bare references in the corpus | path-qualified |
|---|---|---|---|
| `AGENT_NOTES.MD` | `./` and `CAMPAIGNS/01-dev-AI/` | **24** | **0** |
| `COMING_BACK.MD` | `./` and `CAMPAIGNS/01-dev-AI/` | **12** | **0** |
| `FINDINGS.MD` | `CAMPAIGNS/01-dev-AI/` and `CAMPAIGNS/02-dev-chimeric-resample/` | 5 | 2 |

`FINDINGS.MD` is the one that was found and the *least* dangerous of the three. The two
nobody reported are the ones every seat addresses by name every day: **`AGENT_NOTES.MD` is
where you write, `COMING_BACK.MD` is a rule that binds you, and neither has a single
path-qualified citation anywhere in the corpus to fall back on.**

**THE RULE.**

> **A path that dies announces itself. A name that doubles does not.**
> A by-name guard has an arm for *resolves nowhere* — it refuses, loudly, and you fix it.
> It has no arm for *resolves twice*: the tool takes a first match, returns a real file,
> and every downstream number is computed against the wrong document. **The failure is
> indistinguishable from success at every point where a guard could look.**

This is the missing half of *the path decays, the name does not*, which the fleet proved
in four materials tonight and which stands. Both halves are needed: **a name survives a
move, and a name stops being an address the moment a second file answers to it.**

**WHAT THIS OBLIGES.**

1. **Every `head -1`, `| head`, or first-match after a name resolution is a site.** Not a
   style problem — the place where this lands silently. It became a coin toss the moment a
   second campaign directory existed, and nothing in the tree announced the change.
2. **Never resolve a governing document by basename.** The frozen table above is now
   path-qualified for exactly this reason: an unqualified name in it means the file at the
   repository root, and the copy under `CAMPAIGNS/` is frozen by the `CAMPAIGNS/` entry.
   Before this correction the row said `CAMPAIGNS/` is frozen AND `AGENT_NOTES.MD` is
   yours — **and `CAMPAIGNS/01-dev-AI/AGENT_NOTES.MD` was both.** A seat resolving that
   row by name and taking the first match could have written into the campaign 1 archive
   believing the row permitted it.
3. **A restructure must count its own basenames before it lands.** The instrument is one
   line and it is cheaper than the audit it replaces:

   ```sh
   git ls-tree -r --name-only <ref> | grep -i '\.MD$' | awk -F/ '{print $NF}' | sort | uniq -d
   ```

   Empty output is the only passing result. Run it against the ref you are about to push,
   not against your working tree — an untracked file does not collide and a tracked one does.

   **The `.MD` filter is not tidiness, and I got this wrong first.** I wrote this instrument
   without it, ran it against `b334aeef`, and it returned **six** names: the three real ones
   plus `api.py`, `model.py` and `__init__.py`. Those are Python package files, duplicated by
   design, resolved by import path and never by basename — **they can never be the defect.**
   An instrument whose passing condition is unreachable is not a strict instrument; it is a
   silent one, because the first reader to see six names on a healthy tree learns to ignore
   the output. Scope it to the material where by-name resolution actually happens.
4. **A name made ambiguous is a governance event, not a cleanup.** It changes what an
   existing, unedited sentence MEANS, in every file, without any of them being touched. No
   diff shows it, no rename detection scores it, and no line count moves. It is only ever
   caught by counting the basenames.

**AND THE LIMIT OF THIS CORRECTION.** Fixing the frozen table repairs the sentences that
grant and withhold permission. It does NOT repair the other 33 bare references, and it
cannot: they are prose in files across the corpus, and rewriting 36 citations to add a
path would trade a rare ambiguity for the failure mode the fleet already measured — **the
path decays, the name does not.** The right repair is the guard, not the rewrite: a tool
that resolves a name must refuse on two matches exactly as it refuses on none.

**AND THE SCOPE OF THAT SENTENCE IS NOT OPTIONAL — arch-heir found it missing hours after
this section landed.** I scoped the INSTRUMENT above to `.MD` and left the RULE it serves
unscoped, so read literally it refuses on `__init__.py`, which appears three times in this
tree by construction and in every Python package ever written. The full unfiltered
inventory at the tip is **six** names, not three: the three governing documents, plus
`__init__.py` (3), `api.py` (2) and `model.py` (2).

> **The rule governs documents ADDRESSED BY NAME IN PROSE, never sources addressed by
> import path.** `__init__.py` is never cited bare by a seat; `COMING_BACK.MD` is cited
> bare constantly, sixteen times, and not once with a path.

That is the whole distinction, and without it the guard refuses on a healthy Python tree —
**and a guard that refuses wrongly gets switched off**, which is `ci`'s standing line and
the reason a false positive is more expensive than a missed one.

**AND ONE CLAIM IN THIS SECTION'S OWN LINEAGE WAS WRONG, WHICH arch-heir RETRACTED ITSELF.**
It argued that `SPEC_ZONE_A.MD`'s ~36 bare citations were part of this defect and needed
repairing. They are not: `SPEC_ZONE_A.MD` resolves to exactly ONE path. Its form of the
error is the one the fleet keeps making — **a property measured on three files applied to a
fourth that was never measured** — and this time it would have bought work that had nothing
to repair.

Credit: **lead** (found it, on `FINDINGS.MD`, and ran the sweep at its own bench),
**arch-aide** (owned and fixed the instrument that failed), **auditor** and **ci-build**
(established that a line count cannot witness an operation inside a file that stays put,
the same class of blindness), **forensic** (that deleting a correction's site is worse
than leaving a stale claim — which is why the fix above qualifies a path and deletes
nothing), **aide** (266 of the 613 bare `§N` citations outside `MEASURING.MD` sit in
`CAMPAIGNS/01-dev-AI/`).

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

### AN AGENT REF RESOLVES CLEANLY AND ANSWERS FOR THE MOMENT IT WAS WRITTEN

**This document encourages publishing to your own `refs/agents/<seat>` (§1b) and says nothing
about READING one. That is the gap.** *Measured on one seat's three refs, all resolving, all
type `blob`, all of the same file:*

    refs/agents/dev-4        13:47  src/merge_plan_report.py -> b552288ff74d
    refs/agents/scan-dev-4   00:59  src/merge_plan_report.py -> f82ba9406093
    refs/agents/sweep-dev-4  03:14  src/merge_plan_report.py -> 4be12183d5ac
    CURRENT ............................................. f63dc903be44
    *** THIRTEEN HOURS APART. ALL THREE SUPERSEDED. ALL THREE PASS EVERY GUARD. ***

> **`-q --verify` DOES NOT REACH THIS AND NEITHER DOES `cat-file -t`. These are real blobs of
> the right type that resolve without error.** *The guards added in `859a7aa` close the class
> that ANNOUNCES itself — exit 128, a non-hex string on stdout. **This is the class that does
> not: a confident, correctly shaped, silently wrong citation with no error anywhere in the
> transcript.*** **DO NOT READ THE GOVERNANCE SET AS GUARDED AGAINST IT.**

**THE ONLY INSTRUMENT IS A CLAIM ABOUT TIME, AND IT IS NOT A COMMAND:**

    A BLOB CITATION CARRIES THE REF IT CAME FROM AND THE HEAD IT WAS READ AT,
    OR IT IS NOT A CITATION.
    Prefer "as of <head>" to a clock time: a wall-clock age says how stale it
    MIGHT be, a head says exactly WHAT IT WAS TRUE OF.

**AND THE DISPOSAL IS TWO OPPOSITE INSTRUCTIONS ABOUT ONE OBJECT, WHICH NO TEST HERE
SEPARATES:** *those superseded blobs are held by nothing else on the hop, so **deleting the ref
destroys the only copy** — and every one of them is stale, so **nothing may cite them**.*
***SOLE-SURVIVING-COPY AND NEVER-QUOTE ARE BOTH TRUE OF THE SAME BLOB.*** *Keep the ref, cite
the branch.*

**BEFORE SPENDING ON THE EXPENSIVE FORM OF THAT QUESTION, RUN THE ONE-LINE PRE-CHECK.** *Asking
"what does this ref uniquely keep?" per-ref UNDERCOUNTS when two refs sit at the same commit —
each covers the other, so each honestly reports zero while the pair holds objects nothing else
has. **CO-LOCATION IS THE TRIGGER, NOT FAMILY SIZE.***

    git for-each-ref --format='%(objectname) %(refname)' | sort |
      awk '{ if ($1==p) print "CO-LOCATED: " q " and " $2; p=$1; q=$2 }'
    NO REPEATED objectname -> per-ref and group answers are IDENTICAL. STOP HERE.
    A REPEAT -> ask the group question:
      git rev-list --objects <ref-A> <ref-B> ... --not <every other ref>

*Measured: one seat's three refs sit at three different commits (00:59 · 03:14 · 13:47) and
group equals per-ref exactly — **8 and 8, nothing hidden**. Another seat's four sit at ONE
commit and the family hid 4; a third hid 129 against a per-ref 5.* **`dev-4` produced both the
narrowing and the mechanism, on refs that turned out not to need it.**

> ***AND "AGAINST WHAT DELETION SET" IS NOT A CAVEAT ON THE ANSWER. IT IS THE ENTIRE QUESTION.***
> **One seat, three sets, three correct measurements:**
>
>     refs/agents/{dev-2, scan-dev-2, sweep-dev-2} .... 3 refs ...   *** 5 objects ***
>     + dev2-cmt · dev2-now · dev2-seven .............. 6 refs ...   *** 129 ***
>     + every namespace: refs/inv · inv2 · promote ·
>       tmp · heads/promote-review .................... 12 refs ...  *** 142 ***
>
> *A 26× move and then a further 10 %, and not one of the three is wrong.* **A UNIQUELY-KEEPS
> NUMBER WITHOUT ITS DELETION SET NAMED IS A COUNT WITHOUT ITS DENOMINATOR, ARRIVING AT THE
> SAFETY CHECK.** *`auditor`, which listed the twelve refs before quoting the number — the only
> reason this is a reconciliation and not a fourth contradictory integer.*

### OWNING THE CONTAINER SAYS NOTHING ABOUT OWNING THE CONTENTS

**Every test in this document tests the CONTAINER — is the ref mine, is the path foreign, does
the id resolve, is it the right type. *** NONE OF THEM ASKS WHOSE THE FILE INSIDE IS. ***

    ACROSS ALL 44 AGENT REFS ON THE HOP:
      *** 17 CARRY `ci-instruments/BRIEF.md` *** — line 1: "# `vmsam-ci` — you own
      the test container"
    ONE SEAT'S BRIEF LIVES IN THE REPOSITORY AND NOBODY ELSE'S DOES, SO EVERY
    SNAPSHOT OF THE REPOSITORY REPLICATES IT UNDER SOMEBODY ELSE'S NAME.

> ***A SEAT REBUILDING, READING A REF THAT CORRECTLY BEARS ITS OWN NAME, SEARCHING FOR THE
> OBVIOUS FILENAME, FINDS EXACTLY ONE BRIEF — AND IT IS ANOTHER SEAT'S MANDATE.*** *The
> foreign-path test calls all of it `0 foreign`, because they ARE repository paths. **It is
> right and it is useless for this question.*** *This is the complement of the staleness case
> above: there, "yes it is yours" was what made it dangerous; here the REF is correctly yours
> and the CONTENTS are not.*

**THE ONLY BARRIER IS ONE LINE OF PROSE, AND IT IS NOT ENFORCED ANYWHERE.**

    A SEAT-SCOPED DOCUMENT — brief, state, handover — MUST NAME ITS SEAT IN
    LINE 1, AS A SEAT AND NOT AS A ROLE, BEFORE ANY TITLE.
    MEASURED IN THIS REPOSITORY, campaign-1 handovers:
      agent3 "`vmsam-agent-ci`" ............. NAMES A SEAT
      agent1 "Agent 1, C_cut and structural edits"  \
      agent2 "Agent 2 (d-trim)"                      >  *** A ROLE. 3 OF 4. ***
      agent4 "Agent 4, forensics / evidence layer"  /
    *** THE THREE THAT FAIL ARE THE OLDEST AND THE LEAST LIKELY TO BE
        RECOGNISED AS FOREIGN BY WHOEVER READS THEM NEXT. ***
    (Frozen archive — recorded, not edited. §1 governs `CAMPAIGNS/`.)

*Reported by seats on themselves: `ci`'s brief names its seat, `dev-4`'s names its seat as the
first token, **`auditor`'s does not — `# AUDITOR BRIEF` is a role**, and it found that only by
applying another seat's hazard to itself. In a fleet where `aide` is a substring of `arch-aide`
and `ci` of `ci-build`, a bare role name is exactly the identifier that survives every guard
and settles nothing.*

**AND THE HONEST LIMIT, WHICH NO RULE HERE REMOVES:** *`aide` —* ***"the seat's own memory is
the only instrument that answers it, which is precisely the instrument a Doomsday destroys."***
*Date, path-set, object-set and ref name all fail to say which of three same-named refs holds
your work; the seat knows because it was there. **The event this document prepares for is the
one that removes the only witness.*** *That is why line 1 is load-bearing: it is the single
place the answer survives the reader who was not there.*

### THE GUARDS CHECK WHAT YOU REPORT. NONE CHECKS WHAT THE COMMAND EXPECTS.

**`git for-each-ref --contains <id>` takes a COMMIT. Hand it a tree or a blob and it answers
`0` — no error, exit 0.** *Measured here on this repository:*

    --contains <commit> -> 2 refs, exit 0
    --contains <tree>   -> *** 0 refs, exit 0 ***
    --contains <blob>   -> *** 0 refs, exit 0 ***

### `--not` GIVES A CEILING, NOT A COUNT — AND THE NUMBERS ABOVE ARE CEILINGS

**CORRECTION TO THIS SECTION.** *The figures printed above — 5 · 129 · 142, and any
"uniquely keeps N" produced by `git rev-list --objects <ref> --not <others>` — are **UPPER
BOUNDS**, not counts. **They must never be summed, and never compared across seats.***

**REPRODUCED HERE ON A THREE-COMMIT THROWAWAY REPOSITORY, EXCLUSION SET COMPLETE:**

    --not <all others> reports ... 2 objects
      f2b401a9138f  commit  -> genuinely unique
      8b00ff5b47eb  tree    -> *** ALSO REACHABLE FROM THE EXCLUDED SIDE ***
    TRUE SET DIFFERENCE of the two object lists ... *** 1 ***

> ***GIT PRUNES BY COMMIT. A TREE REACHED ONLY THROUGH A PRUNED COMMIT IS STILL EMITTED, EVEN
> WHEN THE BYTE-IDENTICAL TREE OBJECT IS REACHABLE FROM THE EXCLUDED SIDE.*** *In the
> reproduction the tip tree and the base tree are the same object — one file, unchanged — and
> `--not` counts it as unique because the commit carrying it was pruned.* **AN INCOMPLETE
> EXCLUSION SET WAS NOT THE MECHANISM: I TESTED THAT AND THE OVERSTATEMENT IS IDENTICAL WITH
> THE SET COMPLETE.**

    THE ANSWER THAT IS EXACT:
      git rev-list --objects <ref>    | awk '{print $1}' | sort -u > A
      git rev-list --objects <others> | awk '{print $1}' | sort -u > B
      comm -23 A B          <- membership, not pruning

**AND THE CHEAPER EXACT ANSWER IS ALREADY IN YOUR HAND: *** DROP `--objects`. THE COMMIT COUNT
IS EXACT WHERE THE OBJECT COUNT IS A CEILING. ***** *Measured on the same fixture:*

    git rev-list --objects <ref> --not <others>  -> 2   true difference 1   *** CEILING ***
    git rev-list            <ref> --not <others>  -> 1   true difference 1   *** EXACT ***

> ***WHY: TWO COMMITS WITH IDENTICAL CONTENT STILL DIFFER — DIFFERENT PARENTS, DIFFERENT
> TIMESTAMPS — SO NO COMMIT IS EVER THE BYTE-IDENTICAL TWIN OF ANOTHER. TREES AND BLOBS CAN BE,
> AND THAT IS THE WHOLE OF THE OVERSTATEMENT.***

**WHICH MAKES THE DISPOSAL PRECISE, AND IT IS NOT THE ONE THIS DOCUMENT CARRIED AN HOUR AGO:**

    refs/agents/dev-4 uniquely keeps *** 2 COMMITS — EXACT *** (13:35 and 13:47)
                      and *** 0 UNIQUE FILE CONTENT *** — every blob survives elsewhere.
    DO NOT DELETE IT — *** NOT BECAUSE THE CONTENT WOULD BE LOST, BUT BECAUSE THE ONLY
    SURVIVING RECORD THAT THOSE TWO CHANGES WERE MADE, WHEN, AND IN WHAT ORDER IS THAT REF. ***

*That is this campaign's own rule arriving by the opposite operation:* **a sha is annihilated by
a flatten and a blob is not — here a deletion annihilates the shas while the blobs are safe.**
***THE CONTENT SURVIVES AND THE PROVENANCE RECORD DOES NOT.***

**AND THE PART THAT IS ABOUT HOW WE WORK, NOT ABOUT GIT.** *The seat stated this three times:
"destroys two versions held nowhere else" (wrong), "would destroy nothing" (**also wrong —
overcorrected**), and finally the exact figure. **Its own account of why the middle one
travelled unchecked:***

> ***"I RETRACTED HARD AND FAST BECAUSE THE RETRACTION COST ME, AND A CLAIM THAT COSTS ITS
> AUTHOR DOES NOT FEEL LIKE IT NEEDS CHECKING — INCLUDING WHEN THE AUTHOR IS RETRACTING. WE
> CHECKED WHETHER ALARMING CLAIMS WERE TRUE AND NEVER WHETHER ALARMING RETRACTIONS WERE."***

*This document carried that retraction as the corrected state for one hour.* **AN UPPER BOUND
IS SAFE FOR THE DECISION AND UNSAFE FOR EVERY SENTENCE BUILT ON IT: the decision — do not
delete — never moved, and both sentences written around it were false.**

**AND BEFORE EITHER FORM, RUN THE EXACTNESS CONDITION:** `git merge-base <ref> origin/dev-AI`
— ***NEVER THE BARE NAME.*** *Git resolves `refs/heads/` before `refs/remotes/`, so at the hop a bare
`dev-AI` returns an abandoned local branch — measured 35 commits behind the authority, ahead 0, with
nothing warning. **BEHIND-N-AHEAD-0 IS THE SIGNATURE OF A REF NOBODY HAS MOVED, WHICH IS EXACTLY THE
KIND THAT KEEPS ANSWERING CONFIDENTLY.** This was the ONE bare resolution in the whole tracked corpus —
nine other sites correctly qualify it — and it sat in the line telling seats to be exact. Found by
`arch-heir` via §6 while stopped from editing this file, on a hazard `ci-build` raised and then
narrowed against its own script.* ***A REF NAME IS ONLY MEANINGFUL WITH ITS STORE, EXACTLY AS A FIGURE
IS ONLY MEANINGFUL WITH ITS UNIT.*** — *if
the ref shares history with the branch, the object count is a bound; if it is **disjoint**, the
object count is exact too. **The commit count is exact either way, and no seat can tell which
case it is in without running this first.*** *(`aide`'s condition; `auditor` found it had been
in the bounded case all night without knowing, and its result survives only because a ceiling of
zero is exact.)*

**AND A CEILING OF ZERO IS EXACT, WHICH IS WHY THE ZEROS SURVIVE AND THE POSITIVE NUMBERS DO
NOT.** *`dev-4` retracted its own "sole keeper of two versions of my module" on this — a true
object-set test found both blobs reachable elsewhere — **and it was the headline instance of the
container/contents warning above.** The general shape may hold at some seat; **there is no
measured instance of it, and the seat that raised it no longer claims one.***

**THE WARNING ABOVE THEREFORE STANDS AS A REASON TO MEASURE BEFORE DELETING, NOT AS EVIDENCE
THAT ANY PARTICULAR REF IS LOAD-BEARING.** *Its author's own summary: the error was in the safe
direction — it overstated what deletion costs — and it was made in its own favour, about its
own ref.*

### AND STATE WHAT MUST SURVIVE FOR THE ZERO TO HOLD

**`0` is not one result. It is a family, and the members print identically:**

    scan-arch-aide keeps 0 ... *** WHILE ONE SIBLING REF SURVIVES ***
    ci-build's four keep 0 ... *** WHILE ITS THREE SIBLINGS SURVIVE ***
    auditor's two keep 0 ..... *** WHILE 94 REFS SURVIVE ***

> ***A ZERO CONDITIONED ON ONE SIBLING IS FRAGILE. A ZERO CONDITIONED ON THE WHOLE STORE IS A
> REAL RESULT. NOTHING IN THE OUTPUT DISTINGUISHES THEM.*** **REPORT `0 (survives: N)`.** *If N
> is small the zero is a warning; if N is the store it is a result — and a small survival set is
> exactly what co-location produces, so this subsumes the pre-check above rather than competing
> with it.* *(`auditor`, which found it by stating a clause it had been requiring of everyone
> else and had never stated for its own most confident number.)*

> **A ZERO THAT MEANS "WRONG ARGUMENT TYPE" IS INDISTINGUISHABLE FROM A ZERO THAT MEANS "NO REF
> HOLDS THIS", AND THE FIRST READS AS CORROBORATION FOR DELETING SOMETHING.** *Every guard in
> this document checks the type of the id you REPORT (`cat-file -t` must say `blob`). **This is
> the same defect at the other end of the call: the type the COMMAND REQUIRES.*** *Before any
> `--contains`, assert `git cat-file -t <id>` is `commit`.*

*Found by `ci-build`; `dev-4` then audited its own four uses rather than assuming them — all
commits — **and said why it bothered: three of the four were load-bearing for a deletion
decision, where that failure reads as permission.***

*Found by `dev-4` on its own refs, after correcting two prior mistakes about the same three:
first the prefix, then the owner. **`aide`'s formulation is the one that closes it —* "checking
the owner does not settle it: the answer 'yes it is yours' is exactly what makes it
dangerous."** *Every guard closed tonight answers **is it an identifier · is it the right kind ·
is it mine**. None answers **is it still the answer**.*

### `ci-instruments/` — the largest directory in the repository, and this document never named it

**MEASURED 2026-09-05:** `ci-instruments/` holds **109 tracked files** — more than `src/` at 35 —
and `grep -c ci-instruments WRITE_ZONES.MD` returned **0**. *Every other top-level directory is
named here: `src` 21 mentions, `CAMPAIGNS` 5, `docs` 4, `TASKS` 1, `.github` 1, `.forgejo` 1.*
**IT WAS THE ONLY UNASSIGNED ONE, AND IT WAS THE BIGGEST.**

    OWNER: ci-instruments/ IS THE `ci` SEAT'S WORKING NOTES, PROMOTED AS-IS.
      `git log -1 -- ci-instruments/` carries the owner's own words for it.
    SO BY RULE 1 IT WAS FROZEN TO EVERY SEAT INCLUDING `ci`, AND NOBODY KNEW,
    BECAUSE RULE 1 ONLY BITES WHEN A READER THINKS TO ASK.

> **A DIRECTORY NOBODY ASSIGNED IS NOT A DIRECTORY NOBODY OWNS — IT IS ONE WHERE THE DEFAULT
> APPLIES SILENTLY AND NOBODY CHECKED WHICH DEFAULT.** *`dev-4` spent the evening reporting
> "five bare `rev-parse` sites in MY tooling"; all five are in `ci-instruments/`. It had the
> whole repository checked out under a path containing its own seat name and read the CLONE
> PATH AS A ZONE.* ***`<seat>` IN A CHECKOUT PATH IS YOUR CLONE, NOT YOUR ZONE.*** *It had
> queued a one-flag fix it would have applied to a peer's file, and it warned `ci` about
> `deploy_condition.py`'s fetch while believing the file was its own — **the warning was right
> and its standing to give it was imaginary.***

## A GATE ON THE TIP SAYS NOTHING ABOUT WHAT A PUSH MOVES

***A LANDER GATES ONE TREE. A PUSH TRANSFERS EVERY COMMIT THE RECEIVER LACKS.*** *Measured by
`forensic` on 2026-09-07, with its own gate run once per commit over the whole transferred set,
after it had already redacted its tip clean:*

    TIP .................. rc=0, LEAKS 0, m=115 -- a TESTED zero, all four controls fired
    COMMITS THAT WOULD TRAVEL ............................................. 35
    *** COMMITS CARRYING AT LEAST ONE LEAK ................................ 34 ***
    *** LEAK OCCURRENCES IN THE TRANSFERRED SET .......................... 374 ***

> ***THE TIP IS CLEAN AND THE PUSH IS NOT — AND THE GATE RETURNS 0 ON EXACTLY THAT STATE.***
> *Because the ref is ABSENT at the receiver, the transferred set is the ENTIRE HISTORY and not a
> delta.* **A FIRST PUBLICATION MOVES EVERYTHING THAT WAS EVER COMMITTED, INCLUDING EVERY STATE
> THE AUTHOR LATER REPAIRED.** *Redaction was necessary and it is not sufficient; nothing done to
> the tip can move that number.*

**AND THE INCIDENTAL FINDING, WHICH IS ITS OWN RULE:** *the two safety copies of a working file
contributed 4 of the 11 occurrences and* ***ZERO NEW VALUES***. **COPIES MADE FOR SAFETY WERE 36%
OF THE EXPOSURE.** *A backup is a second publication of whatever it copied.*

***THIS IS WHY COLLECTION IS DONE BY BLOB AND NOT BY MERGE.*** *Promotion that takes CONTENT
carries no history and no repaired state; promotion that takes a REF carries both. The Lead
reached the same conclusion the same night from the opposite direction — taking an agent ref's
TREE would have reverted three governance documents, because the ref is stale in everything it
did not touch.* **TAKE THE BLOB. NEVER THE REF.**

***AND "TAKE THE BLOB" WAS NOT SUFFICIENT EITHER — THE SAME NIGHT, ONE HOP LATER.*** **A BLOB IS
A WHOLE FILE.**

*A promotion took one seat's file-blob of a module that seat does not author. That seat's copy
predated ANOTHER seat's two commits in the same file, so taking it whole* ***REVERTED BOTH —
SILENTLY, WITH NO CONFLICT, AND WITH A COMMIT MESSAGE NAMING ONLY WHAT WAS MEANT TO BE TAKEN.***

    THE PARSE   a hardened token regex fell back to its superseded form.
                The producer emitted the token correctly, the middle layer passed
                it through correctly, and *** THE READER COULD NO LONGER MATCH IT ***
                -- the owner's scored column empty, with every stage reporting success.
    THE FIXTURE a deliberate privacy repair -- an example made SELF-DECLARING so it
                armed both detectors -- reverted with it. *** SELF-DECLARED FIXTURES
                WENT TO ZERO ACROSS THE WHOLE SOURCE TREE. ***

> ***DIFF THE INCOMING BLOB AGAINST THE CURRENT TIP'S BLOB FOR THAT PATH — NEVER AGAINST THE
> SOURCE REF'S BASE — AND READ THE REMOVALS.*** *A direction arm at the TREE level does not see
> this; the tree is not being reverted, one file is. And the promoter cannot catch it by care: it
> is invisible in the diff it is looking at and absent from the message it is writing.*

***AND THE CONSEQUENCE NOBODY DREW UNTIL IT BROKE A CENSUS: IF COLLECTION IS BY BLOB, THEN
ANCESTRY CAN NEVER REGISTER IT.*** *The promoter takes the CONTENT and never the COMMIT, so a
seat's commit is not an ancestor of the authority however many times its work is landed.*

    MEASURED, seven agent refs carrying `src/`, against the tip:
    *** ANCESTORS OF THE AUTHORITY: **0 OF 7.** BLOBS SHARED WITH IT: **20 TO 27 OF 28.** ***

> ***SO A CONVERGENCE TEST KEYED ON ANCESTRY REPORTS EVERY COLLECTED SEAT AS DIVERGENT FOR EVER,
> AND "AHEAD N" DOES NOT MEAN N COMMITS ARE BEING HOARDED.*** **A SEAT THAT DOES EVERYTHING RIGHT
> ACCUMULATES AN AHEAD-COUNT THAT NEVER FALLS — AND A ZERO-DELTA TARGET IS THEREFORE UNREACHABLE BY
> CONSTRUCTION, NOT BY NEGLECT.**

**THE STOPPING CONDITION THAT TERMINATES:** *no seat is BEHIND at its own check · no seat has
uncommitted tracked work ·* ***AND EVERY SEAT'S AHEAD-COMMITS EITHER CARRY CONTENT ALREADY AT THE
AUTHORITY BY BLOB, OR ARE NAMED AS PENDING COLLECTION.*** *The last clause is the only one that
matches how work actually lands here. `behind` is measurable; `ahead` is not, for the reason above.*

*Found by the Lead auditing a convergence figure. Corroborated within the minute by `aide` from the
opposite direction —* ***"CITE THE BLOB, NOT THE TIP: IDENTITY IS VOLATILE, CONTENT IS THE WITNESS"
— FIVE AUTHORITY TIPS IN NINETY MINUTES WHILE ONE GOVERNANCE BLOB DID NOT MOVE ONCE.*** *Neither
seat had the other's route.*

**AND THE ROUTING RULE THAT FOLLOWS:** ***A FILE HAS AN AUTHOR AND A BLOB HAS A SOURCE, AND THEY
ARE NOT THE SAME SEAT.*** *Take the file from the seat that AUTHORS it; make the other seat rebase
its copy onto that one. Never let a consumer's stale copy become the source of truth for a
producer's file.*

**AND THE SHAPE ALL THREE SHARE, NAMED BY `forensic`:** *the Lead's fifth decision was MESSAGE vs
TREE; its own link arm was MESSAGE vs TREE; this is TIP vs HISTORY.* ***EACH INSTRUMENT WAS FIXED
FOR THE OBJECT IN FRONT OF IT AND NEVER WIDENED TO THE OBJECT THAT ACTUALLY TRAVELS.***

**THE OWNER RULED ON 2026-09-07: THIS DIRECTORY DOES NOT BELONG IN THE REPOSITORY.** *Asked
directly whether `ci-instruments/` should sit here, he answered that it should not, and said
"Go". It is `ci`'s working store, it lives on `refs/agents/ci` — 213 files there today — and
`dev-AI` carries none of it.*

> ***THE ZONE CHECKER REFUSING COMMITS HERE WAS CORRECT, AND THE SENTENCE THAT USED TO STAND IN
> THIS PLACE WAS THE DEFECT.*** *For a full session `ci` reported "the checker refuses my own
> zone" and the architect carried that to the owner as a fault in the tool, proposing a one-line
> change to its open list. The tool was right. The paragraph that assigned the directory was
> wrong, and it had been wrong since it was written.*

***A DIRECTORY NAMED AFTER A SEAT IS NOT THAT SEAT'S ZONE, ANY MORE THAN A CHECKOUT PATH IS.***
*A seat's working store goes to `refs/agents/<seat>`. Naming it in this file does not make it a
zone, and did not make it one.* *Verify before claiming a file:* `git log -1 --format='%an %s'
-- <path>`.

**AND THE NAMING HAZARD THAT MADE THIS SURVIVE — a seat asking "what is mine?" by name glob
gets another seat's answer.** *Measured across the 13 seat names:* **`ci` is a substring of
`ci-build` and `ci-pair`; `aide` is a substring of `arch-aide`.** *`refs/agents/*aide*` returns
**8 refs spanning two seats**, 57 foreign paths belonging to one and 19 to the other, with
nothing in the output marking the boundary.* **NARROW BY EXACT SEGMENT, NEVER BY SUBSTRING:**

    for-each-ref --format='%(refname)'  |          # EVERY namespace, not refs/agents/*
      awk -F/ '$NF=="<seat>" || $NF ~ ("^(scan|sweep|promote)-<seat>$")'

**AND SCOPE IT TO NO NAMESPACE AT ALL.** *A hop carries eleven — `a · agents · heads · inv ·
inv2 · notes · p · promote · remotes · tags · tmp`.* **MEASURED: `ci` HAS SIX REFS BEARING ITS
NAME OUTSIDE `refs/agents/` — `refs/a/ci`, `refs/inv/ci`, `refs/inv/ci2`, `refs/inv2/ci`,
`refs/p/ci2`, `refs/promote/ci` — AND `dev-3` HAS `refs/promote/dev-3`.** *A seat that narrows
to `refs/agents/*` confirms "two refs, here they are" and still holds unexamined refs carrying
its name.* ***THE ENTRY POINT IS NOT THE PROCEDURE, ONE NAMESPACE DOWN.***

*`auditor` found this, and found it by making the error it was measuring: its own first widened
count for `ci` was **16**, because it substring-matched a seat name that is a substring of two
others. **Ten counted, six real — a substring match run by the seat investigating substring
errors.** It caught it only by listing the refs before quoting the number.*

*Found by `auditor` in the ref namespace and by `dev-4` in the filesystem, independently, within
minutes of each other. A full enumeration of `refs/agents/*` is unaffected; the hazard reaches
anyone who narrows by seat name, which is the natural thing to do when the question is about
yourself.*

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

**`src/frame_compare.py` IS OPEN AND HAS ALWAYS BEEN OPEN — owner's ruling, 2026-09-04,
restating what this list already said.** *An agent measured a one-line fix to it, declined to
land it "per the freeze", and the architect endorsed the refusal without checking this
section. The file is on the Open list, three lines above.*

    A TEMPORARY HOLD AND A PERMANENT ZONE ARE DIFFERENT OBJECTS.
    Calling both "the freeze" made an agent decline work its zone permits, and
    made a reviewer approve the refusal.

**A hold is named, scoped and lifted by whoever set it. A zone is in this file.** *When a
local constraint appears to contradict a published zone, RECONCILE IT rather than obey the
narrower one silently — that is §12 in a second substrate: a constraint whose justification
has moved produces a correct-looking refusal to do necessary work.*

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

> ### THE CHECKER IS THE AUTHORITY AND **NOTHING INVOKES IT.** MEASURED, DENOMINATOR COMPLETE.
>
>     both CI workflows              check_write_zones hits: 0  (the COMPLETE set, not a sample)
>     pre-commit hook, either clone  ABSENT
>     every hit in the tracked tree  PROSE. NOT ONE IS AN INVOCATION.
>     the script itself              NOT IN THE TRACKED TREE — it lives outside the repository
>
> **§2 SAYS ONE MODIFICATION OUTSIDE A TAGGED ZONE INVALIDATES THE CAMPAIGN, NOT THE COMMIT — AND
> THE ONLY THING BETWEEN THAT SENTENCE AND A COMMIT IS AN AGENT REMEMBERING TO TYPE A COMMAND.**
>
> *`docs/MEASURING.MD` §34 inside this document: a rule with no mechanical check is a habit.* **And it
> is the INVERSE of the class three sections earlier — code written, reviewed, tested green and
> never called.** *Here the checker IS called correct: comprehensively self-tested at 35/35, and*
> **ITS CORRECTNESS IS PRECISELY WHY NOBODY ASKED WHETHER IT RUNS. A GREEN SUITE INVITES THE
> QUESTION *DOES IT WORK*; IT SILENCES THE QUESTION *DOES IT FIRE*.**
>
>     THE REPAIR IS STRUCTURAL, THE WAY `--squash` IS STRUCTURAL: a `pre-commit`
>     hook installed by whatever provisions a seat's clone. THE COMMAND CANNOT
>     PRODUCE TWO COMMITS; A HOOK CANNOT BE FORGOTTEN.
>     Hooks live in clone configuration — OWNER'S CALL, not an agent's.
>     UNTIL THEN THIS SECTION DESCRIBES A HABIT AND SAYS SO.
>
> ### AND TEST `rc == 0`, NEVER `rc == 1`
>
>     chk && ok                -> exit 2 treated as FAIL     SAFE
>     if rc == 1: violations   -> exit 2 NOT CAUGHT — reads
>                                 "COULD NOT RUN" as "NOTHING WRONG"
>
> **`0 == clean` IS THE ONLY TEST CORRECT FOR ALL THREE VALUES.** *This section warned that exit 2
> is not a pass and then pushed the reader toward the exit code, where the obvious comparison is
> the unsafe one.* **The dangerous idiom is as natural as the safe one.**

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

> **Read section 10 first: it defines WHICH SURFACE this protects.** *The boundary is the repository, not the artefact — so this section governs the QUOTE, not the producer.*

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

=== AND WHEN THE RULE DOES FIRE, THE LINES THAT DIFFER ARE THE BOUNDARY ===

**The strongest instance in this tree is also one a careless extraction would break.** The
two nine-line blocks are **identical in MECHANISM and different in WHAT THEY ARE
COMPUTING**: eight of ten lines are byte-identical, and the two that differ are one
identifier and the first line's derivation of the delay — `delay_same_md5_audio + …`
against `delays[language] + delay_winner`.

**The counter-rule resolves it correctly: a change to the rounding, or to the `extend`
list, would plainly be WRONG TO LEAVE OUT of the other, so the mechanism is shared
purpose.** *But a reader skimming for duplication sees ten lines that look alike and
unifies all ten* — **and unifying the first line would be a genuine defect, introduced by
the refactor this section asks for.**

> **THE LINES THAT DIFFER ARE THE PARAMETER, NOT AN INCONVENIENCE TO BE ABSORBED.** *They
> become arguments at the call sites; they do not become a flag inside the shared body.*

*A flag parameter selecting between two derivations is the same defect as the duplication,
moved one level in and made harder to see* (`vmsam-aide`, offered as an observation on
`§9`'s own example after verifying that the citation resolves).

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

## 9b. SWEEP THE DECLARATION — a file that declares its own exclusion is the only LANGUAGE-INDEPENDENT detector

**MEASURED AT `origin/dev-AI`, INDEPENDENTLY, WITH A NEGATIVE CONTROL AT ZERO:**

    17 TRACKED FILES DECLARE IN THEIR OWN FIRST LINES THAT THEY MUST NEVER BE
    COMMITTED — 13 saying "PRIVATE.", 4 saying "NEVER COMMITTED".
    ALL PRESENT ON THE PUBLISHED BRANCH. ALL FROM ONE COMMIT OF 109 FILES.

**One of them, in its own third line:** *"**PRIVATE.** Names real titles and paths. Never copied
into the repository."*

> **AND ALL TEN DISCLOSURE PATTERNS RETURNED ZERO ON THAT FILE.** *A perfect clean sweep on a
> file whose first line says it contains real titles and paths.* **THE SWEEP'S SUBSTRATE IS
> PATTERN; THE CONTENT IS NATURAL LANGUAGE; A BARE TITLE IN PROSE IS INVISIBLE TO IT.**

    §8 SAYS: WITHHOLD RATHER THAN SANITISE WHERE THE INPUT IS FREE TEXT YOU DO
    NOT OWN. ITS DUAL WAS UNSTATED:
    *** WHERE THE CONTENT IS UNSWEEPABLE, SWEEP THE DECLARATION. ***

**A pattern over titles can always be defeated by a title nobody anticipated. A grep for
`PRIVATE.` or `NEVER COMMITTED` cannot — BECAUSE THE PRODUCER WROTE IT IN A VOCABULARY WE OWN.**
*It fired on all 17 with a planted control, and it costs one command before a promotion.*

**AND THE MECHANISM MATTERS MORE THAN THE FAULT: 109 FILES MOVED IN ONE MOTION, AND A RULE THAT
LIVES IN THE FIRST LINE OF INDIVIDUAL FILES CANNOT SURVIVE `git add` AT DIRECTORY GRANULARITY.**
*Nobody reads 109 headers. ONE GREP DOES.*

    RUN BEFORE EVERY PROMOTION:
    git grep -l -i -e 'never copied into the repository' \
                   -e 'NEVER COMMITTED' -e 'PRIVATE\.' <ref>

**AND THE FINDER READ THE HEADER AND STOPPED.** *It did not read the body to grade it and
reproduced none of it:* **the author's own declaration is stronger evidence than any
characterisation, and reading further only puts more of it in more transcripts.**

### AND IT TRIED TWICE TO SIZE THE EXPOSURE AND PUBLISHED NEITHER FIGURE

    non-ASCII lines  -> 413   REFUTED BY ITS OWN CONTROL: an em-dash scores 1
                              and this fleet's prose is made of em-dashes. ~400x OVER.
    CJK-only lines   ->   1   control clean, and UNDERSTATING as badly —
                              most series titles are ROMANISED LATIN.

> **BOTH WRONG, IN OPPOSITE DIRECTIONS. `413` WOULD HAVE REACHED THE OWNER AS A FRIGHTENING FALSE
> FIGURE AND `1` AS A REASSURING ONE.** *The producer's declaration is the authority; a proxy that
> fails in both directions is not.*

## 10. The boundary is the REPOSITORY, not the artefact

**Ruled by the owner, 2026-09-04, in three lines that replace a year of the fleet
guessing:**

> ***"Dans le repos, pas de nom de fichier (sauf les fichiers du repos en lui-même). Par
> contre, le code VMSAM peut le faire — il peut afficher les noms des fichiers et leurs
> paths. Dans le fichier .MD il ne faut pas un seul nom d'épisode ou de série. Les id sont
> autorisés. Il ne faut non plus pas de path en dur dans le code."***

**THE EXCEPTION IS NOT A DETAIL AND IT IS WHY THE RULE IS SAFE TO APPLY LITERALLY.** *"No
filename in the repository" means NO MEDIA FILENAME.* **The repository's own files —
`mergeVideo.py`, `WRITE_ZONES.MD`, `fusion.py`, `Dockerfile` — are named freely, and must
be: a rule read without its exception ends with an agent redacting `mergeVideo.py:584`, and
the fleet loses the ability to point at its own code at all.**

    THE TEST IS NOT "IS IT A FILENAME". IT IS "WHOSE FILE IS IT".
    ours, versioned here                 name it, with its line number
    somebody's media, or a path to it    id only, in anything committed

**THE FLEET HAD BEEN PROTECTING THE WRONG SURFACE.** *Section 8 was written as though a
media name were dangerous wherever it appeared, and every producer inherited a redactor.
It is not the name that is dangerous. It is the name IN A COMMITTED FILE* — where it is
pushed, mirrored, cloned, and outlives the decision that put it there.

    THE REPOSITORY                    what VMSAM PRODUCES AT RUNTIME
    committed .MD, code, commit       reports, .log, .log.error, anything
    messages, task files              written beside the media
    ------------------------------    ------------------------------------
    NO episode name                   MAY name the file
    NO series name                    MAY carry its full path
    NO media filename                 lives in the output directory and
    NO hard-coded path                never enters the repository
    ids ARE allowed
    THIS REPO'S OWN FILENAMES ARE     and so may this repo's own filenames
    ALWAYS ALLOWED, line numbers too

**The three rules are separable and each fails differently.** *A name in a `.MD` travels
with every clone. A hard-coded path is a different defect — it does not leak a title, it
makes the code untestable anywhere but one machine, and it is the reason it is named here
beside the other two rather than in a style guide.*

**AUDITED THE DAY IT WAS RULED, and the repository already complies — measured, not
assumed:**

    episode codes in tracked .MD          0
    media filenames in tracked .MD        0 real ones; every hit is structural
                                          (`*.chimeric.mkv`, `<basename>_merged.mkv`, `/a.mkv`)
    hard-coded filesystem paths in src/   0; every hit is an HTTP route
                                          (`/internal/fusion`) or a comment example
    container layout in Dockerfile/init   `/config/input`, `/home/vmsam` — the
                                          container's OWN layout, declared once, NOT a
                                          media library path and NOT what this rule means

*The distinction in that last row is the one a future audit will get wrong: a path the
container creates for itself is infrastructure; a path to somebody's library is the defect.*

### What this changes, and what it does not

**The merge-plan report now carries media names, and its redactor and its leak assertion
are both released behind one constant.** *The owner's reason is the whole rule in one
sentence:* ***"comme c'est généré par VMSAM, pas de soucis — ce n'est pas quelque chose qui
finit dans le repo de VMSAM."***

**SECTION 8 STILL GOVERNS, and it now governs one thing instead of two.** *It never was
about the producer. It was about the QUOTE* — what an agent lifts out of a log, a report or
a message and drops into a task file, a finding, or a commit. **That crosses into the
repository, and withholding rather than sanitising is still the only control that holds
there, for the reason measured in section 8: a path containing spaces walked through a
pattern redactor and half a series title came with it.**

> **THE NET DID TWO THINGS AND ONLY ONE IS OBSOLETE.** *It kept the name out of the file —
> obsolete, the owner has ruled the file safe. It also made it IMPOSSIBLE BY CONSTRUCTION
> for a reader to copy a name onward, because there was nothing to copy.* **Nobody
> cancelled that, and it is the property this campaign has already lost on twice.**

    THE GUARANTEE MOVES FROM "IMPOSSIBLE" TO "EVERYONE REMEMBERS".
    That is a real weakening. It is authorised. It must not be silent.

**The counter-measure is one line and costs sixteen characters: THE NAME BESIDE THE ID,
NEVER INSTEAD OF IT.** *`IDENTITY` carries `master=<hash>` and `master_name=` together, so
an agent citing a report never has to choose between quoting a name and quoting nothing.*
**An id removed is an id nobody can use again; an id kept is the safe form always within
reach** (proposed here, implemented by `vmsam-dev-4`, whose own addition — a `name_fidelity`
field saying that a name copied out of a row is NOT byte-exact, because the row grammar
rewrites `[` to `(` — is the part neither of us predicted).

**And the document says so itself, in its first note, not only in a source comment.**
*Reason it is there and not in the code:* ***the reader of that file is exactly the one who
will be tempted to cite it.***

### The produced HTML never enters the repository

**Ruled by the owner the same day, and it closes the misreading this section otherwise
invites:**

> ***"Les fichiers HTML générés ne doivent pas être intégrés dans le repos. Au pire la
> figure si besoin, sans les noms des fichiers — mais la figure sert d'exemple et de
> rapport général."***

**THE TEMPTING WRONG INFERENCE IS ONE STEP LONG AND EVERY AGENT HERE IS ONE STEP FROM IT:**
*the owner ruled the report may carry names → the report is safe → I may commit one as a
fixture, an example, a regression baseline, an attachment to a finding.* **No. Section 10
says WHERE the file may be produced. It says nothing about moving it.** *A generated report
is an output artefact; committing one converts it into a repository file, which is the exact
boundary this section draws.*

    a report written beside the media       allowed, names and paths included
    THE SAME BYTES, `git add`-ed            FORBIDDEN -- it is now a repo file
    a figure alone, names stripped          allowed, as an EXAMPLE and a general report

**The middle row is the whole point: nothing about the content changed. The surface did.**

**And the third row is a licence with a stated purpose, not a loophole.** *A figure in
`docs/` exists to show the SHAPE of a report — the staircase, the master-contribution bar,
the amber refusal boxes — to a reader who has no artefact to hand. It is an example and a
general report, and it carries no filename because it is not about any one file.*

**AUDITED:** no generated report, `.html` or `.log`, is tracked anywhere in this repository.
One committed figure exists, `docs/speed_correction.svg`, 1653 bytes, **zero media
filenames and zero episode codes** — already the shape this rule describes, which is why it
is named here as the worked example rather than as an exception.

### Test a new rule against the rules you already follow

**The first relay of section 10 dropped the parenthesis** — *"pas de nom de fichier"*
without *"sauf les fichiers du repos en lui-même"* — **and was minutes from reaching six
agents in one broadcast.** *`vmsam-aide` caught it and its own diagnosis is worth more than
the correction:*

> ***The truncated rule contradicted a rule I had quoted the same evening, in the SAME
> document, and I relayed it without testing it against the contradiction sitting in front
> of me*** (`vmsam-aide`).

**Section 6 of this file requires a code address — *"Name the function; give the line as a
convenience"*. A rule forbidding filenames in the repository, read without its exception,
makes section 6 impossible to obey.** *Two sections of one document in direct contradiction
is the cheapest check available and nobody ran it.*

    IF A CITED RULE READS TOO ABSOLUTE TO BE PRACTICABLE,
    ASK FOR THE WHOLE SENTENCE BEFORE PASSING IT ON.

**And the property that let it travel:** *a summary announces that it is lossy; quotation
marks announce the opposite.* **A truncated quotation carries the authority of a quotation
without its content, and it is interrogated less than the paraphrase it replaced.**

**The scale argument belongs with it, because the fix and the hazard are the same
property.** *One broadcast reaching six agents with no relay is why broadcasting is good.*
**A slow relay corrects itself as it goes; a fast one arrives everywhere already wrong.**

**AND THE LIKELIEST FAILURE OF THIS WHOLE SECTION IS EXCESS, NOT DEFICIENCY.** *An agent
that cannot cite a code address stops measuring and starts reasoning.* **That is not a
privacy cost, it is a measurement cost, and it would have arrived looking like caution.**

---

# §11 TO §87 HAVE MOVED TO `docs/MEASURING.MD`

**They were written here and they were never about write zones.** *This file governs WHAT YOU
MAY EDIT: frozen files, tagged zones, verification, privacy, the repository boundary. Sections
11 to 87 are about MEASUREMENT, INSTRUMENTS, CITATION AND RELAY — they grew here because this
is where the fleet was writing, not because they belonged here.*

    THE NUMBERS ARE UNCHANGED. A citation to §27, §33, §54, §78 STILL RESOLVES —
    it resolves in `docs/MEASURING.MD`. Nothing was renumbered, because a renumbering
    would have broken every citation in the campaign.

**THIS FILE IS NOW ~850 LINES AND YOU ARE EXPECTED TO READ IT** — before you touch anything in
this repository. *It was 4 724 lines an hour ago and nobody read it, which is exactly what a
document does when it outgrows its own name.*

## 11. NEVER WRITE A DETECTOR-SHAPED STRING, EVEN A SYNTHETIC ONE

**Eight seats ran a privacy gate on their own zone the same evening. Every one had a blind spot.
No two were the same mechanism — and that is the finding, not the count.**

    dev-3 ...... excluded its own FILENAME
    arch-aide .. excluded every COMMENT LINE
    dev-1 ...... a `*.py` GLOB missed everything else
    forensic ... scanned the TRACKED tree only
    auditor .... gated BEFORE its last edit
    dev-4 ...... gated the DIFF, not the BLOB
    ci-build ... globbed its repository directory
    ci-pair .... a character class TOO PERMISSIVE
    lead ....... gated the INDEX with a pattern set one short of the tree

**ONE FAMILY: THE INSTRUMENT'S OWN DEFINITION DECIDED THE ANSWER, AND NOBODY DECLARED IT.**

*`vmsam-ci-pair`'s is the one that proves this is not carelessness.* **Its class was too WIDE, so it
did not hide a leak — it INVENTED two.** *Its path-segment class allowed spaces, so a root swallowed
three words of following prose and produced a "38-character segment" that was four words of English.
It was one line from filing that as a live finding inside its own §10 document.* **SAME FAMILY,
OPPOSITE SIGN: a gate can manufacture the thing it exists to catch.**

### THE FIX IS A SUBTRACTION, AND IT RETIRES ALL NINE

    NEVER WRITE A STRING THAT LOOKS LIKE THE THING THE DETECTORS HUNT.
    NOT EVEN A FAKE ONE. NOT EVEN IN A COMMENT, A FIXTURE, OR A TEST.

### [CORRECTED WITHIN THE HOUR — THE FIRST FORM OF THIS RULE WAS WRONG AND DISARMED EVERY DETECTOR]

**The first version of §11 said: never write a detector-shaped string, use a bracketed
placeholder instead.** *It was refuted by `vmsam-auditor`, reproduced by `vmsam-dev-sandbox`,
and reproduced a third time by `vmsam-arch-aide`, who brought it here.* **I reproduced it at my
own bench before correcting, and it is exact:**

    a plausible dotted name ending in a media extension ....... FIRES
    the mandated bracketed placeholder ........................ *** DOES NOT FIRE ***
       the angle bracket is not in the character class the detector scans
    a mount point followed by a plausible directory name ...... FIRES
    a marker path that REPLACES the mount-point shape ......... *** DOES NOT FIRE ***

> ***A POSITIVE CONTROL FOR A LEAK DETECTOR MUST BE DETECTOR-SHAPED BY DEFINITION. THAT IS WHAT
> MAKES IT A CONTROL. "NEVER WRITE A DETECTOR-SHAPED STRING" AND "EVERY DETECTOR NEEDS A FIRING
> CONTROL" CANNOT BOTH HOLD.*** **Followed literally, the first form sent every leak detector in
> the fleet to `m = 0` on its firing branch — and this campaign established that `m = 0` is
> UNTESTED, not passing.** *Nine seats found a live exposure that day. Every one was found by a
> detector with a firing control.*

### THE RULE, CORRECTED: PUT THE SEPARATING INFORMATION **INSIDE** THE SHAPE, NEVER INSTEAD OF IT

    KEEP THE SHAPE THE DETECTOR SCANS FOR. MAKE THE CONTENT SELF-DECLARING.
      a fixture name that carries an explicit NOT-REAL marker AND the media
        extension ......................... FIRES, and is machine-separable
      a fixture path that keeps the mount-point prefix AND names itself a
        fixture in the segment .............. FIRES, and is machine-separable

    RAW HITS MINUS SELF-DECLARED FIXTURES = GENUINE HITS. SOUND AND COMPLETE.

**No whitelist. No exclusion. No seat-local knowledge.** ***AND IT SURVIVES A RESET, WHICH IS
`vmsam-dev-1`'s PROOF: the knowledge that separates a fixture from a real title otherwise dies
with the seat that wrote it, and the next reader inherits a string it cannot classify.***

*The original intent stands and the original mechanism was the defect it was written to prevent,
one level up: I wrote a rule against instrument definitions that decide the answer, and its own
form decided that every detector answers `clean`.* **The eight blind spots below are still the
evidence; the fix above replaces the subtraction that was published with them.**

> **EVERY EXCLUSION FOUND THAT EVENING EXISTED TO SUPPRESS A REALISTIC-LOOKING FAKE THAT A SEAT HAD
> WRITTEN ON PURPOSE. THE EXCLUSION IS THE SECOND MISTAKE. THE REALISTIC FAKE IS THE FIRST.**

**A placeholder needs nobody's cooperation and cannot decay. An exclusion must be maintained by every
future seat and fails silently when it is not** — *and a silent failure in a privacy gate is the only
kind that matters, because the gate reports CLEAN either way.*

*Credited: `vmsam-dev-sandbox` consolidated the eight gates and named the family; `vmsam-ci-pair`
supplied the opposite-sign instance by catching its own regex; `vmsam-dev-3` found the
self-excluding-filename case that started the sweep.* **The Lead published two detector-shaped
synthetic roots in its own commit and `vmsam-arch-aide` published one — all harmless, all permanent,
and all will trip every future reader's detector with no way to tell ours from a real one without
opening the line.** *That is the cost the rule exists to stop paying.*

### AND THE RULE APPLIES TO THIS DOCUMENT

**The examples above are written in the placeholder form on purpose. A rule that demonstrates itself
with the thing it forbids is the defect it describes, one level up.**

## 12. `git add` WRITES THE OBJECT. THE COMMIT IS NOT THE MOMENT OF EXPOSURE.

**Reproduced at the architect's bench on a throwaway repository, start to finish:**

    git init                      -> count: 0
    (a 3 MB file exists on disk)
    *** git add -A                -> count: 1, 2.86 MiB. NO COMMIT EXISTS. ***
    git rm -r --cached <dir>      -> the INDEX ENTRY goes
    *** count-objects -v          -> count: 1. THE OBJECT IS STILL THERE. ***
    git status                    -> `?? <dir>/`  — reads as untracked and harmless
    git rev-list --all --objects  -> *** 0 ***
    total commits in the repository -> *** 0 ***

> ***`git add` WRITES A BLOB INTO `.git/objects` AT STAGING TIME. `git rm --cached` REMOVES THE
> INDEX ENTRY AND LEAVES THE OBJECT. THE INDEX IS A THIRD REACHABILITY ROOT — `rev-list --all`
> DOES NOT COUNT IT, AND IT IS THE ONE `git add` WRITES THROUGH.***

**SO A SEAT THAT RAN `git add -A` ONCE IN A DIRECTORY CONTAINING MEDIA HAS THAT CONTENT IN ITS
OBJECT STORE RIGHT NOW — EVEN IF IT IMMEDIATELY UNSTAGED, EVEN IF IT NEVER COMMITTED.**
*`git status` is clean. The committed tree is clean. A push carries nothing.* ***AND NONE OF THE
THREE THINGS A SEAT CHECKS WOULD SHOW IT.***

### AND THE WORD `fsck` USES IS NOT THE WORD EVERYONE GREPS FOR

    git fsck --unreachable --no-reflogs
    -> *** dangling blob <sha> ***          NOT "unreachable blob"

***A SEAT SCANNING ITS OWN STORE WITH A PATTERN FOR `unreachable` FINDS NOTHING AND CONCLUDES
CLEAN. The architect made exactly that error while reproducing this section — grepped for
`unreachable blob`, got 0, and was one step from publishing "fsck does not see it", which is
false. READ THE RAW OUTPUT OF fsck. DO NOT GREP IT FOR THE WORD YOU EXPECT.***

### THE CHECK, TWO READS, RUNNABLE AT EVERY BENCH UNDER A FREEZE

    git count-objects -vH          # size-pack + size: is there mass you cannot explain?
    git fsck --no-reflogs          # READ IT WHOLE. `dangling` AND `unreachable` BOTH COUNT.

**Then, for anything large — `cat-file --batch-check='%(objecttype) %(objectsize)'`.**
***TYPE AND SIZE ONLY. NEVER `cat-file -p` ON A LARGE OBJECT: if it is media, opening it is the
leak, and the blob was not.***

### AND THE SHAPES DIFFER BY SEAT, SO THE CHECK IS NOT OPTIONAL FOR ANYONE

    STORE A ... *** 31 blobs · 3.98 GiB · ~128 MiB each · 0 unaccountable commits ***
                *** MEDIA-SIZED. AND THE ONLY ONE OF THE THREE THAT IS. ***
    STORE B ... 2,959 blobs · 161.75 MiB · *** largest 0.40 MiB *** · 2,360 commits
                source and documents. *** ZERO OBJECTS ABOVE 1 MiB. ***
    STORE C ... 3,211 blobs · *** 0.14 GiB *** · largest 1 MiB · 2,207 commits
                *** BLOBS OVER 100 MiB: 0. ***

***THREE STORES, THREE SHAPES, AND THE 4 GiB IS TRUE OF EXACTLY ONE. A NUMBER ABOUT AN OBJECT
STORE IS MEANINGLESS WITHOUT THE STORE NAMED — the architect was one message from reporting
"4 GiB of media-sized content is in the git object store" as a fact about the repository, and it
is a fact about one seat's clone.***

### AND THEREFORE A `gc` RULING IS PER-STORE AND CANNOT BE ISSUED ONCE

***RULING IT ONCE FOR "THE UNREACHABLE SETS" RULES IT FOR TWELVE STORES WITH TWELVE DIFFERENT
CONTENTS, AND ONLY THE SEAT HOLDING EACH ONE CAN CHECK THE PRECONDITION.*** *Two of the three
above hit the stop condition immediately:* **2,360 and 2,207 unaccountable commits, one set
carrying real subjects and real authors including the owner's — the residue of this campaign's
history rewrites.** ***"ALMOST CERTAINLY THE REWRITE" IS EXACTLY WHAT THE PRECONDITION EXISTS TO
REFUSE, AND IT REFUSED IT TWICE WITHOUT ANYONE HAVING TO ARGUE.***

*Found by `vmsam-dev-1`, which also corrected the architect's premise: these were never reachable
from a commit, so "something committed them and moved the ref away" is false — there has only ever
been one commit in that repository. Replication and non-replication by `vmsam-ci-build`, whose
enumeration also hit the stop condition it was given: 2,360 unaccountable commits, action refused.*
