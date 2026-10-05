# Owner ruling, 2026-09-08 — the derived permission on repair/resample is void

Archived verbatim from the pre-restructure WRITE_ZONES.MD (commits 03464faf, a7a2761f,
author studyfranco). The operative rule lives in WRITE_ZONES.MD §4; this is the full
reasoning, measurements and credits.

> **THAT DERIVATION IS VOID AS OF 2026-09-08, AND NOBODY MAY EDIT `src/merge_video_repair.py`
> OR `src/merge_video_resample.py` ON THE STRENGTH OF THE SENTENCE ABOVE.** I am not
> withdrawing a permission the owner gave — he gave one, for `merge_video_chimeric.py`, and
> that one stands untouched. I am withdrawing an inference *I* let stand, because its premise
> no longer exists and because the paragraph immediately below already contradicts it.

**FIRST: THE PREMISE IS UNAVAILABLE, AND FOR THE ONE FILE WHERE IT IS NEEDED.** `lead` found
this against its own operation. Measured at the authority:

| file | blob | lines | commits | beyond the squash |
|---|---|---|---|---|
| `src/merge_video_resample.py` | `8db18543923a` | 184 | **1 — the squash, and nothing else** | **0** |
| `src/merge_video_repair.py` | `ea523c009683` | 2280 | 4 | 3 |
| `src/merge_video_chimeric.py` | `46fb07f0ec5a` | 3064 | 2 | 1 |

*Control: 483 `src/` commits all time, so the predicate fires and a zero here is a real zero.*

`8b9a5b8a` is titled *"squash: collapse the multi-agent campaign's working history into one
commit"* and touched 49 files. For `merge_video_resample.py` it is the ONLY commit, and the
store does not say so: `git log` answers `vmsam-lead` with complete confidence and no
information.

> **CORRECTION, 2026-09-08, TO THIS SECTION'S OWN REASONING.** I first wrote here that *"the
> single fact this permission rests on is precisely the fact the squash erased."* **THAT IS
> WRONG, AND THE SQUASH IS NOT WHAT MADE THE PREMISE FALSE.** `lead` found that
> `refs/backup/pre-squash-dev-AI` exists at the authority — the branch state immediately
> before the operation — and reported it precisely because both of us had reasoned from the
> evidence's absence. I fetched it into a scratch namespace, measured, and deleted my import.
> The pre-squash state says:
>
> | file | commits before the squash | author |
> |---|---|---|
> | `src/merge_video_resample.py` | **1** | **`studyfranco` — the OWNER, not an agent** |
> | `src/merge_video_repair.py` | 4 | 3 `vmsam-lead`, 1 `studyfranco` |
> | `src/merge_video_chimeric.py` | 2 | 1 `vmsam-lead`, 1 `studyfranco` |
>
> *Control: the pre-squash state distinguishes five author identities across `src/`, so the
> field carries information there and a uniform answer would have been a real finding.*
>
> **SO THE PREMISE WAS NEVER TRUE FOR `merge_video_resample.py`.** It did not arrive from the
> agent the derivation names; it arrived in a commit authored by the owner. The evidence was
> not destroyed — **it exists, it is readable, and it REFUTES the premise rather than being
> silent about it.** The ruling above stands and is strengthened; only my account of why was
> wrong, and `lead` should not carry a self-charge for erasing something that says the
> opposite of what either of us assumed.
>
> **AND THE GENERAL FORM IS SHARPER THAN THE ONE I ORIGINALLY DREW.** I inferred a fact's
> destruction from its absence at the tip, when a namespace outside the branch held it. So:
> **ABSENT FROM THE HISTORY YOU LOOKED AT IS NOT ABSENT.** Before asserting that a repository
> operation destroyed evidence, look in `refs/backup/*` — those refs are named for the
> operations that remove things (`pre-rewrite`, `prereset`, `prerebase`, `pre-squash`), which
> is exactly why they hold what an operation took, and they are reached by no lander and
> therefore gated by nothing. Credit: **lead**, who reported the ref's existence knowing it
> undercut its own confession; **dev-3**, whose rule that a `fetch` is a write to your
> repository's reachable set is why the import was scratch-namespaced and deleted.

> **A DEAD PATH ANNOUNCES ITSELF. AN ATTRIBUTION FLATTENED BY A SQUASH RETURNS A CONFIDENT
> WRONG NAME.** §1c's defect in a third material — there a name DOUBLED, here a name ABSORBED,
> and neither has an arm that fires.

**SECOND, AND IT DOES NOT DEPEND ON THE SQUASH AT ALL: THE NEXT PARAGRAPH ALREADY SAYS NO.**
Six lines below, this document rules *"a file becomes shared by being useful and imported, not
by anyone deciding it is."* Measured at the authority tip, importers inside `src/` excluding
the file itself:

    merge_video_resample.py   2 importers      merge_video_chimeric.py   2 importers
    merge_video_repair.py     3 importers

**All three are imported. By this document's own test they are shared files, and the
derivation was arguing the opposite of the rule printed underneath it.** That defect predates
the squash and would have been there had `lead` never run it.

**THIRD: `zone A` IS NOT THIS FAMILY, AND THE DOCUMENT INVITES THE MISREADING.** `zone A` is
`remove_not_compatible_video` in `mergeVideo.py` — see the zone table above.
`src/merge_video_resample.py` appears in this entire document exactly ONCE, in the prose
sentence above, and in no zone table anywhere. `dev-2` read it as "zone A, mine" in good
faith, and this family has already misread itself once: an instrument scored
`merge_video_chimeric.py` FROZEN while the sentence above it said fully editable.

**WHAT HAPPENS NOW.** The permission on these two files is the owner's to grant or refuse, and
it is with him as an open item. Until he rules, work on them is built inert in the agent's own
lab and does not land. `lead` has already told `dev-2` exactly that, before asking me.

**AND THE RULE THIS LEAVES BEHIND, WHICH IS THE PART THAT GENERALISES:**

> **A DERIVED PERMISSION IS ONLY AS DURABLE AS ITS PREMISE, AND A REPOSITORY OPERATION CAN
> DESTROY A PREMISE WITHOUT TOUCHING THE SENTENCE THAT USES IT.** Same signature as §1c: no
> diff shows it, no line count moves, and the sentence still reads correctly. **So a permission
> must rest on a property that is CHECKABLE AT THE TIP** — absent from `master`, no importer,
> named in a zone table — **never on a property of the history, which any rewrite can flatten.**

Credit: **lead** (found it, against its own squash, and proposed nothing); **auditor** (whose
four-commit authorship check on `merge_video_repair.py` survives on its three post-squash
commits — the verdict holds, the METHOD does not generalise, and it fails silently on exactly
the file where it is needed); **arch-aide** (the doubling half of the same defect);
**dev-2** (whose good-faith misreading located the zone-table gap).

