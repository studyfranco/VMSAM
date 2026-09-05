# `showgestionar-test2` — announced 2026-09-05, DOES NOT EXIST YET

**PRIVATE.** Stored durably rather than acknowledged, because a message does not survive
a session and this changes what my measurements mean.

    OWNER, relayed by the architect 2026-09-05:
    "Lors de la prochaine mise a jour de container, ci aura un SECOND container
     pour ses tests end-to-end: showgestionar-test2."

**STATE AS OF THIS WRITING: it does not exist.** `WRITE_ZONES.MD` §14 — a capability
claim carries the build it was judged against, and *"I have two containers"* is a
capability claim that is **FALSE until the next container update lands.** Anything I
measure before then is single-container work and must not be described otherwise.

## What it buys, and it is the thing tonight lacked

**I CAN RUN THE NULL.**

With one container a control run and a treatment run are **sequential on the same
apparatus**, so they share the image, the queue state, the output tree and the
preserver. **Every before/after I produced tonight was two readings of one apparatus at
two times, and the apparatus changed twice** — `PLATEAU_TOLERANCE_MS` 50→5, and a
41-second window in which a tree that could not import its own locator was published.

**Two containers on ONE image is a paired run: one arm gets the change, one does not,
both at the same instant.** That is the difference between *the number moved* and *the
change moved it*.

**Second, smaller, immediate: the deploy gate stops competing with the batch.** I came
close to recreating onto a broken tree while a 100-file run was in flight. With two,
`test2` is rebuilt without touching the one that is measuring.

## THE COLLISION TO DESIGN AGAINST BEFORE THE BUILD

dev-2's durable store, landed tonight:

    UNDELIVERED_STORE_DEFAULT = "/config/output/undelivered"
    nested under stable_case_key(candidate_path) = md5(path)[:16]
    never overwrites -- SUFFIXES .1 .2 on collision

**`stable_case_key` is keyed on the CANDIDATE PATH and deliberately not on container
startup** — dev-2's reason is that a startup-keyed path would make the same refusal look
like a new artefact after every recreate. **Correct for one container.**

> **WITH TWO: ONE CASE, REFUSED ONCE, OBSERVED BY BOTH CONTAINERS, LANDS AT THE SAME
> `case_key` — AND THE NON-OVERWRITE SUFFIX MAKES ONE REFUSAL LOOK LIKE TWO.**

**That is `absent is never zero` inverted: a count inflated by an observer rather than
deflated by a silence.** And it is invisible in a directory listing — `.1` is exactly
what a genuine second refusal looks like.

**Three possible fixes, none mine to choose alone:** the store path carries the
observing container; or the artefact carries it; or **my counting deduplicates on
`case_key` + CONTENT DIGEST rather than on filename.** The third is entirely mine and
needs no agreement — and my `DECLINED/` ledger already records `size_at_capture`, which
is the cheap half of a content check.

## What must change in my instruments

**Every measurement I publish has named the IMAGE. From the next update it must also
name WHICH CONTAINER** — two containers on one image can diverge in queue state, output
tree and elapsed time, so *"on image `<id>`"* stops being a complete reference.

    MEASURED 2026-09-05: six hardcoded `showgestionar-test` endpoints across five files

        deploy.py                  the deploy/health probe
        merge_queue.py             API constant -- the runner's whole conversation
        start_when_idle.sh         the idle-wait launcher
        targeted_probe.py          one-off probes
        test_merge_plan_report.py  the provenance banner's /health read

**None of them can express "which container".** Parameterising is a one-line change per
site (`os.environ.get('VMSAM_TEST_HOST', 'showgestionar-test')`) and is worth doing
BEFORE `test2` exists, not after — because the failure mode is silent: a probe aimed at
the wrong container returns a valid answer about the wrong apparatus.

**And `merge_queue.py` is the one that matters**: its rows would carry
`image_git_commit` and no container identity, so a paired run's two arms would be
indistinguishable in the record. That is the image-label defect again, one substrate
out, and I have already had to amend that correction twice tonight.
