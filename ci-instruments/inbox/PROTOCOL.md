# THE SHARED CI MAILBOX — one box, three pickers, claimed by atomic rename

**The owner's design:** *"Qu'ils aient une boîte aux lettres COMMUNE et que chacun traite
les messages en PIOCHANT dedans. Les 3 agents doivent avoir un LOOP sur la lecture de leur
dossier."*

    vmsam-ci         MEASUREMENT -- the batch, THE RUNNER, the ledger
    vmsam-ci-build   BUILD -- deploy, image recreation, the corpus generator, fixtures
    vmsam-ci-pair    PAIRED RUNS on test2 when it exists, and the harness

## CLAIM BY ATOMIC RENAME. NOTHING ELSE WORKS.

    os.rename(f"inbox/{msg}", f"inbox/claimed/{me}/{msg}")

**Two agents WILL read the same file in the same instant.** *No "check before you take"
fixes it -- the check and the take are two operations and the gap between them is where the
second agent arrives.* **POSIX `rename(2)` is atomic: exactly one wins, the losers get
`ENOENT` and move to the next file.** No lock, no protocol, no polling.

**VERIFIED HERE before this file was written:** inbox and `claimed/` are on device 69, one
rename raced two ways, one winner, the loser got `ENOENT`.

### THE TRAP, AND THIS FLEET HIT IT THIS WEEK

> **`os.replace` returns `EXDEV` ACROSS A MOUNT BOUNDARY.** dev-2 met it building the
> durable store and switched to `shutil.move` -- **and `shutil.move` FALLS BACK TO
> COPY-THEN-DELETE, WHICH IS NOT ATOMIC** and would silently reintroduce double processing
> with nothing in any log.

    THE CLAIM DIRECTORY MUST BE ON THE SAME FILESYSTEM AS THE INBOX.
    IT IS INSIDE THE INBOX SO IT CANNOT BE ANYWHERE ELSE BY CONSTRUCTION.
    NEVER MOVE `claimed/` SOMEWHERE TIDIER. USE os.rename, NEVER shutil.move.

## LOOP ON THE FOLDER WITH A MONITOR, NOT A POLL

**USE `VMSAM_HELP_AI/ci-build/watch_inbox.py`, sha256:33a2c9fc6df2 (202 lines -- sha256:80775921bc2e ANNOUNCED ONE MESSAGE TWICE ON THE REAL BOX AND MUST NOT BE USED).** *A polling loop burns the
window on empty checks; a monitor costs nothing while the box is empty.*

> **CORRECTED ~23:45Z. THIS FILE AND TWO BRIEFS SAID "COPY AIDE'S MONITOR". THERE IS NO SUCH
> ARTEFACT.** aide ships `sweep.sh` and `cadence.sh` and **BOTH ARE SINGLE-PASS — they run
> once, read, and exit.** *What aide has is a HARNESS FEATURE: a background task in its
> runtime that delivers arrival events as notifications. Not a file, not a daemon, and not
> obtainable by reading anything it has written.*

**aide's own diagnosis, against its own description:**

    A CAPABILITY DESCRIBED WITHOUT ITS SUBSTRATE READS AS AN ARTEFACT.
    "I HAVE A MONITOR" AND "MY RUNTIME NOTIFIES ME" ARE IDENTICAL SENTENCES
    AND COMPLETELY DIFFERENT INSTRUCTIONS TO A THIRD PARTY.

**The Lead asserted it from a description, never checked, and put it in three briefs.**

### WHAT TO USE, AND WHY IT IS NOT A SECOND DIALECT

**`inotifywait` is not installed, `watchdog` does not import, `inotify_simple` does not
import — so `watch_inbox.py` calls `inotify(7)` THROUGH LIBC VIA `ctypes`, stdlib only.**
*The same kernel mechanism `inotifywait` uses: the protocol's own mechanism without the
missing binary.*

**AND IT DOES AN INITIAL SCAN AT STARTUP**, so a message that arrived while nobody was
watching is not missed — *the failure mode a naive watcher has.*

    A DIALECT CHOSEN BY WHOEVER WROTE FIRST IS NOT AUTOMATICALLY WRONG.
    IT IS AUTOMATICALLY UNDECLARED.   (aide)

### THE MONITOR DOES NOT CLAIM, AND THAT IS LOAD-BEARING

**It prints one line per message so the agent wakes; THE AGENT CLAIMS IN ITS OWN TURN.**
*A watcher that claims eagerly takes messages while its agent is forty minutes into a build
— and a message in `claimed/<me>/` IS INVISIBLE TO THE OTHER TWO PICKERS, where the same
message left in the box is visible to all three.* **A claim is a promise to process, and a
promise made by a background process on behalf of a busy agent is a message quietly removed
from a shared queue.**

### NOT YET EXECUTED

**aide checked its syntax and import resolution, explicitly did NOT run it, and does NOT
claim it works** — *it refused to enter ci-build's zone to test it, which is right.*
**ci-build runs it once and reports before the other two adopt: SYNTAX AND IMPORTS ARE NOT
EXECUTION**, the same distinction as a gate that checks for a token instead of interrogating
the thing.

## THE STORE IS THE AUTHORITY, NOT THE AGENT

**The owner ruled that ci is the entry point for the test containers. With three agents that
ruling is KEPT and made safer:**

    `VMSAM_HELP_AI/ci/` HOLDS THE CONTAINER RECORDS. ANY OF THE THREE READS IT.
    NONE KEEPS A PRIVATE COPY. NONE ANSWERS FROM CONTEXT.

**One agent holding it in its head was a single point of failure; three agents reading one
file is not.** *There is still ONE authoritative source for "what are the containers" -- it
is a FILE rather than a SESSION.*

    WHEN ANY OF THE THREE LEARNS SOMETHING ABOUT A CONTAINER, IT WRITES IT TO
    THE STORE **BEFORE** ANSWERING ANYONE. Otherwise the answer lives in one
    context and dies with it -- which is why ci wrote SECOND_CONTAINER.md
    instead of acknowledging a message.

## WHAT IS NOT SHARED

**THE RUNNER.** *Two agents driving one batch is the collision the mailbox cannot solve,
because it is not a message.* **vmsam-ci owns it alone.**
