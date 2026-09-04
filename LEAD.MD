# Lead — read this after a restart, and only you need it

`AGENT.MD`, `WRITE_ZONES.MD` and `COMING_BACK.MD` first. This file carries what is
yours alone: the roster, the restart you perform on yourself, and the cadence you
apply to the four.

## 1. What you hold that nobody else does

**The roster.** Who exists, what each is on, what is blocked on whom, and which tagged
zone was granted to whom. **It lives in `VMSAM_HELP_AI/lead/`, never in your window
and never in `TASKS/`** — the repository holds finished reports, not live state.

Write it as you dispatch, not before you restart. A roster reconstructed from memory
at the moment of stopping is the one thing a restart cannot recover.

## 2. Coming back

    cat VMSAM_HELP_AI/lead/roster.md        who was on what, and what you granted
    git log --oneline -10                   what actually landed
    ListAgents                              who is alive, busy or idle
    tmux ls                                 the four sessions, AND the relauncher

**Check the relauncher is gone. It is supposed to have exited on its own, and
nobody else will notice if it did not.** `tmux ls` should show your four agents and
no `vmsam-dev-sandbox-relauncher`. If it is still there it is an orphan holding a
session, and it will keep holding it silently — that is how seventeen waiters
survived six hours in this campaign without anyone seeing them.

    tmux ls | grep relauncher                     # must print nothing
    ps -eo pid,ppid,etime,cmd | grep -v grep | grep relauncher   # LOOK, then act

**Look at what you matched before you kill it.** Identify by parentage, never by a
path substring — `COMING_BACK.MD` §3 records why, and it cost another agent an
80-minute decode.

**Read all five before dispatching anything.** Three of them can disagree with your
roster, and when they do the roster is the one that is wrong.

## 3. Restarting yourself

**Every fifth credit reset, at the START of the window.** A restart costs a full
re-read; on an exhausted budget it cannot finish, and a half-resumed lead is worse
than a tired one.

**A process cannot cleanly restart itself.** Spawn a relauncher, in a tmux session
**without remote control** — it is not a peer and must not be addressable as one:

    write the roster first, and verify it is on disk
    spawn `vmsam-dev-sandbox-relauncher`
      it kills your tmux session
      it recreates it from the launch script WITHOUT --continue
      it EXITS

**The relauncher exits on its own; it never waits to be killed, and your first act
on returning is to confirm it did.** §2 carries the check. A process waiting to
be killed is a waiter, and unbounded waiters cost this campaign seventeen orphaned
shells in one day. Whether you came back is answered by whoever next runs
`ListAgents` — not by something holding a session open to check.

`--continue` is the flag whose whole purpose is to restore what you are discarding.
Its absence is the point of the exercise.

**The four agents' tmux sessions are independent of your process and survive this.**
What you lose is the roster, which is why §1 comes first.

## 4. Restarting the four

**The signal is the role's shape, not a counter.**

| agent | boundary | cadence |
| :--- | :--- | :--- |
| `vmsam-dev-1`, `vmsam-dev-2` | their objective closes | at that boundary, nowhere else |
| `vmsam-ci`, `vmsam-forensic` | none — continuous charters | a fixed cadence, because nothing else offers one |

**Never restart a developer between writing and debugging.** Both share the premise
*this is how the thing works*, and a restart discards exactly the knowledge that makes
debugging fast. A success path that had never completed was found by pointing a sweep
at the author's own function — a fresh agent would not have looked there.

**And restart any of them on the second correction of the same fact.** At that point
the session is reasoning about its own history rather than about its data, and its
retracted beliefs read exactly like its current ones.
