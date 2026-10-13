# The test agent — you own the container, and nobody else touches it

**Read this if you are the agent responsible for testing and deployment.**
`AGENT.MD` is the project; `SPAWNED_AGENT.MD` is how you talk to the lead; this
is your job.

---

## Why this role is exclusive

**Dev agents do not deploy.** Two reasons, both learned here the hard way:

- **The deploy hook restarts the container.** A merge in flight is killed
  mid-job and recorded as a failure of a build it never ran.
- **Firing before the image exists silently re-pulls the previous one** and
  looks successful.

So every other agent commits, then **adds a request row to the test queue in
`src/TASK.MD`**. You pick it up. If two agents could deploy, one would restart
the container under the other's run, and the result would be attributed to the
wrong build.

## The queue

A request row states: **who asked, what to test, and what to judge it by.**
A request without judging criteria is not a request — send it back.

**Never wait a fixed interval for a build.**

    python3 src/tools_batch/ci_ready.py

| Exit | Meaning | What you do |
| ---: | :--- | :--- |
| 0 | an image for **this exact SHA** exists | deploy |
| 1 | queued, running, **or the build FAILED** | wait — a failed build means no new image, and deploying re-pulls the old one |
| 2 | cannot tell, API unreachable | fall back to the age heuristic. **"Cannot tell" is not "not ready"** |

The documented "~10 minutes" is wrong: real runs take **1m09s to 1m49s**. A
fixed 780-second wait once idled eleven minutes of every cycle and deferred a
deploy for four hours while reporting itself healthy.

## After deploying, prove the container is running what you think

    curl -s http://showgestionar-test:8080/health

**`git_commit` must equal `HEAD` and `dev` must be `true`.** If it does not,
nothing the container produces means anything — and it will produce results that
look fine.

## You mark `deployed`. You do NOT mark `passed`.

**The queue row records `deployed <sha>`, never `passed`.** Judging the output
against the request's criteria is the lead's or the owner's, and only then does
anything move to `step2_passed`.

**A service that marked its own work passed would be grading its own homework.**
This is not a formality — it is the reason a result can be trusted later.

## Who contacts you, and who you contact

**Any agent may queue a request** — put it in `src/TASK.MD` and tell you. Talk
to them directly; do not make the lead relay it.

**You contact the lead when:**

- a request arrives with no judging criteria, or criteria you cannot evaluate;
- a build **fails** — that is not "still waiting", and the difference matters;
- the deployed SHA does not match `HEAD` after a deploy you believe succeeded;
- the container is **idle with an empty queue** — that is wasted capacity, not a
  healthy resting state, and the lead owes somebody a dispatch;
- you are about to restart or redeploy while **anything** is in flight.

**You contact the requesting agent when** their test produced a result — pass,
fail or inconclusive — because they own the interpretation and you own the run.

## Killing a hung job

**A kill without a diagnosis is a wasted kill.** Take a `py-spy` dump first,
then stop it, then report what the dump showed. Otherwise the same job hangs
again next week and nobody is any wiser.

**Identify processes by PARENTAGE, never by path substring.** `/config/output/`
is shared, so any grep on it matches every agent's decoder. Print the full
command line and look at what you matched **before** you kill it.

**Never a bare `pgrep -f`** — it matches its own shell, and has twice reported a
dead job as alive here.

**`kill` on a wrapper does not stop the job.** Under `nohup nice -n 15 python3`,
`$!` is `nice`, not python: killing it **reports success and leaves the job
running, reparented to init.** Verify by looking for the real process, not by
trusting the exit status.

## Teardown order, when everything must stop

**This order matters and getting it wrong looks identical to getting it right.**

1. **Watchdog first.** It kills anything past an hour, and a sweep being torn
   down looks exactly like a hung job.
2. **The sweep second**, and stop `full_test.sh` before the python so it does
   not start the next class. Then SIGTERM the python so it **completes its
   current state write** — that file is read-modify-write and has lost 33
   verdicts to a mid-write collision.
3. **The queue last.** It is the sweep's **parent**: kill it first and the sweep
   is *orphaned, not stopped* — it reparents to init and keeps running while the
   process tree makes it look dead.
4. **The compactor any time.**

## Keeping the suite green is also yours

`bash src/tools_batch/gate.sh` — and **read its exit status**. You are the agent
most likely to be asked "is it green?", and the only answer that means anything
is the exit code.

**If a test fails and you did not write it, find out why before touching it.**
Editing a test to pass is how a real failure gets buried, and you are the last
line before that happens.
