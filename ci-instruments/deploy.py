#!/usr/bin/env python3
"""Deploy a pushed commit to the test instance, and prove which commit is running.

PRIVATE — lives beside the repository, never inside it (WRITE_ZONES.MD §4).
It names the API host and reads the credential file, so it may not be committed.

The failure mode this exists to remove
--------------------------------------
DEPLOYMENT_WORKFLOW.md step 2 says "CI takes ~10 min; do not fire the hook early".
A hook fired early re-pulls the *previous* image and redeploys the old commit, which
looks like a successful deploy and silently invalidates the test.

"~10 min" is a duration, and a duration cannot be checked. The registry can:
the published image carries the commit it was built from, in the OCI label
`org.opencontainers.image.revision` and in its baked `VMSAM_GIT_COMMIT` env.
So step 2 becomes a measured precondition -- wait until the image *is* the target --
instead of a timer someone has to trust.

Measured 2026-09-03: a documents-only push published in ~90 s, not ~10 min
(cache-from: type=gha, every layer a cache hit). The 10-minute figure is a
worst case for a cold build, not a constant. Waiting it out blind is both
unnecessary when the image is ready and insufficient when the build is slow.

Exit status -- read this, never the output
------------------------------------------
  0  the instance is running the target commit. Verified.
  1  it is not. The deployment did not happen; this says NOTHING about the code.
  2  the instrument could not run. NOT a pass.

Outcome 2 is the one that gets lost. A reader with two outcomes describing a
world with three loses the third, and the missing outcome is always
"the instrument did not run" (RESTART.MD §3.4).

Usage
-----
  ./deploy.py --target <sha>     deploy that commit
  ./deploy.py                    deploy the campaign clone's current HEAD
  ./deploy.py --check            report what is deployed now; fire nothing
  ./deploy.py --wait-image-only  wait for the image, do not fire the hook
"""

import argparse
import json
import subprocess
import os
import sys
import time
import urllib.request

API = os.environ.get("VMSAM_TEST_API", "http://" + os.environ.get("VMSAM_TEST_HOST","showgestionar-test") + ":8080")
REGISTRY_REPO = "studyfranco/vmsam"
IMAGE_TAG = "dev-AI"
HOOK_DIR = "/home/vmsam/src"
HOOK_SCRIPT = "./hook_executor_parameters"
LEAD_CLONE = "/home/vmsam/src/VMSAM"

# git_commit values that are not a SHA. Each means something different.
NOT_A_SHA = {
    # tools.py:226  os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"
    "unknown": "VMSAM_GIT_COMMIT arrived EMPTY or absent at runtime -- the compose "
               "file overrode it (Dockerfile:163 already sets it empty). "
               "The deployment is UNVERIFIABLE, not broken.",
    # Dockerfile:137  ARG VMSAM_GIT_COMMIT="IDK"
    "IDK": "the image was built WITHOUT --build-arg VMSAM_GIT_COMMIT, so it carries "
           "the Dockerfile's ARG default. The deployment is UNVERIFIABLE, not broken. "
           "Note this is a DIFFERENT cause from 'unknown' -- fix it in CI, not compose.",
}


def fail(msg):
    print(f"[instrument] {msg}", file=sys.stderr)
    sys.exit(2)


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


# --- step 2: is the published image the target commit? -----------------------

def image_revision():
    """(revision, created) of the currently published tag, or (None, reason)."""
    try:
        tok = get_json(
            f"https://ghcr.io/token?scope=repository:{REGISTRY_REPO}:pull"
            f"&service=ghcr.io"
        )["token"]
    except Exception as e:
        return None, f"could not get a registry token: {e}"

    accept = ", ".join([
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ])
    base = f"https://ghcr.io/v2/{REGISTRY_REPO}"

    def fetch(ref):
        req = urllib.request.Request(f"{base}/manifests/{ref}")
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("Accept", accept)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)

    try:
        man = fetch(IMAGE_TAG)
        if "manifests" in man:  # multi-arch index -> pick the real platform
            child = next(
                m for m in man["manifests"]
                if m.get("platform", {}).get("architecture") not in (None, "unknown")
            )
            man = fetch(child["digest"])
        req = urllib.request.Request(f"{base}/blobs/{man['config']['digest']}")
        req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=20) as r:
            cfg = json.load(r)
    except Exception as e:
        return None, f"could not read the image manifest/config: {e}"

    rev = (cfg.get("config", {}).get("Labels") or {}).get(
        "org.opencontainers.image.revision")
    if not rev:  # fall back to the baked env, which is what /health will report
        for e in cfg.get("config", {}).get("Env", []):
            if e.startswith("VMSAM_GIT_COMMIT="):
                rev = e.split("=", 1)[1]
    return rev, cfg.get("created")


def _looks_overtaken(rev, target):
    """The registry moved to a DIFFERENT sha. Cannot prove it is newer without the
    repo, so the message says what was seen and does not assert an ordering."""
    return bool(rev) and bool(target) and rev != target

def wait_for_image(target, budget_s, poll_s):
    """Block until the published image carries `target`. True if it does.

    THE REGISTRY HOLDS ONE TAG AND EACH BUILD OVERWRITES IT. So a target can be
    OVERTAKEN: I asked for e1b2b0e while the branch moved on, the registry rebuilt to
    278471a0, and this loop waited for a value that would never return -- silently,
    with an empty log, until the budget ran out.

    A wait for an exact value on a mutable single-slot pointer is a wait that can
    never succeed, and it fails by TIMING OUT rather than by saying why. Report the
    overtake as its own outcome instead of letting it read as "not built yet".
    """
    deadline = time.time() + budget_s
    seen = None
    while True:
        rev, info = image_revision()
        if rev is None:
            fail(f"registry unreadable: {info}")
        if rev != seen:
            print(f"  registry {IMAGE_TAG} = {rev[:12]}  (built {info})")
            seen = rev
        if rev and rev != target and _looks_overtaken(rev, target):
            fail(f"OVERTAKEN: the registry now carries {rev[:12]}, not the requested "
                 f"{target[:12]}. The tag was rebuilt while this waited. Re-target the "
                 f"published image, or the branch has moved past your commit.")
        if rev == target:
            # A READINESS SIGNAL WITH NO ORDERING IS NOT A READINESS SIGNAL.
            # This printed `IMAGE READY: 575f55729fde is published` for an image that
            # is the PARENT of the running container. It was true and useless: a
            # watcher that announces publication without comparing to what is live
            # announces a REGRESSION in exactly the same words as an upgrade, and the
            # recreate that follows moves the container backwards a commit.
            _live = None
            try:
                _h = deployed_commit()
                _live = (_h or {}).get("git_commit")
            except Exception:
                _live = None
            if _live and _live not in NOT_A_SHA and _live != target:
                _anc = subprocess.run(
                    ["git", "-C", LEAD_CLONE, "merge-base", "--is-ancestor",
                     target, _live],
                    capture_output=True)
                if _anc.returncode == 0:
                    fail(f"BEHIND, NOT READY: the published {target[:12]} is an "
                         f"ANCESTOR of the running {_live[:12]}. Recreating onto it "
                         f"would move the container backwards. Wait for a build past "
                         f"the live commit.")
                if _anc.returncode not in (0, 1):
                    print(f"  ORDERING UNKNOWN: cannot compare {target[:12]} to the "
                          f"running {_live[:12]} (git said rc={_anc.returncode}). "
                          f"NOT claiming ready -- an unordered image is not an upgrade.")
                    return False
            print(f"  IMAGE READY: {target[:12]} is published"
                  + (f" and is ahead of the running {_live[:12]}" if _live else
                     " -- LIVE COMMIT UNREADABLE, ordering unverified"))
            return True
        if time.time() >= deadline:
            print(f"  IMAGE NOT READY after {budget_s}s: published tag is still "
                  f"{rev[:12]}, target is {target[:12]}")
            return False
        time.sleep(poll_s)


# --- step 3: fire the hook ---------------------------------------------------

def fire_hook(raw_path):
    """Fire the recreate hook. Returns the parsed head object, or None.

    The response is NOT a single JSON document: it is one JSON object followed by
    trailing data, so `json.load(sys.stdin)` -- the filter DEPLOYMENT_WORKFLOW.md
    prints -- dies with "Extra data". Measured 2026-09-03: extra data at char 31571.
    raw_decode() reads the leading object and ignores the rest. The full response is
    kept on disk because it embeds the host firewall log, which is noise until the
    day it is not.
    """
    try:
        p = subprocess.run(
            ["bash", HOOK_SCRIPT], cwd=HOOK_DIR, capture_output=True, timeout=300
        )
    except Exception as e:
        fail(f"the hook did not run: {e}")

    out = p.stdout.decode("utf-8", "replace")
    with open(raw_path, "w") as fh:
        fh.write(out)

    try:
        head, _ = json.JSONDecoder().raw_decode(out.lstrip())
    except Exception as e:
        print(f"  hook output is not JSON at all ({e}); raw kept at {raw_path}")
        return None
    print(f"  hook: status={head.get('status')} exit_code={head.get('exit_code')} "
          f"duration_ms={head.get('duration_ms')}")
    return head


# --- step 4: what is actually running? ---------------------------------------

def deployed_commit():
    try:
        return get_json(f"{API}/health", timeout=5)
    except Exception:
        return None


def poll_health(target, tries, gap):
    """Returns (verdict, health). verdict in {ok, mismatch, unverifiable, silent}."""
    last = None
    for _ in range(tries):
        h = deployed_commit()
        if h:
            last = h
            got = h.get("git_commit", "")
            if got == target:
                return "ok", h
            if got in NOT_A_SHA:
                return "unverifiable", h
        time.sleep(gap)
    return ("mismatch", last) if last else ("silent", None)


def report(verdict, health, target):
    if verdict == "ok":
        # A RECREATE IS THE FREE MOMENT TO RESTART THE RUNNER, AND ONLY HERE.
        # merge_queue.py reads the container image ONCE at startup and stamps every row
        # with it; the patched image_now() reads it PER ROW and is in the file, unloaded.
        # Every recreate while the old process runs adds another boundary to a correction
        # record I have already had to amend once -- and I am the only reader who would
        # notice it had gone stale.
        #
        # A mid-run restart risks a double-POST, because the container keeps working on
        # the job the old runner was tracking. THAT COST IS ALREADY PAID HERE: this
        # recreate has just killed that job. So this is the one instant where the restart
        # is free, and the instrument says so rather than my remembering it.
        print("\n  REMINDER: this recreate just forfeited the in-flight job, which makes")
        print("  NOW the only free moment to restart merge_queue.py. That loads")
        print("  image_now() and ends the per-row image-label correction; skipping it")
        print("  adds another boundary to that record.")
        print(f"\nDEPLOYED: {target}")
        print(f"  mode={health.get('mode')} dev={health.get('dev')} "
              f"is_running={health.get('is_running')}")
        return 0
    if verdict == "unverifiable":
        got = health.get("git_commit")
        print(f"\nUNVERIFIABLE: /health reports git_commit={got!r}")
        print(f"  {NOT_A_SHA[got]}")
        print("  No result obtained in this state proves anything. This is the "
              "third outcome, and it is the one that gets lost.")
        return 1
    if verdict == "mismatch":
        print(f"\nNOT DEPLOYED: /health reports {health.get('git_commit')}, "
              f"target is {target}")
        print("  The deployment did not happen. This says NOTHING about the code "
              "under test -- never report it as a failure of a change.")
        print("  Usual cause: the image was not published yet when the hook fired. "
              "Re-run this script (it waits on the registry); do not push again.")
        return 1
    print("\nNO ANSWER from /health: the container did not come back.")
    print("  Re-fire the hook and read status/exit_code.")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="SHA to deploy (default: campaign clone HEAD)")
    ap.add_argument("--check", action="store_true",
                    help="report what is deployed now; fire nothing")
    ap.add_argument("--wait-image-only", action="store_true",
                    help="wait for the image; do not fire the hook")
    ap.add_argument("--image-budget", type=int, default=1800,
                    help="seconds to wait for CI to publish (default 1800)")
    ap.add_argument("--poll", type=int, default=20)
    ap.add_argument("--raw", default="/home/vmsam/src/VMSAM_HELP_AI/ci/runs/hook-last.json")
    a = ap.parse_args()

    if a.check:
        h = deployed_commit()
        if not h:
            fail("no answer from /health")
        rev, created = image_revision()
        print(f"deployed : {h.get('git_commit')}")
        print(f"           mode={h.get('mode')} dev={h.get('dev')} "
              f"is_running={h.get('is_running')}")
        print(f"published: {rev}  (built {created})")
        if h.get("git_commit") in NOT_A_SHA:
            print(f"NOTE: {NOT_A_SHA[h['git_commit']]}")
        print("in sync  :", h.get("git_commit") == rev)
        return 0

    target = a.target
    if not target:
        try:
            target = subprocess.run(
                ["git", "-C", LEAD_CLONE, "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception as e:
            fail(f"could not read the campaign clone HEAD: {e}")
    if len(target) != 40:
        fail(f"target {target!r} is not a full 40-character SHA")

    print(f"target: {target}")
    print("step 2 -- waiting for the published image to carry the target")
    if not wait_for_image(target, a.image_budget, a.poll):
        print("\nSTOPPING before the hook. Firing now would redeploy the OLD image "
              "and look like a success.")
        return 1
    if a.wait_image_only:
        return 0

    print("step 3 -- firing the recreate hook")
    fire_hook(a.raw)

    print("step 4 -- verifying the deployed commit")
    verdict, health = poll_health(target, tries=20, gap=6)
    return report(verdict, health, target)


if __name__ == "__main__":
    sys.exit(main())
