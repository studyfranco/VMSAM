#!/usr/bin/env python3
"""Is the owner's condition for updating the test container met?

    "Quand tous les push sont faits et que tout est bon pour le HTML,
     ci peut mettre a jour le container de test."

EVERY ANSWER COMES FROM A LIVE QUERY, NAMING THE URL. A tracking ref answers
"what did I see last time", never "what is there now" -- and a cache answers
always, with confidence. The architect reported two falsehoods to the owner in one
hour from a stale ref; I inferred a failed push from an absence twice.

THREE OUTCOMES BY NAME, never a boolean:
    0  MET        -- the wiring is on the published branch
    1  NOT MET    -- measured, with the missing piece named
    2  CANNOT TELL -- the remote did not answer; NOT the same as not met
"""
import subprocess, sys

URL  = "ssh://git@forgejo:2222/fallrik/VMSAM.git"
# EVERY git COMMAND MUST NAME ITS REPOSITORY. The first version ran bare `git fetch`
# and inherited the caller's cwd -- run from my own directory it found no repo and
# reported CANNOT TELL on a condition that was MET. The three-outcome design is what
# saved it: a boolean would have said NOT MET and I would have told the architect its
# push had not landed, which is precisely the false report I made about dev-2 tonight.
REPO = "/home/vmsam/src/VMSAM"

def run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def main():
    r = run(["git","-C",REPO,"ls-remote",URL,"refs/heads/dev-AI"])
    if r.returncode != 0 or not r.stdout.strip():
        print(f"  CANNOT TELL -- {URL} did not answer")
        print(f"    {r.stderr.strip()[:120]}")
        return 2
    tip = r.stdout.split()[0]
    print(f"  live tip   {tip[:12]}   from {URL}")

    # SYMMETRIC FRESHNESS. I checked forgejo against my tracking ref all evening --
    # ONE DIRECTION. dev-1 found the hop AHEAD of forgejo by two commits, which is the
    # case that reports "fine" while diverging: every agent's `origin` IS the hop, so
    # work can sit there unseen by the true remote indefinitely.
    # A FRESHNESS CHECK THAT LOOKS ONE WAY REPORTS FINE WHEN IT FINDS THE OTHER.
    hop = run(["git","-C",REPO,"rev-parse","refs/heads/dev-AI"]).stdout.strip()
    if hop and hop != tip:
        ahead = run(["git","-C",REPO,"rev-list","--count",f"{tip}..{hop}"]).stdout.strip()
        behind = run(["git","-C",REPO,"rev-list","--count",f"{hop}..{tip}"]).stdout.strip()
        print(f"  hop        {hop[:12]}   AHEAD {ahead or '?'}  BEHIND {behind or '?'}")
        if ahead and ahead != "0":
            print(f"    the hop carries {ahead} commit(s) forgejo has not seen --")
            print(f"    they are NOT in any image built from a forgejo clone")
    elif hop:
        print(f"  hop        {hop[:12]}   identical to forgejo")

    # fetch so the tip's tree is readable, then ask the TREE, not a ref
    run(["git","-C",REPO,"fetch","origin"], timeout=180)
    if run(["git","-C",REPO,"cat-file","-e",tip]).returncode != 0:
        print("  CANNOT TELL -- tip not fetchable into this store")
        return 2

    # THE CONDITION IS A CALL SITE, NOT A COMMIT NAME. A commit list can be
    # satisfied by commits that do not wire anything; the call site cannot.
    # A BARE NAME IS A PROXY FOR WIRING, AND IT REPORTED A CHANGE WHEN A COMMENT
    # CHANGED. `vmsam-dev-1` caught this: their new `_emit` docstring cites
    # "merge_plan_report.py:489" and "the allowlist entry in merge_plan_report.py" to
    # record that their line and dev-4's allowlist must land together. My gate counted
    # both as CALL SITES and went from 1 to 2 -- so a number stable all night moved
    # because someone wrote a citation.
    #
    # WORSE THAN A MISCOUNT: if the only match anywhere were prose, this gate would
    # report MET over a module nothing invokes -- which is the exact failure it was
    # built to replace, when it checked a COMMIT NAME instead of a call site.
    #
    # So: match the two things that are actually wiring. `merge_plan_report.py:489`
    # does not match `\.\w+\(`, and a citation in prose does not match an import.
    pat = r"(^|[^A-Za-z_.])merge_plan_report\.[A-Za-z_][A-Za-z_0-9]*\(|^[[:space:]]*(import|from)[[:space:]]+merge_plan_report"
    g = run(["git","-C",REPO,"grep","-lE",pat,tip,"--","src/"])
    files = [l.split(":",1)[1] for l in g.stdout.splitlines() if ":" in l]
    callers = [f for f in files if not f.endswith("merge_plan_report.py")]
    print(f"  files mentioning merge_plan_report in src/: {len(files)}")
    for f in files:
        print(f"    {f}{'   <- CALL SITE' if f in callers else '   (the module itself)'}")
    if not callers:
        print("\n  NOT MET -- zero call sites on the published branch.")
        print("  The module ships and nothing invokes it, so no report is produced")
        print("  and there is nothing for the container to test.")
        return 1
    # A CALL SITE IS STILL AN EXISTENCE CHECK ON A TOKEN. Measured on the poison ref
    # f99bb98 (published for ~1 minute on 2026-09-04): this gate printed
    # "MET -- 1 call site(s)" over a tree whose src/change_point_locator.py carried
    # three conflict markers and died at line 159. THE BROKEN FILE WAS NOT THE FILE
    # I GREPPED, so grepping harder could never have found it.
    #
    # A force-push moves a ref; it does not unfetch an object. Nothing here pins a
    # sha -- `tip` comes from ls-remote, so a repaired branch repairs this gate --
    # but a BRANCH CAN CARRY POISON, and for that minute it did.
    #
    # Compiling is the first check that interrogates the thing rather than its name.
    import tempfile, shutil, os, pathlib, py_compile
    tmp = tempfile.mkdtemp(prefix="deploycond-")
    try:
        ar = subprocess.run(["git","-C",REPO,"archive",tip,"src/"],
                            capture_output=True, timeout=120)
        if ar.returncode != 0:
            print("\n  CANNOT TELL -- could not read the tip's src/ tree")
            return 2
        subprocess.run(["tar","-x","-C",tmp], input=ar.stdout, timeout=120)
        srcs = sorted(pathlib.Path(tmp, "src").rglob("*.py"))
        bad = []
        for f in srcs:
            cf = os.path.join(tmp, "__c", str(f).replace(os.sep, "_") + "c")
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            try:
                py_compile.compile(str(f), cfile=cf, doraise=True)
            except py_compile.PyCompileError as e:
                bad.append((os.path.relpath(str(f), tmp),
                            str(e).strip().splitlines()[-1][:80]))
        print(f"  src/ compiles: {len(srcs) - len(bad)} of {len(srcs)} files")
        if bad:
            print(f"\n  NOT MET -- {len(bad)} file(s) on the published branch do not compile.")
            for f, err in bad:
                print(f"    {f}\n      {err}")
            print("  The wiring is present and the tree is broken. Deploying this")
            print("  builds an image that cannot run the code the call site names.")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n  MET -- {len(callers)} call site(s) on the published branch, tree compiles.")
    return 0

sys.exit(main())
