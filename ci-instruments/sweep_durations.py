#!/usr/bin/env python3
"""Pair every error entry with its master and compare durations.

PRIVATE — reads real media paths. Never inside the repository.

Why duration and not the filename
---------------------------------
The obvious sweep — flag an error file whose name does not match its folder — is
wrong here, and measurably so. Many folders legitimately mix title conventions for
the SAME work: a folder named for the romaji title holding an English-titled
release, or a Chinese-titled one. Screening on names would flag all of those and
call them misfiled.

Duration is a property of the content. It does not settle the question by itself
either — a different cut of the same episode also differs — so this produces
CANDIDATES, ranked, to be verified by looking at frames. It is a screen, not a
verdict.

Media is mounted read-only; this only reads.
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

RUNS = "/home/vmsam/src/VMSAM_HELP_AI/ci/runs"
EP_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})")


def duration(path):
    """Seconds, or None if ffprobe could not read it. None is a third outcome."""
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120)
        v = p.stdout.strip()
        return float(v) if v and v != "N/A" else None
    except Exception:
        return None


def main():
    errs = json.load(open(f"{RUNS}/post-deploy-errors.json"))
    errs = errs if isinstance(errs, list) else errs["incompatible_files"]
    fold = json.load(open(f"{RUNS}/folders.json"))
    fold = fold if isinstance(fold, list) else list(fold.values())[0]
    dest = {x["id"]: x["destination_path"] for x in fold}

    # index the master files of every folder that has errors, by episode number
    masters = {}
    for fid in {e["folder_id"] for e in errs}:
        dp = dest.get(fid)
        masters[fid] = {}
        if not dp or not os.path.isdir(dp):
            continue
        for name in os.listdir(dp):
            if not name.lower().endswith((".mkv", ".mp4", ".mka")):
                continue
            m = EP_RE.search(name)
            if m:
                masters[fid][int(m.group(2))] = os.path.join(dp, name)

    rows = []
    for e in errs:
        row = dict(e)
        row["master_path"] = masters.get(e["folder_id"], {}).get(e["episode_number"])
        row["dest"] = dest.get(e["folder_id"])
        rows.append(row)

    # probe every distinct file once
    paths = set()
    for r in rows:
        paths.add(r["file_path"])
        if r["master_path"]:
            paths.add(r["master_path"])
    paths = sorted(paths)
    print(f"probing {len(paths)} distinct files...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=12) as ex:
        durs = dict(zip(paths, ex.map(duration, paths)))

    for r in rows:
        r["err_dur"] = durs.get(r["file_path"])
        r["mst_dur"] = durs.get(r["master_path"]) if r["master_path"] else None
        if r["err_dur"] and r["mst_dur"]:
            r["delta_s"] = round(r["err_dur"] - r["mst_dur"], 2)
            r["delta_pct"] = round(100.0 * abs(r["delta_s"]) / r["mst_dur"], 3)
        else:
            r["delta_s"] = None
            r["delta_pct"] = None

    json.dump(rows, open(f"{RUNS}/sweep-durations.json", "w"), indent=1)
    print(f"wrote {RUNS}/sweep-durations.json  ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
