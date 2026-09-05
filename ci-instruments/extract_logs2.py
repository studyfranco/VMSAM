#!/usr/bin/env python3
"""Stage-1 matrices, segmented BY TRACK PAIR. Supersedes extract_logs.py.

PRIVATE — reads real media paths. Never inside the repository.

Why this replaces the first version
-----------------------------------
`extract_logs.py` flattened every `(fidelity, offset_points, delay_ms)` triple in
the log into ONE list, across all track pairs. dev-1 caught the consequence:

    id 23   '0-0' = [0]   x10, fid 0.993-0.997
            '1-0' = [125] x10, fid 0.981-0.984

Each track pair is perfectly CONSTANT. The two values are the MASTER'S OWN two
audio tracks sitting 125 ms apart — not a change point in the media. Flattened,
that reads as `delays=[0, 125]` and my classifier called it a `step`, and I then
recommended those files as the best change-point exemplars. A chimeric repair
driven off that would have been repairing the wrong thing.

So variation BETWEEN track pairs and variation WITHIN a track pair are different
phenomena and must never be pooled. Only within-pair variation can be a staircase.

The second defect, which this file can only label and not fix
--------------------------------------------------------------
dev-1 measured that the recorded delays are **residuals, not absolute offsets**.
`first_delay_test` calls `recreate_files_for_delay_adjuster` (mergeVideo.py:333),
re-extracting the candidate shifted by stage 1's own answer, so everything
afterwards is measured against an already-shifted candidate. The shift is an
arbitrary millisecond value and almost never a whole multiple of the chromaprint
hop (4096/3/11025 = 123.840 ms), and a fractional-hop shift misaligns the two
fingerprint grids.

Their measurement on id 318, same pair, only the shift changing:

    shift 0.000 ms  = 0.000 hops -> fid 0.968-0.975, residual +552
    shift 495.360   = 4.000 hops -> fid 0.967-0.974, residual    0
    shift 552.000   = 4.457 hops -> fid 0.918-0.936, residual -138   <- the pipeline
    shift 619.200   = 5.000 hops -> fid 0.967-0.974, residual -138

Whole-hop shifts hold fidelity; the fractional one drops it into exactly the band
the log records. **So a ONE-POINT step in recorded residuals may be an artefact of
the re-extraction rather than a real change point**, and the recorded fidelities
are depressed by the instrument rather than by the media.

That is dev-1's measurement on **4 files of 157** — two artefacts, one real, one
unreadable. Four files is not a rate, and it is not carried here as one. What
follows from it is only this: the recorded data **cannot distinguish** a one-point
step from a shift artefact, so one-point steps are ranked as SUSPECT and
multi-point steps as the ones worth trusting.
"""

import ast
import json
import os
import re
import sys
from collections import Counter

RUNS = "/home/vmsam/src/VMSAM_HELP_AI/ci/runs"
M1 = re.compile(r"Multiple delay found with the method 1[^{]*(\{)")
CMDERR = re.compile(r"This cmd is in error")


def parse_all(text):
    """[(dict, (fileA, fileB)), ...] -- EVERY method-1 message in the log.

    DEFECT FIXED 2026-09-03: this used `M1.search`, reading only the FIRST
    message. 73 of the 315 logs carry more (64 have two, 8 have three, 1 has
    four), because one merge attempt compares several releases pairwise. Reading
    one message and treating it as the file's whole record is how I reported a
    single track pair and a single quantum for a log that contained three
    comparisons at two different window geometries.

    That also resolved dev-1's 126 ms puzzle: a folder holding a 918 s release
    alongside 1442 s ones yields a SHORTER window for the pairs involving it, so
    the same log carries 125 ms for some comparisons and 126 ms for others. The
    quantum is per call, and a log is not one call.
    """
    out = []
    for m in M1.finditer(text):
        s = text[m.start(1):]
        depth = 0
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = ast.literal_eval(s[:i + 1])
                    except Exception:
                        d = None
                    fm = re.search(r"\s*for (.+?) and (.+?)(?:\\n|\"|\'|$)",
                                   s[i + 1:i + 800], re.S)
                    if isinstance(d, dict):
                        out.append(({k: list(v) for k, v in d.items()},
                                    (fm.group(1).strip(), fm.group(2).strip())
                                    if fm else None))
                    break
    return out


def point_size(pairs):
    """ms per point for THIS file, recovered as -delay/offset. dev-1's method."""
    vals = []
    for tr in pairs.values():
        for f, o, dl in tr:
            if o:
                vals.append(round(-dl / o, 3))
    if not vals:
        return None, False
    c = Counter(vals)
    return c.most_common(1)[0][0], len(c) > 1


def classify(pairs):
    """Separate within-pair variation from between-pair variation."""
    per = {}
    for k, tr in pairs.items():
        dl = [x[2] for x in tr]
        fd = [x[0] for x in tr]
        per[k] = {"delays": sorted(set(dl)), "n": len(dl),
                  "fid_min": min(fd), "fid_max": max(fd),
                  "constant": len(set(dl)) == 1}
    within = any(not v["constant"] for v in per.values())
    consts = {v["delays"][0] for v in per.values() if v["constant"]}
    between = len(consts) > 1
    return per, within, between


def main():
    rows = json.load(open(f"{RUNS}/sweep-durations.json"))
    out = []
    for r in rows:
        p = r["file_path"] + ".log.error"
        rec = {k: r[k] for k in ("id", "folder_id", "episode_number",
                                 "file_path", "master_path")}
        if not os.path.exists(p):
            rec.update(shape="no-log")
            out.append(rec)
            continue
        t = open(p, encoding="utf-8", errors="replace").read()
        rec["cmd_error"] = bool(CMDERR.search(t))
        msgs = parse_all(t)
        rec["n_messages"] = len(msgs)
        # Pick the message that actually compares THIS error file against THIS
        # master. A log can hold several comparisons between other releases in the
        # same folder, at different window geometries; classifying the wrong one
        # attributes another pair's delays to this row.
        want_e = os.path.basename(r["file_path"])
        want_m = os.path.basename(r["master_path"] or "")
        pairs, matched = None, False
        for d, names in msgs:
            if not names:
                continue
            bn = [os.path.basename(x) for x in names]
            if want_e in bn and want_m in bn:
                pairs, matched = d, True
                break
        if pairs is None and msgs:
            pairs = msgs[0][0]          # fall back, and say so
        rec["pair_matched"] = matched
        if not pairs:
            rec["shape"] = "cmd-error" if rec["cmd_error"] else "no-matrix"
            out.append(rec)
            continue

        per, within, between = classify(pairs)
        ps, ps_conflict = point_size(pairs)
        rec["n_track_pairs"] = len(pairs)
        rec["per_pair"] = per
        rec["point_ms"] = ps
        rec["point_conflict"] = ps_conflict
        rec["within_pair_varies"] = within
        rec["between_pair_varies"] = between

        fids = [x[0] for tr in pairs.values() for x in tr]
        rec["fid_min"], rec["fid_max"] = min(fids), max(fids)
        floor = rec["fid_max"] < 0.65

        # step magnitude IN POINTS, within a single track pair only
        span = 0
        for v in per.values():
            if len(v["delays"]) > 1 and ps:
                span = max(span, (max(v["delays"]) - min(v["delays"])) / ps)
        rec["within_step_points"] = round(span, 2)

        if floor:
            allv = [x[2] for tr in pairs.values() for x in tr]
            rec["shape"] = ("floor-scatter" if len(set(allv)) >= 4
                            else "floor-constant")
        elif within:
            rec["shape"] = "within-step"
        elif between:
            # every track pair constant, pairs disagree: the MASTER's own tracks
            rec["shape"] = "between-pair-offset"
        else:
            rec["shape"] = "constant-offset"
        out.append(rec)

    json.dump(out, open(f"{RUNS}/log-matrices2.json", "w"), indent=1)
    print("shape distribution (track-pair aware):", file=sys.stderr)
    for k, v in Counter(r["shape"] for r in out).most_common():
        print(f"  {k:<22}: {v:3d}", file=sys.stderr)
    ps = Counter(r.get("point_ms") for r in out if r.get("point_ms"))
    print(f"\npoint size recovered: {dict(ps.most_common())}", file=sys.stderr)
    print(f"point-size conflicts within a file: "
          f"{sum(1 for r in out if r.get('point_conflict'))}", file=sys.stderr)
    print(f"logs with >1 method-1 message: "
          f"{sum(1 for r in out if (r.get('n_messages') or 0) > 1)}", file=sys.stderr)
    print(f"rows where the right comparison was identified by filename: "
          f"{sum(1 for r in out if r.get('pair_matched'))}"
          f" / {sum(1 for r in out if r.get('n_messages'))}", file=sys.stderr)
    print(f"wrote {RUNS}/log-matrices2.json", file=sys.stderr)


if __name__ == "__main__":
    main()
