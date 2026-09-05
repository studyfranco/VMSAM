#!/usr/bin/env python3
"""Pass 2: re-measure every file whose step span exceeds the old fixed fine window.

WRITES INCREMENTALLY. Pass 1 wrote its output only at the end, so a crash would
have lost every sequence while the labels survived in the log -- an exposure I
named and carried rather than fixed. Fixed here: one JSON rewrite per file. The
cost is trivial against 26 s of ffmpeg per file.
"""
import json, sys
from redact import safe_text as _safe
sys.path.insert(0, '.')
import scan_hidden as S

samp = {r["id"]: r for r in json.load(open("runs/sampleA.json"))}
rec  = {r["id"]: r for r in json.load(open("runs/log-matrices2.json"))}
ids  = json.load(open("runs/pass2-ids.json"))
ids  = [i for i in ids if i not in (9, 403)]          # dropped for cause

# RESUMABLE. The machine is oversubscribed and this will be interrupted; a run that
# cannot resume gets restarted from zero and never finishes.
import os
done = {}
if os.path.exists("runs/pass2-out.json"):
    done = {r["id"]: r for r in json.load(open("runs/pass2-out.json"))}
ids = [i for i in ids if i not in done]
out = list(done.values())
for n, i in enumerate(ids, 1):
    r = dict(samp[i]); r["per_pair"] = rec[i]["per_pair"]
    try:
        res = S.scan(r, n_probes=15)
    except Exception as e:
        res = {"id": i, "label": "instrument-failed", "detail": _safe(e),
               "sequence": []}
    # a depth within 15% of the file's own window is SUSPECT: it may be the
    # ceiling rather than the media. Flagged in the data, not in a caveat.
    fw = res.get("fine_maxlag_s")
    if fw and "depth" in res.get("detail", ""):
        try:
            d = float(res["detail"].split("depth")[1].split("ms")[0])
            res["near_window_ceiling"] = d > 0.85 * fw * 1000
        except (ValueError, IndexError):
            pass
    out.append(res)
    json.dump(out, open("runs/pass2-out.json", "w"), indent=1)   # incremental
    print(f"[{n}/{len(ids)}] id={i} {res['label']}: {res.get('detail','')[:60]}"
          f" (fine=+-{res.get('fine_maxlag_s','?')}s"
          f"{' CEILING?' if res.get('near_window_ceiling') else ''})",
          file=sys.stderr, flush=True)
print("PASS2 DONE", file=sys.stderr)
