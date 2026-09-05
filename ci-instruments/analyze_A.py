#!/usr/bin/env python3
"""Score sample A: JOIN my scan labels against the stage-1 record.

WHY A JOIN AND NOT A MARGINAL COUNT
-----------------------------------
The sample is drawn from the REFUSED corpus. A file is in it BECAUSE the pipeline
saw something wrong. So a `step` in my scan is, for most of these files, the very
step the record already shows -- it is the reason the file was refused, not a miss.

Reporting "N% of sampled files carry a step" as a miss rate would be one of the
worst errors available here: it would count the record's OWN successes as its
failures, and it would be a large, confident, quotable number.

A MISS is a file where the record could not see what I can:

  no-matrix        the record shows NOTHING. Any structure here is unseen.
  island           offset departs and RETURNS. Every stage-1 window spans both
                   transitions, so the record reads constant -- and where the
                   record DID refuse on something else, its shape is still wrong,
                   which matters because a repair acts on the shape.
  *-subquantum     below ~62 ms: moves no window's reading, at any position.

`step` against a record that already says `within-step` is AGREEMENT. It is
reported, because agreement is evidence the instrument works, but it is not a miss.
"""
import json, sys, collections, math

def assert_one_instrument(recs):
    """REFUSE to score rows produced by different instruments.

    Found in my own pass-2 output: a restart's resume logic preserved id 29's row
    from BEFORE the probe-count and window cuts -- 30 probes and a 15 s window
    against 15 probes and 12 s for every other row. Each row was individually
    well-formed and the file was silently mixed, which is dev-2's duplicate-producer
    hazard arriving by a different route: not two processes, but one process
    inheriting another's output.

    A rate computed across two instruments is not a rate. This raises rather than
    warns, because the failure is invisible in every summary statistic downstream
    and I would not have remembered.
    """
    sig = {}
    for r in recs:
        if not r.get("sequence"):
            continue
        # KEY ON PROBE COUNT ONLY. probe_s and fine_maxlag_s vary per file BY
        # DESIGN -- the window is sized from each file's own step span -- so keying
        # on them flagged 20.0 / 21.9 / 25.9 as three instruments when they are one.
        # A guard that cries wolf on correct data gets bypassed, which is worse than
        # no guard: it trains you to pass the flag rather than read it.
        sig.setdefault(len(r["sequence"]), []).append(r["id"])
    if len(sig) > 1:
        lines = "\n".join(f"    {k} probes: {len(v)} rows e.g. {v[:5]}"
                          for k, v in sorted(sig.items()))
        raise SystemExit("REFUSING TO SCORE -- MIXED INSTRUMENTS:\n" + lines +
                         "\n  re-run the minority rows on the current instrument.")
    return sig



_raw = json.load(open(sys.argv[1]))
assert_one_instrument(_raw)
scan = {r["id"]: r for r in _raw}
rec  = {r["id"]: r for r in json.load(open("runs/log-matrices2.json"))}
excl = set(json.load(open("runs/sampleA-exclusions.json"))["excluded_from_denominator"])

# forensic's master-side / per-language-divergence files: flat AND CORRECT.
# They answer a different question. A denominator matter, not contamination.
MASTER_SIDE = {6, 10, 17, 20, 24}

DID_NOT_RUN = {"partial-coverage", "unreadable", "instrument-failed",
               "baseline-not-found"}
MISS_CLASSES = {"island", "island-subquantum", "step-subquantum"}


def wilson(k, n, z=1.96, fpc=1.0):
    """Two-sided Wilson interval, finite-population corrected."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) * fpc
    lo, hi = (c - h) / d, (c + h) / d
    # DEFECT: the fpc scales the half-width but not the centre, so at k = 0 it
    # lifted the LOWER bound off zero -- printing [0.003, 0.146] for a count of
    # ZERO. That asserts the rate is certainly non-zero, which is exactly what
    # observing no events cannot support. An interval that excludes the value
    # you actually observed is not a correction, it is a bug.
    # Observing 0 of n is consistent with a true rate of 0; observing n of n is
    # consistent with 1. Pin those two ends rather than letting the fpc move them.
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (max(0.0, lo), min(1.0, hi))


rows = []
for i, s in scan.items():
    if i in excl:
        continue
    r = rec.get(i, {})
    rows.append({"id": i, "label": s["label"], "detail": s.get("detail", ""),
                 "shape": r.get("shape", "NOT-IN-RECORD"),
                 "master_side": i in MASTER_SIDE,
                 "quantum": r.get("point_ms"), "seq": s.get("sequence", [])})

ran  = [r for r in rows if r["label"] not in DID_NOT_RUN]
dead = [r for r in rows if r["label"] in DID_NOT_RUN]

print(f"scanned {len(rows)}   instrument ran on {len(ran)}   did NOT run on {len(dead)}")
print("\nTHE INSTRUMENT-DID-NOT-RUN SET IS NOT A CLEAN SET. It is a third outcome.")
for k, v in collections.Counter(r["label"] for r in dead).most_common():
    print(f"  {v:3d}  {k}")

print("\n=== JOIN: my label x what the record shows ===")
tab = collections.Counter((r["shape"], r["label"]) for r in ran)
shapes = sorted({s for s, _ in tab}); labels = sorted({l for _, l in tab})
print(f"{'record shape':<22}" + "".join(f"{l[:13]:>15}" for l in labels))
for s in shapes:
    print(f"{s:<22}" + "".join(f"{tab.get((s,l),0):>15}" for l in labels))

# DEFECT FOUND WHILE SCORING THE PARTIAL RUN, and it inflated the rate.
# The first version counted `shape == "no-matrix"` as a miss on its own. But a
# file whose record shows nothing AND in which I find nothing is not a miss --
# it is AGREEMENT that there is nothing there. id 24 was being counted as a
# miss while my own scan reported it flat at a 3.5 ms spread.
# A miss needs BOTH halves: the record blind, AND me finding structure.
STRUCTURE = MISS_CLASSES | {"step", "scatter"}
miss = [r for r in ran
        if r["label"] in MISS_CLASSES
        or (r["shape"] in ("no-matrix", "NOT-IN-RECORD") and r["label"] in STRUCTURE)]
print(f"\n=== MISSES: structure the record could not show ({len(miss)} of {len(ran)}) ===")
for r in sorted(miss, key=lambda x: x["id"]):
    print(f"  id {r['id']:<5} {r['label']:<20} rec={r['shape']:<18} {r['detail'][:60]}")

fpc = math.sqrt((254 - len(ran)) / (254 - 1)) if len(ran) < 254 else 0.0
lo, hi = wilson(len(miss), len(ran), fpc=fpc)
print(f"\nmiss rate {len(miss)}/{len(ran)} = {len(miss)/max(1,len(ran)):.3f}")
print(f"Wilson 95% (fpc={fpc:.3f}): [{lo:.3f}, {hi:.3f}]  -- UPPER LIMIT {hi:.1%}")
print("REFUTED if the upper limit exceeds ~10%: that makes the miss rate material")
print("and objective 3's three counts unreportable without it.")

# ---- forensic's quantum-artefact test, turned on MYSELF ----
print("\n=== QUANTUM-ARTEFACT CHECK (forensic's test, applied to me) ===")
print("If my depths cluster on multiples of the file's own quantum, they are")
print("suspect: continuous measurements landing on a grid is a signature, not luck.")
depths = []
for r in ran:
    if "depth" in r["detail"]:
        try:
            d = float(r["detail"].split("depth")[1].split("ms")[0])
            depths.append((r["id"], d, r["quantum"] or 125.0))
        except (ValueError, IndexError):
            pass
if not depths:
    print("  no depths yet")
else:
    off = []
    for i, d, q in sorted(depths, key=lambda x: x[1]):
        n = d / q
        near = abs(n - round(n))
        off.append(near)
        flag = " <-- ON THE GRID" if near < 0.12 and d > q * 0.5 else ""
        print(f"  id {i:<5} depth {d:8.1f} ms  = {n:6.2f} quanta (q={q:.0f})"
              f"  off-grid {near:.3f}{flag}")
    import statistics
    print(f"\n  mean |distance to nearest multiple|: {statistics.mean(off):.3f}")
    print("  0.25 = uniform (no grid effect). Near 0 = my depths ARE the grid.")

json.dump(rows, open(sys.argv[2], "w"), indent=1)
print(f"\nwrote {sys.argv[2]}")
