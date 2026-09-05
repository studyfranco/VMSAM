#!/usr/bin/env python3
"""Derive steps from probe SEQUENCES, with the threshold in the instrument's units.

WHY THIS EXISTS: my labeller reported "2 deviant runs" on a file whose own sequence
carried four steps. The labels are unusable (121 of 254 disagree with their own data).
This does NOT re-label -- it derives, and it declares its own resolution.

THE THRESHOLD IS NOT A MILLISECOND CONSTANT. forensic's 150 ms sat at 1.21 quanta of
its own 124.1 ms resolution and silently deleted every one-quantum step. Taking its FIX
as the shape rather than its number, the threshold here is expressed in the two units
this instrument actually has:

  CORRELATION QUANTUM  1/SR = 1/8000 s = 0.125 ms. The smallest lag the argmax can
                       return. A step below this cannot be represented AT ALL.
  DOMAIN QUANTUM       one video frame at THE FILE'S OWN declared rate, 1000/fps.
                       dev-1 measured every id-13 step as a whole frame to within 0.1 ms
                       (RETRACTED 0.067: that figure was its rounding, not a bound),
                       so a real edit is expected to be >= 1 frame.

  threshold = 0.5 frame, floored at 8 x correlation quantum and at 6 x measured noise

USING THE FILE'S OWN RATE MATTERS: dev-1 tested id 13's steps against 23.976 -- the
corpus mode, 516 of 561 -- and got a 1.500 ms residual that read as "not frame
aligned". id 13 is 24.000. A property of the population used where a property of the
individual was required.

THE WHOLE-FRAME FIGURE NEVER TRAVELS WITHOUT ITS TOLERANCE. "403 of 469 steps are
whole frames (85.9 %)" is a claim AT A 2.09 ms TOLERANCE (0.05 frames at 23.976).
dev-1's independent instrument has a repeatability bound of <=0.1 ms, so my tolerance
is about FORTY TIMES its resolution: my "frame-aligned" set provably contains steps
that instrument can resolve as off-grid. The two are not the same claim, and quoting
85.9 % as "whole frames" would read as dev-1's <=0.1 ms result at 254 files, which it
is not. The sensitivity:

    tolerance (frames)   = ms @23.976   whole-frame   %
         0.05                 2.09          403     85.9
         0.03                 1.25          385     82.1
         0.02                 0.83          336     71.6
         0.01                 0.42          217     46.3
         0.005                0.21          144     30.7
         0.0024               0.10          114     24.3

dev-1's PRECISION IS <=0.1 ms, NOT 0.067 ms. It retracted 0.067 after this curve made
it check: its correlation quantum is 0.0227 ms but it STORES offsets rounded to one
decimal place, so every id-13 residual (0.000, 0.067, 0.033, 0.033) is smaller than its
own storage quantum -- they are what rounding an exact frame multiple to 0.1 ms
produces. A residual quoted finer than the quantum of the values it was computed from.
So the gap between my tolerance and its resolution is 2.09 ms against 0.1 ms -- still
twentyfold, still two different claims, and no longer thirtyfold.

READ ONLY THE ROWS ABOVE MY OWN RESOLUTION. The correlation quantum is 0.125 ms and
sequences are stored rounded to 0.1 ms, so the bottom two rows are at or below what
this instrument can represent -- their decline measures MY noise, not the media. The
lowest interpretable row is 0.01 frames / 0.42 ms, at 46.3 %.

So the defensible statements are: 85.9 % frame-aligned at a tolerance matched to a
44 s grid's straddle risk, falling to 46.3 % at the finest tolerance this instrument
can honestly assert. The decay is steep, and a single figure hides it.

EVERY step carries its size in FRAMES and the resolution that produced it, so a
one-frame reading declares what it could and could not have seen.
"""
import json, sys, collections

CORR_QUANTUM_MS = 1000.0 / 8000.0          # scan_hidden.SR = 8000, lag is an int sample

def noise_estimate(offs):
    """Median |consecutive diff| -- robust to the steps themselves, which are few."""
    d = sorted(abs(b - a) for a, b in zip(offs, offs[1:]))
    return d[len(d) // 2] if d else 0.0

def threshold_ms(fps, offs):
    frame_ms = 1000.0 / fps if fps else 41.708      # fallback stated, not hidden
    return (max(0.5 * frame_ms, 8 * CORR_QUANTUM_MS, 6 * noise_estimate(offs)),
            frame_ms)

def segments(pts, tol):
    """Group consecutive probes into PLATEAUS. A plateau is >=2 probes within tol of
    its running median. A lone probe between plateaus is a STRADDLE -- a probe that
    landed on the transition and reads an intermediate value.

    WHY NOT CONSECUTIVE DIFFS: my first version differenced adjacent probes and read
    id 13's third step as 64.7 ms because the probe at t=1160 straddles the change
    point (corr 0.8991 against ~0.98 either side). The true step, plateau to plateau,
    is 83.5 ms. A straddling probe SPLITS a step across two diffs and under-reports
    both halves. dev-1's independent instrument measures between segment LEVELS, which
    is why its steps come out as whole frames to within 0.1 ms and mine did not.
    """
    segs, cur = [], [pts[0]]
    for p in pts[1:]:
        med = sorted(o for _, o in cur)[len(cur) // 2]
        if abs(p[1] - med) <= tol:
            cur.append(p)
        else:
            segs.append(cur); cur = [p]
    segs.append(cur)
    return segs

def derive(seq, fps):
    """(confirmed, candidates, thr, frame_ms, n_used, n_null).

    THREE OUTCOMES, NOT TWO. A level change supported by TWO plateaus is a step. A
    level change supported by a SINGLE probe is a CANDIDATE -- at this grid it cannot
    be told from an edge artefact, and my first probe is an outlier by >100 ms in
    26.0 % of rows.

    I DO NOT DROP EDGES ANY MORE. Dropping them deleted id 13's fourth step, which
    dev-1 then CONFIRMED with a flat-region control: 17 probes at W=15 over a known-
    flat span, all identical to the last decimal, change point bracketed (1285,1290),
    step -41.7 ms = one frame at 24.000 within 0.1 ms. My edge rule turned a real step
    into nothing -- and nothing reads as flat, which is the defect dev-1 had at the
    other end of the same file. A candidate is reported AS a candidate instead.
    """
    pts = [(t, o) for t, o, *_ in seq if isinstance(o, (int, float))]
    n_null = len(seq) - len(pts)
    if len(pts) < 3:
        return None, None, None, None, len(pts), n_null
    thr, frame_ms = threshold_ms(fps, [o for _, o in pts])
    segs = segments(pts, max(0.25 * frame_ms, 6 * CORR_QUANTUM_MS))
    lev = [(s, sorted(o for _, o in s)[len(s) // 2]) for s in segs]
    confirmed, cand = [], []
    for (sa, la), (sb, lb) in zip(lev, lev[1:]):
        d = lb - la
        if abs(d) <= thr:
            continue
        # WITHIN-LEVEL SPREAD IS WHAT THE INSTRUMENT ACHIEVED ON THIS FILE; the
        # correlation quantum is only what it cannot go below. dev-1's point, and it is
        # one step up from the 0.067 error: quoting the floor as if it were the achieved
        # precision. Its id 269 had spread 0.00 on all six levels while id 52 had
        # 0.40-0.50, so the floor is identical and the achieved precision is not.
        # With these, a residual can be tested against THIS FILE'S noise -- se of each
        # level mean ~ spread/sqrt(n) -- instead of against a global constant.
        spa = max(o for _, o in sa) - min(o for _, o in sa)
        spb = max(o for _, o in sb) - min(o for _, o in sb)
        rec = {"bracket_s": [round(sa[-1][0], 1), round(sb[0][0], 1)],
               "step_ms": round(d, 1),
               "step_frames": round(d / frame_ms, 3),
               "step_corr_quanta": round(d / CORR_QUANTUM_MS, 1),
               "probes_before": len(sa), "probes_after": len(sb),
               "spread_before_ms": round(spa, 2), "spread_after_ms": round(spb, 2),
               # se of a difference of two means, from THIS file's own scatter
               "step_se_ms": round(((spa / 2) ** 2 / max(len(sa), 1)
                                    + (spb / 2) ** 2 / max(len(sb), 1)) ** 0.5, 3)}
        (confirmed if len(sa) >= 2 and len(sb) >= 2 else cand).append(rec)
    return confirmed, cand, thr, frame_ms, len(pts), n_null

if __name__ == "__main__":
    fps = {}
    for l in open('runs/framerate-pairs.jsonl'):
        try:
            r = json.loads(l); fps[r['id']] = r.get('master_fps')
        except Exception: pass
    rows = []
    for l in open('runs/full-out.jsonl'):
        try: r = json.loads(l)
        except Exception: continue
        if isinstance(r, dict) and r.get('sequence'): rows.append(r)

    only = sys.argv[1:] and [int(x) for x in sys.argv[1:]]
    hist = collections.Counter(); out = []
    for r in rows:
        if only and r['id'] not in only: continue
        confirmed, cand, thr, frame_ms, n_used, n_null = derive(r['sequence'], fps.get(r['id']))
        if confirmed is None:
            hist['TOO_SHORT'] += 1; continue
        hist[len(confirmed)] += 1
        out.append((r['id'], r['label'], confirmed, cand, thr, frame_ms, n_used, n_null))
        if only:
            print(f"id {r['id']}  label={r['label']}  fps={fps.get(r['id'])}  "
                  f"frame={frame_ms:.3f} ms  threshold={thr:.2f} ms "
                  f"({thr/frame_ms:.2f} frames, {thr/CORR_QUANTUM_MS:.0f} corr-quanta)")
            print(f"  probes used {n_used}, unmeasurable {n_null}")
            for s in confirmed + [dict(c, CANDIDATE=True) for c in cand]:
                tag = "  <-- CANDIDATE (single probe, cannot be told from an edge artefact)" if s.get('CANDIDATE') else ""
                print(f"   {str(s['bracket_s']):>18}  {s['step_ms']:>9} ms  "
                      f"{s['step_frames']:>8} frames  {s['step_corr_quanta']:>8} q"
                      f"  [{s['probes_before']}|{s['probes_after']} probes]{tag}")
    if not only:
        print("derived step-count distribution over", sum(hist.values()), "rows:")
        for k in sorted(hist, key=str): print(f"   {k}: {hist[k]}")
