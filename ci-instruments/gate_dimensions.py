#!/usr/bin/env python3
"""Two NAMED dimensions, never folded into one verdict.

id 134 is the proof they can disagree: verified_coverage 2/2 -- every track the
verifier examined, verified -- while all three audio tracks were 78.4 SECONDS SHORT
of a full-length video. A single token for two questions prints a pass.

    ALIGNMENT     did the tracks line up where they were checked
    COMPLETENESS  do audio and video run the full length

They are independent. A file can pass either and fail the other, and the failure that
shipped was in the dimension nobody printed.

THREE OUTCOMES PER DIMENSION, BY NAME. UNKNOWN is not PASS.
"""
import json, sys

TOL = 0.02      # check_output.TOL, and it is printed beside the verdict, not implied

def alignment(row):
    """From the derived coverage where present, else the emitted field."""
    d = row.get('verified_derived')
    e = row.get('verified_coverage')
    src = d or (e if isinstance(e, str) and '/' in e else None)
    if not src:
        return {"dimension": "alignment", "state": "UNKNOWN",
                "why": "no verified= emitted and no per-track verify= lines to derive from",
                "coverage": None}
    n, m = (int(x) for x in src.split('/'))
    return {"dimension": "alignment",
            "state": "PASS" if n == m else ("NONE_VERIFIED" if n == 0 else "PARTIAL"),
            "coverage": src, "verified": n, "examined": m,
            "unverifiable": m - n,
            "why": ("every examined track verified" if n == m else
                    f"{m - n} of {m} tracks could not be verified -- structural where the "
                    "fill was cross-language, see the §4c finding")}

def completeness(row):
    """Audio and video must run the length. SUBTITLES ARE EXCLUDED BY DESIGN --
    a subtitle ends at its last cue, and a known-good file reported 24 short
    subtitle tracks with every audio track at ratio 1.0."""
    tracks = row.get('tracks')
    m = row.get('master_dur_s')
    if not tracks or not m:
        return {"dimension": "completeness", "state": "UNKNOWN",
                "why": "no per-track durations in this row"}
    av = [t for t in tracks if t.get('type') in ('audio', 'video')
          and t.get('dur_s') is not None]
    if not av:
        return {"dimension": "completeness", "state": "UNKNOWN",
                "why": "no audio or video track carried a duration"}
    short = [t for t in av if t['dur_s'] < m * (1 - TOL)]
    worst = min((t['dur_s'] for t in av), default=None)
    return {"dimension": "completeness",
            "state": "FAIL" if short else "PASS",
            "tolerance_frac": TOL,
            "tolerance_s": round(m * TOL, 1),
            "shortest_av_s": worst,
            "shortfall_s": round(m - worst, 2) if worst is not None else None,
            "n_short": len(short),
            "n_examined": len(av),
            "why": (f"{len(short)} of {len(av)} audio/video tracks below "
                    f"{100*(1-TOL):.0f}% of master duration" if short
                    else f"all {len(av)} audio/video tracks within {TOL*100:.0f}%")}

def dimensions(row):
    a, c = alignment(row), completeness(row)
    # THE COMBINED LINE EXISTS ONLY TO SAY THEY ARE SEPARATE. It never replaces them.
    return {"alignment": a, "completeness": c,
            "both_pass": a['state'] == 'PASS' and c['state'] == 'PASS',
            "disagree": (a['state'] == 'PASS') != (c['state'] == 'PASS')}

if __name__ == '__main__':
    path = sys.argv[1]
    for l in open(path):
        try: r = json.loads(l)
        except Exception: continue
        if not (isinstance(r, dict) and 'verdict' in r and 'id' in r): continue
        d = dimensions(r)
        flag = "   <== DIMENSIONS DISAGREE" if d['disagree'] else ""
        print(f"  id {r['id']:>4} {r['verdict']:<17} "
              f"alignment={d['alignment']['state']:<14} "
              f"completeness={d['completeness']['state']:<8}{flag}")
        if d['disagree']:
            print(f"        alignment: {d['alignment']['why']}")
            print(f"        completeness: {d['completeness']['why']} "
                  f"(shortfall {d['completeness']['shortfall_s']} s, tolerance {d['completeness']['tolerance_s']} s)")
