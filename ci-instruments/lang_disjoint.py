#!/usr/bin/env python3
"""How many RELEASE PAIRINGS have a master language set disjoint from the candidate's?

dev-3's reframing, the Lead's ask. This is the §4c exposure in its actionable unit:
a property of the PAIRING, constant across a release's episodes, multiplying by
episode count -- and answerable from metadata alone, with no artefact produced.

A candidate track whose language the master lacks is filled cross-language and is
therefore STRUCTURALLY UNVERIFIABLE. So per pairing:

    unverifiable_tracks = candidate languages NOT present in the master
    exposure            = that count x the number of episodes in the pairing

THE UNIT IS THE PAIRING, NOT THE FILE. Every clustering correction tonight went the
same way, and this is the same fact: episodes of one release share a master release
and therefore share their language relation exactly.

Metadata only -- ffprobe stream tags, no decoding.
"""
import json, subprocess, collections, sys

def langs(p):
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index:stream_tags=language",
                        "-of", "csv=p=0", p], capture_output=True, timeout=45)
    out = []
    for line in q.stdout.decode().split():
        parts = line.split(',')
        if parts and parts[0].isdigit():
            out.append(parts[1] if len(parts) > 1 else None)
    return out

def main():
    recs = json.load(open('runs/audio-durations.json'))
    by_pairing = collections.defaultdict(list)
    fail = 0
    for n, r in enumerate(recs, 1):
        try:
            m, c = langs(r['master_path']), langs(r['file_path'])
        except Exception:
            fail += 1
            continue
        mset = {x for x in m if x}
        cset = {x for x in c if x}
        # NO LANGUAGE TAG IS NOT A LANGUAGE. An untagged track cannot be shown to be
        # cross-language, so it is counted separately and never as unverifiable.
        untagged = sum(1 for x in c if not x)
        missing = sorted(cset - mset)
        by_pairing[r.get('folder_id')].append(
            {"id": r['id'], "n_cand": len(c), "untagged": untagged,
             "missing": missing, "n_missing_tracks": sum(1 for x in c if x and x not in mset),
             "master_langs": sorted(mset), "cand_langs": sorted(cset)})
        if n % 50 == 0:
            print(f"  ...{n}/{len(recs)}", file=sys.stderr, flush=True)
    json.dump({str(k): v for k, v in by_pairing.items()},
              open('runs/lang-disjoint.json', 'w'), indent=1)
    print(f"pairings (by folder): {len(by_pairing)}   ffprobe failures: {fail}")
    tot_ep = tot_tracks = 0
    rows = []
    for fo, eps in by_pairing.items():
        miss = [e for e in eps if e['n_missing_tracks'] > 0]
        if not miss:
            continue
        sets = {tuple(e['missing']) for e in eps}
        exposure = sum(e['n_missing_tracks'] for e in eps)
        rows.append((exposure, fo, len(eps), len(miss), sorted(sets)[:1], len(sets)))
        tot_ep += len(miss); tot_tracks += exposure
    rows.sort(reverse=True)
    print(f"\n  pairings with ANY cross-language fill: {len(rows)} of {len(by_pairing)}")
    print(f"  episodes affected: {tot_ep}    unverifiable tracks (exposure): {tot_tracks}\n")
    print(f"  {'folder':>7}{'eps':>5}{'affected':>10}{'exposure':>10}  missing-language set (constant?)")
    for exp, fo, neps, nmiss, ex, nsets in rows[:16]:
        const = "CONSTANT" if nsets == 1 else f"{nsets} DIFFERENT SETS"
        print(f"  {str(fo):>7}{neps:>5}{nmiss:>10}{exp:>10}  {','.join(ex[0]) or '-':<28} {const}")

main()
