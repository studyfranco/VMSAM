#!/usr/bin/env python3
"""Hunt a sub-quantum step in files forensic's discriminator calls CLEAN.

SCOPE, STATED SO IT CANNOT BE MISREAD LATER: this frame is AIMED BY ANOTHER
AGENT'S INSTRUMENT. It therefore CANNOT contribute to the 2.3 % bound, whose
manifest was written before its first probe -- that is the only thing making it
uncontaminated. This is POINTING, not rate-making: I am looking for ONE exemplar
to run through the repair, not counting anything. Any rate from this frame would
need its own design frozen in advance and its own denominator.
"""
import json, subprocess, sys
sys.path.insert(0, '.')
import scan_hidden as S


def audio_langs(path):
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream_tags=language", "-of", "csv=p=0",
                        path], capture_output=True, timeout=60)
    return [x.strip() for x in q.stdout.decode().split() if x.strip()]


def matched_pair(mpath, cpath):
    """(master_idx, candidate_idx, language) for a SHARED language.

    DEFECT THIS FIXES, and it voided a whole run. The first version hardcoded
    stream 0 on both sides. On language-ASYMMETRIC files -- which is the entire
    point of this frame -- master a:0 was `fre` and candidate a:0 was `jpn`, so
    every probe correlated two different languages.

    It failed the way everything else has: four files came back `partial-coverage`,
    which is honest, and ONE came back `step, depth 9968.1 ms`, which is a spurious
    peak between unrelated audio wearing the shape of a measurement.
    """
    ml, cl = audio_langs(mpath), audio_langs(cpath)
    common = [l for l in ml if l in cl]
    if not common:
        return None
    # prefer the original-language track: it is the one both sides most likely
    # carry unmodified, and dubs are re-mastered per territory.
    lang = "jpn" if "jpn" in common else common[0]
    return ml.index(lang), cl.index(lang), lang

CLEAN_ASYM   = [114, 120, 122, 127, 131]     # forensic: clean x asymmetric
SPLICED_ASYM = [125, 128, 130]               # control: structure it CAN see

ap = {r['id']: r for r in json.load(open('runs/all-pairs.json'))}
ad = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
rec = {r['id']: r for r in json.load(open('runs/log-matrices2.json'))}

for label, ids in (("CLEAN x asymmetric -- the hunting ground", CLEAN_ASYM),
                   ("SPLICED x asymmetric -- control", SPLICED_ASYM)):
    print(f"\n=== {label} ===", file=sys.stderr, flush=True)
    for i in ids:
        if i not in ap or i not in ad:
            print(f"  id {i}: not in my pair tables -- COULD NOT MEASURE",
                  file=sys.stderr, flush=True); continue
        d = ad[i]
        mp = matched_pair(d['master_path'], d['file_path'])
        if mp is None:
            print(f"  id {i:<5} NO SHARED AUDIO LANGUAGE -- COULD NOT MEASURE",
                  file=sys.stderr, flush=True); continue
        ms, cs, lang = mp
        r = {'id': i, 'master_path': d['master_path'], 'file_path': d['file_path'],
             'ms': ms, 'cs': cs, 'err_audio_s': d['err_audio_s'],
             'mst_audio_s': d['mst_audio_s'],
             'per_pair': rec.get(i, {}).get('per_pair', {})}
        try:
            res = S.scan(r, n_probes=30)
        except Exception as e:
            res = {'id': i, 'label': 'instrument-failed', 'detail': str(e)[:70]}
        print(f"  id {i:<5} [{lang} a:{ms}/a:{cs}] {res['label']:<20} "
              f"{res.get('detail','')[:52]}",
              file=sys.stderr, flush=True)
