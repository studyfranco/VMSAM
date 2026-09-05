#!/usr/bin/env python3
"""The full corpus through the whole-file scanner. Owner instruction: all error
files, not a sample.

DENOMINATOR, resolved in this instrument and written into the artefact:
  315 error records from the live API; all 315 have both media files present on
  disk; 301 carry a per-pair delay list the window can be sized from. The 14
  without one get the default window and ARE FLAGGED, not silently defaulted.
  ids 9 and 403 are scanned but marked: id 9's master was REPLACED since the error
  was recorded, id 403's log names no master, so their pairs never existed as
  recorded. They are reported, never dropped.

RESUMABLE and INCREMENTAL: a run this long will be interrupted, and a run that
cannot resume gets restarted from zero and never finishes.
"""
import json, os, subprocess, sys
sys.path.insert(0, '.')
import scan_hidden as S


def matched_pair(mpath, cpath):
    """(master_idx, candidate_idx, language) for a SHARED language.

    THIRD INSTANCE OF THIS DEFECT TODAY. The first version hardcoded stream 0 on
    both sides. Measured: on 25 randomly sampled pairs, stream 0 carries a DIFFERENT
    language on the two sides for 18 of them -- 72 % OF THE CORPUS WOULD BE
    CORRELATED ACROSS LANGUAGES. The first 91 rows of the full run showed 67 %
    instrument-did-not-run against 10 % on the sampled run, and that gap was my
    defect and not the media.

    sampleA.json carried matched indices -- (1,7), (8,8) -- computed once and
    correctly. I did not reuse them and re-derived the wrong thing.
    """
    def langs(p):
        try:
            q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                                "-show_entries", "stream_tags=language",
                                "-of", "csv=p=0", p], capture_output=True, timeout=45)
            return [x.strip() for x in q.stdout.decode().split() if x.strip()]
        except Exception:
            return []
    ml, cl = langs(mpath), langs(cpath)
    common = [l for l in ml if l in cl]
    if not common:
        return None
    lang = "jpn" if "jpn" in common else common[0]
    return ml.index(lang), cl.index(lang), lang

def _run():
    ad = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}
    rec = {r['id']: r for r in json.load(open('runs/log-matrices2.json'))}
    ids = sorted(ad)
    EXCL = {9: "master REPLACED since the error was recorded",
            403: "log names no master"}

    done = {}
    if os.path.exists('runs/full-out.jsonl'):
        for line in open('runs/full-out.jsonl'):
            try:
                d = json.loads(line)
                if 'id' in d: done[d['id']] = d
            except Exception:
                pass

    out = open('runs/full-out.jsonl', 'a')
    if not done:
        out.write(json.dumps({
            "ask": "every error record in the corpus",
            "asked_ids": ids, "n_asked": len(ids),
            "with_delay_list": sum(1 for i in ids if rec.get(i, {}).get('per_pair')),
            "flagged_no_delay_list": [i for i in ids if not rec.get(i, {}).get('per_pair')],
            "excluded_for_cause": EXCL, "probes": 30}) + "\n")
        out.flush()

    todo = [i for i in ids if i not in done]
    print(f"asked {len(ids)}  already done {len(done)}  to run {len(todo)}",
          file=sys.stderr, flush=True)
    for n, i in enumerate(todo, 1):
        d = ad[i]
        pp = rec.get(i, {}).get('per_pair', {})
        mp = matched_pair(d['master_path'], d['file_path'])
        if mp is None:
            out.write(json.dumps({'id': i, 'label': 'no-shared-language',
                                  'detail': 'COULD NOT MEASURE: no audio language in common',
                                  'sequence': [], 'no_delay_list': not bool(pp)}) + "\n")
            out.flush()
            print(f"[{n}/{len(todo)}] id={i} no-shared-language", file=sys.stderr, flush=True)
            continue
        ms, cs, lang = mp
        r = {'id': i, 'master_path': d['master_path'], 'file_path': d['file_path'],
             'ms': ms, 'cs': cs, 'err_audio_s': d['err_audio_s'],
             'mst_audio_s': d['mst_audio_s'], 'per_pair': pp}
        try:
            res = S.scan(r, n_probes=30)
        except Exception as e:
            res = {'id': i, 'label': 'instrument-failed', 'detail': str(e)[:90],
                   'sequence': []}
        res['no_delay_list'] = not bool(pp)
        res['track_pair'] = {'lang': lang, 'master_a': ms, 'candidate_a': cs}
        if i in EXCL:
            res['excluded_for_cause'] = EXCL[i]
        out.write(json.dumps(res) + "\n"); out.flush()
        print(f"[{n}/{len(todo)}] id={i} {res['label']}: {res.get('detail','')[:52]}",
              file=sys.stderr, flush=True)
    out.write(json.dumps({"summary": True, "n_asked": len(ids)}) + "\n")
    out.close()
    print("FULL RUN DONE", file=sys.stderr)


if __name__ == '__main__':
    _run()
