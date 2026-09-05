#!/usr/bin/env python3
"""dev-1's test: is a non-whole-frame step a property of the EDIT or of the TRACK?

dev-1 measured one cut on id 100 three ways: 5.000 frames on one audio track, 4.786 on
another, and NO STEP AT ALL on a third. If frame-alignment is a track property, some of
my 66 "off-grid" steps are frame-aligned edits measured on a track carrying its own
offset -- which is a completely different finding from an off-grid edit, and only the
latter bears on MERGE_CRITERIA criterion 1.

I cannot answer this from stored data: full-out.jsonl holds exactly ONE track_pair per
file. This probes the SAME BRACKET on the SIBLING pairs.

Usage: sibling_probe.py <id> <t_before> <t_after>
"""
import json, subprocess, sys, os
sys.path.insert(0, '.')
import scan_hidden as S

def audio_streams(p):
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index:stream_tags=language",
                        "-of", "csv=p=0", p], capture_output=True, timeout=45)
    out = []
    for line in q.stdout.decode().split():
        parts = line.split(',')
        if parts and parts[0].isdigit():
            out.append((int(parts[0]), parts[1] if len(parts) > 1 else None))
    return out

def main():
    i = int(sys.argv[1]); tb = float(sys.argv[2]); ta = float(sys.argv[3])
    rec = {r['id']: r for r in json.load(open('runs/audio-durations.json'))}[i]
    M, C = rec['master_path'], rec['file_path']
    ms, cs = audio_streams(M), audio_streams(C)
    print(f"id {i}: master {len(ms)} audio, candidate {len(cs)} audio")
    # RELATIVE audio index is what pcm() wants (a:N), not the absolute stream index.
    mrel = {absi: n for n, (absi, _) in enumerate(ms)}
    crel = {absi: n for n, (absi, _) in enumerate(cs)}
    fr = None
    for l in open('runs/framerate-pairs.jsonl'):
        r = json.loads(l)
        if r['id'] == i: fr = r.get('master_fps')
    frame_ms = 1000.0 / fr if fr else None
    print(f"  declared master fps {fr}  frame {frame_ms:.4f} ms" if fr else "  no rate")
    print(f"  {'pair':<22}{'before':>11}{'after':>11}{'step_ms':>11}{'frames':>10}{'resid_ms':>10}")
    for mi, mlang in ms:
        for ci, clang in cs:
            if mlang != clang or mlang is None:
                continue          # ONLY same-language pairs; cross-language is meaningless
            try:
                b, bc = S.offset_at(M, mrel[mi], C, crel[ci], tb, dur=20.0, maxlag=4.0)
                a, ac = S.offset_at(M, mrel[mi], C, crel[ci], ta, dur=20.0, maxlag=4.0)
            except Exception as e:
                print(f"  a:{mrel[mi]} x a:{crel[ci]} ({mlang}) -- probe failed: {str(e)[:40]}")
                continue
            step = a - b
            fpart = step / frame_ms if frame_ms else float('nan')
            resid = (fpart - round(fpart)) * frame_ms if frame_ms else float('nan')
            flag = "  <== WHOLE FRAME" if abs(resid) <= 0.2 else ""
            print(f"  a:{mrel[mi]} x a:{crel[ci]} ({mlang}){'':<6}{b:>11.1f}{a:>11.1f}"
                  f"{step:>11.1f}{fpart:>10.3f}{resid:>10.3f}{flag}   corr {bc:.3f}/{ac:.3f}")

main()
