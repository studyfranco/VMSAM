#!/usr/bin/env python3
"""A POSITIVE CONTROL for the accepted population, with a known answer.

THE BLOCKER THIS ADDRESSES. The pipeline's ACCEPTED output is the population no
blind-spot figure covers -- the error corpus is defined by an inclusion criterion
that excludes sub-quantum steps by construction. A merged file's tracks can be
correlated against each other with no candidate, but the pilot showed cross-language
correlation mostly falls below my 0.25 floor, and I HAVE NO FILE WITH A KNOWN ANSWER
to tell whether low-correlation lag agreement is real or spurious.

Every instrument defect found in this campaign was caught by a case whose answer came
from OUTSIDE the instrument. There is no such case here, so I am constructing one.

ADMISSIBLE BECAUSE THE THING UNDER TEST IS THE INSTRUMENT'S SENSITIVITY, NOT THE MEDIA.
The step is mine, so the expected answer is mine -- that is exactly the fixture trap,
and it is licensed only for measuring what the scanner can detect. It says nothing
about how often such steps occur.

READ-ONLY on the library. Writes only to the scratchpad.
"""
import json, os, random, subprocess, sys
import numpy as np

SR = 8000
OUT = "/tmp/claude-1000/-home-vmsam-src-VMSAM-WIP-ci/45d5406a-a849-4c6b-8a9a-8830f03f753c/scratchpad/ctrl"
os.makedirs(OUT, exist_ok=True)

def pcm(path, stream, ss, dur):
    p = subprocess.run(["ffmpeg","-v","error","-ss",f"{ss}","-t",f"{dur}","-i",path,
        "-map",f"0:a:{stream}","-ac","1","-ar",str(SR),"-c:a","pcm_s16le","-f","s16le","-"],
        capture_output=True, timeout=900)
    return np.frombuffer(p.stdout, dtype=np.int16)

def write_wav(a, path):
    subprocess.run(["ffmpeg","-v","error","-y","-f","s16le","-ar",str(SR),"-ac","1",
                    "-i","pipe:0","-c:a","pcm_s16le",path],
                   input=a.astype(np.int16).tobytes(), capture_output=True, timeout=300)

fol = json.load(open('runs/folders.json'))['folders']
dests = [f['destination_path'] for f in fol if f.get('destination_path')
         and os.path.isdir(f['destination_path'])]
random.seed(20260903); random.shuffle(dests)

src = None
for d in dests:
    try: files = sorted(x for x in os.listdir(d) if x.endswith('.mkv'))
    except Exception: continue
    if not files: continue
    p = os.path.join(d, files[0])
    q = subprocess.run(["ffprobe","-v","error","-select_streams","a",
        "-show_entries","stream=index","-of","csv=p=0",p], capture_output=True, timeout=60)
    if len([l for l in q.stdout.decode().split() if l]) >= 2:
        src = p; break
if not src:
    sys.exit("no accepted file with two audio tracks found")

DUR, STEP_AT, STEP_MS = 1200.0, 600.0, 40.0
print(f"source: an accepted library file, 2+ audio tracks", file=sys.stderr)
print(f"building: track0 unchanged; track1 with a KNOWN {STEP_MS:.0f} ms step at t={STEP_AT:.0f}s",
      file=sys.stderr, flush=True)

a0 = pcm(src, 0, 0.0, DUR)
a1 = pcm(src, 1, 0.0, DUR)
n = min(len(a0), len(a1))
a0, a1 = a0[:n], a1[:n]
cut = int(STEP_AT * SR)
pad = np.zeros(int(STEP_MS / 1000.0 * SR), dtype=np.int16)
a1_stepped = np.concatenate([a1[:cut], pad, a1[cut:]])[:n]

write_wav(a0, f"{OUT}/t0.wav")
write_wav(a1, f"{OUT}/t1_flat.wav")
write_wav(a1_stepped, f"{OUT}/t1_step.wav")
print(f"wrote {OUT}: t0, t1_flat (no step), t1_step (+{STEP_MS:.0f} ms after {STEP_AT:.0f}s)",
      file=sys.stderr)
