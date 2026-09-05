#!/usr/bin/env python3
"""A diagnostic BESIDE the corpus run, never inside it.

Every row carries `targeted_probe: true` so it CANNOT enter any incidence by
construction. That distinction was being relied on from the queue ordering, and it
does not exist for free -- it has to be a field.

TWO GROUPS, and the second is the stronger test:
  144, 157, 174  declined at 1103.4 / 1103.8 / 1059.4 ms on the pre-fix image and
                 repair at 12.12 / 7.75 / 34.88 ms in the author's clone under
                 current code. A PREDICTED ANSWER THAT CAN FAIL.
  8, 9, 12, 13, 25  splice-family: does `duration_source` come back "matroska tag"
                 in the container as it does in a lab, and does `measured` come back
                 true. Better to learn on five files than on three hundred.
"""
import json, os, re, subprocess, sys, time
from redact import safe_text as _safe
from redact import redact_checked
API=os.environ.get("VMSAM_TEST_API", "http://" + os.environ.get("VMSAM_TEST_HOST","showgestionar-test") + ":8080"); OUT="/config/output/srv"
IDS=[8,9,12,13,25]   # the three are done on f7b0e6b; these five are the splice population the guard is for
def api(p,d=None):
    c=["curl","-s","--max-time","60",f"{API}{p}"]
    if d is not None: c+=["-X","POST","-H","Content-Type: application/json","-d",json.dumps(d)]
    try: return json.loads(subprocess.run(c,capture_output=True,timeout=90).stdout.decode())
    except Exception: return None
def arts(since):
    # CAPTURE THE SUCCESS LOG TOO. The first version matched only *.mkv and
    # *.log.error -- so a probe built to answer "does the gate report measured"
    # COULD NOT SEE THE FILE THE ANSWER LIVES IN. The gate line (measured=,
    # would_refuse=, enforcing=, tolerance_ms=) is written to a plain *.log.
    r=subprocess.run(["find",OUT,"-newermt",since,"(","-name","*.mkv","-o","-name","*.log.error","-o","-name","*.log",")"],
                     capture_output=True,timeout=120)
    return [l for l in r.stdout.decode().splitlines() if l.strip()]
ad={r['id']:r for r in json.load(open('runs/audio-durations.json'))}
errs={e['id']:e['file_path'] for e in json.load(open('runs/errors-7b83af4.json'))['incompatible_files']}
img=api("/health") or {}
out=open('runs/targeted-probe.jsonl','a')
out.write(json.dumps({"ask":"diagnostic beside the corpus run","ids":IDS,
                      "image_git_commit":img.get("git_commit","UNKNOWN"),
                      "excluded_from_incidence":True}) + "\n"); out.flush()
for n,i in enumerate(IDS,1):
    since=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()-5))
    r=api("/fusion",{"error_file_path":errs[i]})
    if not r or "queued" not in str(r.get("message","")).lower():
        rec={"id":i,"targeted_probe":True,"verdict":"COULD_NOT_QUEUE","why":_safe(r)}
        out.write(json.dumps(rec)+"\n"); out.flush()
        print(f"[{n}/{len(IDS)}] id={i} COULD_NOT_QUEUE",file=sys.stderr,flush=True); continue
    t0=time.time(); found=[]
    while time.time()-t0 < 3*3600:
        time.sleep(30); found=arts(since)
        if found: break
        st=api("/fusion")
        if st and st.get("status")=="idle" and not st.get("current_job"):
            found=arts(since); break
    if not found: rec={"id":i,"verdict":"COULD_NOT_MEASURE","why":f"no artefact in {int(time.time()-t0)}s"}
    else:
        mkv=[f for f in found if f.endswith('.mkv')]
        logs=[f for f in found if f.endswith('.log.error')]
        # READ THE REASON BEFORE DELETING IT. The first version of this probe recorded
        # verdict=DECLINED and then deleted the log.error -- so it could say THAT three
        # frozen predictions failed and not one word about WHY. A decline at 1103 ms and
        # a decline at 12 ms are OPPOSITE findings (fix inert vs fix works, gate rejects)
        # and the artefact that separates them was the artefact being thrown away.
        if logs:
            raw=open(logs[0],encoding='utf-8',errors='replace').read()
            red=redact_checked(raw)
            open(f'runs/decline-{i}.log.redacted','w').write(red)
            first=[l for l in red.splitlines() if 'Error processing file' in l]
            frames=re.findall(r'line \d+, in (\w+)', red)
            fdt=[l.strip()[:200] for l in red.splitlines() if 'first_delay_test' in l]
            rec={"id":i,"verdict":"DECLINED",
                 "decline_msg": (first[0][:200] if first else None),
                 "decline_stage": (frames[-1] if frames else None),
                 "repair_attempted": ('repair:' in red),
                 "repair_failed": ('repair: failed' in red),
                 "ffmpeg_input_error": ('Error opening input' in red),
                 "n_duration_na": red.count('Duration: N/A'),
                 "first_delay_test": (fdt[0] if fdt else None),
                 "reason_saved": f'runs/decline-{i}.log.redacted'}
        elif not mkv:
            rec={"id":i,"verdict":"COULD_NOT_MEASURE","why":"artefact was neither mkv nor log.error"}
        if mkv:
            c=subprocess.run([sys.executable,"check_output.py",mkv[0],ad[i]['master_path']],
                             capture_output=True,timeout=300)
            try: rec={"id":i,**json.loads(c.stdout.decode())}
            except Exception as e: rec={"id":i,"verdict":"COULD_NOT_MEASURE","why":_safe(e)}
        for f in found:
            try: os.remove(f)
            except Exception: pass
    # FOLDER-HASH IN EVERY ROW. Three files whose figures matched to the millisecond
    # turned out to be one season sampled three times, and I only found that by
    # getting suspicious first. A ROW THAT DOES NOT NAME ITS POPULATION CANNOT BE
    # COUNTED CORRECTLY LATER -- same reason image_git_commit is in every row.
    import hashlib as _h
    rec["folder_hash"]=_h.sha256(os.path.dirname(errs[i]).encode()).hexdigest()[:10]
    rec["targeted_probe"]=True
    rec["image_git_commit"]=img.get("git_commit","UNKNOWN")
    out.write(json.dumps(rec)+"\n"); out.flush()
    print(f"[{n}/{len(IDS)}] id={i} {rec.get('verdict')}",file=sys.stderr,flush=True)
print("PROBE DONE",file=sys.stderr)
