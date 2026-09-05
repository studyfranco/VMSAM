#!/usr/bin/env python3
"""One command for my own state, because I went 44 minutes without checking it.

The architect asked the fleet why everyone stopped. I had not stopped -- the runner
was continuous -- but I admitted a different fault: I was corresponding, not
monitoring, and a stalled job would have reached me late.

A HABIT I HAVE TO REMEMBER IS NOT A CONTROL. This makes the check one command, and it
DETECTS a stall rather than leaving me to notice one.

STALL THRESHOLD FROM MEASURED HISTORY, not from taste:
    id 12   137 min   longest job observed
    id 695   80 min   killed at 80, still running
So >150 min is beyond anything this corpus has produced and is reported as STALLED.
Between 90 and 150 is LONG -- named, not alarmed.
"""
import os, json, re, subprocess, sys, datetime, collections, glob

# ANCHOR ON THE SCRIPT'S OWN LOCATION. The first version used relative paths and
# died the moment I ran it from another directory -- a self-check that only works
# from one cwd is a self-check I will forget to run.
HERE = os.path.dirname(os.path.abspath(__file__))

def _latest_run():
    """THE NEWEST RUN FILE, not a hardcoded one. I redeploy and start a new file each
    time; a pinned name reports a finished run forever and looks healthy doing it."""
    import glob as _g
    c = sorted(_g.glob(os.path.join(HERE, 'runs/weekend-100-*.jsonl')), key=os.path.getmtime)
    return c[-1] if c else os.path.join(HERE, 'runs/weekend-100-aefdf8a.jsonl')

RUN = _latest_run()
LONG_MIN, STALL_MIN = 90, 150

def sh(cmd, timeout=30):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception: return ""

def main():
    problems = []
    print("  == processes ==")
    # ARGV POSITION, NEVER SUBSTRING. This counted any line CONTAINING the name and
    # reported "4 copies of preserve_artefacts.sh" -- two of which were MY OWN SHELL,
    # whose `bash -c '...preserve_artefacts.sh...'` carries the string in argv[2].
    #
    # FIFTH INSTANCE TONIGHT of the same trap: `pgrep -f merge_queue.py` killed my shell
    # twice, my own kill-guard's `case "$CMD" in *merge_queue.py*` killed it a third
    # time, `grep "id 29"` matched `id 293` inside the check for whether a claim had
    # spread -- and now the status tool built to catch problems inventing one.
    #
    # The fix is the one already in stop_runner.sh: read /proc/PID/cmdline and test the
    # BASENAME OF ARGV[1]. A process is the runner because of what it was invoked as,
    # not because its command line mentions a filename.
    def _instances(name):
        """Matching processes, EXCLUDING a match's own children.

        THIRD REVISION OF THIS CHECK IN ONE NIGHT, and each fired on a population the
        claim was never about:
          1. substring over `ps` output -> counted MY OWN SHELL, whose `bash -c '...'`
             carries the name in argv[2].
          2. basename(argv[1]) -> correct, and still reported "2 copies" because
             `while read` in a PIPELINE makes bash fork a SUBSHELL THAT INHERITS THE
             PARENT'S CMDLINE. A process is not a second instance because it is the
             first one's own plumbing.
          3. this: a match whose PPID is also a match is a subshell, not a copy.

        The guard is now about what LAUNCHED a process, not what its command line says
        and not what forked from it.
        """
        raw = {}
        for d in os.listdir('/proc'):
            if not d.isdigit(): continue
            try:
                with open(f'/proc/{d}/cmdline', 'rb') as fh:
                    parts = fh.read().split(b'\0')
                with open(f'/proc/{d}/stat', 'rb') as fh:
                    ppid = fh.read().rsplit(b')', 1)[1].split()[1].decode()
            except (OSError, IndexError):
                continue
            if len(parts) < 2: continue
            if os.path.basename(parts[1].decode('utf-8', 'replace')) != name: continue
            raw[d] = ppid
        out = []
        for d, ppid in raw.items():
            if ppid in raw:          # a match's own subshell, not a second copy
                continue
            et = sh(["ps", "-o", "etime=", "-p", d]).strip()
            out.append((d, et))
        return out
    ps = sh(["ps","-eo","pid,etime,args"])
    for name in ("merge_queue.py","preserve_artefacts.sh"):
        inst = _instances(name)
        lines = [f"{pid} {et} {name}" for pid, et in inst]
        if not lines:
            print(f"    {name:<24} ABSENT"); problems.append(f"{name} not running")
        else:
            for l in lines:
                f = l.split(None, 2)
                print(f"    {name:<24} pid {f[0]:<8} up {f[1]}")
            if len(lines) > 1:
                print(f"    {'':24} {len(lines)} INSTANCES -- expected 1")
                problems.append(f"{len(lines)} copies of {name}")

    print("  == container ==")
    try:
        # merge_queue USES ITS OWN RELATIVE PATHS. Importing it from another cwd
        # raised FileNotFoundError and my status tool reported "container unreachable"
        # -- a FALSE ALARM about the subject, caused by the instrument's environment.
        # A monitor that cries wolf is worse than no monitor: I would learn to ignore it.
        sys.path.insert(0, HERE)
        os.chdir(HERE)
        import merge_queue as m
        f = m.api('/fusion'); h = m.api('/health')
        cj = f.get('current_job') or {}
        print(f"    image {h.get('git_commit','?')[:12]}  (a LABEL, not the running code)")
        if cj.get('started_at'):
            st = datetime.datetime.fromisoformat(cj['started_at'].replace('Z','+00:00'))
            mins = (datetime.datetime.now(datetime.timezone.utc)-st).total_seconds()/60
            lst = m.api('/errors')['incompatible_files']
            hit = next((e.get('id') for e in lst if e.get('file_path')==cj.get('error_file_path','')), None)
            tag = ("STALLED -- beyond anything this corpus has produced" if mins > STALL_MIN
                   else "LONG -- named, not alarmed" if mins > LONG_MIN else "normal")
            print(f"    job id {hit}  elapsed {mins:.0f} min   {tag}")
            if mins > STALL_MIN: problems.append(f"job {hit} at {mins:.0f} min")
        else:
            print(f"    status {f.get('status')}  NO CURRENT JOB")
    except Exception as e:
        print(f"    UNREACHABLE -- {str(e)[:60]}"); problems.append("container unreachable")

    print(f"  == run ==  ({os.path.basename(RUN)})")
    rows = []
    for l in open(RUN):
        try: d = json.loads(l)
        except Exception: continue
        if isinstance(d, dict) and 'verdict' in d and 'id' in d: rows.append(d)
    uniq = {d['id']: d for d in rows}          # BY ID, never by row -- my 6-of-14 error
    c = collections.Counter(d['verdict'] for d in uniq.values())
    measured = c['FULL_LENGTH'] + c['TRUNCATED']
    # THE DENOMINATOR IS READ, NEVER TYPED. This line said `of 100` for hours because
    # I took the number from the run FILE'S NAME (`weekend-100-...jsonl`) instead of the
    # population. The name records what was ASKED FOR; `ONLY_IDS` records what the runner
    # was actually given, and after filtering against the corpus, what it can ever do.
    # A hardcoded denominator is the exact defect I keep finding in other agents' work:
    # a number nothing can contradict, so nothing ever does.
    _den, _basis = None, 'UNKNOWN'
    for _pid, _ in _instances('merge_queue.py'):
        try:
            _env = dict(l.split('=', 1) for l in
                        open(f'/proc/{_pid}/environ', 'rb').read()
                        .decode('utf8', 'replace').split('\0') if '=' in l)
        except OSError:
            continue
        _only = _env.get('ONLY_IDS')
        if _only:
            try:
                _corpus = {e['id'] for e in json.load(
                    open('runs/errors-7b83af4.json'))['incompatible_files']}
                _den = len([i for i in (int(x) for x in _only.split(',') if x.strip())
                            if i in _corpus])
                _basis = f"ONLY_IDS of pid {_pid}, filtered against the corpus"
            except Exception:
                _den = len([x for x in _only.split(',') if x.strip()])
                _basis = f"ONLY_IDS of pid {_pid} (corpus unreadable -- UNFILTERED)"
        break
    if _den is None:
        print(f"    {len(uniq)} done by DISTINCT id  ({len(rows)} rows) "
              f"-- DENOMINATOR UNKNOWN: no live runner to ask")
    else:
        print(f"    {len(uniq)} of {_den} by DISTINCT id  ({len(rows)} rows)")
        print(f"      denominator basis: {_basis}")
    # PROVENANCE, EXECUTED NOT RECITED. The rows carry a startup-cached image label and
    # the truth is in correction records keyed on file order. `vmsam-dev-2`'s rule: the
    # provenance has to be on the row and checked by the reader, not in a note the
    # reader has to remember exists.
    try:
        _pv = subprocess.run([sys.executable, 'row_provenance.py'],
                             capture_output=True, text=True, timeout=60)
        _last = [l for l in _pv.stdout.splitlines() if 'name an image that is NOT' in l]
        if _last:
            _m = re.search(r'(\d+) of (\d+) rows', _last[0])
            if _m:
                print(f"    image label: {_m.group(1)} of {_m.group(2)} rows name an "
                      f"image that did NOT run -- `row_provenance.py` resolves each")
    except Exception as _e:
        print(f"    image label: PROVENANCE UNCHECKED ({_e.__class__.__name__}) -- "
              f"not the same as 'labels are correct'")
    print(f"    {dict(c)}")
    print(f"    measured artefacts {measured} of the 20 I report at")

    print("  == preservation ==")
    kinds = collections.Counter()
    led = '/config/output/KEEP/ci-preservation-ledger.tsv'
    if os.path.exists(led):
        for i, l in enumerate(open(led)):
            if i == 0 or l.startswith('#'): continue
            p = l.split('\t')
            if len(p) > 1: kinds[p[1]] += 1
    print(f"    ledger kinds {dict(kinds)}")
    free = sh(["df","-h","/config/output"]).splitlines()[-1].split()[3]
    print(f"    KEEP {len(glob.glob('/config/output/KEEP/*.mkv'))} mkv, "
          f"{len(glob.glob('/config/output/KEEP/frozen/*.mkv'))} frozen, {free} free")

    # CLASSIFIER CENSUS. Three times tonight a CORRECT classifier in this tool
    # produced a plausible ZERO while the evidence sat on disk: the declined kinds
    # (sweep rooted where no such file lives), the per-row image (cached at startup),
    # and `kind` from the extension (marker is in the CONTENT). None was found by
    # reading code -- each was found by someone counting and getting a number that
    # could not be true.
    #
    # A CLASSIFIER THAT NEVER FIRES PRODUCES NO ERROR, NO EMPTY RESULT AND NO ANOMALY.
    # It produces a plausible zero, indistinguishable from a true zero to every reader
    # including its author. So: count what the classifier CLAIMS against what is ON
    # THE DISK, every time status runs.
    import subprocess as _sp
    on_disk = 0
    for f in glob.glob('/config/output/KEEP/*.error') + glob.glob('/config/output/KEEP/*.log'):
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                if 'undelivered state=' in fh.read():
                    on_disk += 1
        except OSError:
            pass
    claimed = kinds.get('declined_by_gate_record', 0) + kinds.get('unadjudicated_record', 0)
    # ENUMERATED COUNTS AS ACCOUNTED FOR. The sweep skips names it already holds, so
    # records captured before the content classifier existed keep `kind=error` forever
    # and are listed in ci-refusal-records.tsv instead. Treating those as an unfired
    # classifier makes this check cry wolf permanently -- and a check that is always
    # red is a check nobody reads, which is the failure it was built to prevent.
    enum = '/config/output/KEEP/ci-refusal-records.tsv'
    listed = 0
    if os.path.exists(enum):
        listed = max(0, sum(1 for _ in open(enum)) - 1)
    accounted = claimed + listed
    print(f"    refusal records: {on_disk} on disk, {claimed} kinded + {listed} enumerated")
    if on_disk > accounted:
        problems.append(f"{on_disk - accounted} refusal record(s) on disk are NEITHER "
                        f"kinded NOR enumerated -- a classifier that is not firing")
    elif on_disk != claimed:
        # NOT a problem by itself: rows captured before the content-marker classifier
        # existed keep kind=error and the sweep skips names it already holds.
        print(f"      ({on_disk - claimed} preserved before the content classifier existed; "
              f"enumerated in KEEP/ci-refusal-records.tsv)")

    # ASSERTIONS THAT CANNOT FAIL. Two of mine printed green tonight: `bool(accents)
    # or True`, and a whole test block that landed after `sys.exit(...)` while the suite
    # reported `11 passed, 0 failed`. A guard kept in my memory is a guard that runs when
    # I remember it; this one runs every time status does.
    # JOIN COVERAGE. `vmsam-dev-4`'s general test, after they found that my
    # merge_plan_report rows carried a stem NO OTHER ROW COULD SHARE: the key was
    # populated on every row and joined nothing, and the ledger looked complete from
    # inside. A JOIN THAT RETURNS ZERO ON A NON-EMPTY CORPUS IS A DEFECT, NEVER A
    # MEASUREMENT.
    #
    # Only a kind that joins NOTHING is a problem. A PARTIAL join is expected and has a
    # known cause: an `.error` record is named from the ERROR file's basename while an
    # artefact is named from the MASTER's, so the two bases cannot meet. That is a
    # limitation of the key, not a fault in a row.
    joins = collections.defaultdict(lambda: [0, 0])
    if os.path.exists(led):
        import csv as _csv
        _rows = list(_csv.DictReader(open(led), delimiter='\t'))
        _by = collections.defaultdict(list)
        for r in _rows: _by[r.get('stem_sha256_16')].append(r)
        # ROWS WHOSE KEY WAS CORRECTED OUT OF BAND STILL COUNT AS JOINED. The seven
        # merge_plan_report rows keep their original stem forever -- the sweep skips
        # names it already holds -- and their correct stems are backfilled in
        # ci-report-bindings.tsv. Treating those as unjoined makes this check
        # permanently red, and a check that is always red is a check nobody reads.
        _fix = {}
        _bind = '/config/output/KEEP/ci-report-bindings.tsv'
        if os.path.exists(_bind):
            for b in _csv.DictReader(open(_bind), delimiter='\t'):
                _fix[b.get('report_keep_name')] = b.get('correct_stem_sha256_16')
        # DEGENERACY, NOT FORMAT. My first fix rejected any stem that was not 16 hex
        # and was WRONG ABOUT THREE OF THE TWENTY-ONE: `FROZEN-COPY-of-<id>` values are
        # UNIQUE and NAME THEIR SOURCE -- perfectly good keys in a different format, and
        # rejecting them would force re-deriving provenance they already carry.
        # `vmsam-forensic` caught the over-correction.
        #
        # THE PRINCIPLED TEST, format-independent and with no threshold:
        #
        #     A JOB PRODUCES AT MOST ONE ARTEFACT OF EACH KIND.
        #     SO A VALID STEM COVERS AT MOST ONE ROW PER KIND.
        #
        # Measured: every real stem covers log+mkv+report, one each. Every
        # FROZEN-COPY-of-<id> covers one mkv. `UNKNOWN-backfill` covers EIGHTEEN rows
        # across error, log and mkv -- many per kind, rows that cannot be one job.
        #
        # The defect is a key that matches EVERYTHING, which passes a check written for
        # a key that matches NOTHING. Only asking what the matched rows ARE separates
        # them.
        # AT MOST ONE ROW PER KIND *PER INODE*, not per name. `vmsam-forensic` tried to
        # break the per-name form and found the attacker: the three container refusals
        # each exist under TWO NAMES on ONE INODE -- the `undelivered/` name and the
        # `DECLINED/` name. One job, two `mkv` rows, and the per-name rule would call a
        # valid key degenerate.
        #
        # IT CANNOT BITE TODAY only because those live in `ci-declined-ledger.tsv` and
        # never meet this one. THAT IS A PROPERTY OF THE POPULATIONS STAYING APART, and
        # I may well merge them. The inode form does not depend on that, and `inode` is
        # already 98/98 distinct here, so the stronger rule costs nothing.
        _kindcount = {}
        for k, v in _by.items():
            seen = collections.Counter()
            for m in v:
                seen[(m.get('kind'), m.get('inode'))] += 1
            _kindcount[k] = collections.Counter(
                kind for (kind, _ino) in {kk for kk in seen})
        def _is_key(k):
            c = _kindcount.get(k)
            return bool(c) and max(c.values()) <= 1
        for r in _rows:
            key = _fix.get(r.get('keep_name'), r.get('stem_sha256_16'))
            mates = ([m for m in _by[key] if m.get('keep_name') != r.get('keep_name')]
                     if _is_key(key) else [])
            joins[r.get('kind')][0] += 1
            if mates: joins[r.get('kind')][1] += 1
    # `vmsam-dev-2`'s third state, which is better than printing the rate and burying
    # the cause in a comment as I first did: `3/23 (structurally cannot meet)` and
    # `3/23 (cause not established)` are DIFFERENT ROWS, and the second is not a worse
    # version of the first. Two states where there are three is the collapse this whole
    # campaign keeps rebuilding.
    # LEDGER vs DISK, BOTH DIRECTIONS. A ledger row asserting a preserved file that is
    # not there is the same class as a key that joins nothing: the record looks complete
    # and the thing it records is absent. And an ORPHAN -- a file in KEEP with no row --
    # is the reverse: preserved, and unattributable the moment its source is gone.
    #
    # Found because dev-2's third state made me print "CAUSE NOT ESTABLISHED" for a
    # partial I had not investigated, and going to establish it turned up a row whose
    # file does not exist. THE CHECK THAT NAMES ITS OWN IGNORANCE IS THE ONE THAT GETS
    # ANSWERED.
    if os.path.exists(led):
        _names = {r.get('keep_name') for r in _rows}
        _gone = [r.get('keep_name') for r in _rows
                 if not os.path.exists('/config/output/KEEP/' + (r.get('keep_name') or ''))]
        _orph = []
        for _f in glob.glob('/config/output/KEEP/*') + glob.glob('/config/output/KEEP/frozen/*'):
            if os.path.isdir(_f): continue
            _rel = _f.replace('/config/output/KEEP/', '')
            # ONLY FILES OF A KIND THE SWEEP PRESERVES. This flagged
            # `validation-status.tsv.bak` -- vmsam-forensic's own backup, which is not
            # an artefact and was never going to have a ledger row. A check for
            # "preserved and unattributable" must first ask whether the thing was ever
            # something I preserve.
            if not _rel.endswith(('.mkv', '.log', '.error', '.html')): continue
            if _rel not in _names: _orph.append(_rel)
        print(f"    ledger vs disk: {len(_rows) - len(_gone)}/{len(_rows)} rows have their file, "
              f"{len(_orph)} orphan file(s)")
        if _gone:
            # A ROW MAY EXPLAIN ITS OWN ABSENCE. `875f3374d854162a.log` was reported as
            # an unexplained gap all night while its own `note` field said "WITHDRAWN:
            # ci test probe, NOT an artefact" -- unreadable because the header had eight
            # columns and the row had nine.
            _notes = {r.get('keep_name'): (r.get('note') or '') for r in _rows}
            for g in _gone[:4]:
                why = _notes.get(g, '')
                print(f"      MISSING: {g}" + (f"  -- {why[:70]}" if why else
                                               "  -- NO EXPLANATION IN ITS ROW"))
        if _orph:
            problems.append(f"{len(_orph)} file(s) in KEEP with no ledger row -- "
                            f"preserved and unattributable")

    # CAUSES RE-ESTABLISHED AFTER PLACEHOLDER STEMS WERE EXCLUDED. The previous notes
    # described the INFLATED counts -- `log` said "the 1 unjoined is the missing file"
    # when 11 are unjoined, and `mkv` said "the 3 are frozen" when 8 are. A cause note
    # that no longer describes its count is the labels-stopped-describing-the-values
    # defect, in the field built to name causes.
    KNOWN_CAUSE = {
        'error': "21 have a real stem and genuinely no sibling -- an .error record is "
                 "named from the ERROR file's basename and an artefact from the "
                 "MASTER's, and those two bases structurally cannot meet; 3 carry the "
                 "placeholder stem `UNKNOWN-backfill`",
        'declined_by_gate_record': "same as `error` -- .error records kinded by content; "
                 "the two naming bases cannot meet",
        'unadjudicated_record': "same as `error` -- kinded by content; the two naming "
                 "bases cannot meet",
        'log':   "10 carry the DEGENERATE stem `UNKNOWN-backfill`; 1 is "
                 "875f3374d854162a.log, and its own row now explains it -- "
                 "`WITHDRAWN: ci test probe, NOT an artefact ... removed from KEEP`. "
                 "I reported that as an unexplained ledger-vs-disk gap all night while "
                 "the answer sat in a NINTH field the 8-column header made unreadable",
        'mkv':   "5 carry the DEGENERATE stem `UNKNOWN-backfill`; 3 are KEEP/frozen/ "
                 "entries whose `FROZEN-COPY-of-<id>` stem is a VALID key -- unique and "
                 "naming its source -- with genuinely no sibling, because a frozen copy "
                 "is held outside the flow by design",
        'merge_plan_report': "7 have a real stem and no sibling; the other 5 join via "
                 "the ci-report-bindings.tsv backfill",
    }
    print("    join coverage by kind:")
    for k, (tot, jn) in sorted(joins.items()):
        if not tot:
            continue
        if jn == tot:
            note = ""
        elif k in KNOWN_CAUSE:
            note = f"  PARTIAL -- {KNOWN_CAUSE[k]}"
        else:
            note = "  PARTIAL -- CAUSE NOT ESTABLISHED"
        print(f"      {k:22} {jn}/{tot}{note}")
    for k, (tot, jn) in sorted(joins.items()):
        # ZERO JOINS IS A PROBLEM ONLY WHERE NO CAUSE IS ESTABLISHED. A kind whose rows
        # STRUCTURALLY cannot join -- an .error record named from the error file's
        # basename against an artefact named from the master's -- would otherwise raise
        # this alarm forever on its very first row, which is the opposite of the truth:
        # the classifier fired, and the join was never possible.
        #
        # AND A CHECK THAT IS ALWAYS RED IS A CHECK NOBODY READS, which is the failure
        # this was built to prevent.
        if tot and jn == 0 and k not in KNOWN_CAUSE:
            problems.append(f"ledger kind '{k}': {tot} row(s), ZERO join to a sibling, "
                            f"and NO CAUSE ESTABLISHED -- a key that matches nothing "
                            f"is not a key")

    # A TRACK FILLED WITH THE WRONG LANGUAGE HAS A PERFECTLY CORRECT DURATION.
    # forensic found two DELIVERED artefacts carrying a third of their non-Japanese
    # audio as Japanese -- the master holds only ja and fr, and every other language the
    # candidate supplies is filled from Japanese. FULL_LENGTH is right, coverage is
    # right, deficit_vs_master_ms is right, and NOT ONE OF THEM CAN SEE IT.
    #
    # This is the third member of the family: a uniformly shifted subtitle track
    # (currently empty), an unnecessary interior fill (occupied), and this (occupied,
    # and it SHIPPED). Surfaced here because nothing else in my tooling reaches it.
    print("  == content ==")
    out = sh(["python3", os.path.join(HERE, "cross_language_fill.py")], timeout=180)
    dl = [l for l in out.splitlines() if "DELIVERED carrying" in l]
    worst = [l for l in out.splitlines() if "worst delivered" in l]
    # MATCH A LABEL I CONTROL, AND NOTICE WHEN IT IS ABSENT. This grepped for
    # "does not fill from" -- a phrase I changed in the tool two minutes after writing
    # the grep, so the line silently vanished from the report. FIFTH instance tonight
    # of a pattern written against a format that then changed, and the first where I
    # BROKE THE FORMAT MYSELF.
    #
    # So: match the stable half, and say so when nothing matches rather than printing
    # nothing at all -- a missing line and a zero must not look the same.
    opp = [l for l in out.splitlines() if "condition could arise" in l]
    # MATCH THE SHAPE OF A TIMESTAMP, NOT THE WORD "captured" -- the section HEADING
    # contains that word too and leaked into the set as
    # "-- historical damage is not ongoing damage:". A substring filter picking up its
    # own label is the same trap as `pgrep -f` matching my shell, in a string.
    when = sorted({m.group(0) for l in out.splitlines()
                   for m in [re.search(r'\d{4}-\d{2}-\d{2}T[\d:]+Z', l)] if m})
    print(f"    {dl[-1].strip() if dl else 'no answer'}")
    if worst: print(f"    {worst[-1].strip()}")
    if opp:   print(f"    condition could arise in{opp[-1].split(':')[-1]}")
    else:     print("    condition-opportunity line NOT FOUND in the detector's output "
                    "-- the label changed, this is a parse gap, not a zero")
    # TRACK-LEVEL EXPOSURE. forensic's metric, and it is where the defect lives: the
    # condition can only arise on a REBUILT track, so a job count understates nothing
    # and overstates the unit. Both F-54 artefacts rebuilt every track they had.
    for lbl in ("tracks rebuilt across", "in a language the master lacks"):
        m = [l for l in out.splitlines() if lbl in l]
        if m: print(f"    {m[-1].strip()}")
    try:
        n = int(dl[-1].split(":")[1].strip()) if dl else 0
    except Exception:
        n = 0
    if n:
        # SCOPED TWO WAYS, BECAUSE A LIVE `PROBLEMS` LINE IS READ AS ONGOING.
        #   WHEN   both hits were captured 2026-09-04T15:14:53Z, five hours before
        #          this run began and by an earlier image.
        #   WHETHER THE QUESTION WAS EVEN ASKED  `vmsam-forensic`'s point: F-54 needs a
        #          candidate supplying a language THE MASTER CANNOT FILL. Measured over
        #          this run's twelve delivered jobs: ZERO. Eight are single-language.
        #
        #     NOT  "0 artefacts carry a foreign-language fill"
        #     BUT  "0 of N carry it; the condition was present in M of N jobs"
        #
        # WITH M = 0 A CLEAN RESULT IS *UNTESTED*, NOT *PASSED* -- a green light with
        # nothing behind it.
        problems.append(
            f"{n} REPAIR-PATH artefact(s) carry a foreign-language audio fill -- a correct "
            f"duration in the wrong language, invisible to every duration check. "
            f"SCOPE: the repair path over PREVIOUSLY-REFUSED files in a test container -- ci-pair joined my 28 accepted ids against the 315-entry /errors corpus and found 28 of 28 inside it, so this is NOT a rate over production deliveries. HISTORICAL (captured {', '.join(when)}); THIS RUN'S DELIVERIES ARE "
            f"UNTESTED, NOT CLEAN -- the condition arose in 0 of 12 jobs")

    # DRIFT SINCE CAPTURE. `size_at_capture` and `mtime_at_capture` were added because
    # A HARD LINK IS NOT A SNAPSHOT -- a producer rewriting in place rewrites the
    # preserved copy with it. THE COLUMNS EXISTED ALL NIGHT AND WERE NEVER READ.
    #
    # First run found a hit: 9c22d7d987840f1f.mkv at +385 MB with the mtime IDENTICAL,
    # which cannot be a rewrite. It was a TORN METADATA READ -- four separate `stat`
    # processes, so `%s` caught the file mid-write and `%Y` caught the final timestamp
    # in the same second. Now one stat call per row; the row is annotated, not corrected.
    if os.path.exists(led):
        _drift, _ok, _nocol = [], 0, 0
        for r in _rows:
            _p = '/config/output/KEEP/' + (r.get('keep_name') or '')
            _sz, _mt = r.get('size_at_capture'), r.get('mtime_at_capture')
            if not (_sz or '').isdigit() or not (_mt or '').isdigit():
                _nocol += 1; continue
            if not os.path.exists(_p): continue
            if os.path.getsize(_p) != int(_sz) or int(os.path.getmtime(_p)) != int(_mt):
                _drift.append(r.get('keep_name'))
            else:
                _ok += 1
        print(f"    drift since capture: {_ok} unchanged, {len(_drift)} CHANGED, "
              f"{_nocol} uncheckable (predate the columns)")
        # A SETTLED READING COUNTS AS AN EXPLANATION. `ci-settled-sizes.tsv` records
        # capture-time against settled size for artefacts linked mid-write -- the link
        # is eager because the runner deletes its outputs, so the second reading is the
        # fix rather than a later link.
        _settled = {}
        _sf = '/config/output/KEEP/ci-settled-sizes.tsv'
        if os.path.exists(_sf):
            for _l in open(_sf):
                if _l.startswith('#') or _l.startswith('keep_name'): continue
                _p = _l.rstrip('\n').split('\t')
                if len(_p) >= 4: _settled[_p[0]] = (_p[2], _p[3])
        for d in _drift[:3]:
            _note = next((x.get('note') or '' for x in _rows if x.get('keep_name') == d), '')
            _s = _settled.get(d)
            if _s and _s[0]:
                _dl = _s[1]
                _dl = f"+{_dl}" if _dl and not _dl.startswith('-') else _dl
                print(f"      CHANGED: {d}  -- settled at {_s[0]} ({_dl} bytes), recorded")
            elif _note:
                print(f"      CHANGED: {d}  -- {_note[:60]}")
            else:
                print(f"      CHANGED: {d}  -- NO EXPLANATION IN ITS ROW")
                problems.append(f"{d} has changed since capture with no explanation "
                                f"in its row -- a hard link is not a snapshot")

    # DEGENERATE KEYS. A key that matches EVERYTHING passes a check written for a key
    # that matches NOTHING; only asking what the matched rows ARE separates them.
    # Gated on a positive control -- the sweep must re-find the known defect before its
    # result is readable, because its first version was structurally blind to it.
    out = sh(["python3", os.path.join(HERE, "check_degenerate_keys.py")], timeout=120)
    ctl = [l for l in out.splitlines() if "positive control" in l]
    deg = [l for l in out.splitlines() if "covers" in l and "rows" in l]
    print(f"    degenerate keys: {len(deg)} " +
          ("(control PASS)" if ctl and "PASS" in ctl[-1] else "-- CONTROL FAILED, result unreadable"))
    for d in deg[:3]:
        print(f"      {d.strip()}")
    if ctl and "PASS" not in ctl[-1]:
        problems.append("the degenerate-key sweep FAILED ITS POSITIVE CONTROL "
                        "-- it cannot find a defect it is known to contain")
    elif len(deg) > 1:
        problems.append(f"{len(deg)} degenerate key(s) in my ledgers -- one value "
                        f"covering rows that cannot be one thing")

    print("  == suites ==")
    out = sh(["python3", os.path.join(HERE, "check_empty_assertions.py")], timeout=120)
    tail = [l for l in out.splitlines() if "finding(s)" in l]
    hits = [l for l in out.splitlines() if l.strip() and "finding(s)" not in l
            and "SHAPES ONLY" not in l]
    print(f"    {tail[-1].strip() if tail else 'no answer'}")
    for h in hits[:5]:
        print(f"      {h.strip()}")
        problems.append(f"assertion that cannot fail: {h.strip()[:80]}")

    print("  == deploy condition ==")
    out = sh(["python3", os.path.join(HERE,"deploy_condition.py")], timeout=120)
    line = [l for l in out.splitlines() if "MET" in l or "CANNOT" in l]
    print(f"    {line[-1].strip() if line else 'no answer'}")

    print()
    if problems:
        print(f"  PROBLEMS: {len(problems)}")
        for p in problems: print(f"    - {p}")
        return 1
    print("  no problems detected")
    return 0

sys.exit(main())
