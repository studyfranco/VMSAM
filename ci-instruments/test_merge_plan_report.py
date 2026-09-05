#!/usr/bin/env python3
"""The owner's report, tested on the four parts the architect named.

He is wiring the call site himself and wants the document in the test container. The
module is 2740 lines, is present at the DEPLOYED commit aefdf8a, and `grep -rl
merge_plan_report src/` returns only the module -- SO NOTHING HAS EVER INVOKED IT. The
first run in the container will be its first run anywhere except by hand.

FOUR PARTS, EACH FAILING DIFFERENTLY:
  1 does it run at all -- a traceback in the repair path is the failure that matters
  2 does the file land at <produced>.merge_plan.log, WRITTEN BEFORE transport returns
  3 does the extraction round-trip -- BY BYTES; and a CHARACTER slice must FAIL,
    or the test cannot tell the two apart
  4 does it open -- <!doctype> as the FIRST BYTE, meta charset utf-8, no external refs

Part 3's negative control is the architect's specific instruction and it is the whole
point: dev-4's own first check sliced by characters and got 17467 where the header said
17466. The prefix was right and the reader was wrong.
"""
import sys, os, re, json, traceback
sys.path.insert(0, '/home/vmsam/src/VMSAM/src')

OUT = '/tmp/claude-1000/-home-vmsam-src-VMSAM-WIP-ci/45d5406a-a849-4c6b-8a9a-8830f03f753c/scratchpad/rpt'
results = []
def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

# PROVENANCE BEFORE VERDICT. `vmsam-dev-4` measured that their lab validator had
# `sys.path.insert(0, "/home/vmsam/src/VMSAM/src")` hardcoded all campaign, and that
# checkout was sitting on a PEER'S REVIEW BRANCH -- so every `VALIDATED 23/23` was
# about a tree they did not control, unnoticed BECAUSE THE NUMBER WAS PLAUSIBLE.
# I checked mine on their warning: a different miss, same shape. My tree tracks the
# TIP; the container runs an OLDER IMAGE. My 15/15 was about the tip.
#
# A SUITE THAT DOES NOT NAME THE BYTES IT TESTED HAS NOT SAID WHAT PASSED.
import subprocess as _sp
def _sh(c):
    try: return _sp.run(c, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return "?"
REPO = '/home/vmsam/src/VMSAM'
_tested = _sh(["git","-C",REPO,"rev-parse","HEAD:src/merge_plan_report.py"])
_head   = _sh(["git","-C",REPO,"rev-parse","HEAD"])
_img = os.environ.get('CI_CONTAINER_COMMIT','')
if not _img:
    _h = _sh(["curl","-s","--max-time","15",os.environ.get("VMSAM_TEST_API", "http://" + os.environ.get("VMSAM_TEST_HOST","showgestionar-test") + ":8080") + "/health"])
    try: _img = json.loads(_h).get("git_commit","")
    except Exception: _img = ""
_img_blob = _sh(["git","-C",REPO,"rev-parse",f"{_img}:src/merge_plan_report.py"]) if _img else ""
print(f"  WORKTREE  {REPO}/src/merge_plan_report.py")
print(f"            worktree HEAD {_head[:12]}   blob {_tested[:12]}")
if _img_blob and _img_blob != _tested:
    print(f"  CONTAINER {_img[:12]}   blob {_img_blob[:12]}   <-- WORKTREE DIFFERS")
    print( "            the worktree is NOT what runs; this suite imports the container's")
elif _img_blob:
    print(f"  CONTAINER {_img[:12]}   blob {_img_blob[:12]}   same bytes")
else:
    print( "  CONTAINER unreachable -- cannot say whether these are the running bytes")
print()

# TEST THE DEPLOYED BYTES, NOT THE WORKTREE'S. Printing the mismatch was the first
# fix and it is not the fix: a banner saying "this suite does not test what the
# container is running" is a warning I would learn to read past. `vmsam-dev-4`'s
# version of this ran a whole campaign against a peer's review branch.
#
# So: extract src/ AT THE CONTAINER'S OWN COMMIT and import from there. The worktree
# stops mattering. If the container is unreachable, fall back to the worktree AND SAY
# SO -- an untested fallback that pretends to be the real thing is the defect itself.
_srcdir = REPO + '/src'
_mode = "WORKTREE (container unreachable)"
if _img:
    import tempfile, tarfile, io
    _ar = _sp.run(["git","-C",REPO,"archive",_img,"src/"], capture_output=True, timeout=120)
    if _ar.returncode == 0:
        _tmp = tempfile.mkdtemp(prefix="ci-deployed-")
        tarfile.open(fileobj=io.BytesIO(_ar.stdout)).extractall(_tmp)
        _srcdir = os.path.join(_tmp, 'src')
        _mode = f"DEPLOYED BYTES from {_img[:12]}"
    else:
        _mode = f"WORKTREE (could not archive {_img[:12]})"
sys.path.insert(0, _srcdir)
print(f"  IMPORTING FROM: {_mode}")
print()

try:
    import merge_plan_report as R
except Exception as e:
    print("  [FAIL] import merge_plan_report --", str(e)[:120]); sys.exit(2)

# A job dict with the shape the module reads. Every access in the module is .get(),
# so a sparse dict exercises the DEFAULT paths -- which is what a first real call
# will hit if the owner's wiring omits a key.
# THE CONTRACT IS DERIVED, NOT COPIED. The module's own refusal says how:
#   "Pass `parse_job_log(<the emitted bytes>)`"
# and dev-4 built validate_job's key set as frozenset(parse_job_log("").keys()) so it
# cannot drift from the reader that builds the dict.
#
# My first fixture hand-listed FIFTEEN keys, derived from bracket accesses. The tip now
# requires TWENTY-TWO -- build, candidate_path, declined, failed, master_path,
# undelivered, unparsed were added -- and a hand-written list would have gone stale in
# the hours between. A SECOND IMPLEMENTATION OF A CONTRACT IS A SECOND THING TO KEEP
# CORRECT, which is the argument dev-4 used to refuse hand-listing it and I ignored
# when I wrote the fixture.
job = dict(R.parse_job_log(""))
for k, v in {
    "summary_counts": "2 pieces master, 3 pieces candidate",
    "master_line_present": True,
    "output_check": {"frame_rate": "23.976", "frame_rate_original": None},
}.items():
    if k in job:
        job[k] = v
produced = os.path.join(OUT, "artefact_test.mkv")
open(produced, "wb").write(b"\x00" * 16)

# --- 1 DOES IT RUN ---------------------------------------------------------------
try:
    dest, entry = R.write_report(job, "ci-test-001", "ci self-test", produced,
                                 caveats=("this is a ci self-test, not a real merge",))
    check("1 runs without raising", True, f"returned {os.path.basename(dest)}")
except Exception:
    check("1 runs without raising", False, traceback.format_exc().strip().splitlines()[-1][:100])
    print(traceback.format_exc())
    sys.exit(1)

# --- 2 DESTINATION ---------------------------------------------------------------
# CHANGED BEHAVIOUR, NOT A REGRESSION: the owner moved the report from .log to .html
# so it can be double-clicked. Updated only after RUNNING against the published tip --
# the architect's description of the change was right, but I do not edit assertions to
# match a description, because a suite that does asserts someone's sentence.
expect = produced + ".merge_plan.html"
check("2a destination is <produced>.merge_plan.html", dest == expect,
      f"got {os.path.basename(dest)}")
check("2b file exists on disk", os.path.exists(dest),
      f"{os.path.getsize(dest) if os.path.exists(dest) else 0} bytes")
disk = open(dest, encoding="utf-8").read()

# --- 3 EXTRACTION ROUND-TRIP -----------------------------------------------------
m = re.search(r'MERGE_PLAN_HTML\s+(\S+)\s+bytes=(\d+)', entry)
check("3a transport entry carries a byte-length header", bool(m),
      m.group(0) if m else entry[:70])
if m:
    n = int(m.group(2))
    tail = entry[m.end():]
    tail = tail[1:] if tail[:1] in ("\n", " ") else tail
    raw = tail.encode("utf-8")
    sliced_bytes = raw[:n].decode("utf-8", errors="replace")
    check("3b BYTE slice of N reproduces the document", sliced_bytes == disk,
          f"header says {n} bytes; document is {len(disk.encode('utf-8'))} bytes")
    # THE NEGATIVE CONTROL THE ARCHITECT ASKED FOR
    sliced_chars = tail[:n]
    differs = sliced_chars != disk
    check("3c NEGATIVE CONTROL: character slice must NOT match", differs,
          f"char slice {len(sliced_chars)} chars vs {len(disk)} chars"
          + ("" if differs else "  <-- test is BLIND: it cannot tell bytes from chars"))

# --- 4 DOES IT OPEN --------------------------------------------------------------
rawdoc = open(dest, "rb").read()
check("4a <!doctype is the FIRST BYTE", rawdoc[:9].lower() == b"<!doctype",
      repr(rawdoc[:20]))
check("4b declares utf-8", b"charset=utf-8" in rawdoc.lower() or b'charset="utf-8"' in rawdoc.lower())
ext = re.findall(rb'(?:src|href)\s*=\s*["\'](https?:|//)', rawdoc)
check("4c zero external references", not ext, f"{len(ext)} found")
try:
    rawdoc.decode("utf-8"); okdec = True
except UnicodeDecodeError as e:
    okdec = False
check("4d decodes as utf-8 end to end", okdec)
# 4e WAS A NO-OP AND I ONLY SAW IT WHEN IT PASSED SAYING "none present".
# `bool(accents) or True` is True unconditionally -- it asserted nothing, in a suite
# whose whole value is that its negative control can fail. Replaced with the actual
# behaviour change: the document is now ENGLISH.
lang_en = b'lang="en"' in rawdoc
check("4e document declares lang=en (was fr)", lang_en,
      "the language change is the measurable half; the DECIMAL POINT and media-name "
      "behaviour need a REAL job -- a synthetic fixture carries no media names, so "
      "asserting they traverse would test my fixture, not the module")

# --- PART 5: THE PARSER'S FOREIGN-LINE COUNTER ---------------------------------
# ADDED 2026-09-05. My four parts test the report's PACKAGING -- destination, byte
# round-trip, doctype, encoding. They test NOTHING about its PARSING, and I only
# discovered that when `vmsam-dev-1` proposed ungating a `[change_point_locator]`
# line and traced it to `merge_plan_report.py:489-492`, where any line not starting
# `repair: ` and not in a four-item allowlist is counted as FOREIGN.
#
# I OWN A HARNESS FOR THIS MODULE AND IT WOULD NOT HAVE SEEN THAT CHANGE AT ALL.
# 11 of 11 passing, and zero of them touch the counter the change moves.
#
# This is a CHARACTERISATION test, not a specification: it pins what the parser does
# TODAY so the change becomes visible. If dev-2 adds "[change_point_locator]" to the
# allowlist, 5c flips and SHOULD be updated -- a failure here after that change is
# the test working, not the module breaking.
try:
    def _foreign(text):
        j = R.parse_job_log(text)
        return list(j.get("foreign_lines") or [])

    known_ok = "repair: plan piecewise_constant language=en quantum=129 pieces=m0-1\n"
    check("5a a `repair: ` line is NOT foreign",
          _foreign(known_ok) == [],
          f"foreign={_foreign(known_ok)}")

    allowed = "Merged errors:\nLogs:\nWe was in first_delay_test\nMultiple delay found\n"
    check("5b the four allowlisted prefixes are NOT foreign",
          _foreign(allowed) == [],
          f"foreign={_foreign(allowed)}")

    # UPDATED 2026-09-05 when dev-4 landed 20e1d17, which is what 5c was for.
    #
    # AND 5c AS FIRST WRITTEN COULD NOT HAVE DONE ITS JOB. dev-4 found it: the ruling
    # was to add "[change_point_locator]" to the allowlist tuple, and their branch
    # tested `line.startswith(...)` on the RAW line -- which begins with TWO TABS
    # (`change_point_locator.py:233`). The specified patch was a NO-OP and 5c would
    # have passed before and after it. My fixture bytes were right; the assertion was
    # about a prefix test those bytes never reach.
    #
    # ASSERT BOTH HALVES. "the counter went quiet" passes equally on a SILENCE LIST,
    # and a silence list is the version of this change that throws the numbers away.
    succ = ("\t\t[change_point_locator] offset_ms=-983.5 points=-8 quantum_ms=129 "
            "window_s=60.0 segments=3 change_points=2\n")
    j = R.parse_job_log(succ)
    check("5c a [change_point_locator] measurement is NOT foreign",
          list(j.get("foreign_lines") or []) == [],
          f"foreign={list(j.get('foreign_lines') or [])}")

    meas = list(j.get("locator_measurements") or [])
    check("5c2 and the NUMBERS ARE RENDERED, not merely silenced",
          len(meas) == 1 and meas[0].get("offset_ms") == "-983.5"
          and meas[0].get("points") == "-8" and meas[0].get("quantum_ms") == "129"
          and meas[0].get("window_s") == "60.0" and meas[0].get("segments") == "3"
          and meas[0].get("change_points") == "2",
          f"locator_measurements={meas}")

    # 5e: THE PROSE HALF. `if tools.dev:` sits on `_log` ITSELF, so ungating the
    # function ungates 25 call sites, 24 of them prose -- including :324
    # f"probe at {s}s failed: {error}" where {error} is an arbitrary ffprobe exception
    # CARRYING THE INPUT PATH. That is the confidentiality rule verbatim, and my
    # `carries_path: False` measurement settled it for ONE line and not for the change.
    prose = ("\t\t[change_point_locator] probe at 480s failed: "
             "ffprobe: /srv/SYNTHETIC-LIBRARY-ROOT/Some Show Name/S01E01 - Title.mkv: no such file\n")
    jp = R.parse_job_log(prose)
    notes = list(jp.get("locator_notes") or [])
    body = json.dumps(jp)
    check("5e a PROSE locator line is kept as a FACT in locator_notes",
          len(notes) == 1 and notes[0].get("carries_path") is True,
          f"locator_notes={notes}")
    check("5e2 and NO PATH survives into the row",
          "/srv/SYNTHETIC-LIBRARY-ROOT" not in body and "/" not in (notes[0].get("tail","") if notes else "/"),
          "the path is stripped, which is the half the redaction was built for")
    # 5e3 IS A KNOWN FAILURE AND IS LEFT RED DELIBERATELY.
    # dev-4 measured "no fragment of the title survives in the row" on THEIR probe
    # failure. On mine the PATH goes and the TITLE WORDS REMAIN: a fixture titled
    # "Some Show Name/S01E01 - Title.mkv" yields tail "show namese  titlemkv ...".
    # Lowercased and de-punctuated, but the words are the title's.
    #
    # NOT A DEFECT IN THE REPORT: a report written beside the media is runtime output
    # and may name its own file. IT IS A DEFECT IN TREATING `tail` AS REDACTED --
    # the redaction is for PATHS, not for TITLES, and anything I quote from a `tail`
    # into a message, a commit or a filing would carry title words. That is the
    # confidentiality rule verbatim and the field's name does not warn you.
    tail = (notes[0].get("tail","") if notes else "")
    check("5e3 KNOWN RED: title words survive in `tail` -- do not cite it",
          not any(w in tail.lower() for w in ("show name", "title")),
          f"tail={tail!r} -- LEFT FAILING ON PURPOSE as a standing warning, not a bug report")

    # NEGATIVE CONTROL: the counter must be able to stay at zero, or 5c proves nothing.
    check("5d NEGATIVE CONTROL: the counter is not simply always non-zero",
          _foreign(known_ok + allowed) == [],
          "a mix of repair: and allowlisted lines yields zero foreign")
except Exception as e:
    check("5 parser foreign-line coverage", False, f"{type(e).__name__}: {str(e)[:110]}")

print()
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"  {len(results)-n_fail} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
