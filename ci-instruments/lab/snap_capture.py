"""Capture every adjust_delay_to_frame call: delay in, delay out, both rates, call site.

LAB TOOL. NEVER COMMITTED. Separate image at build time -- the image serving the
hundred-file run is not touched.

WHY THIS AND NOT AN ARTEFACT: the post-snap delay is written to sys.stdout at
mergeVideo :1495/:1511 and the pre-snap set to tools.logs at :2064. TWO SINKS, ONLY
ONE OF WHICH BECOMES AN ARTEFACT -- so no number of runs puts both in one file. The
hook is the only place both exist at once.

THE FIELD THAT DECIDES dev-3's HEADLINE is best_quality_fps: get_good_frame snaps to
the MINIMUM of the two declared rates, this method snaps to the BEST-QUALITY video's
own rate, and they coincide exactly when the best-quality video is the lower-rate one.
Every mixed pair in this corpus has 23.976 on the master side and every other rate is
higher, so IF the best-quality video is always the 23.976 file, P-C's population is
empty and the second snap is a no-op on the tournament path everywhere. That is the
outcome that REFUTES the stronger claim, and it is why the rate is carried
EXPLICITLY rather than inferred from the pair.
"""
import datetime, inspect, json, os, traceback

OUT = os.environ.get('SNAP_FIXTURE_DIR', '/config/output/fixtures/snap')
_seq = [0]


def _fps(obj):
    try:
        v = getattr(obj, 'video', None) or {}
        return {"FrameRate": v.get("FrameRate"), "FrameRate_Mode": v.get("FrameRate_Mode")}
    except Exception:
        return {"FrameRate": None, "FrameRate_Mode": None}




def _pair(self_obj):
    """WHICH TWO FILES WERE COMPARED -- digests only, never paths.

    dev-3's P-0 ask, and it is what makes an EMPTY table readable. A job with three
    or more files compares best_video against each one, so a row from a mixed-rate
    JOB may belong to a same-rate PAIR inside it. Without this, P-A..P-E silently
    stop being testable; with it, zero rows for the mixed-rate pairing means the snap
    NEVER RAN on that pairing -- a finding, not a null.

    dev-2 measured zero adjust_delay_to_frame calls across nine cases because
    first_delay_test RAISES on files that are in the error corpus BECAUSE their delay
    measurement failed. The snap only ever sees pairs that get that far.
    """
    import hashlib
    def d(o):
        p = getattr(o, 'filePath', None) or getattr(o, 'file_path', None)
        if not p:
            return None
        return hashlib.sha256(str(p).encode()).hexdigest()[:16]
    o1 = getattr(self_obj, 'video_obj_1', None)
    o2 = getattr(self_obj, 'video_obj_2', None)
    best = getattr(self_obj, 'video_obj_with_best_quality', None)
    return {"obj_1_sha16": d(o1), "obj_2_sha16": d(o2), "best_sha16": d(best),
            "language": getattr(self_obj, 'language', None)}


def _write(rec, prefix):
    """Serialise one capture record. THE ONLY WRITER -- both record() and
    record_gate() go through it.

    Owner's rule, applied to my own lab: over five lines, twice, same purpose. It was
    already twice when I wrote record_gate, and the second copy did not exist yet --
    the rule FIRES WHEN YOU WRITE THE SECOND COPY, which is exactly when I was about
    to. The drift this prevents is concrete: the tmp-then-rename discipline below is
    what stops a half-written fixture being read as data, and a second hand-rolled
    writer would have been one refactor away from losing it.

    AN INSTRUMENT MUST NEVER BREAK THE THING IT MEASURES. Every failure is swallowed
    and logged; a capture hook that raises into the merge turns an observation into a
    lost job. My first record_gate called a _write() that did not exist and would have
    raised NameError straight into test_if_constant_good_delay.
    """
    try:
        os.makedirs(OUT, exist_ok=True)
        _seq[0] += 1
        rec.setdefault("seq", _seq[0])
        rec.setdefault("captured_utc",
                       datetime.datetime.now(datetime.timezone.utc)
                               .strftime('%Y-%m-%dT%H:%M:%SZ'))
        path = os.path.join(
            OUT, f"{prefix}_{rec['captured_utc'].replace(':', '')}_{rec['seq']:05d}.json")
        tmp = path + '.part'
        with open(tmp, 'w') as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, path)      # compute, then rename: never a half-written fixture
    except Exception:
        try:
            with open(os.path.join(OUT, 'capture-errors.log'), 'a') as fh:
                fh.write(traceback.format_exc() + "\n---\n")
        except Exception:
            pass


def record(self_obj, delay_in, delay_out, caller_line):
    try:
        # THE COUNTER IS _write's. It was incremented here AND there, so seq advanced
        # by two per snap and the series read 1, 3, 5 -- and a GAP IN A SEQUENCE FIELD
        # IS INDISTINGUISHABLE FROM A LOST CAPTURE, which is the one thing seq exists
        # to rule out. Double-counting turned the completeness check into a false alarm.
        os.makedirs(OUT, exist_ok=True)
        best = getattr(self_obj, 'video_obj_with_best_quality', None)
        o1 = getattr(self_obj, 'video_obj_1', None)
        o2 = getattr(self_obj, 'video_obj_2', None)
        din = float(delay_in) if delay_in is not None else None
        dout = float(delay_out) if delay_out is not None else None
        rec = {
            "schema": "adjust_delay_to_frame/1",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc)
                                    .strftime('%Y-%m-%dT%H:%M:%SZ'),
            # CALL SITE by the CALLER's line number, not by guessing from state.
            # :229 is the forced path; :650 and :656 are the tournament.
            "caller_line": caller_line,
            "call_site": ("forced_229" if caller_line and abs(caller_line - 229) < 4
                          else "tournament" if caller_line else "unknown"),
            "delay_in_ms": din,
            "delay_out_ms": dout,
            # moved_ms is the whole question. NOT rounded here -- a rounding in the
            # instrument would sit exactly where the finding is.
            "moved_ms": (None if (din is None or dout is None) else dout - din),
            "best_quality": _fps(best),
            "obj_1": _fps(o1),
            "obj_2": _fps(o2),
            # which of the pair IS the best-quality one -- dev-3's deciding field
            "best_is_obj_1": (best is o1) if (best is not None and o1 is not None) else None,
            "best_is_obj_2": (best is o2) if (best is not None and o2 is not None) else None,
            "language": getattr(self_obj, 'language', None),
            "pair": _pair(self_obj),
        }
        _write(rec, "snap")            # THE ONLY WRITER -- see _write's docstring
    except Exception:
        try:
            with open(os.path.join(OUT, 'capture-errors.log'), 'a') as fh:
                fh.write(traceback.format_exc() + "\n---\n")
        except Exception:
            pass


def caller_line():
    """The line that called adjust_delay_to_frame, two frames up from here."""
    try:
        return inspect.stack()[2].lineno
    except Exception:
        return None

GATE_BOUND_MS = 500.0   # mergeVideo.py:244, abs(calculated - first) < 500

def record_gate(self_obj, first, second, gate_passed, exc):
    """dev-3's selection-gate column.

    THE PREDICATE MENTIONS THE QUANTITY BEING MEASURED, so its bound travels beside
    the statistic. The gate reduces to abs(second*1000) < 500, which means every
    delay the SNAP hook sees was already selected on a property of the delay. A
    spread computed over delay_in_ms without this column inherits that selection and
    would report "no large stage-1/stage-2 disagreements" as a fact when it is a
    consequence of membership.

    RECORDED ON BOTH PATHS. gate_passed=False rows are the ones absent from the snap
    capture by construction, and they are the reason this exists -- a capture of the
    successes alone reproduces the bias it was built to measure.

    ratio_to_bound near 1 means the gate is binding on real data. If it comes back
    far below 1 on every row, the threshold never fires, which is a more useful
    finding than the guard it came from: a threshold that never fires is decoration
    with a number on it.
    """
    try:
        dev = None if second is None else abs(float(second) * 1000.0)
    except Exception:
        dev = None
    _write({
        "kind": "gate",
        "delay_first_method_ms": None if first is None else float(first),
        "delay_second_method_s": None if second is None else float(second),
        "gate_deviation_ms": dev,                      # abs(second*1000)
        "gate_bound_ms": GATE_BOUND_MS,
        "ratio_to_bound": None if dev is None else round(dev / GATE_BOUND_MS, 4),
        "gate_passed": bool(gate_passed),
        # A refusal that carries no reason is a count, not an observation.
        "refusal": None if exc is None else str(exc)[:120],
        "pair": _pair(self_obj),
        "call_site": caller_line(),
    }, "gate")
