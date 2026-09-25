"""repair_log.py -- the repair's own timing lines (ADDENDUM 26.5, owner's precision 2026-09-25).

The owner withdrew the dev-mode wrapper on the frozen launchers (c09234de, reverted by 26068c1e):
"what he wants is `tools.dev_log(...)` lines IN OUR OWN MODULES" -- a START / END pair around
every external command the repair launches, naming the tool, the input's BASENAME (never the
full path), the seconds and the exit code; and an ENTER / EXIT pair on every phase of repair(),
chimeric() and apply_plan() with the elapsed seconds (the UTC millisecond stamp is `tools.stamp`'s,
on the `orchestrator:` / `chimeric:` / `scene_anchor:` prefixes). The frozen legacy phases stay
silent -- the owner's call. Both helpers only WRITE LINES: nothing here changes what a call does.
"""
from contextlib import contextmanager
import functools
from os import path
import time

import tools


def _basename(input_path):
    return path.basename(str(input_path)) if input_path else "-"


@contextmanager
def announced(prefix, tool, input_path):
    """START before the command, END after it -- also when it raises (`exit=raised:<class>`),
    because a command that hangs or throws is exactly the one a reader needs located. The caller
    sets `call["exit"]` to the exit code it read."""
    call = {"exit": None}
    name = _basename(input_path)
    tools.dev_log(f"{prefix}: command START tool={tool} input={name}\n")
    started = time.monotonic()
    try:
        yield call
    except BaseException as error:
        call["exit"] = f"raised:{type(error).__name__}"
        raise
    finally:
        tools.dev_log(f"{prefix}: command END tool={tool} input={name} "
                      f"seconds={round(time.monotonic() - started, 2)} exit={call['exit']}\n")


def _outcome(value):
    if isinstance(value, tuple) and value:
        return "ok" if value[0] else f"refused:{value[1] if len(value) > 1 else None}"
    return "ok" if value else "refused"


def timed_phase(prefix, name, input_of):
    """ENTER / EXIT around one phase function; `input_of(*args, **kwargs)` names its input."""
    def wrap(function):
        @functools.wraps(function)
        def run(*args, **kwargs):
            subject = _basename(input_of(*args, **kwargs))
            tools.dev_log(f"{prefix}: phase ENTER name={name} input={subject}\n")
            started = time.monotonic()
            outcome = "raised"
            try:
                value = function(*args, **kwargs)
                outcome = _outcome(value)
                return value
            except BaseException as error:
                outcome = f"raised:{type(error).__name__}"
                raise
            finally:
                tools.dev_log(f"{prefix}: phase EXIT name={name} input={subject} "
                              f"seconds={round(time.monotonic() - started, 2)} "
                              f"outcome={outcome}\n")
        return run
    return wrap
