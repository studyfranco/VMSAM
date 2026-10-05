"""Timing log lines for the repair modules.

announced() logs START/END around an external command (tool, input basename, seconds, exit
code); timed_phase() logs ENTER/EXIT with elapsed seconds around a phase function. Lines go
through tools.dev_log under the caller's prefix. Neither helper changes what a call does.
"""
from contextlib import contextmanager
import functools
from os import path
import time

import tools


# Build stamp written on fabricated tracks (`VMSAM=<sha>`) and the plan line (`build=<sha>`).
# Read from VMSAM_GIT_COMMIT via tools.get_git_commit() (the image ships no .git).
BUILD_SHA_LENGTH = 8
_BUILD_LOGGED = []


def build_sha():
    """Short commit sha of the running build, or 'unknown' (logged once)."""
    commit = tools.get_git_commit()
    sha = commit[:BUILD_SHA_LENGTH] if commit and commit not in ("unknown", "IDK") else "unknown"
    if sha == "unknown" and not _BUILD_LOGGED:
        _BUILD_LOGGED.append(True)
        tools.log_always(f"repair: build_commit_unknown VMSAM_GIT_COMMIT={commit!r} -- fabricated "
                         f"tracks and the plan line carry VMSAM=unknown\n")
    return sha


def _basename(input_path):
    return path.basename(str(input_path)) if input_path else "-"


@contextmanager
def announced(prefix, tool, input_path, media_s=None):
    """Log START before a command and END after it, including when it raises.

    Yields a dict whose "exit" key the caller sets to the command's exit code. With media_s
    (seconds of media decoded), END also reports media_per_wall, the decode speed.
    """
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
        wall = time.monotonic() - started
        speed = (f" media_s={round(float(media_s), 2)} media_per_wall="
                 f"{round(float(media_s) / wall, 2) if wall > 0 else None}"
                 if media_s is not None else "")
        tools.dev_log(f"{prefix}: command END tool={tool} input={name} "
                      f"seconds={round(wall, 2)} exit={call['exit']}{speed}\n")


def _outcome(value):
    if isinstance(value, tuple) and value:
        return "ok" if value[0] else f"refused:{value[1] if len(value) > 1 else None}"
    return "ok" if value else "refused"


def timed_phase(prefix, name, input_of):
    """Decorator logging ENTER/EXIT around a phase; input_of(*args, **kwargs) names its input."""
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
