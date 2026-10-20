"""Shared pool for the repair stage's low-memory parallel jobs (decode, scene-detect, hash,
fingerprint extraction).

`video.ffmpeg_pool_audio_convert` is closed by `merge_video_repair.retire_ffmpeg_pools` before
repair starts, so repair code cannot submit to it; this is a separate, repair-local pool for the
same class of job. Thread-based, not process-based: every job here wraps an ffmpeg/ffprobe/fpcalc
subprocess, so the job's memory lives in the subprocess, not in this process's thread -- a thread
pool gets the same overlap as a process pool here without a second process pool's fixed cost.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import tools

_pool = None
_pool_lock = threading.Lock()


def pool_size():
    """Worker count, sized the same way `main_gestionar_show.py` sizes its own ffmpeg pool."""
    return max(1, int(tools.core_to_use / 1.6))


def get_pool():
    """Return the shared repair-stage pool, built lazily at the current `pool_size()`.

    Process-lifetime, like the pipeline's own ffmpeg pools: never closed by a caller.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(max_workers=pool_size())
    return _pool


def run_parallel(jobs):
    """Submit independent, bounded-memory zero-arg callables and return their results in order.

    Equivalent to calling each job in sequence and collecting its result: the first job to raise
    raises here too, once its turn in `jobs`'s order is reached (not necessarily the first to
    finish) -- a caller that wants its own exception priority among several jobs should submit
    to `get_pool()` directly instead, as `video_offset_plan.measure_video_offset` does.
    """
    futures = [get_pool().submit(job) for job in jobs]
    return [future.result() for future in futures]
