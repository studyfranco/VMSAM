"""Fusion worker of the movie side: merges a rejected file with its movie's file.

Same machinery as the show's fusion worker: an in-memory queue, one thread
consuming it, and a single child process running each merge, so a crash of
merge_videos kills the child, not the loop, and the merge memory is returned to
the system after every job.

The execution mode comes from VMSAM_MODE (anything but `production` is test):

* test       -- nothing is written outside VMSAM_TEST_OUTPUT_DIR, on disk or in
                the database.
* production -- the movie file is replaced, the movie_incompatible_files row
                deleted and the rejected file removed. On failure, nothing
                changes.

The result is named like the daemon's (and the show fusion's): the movie file's
path when it weighs at least as much as the rejected file, else the name given
by the rejected file's regex (rename pattern + suffix, or its own name). A merge
produces Matroska, so that name gets the .mkv extension.
"""

import os
import queue
import re
import shutil
import threading
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import Pool, get_context
from sys import stderr

import tools

from .model import setup_database, get_movie_data, get_regex_by_movie_id, get_incompatible_file_by_path, delete_incompatible_file, build_file_name
from gestionar_common.database import wait_for_database, is_database_unavailable

status_idle = "idle"
status_processing = "processing"

fusion_queue = queue.Queue()
state_lock = threading.Lock()
pending_jobs = deque()
current_job = None
worker_thread = None
parrallel_jobs = None
database_url = None
shutdown_sentinel = object()
# Returned by run_fusion_job when the database was lost BEFORE the merge
# started: nothing was written, the job can be replayed as is. A string, not an
# object(): the value crosses the child process pickle.
database_unavailable = "database_unavailable"
# Replays of one job when the database answers SELECT 1 but the job still
# loses it; beyond that the problem is not an outage and the job is dropped.
max_database_retries = 10


def get_test_output_dir():
    """Root of the test outputs. Empty means not configured, the internal API refuses."""
    return os.environ.get("VMSAM_TEST_OUTPUT_DIR", "").strip()


def is_test_mode():
    """Test by default: only the exact word `production` opens the destructive branch."""
    return tools.get_execution_mode() != tools.mode_production


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


''' Queue '''
def enqueue_fusion_job(error_file_path):
    """Add a job. Returns its position among the waiting ones."""
    with state_lock:
        if current_job != None and current_job["error_file_path"] == error_file_path:
            raise ValueError(f"{error_file_path} is already being processed")
        if error_file_path in pending_jobs:
            raise ValueError(f"{error_file_path} is already queued")
        pending_jobs.append(error_file_path)
        position = len(pending_jobs)
    fusion_queue.put(error_file_path)
    return position


def is_job_running():
    with state_lock:
        return current_job != None


def get_fusion_status():
    with state_lock:
        active_job = None if current_job == None else dict(current_job)
        waiting = list(pending_jobs)
    return {
        "status": status_idle if active_job == None else status_processing,
        "current_job": active_job,
        "queue_length": len(waiting),
        "pending_jobs": [{"error_file_path": path} for path in waiting]
    }


def start_worker(new_database_url):
    """Build the one-process pool and start the consumer thread (one merge at a time)."""
    global database_url, parrallel_jobs, worker_thread
    database_url = new_database_url
    if parrallel_jobs == None:
        parrallel_jobs = ProcessPoolExecutor(max_workers=1, mp_context=get_context("fork"))
    if worker_thread == None or (not worker_thread.is_alive()):
        worker_thread = threading.Thread(target=worker_loop, name="movie_fusion_worker", daemon=True)
        worker_thread.start()
    return worker_thread


def stop_worker():
    fusion_queue.put(shutdown_sentinel)


def worker_loop():
    """Consume the queue, one job at a time, with no time limit."""
    global current_job, parrallel_jobs
    while True:
        job = fusion_queue.get()
        if job is shutdown_sentinel:
            fusion_queue.task_done()
            # shutdown() makes the pool unusable for good: reset it so that
            # start_worker can build a new one.
            parrallel_jobs.shutdown()
            parrallel_jobs = None
            return

        with state_lock:
            tools.remove_element_without_bug(pending_jobs, job)
            current_job = {"error_file_path": job, "started_at": now_iso()}

        try:
            # A database outage would empty the queue, every job failing on its
            # first read: wait for the database before running the job.
            wait_for_database(database_url)
            result = parrallel_jobs.submit(run_fusion_job, database_url, job).result()
            database_retries = 0
            while result == database_unavailable and database_retries < max_database_retries:
                database_retries += 1
                stderr.write(f"Database lost before the merge of {job}, waiting to replay it ({database_retries}/{max_database_retries})\n")
                wait_for_database(database_url)
                result = parrallel_jobs.submit(run_fusion_job, database_url, job).result()
            if result == database_unavailable:
                stderr.write(f"Movie fusion job dropped for {job}: the database answers but the job kept losing it, resubmit it with POST /fusion\n")
        except Exception as e:
            stderr.write(f"Movie fusion job failed for {job}: {e}\n")
        finally:
            with state_lock:
                current_job = None
            fusion_queue.task_done()


''' Fusion, in the child process '''
def resolve_regex(movie_id, incompatible_file, session):
    """The regex that caught the rejected file, or None.

    Same as the show's resolve_rename_pattern: movie_incompatible_files does not
    keep the regex, so it is found by the weight, which comes from it; when
    several regexes of the movie share that weight, the base name, kept by
    process_rejected_files, is matched again.
    """
    candidates = [regex for regex in get_regex_by_movie_id(movie_id, session) if regex.weight == incompatible_file.file_weight]
    if len(candidates) == 1:
        return candidates[0]
    file_name = os.path.basename(incompatible_file.file_path)
    for regex in candidates:
        try:
            if re.search(regex.regex_pattern, file_name) != None:
                return regex
        except re.error:
            # A stored pattern that does not compile matches nothing.
            pass
    return None


def write_log_file(log_file_path, content):
    """Write a log without ever failing the job for it."""
    if not tools.make_dirs(os.path.dirname(log_file_path)):
        stderr.write(f"Cannot create the folder holding {log_file_path}\n")
        return
    try:
        with open(log_file_path, "w") as log:
            log.write(content)
    except OSError as e:
        stderr.write(f"Cannot write {log_file_path}: {e}\n")


def run_fusion_job(database_url, error_file_path):
    """Merge a rejected file with its movie's file. Runs in the child process."""
    import mergeVideo
    import video

    tools.dev = tools.get_dev_env_var()
    test_mode = is_test_mode()
    session = setup_database(database_url)
    lock_handle = None
    merge_started = False
    # Only THIS job's temp folder is removed on exit. Until it is chosen,
    # tools.tmpFolder still holds the inherited root (the daemon's --tmp, /tmp
    # by default): removing it on an early failure would wipe the whole tree.
    job_tmp_folder = None
    try:
        # First read: only to know WHICH lock to take.
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} is not registered in movie_incompatible_files")

        # The movie daemon may write the same movie, and the lock holds in BOTH
        # modes: even in test the fusion READS the movie file the daemon may be
        # replacing. `movie_<id>` can never be a show folder id (an integer).
        lock_handle = tools.acquire_episode_lock("movie_"+str(incompatible_file.movie_id), 0, blocking=True)
        # The wait may have lasted a whole integration: without expire_all() the
        # session would serve the pre-wait objects of its identity map.
        session.expire_all()
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} no longer registered in movie_incompatible_files after waiting for the lock")

        # Everything below is read AFTER the lock, so up to date.
        current_movie = get_movie_data(incompatible_file.movie_id, session)
        if current_movie == None:
            raise Exception(f"Movie {incompatible_file.movie_id} not found")
        if current_movie.file_path == None:
            raise Exception(f"Movie {current_movie.id} has no registered file, nothing to merge with")
        if not os.path.isfile(incompatible_file.file_path):
            raise Exception(f"{incompatible_file.file_path} is missing on disk")
        if not os.path.isfile(current_movie.file_path):
            raise Exception(f"{current_movie.file_path} is missing on disk")

        # Same rule as the daemon: the heavier file is the reference, the movie
        # file wins a tie and then keeps its path; otherwise the result takes
        # the name given by the rejected file's regex. No regex found: the
        # movie file's path is kept rather than inventing a name.
        if current_movie.file_weight >= incompatible_file.file_weight:
            tools.special_params["forced_best_video"] = current_movie.file_path
            new_file_path = current_movie.file_path
            new_file_weight = current_movie.file_weight
        else:
            tools.special_params["forced_best_video"] = incompatible_file.file_path
            matched_regex = resolve_regex(current_movie.id, incompatible_file, session)
            if matched_regex == None:
                new_file_path = current_movie.file_path
            else:
                new_file_path = os.path.join(current_movie.destination_path,
                                             build_file_name(matched_regex.rename_pattern, current_movie.suffix, os.path.basename(incompatible_file.file_path)))
            new_file_weight = incompatible_file.file_weight
        # A merge always produces a .mkv. When the chosen name has another
        # extension, the final name is a different file that must not exist.
        new_file_path = os.path.splitext(new_file_path)[0]+'.mkv'
        if os.path.abspath(new_file_path) != os.path.abspath(current_movie.file_path) and os.path.lexists(new_file_path):
            raise Exception(f"{new_file_path} already exists and is not the movie's file")

        video.number_cut = current_movie.number_cut
        mergeVideo.cut_file_to_get_delay_second_method = current_movie.cut_file_to_get_delay_second_method
        tools.default_language_for_undetermine = current_movie.original_language
        tools.special_params["original_language"] = current_movie.original_language

        # Temp tree apart from the daemon's (tmpFolder_original/movie_<id>),
        # which the daemon wipes when it finishes the movie.
        job_tmp_folder = os.path.join(tools.tmpFolder_original, "fusion_movie_"+str(current_movie.id))
        tools.tmpFolder = job_tmp_folder
        out_folder = os.path.join(tools.tmpFolder, "final_file")

        tools.remove_dir(tools.tmpFolder, printError=False)
        tools.make_dirs(tools.tmpFolder)
        tools.make_dirs(out_folder)

        mergeVideo.default_audio = True
        mergeVideo.errors_merge = []
        tools.logs = []
        if not tools.dev:
            mergeVideo.show_not_compatible_error = False

        # From here the job may write (files, then database): losing the
        # database can no longer be replayed without risking a double merge.
        merge_started = True
        video.ffmpeg_pool_audio_convert = Pool(processes=max(1, int(tools.core_to_use/1.6)))
        video.ffmpeg_pool_big_job = Pool(processes=1)
        merged_file_path = None
        job_error = None
        try:
            mergeVideo.merge_videos([incompatible_file.file_path, current_movie.file_path], out_folder, True)
            # merge_videos names the result `<best base name>_merged.mkv`, with
            # `_<n>` added to a base name both inputs share: take the only one.
            merged_file_path = os.path.join(out_folder, [file_name for file_name in os.listdir(out_folder) if file_name.endswith('_merged.mkv')][0])
        except Exception as e:
            job_error = {"error": e, "traceback": traceback.format_exc()}
        finally:
            try:
                video.ffmpeg_pool_audio_convert.close()
                video.ffmpeg_pool_big_job.close()
                video.ffmpeg_pool_audio_convert.terminate()
                video.ffmpeg_pool_big_job.terminate()
            except Exception as e:
                stderr.write(f"Error close pool: {e}\n")

        if test_mode:
            apply_test_outcome(current_movie, incompatible_file, new_file_path,
                               merged_file_path, job_error, mergeVideo.errors_merge)
        elif job_error == None:
            apply_production_outcome(session, incompatible_file, current_movie,
                                     new_file_path, new_file_weight, merged_file_path)
        elif tools.dev:
            # In production a failure touches neither files nor database: the
            # rejected file stays where it is, which IS the information. The
            # trace is written in dev mode only.
            stderr.write(f"Movie fusion failed for {incompatible_file.file_path}: {job_error['error']}\n")
            write_log_file(incompatible_file.file_path+".log.error",
                           f"Error processing file {os.path.basename(incompatible_file.file_path)}: "
                           f"{job_error['error']}\n{job_error['traceback']}\n\n"
                           f"Merged errors:\n{chr(10).join(mergeVideo.errors_merge)}\n\n"
                           f"Logs:\n{chr(10).join(tools.logs)}\n")
    except Exception as e:
        if (not merge_started) and is_database_unavailable(e):
            # Only reads happened: the worker will wait for the database and
            # replay the job instead of losing it.
            stderr.write(f"Database unavailable before the merge of {error_file_path}: {str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__}\n")
            return database_unavailable
        stderr.write(f"Error with the movie merge: {e}\n")
    finally:
        tools.release_episode_lock(lock_handle)
        if job_tmp_folder != None:
            tools.remove_dir(job_tmp_folder, printError=False)
        session.close()


def apply_test_outcome(current_movie, incompatible_file, new_file_path,
                       merged_file_path, job_error, merged_errors):
    """Test mode: nothing exists outside VMSAM_TEST_OUTPUT_DIR.

    Success drops the .mkv (final name) and its .log under
    <test output>/<destination_path>/; failure drops only the .log.error.
    """
    out_folder_final = os.path.join(get_test_output_dir(), current_movie.destination_path.lstrip(os.sep))
    if not tools.make_dirs(out_folder_final):
        raise Exception(f"Cannot create the test output folder {out_folder_final}")

    if job_error == None and merged_file_path != None:
        published = os.path.join(out_folder_final, os.path.basename(new_file_path))
        shutil.move(merged_file_path, published)
        write_log_file(published+".log",
                       f"Merged {incompatible_file.file_path} with the movie file into {published}\n\n"
                       f"Merged errors:\n{chr(10).join(merged_errors)}\n\n"
                       f"Logs:\n{chr(10).join(tools.logs)}\n")
    else:
        write_log_file(os.path.join(out_folder_final, os.path.basename(incompatible_file.file_path))+".log.error",
                       f"Error processing file {os.path.basename(incompatible_file.file_path)}: "
                       f"{job_error['error']}\n{job_error['traceback']}\n\n"
                       f"Merged errors:\n{chr(10).join(merged_errors)}\n\n"
                       f"Logs:\n{chr(10).join(tools.logs)}\n")


def apply_production_outcome(session, incompatible_file, current_movie,
                             new_file_path, new_file_weight, merged_file_path):
    """Production, success only: publish the result and settle the error.

    Written beside the target then swapped with os.replace (atomic on one file
    system), so the movie never lacks its file. The previous file is removed
    when its name differs (regex name or other extension), then the rejected
    file and its row.
    """
    shutil.move(merged_file_path, new_file_path+'.tmp')
    os.replace(new_file_path+'.tmp', new_file_path)
    if os.path.abspath(current_movie.file_path) != os.path.abspath(new_file_path):
        os.remove(current_movie.file_path)
    os.remove(incompatible_file.file_path)
    current_movie.file_path = new_file_path
    current_movie.file_weight = new_file_weight
    # delete_incompatible_file commits: the movie update and the row deletion
    # leave in the same transaction.
    delete_incompatible_file(incompatible_file, session)
