"""Merge retry engine for files rejected by the pipeline.

An in-memory queue, one consumer thread, and a single child process that runs
the merge, so a crash in merge_videos kills the child, not the loop, and the
merge's memory is returned to the system after every job.

The execution mode comes from VMSAM_MODE (anything but `production` is test):

* test       -- nothing is written outside VMSAM_TEST_OUTPUT_DIR, on disk or
                in the database.
* production -- the master file is replaced, the incompatible_files row and
                the error file are deleted. On failure, nothing changes.
"""

import os
import queue
import re
import shutil
import threading
import traceback
import signal
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from multiprocessing import Pool, get_context
from sys import stderr

import tools

from .api import episode_pattern_insert
from .model import setup_database, get_folder_data, get_episode_data, get_regex_by_folder_id, get_incompatible_file_by_path, delete_incompatible_file

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


def get_test_output_dir():
    """Root of test outputs; empty means unconfigured and the internal API refuses to start."""
    return os.environ.get("VMSAM_TEST_OUTPUT_DIR", "").strip()


def is_test_mode():
    """Return True unless the mode is exactly `production`.

    Tested against mode_production so an empty or misspelled value stays in test mode.
    """
    return tools.get_execution_mode() != tools.mode_production


def now_iso():
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


''' Gestion de la file '''
def enqueue_fusion_job(error_file_path):
    """Queue a job and return its position among pending jobs."""
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
    """Return the worker status, current job and pending jobs."""
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
    """Create the single-process pool and start the consumer thread.

    The pool is built here rather than at import so it never forks from a
    half-initialised interpreter; max_workers=1 runs one merge at a time.
    """
    global database_url, parrallel_jobs, worker_thread
    database_url = new_database_url
    if parrallel_jobs == None:
        parrallel_jobs = ProcessPoolExecutor(max_workers=1, mp_context=get_context("fork"))
    if worker_thread == None or (not worker_thread.is_alive()):
        worker_thread = threading.Thread(target=worker_loop, name="fusion_worker", daemon=True)
        worker_thread.start()
    return worker_thread


def stop_worker():
    """Ask the worker thread to stop after the jobs already queued."""
    fusion_queue.put(shutdown_sentinel)


def worker_loop():
    """Consume the queue one job at a time, with no time limit.

    `get()` blocks until a job arrives; shutdown is signalled by a sentinel.
    """
    global current_job, parrallel_jobs
    while True:
        job = fusion_queue.get()
        if job is shutdown_sentinel:
            fusion_queue.task_done()
            # A shut-down pool is unusable; reset to None so start_worker can recreate it.
            parrallel_jobs.shutdown()
            parrallel_jobs = None
            return

        with state_lock:
            tools.remove_element_without_bug(pending_jobs, job)
            current_job = {"error_file_path": job, "started_at": now_iso()}

        try:
            # Wait without bound: a merge must never be interrupted here.
            parrallel_jobs.submit(run_fusion_job, database_url, job).result()
        except BrokenProcessPool as e:
            # A dead child (OOM, segfault, kill -9) breaks the pool permanently, so it is
            # rebuilt; the failing job is not replayed, since it would crash the child again.
            stderr.write(f"Fusion job failed for {job}: {e}\n")
            parrallel_jobs.shutdown(wait=False)
            parrallel_jobs = ProcessPoolExecutor(max_workers=1, mp_context=get_context("fork"))
        except Exception as e:
            stderr.write(f"Fusion job failed for {job}: {e}\n")
        finally:
            with state_lock:
                current_job = None
            fusion_queue.task_done()


''' Exécution de la fusion, dans le process fils '''
def resolve_rename_pattern(folder_id, incompatible_file, session):
    """Return the rename_pattern of the error file's regex, or None.

    The regex is found by its weight; ties are broken by re-matching the file's
    base name.
    """
    candidates = [regex for regex in get_regex_by_folder_id(folder_id, session)
                  if regex.weight == incompatible_file.file_weight and regex.rename_pattern]
    if len(candidates) == 1:
        return candidates[0].rename_pattern
    if len(candidates) > 1:
        file_name = os.path.basename(incompatible_file.file_path)
        for regex in candidates:
            if re.search(regex.regex_pattern, file_name) != None:
                return regex.rename_pattern
    return None


def write_log_file(log_file_path, content, merge_plan=None, plan_anchor=None,
                   candidate_path=None):
    """Write a log file without ever failing the job.

    When `merge_plan` is not None, also write the merge plan report next to the
    file, under the name `merge_plan_report.write_report` returns; None means no
    resample/chimeric version was produced. Report errors are caught broadly so
    a report failure never fails a successful merge; the outcome is written to
    the log either way.
    """
    if not tools.make_dirs(os.path.dirname(log_file_path)):
        stderr.write(f"Cannot create the folder holding {log_file_path}\n")
        return
    if merge_plan != None:
        try:
            import merge_plan_report
            destination, transport_entry = merge_plan_report.write_report(
                merge_plan_report.parse_job_log(merge_plan),
                merge_plan_report.opaque_id(candidate_path),
                # The log's base name; the report module redacts it.
                os.path.basename(log_file_path),
                plan_anchor)
                # No `caveats`/`corpus`: a production render covers a single merge, so
                # the report correctly states that no corpus was supplied.
            # Log a pointer to the report, not its full transport copy.
            content += (f"\nMerge plan report: {os.path.basename(destination)} "
                        f"({len(transport_entry.encode('utf-8'))} bytes transportable)\n")
        except Exception as e:
            # Match the class by name: if the import itself failed, the module is absent.
            # A contract refusal (merge_plan_error) is reported apart from a render failure.
            kind = ("REFUSED (job contract)"
                    if type(e).__name__ == "merge_plan_error" else "NOT produced")
            stderr.write(f"Merge plan report {kind}: {e}\n")
            content += f"\nMerge plan report {kind}: {e}\n"
    try:
        with open(log_file_path, "w") as log:
            log.write(content)
    except OSError as e:
        stderr.write(f"Cannot write {log_file_path}: {e}\n")


def run_fusion_job(database_url, error_file_path):
    """Retry the merge of one error file; runs in the child process."""
    import mergeVideo
    import video

    tools.dev = tools.get_dev_env_var()
    test_mode = is_test_mode()
    session = setup_database(database_url)
    lock_handle = None
    try:
        # First read only identifies which lock to take.
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} is not registered in incompatible_files")

        folder_id = incompatible_file.folder_id
        episode_number = incompatible_file.episode_number

        # Locked in both modes: even in test mode the merge reads the master, which the
        # integration loop may replace concurrently.
        lock_handle = tools.acquire_episode_lock(folder_id, episode_number, blocking=True)
        # The rows may have changed while waiting; expire_all() forces a fresh re-read.
        session.expire_all()
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} no longer registered in incompatible_files after waiting for the lock")
        folder_id = incompatible_file.folder_id
        episode_number = incompatible_file.episode_number

        current_folder = get_folder_data(folder_id, session)
        if current_folder == None:
            raise Exception(f"Folder {folder_id} not found")

        previous_file = get_episode_data(folder_id, episode_number, session)
        if previous_file == None:
            raise Exception(f"No episode {episode_number} for folder {folder_id}, nothing to merge with")

        if not os.path.isfile(incompatible_file.file_path):
            raise Exception(f"{incompatible_file.file_path} is missing on disk")
        if not os.path.isfile(previous_file.file_path):
            raise Exception(f"{previous_file.file_path} is missing on disk")

        video.number_cut = current_folder.number_cut
        mergeVideo.cut_file_to_get_delay_second_method = current_folder.cut_file_to_get_delay_second_method
        tools.default_language_for_undetermine = current_folder.original_language
        tools.special_params["original_language"] = current_folder.original_language

        # Separate temp tree: the integration loop wipes tmpFolder_original/<folder_id>.
        tools.tmpFolder = os.path.join(tools.tmpFolder_original, "fusion_"+str(folder_id)+"_"+str(episode_number))
        out_folder = os.path.join(tools.tmpFolder, "final_file")

        # Same rule as process_episode: the weight decides the reference and the output
        # name, with the error file acting as the new file.
        if previous_file.file_weight >= incompatible_file.file_weight:
            tools.special_params["forced_best_video"] = previous_file.file_path
            new_file_path = previous_file.file_path
            new_file_weight = previous_file.file_weight
            merged_source = previous_file.file_path
        else:
            tools.special_params["forced_best_video"] = incompatible_file.file_path
            rename_pattern = resolve_rename_pattern(folder_id, incompatible_file, session)
            if rename_pattern == None:
                # No matching regex: keep the master's path rather than invent a name.
                new_file_path = previous_file.file_path
            else:
                new_file_path = os.path.join(current_folder.destination_path,
                                             rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}"))
            new_file_weight = incompatible_file.file_weight
            merged_source = incompatible_file.file_path

        tools.remove_dir(tools.tmpFolder, printError=False)
        tools.make_dirs(tools.tmpFolder)
        tools.make_dirs(out_folder)

        mergeVideo.default_audio = True
        mergeVideo.errors_merge = []
        # None (not []) means no resample/chimeric version was produced.
        mergeVideo.merge_plan = None
        tools.logs = []
        if not tools.dev:
            mergeVideo.show_not_compatible_error = False

        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        video.ffmpeg_pool_audio_convert = Pool(processes=max(1, int(tools.core_to_use/1.6)))
        video.ffmpeg_pool_big_job = Pool(processes=1)
        merged_file_path = None
        job_error = None
        try:
            mergeVideo.merge_videos([incompatible_file.file_path, previous_file.file_path], out_folder, True)
            merged_file_path = os.path.join(out_folder, os.path.splitext(os.path.basename(merged_source))[0]+'_merged.mkv')
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
            apply_test_outcome(current_folder, incompatible_file, new_file_path,
                               merged_file_path, job_error, mergeVideo.errors_merge,
                               mergeVideo.merge_plan)
        elif job_error == None:
            apply_production_outcome(session, incompatible_file, previous_file,
                                     new_file_path, new_file_weight, merged_file_path)
        elif tools.dev:
            # A production failure changes nothing; in dev mode the error is logged,
            # since job_error is caught above and would otherwise go unreported.
            stderr.write(f"Fusion failed for {incompatible_file.file_path}: {job_error['error']}\n")
            write_log_file(incompatible_file.file_path+".log.error",
                           f"Error processing file {os.path.basename(incompatible_file.file_path)}: "
                           f"{job_error['error']}\n{job_error['traceback']}\n\n"
                           f"Merged errors:\n{chr(10).join(mergeVideo.errors_merge)}\n\n"
                           f"Logs:\n{chr(10).join(tools.logs)}\n")
    except Exception as e:
        stderr.write(f"Error with the merge: {e}\n")
    finally:
        tools.release_episode_lock(lock_handle)
        tools.remove_dir(tools.tmpFolder, printError=False)
        session.close()


def apply_test_outcome(current_folder, incompatible_file, new_file_path,
                       merged_file_path, job_error, merged_errors, merge_plan=None):
    """Test mode: write only under VMSAM_TEST_OUTPUT_DIR.

    Sources and database are untouched. Success writes the mkv and its .log;
    failure writes only a .log.error.
    """
    # internal_api refuses to start the worker when VMSAM_TEST_OUTPUT_DIR is empty.
    out_folder_final = os.path.join(get_test_output_dir(), current_folder.destination_path.lstrip(os.sep))
    if not tools.make_dirs(out_folder_final):
        raise Exception(f"Cannot create the test output folder {out_folder_final}")

    if job_error == None and merged_file_path != None:
        published = os.path.join(out_folder_final, os.path.basename(new_file_path))
        shutil.move(merged_file_path, published)
        write_log_file(published+".log",
                       f"Merged {incompatible_file.file_path} with the master into {published}\n\n"
                       f"Merged errors:\n{chr(10).join(merged_errors)}\n\n"
                       f"Logs:\n{chr(10).join(tools.logs)}\n",
                       merge_plan, published, incompatible_file.file_path)
    else:
        # A plan may exist without an output file; anchor it on the candidate's name.
        failed_anchor = os.path.join(out_folder_final, os.path.basename(incompatible_file.file_path))
        write_log_file(failed_anchor+".log.error",
                       f"Error processing file {os.path.basename(incompatible_file.file_path)}: "
                       f"{job_error['error']}\n{job_error['traceback']}\n\n"
                       f"Merged errors:\n{chr(10).join(merged_errors)}\n\n"
                       f"Logs:\n{chr(10).join(tools.logs)}\n",
                       merge_plan, failed_anchor, incompatible_file.file_path)


def apply_production_outcome(session, incompatible_file, previous_file,
                             new_file_path, new_file_weight, merged_file_path):
    """Production mode, success only: publish the merge and clear the error.

    Writes under .tmp, swaps it in, removes the old master if renamed, then
    deletes the error file and its row.
    """
    # os.replace is atomic on one filesystem, so the episode always has a file.
    shutil.move(merged_file_path, new_file_path+'.tmp')
    os.replace(new_file_path+'.tmp', new_file_path)
    # Remove the old master only when the name changed; otherwise it was just replaced.
    if os.path.abspath(previous_file.file_path) != os.path.abspath(new_file_path):
        os.remove(previous_file.file_path)
    os.remove(incompatible_file.file_path)
    previous_file.file_path = new_file_path
    previous_file.file_weight = new_file_weight
    # delete_incompatible_file commits the episode update and the row deletion together.
    delete_incompatible_file(incompatible_file, session)
    session.commit()
