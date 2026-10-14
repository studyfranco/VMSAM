"""Movie daemon: integrates the files of the watch folder caught by a movie regex.

Same flow as main_gestionar_show without an episode number: every file a
movie's regexes catch is merged into the movie's one kept file, in its
destination_path. As for an episode, the file name comes from the regex that
caught the file: its rename pattern with `{<suffix>}` replaced by the movie's
suffix, or the file's own name when the regex has none. Started by
main_gestionar.py, before the show steps of a round.
"""

import gc
import os
import re
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, get_context
from sys import stderr
from time import sleep

import uvicorn

import mergeVideo
import tools
import video
from gestionar_movie.model import setup_database, get_movie_data, get_all_regex, set_movie_file, get_incompatible_files_by_movie_id, insert_incompatible_file, build_file_name
from main_gestionar_show import write_api_env_file

# Seconds between two movie jobs handed to the pool, like the show's spacing
# between episodes: it staggers the start of the heavy extraction phases.
job_spacing = 2

def unique_file_path(file_path):
    """file_path if free, else `<stem>.<n><ext>` with the first free n."""
    stem, extension = os.path.splitext(file_path)
    number = 0
    while os.path.lexists(file_path):
        number += 1
        file_path = f"{stem}.{number}{extension}"
    return file_path

def get_merged_file_path(out_folder):
    """merge_videos names its result `<best base name>_merged.mkv`, with `_<n>`
    added to a base name both inputs share: take the only one of the folder."""
    return os.path.join(out_folder, [file_name for file_name in os.listdir(out_folder) if file_name.endswith('_merged.mkv')][0])

def process_rejected_files(file, current_movie, session, data_rejected):
    """Send a file that does not merge with the movie's file to the movie's error directory.

    Same contract as the show's: the file is first tried against the movie's
    incompatible files, heaviest first; on the first success the result replaces
    that incompatible file (same row) and the file is removed. Otherwise the file
    is moved to `<error root>/<destination_path>/<title>/` (one error directory
    per movie: several movies may share a destination) and registered.
    A `.log.error` beside the kept file is appended, so a merged incompatible
    file keeps the history of every file that went into it.
    """
    tools.tmpFolder = os.path.join(tools.tmpFolder_original, "movie_"+str(current_movie.id))
    out_folder = os.path.join(tools.tmpFolder, "final_file")

    out_folder_final = os.path.join(tools.folder_error, current_movie.destination_path.lstrip(os.sep), current_movie.title)
    if not tools.make_dirs(out_folder_final):
        raise Exception(f"Cannot create the error directory {out_folder_final}")

    kept_file_path = None
    other_rejected_files = []
    for incompatible_file in get_incompatible_files_by_movie_id(current_movie.id, session):
        if not os.path.isfile(incompatible_file.file_path):
            stderr.write(f"\t\tIncompatible file {incompatible_file.file_path} is missing on disk, not tried\n")
            continue
        if (incompatible_file.file_weight < file['weight'] or (incompatible_file.file_weight == file['weight']
                and os.path.getsize(file['chemin']) >= os.path.getsize(incompatible_file.file_path))):
            tools.special_params["forced_best_video"] = file['chemin']
            new_file_weight = file['weight']
        else:
            tools.special_params["forced_best_video"] = incompatible_file.file_path
            new_file_weight = incompatible_file.file_weight
        new_file_path = os.path.join(out_folder_final, os.path.splitext(os.path.basename(tools.special_params["forced_best_video"]))[0]+'.mkv')
        if os.path.abspath(new_file_path) != os.path.abspath(incompatible_file.file_path):
            new_file_path = unique_file_path(new_file_path)

        tools.remove_dir(tools.tmpFolder, printError=False)
        tools.make_dirs(out_folder)
        mergeVideo.errors_merge = []
        tools.logs = []
        try:
            if tools.dev:
                stderr.write(f"\t\tStart merging {file['nom']} and {incompatible_file.file_path} into {out_folder_final}\n")
            mergeVideo.merge_videos([file['chemin'], incompatible_file.file_path], out_folder, True)
            merged_file_path = get_merged_file_path(out_folder)
        except Exception as e:
            other_rejected_files.append({"error": e, "traceback": traceback.format_exc(), "merged_errors": mergeVideo.errors_merge, "merged_logs": tools.logs})
            continue
        if tools.dev:
            stderr.write(f"\t\tEnd merging {file['nom']} and {incompatible_file.file_path} into {out_folder_final}\n")

        shutil.move(merged_file_path, new_file_path+'.tmp')
        os.replace(new_file_path+'.tmp', new_file_path)
        if os.path.abspath(incompatible_file.file_path) != os.path.abspath(new_file_path):
            os.remove(incompatible_file.file_path)
        os.remove(file['chemin'])
        incompatible_file.file_path = new_file_path
        incompatible_file.file_weight = new_file_weight
        session.commit()
        kept_file_path = new_file_path
        break

    if kept_file_path == None:
        kept_file_path = unique_file_path(os.path.join(out_folder_final, os.path.basename(file['chemin'])))
        shutil.move(file['chemin'], kept_file_path)
        insert_incompatible_file(current_movie.id, kept_file_path, file['weight'], session)

    with open(kept_file_path+".log.error", "a") as log:
        log.write(f"Error processing file {file['nom']}: {data_rejected['error']}\n{data_rejected['traceback']}\n\nMerged errors: {"\n".join(data_rejected['merged_errors'])}\n\nLogs:\n{"\n".join(data_rejected['merged_logs'])}\n\nOther rejected files:")
        for rejected_file in other_rejected_files:
            log.write(f"\n\n{rejected_file['error']}\n{rejected_file['traceback']}\n\nMerged errors:\n{"\n".join(rejected_file['merged_errors'])}\n\nLogs:\n{"\n".join(rejected_file['merged_logs'])}\n\n")
        log.write("\n")

def process_movie(files, movie_id, database_url):
    """Integrate every file of one movie, heaviest first (regex weight, then size). Runs in a pool process.

    Same rules as the show's process_episode, a file's name being given by its
    regex (rename pattern + movie suffix, or the file's own name):
    - movie without a file: the file is moved to the name given by its regex
      and registered;
    - movie with a file: both are merged, the heavier as the forced best video
      (the registered file wins a tie). When the registered file weighs at
      least as much, the result keeps the registered file's path; otherwise it
      takes the name given by the new file's regex, extension included as the
      rename pattern writes it (the owner's choice), and a previous file under
      another name is removed. On failure the lighter file goes to the error
      directory; when it is the registered one, the new file is then moved
      in as if the movie had no file.
    A final name already taken by another file stops the movie for this round,
    its remaining files staying in the watch folder.
    """
    session = setup_database(database_url)

    # The fusion worker may hold this movie, in both modes: even in test it reads
    # the movie file replaced here. Like the show, skip the movie this round
    # rather than freezing a pool process for a whole fusion.
    lock_handle = tools.acquire_episode_lock("movie_"+str(movie_id), 0, blocking=False)
    if lock_handle == None:
        stderr.write(f"Movie {movie_id} is held by the fusion worker, skipped this round\n")
        session.close()
        return

    # Temp tree apart from the show's (tmpFolder_original/<folder_id>) and from
    # the movie fusion's (tmpFolder_original/fusion_movie_<id>).
    tools.tmpFolder = os.path.join(tools.tmpFolder_original, "movie_"+str(movie_id))
    out_folder = os.path.join(tools.tmpFolder, "final_file")
    try:
        current_movie = get_movie_data(movie_id, session)
        if current_movie == None:
            raise Exception(f"Movie {movie_id} not found")
        if tools.dev:
            stderr.write(f"\tStart integration of movie {movie_id} in {current_movie.destination_path}\n")

        video.number_cut = current_movie.number_cut
        mergeVideo.cut_file_to_get_delay_second_method = current_movie.cut_file_to_get_delay_second_method
        tools.default_language_for_undetermine = current_movie.original_language
        tools.special_params["original_language"] = current_movie.original_language

        for file in sorted(files, key=lambda candidate: (candidate['weight'], os.path.getsize(candidate['chemin'])), reverse=True):
            if current_movie.file_path != None:
                previous_file_path = current_movie.file_path
                previous_file_weight = current_movie.file_weight
                if not os.path.isfile(previous_file_path):
                    raise Exception(f"The registered file {previous_file_path} is missing on disk, {file['nom']} left in the watch folder")
                if previous_file_weight >= file['weight']:
                    tools.special_params["forced_best_video"] = previous_file_path
                    new_file_path = previous_file_path
                    new_file_weight = previous_file_weight
                else:
                    tools.special_params["forced_best_video"] = file['chemin']
                    new_file_path = os.path.join(current_movie.destination_path,
                                                                      build_file_name(file['rename_pattern'], current_movie.suffix, file['nom']))
                    new_file_weight = file['weight']
                if os.path.abspath(new_file_path) != os.path.abspath(previous_file_path) and os.path.lexists(new_file_path):
                    raise Exception(f"{new_file_path} already exists and is not the movie's file, {file['nom']} left in the watch folder")

                tools.remove_dir(tools.tmpFolder, printError=False)
                tools.make_dirs(out_folder)
                mergeVideo.default_audio = True
                mergeVideo.errors_merge = []
                tools.logs = []
                if not tools.dev:
                    mergeVideo.show_not_compatible_error = False

                video.ffmpeg_pool_audio_convert = Pool(processes=max(1, int(tools.core_to_use/1.6)))
                video.ffmpeg_pool_big_job = Pool(processes=1)
                merged_file_path = None
                try:
                    if tools.dev:
                        stderr.write(f"\t\tStart merging {file['nom']} and {previous_file_path} into {new_file_path}\n")
                    mergeVideo.merge_videos([file['chemin'], previous_file_path], out_folder, True)
                    merged_file_path = get_merged_file_path(out_folder)
                    if tools.dev:
                        stderr.write(f"\t\tEnd merging {file['nom']} and {previous_file_path} into {new_file_path}\n")
                except Exception as e:
                    stderr.write(f"Error merging: {file['nom']} and {previous_file_path} are incompatibles\n")
                    data_rejected = {"error": e, "traceback": traceback.format_exc(), "merged_errors": mergeVideo.errors_merge, "merged_logs": tools.logs}
                    if previous_file_weight >= file['weight']:
                        process_rejected_files(file, current_movie, session, data_rejected)
                    else:
                        previous_file = {'nom': os.path.basename(previous_file_path), 'chemin': previous_file_path, 'weight': previous_file_weight}
                        process_rejected_files(previous_file, current_movie, session, data_rejected)
                        # Cleared before the new file is moved below: if the move
                        # fails the movie has no file and the new file, still in
                        # the watch folder, is integrated afresh next round.
                        set_movie_file(current_movie, None, None, session)
                finally:
                    try:
                        video.ffmpeg_pool_audio_convert.close()
                        video.ffmpeg_pool_big_job.close()
                        video.ffmpeg_pool_audio_convert.terminate()
                        video.ffmpeg_pool_big_job.terminate()
                    except Exception as e:
                        stderr.write(f"Error close pool: {e}\n")

                if merged_file_path != None:
                    # Written beside the target then swapped in one call: os.replace
                    # is atomic on one file system, so the movie never lacks its file.
                    shutil.move(merged_file_path, new_file_path+'.tmp')
                    os.replace(new_file_path+'.tmp', new_file_path)
                    if os.path.abspath(previous_file_path) != os.path.abspath(new_file_path):
                        os.remove(previous_file_path)
                    os.remove(file['chemin'])
                    set_movie_file(current_movie, new_file_path, new_file_weight, session)
                tools.remove_dir(tools.tmpFolder, printError=False)

            if current_movie.file_path == None and os.path.isfile(file['chemin']):
                new_file_path = os.path.join(current_movie.destination_path, build_file_name(file['rename_pattern'], current_movie.suffix, file['nom']))
                if os.path.lexists(new_file_path):
                    raise Exception(f"{new_file_path} already exists and is not registered, {file['nom']} left in the watch folder")
                tools.make_dirs(current_movie.destination_path)
                shutil.move(file['chemin'], new_file_path)
                set_movie_file(current_movie, new_file_path, file['weight'], session)
                if tools.dev:
                    stderr.write(f"\t\t{file['nom']} integrated as {new_file_path}\n")
            gc.collect()
    except Exception as e:
        session.rollback()
        stderr.write(f"Error processing files for movie {movie_id}: {e}\n")
    finally:
        tools.remove_dir(tools.tmpFolder, printError=False)
        tools.release_episode_lock(lock_handle)
        session.close()

def process_files_in_folder(folder_files, database_url):
    stderr.write("Start movie integrator !\n")
    files = [
        {'nom': file_name, 'chemin': os.path.join(folder_files, file_name)}
        for file_name in os.listdir(folder_files)
        if os.path.isfile(os.path.join(folder_files, file_name))
    ]
    if not files:
        return

    # A file goes to the first regex that matches it, heaviest regex first.
    files_by_movie = {}
    with setup_database(database_url) as session:
        for regex in get_all_regex(session):
            if not files:
                break
            try:
                compiled_regex = re.compile(regex.regex_pattern)
            except re.error as e:
                stderr.write(f"Movie regex `{regex.regex_pattern}` does not compile, skipped: {e}\n")
                continue
            for matched_file in [file for file in files if compiled_regex.search(file['nom'])]:
                matched_file['regex'] = regex.regex_pattern
                matched_file['rename_pattern'] = regex.rename_pattern
                matched_file['weight'] = regex.weight
                files_by_movie.setdefault(regex.movie_id, []).append(matched_file)
                files.remove(matched_file)
    if not files_by_movie:
        return

    parrallel_jobs = ProcessPoolExecutor(max_workers=2, mp_context=get_context("fork"))
    list_jobs = []
    for movie_id, movie_files in files_by_movie.items():
        list_jobs.append(parrallel_jobs.submit(process_movie, movie_files, movie_id, database_url))
        sleep(job_spacing)
    for job in list_jobs:
        try:
            job.result()
        except Exception as e:
            stderr.write(f"Error processing movie files: {e}\n")
    parrallel_jobs.shutdown()

def run_uvicorn_public(database_url, tmpFolder):
    """Public movie instance: exposed on 0.0.0.0:8081, multi-worker, no queue state."""
    import sys
    sys.stdin = None

    env_path = os.path.join(tmpFolder, "gestionar_movie_api.env")
    write_api_env_file(env_path, database_url)
    stderr.write("Start public movie API on 0.0.0.0:8081\n")
    uvicorn.run("gestionar_movie.api:app", host="0.0.0.0", port=8081, env_file=env_path, workers=5, log_level="error")

def run_uvicorn_internal(database_url, tmpFolder):
    """Internal movie instance: 127.0.0.1:42086 only, one worker (it owns the
    fusion queue). The port must differ from the show's internal 42085."""
    import sys
    sys.stdin = None

    env_path = os.path.join(tmpFolder, "gestionar_movie_internal_api.env")
    write_api_env_file(env_path, database_url, with_merge_runtime=True)
    stderr.write("Start internal movie fusion worker on 127.0.0.1:42086\n")
    uvicorn.run("gestionar_movie.internal_api:app", host="127.0.0.1", port=42086, env_file=env_path, workers=1, log_level="error")
