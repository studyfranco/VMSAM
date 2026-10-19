"""Entry point of the gestionar daemon: shows and movies.

Builds the runtime state of `tools`, creates the show and movie tables, starts
the four uvicorn instances (show public 8080 / internal 42085, movie public
8081 / internal 42086) and runs the watch loop.
Each round: wait for the database, then the movies, the show incrementaller and
the show integrator. The movies run first: a movie regex is refused by the
movie API when a show regex would take the same example as an episode.
"""

import argparse
import os
import traceback
from datetime import datetime
from multiprocessing import Process, set_start_method
from sys import stderr
from time import sleep, time

import tools
import main_gestionar_movie
import main_gestionar_show
from gestionar_common.database import wait_for_database
from gestionar_show.model import setup_database as setup_show_database
from gestionar_movie.model import setup_database as setup_movie_database


def run_round(folder, database_url):
    """One round of the loop. Returns the time the round's work started.

    A network cut or a PostgreSQL restart would make every step fail file by
    file: the round waits for the database first. The wait does not count in
    the delay between two rounds, which starts again from here.
    """
    begin = time()
    if wait_for_database(database_url):
        begin = time()
    main_gestionar_movie.process_files_in_folder(folder, database_url)
    main_gestionar_show.incrementaller(folder, database_url)
    main_gestionar_show.process_files_in_folder(folder, database_url)
    return begin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is the wrapper to process mkv,mp4 file to generate best file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--folder", metavar='folder', type=str,
                       required=True, help="Folder to watch for new files to process")
    parser.add_argument("-e","--error", metavar='error', type=str, default=".", help="Folder where send files with errors")
    parser.add_argument("--tmp", metavar='tmpdir', type=str,
                        default="/tmp", help="Folder where send temporar files")
    parser.add_argument("-c","--core", metavar='core', type=int, default=1, help="number of core the merge software can use")
    parser.add_argument("-w","--wait", metavar='wait', type=int, default=600, help="Time in second between folder check")
    parser.add_argument("--config", metavar='configFile', type=str,
                        default="config.ini", help="Path to the config file, by default use the config in the software folder. This config is for configure the path to your softwares")
    parser.add_argument("--database_url_file", metavar='database_url_file', type=str,
                        help="Database URL to connect to the gestionar_show database in a file.")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the software, put it if you use the folder from another folder")
    parser.add_argument("--dev", dest='dev', default=False, action='store_true', help="Print more errors and write all logs")
    args = parser.parse_args()
    try:
        set_start_method('fork', force=True)
        if (not os.access(args.error, os.W_OK)):
            raise Exception(f"{args.error} not writable")
        if (not os.access(args.folder, os.W_OK)):
            raise Exception(f"{args.folder} not writable")
        if (not os.access(args.tmp, os.W_OK)):
            raise Exception(f"{args.tmp} not writable")
        os.chdir(args.pwd)
        tools.dev = args.dev
        tools.config_file = args.config
        tools.software = tools.config_loader(args.config, "software")
        tools.core_to_use = max(1, args.core-1)
        tools.folder_error = args.error
        tools.tmpFolder = args.tmp
        tools.tmpFolder_original = os.path.join(args.tmp,"gestionar_show_"+str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
        tools.mergeRules = tools.config_loader(args.config,"mergerules")
        import json
        with open("titles_subs_group.json") as titles_subs_group_file:
            tools.group_title_sub = json.load(titles_subs_group_file)

        with open("config.json") as config_file:
            config = json.load(config_file)
        tools.language_to_keep = config["language_to_keep"]
        tools.language_to_completely_remove = set(config["language_to_completely_remove"])
        tools.language_to_try_to_keep = config["language_to_try_to_keep"]

        tools.special_params = {"change_all_und":True, "remove_commentary":True, "remove_descriptive":True, "forced_best_video_contain":False}
        with open(args.database_url_file) as database_url_file:
            database_url_param = json.load(database_url_file)
        database_url = database_url_param["database_url"]
        stderr.write("Setup database\n")
        with setup_show_database(database_url, create_tables=True) as session:
            pass
        with setup_movie_database(database_url, create_tables=True) as session:
            pass

        uvicorn_processes = [
            Process(target=main_gestionar_show.run_uvicorn_internal, args=(database_url, tools.tmpFolder)),
            Process(target=main_gestionar_show.run_uvicorn_public, args=(database_url, tools.tmpFolder)),
            Process(target=main_gestionar_movie.run_uvicorn_internal, args=(database_url, tools.tmpFolder)),
            Process(target=main_gestionar_movie.run_uvicorn_public, args=(database_url, tools.tmpFolder))
        ]
        for uvicorn_process in uvicorn_processes:
            uvicorn_process.start()

        stderr.write("Start !\n")
        while True:
            begin = time()
            try:
                begin = run_round(args.folder, database_url)
            except Exception as e:
                stderr.write(f"Error: {e}\n")
            import gc
            gc.collect()
            stderr.write("\n\n\n\nPOSO !!!!!!!\n\n\n\n\n")
            waiting_time = int(args.wait-(time()-begin))
            if waiting_time > 0:
                sleep(waiting_time)

        for uvicorn_process in uvicorn_processes:
            uvicorn_process.terminate()
            uvicorn_process.join()
    except:
        traceback.print_exc()
        exit(1)
    exit(0)
