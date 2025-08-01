import argparse
import os
from datetime import datetime
from multiprocessing import Pool, Process
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from sys import stderr,stdout
from time import sleep
import tools
from gestionar_show_model import setup_database, get_folder_data, get_all_regex, get_episode_data, get_regex_data, insert_episode, get_all_incrementaller
import re
import mergeVideo
import video
import shutil
import uvicorn
import traceback

episode_pattern_insert = "{<episode>}"
parrallel_jobs = None

def process_episode(files, folder_id, episode_number, database_url):
    dic_weight_files = {}
    for file in files:
        if file['weight'] not in dic_weight_files:
            dic_weight_files[file['weight']] = [file]
        else:
            dic_weight_files[file['weight']].append(file)
    """Process files for a specific folder and extract episodes"""
    session = setup_database(database_url)

    try:
        # Récupérer le dossier
        current_folder = get_folder_data(folder_id, session)
        
        stderr.write(f"\tStart integration episode {episode_number} of {current_folder.destination_path}\n")
        
        video.number_cut = current_folder.number_cut
        mergeVideo.cut_file_to_get_delay_second_method = current_folder.cut_file_to_get_delay_second_method
        tools.default_language_for_undetermine = current_folder.original_language
        tools.special_params["original_language"] = current_folder.original_language

        tools.tmpFolder = os.path.join(os.path.join(tools.tmpFolder_original, str(current_folder.id)), str(episode_number))
        out_folder = os.path.join(tools.tmpFolder, "final_file")
        
        # Traiter les fichiers
        for weight in reversed(sorted(dic_weight_files)):
            for file in dic_weight_files[weight]:
                previous_file = get_episode_data(folder_id, episode_number, session)
                if previous_file != None:
                    # Si l'épisode existe déjà, on le fusionne
                    if previous_file.file_weight < file['weight']:
                        regex_data = get_regex_data(file['regex'], session)
                        tools.special_params["forced_best_video"] = file['chemin']
                        new_file_path = os.path.join(current_folder.destination_path, regex_data.rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}"))
                        new_file_weight = file['weight']
                    elif previous_file.file_weight > file['weight']:
                        tools.special_params["forced_best_video"] = previous_file.file_path
                        new_file_path = previous_file.file_path
                        new_file_weight = previous_file.file_weight
                    else:
                        regex_data = get_regex_data(file['regex'], session)
                        tools.special_params["forced_best_video"] = file['chemin']
                        new_file_path = os.path.join(current_folder.destination_path, regex_data.rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}"))
                        new_file_weight = file['weight']
                    
                    tools.make_dirs(tools.tmpFolder)
                    tools.make_dirs(out_folder)

                    mergeVideo.default_audio = True
                    mergeVideo.errors_merge = []

                    video.ffmpeg_pool_audio_convert = Pool(processes=tools.core_to_use)
                    video.ffmpeg_pool_big_job = Pool(processes=1)
                    try:
                        mergeVideo.merge_videos([file['chemin'],previous_file.file_path],out_folder,True)
                        if previous_file.file_weight < file['weight']:
                            shutil.move(os.path.join(out_folder, os.path.splitext(os.path.basename(file['chemin']))[0]+'_merged.mkv'), new_file_path+'.tmp')
                        elif previous_file.file_weight > file['weight']:
                            shutil.move(os.path.join(out_folder, os.path.splitext(os.path.basename(previous_file.file_path))[0]+'_merged.mkv'), new_file_path+'.tmp')
                        else:
                            shutil.move(os.path.join(out_folder, os.path.splitext(os.path.basename(file['chemin']))[0]+'_merged.mkv'), new_file_path+'.tmp')
                        os.remove(previous_file.file_path)
                        shutil.move(new_file_path+'.tmp', new_file_path)
                        os.remove(file['chemin'])
                    except Exception as e:
                        stderr.write(f"Error processing file {file['nom']}: {e}\n")
                        if previous_file.file_weight < file['weight'] or previous_file.file_weight == file['weight']:
                            shutil.move(previous_file.file_path, os.path.join(tools.folder_error, os.path.basename(previous_file.file_path)))
                            shutil.move(file['chemin'], new_file_path)
                            with open(os.path.join(tools.folder_error, os.path.basename(previous_file.file_path))+".log.error","w") as log:
                                log.write(f"Error processing file {os.path.basename(previous_file.file_path)}: {e}\n{traceback.print_exc()}\n\nMerged errors: {mergeVideo.errors_merge}")
                        else:
                            shutil.move(file['chemin'], os.path.join(tools.folder_error, os.path.basename(file['chemin'])))
                            with open(os.path.join(tools.folder_error, os.path.basename(file['chemin']))+".log.error","w") as log:
                                log.write(f"Error processing file {file['nom']}: {e}\n{traceback.print_exc()}\n\nMerged errors: {mergeVideo.errors_merge}")
                    finally:
                        try:
                            video.ffmpeg_pool_audio_convert.close()
                            video.ffmpeg_pool_big_job.close()
                            video.ffmpeg_pool_audio_convert.terminate()
                            video.ffmpeg_pool_big_job.terminate()
                        except Exception as e:
                            stderr.write(f"Error close pool: {e}\n")
                        tools.remove_dir(tools.tmpFolder)
                    previous_file.file_path = new_file_path
                    previous_file.file_weight = new_file_weight
                    session.commit()
                else:
                    # Si l'épisode n'existe pas, on l'ajoute
                    regex_data = get_regex_data(file['regex'], session)
                    new_file_path = os.path.join(current_folder.destination_path, regex_data.rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}"))
                    shutil.move(file['chemin'], new_file_path)
                    insert_episode(folder_id, episode_number, new_file_path, file['weight'], session)
                import gc
                gc.collect()
    except Exception as e:
        stderr.write(f"Error processing files for folder {folder_id}, episode {episode_number}: {e}\n")
    session.close()

def extraire_episode(nom_fichier, regex_pattern):
    """Extrait le numéro d'épisode selon le pattern regex"""
    match = re.search(regex_pattern, nom_fichier)
    if match:
        if 'episode' in match.groupdict():
            return match.group('episode')
        elif match.groups():
            return match.group(1)
    return None

def process_file_by_folder(files, folder_id, database_url):
    group_files_by_episode = {}
    for file in files:
        episode_number = extraire_episode(file['nom'], file['regex'])
        if episode_number != None and episode_number.isdigit() and int(episode_number) > 0:
            episode_number = int(episode_number)
            if episode_number not in group_files_by_episode:
                group_files_by_episode[episode_number] = []
            group_files_by_episode[episode_number].append(file)
        else:
            stderr.write(f"Episode number not found for file {file['nom']}\n")
            
    if len(group_files_by_episode):
        with setup_database(database_url) as session:
            # Récupérer le dossier
            current_folder = get_folder_data(folder_id, session)
            tmp_folder_group = os.path.join(tools.tmpFolder_original, str(current_folder.id))
        tools.make_dirs(tmp_folder_group)

        global parrallel_jobs
        list_jobs = []
        for episode_number, files in group_files_by_episode.items():
            if episode_number <= current_folder.max_episode_number:
                # Lancer le traitement des fichiers en parallèle
                list_jobs.append(parrallel_jobs.submit(
                    process_episode,files, folder_id, episode_number, database_url
                ))
        group_files_by_episode = None  # Libérer la mémoire
        current_folder = None  # Libérer la mémoire
        
        for job in list_jobs:
            try:
                job.result()
            except Exception as e:
                stderr.write(f"Error processing files: {e}\n")
        tools.remove_dir(tmp_folder_group)
    return

def process_files_in_folder(folder_files,database_url):
    stderr.write("Start integrator !\n")
    fichiers = [
            {'nom': fichier, 'chemin': os.path.join(folder_files, fichier)}
            for fichier in os.listdir(folder_files)
            if os.path.isfile(os.path.join(folder_files, fichier))
        ]
        
    if not fichiers:
        return
    
    with setup_database(database_url) as session:
        # Récupérer toutes les regex triées par poids décroissant
        all_regex = get_all_regex(session)
        
        # Traiter chaque regex et supprimer les fichiers matchés directement
        resultats_finaux = {}
        
        for regex in all_regex:
            if not fichiers:  # Plus de fichiers à traiter
                break
                
            # Compiler la regex
            regex_compilee = re.compile(regex.regex_pattern)
            
            # Filtrer les fichiers qui matchent
            fichiers_matches = list(filter(
                lambda f: regex_compilee.search(f['nom']), 
                fichiers
            ))
            
            if len(fichiers_matches):
                if regex.folder_id not in resultats_finaux:
                    resultats_finaux[regex.folder_id] = []
                # Retirer les fichiers matchés directement de la liste principale
                for fichier_match in fichiers_matches:
                    fichier_match['regex'] = regex.regex_pattern
                    fichier_match['weight'] = regex.weight
                    resultats_finaux[regex.folder_id].append(fichier_match)
                    fichiers.remove(fichier_match)
    
        all_regex = None  # Libérer la mémoire
        fichiers_matches = None  # Libérer la mémoire
    fichiers = None  # Libérer la mémoire
    
    list_jobs = []
    global parrallel_jobs
    parrallel_jobs = ProcessPoolExecutor(max_workers=1)

    for folder_id, files in resultats_finaux.items():
        # Lancer le traitement des fichiers en parallèle
        list_jobs.append(Thread(target=process_file_by_folder, args=(files, folder_id, database_url)))
        list_jobs[-1].start()
    resultats_finaux = None  # Libérer la mémoire
    
    for job in list_jobs:
        try:
            job.join()
        except Exception as e:
            stderr.write(f"Error processing files: {e}\n")
    
    parrallel_jobs.shutdown()

    return

def incrementaller(folder_files,database_url):
    fichiers = [
            {'nom': fichier, 'chemin': os.path.join(folder_files, fichier)}
            for fichier in os.listdir(folder_files)
            if os.path.isfile(os.path.join(folder_files, fichier))
        ]
        
    if not fichiers:
        return
    
    with setup_database(database_url) as session:
        stderr.write("Start Renamer !\n")
        # Récupérer toutes les regex triées par poids décroissant
        all_regex = get_all_incrementaller(session)
        
        # Traiter chaque regex et supprimer les fichiers matchés directement
        resultats_finaux = {}
        
        for regex in all_regex:
            if not fichiers:  # Plus de fichiers à traiter
                break
                
            # Compiler la regex
            regex_compilee = re.compile(regex.regex_pattern)
            
            # Filtrer les fichiers qui matchent
            fichiers_matches = list(filter(
                lambda f: regex_compilee.search(f['nom']), 
                fichiers
            ))
            
            if len(fichiers_matches):
                for fichier_match in fichiers_matches:
                    fichiers.remove(fichier_match)
                    try:
                        episode_number = extraire_episode(fichier_match['nom'], regex.regex_pattern)
                        if episode_number == None or (not episode_number.isdigit()):
                            raise Exception(f"Episode not found")
                        elif int(episode_number) > 0:
                            new_file_path = os.path.join(os.path.dirname(fichier_match['chemin']), regex.rename_pattern.replace(episode_pattern_insert, f"{(int(episode_number)+regex.episode_incremental):02}"))
                            shutil.move(fichier_match['chemin'], new_file_path)
                            stderr.write(f'\tFile {fichier_match['nom']} rename in {os.path.basename(new_file_path)}\n')
                    except Exception as e:
                        stderr.write(f"Error processing {fichier_match['nom']}: {e}\n")

def run_uvicorn():
    env_path = os.path.join(tools.tmpFolder, "gestionar_show_api.env")
    
    # Écrit la variable DATABASE_URL dans un fichier .env
    with open(env_path, "w") as env_file:
        env_file.write(f"DATABASE_URL={database_url_param['database_url']}\n")
    uvicorn.run("gestionar_show_api:app", host="0.0.0.0", port=8080, workers=5, env_file=env_path, log_level="error")

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
        if (not os.access(args.error, os.W_OK)):
            raise Exception(f"{args.error} not writable")
        if (not os.access(args.folder, os.W_OK)):
            raise Exception(f"{args.folder} not writable")
        if (not os.access(args.tmp, os.W_OK)):
            raise Exception(f"{args.tmp} not writable")
        os.chdir(args.pwd)
        tools.dev = args.dev
        tools.software = tools.config_loader(args.config, "software")
        if args.core > 1:
            tools.core_to_use = args.core-1
        else:
            tools.core_to_use = 1
        tools.folder_error = args.error
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

        tools.special_params = {"change_all_und":True, "remove_commentary":True, "forced_best_video_contain":False}
        with open(args.database_url_file) as database_url_file:
            database_url_param = json.load(database_url_file)
        stderr.write("Setup database\n")
        with setup_database(database_url_param["database_url"], create_tables=True) as session:
            pass

        uvicorn_process = Process(target=run_uvicorn)
        uvicorn_process.start()
        
        stderr.write("Start !\n")
        while True:
            incrementaller(args.folder,database_url_param["database_url"])
            process_files_in_folder(args.folder,database_url_param["database_url"])
            import gc
            gc.collect()
            stderr.write("\n\n\n\nPOSO !!!!!!!\n\n\n\n\n")
            sleep(args.wait)
        
        uvicorn_process.terminate()
        uvicorn_process.join()
    except:
        traceback.print_exc()
        exit(1)
    exit(0)