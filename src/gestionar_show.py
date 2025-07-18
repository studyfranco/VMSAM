import argparse
import os
from multiprocessing import Pool
from sys import stderr
from time import sleep
import tools
from gestionar_show_model import setup_database, get_folder_data, get_all_regex, get_episode_data, get_regex_data, insert_episode
import re
from mergeVideo import merge_videos

episode_pattern_insert = "{<episode>}"

def process_episode(files, folder_id, episode_number, database_url):
    """Process files for a specific folder and extract episodes"""
    session = setup_database(database_url)
    
    try:
        # Récupérer le dossier
        current_folder = get_folder_data(folder_id, session)
        
        # Traiter les fichiers
        for file in files:
            previous_file = get_episode_data(folder_id, episode_number, session)
            if previous_file is not None and len(previous_file.file_path):
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
                
                tools.tmpFolder = os.path.join(tools.tmpFolder, str(episode_number))
                tools.make_dirs(tools.tmpFolder)
                out_folder = os.path.join(tools.tmpFolder, "final_file")
                tools.make_dirs(out_folder)
                try:
                    merge_videos([file['chemin'],previous_file.file_path],out_folder,True)
                    os.remove(previous_file.file_path)
                    if previous_file.file_weight < file['weight']:
                        os.rename(os.path.join(out_folder, os.path.splitext(os.path.basename(file['chemin']))[0]+'_merged.mkv'), new_file_path)
                    elif previous_file.file_weight > file['weight']:
                        os.rename(os.path.join(out_folder, os.path.splitext(os.path.basename(previous_file.file_path))[0]+'_merged.mkv'), new_file_path)
                    else:
                        os.rename(os.path.join(out_folder, os.path.splitext(os.path.basename(file['chemin']))[0]+'_merged.mkv'), new_file_path)
                except Exception as e:
                    stderr.write(f"Error processing file {file['nom']}: {e}\n")
                    if previous_file.file_weight < file['weight'] or previous_file.file_weight == file['weight']:
                        os.rename(previous_file.file_path, os.path.join(tools.folder_error, os.path.basename(previous_file.file_path)))
                        os.rename(file['chemin'], new_file_path)
                        with open(os.path.join(tools.folder_error, os.path.join(tools.folder_error, os.path.basename(previous_file.file_path))+"log.error"),"w") as log:
                            log.write(f"{e}")
                    else:
                        os.rename(file['chemin'], os.path.join(tools.folder_error, os.path.basename(file['chemin'])))
                        with open(os.path.join(tools.folder_error, os.path.join(tools.folder_error, os.path.basename(file['chemin']))+"log.error"),"w") as log:
                            log.write(f"{e}")
                previous_file.file_path = new_file_path
                previous_file.file_weight = new_file_weight
                session.commit()
            else:
                # Si l'épisode n'existe pas, on l'ajoute
                regex_data = get_regex_data(file['regex'], session)
                new_file_path = os.path.join(current_folder.destination_path, regex_data.rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}"))
                os.rename(file['chemin'], new_file_path)
                insert_episode(folder_id, episode_number, new_file_path, file['weight'], session)
    except Exception as e:
        stderr.write(f"Error processing files for folder {folder_id}, episode {episode_number}: {e}\n")
    finally:
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
        session = setup_database(database_url)
        # Récupérer le dossier
        current_folder = get_folder_data(folder_id, session)
        tools.tmpFolder = os.path.join(tools.tmpFolder, str(current_folder.id))
        if not tools.make_dirs(tools.tmpFolder):
            session.close()
            raise Exception("Impossible to create the temporar dir")
        tools.default_language_for_undetermine = current_folder.original_language
        tools.special_params["original_language"] = current_folder.original_language
        session.close()
        parrallel_jobs = Pool(processes=tools.core_to_use)
        list_jobs = []
        for episode_number, files in group_files_by_episode.items():
            # Lancer le traitement des fichiers en parallèle
            list_jobs.append(parrallel_jobs.apply_async(
                process_episode, (files, folder_id, episode_number, database_url)
            ))
        for job in list_jobs:
            try:
                job.get()
            except Exception as e:
                stderr.write(f"Error processing files: {e}\n")

def process_files_in_folder(folder_files,database_url):
    fichiers = [
            {'nom': fichier, 'chemin': os.path.join(folder_files, fichier)}
            for fichier in os.listdir(folder_files)
            if os.path.isfile(os.path.join(folder_files, fichier))
        ]
        
    if not fichiers:
        return
    
    session = setup_database(database_url)
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
    
    session.close()
    parrallel_jobs = Pool(processes=tools.core_to_use)
    list_jobs = []
    for folder_id, files in resultats_finaux.items():       
        # Lancer le traitement des fichiers en parallèle
        list_jobs.append(parrallel_jobs.apply_async(
            process_file_by_folder, (files, folder_id, database_url)
        ))
    
    for job in list_jobs:
        try:
            job.get()
        except Exception as e:
            stderr.write(f"Error processing files: {e}\n")
    
    return


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
    parser.add_argument("database_url", metavar='database_url', type=str,
                        help="Database URL to connect to the gestionar_show database")
    parser.add_argument("--pwd", metavar='pwd', type=str,
                        default=".", help="Path to the software, put it if you use the folder from another folder")
    parser.add_argument("--dev", dest='dev', default=False, action='store_true', help="Print more errors and write all logs")
    args = parser.parse_args()
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
    tools.special_params = {"change_all_und":True, "remove_commentary":True, "forced_best_video_contain":False}
    while True:
        process_files_in_folder(args.folder,args.database_url)
        sleep(args.wait)