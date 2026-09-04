"""Moteur de reprise de fusion pour les fichiers rejetés par le pipeline.

Une file en mémoire, un thread qui la consomme, et un unique process fils qui
exécute la fusion. Le fils est ce qui rend le worker increvable: un plantage de
merge_videos tue le fils, pas la boucle, et la mémoire du merge est rendue au
système à chaque job.

Le mode d'exécution vient de VMSAM_MODE (tout sauf `production` => test):

* test       -- rien n'est écrit hors de VMSAM_TEST_OUTPUT_DIR, ni sur disque
                ni en base.
* production -- le fichier maître est remplacé, la ligne incompatible_files
                supprimée et le fichier en erreur effacé. En cas d'échec, rien.
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
    """Racine des sorties de test. Vide = non configuré, l'API interne refuse."""
    return os.environ.get("VMSAM_TEST_OUTPUT_DIR", "").strip()


def is_test_mode():
    """Test par defaut: seul le mot exact `production` ouvre la branche destructive.

    Le predicat porte sur mode_production et non sur mode_test, sinon une valeur
    vide -- ce qu'une variable declaree sans valeur dans un compose produit tout
    seul -- ou une faute de frappe basculerait en production et remplacerait le
    fichier maitre.
    """
    return tools.get_execution_mode() != tools.mode_production


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


''' Gestion de la file '''
def enqueue_fusion_job(error_file_path):
    """Ajoute un job. Retourne sa position parmi ceux en attente."""
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
    """Crée le pool d'un seul process et démarre le thread consommateur.

    Le pool est construit ici plutôt qu'à l'import: un ProcessPoolExecutor créé
    au chargement du module forkerait depuis un interpréteur encore en train de
    s'initialiser. max_workers=1 est le mandat: un seul merge à la fois.
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
    fusion_queue.put(shutdown_sentinel)


def worker_loop():
    """Consomme la file, un job à la fois, sans limite de durée.

    `get()` bloque jusqu'à l'arrivée d'un job: pas de timeout, donc pas de tour
    de boucle à vide, et un job soumis réveille le thread immédiatement. L'arrêt
    passe par une sentinelle plutôt que par un drapeau relu périodiquement.
    """
    global current_job, parrallel_jobs
    while True:
        job = fusion_queue.get()
        if job is shutdown_sentinel:
            fusion_queue.task_done()
            # shutdown() rend le pool inutilisable definitivement: on le remet a
            # None pour que start_worker puisse en recreer un. Sans cela, sa
            # garde `if parrallel_jobs == None` le laisserait tel quel et le
            # premier submit leverait "cannot schedule new futures after
            # shutdown".
            parrallel_jobs.shutdown()
            parrallel_jobs = None
            return

        with state_lock:
            tools.remove_element_without_bug(pending_jobs, job)
            current_job = {"error_file_path": job, "started_at": now_iso()}

        try:
            # .result() attend la fin du fils sans borne: une fusion dure ce
            # qu'elle dure, et rien ici ne doit l'interrompre.
            parrallel_jobs.submit(run_fusion_job, database_url, job).result()
        except Exception as e:
            stderr.write(f"Fusion job failed for {job}: {e}\n")
        finally:
            with state_lock:
                current_job = None
            fusion_queue.task_done()


''' Exécution de la fusion, dans le process fils '''
def resolve_rename_pattern(folder_id, incompatible_file, session):
    """Le rename_pattern du fichier en erreur, ou None.

    incompatible_files ne garde pas la regex qui a identifié le fichier. On la
    retrouve par le poids, qui vient d'elle; si plusieurs regex du dossier
    partagent ce poids, on départage en re-matchant le nom de base, que
    process_rejected_files a conservé en déplaçant le fichier.
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
    """Écrit un log sans jamais faire échouer le job pour autant.

    Et, quand `merge_plan` n'est pas `None`, le rapport de plan A COTE du
    fichier. LE NOM COMPLET EST CELUI QUE `write_report` REND, jamais un nom
    reconstruit ici: l'extension appartient a `merge_plan_report.report_path` et
    elle a deja change une fois. `merge_plan` vaut
    `mergeVideo.merge_plan`: `None` veut dire qu'AUCUNE version
    resample/chimerique n'a ete produite -- il n'y a pas de plan a rendre, et ce
    n'est pas une erreur.

    Import tardif et except large, pour la meme raison que la reparation
    elle-meme: le rapport est un lecteur greffe sur un chemin qui marchait sans
    lui. Un module absent d'un deploiement partiel, ou une cle que le rapport
    attend et que le journal ne porte pas, n'ont pas le droit de faire tomber
    une fusion reussie. Le module lit QUINZE cles en `job["x"]` -- dont treize
    lues aussi en `.get()` ailleurs -- donc une cle manquante leve ou non selon
    la branche qui s'execute, et cela ne se decide pas ici.

    L'ISSUE EST ECRITE DANS LE LOG, succes comme echec. Un rapport qui manque en
    silence se lit comme un fichier qui n'avait rien a rapporter, ce qui est
    exactement ce que `None` veut dire par ailleurs.
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
                # LE NOM DU JOURNAL D'OU VIENNENT CES OCTETS, et c'est bien le
                # nom de base du fichier -- il porte le titre et le numero
                # d'episode, et le proprietaire l'a autorise: ce rapport est
                # genere par VMSAM, s'ecrit a cote du media et n'entre jamais
                # dans le depot.
                #
                # CORRIGE: je passais ici un libelle neutre en croyant que
                # c'etait lui qui gardait le nom hors du document. C'etait FAUX
                # -- le redacteur du module le remplacait deja par
                # `<redacted:...>`. Un garde redondant qui survit a ce qu'il
                # doublait ne fait plus que du mal: celui-la aurait defait la
                # decision du proprietaire sans que rien ne le dise.
                os.path.basename(log_file_path),
                plan_anchor)
            # Un POINTEUR, et non la copie de transport. `transport_entry` porte
            # le document entier prefixe par sa longueur EN OCTETS, pour un
            # lecteur qui n'aurait que le journal; ici le fichier est ecrit a
            # cote, et l'inclure ajouterait quatorze kilo-octets de HTML au
            # milieu d'un journal qui se lit. Pour revenir a l'extraction depuis
            # le journal, ecrire `transport_entry` au lieu de cette ligne.
            content += (f"\nMerge plan report: {os.path.basename(destination)} "
                        f"({len(transport_entry.encode('utf-8'))} bytes transportable)\n")
        except Exception as e:
            # UN REFUS DE CONTRAT N'EST PAS UN PLANTAGE. `validate_job` leve
            # `merge_plan_error` PAR SON NOM quand le `job` n'a pas les quinze
            # cles que le rapport lit; confondre les deux dans le journal
            # ferait lire un defaut d'appel comme un bug du rendu.
            #
            # Le NOM de la classe et non la classe: si l'import lui-meme a
            # echoue, `merge_plan_report` n'existe pas, et un
            # `except merge_plan_report.merge_plan_error` leverait DANS le
            # gestionnaire d'erreur. Un module anterieur a cette classe tombe
            # aussi tout seul du bon cote.
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
    """Rejoue une fusion pour un fichier en erreur. Exécuté dans le process fils."""
    import mergeVideo
    import video

    tools.dev = tools.get_dev_env_var()
    test_mode = is_test_mode()
    session = setup_database(database_url)
    lock_handle = None
    try:
        # Premiere lecture: elle ne sert qu'a savoir QUEL verrou prendre.
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} is not registered in incompatible_files")

        folder_id = incompatible_file.folder_id
        episode_number = incompatible_file.episode_number

        # La boucle d'integration peut ecrire le meme episode, et le verrou vaut
        # dans les DEUX modes: meme en test, ou la fusion n'ecrit rien hors du
        # dossier de sortie, elle LIT le fichier maitre pendant que l'autre
        # process peut le supprimer puis le remplacer. On obtiendrait un resultat
        # de test faux, c'est-a-dire la pire sortie possible pour une mesure qui
        # sert a decider.
        lock_handle = tools.acquire_episode_lock(folder_id, episode_number, blocking=True)
        # L'attente a pu durer une fusion entiere, pendant laquelle l'autre
        # process a pu deplacer le fichier en erreur, changer son poids ou
        # remplacer le maitre. expire_all() est indispensable: sans lui, la
        # session reservirait les objets de son identity map et la re-lecture
        # ci-dessous renverrait les valeurs d'avant l'attente.
        session.expire_all()
        incompatible_file = get_incompatible_file_by_path(error_file_path, session)
        if incompatible_file == None:
            raise Exception(f"{error_file_path} no longer registered in incompatible_files after waiting for the lock")
        folder_id = incompatible_file.folder_id
        episode_number = incompatible_file.episode_number

        # Tout ce qui suit est lu APRES le verrou, donc a jour.
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

        # Arbre temporaire distinct de celui de la boucle d'integration: celle-ci
        # balaie tmpFolder_original/<folder_id> en fin de dossier, ce qui
        # effacerait le repertoire de travail d'une fusion en cours sur
        # n'importe quel episode de ce meme dossier.
        tools.tmpFolder = os.path.join(tools.tmpFolder_original, "fusion", str(folder_id), str(episode_number))
        out_folder = os.path.join(tools.tmpFolder, "final_file")

        # Même règle que process_episode: le poids décide de la référence et du
        # nom de sortie, le fichier en erreur tenant le rôle du nouveau fichier.
        if previous_file.file_weight >= incompatible_file.file_weight:
            tools.special_params["forced_best_video"] = previous_file.file_path
            new_file_path = previous_file.file_path
            new_file_weight = previous_file.file_weight
            merged_source = previous_file.file_path
        else:
            tools.special_params["forced_best_video"] = incompatible_file.file_path
            rename_pattern = resolve_rename_pattern(folder_id, incompatible_file, session)
            if rename_pattern == None:
                # Aucune regex ne se rattache au fichier: on garde le chemin du
                # maître plutôt que d'inventer un nom.
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
        # Remise a `None` et non a `[]`: `None` veut dire "aucune version
        # resample/chimerique", et c'est ce qui distingue un rapport absent d'un
        # rapport perdu.
        mergeVideo.merge_plan = None
        tools.logs = []
        if not tools.dev:
            mergeVideo.show_not_compatible_error = False

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
            # En production un echec ne touche NI les fichiers NI la base: le
            # fichier en erreur reste dans son dossier, ce qui EST deja
            # l'information. La trace n'est donc ecrite qu'en mode dev, ou l'on
            # cherche a comprendre l'echec.
            # Elle est necessaire meme la: job_error est capture plus haut, donc
            # il ne remonte jamais jusqu'au except de cette fonction et un echec
            # de fusion ne dirait rien nulle part.
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
    """Mode test: rien n'existe hors de VMSAM_TEST_OUTPUT_DIR.

    Les sources et la base ne sont pas touchées. Le succès dépose le mkv et son
    .log; l'échec ne dépose que le .log.error.
    """
    # Pas de controle sur VMSAM_TEST_OUTPUT_DIR ici: internal_api refuse de
    # demarrer le worker quand il est vide, donc aucun job n'atteint ce point.
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
        # Une version resample/chimerique a pu etre produite et le merge echouer
        # apres: le plan existe alors sans fichier produit, et il s'ancre sur le
        # nom du candidat, a cote du .log.error.
        failed_anchor = os.path.join(out_folder_final, os.path.basename(incompatible_file.file_path))
        write_log_file(failed_anchor+".log.error",
                       f"Error processing file {os.path.basename(incompatible_file.file_path)}: "
                       f"{job_error['error']}\n{job_error['traceback']}\n\n"
                       f"Merged errors:\n{chr(10).join(merged_errors)}\n\n"
                       f"Logs:\n{chr(10).join(tools.logs)}\n",
                       merge_plan, failed_anchor, incompatible_file.file_path)


def apply_production_outcome(session, incompatible_file, previous_file,
                             new_file_path, new_file_weight, merged_file_path):
    """Mode production, succès seulement: on publie et on solde l'erreur.

    Même séquence que process_episode -- écriture sous .tmp, retrait de l'ancien
    maître, renommage -- puis suppression du fichier en erreur et de sa ligne.
    """
    # Ecrire a cote de la cible puis basculer en UN appel: os.replace est
    # atomique sur un meme systeme de fichiers, et le .tmp est dans le dossier
    # de destination donc la condition tient. La sequence precedente --
    # os.remove(maitre) puis move -- laissait une fenetre, courte mais reelle,
    # ou l'episode n'avait plus aucun fichier: un plantage a cet instant perdait
    # le maitre sans que le nouveau soit en place.
    shutil.move(merged_file_path, new_file_path+'.tmp')
    os.replace(new_file_path+'.tmp', new_file_path)
    # L'ancien maitre n'est supprime que si le nom a change (cas du renommage
    # par rename_pattern): sinon os.replace vient deja de le remplacer.
    if os.path.abspath(previous_file.file_path) != os.path.abspath(new_file_path):
        os.remove(previous_file.file_path)
    os.remove(incompatible_file.file_path)
    previous_file.file_path = new_file_path
    previous_file.file_weight = new_file_weight
    # delete_incompatible_file committe: la mise à jour de l'épisode et la
    # suppression de la ligne partent dans la même transaction.
    delete_incompatible_file(incompatible_file, session)
    session.commit()
