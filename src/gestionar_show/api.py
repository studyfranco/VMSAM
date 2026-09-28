from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from typing import Optional
from time import sleep
from .model import get_folder_by_path, insert_folder, get_all_regex, insert_regex, get_regex_data, update_regex, get_incrementaller_data,get_all_incrementaller, insert_incrementaller, update_incrementaller, search_like_folder, get_regex_by_folder_id, get_all_folder, get_all_incompatible_files, get_incompatible_file_by_path, get_episode_data, get_folder_data, get_episodes_by_folder_id, insert_episode, get_special_rename_data, get_all_special_rename, insert_special_rename, update_special_rename
import tools
import urllib.error
from . import internal_client
from .settings import Settings

episode_pattern_insert = "{<episode>}"

# Initialise la DB
engine = None

def get_session():
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        try:
            session.close()
            import gc
            gc.collect()
        except Exception as e:
            pass
        raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
    except Exception as e:
        try:
            session.close()
            import gc
            gc.collect()
        except Exception as e:
            pass
        session.rollback()
        raise e
    finally:
        try:
            session.close()
            import gc
            gc.collect()
        except Exception as e:
            pass

class Folder(BaseModel):
    destination_path: str
    original_language: str
    number_cut: int = Field(default=10)
    cut_file_to_get_delay_second_method: float = Field(default=2)
    max_episode_number: int = Field(default=12)
    
class Regex(BaseModel):
    regex_pattern: str
    rename_pattern: Optional[str] = None
    weight: int = Field(default=1)
    example_filename: str
    destination_path: str

class Incrementaller(BaseModel):
    regex_pattern: str
    rename_pattern: str
    episode_incremental: int = Field(default=12)
    example_filename: str

class FusionRequest(BaseModel):
	error_file_path: str

class SpecialRename(BaseModel):
    file_name: str
    new_file_name: str
    destination_path: str

class IndexFolderRequest(BaseModel):
    folder_id: int

app = FastAPI(
    title="Gestionar Show API",
    description="API pour la gestion des folders, regex patterns et épisodes",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    settings = Settings()
    global engine
    engine = create_engine(settings.DATABASE_URL, echo=False)

@app.post("/folders/")
def create_folder(folder_in: Folder, session: Session = Depends(get_session)):
    # Vérifie s’il existe déjà
    existing = get_folder_by_path(folder_in.destination_path, session)
    if existing != None:
        existing.max_episode_number = folder_in.max_episode_number
        existing.number_cut = folder_in.number_cut
        existing.cut_file_to_get_delay_second_method = folder_in.cut_file_to_get_delay_second_method
        existing.original_language = folder_in.original_language
        session.commit()
        return {
            "message": "Folder already exists and updated",
            "folder_id": existing.id
        }

    import os
    if os.path.isfile(folder_in.destination_path):
        raise HTTPException(status_code=400, detail="A regular file already exists at this path")
    elif (not os.path.isdir(folder_in.destination_path)):
        try:
            if (not os.makedirs(folder_in.destination_path, exist_ok=True)):
                raise HTTPException(status_code=400, detail="Folder can't be created")
            sleep(1)
            if (not os.path.isdir(folder_in.destination_path)):
                raise HTTPException(status_code=400, detail="Folder can't be created")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Folder can't be created: {str(e)}")

    elif (not os.access(folder_in.destination_path, os.W_OK)):
        raise HTTPException(status_code=400, detail="Folder not writable")

    # Création avec valeurs Pydantic (défauts inclus automatiquement)
    new_folder = insert_folder(folder_in.destination_path, folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number, session)

    return {
        "message": "Folder created",
        "folder_id": new_folder.id
    }

def test_regex_rename(regex_data):
    if regex_data.rename_pattern != None and episode_pattern_insert not in regex_data.rename_pattern:
        raise HTTPException(status_code=400, detail=f"Rename pattern must contain the episode pattern: {episode_pattern_insert}")

def get_test_folder(regex_data,session):
    folder = get_folder_by_path(regex_data.destination_path, session)
    if folder == None:
        raise HTTPException(status_code=400, detail=f"Folder {regex_data.destination_path} not found")
    return folder

@app.post("/regex/")
def create_regex(regex_data: Regex, session: Session = Depends(get_session)):
    import re
    # Vérifier que la nouvelle regex matche le fichier d'exemple
    # Vérifier que la regex permet d'extraire un numéro d'épisode valide
    match = re.search(regex_data.regex_pattern, regex_data.example_filename)
    if match != None:
        if 'episode' in match.groupdict():
            episode_number = match.group('episode')
            if (not episode_number.isdigit()) or int(episode_number) < 1:
                raise HTTPException(status_code=400, detail=f"Regex does not extract a valid episode number. We get: {episode_number}")
        else:
            raise HTTPException(status_code=400, detail="Regex does not extract a valid episode number from the example filename")
    else:
        raise HTTPException(status_code=400, detail="Regex does not match the example filename")
    
    test_regex_rename(regex_data)
    
    regex = get_regex_data(regex_data.regex_pattern, session)
    if regex == None:
        
        # Vérifier l'existence du dossier via son path
        folder = get_test_folder(regex_data,session)

        # Vérifier les conflits : aucune regex existante ne doit matcher le fichier d'exemple
        all_regex = get_all_regex(session)
        for regex in all_regex:
            if re.search(regex.regex_pattern, regex_data.example_filename) != None:
                raise HTTPException(status_code=400, detail=f"Conflict with existing regex: `{regex.regex_pattern}`")

        # Créer et insérer la nouvelle regex
        try:
            insert_regex(regex_data.regex_pattern, folder.id, regex_data.rename_pattern, regex_data.weight, session)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during insertion of the regex: {e}")

        return {
            "message": "Regex added",
            "regex_pattern": regex_data.regex_pattern,
            "extracted_episode": int(episode_number),
            "folder_id": folder.id
        }
    else:
        # Si la regex existe déjà, on met à jour les champs
        try:
            update_regex(regex, get_test_folder(regex_data,session).id, regex_data.rename_pattern, regex_data.weight, session)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during update of the regex: {e}")
        return {
            "message": "Regex updated",
            "regex_pattern": regex_data.regex_pattern
        }

@app.post("/incrementaller/")
def create_regex_incrementaller(incremental_data: Incrementaller, session: Session = Depends(get_session)):
    import re
    incremental = get_incrementaller_data(incremental_data.regex_pattern, session)
    if incremental == None:
        # Vérifier que la nouvelle regex matche le fichier d'exemple
        # Vérifier que la regex permet d'extraire un numéro d'épisode valide
        match = re.search(incremental_data.regex_pattern, incremental_data.example_filename)
        if match != None:
            if 'episode' in match.groupdict():
                episode_number = match.group('episode')
                if (not episode_number.isdigit()) or int(episode_number) < 1:
                    raise HTTPException(status_code=400, detail=f"Regex does not extract a valid episode number. We get: {episode_number}")
            else:
                raise HTTPException(status_code=400, detail="Regex does not extract a valid episode number from the example filename")
        else:
            raise HTTPException(status_code=400, detail="Regex does not match the example filename")

        test_regex_rename(incremental_data)

        # Vérifier les conflits : aucune regex existante ne doit matcher le fichier d'exemple
        all_incremental = get_all_incrementaller(session)
        for regex in all_incremental:
            if re.search(regex.regex_pattern, incremental_data.example_filename) != None:
                raise HTTPException(status_code=400, detail=f"Conflict with existing regex: `{regex.regex_pattern}`")

        # Créer et insérer la nouvelle regex
        try:
            insert_incrementaller(incremental_data.regex_pattern, incremental_data.rename_pattern, incremental_data.episode_incremental, session)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during insertion of the regex: {e}")

        return {
            "message": "Regex added",
            "regex_pattern": incremental_data.regex_pattern,
            "extracted_episode": int(episode_number),
            "new_file_name": incremental_data.rename_pattern.replace(episode_pattern_insert, f"{(int(episode_number)+incremental_data.episode_incremental):02}")
        }
    else:
        test_regex_rename(incremental_data)
        # Si la regex existe déjà, on met à jour les champs
        try:
            update_incrementaller(incremental, incremental_data.rename_pattern, incremental_data.episode_incremental, session)
            match = re.search(incremental_data.regex_pattern, incremental_data.example_filename)
            episode_number = match.group('episode')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during update of the regex: {e}")
        return {
            "message": "Regex updated",
            "regex_pattern": incremental_data.regex_pattern,
            "new_file_name": incremental_data.rename_pattern.replace(episode_pattern_insert, f"{(int(episode_number)+incremental_data.episode_incremental):02}")
        }

@app.get("/folders_list/")
def get_folders_list(session: Session = Depends(get_session)):
    """Récupère la liste des dossiers"""
    folders = get_all_folder(session)
    if not folders:
        raise HTTPException(status_code=404, detail="No folders found")

    infos = []
    for folder in folders:
        infos.append({
            "id": folder.id,
            "destination_path": folder.destination_path,
            "original_language": folder.original_language,
            "number_cut": folder.number_cut,
            "cut_file_to_get_delay_second_method": folder.cut_file_to_get_delay_second_method,
            "max_episode_number": folder.max_episode_number
        })
    return {
        "folders": infos
    }

@app.get("/folders_infos/")
def get_folder_info(destination_like: str, session: Session = Depends(get_session)):
    """Récupère les infos des dossiers qui matchent le nom partiel"""
    folders = search_like_folder(destination_like, session)
    if not folders:
        raise HTTPException(status_code=404, detail="No folders found matching the criteria")
    
    infos = []
    for folder in folders:
        infos.append({
            "id": folder.id,
            "destination_path": folder.destination_path,
            "original_language": folder.original_language,
            "number_cut": folder.number_cut,
            "cut_file_to_get_delay_second_method": folder.cut_file_to_get_delay_second_method,
            "max_episode_number": folder.max_episode_number
        })
    return {
        "folders": infos
    }
    
@app.get("/regex_folder/")
def get_regex_by_folder(folder_id: int, session: Session = Depends(get_session)):
    """Récupère les regex d'un dossier spécifique"""
    regex_list = get_regex_by_folder_id(folder_id, session)
    if not regex_list:
        raise HTTPException(status_code=404, detail="No regex found for this folder")
    
    infos = []
    for regex in regex_list:
        infos.append({
            "regex_pattern": regex.regex_pattern,
            "rename_pattern": regex.rename_pattern,
            "weight": regex.weight
        })
    return {
        "regex_patterns": infos
    }

@app.get("/errors")
def get_incompatible_files_list(session: Session = Depends(get_session)):
    """Liste les fichiers rejetés par le pipeline avec leurs métadonnées"""
    incompatible_files = get_all_incompatible_files(session)

    infos = []
    for incompatible_file in incompatible_files:
        infos.append({
            "id": incompatible_file.id,
            "folder_id": incompatible_file.folder_id,
            "episode_number": incompatible_file.episode_number,
            "file_path": incompatible_file.file_path,
            "file_weight": incompatible_file.file_weight
        })
    return {
        "incompatible_files": infos
    }

@app.get("/fusion")
def get_fusion_queue_status():
    """Etat de la file d'attente, relaye depuis l'instance interne"""
    try:
        status_code, payload = internal_client.call_internal_api("GET", "/internal/fusion")
    except (urllib.error.URLError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"Internal fusion worker unreachable: {e}")

    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=payload.get("detail", "Internal fusion worker error"))
    return payload

@app.post("/fusion")
def create_fusion_job(fusion_request: FusionRequest, session: Session = Depends(get_session)):
    """Valide une demande de fusion puis la transmet au worker interne"""
    incompatible_file = get_incompatible_file_by_path(fusion_request.error_file_path, session)
    if incompatible_file == None:
        raise HTTPException(status_code=404, detail=f"{fusion_request.error_file_path} not found in incompatible_files")

    master_episode = get_episode_data(incompatible_file.folder_id, incompatible_file.episode_number, session)
    if master_episode == None:
        raise HTTPException(status_code=404, detail=f"No episode {incompatible_file.episode_number} registered for folder {incompatible_file.folder_id}, nothing to merge with")

    try:
        status_code, payload = internal_client.call_internal_api(
            "POST", "/internal/fusion", payload={"error_file_path": fusion_request.error_file_path}
        )
    except (urllib.error.URLError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"Internal fusion worker unreachable: {e}")

    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=payload.get("detail", "Internal fusion worker error"))

    payload["master_file_path"] = master_episode.file_path
    return payload

@app.get("/health")
def get_health():
    """Verification de deploiement: version, mode courant et etat d'execution"""
    status = "ok"
    is_running = False
    # None means "the internal instance did not tell us", which is not the same
    # as False. A caller must be able to tell an unreachable worker from a
    # worker that answered and reported its fusion path disabled.
    fusion_enabled = None
    queue_length = None
    internal_status = None
    try:
        status_code, payload = internal_client.call_internal_api("GET", "/internal/health", timeout=5)
        internal_status = status_code
        if status_code == 200:
            is_running = bool(payload.get("is_running", False))
            queue_length = payload.get("queue_length")
            # `/internal/health` stays answerable precisely so it can say WHY
            # fusion is unavailable (internal_api.py refuses to start the worker
            # when the test output dir is unset). Reading only `is_running`
            # threw that answer away and reported "ok" for an instance that
            # cannot fuse anything.
            fusion_enabled = payload.get("fusion_enabled")
            if fusion_enabled is False:
                status = "degraded"
        else:
            status = "degraded"
    except (urllib.error.URLError, OSError):
        # The public instance keeps serving its own endpoints when the internal
        # worker is down, so this reports degraded rather than failing outright.
        status = "degraded"

    return {
        "status": status,
        "git_commit": tools.get_git_commit(),
        "mode": tools.get_execution_mode(),
        # Whether the verbose diagnostics are actually on. The image has always
        # set dev=true, but until run.sh was fixed nothing passed --dev, so
        # tools.dev stayed False and every gated diagnostic was silently dark.
        # There was no way to tell from outside; now there is.
        "dev": tools.get_dev_env_var(),
        "is_running": is_running,
        # Reported verbatim from `/internal/health`; None means it did not answer.
        "fusion_enabled": fusion_enabled,
        "queue_length": queue_length,
        "internal_status": internal_status
    }

def extract_episode_number(file_name, regex_pattern):
    """Same contract as the daemon's extraire_episode: the `episode` named group,
    else the first group. Returns the raw string, or None when nothing matched."""
    import re
    match = re.search(regex_pattern, file_name)
    if match:
        if 'episode' in match.groupdict():
            return match.group('episode')
        elif match.groups():
            return match.group(1)
    return None

def test_bare_file_name(value, field_name):
    """A special rename is a name inside the watch folder, never a path."""
    import os
    if value == None or len(value.strip()) == 0:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if os.sep in value or (os.altsep != None and os.altsep in value) or value in ('.', '..'):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a file name, not a path")

@app.post("/special/")
def create_special_rename(special_data: SpecialRename, session: Session = Depends(get_session)):
    """Declare (or edit) the final name of a special. The daemon renames the
    exact incoming name in place, then the ordinary folder regex picks the new
    name up.

    The table knows no folder; the endpoint does, as a control against a wrong
    destination and against cross-matches: a new name caught by a regex of any
    other folder, or by an increment rule, is refused. The heaviest regex of
    the given folder that catches the new name is returned so the caller can
    show and edit it instead of creating a second one. An existing special
    (same incoming name) is updated under the same control.
    """
    import re
    test_bare_file_name(special_data.file_name, "file_name")
    test_bare_file_name(special_data.new_file_name, "new_file_name")
    if special_data.file_name == special_data.new_file_name:
        raise HTTPException(status_code=400, detail="new_file_name must differ from file_name")

    folder = get_test_folder(special_data, session)

    # Every regex that would extract an episode number from the new name,
    # heaviest first like the daemon. The heaviest one integrates the file; any
    # other folder among them is a cross-match: the name is ambiguous between
    # shows, and a later weight change would silently move the special.
    matching_regex = []
    for regex in get_all_regex(session):
        episode_number = extract_episode_number(special_data.new_file_name, regex.regex_pattern)
        if episode_number != None and episode_number.isdigit() and int(episode_number) > 0:
            matching_regex.append((regex, int(episode_number)))

    foreign = [regex for regex, episode_number in matching_regex if regex.folder_id != folder.id]
    if foreign:
        raise HTTPException(status_code=400, detail="The new name is caught by a regex of another folder: " + "; ".join(
            f"`{regex.regex_pattern}` of folder {regex.folder_id} ({regex.folder.destination_path})" for regex in foreign
        ) + f". Choose a name only folder {folder.id} recognises")

    # The incrementaller runs after the special renamer in the daemon loop: a
    # new name it matches would be renamed a second time, with a shifted number.
    for incremental in get_all_incrementaller(session):
        if re.search(incremental.regex_pattern, special_data.new_file_name) != None:
            raise HTTPException(status_code=400, detail=f"The new name is caught by the increment rule `{incremental.regex_pattern}`: the daemon would rename it again. Choose a name no increment rule recognises")

    special = get_special_rename_data(special_data.file_name, session)
    try:
        if special == None:
            insert_special_rename(special_data.file_name, special_data.new_file_name, session)
            message = "Special rename added"
        else:
            update_special_rename(special, special_data.new_file_name, session)
            message = "Special rename updated"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during insertion of the special rename: {e}")

    return {
        "message": message,
        "file_name": special_data.file_name,
        "new_file_name": special_data.new_file_name,
        "folder_id": folder.id,
        # None: no regex catches the new name yet, the caller should create one.
        # Otherwise the heaviest regex of this folder, the one that will
        # integrate the file, so the caller can show and edit it.
        "matching_regex": None if not matching_regex else {
            "regex_pattern": matching_regex[0][0].regex_pattern,
            "rename_pattern": matching_regex[0][0].rename_pattern,
            "weight": matching_regex[0][0].weight,
            "folder_id": matching_regex[0][0].folder_id,
            "extracted_episode": matching_regex[0][1]
        }
    }

@app.get("/special_list/")
def get_special_rename_list(session: Session = Depends(get_session)):
    """Liste les renommages de speciaux declares"""
    return {
        "special_renames": [
            {"file_name": special.file_name, "new_file_name": special.new_file_name}
            for special in get_all_special_rename(session)
        ]
    }

@app.get("/incrementaller_list/")
def get_incrementaller_list(session: Session = Depends(get_session)):
    """Liste les regex d'incrementation declarees"""
    return {
        "incrementaller": [
            {
                "regex_pattern": incremental.regex_pattern,
                "rename_pattern": incremental.rename_pattern,
                "episode_incremental": incremental.episode_incremental
            }
            for incremental in get_all_incrementaller(session)
        ]
    }

@app.post("/index_folder/")
def index_folder(index_request: IndexFolderRequest, session: Session = Depends(get_session)):
    """Rename and register the files already present in a folder.

    For every regular file of the folder that the database does not know yet,
    the folder's own regex (heaviest first) extract the episode number; the file
    is renamed with the regex rename pattern and inserted as an episode carrying
    the regex weight. Nothing is merged, nothing is deleted, nothing outside the
    folder is touched. Runs without the episode lock: the public instance has no
    merge runtime, and a file the database does not know cannot be held by the
    integration loop or the fusion worker.
    """
    import os
    current_folder = get_folder_data(index_request.folder_id, session)
    if current_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {index_request.folder_id} not found")
    if not os.path.isdir(current_folder.destination_path):
        raise HTTPException(status_code=400, detail=f"Folder path is not a directory on this instance")

    regex_list = get_regex_by_folder_id(index_request.folder_id, session)
    if not regex_list:
        raise HTTPException(status_code=404, detail="No regex found for this folder")
    regex_list = sorted(regex_list, key=lambda regex: regex.weight, reverse=True)

    known_paths = set(episode.file_path for episode in get_episodes_by_folder_id(index_request.folder_id, session))

    file_names = sorted(
        file_name for file_name in os.listdir(current_folder.destination_path)
        if os.path.isfile(os.path.join(current_folder.destination_path, file_name)) and not file_name.startswith('.')
    )

    indexed = []
    skipped = []
    unmatched = []
    already_indexed = 0
    for file_name in file_names:
        file_path = os.path.join(current_folder.destination_path, file_name)
        if file_path in known_paths:
            already_indexed += 1
            continue

        matched_regex = None
        episode_number = None
        for regex in regex_list:
            episode_number = extract_episode_number(file_name, regex.regex_pattern)
            if episode_number != None:
                matched_regex = regex
                break
        if matched_regex == None:
            unmatched.append(file_name)
            continue

        if (not episode_number.isdigit()) or int(episode_number) < 1:
            skipped.append({"file_name": file_name, "reason": "invalid_episode_number", "regex_pattern": matched_regex.regex_pattern, "extracted": episode_number})
            continue
        episode_number = int(episode_number)
        if episode_number > current_folder.max_episode_number:
            skipped.append({"file_name": file_name, "reason": "above_max_episode_number", "episode_number": episode_number})
            continue

        # The daemon merges a newcomer into the registered episode; indexing does
        # not merge anything, so a file whose episode is already registered is
        # left exactly where it is, untouched and unregistered. The registered
        # file keeps being the master of that episode.
        if get_episode_data(index_request.folder_id, episode_number, session) != None:
            skipped.append({"file_name": file_name, "reason": "episode_already_registered", "episode_number": episode_number})
            continue

        if matched_regex.rename_pattern != None:
            new_file_name = matched_regex.rename_pattern.replace(episode_pattern_insert, f"{episode_number:02}")
        else:
            new_file_name = file_name
        new_file_path = os.path.join(current_folder.destination_path, new_file_name)

        if new_file_path != file_path:
            if os.path.exists(new_file_path):
                skipped.append({"file_name": file_name, "reason": "target_exists", "new_file_name": new_file_name, "episode_number": episode_number})
                continue
            try:
                os.rename(file_path, new_file_path)
            except OSError as e:
                skipped.append({"file_name": file_name, "reason": "rename_failed", "new_file_name": new_file_name, "error": str(e)})
                continue

        try:
            insert_episode(index_request.folder_id, episode_number, new_file_path, matched_regex.weight, session)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File {new_file_name} renamed but not registered: {e}")
        indexed.append({
            "file_name": file_name,
            "new_file_name": new_file_name,
            "episode_number": episode_number,
            "file_weight": matched_regex.weight,
            "regex_pattern": matched_regex.regex_pattern
        })

    return {
        "message": f"{len(indexed)} file(s) indexed",
        "folder_id": current_folder.id,
        "destination_path": current_folder.destination_path,
        "indexed": indexed,
        "skipped": skipped,
        "unmatched": unmatched,
        "already_indexed": already_indexed
    }
