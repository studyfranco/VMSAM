from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from typing import Optional, List
from time import sleep
from .model import get_folder_by_path, insert_folder, get_all_regex, insert_regex, get_regex_data, update_regex, get_incrementaller_data,get_all_incrementaller, insert_incrementaller, update_incrementaller, search_like_folder, get_regex_by_folder_id, get_all_folder, get_all_incompatible_files, get_incompatible_file_by_path, get_episode_data, get_folder_data, get_episodes_by_folder_id, insert_episode, update_folder_parameters, test_conformity_folder_param, replace_regex, delete_regex, delete_incompatible_file, get_episode_by_path
import tools
import urllib.error
from . import internal_client
from gestionar_common.settings import Settings
from gestionar_common.database import get_connect_args

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
        raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
    finally:
        # close() also rolls back a transaction left open by any other error
        session.close()

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

class IndexFolderRequest(BaseModel):
    folder_id: int

class FolderMove(BaseModel):
    folder_id: int
    new_destination_path: str

class EpisodeRename(BaseModel):
    episode_number: int
    new_file_name: str

class EpisodesRenameRequest(BaseModel):
    folder_id: int
    renames: List[EpisodeRename]

class RegexReplace(BaseModel):
    regex_pattern: str
    new_regex_pattern: str
    # Absent keeps the current rename pattern; an explicit null removes it.
    rename_pattern: Optional[str] = None
    weight: Optional[int] = None
    folder_id: Optional[int] = None
    example_filename: str

class RegexDeleteRequest(BaseModel):
    regex_pattern: str

class ErrorDeleteRequest(BaseModel):
    error_file_path: str

app = FastAPI(
    title="Gestionar Show API",
    description="API pour la gestion des folders, regex patterns et épisodes",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    settings = Settings()
    global engine
    # Same options as setup_database: pool_pre_ping drops a connection that died
    # with a database restart instead of failing the next request on it.
    engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=get_connect_args(settings.DATABASE_URL), pool_pre_ping=True)

def folder_to_infos(current_folder):
    """One folder, with the keys used by /folders/."""
    return {
        "folder_id": current_folder.id,
        "destination_path": current_folder.destination_path,
        "original_language": current_folder.original_language,
        "number_cut": current_folder.number_cut,
        "cut_file_to_get_delay_second_method": current_folder.cut_file_to_get_delay_second_method,
        "max_episode_number": current_folder.max_episode_number
    }

@app.post("/folder/")
def create_folder(folder_in: Folder, session: Session = Depends(get_session)):
    import os
    # One spelling per folder: `x/` and `x` must not become two folders, and a
    # relative path would be created relative to the process working directory.
    if not os.path.isabs(folder_in.destination_path):
        raise HTTPException(status_code=400, detail="destination_path must be an absolute path")
    folder_in.destination_path = os.path.normpath(folder_in.destination_path)
    # Checked before anything is written, so a refused value never leaves a
    # created directory behind, and answers 400 instead of an uncaught 500.
    try:
        test_conformity_folder_param(folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Vérifie s’il existe déjà
    existing = get_folder_by_path(folder_in.destination_path, session)
    if existing != None:
        update_folder_parameters(existing, folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number, session)
        session.commit()

        infos = folder_to_infos(existing)
        infos["message"] = "Folder updated"
        return infos

    if os.path.isfile(folder_in.destination_path):
        raise HTTPException(status_code=400, detail="A regular file already exists at this path")
    elif (not os.path.isdir(folder_in.destination_path)):
        try:
            os.makedirs(folder_in.destination_path, exist_ok=True)
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
    try:
        new_folder = insert_folder(folder_in.destination_path, folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    infos = folder_to_infos(new_folder)
    infos["message"] = "Folder created"
    return infos

@app.get("/folder/")
def get_folder(folder_id: int, session: Session = Depends(get_session)):
    """One folder by id, described like an element of /folders/."""
    current_folder = get_folder_data(folder_id, session)
    if current_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")
    return folder_to_infos(current_folder)

@app.get("/folders/")
def get_folders_list(session: Session = Depends(get_session)):
    """Récupère la liste des dossiers"""
    folders = get_all_folder(session)
    if not folders:
        raise HTTPException(status_code=404, detail="No folders found")

    infos = []
    for folder in folders:
        infos.append(folder_to_infos(folder))
    return {
        "folders": infos
    }

@app.get("/folders/destination_like/")
def get_folder_info(destination_like: str, session: Session = Depends(get_session)):
    """Récupère les infos des dossiers qui matchent le nom partiel"""
    folders = search_like_folder(destination_like, session)
    if not folders:
        raise HTTPException(status_code=404, detail="No folders found matching the criteria")
    
    infos = []
    for folder in folders:
        infos.append(folder_to_infos(folder))
    return {
        "folders": infos
    }

def test_regex_rename(rename_pattern):
    if rename_pattern != None and episode_pattern_insert not in rename_pattern:
        raise HTTPException(status_code=400, detail=f"Rename pattern must contain the episode pattern: {episode_pattern_insert}")

def get_test_folder(regex_data,session):
    import os
    folder = get_folder_by_path(os.path.normpath(regex_data.destination_path), session)
    if folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {regex_data.destination_path} not found")
    return folder

def valide_regex(regex_pattern, example_filename, rename_pattern):
    import re
    # Vérifier que la nouvelle regex matche le fichier d'exemple
    # Vérifier que la regex permet d'extraire un numéro d'épisode valide
    try:
        match = re.search(regex_pattern, example_filename)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Regex does not compile: {e}")
    if match != None:
        if 'episode' in match.groupdict():
            episode_number = match.group('episode')
            if (not episode_number.isdigit()) or int(episode_number) < 1:
                raise HTTPException(status_code=400, detail=f"Regex does not extract a valid episode number. We get: {episode_number}")
        else:
            raise HTTPException(status_code=400, detail="Regex does not extract a valid episode number from the example filename")
    else:
        raise HTTPException(status_code=400, detail="Regex does not match the example filename")
    
    test_regex_rename(rename_pattern)

    return episode_number

def regex_to_infos(regex, message, episode_number):
    """The same answer for a created and an updated regex."""
    return {
        "message": message,
        "regex_pattern": regex.regex_pattern,
        "rename_pattern": regex.rename_pattern,
        "weight": regex.weight,
        "folder_id": regex.folder_id,
        "extracted_episode": int(episode_number)
    }

@app.post("/regex/")
def create_regex(regex_data: Regex, session: Session = Depends(get_session)):
    episode_number = valide_regex(regex_data.regex_pattern, regex_data.example_filename, regex_data.rename_pattern)
    
    regex = get_regex_data(regex_data.regex_pattern, session)
    if regex == None:
        
        # Vérifier l'existence du dossier via son path
        folder = get_test_folder(regex_data,session)

        # Vérifier les conflits : aucune regex existante ne doit matcher le fichier d'exemple
        all_regex = get_all_regex(session)
        import re
        for regex in all_regex:
            try:
                if re.search(regex.regex_pattern, regex_data.example_filename) != None:
                    raise HTTPException(status_code=400, detail=f"Conflict with existing regex: `{regex.regex_pattern}`")
            except re.error:
                # A stored pattern that does not compile matches nothing.
                pass

        # Créer et insérer la nouvelle regex
        try:
            new_regex = insert_regex(regex_data.regex_pattern, folder.id, regex_data.rename_pattern, regex_data.weight, session)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Error during insertion of the regex: {e}")
        except SQLAlchemyError as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Database insertion failed, regex not added: {e}")

        return regex_to_infos(new_regex, "Regex added", episode_number)
    else:
        # Si la regex existe déjà, on met à jour les champs
        target_folder = get_test_folder(regex_data,session)
        try:
            update_regex(regex, target_folder.id, regex_data.rename_pattern, regex_data.weight, session)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Error during update of the regex: {e}")
        except SQLAlchemyError as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=f"Database update failed, regex unchanged: {e}")
        return regex_to_infos(regex, "Regex updated", episode_number)

@app.post("/regex/replace/")
def update_regex_pattern(update_data: RegexReplace, session: Session = Depends(get_session)):
    """Edit an existing regex: pattern, rename pattern, weight and folder.

    Absent fields keep their current value; rename_pattern set to null removes
    the rename pattern. The resulting regex is validated like POST /regex/: it
    must compile, match example_filename and extract an `episode` group >= 1.
    When the pattern changes, the new one must not exist yet and no other
    regex may match example_filename (the edited regex itself is excluded).
    The pattern is the primary key, so a pattern change inserts the new row
    and deletes the old one in a single transaction. Moving the regex to
    another folder is allowed.
    """

    if update_data.regex_pattern == update_data.new_regex_pattern:
        raise HTTPException(status_code=400, detail="The regex have to be different.")
    current_regex = get_regex_data(update_data.regex_pattern, session)
    if current_regex == None:
        raise HTTPException(status_code=404, detail=f"Regex `{update_data.regex_pattern}` not found")
    if get_regex_data(update_data.new_regex_pattern, session) != None:
        raise HTTPException(status_code=400, detail=f"Regex `{update_data.new_regex_pattern}` already exists")
    
    folder_id = update_data.folder_id if update_data.folder_id != None else current_regex.folder_id
    rename_pattern = update_data.rename_pattern if "rename_pattern" in update_data.model_fields_set else current_regex.rename_pattern
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    weight = update_data.weight if update_data.weight != None else current_regex.weight

    target_folder = get_folder_data(folder_id, session)
    if target_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

    episode_number = valide_regex(update_data.new_regex_pattern, update_data.example_filename, rename_pattern)

    import re
    for regex in get_all_regex(session):
        if regex.regex_pattern != current_regex.regex_pattern:
            try:
                if re.search(regex.regex_pattern, update_data.example_filename) != None:
                    raise HTTPException(status_code=400, detail=f"Conflict with existing regex: `{regex.regex_pattern}`")
            except re.error:
                # A stored pattern that does not compile matches nothing.
                pass

    try:
        updated_regex = replace_regex(current_regex, update_data.new_regex_pattern, folder_id, rename_pattern, weight, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database update failed, regex unchanged: {e}")

    return {
        "message": "Regex replaced",
        "regex_pattern": updated_regex.regex_pattern,
        "rename_pattern": updated_regex.rename_pattern,
        "weight": updated_regex.weight,
        "folder_id": updated_regex.folder_id,
        "previous_regex_pattern": update_data.regex_pattern,
        "extracted_episode": int(episode_number)
    }

@app.post("/regex/delete/")
def delete_regex_pattern(delete_request: RegexDeleteRequest, session: Session = Depends(get_session)):
    """Delete a regex by its pattern (POST so the UI relay stays simple)."""
    current_regex = get_regex_data(delete_request.regex_pattern, session)
    if current_regex == None:
        raise HTTPException(status_code=404, detail=f"Regex `{delete_request.regex_pattern}` not found")
    try:
        delete_regex(current_regex, session)
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database delete failed, regex kept: {e}")
    return {
        "message": "Regex deleted",
        "regex_pattern": delete_request.regex_pattern
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

        test_regex_rename(incremental_data.rename_pattern)

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
        test_regex_rename(incremental_data.rename_pattern)
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

@app.get("/episodes_folder/")
def get_episodes_by_folder(folder_id: int, session: Session = Depends(get_session)):
    """List the episodes registered for a folder, sorted by episode number.

    An empty list is a valid answer (200): the folder exists and nothing is
    registered yet.
    """
    current_folder = get_folder_data(folder_id, session)
    if current_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

    return {
        "folder_id": current_folder.id,
        "destination_path": current_folder.destination_path,
        "episodes": [
            {
                "episode_number": current_episode.episode_number,
                "file_path": current_episode.file_path,
                "file_weight": current_episode.file_weight
            }
            for current_episode in get_episodes_by_folder_id(folder_id, session)
        ]
    }

@app.post("/folders/move/")
def move_folder(move_request: FolderMove, session: Session = Depends(get_session)):
    """Move (or rename) a registered folder on disk and in the database.

    The directory is moved first, then the folder path and the paths of its
    episodes are rewritten. If the database update fails, the directory is
    moved back so disk and database stay consistent.

    Two deliberate limits:
    - incompatible_files rows are not rewritten. They live in the error tree,
      which mirrors the folder path at the moment of the rejection, and their
      own paths stay valid after the move.
    - No episode lock is taken, same reasoning as index_folder: the public
      instance has no merge runtime, and the owner accepted it. A folder move
      is an administrative action done while the folder is idle.

    Once the move and the database commit have both succeeded, the empty
    parent directories left behind by the old path are pruned, deepest
    first, starting from the parent of the old path and walking upwards.
    The walk stops at the first directory that:
    - is the common ancestor of the old and new paths, or above it;
    - is not empty (hidden files count as content);
    - is a mount point;
    - is the destination_path of a registered folder (it stays even empty);
    - cannot be removed (os.rmdir raised an OSError).
    Only empty directories are removed (os.rmdir), never files. The removed
    directories are listed in "removed_directories".
    """
    import os
    import shutil
    current_folder = get_folder_data(move_request.folder_id, session)
    if current_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {move_request.folder_id} not found")

    old_path = os.path.normpath(current_folder.destination_path)
    new_path = os.path.normpath(move_request.new_destination_path)
    if not os.path.isabs(new_path):
        raise HTTPException(status_code=400, detail="new_destination_path must be an absolute path")
    if os.path.normcase(os.path.abspath(old_path)) == os.path.normcase(os.path.abspath(new_path)):
        raise HTTPException(status_code=400, detail="new_destination_path is the current path of the folder")

    other_folder = get_folder_by_path(new_path, session)
    if other_folder != None and other_folder.id != current_folder.id:
        raise HTTPException(status_code=400, detail=f"{new_path} is already registered to folder {other_folder.id}")

    try:
        if old_path == os.path.commonpath([old_path,new_path]):
            raise HTTPException(status_code=400, detail="A folder cannot be moved inside itself")
    except ValueError as e:
        pass

    if os.path.lexists(new_path):
        raise HTTPException(status_code=400, detail=f"{new_path} already exists on disk")

    if not os.path.isdir(old_path):
        raise HTTPException(status_code=400, detail=f"Current folder path is not a directory on this instance, nothing moved")

    try:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, new_path)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Folder can't be moved, database untouched: {e}")

    old_prefix = os.path.join(old_path, "")
    episodes_updated = 0
    try:
        current_folder.destination_path = new_path
        for current_episode in get_episodes_by_folder_id(current_folder.id, session):
            ep_path = os.path.normpath(current_episode.file_path)
            if ep_path.startswith(old_prefix):
                current_episode.file_path = os.path.join(new_path,ep_path[len(old_prefix):])
                episodes_updated += 1
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        shutil.move(new_path, old_path)
        raise HTTPException(status_code=500, detail=f"Database update failed, the move was reverted: {e}")

    removed_folder = os.path.dirname(old_path)
    removed_directories = []
    registered_paths = set(os.path.normpath(registered_folder.destination_path) for registered_folder in get_all_folder(session))
    # An empty parent that is itself a registered folder stays: removing it would
    # leave that folder pointing to a missing directory.
    if (not os.listdir(removed_folder)) and (not os.path.ismount(removed_folder)) and (os.path.normpath(removed_folder) not in registered_paths):
        try:
            os.rmdir(removed_folder)
            removed_directories.append(removed_folder)
        except OSError:
            pass

    return {
        "message": "Folder moved",
        "folder_id": current_folder.id,
        "old_destination_path": old_path,
        "new_destination_path": new_path,
        "episodes_updated": episodes_updated,
        "removed_directories": removed_directories
    }

def revert_renames(done_renames):
    """Rename back, last first, the (old_path, new_path) pairs already applied.
    Returns the paths that could not be renamed back."""
    import os
    not_reverted = []
    for old_path, new_path in reversed(done_renames):
        try:
            os.rename(new_path, old_path)
        except OSError as e:
            not_reverted.append(f"{new_path} ({e})")
    return not_reverted

@app.post("/episodes/rename/")
def rename_episodes(rename_request: EpisodesRenameRequest, session: Session = Depends(get_session)):
    """Rename registered episode files in place and update their file_path.

    The whole batch is validated before anything is touched; a single problem
    refuses the batch and nothing is renamed:
    - 404 if the folder or one of the episodes is not registered;
    - 400 if an episode is listed twice, a new name is not a bare file name or
      has no extension, the current file is missing on disk, the target exists
      on disk (unless it is the same file) or is the registered path of another
      episode, or two renames of the batch produce the same target.
    A target that exists is refused even when it is itself renamed away in the
    same batch (swap or chain): the owner does that in two calls.

    The files are renamed first, then the database is committed. If a rename
    or the commit fails, the files already renamed are renamed back and 500 is
    returned. The UI's "rename by pattern" computes the names client side and
    sends them here in one batch.
    """
    import os
    current_folder = get_folder_data(rename_request.folder_id, session)
    if current_folder == None:
        raise HTTPException(status_code=404, detail=f"Folder {rename_request.folder_id} not found")
    if not rename_request.renames:
        raise HTTPException(status_code=400, detail="renames cannot be empty")

    registered = {current_episode.episode_number: current_episode for current_episode in get_episodes_by_folder_id(current_folder.id, session)}
    registered_paths = {os.path.normcase(os.path.normpath(current_episode.file_path)): current_episode.episode_number for current_episode in registered.values()}

    missing_episodes = []
    problems = []
    plan = []
    seen_episodes = set()
    seen_targets = {}
    for rename in rename_request.renames:
        label = f"episode {rename.episode_number}"
        if rename.episode_number in seen_episodes:
            problems.append(f"{label}: listed more than once")
            continue
        seen_episodes.add(rename.episode_number)

        current_episode = registered.get(rename.episode_number)
        if current_episode == None:
            missing_episodes.append(rename.episode_number)
            continue

        new_file_name = rename.new_file_name
        if new_file_name == None or len(new_file_name.strip()) == 0:
            problems.append(f"{label}: new_file_name cannot be empty")
            continue
        if os.sep in new_file_name or (os.altsep != None and os.altsep in new_file_name) or new_file_name in ('.', '..') or '\0' in new_file_name:
            problems.append(f"{label}: new_file_name must be a file name, not a path")
            continue
        if os.path.splitext(new_file_name)[1] in ('', '.'):
            problems.append(f"{label}: new_file_name must keep a file extension")
            continue

        old_path = current_episode.file_path
        new_path = os.path.join(os.path.dirname(old_path), new_file_name)
        if not os.path.isfile(old_path):
            problems.append(f"{label}: current file {os.path.basename(old_path)} not found on disk")
            continue

        target_key = os.path.normcase(os.path.normpath(new_path))
        if target_key in seen_targets:
            problems.append(f"{label}: same target {new_file_name} as episode {seen_targets[target_key]}")
            continue
        seen_targets[target_key] = rename.episode_number

        if new_path != old_path:
            # samefile covers a case-only rename on a case-insensitive file system.
            if os.path.lexists(new_path) and not (os.path.exists(new_path) and os.path.samefile(old_path, new_path)):
                problems.append(f"{label}: {new_file_name} already exists on disk")
                continue
            if registered_paths.get(target_key, rename.episode_number) != rename.episode_number:
                problems.append(f"{label}: {new_file_name} is the registered file of episode {registered_paths[target_key]}")
                continue
        plan.append((current_episode, old_path, new_path))

    if missing_episodes:
        raise HTTPException(status_code=404, detail=f"Nothing renamed: episode(s) {', '.join(str(n) for n in missing_episodes)} not registered for folder {current_folder.id}" + ("; " + "; ".join(problems) if problems else ""))
    if problems:
        raise HTTPException(status_code=400, detail="Nothing renamed: " + "; ".join(problems))

    done_renames = []
    try:
        for current_episode, old_path, new_path in plan:
            if new_path != old_path:
                os.rename(old_path, new_path)
                done_renames.append((old_path, new_path))
    except OSError as e:
        not_reverted = revert_renames(done_renames)
        raise HTTPException(status_code=500, detail=f"Rename failed, database untouched: {e}. "
                            + (f"Could not rename back: {'; '.join(not_reverted)}" if not_reverted else f"{len(done_renames)} file(s) renamed back"))

    try:
        for current_episode, old_path, new_path in plan:
            current_episode.file_path = new_path
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        not_reverted = revert_renames(done_renames)
        raise HTTPException(status_code=500, detail=f"Database update failed, the renames were reverted: {e}"
                            + (f". Could not rename back: {'; '.join(not_reverted)}" if not_reverted else ""))

    return {
        "message": f"{len(done_renames)} file(s) renamed",
        "folder_id": current_folder.id,
        "renamed": [
            {
                "episode_number": current_episode.episode_number,
                "old_file_name": os.path.basename(old_path),
                "new_file_name": os.path.basename(new_path)
            }
            for current_episode, old_path, new_path in plan
        ]
    }

def holds_only_error_files(directory):
    """True for an existing directory (not a link) holding nothing but regular
    files named `*.error`; an empty directory counts too."""
    import os
    if os.path.islink(directory) or not os.path.isdir(directory):
        return False
    with os.scandir(directory) as entries:
        return all(entry.is_file(follow_symlinks=False) and entry.name.endswith('.error') for entry in entries)

@app.post("/errors/delete/")
def delete_error_file(delete_request: ErrorDeleteRequest, session: Session = Depends(get_session)):
    """Delete an incompatible file, its `<file>.log.error` and its row.

    A file already missing on disk is reported, not refused. Then the file's
    directory is removed when only `.error` files are left in it.
    """
    import os
    import shutil
    incompatible_file = get_incompatible_file_by_path(delete_request.error_file_path, session)
    if incompatible_file == None:
        raise HTTPException(status_code=404, detail=f"{delete_request.error_file_path} not found in incompatible_files")

    file_path = incompatible_file.file_path
    if os.path.lexists(file_path) and not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail=f"{file_path} is not a regular file, nothing deleted")
    if os.path.islink(file_path):
        raise HTTPException(status_code=400, detail=f"{file_path} is a symbolic link, nothing deleted")
    if get_episode_by_path(file_path, session) != None:
        raise HTTPException(status_code=400, detail=f"{file_path} is also a registered episode file, nothing deleted")

    deleted_files = []
    warnings = []
    file_was_missing = not os.path.lexists(file_path)
    if not file_was_missing:
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"{file_path} can't be deleted, database untouched: {e}")
    log_path = file_path + ".log.error"
    if os.path.islink(log_path) or os.path.isfile(log_path):
        try:
            os.remove(log_path)
            deleted_files.append(log_path)
        except OSError as e:
            warnings.append(f"{log_path} not deleted: {e}")

    try:
        delete_incompatible_file(incompatible_file, session)
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Files deleted ({len(deleted_files)}) but the row was kept, call again to finish: {e}")

    removed_directories = []
    file_dir = os.path.dirname(os.path.normpath(file_path))
    if holds_only_error_files(file_dir):
        try:
            shutil.rmtree(file_dir)
            removed_directories.append(file_dir)
        except OSError as e:
            warnings.append(f"{file_dir} not removed: {e}")

    return {
        "message": "Error deleted",
        "error_file_path": file_path,
        "file_was_missing": file_was_missing,
        "deleted_files": deleted_files,
        "removed_directories": removed_directories,
        "warnings": warnings
    }
