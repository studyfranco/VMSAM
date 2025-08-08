from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from gestionar_show_model import get_folder_by_path, insert_folder, get_all_regex, insert_regex, get_regex_data, update_regex, get_incrementaller_data,get_all_incrementaller, insert_incrementaller, update_incrementaller, search_like_folder, get_regex_by_folder
from gestionar_show import episode_pattern_insert

class Settings(BaseSettings):
    DATABASE_URL: str

    class Config:
        env_file = ".env"  # Ce nom est utilisé *par défaut* dans uvicorn =--env_file

settings = Settings()

# Initialise la DB
engine = create_engine(settings.DATABASE_URL, echo=False)

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

app = FastAPI(
    title="Gestionar Show API",
    description="API pour la gestion des folders, regex patterns et épisodes",
    version="1.0.0"
)

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
    if (not os.path.isdir(folder_in.destination_path)):
        raise HTTPException(status_code=400, detail="Folder not exist")

    if (not os.access(folder_in.destination_path, os.W_OK)):
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
            "folder id": folder.id
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
def create_regex(incremental_data: Incrementaller, session: Session = Depends(get_session)):
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
    regex_list = get_regex_by_folder(folder_id, session)
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