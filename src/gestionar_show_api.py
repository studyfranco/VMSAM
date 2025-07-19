from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from pydantic import BaseSettings, BaseModel, Field, Optional
from gestionar_show_model import get_folder_by_path, insert_folder, get_all_regex, insert_regex, get_regex_data, update_regex
from gestionar_show import episode_pattern_insert

class Settings(BaseSettings):
    DATABASE_URL: str

    class Config:
        env_file = ".env"  # Ce nom est utilisé *par défaut* dans uvicorn =--env_file

settings = Settings()

# Initialise la DB
engine = create_engine(settings.DATABASE_URL, echo=True)

def get_session():
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()

class Folder(BaseModel):
    destination_path: str
    original_language: str
    number_cut: int = Field(default=5)
    cut_file_to_get_delay_second_method: float = Field(default=2.5)
    max_episode_number: int = Field(default=12)
    
class Regex(BaseModel):
    regex_pattern: str
    rename_pattern: Optional[str] = None
    weight: int = Field(default=1)
    example_filename: str
    destination_path: str

app = FastAPI(
    title="Gestionar Show API",
    description="API pour la gestion des folders, regex patterns et épisodes",
    version="1.0.0"
)

@app.post("/folders/")
def create_folder(folder_in: Folder, session: Session = Depends(get_session)):
    # Vérifie s’il existe déjà
    existing = get_folder_by_path(folder_in.destination_path, session)
    if existing != None and len(existing):
        existing.max_episode_number = folder_in.max_episode_number
        existing.number_cut = folder_in.number_cut
        existing.cut_file_to_get_delay_second_method = folder_in.cut_file_to_get_delay_second_method
        existing.original_language = folder_in.original_language
        session.commit()
        return {
            "message": "Folder already exists and updated",
            "folder_id": existing.id
        }

    # Création avec valeurs Pydantic (défauts inclus automatiquement)
    new_folder = insert_folder(folder_in.destination_path, folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number, session)

    return {
        "message": "Folder created",
        "folder_id": new_folder.id
    }

def test_regex_rename(regex_data):
    if regex_data.rename_pattern != None and episode_pattern_insert not in regex_data.rename_pattern:
        raise HTTPException(status_code=400, detail=f"Rename pattern must contain the episode pattern: {episode_pattern_insert}")

@app.post("/regex/")
def create_regex(regex_data: Regex, session: Session = Depends(get_session)):
    regex = get_regex_data(regex_data.regex_pattern, session)
    if regex == None or len(regex) == 0:
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
        
        # Vérifier l'existence du dossier via son path
        folder = get_folder_by_path(regex_data.destination_path, session)
        if folder == None or len(folder) == 0:
            raise HTTPException(status_code=400, detail=f"Folder {regex_data.destination_path} not found")

        test_regex_rename(regex_data)

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
            "extracted_episode": int(episode_number)
        }
    else:
        test_regex_rename(regex_data)
        # Si la regex existe déjà, on met à jour les champs
        try:
            update_regex(regex, regex_data.rename_pattern, regex_data.weight, session)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during update of the regex: {e}")
        return {
            "message": "Regex updated",
            "regex_pattern": regex_data.regex_pattern
        }