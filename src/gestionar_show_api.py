from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from pydantic import BaseSettings, BaseModel, Field
from gestionar_show_model import get_folder_by_path, insert_folder

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

class FolderCreate(BaseModel):
    destination_path: str
    original_language: str
    number_cut: int = Field(default=5)
    cut_file_to_get_delay_second_method: float = Field(default=2.5)
    max_episode_number: int = Field(default=12)

app = FastAPI(
    title="Gestionar Show API",
    description="API pour la gestion des folders, regex patterns et épisodes",
    version="1.0.0"
)

@app.post("/folders/")
def create_folder(folder_in: FolderCreate, session: Session = Depends(get_session)):
    # Vérifie s’il existe déjà
    existing = get_folder_by_path(folder_in.destination_path, session)
    if existing:
        return {
            "message": "Folder already exists",
            "folder_id": existing.id
        }

    # Création avec valeurs Pydantic (défauts inclus automatiquement)
    insert_folder(folder_in.destination_path, folder_in.original_language, folder_in.number_cut, folder_in.cut_file_to_get_delay_second_method, folder_in.max_episode_number, session)
    new_folder = get_folder_by_path(folder_in.destination_path, session)

    return {
        "message": "Folder created",
        "folder_id": new_folder.id
    }