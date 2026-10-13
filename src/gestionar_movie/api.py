"""Public FastAPI app managing movies and their filename regexes."""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Optional
from time import sleep
from .model import get_movie_by_path, insert_movie, get_all_regex, insert_regex, get_regex_data, search_like_movie, get_regex_by_tmdb_id, get_all_movie, get_movie_data

class Settings(BaseSettings):
    """Database settings read from the environment or `.env`."""
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()

engine = create_engine(settings.DATABASE_URL, echo=False)

def get_session():
    """Yield a database session, committing on success and rolling back on error."""
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

class Movie(BaseModel):
    """Request body to create or update a movie."""
    tmdb_id: str
    destination_path: str
    original_language: str
    number_cut: int = Field(default=10)
    cut_file_to_get_delay_second_method: float = Field(default=2)
    
class Regex(BaseModel):
    """Request body to create or update a movie regex."""
    regex_pattern: str
    rename_pattern: Optional[str] = None
    weight: int = Field(default=1)
    example_filename: str
    tmdb_id: str

app = FastAPI(
    title="Gestionar movie API",
    description="API pour la gestion des films, regex patterns et films",
    version="1.0.0"
)

@app.post("/movies/")
def create_movie(movie_in: Movie, session: Session = Depends(get_session)):
    """Create a movie and its folder, or update it if it already exists."""
    existing = get_movie_data(movie_in.tmdb_id, session)
    if existing == None:
        existing = get_movie_by_path(movie_in.destination_path, session)
    if existing != None:
        existing.tmdb_id = movie_in.tmdb_id
        existing.destination_path = movie_in.destination_path
        existing.number_cut = movie_in.number_cut
        existing.cut_file_to_get_delay_second_method = movie_in.cut_file_to_get_delay_second_method
        existing.original_language = movie_in.original_language
        session.commit()
        return {
            "message": "Movie already exists and updated",
            "tmdb_id": existing.tmdb_id,
            "destination_path": existing.destination_path
        }

    import os
    if os.path.isfile(movie_in.destination_path):
        raise HTTPException(status_code=400, detail="A regular file already exists at this path")
    elif (not os.path.isdir(movie_in.destination_path)):
        try:
            if (not os.makedirs(movie_in.destination_path, exist_ok=True)):
                raise HTTPException(status_code=400, detail="Folder can't be created")
            sleep(1)
            if (not os.path.isdir(movie_in.destination_path)):
                raise HTTPException(status_code=400, detail="Folder can't be created")
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Folder can't be created: {str(e)}")

    elif (not os.access(movie_in.destination_path, os.W_OK)):
        raise HTTPException(status_code=400, detail="Folder not writable")

    new_movie = insert_movie(movie_in.tmdb_id, movie_in.destination_path, movie_in.original_language, movie_in.number_cut, movie_in.cut_file_to_get_delay_second_method, session)

    return {
        "message": "Movie created",
        "tmdb_id": new_movie.tmdb_id,
        "destination_path": new_movie.destination_path
    }

def get_test_movie(regex_data,session):
    """Return the movie referenced by `regex_data`, or raise HTTP 400."""
    movie = get_movie_data(regex_data.tmdb_id, session)
    if movie == None:
        raise HTTPException(status_code=400, detail=f"Movie {regex_data.tmdb_id} not found")
    return movie

@app.post("/regex/")
def create_regex(regex_data: Regex, session: Session = Depends(get_session)):
    """Create a movie regex after checking it matches the example and conflicts with none."""
    import re
    match = re.search(regex_data.regex_pattern, regex_data.example_filename)
    if match == None:
        raise HTTPException(status_code=400, detail="Regex does not match the example filename")
    
    movie = get_test_movie(regex_data,session)

    regex = get_regex_data(regex_data.regex_pattern, session)
    if regex == None:

        # No existing regex may also match the example filename.
        all_regex = get_all_regex(session)
        for regex in all_regex:
            if re.search(regex.regex_pattern, regex_data.example_filename) != None:
                raise HTTPException(status_code=400, detail=f"Conflict with existing regex: `{regex.regex_pattern}`")

        try:
            regex = insert_regex(regex_data.regex_pattern, movie.tmdb_id, regex_data.rename_pattern, regex_data.weight, session)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during insertion of the regex: {e}")

        return {
            "message": "Regex added",
            "regex_pattern": regex.regex_pattern,
            "tmdb_id": regex.tmdb_id
        }
    else:
        try:
            regex.tmdb_id = movie.tmdb_id
            regex.rename_pattern = regex_data.rename_pattern
            regex.weight = regex_data.weight
            session.commit()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error during update of the regex: {e}")
        return {
            "message": "Regex updated",
            "regex_pattern": regex.regex_pattern,
            "tmdb_id": regex.tmdb_id
        }

@app.get("/movies_list/")
def get_movies_list(session: Session = Depends(get_session)):
    """List all movies."""
    movies = get_all_movie(session)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found")

    infos = []
    for movie in movies:
        infos.append({
            "tmdb_id": movie.tmdb_id,
            "destination_path": movie.destination_path,
            "original_language": movie.original_language,
            "number_cut": movie.number_cut,
            "cut_file_to_get_delay_second_method": movie.cut_file_to_get_delay_second_method,
            "max_episode_number": movie.max_episode_number,
            "file": movie.file,
            "file_weight": movie.file_weight
        })
    return {
        "movies": infos
    }

@app.get("/movies_infos/")
def get_movie_info(destination_like: str, session: Session = Depends(get_session)):
    """List movies whose path contains `destination_like`."""
    movies = search_like_movie(destination_like, session)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found matching the criteria")
    
    infos = []
    for movie in movies:
        infos.append({
            "tmdb_id": movie.tmdb_id,
            "destination_path": movie.destination_path,
            "original_language": movie.original_language,
            "number_cut": movie.number_cut,
            "cut_file_to_get_delay_second_method": movie.cut_file_to_get_delay_second_method,
            "max_episode_number": movie.max_episode_number,
            "file": movie.file,
            "file_weight": movie.file_weight
        })
    return {
        "movies": infos
    }

@app.get("/movie_infos_by_tmdb_id/")
def get_movie_by_tmdb_id(tmdb_id: str, session: Session = Depends(get_session)):
    """Return one movie by TMDB id."""
    movie = get_movie_data(tmdb_id, session)
    if not movie:
        raise HTTPException(status_code=404, detail="No movie found matching the criteria")
    
    return {
        "movie": {
            "tmdb_id": movie.tmdb_id,
            "destination_path": movie.destination_path,
            "original_language": movie.original_language,
            "number_cut": movie.number_cut,
            "cut_file_to_get_delay_second_method": movie.cut_file_to_get_delay_second_method,
            "max_episode_number": movie.max_episode_number,
            "file": movie.file,
            "file_weight": movie.file_weight
        }
    }
    
@app.get("/regex_tmdb_id/")
def get_regex_by_tmdb_id(tmdb_id: str, session: Session = Depends(get_session)):
    """List the regexes of one movie."""
    regex_list = get_regex_by_tmdb_id(tmdb_id, session)
    if not regex_list:
        raise HTTPException(status_code=404, detail="No regex found for this movie")
    
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