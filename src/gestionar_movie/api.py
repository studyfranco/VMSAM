"""Public API of the movie side (0.0.0.0:8081, several workers, stateless).

A movie is any single file to keep: a film or a standalone show episode, named by
its unique `title`. A regex links downloaded files to a movie and names the kept
file: `<destination_path>/<rename_pattern with {<suffix>} replaced by the suffix>`,
or the file's own name when the regex has no rename pattern.

Must not import the merge engine (mergeVideo, audioCorrelation, numpy), directly
or transitively: merges run on the internal movie instance, reached through
internal_client. Nothing from gestionar_show is imported either; the show tables
read for the cross-checks are read with plain SQL.
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import json
import os
import re
import shutil
import urllib.error
import tools
from gestionar_common.database import get_connect_args
from gestionar_common.settings import Settings
from .model import (get_all_movies, get_movie_data, get_movie_by_title, get_movie_by_file_path, search_like_movie,
                    search_title_like_movie, clean_metadata_ids, test_conformity_movie_param, insert_movie, update_movie_parameters,
                    set_movie_file, delete_movie, build_file_name, suffix_pattern_insert, get_regex_data, get_all_regex,
                    get_regex_by_movie_id, insert_regex, update_regex, replace_regex, delete_regex, get_all_incompatible_files,
                    get_incompatible_files_by_movie_id, get_incompatible_file_by_path, delete_incompatible_file)
from . import internal_client

engine = None
# Read once per process at startup from two editable JSON files:
# providers: provider -> {app: tag}; brackets: app -> {"begin": ..., "end": ...}.
metadata_providers = {}
metadata_brackets = {}
providers_file = os.environ.get("VMSAM_METADATA_PROVIDERS", "").strip() or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata_providers.json")
brackets_file = os.environ.get("VMSAM_METADATA_BRACKETS", "").strip() or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metadata_brackets.json")

def get_session():
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
    finally:
        session.close()

class Movie(BaseModel):
    # The key: the movie with this title is updated, else one is created.
    title: str
    destination_path: str
    suffix: str = Field(default='')
    original_language: str
    number_cut: int = Field(default=10)
    cut_file_to_get_delay_second_method: float = Field(default=2)
    # Full data each time: absent or null means no id.
    metadata_ids: Optional[Dict[str, str]] = None

class MovieDeleteRequest(BaseModel):
    movie_id: int

class MovieMove(BaseModel):
    movie_id: int
    new_destination_path: str

class Regex(BaseModel):
    regex_pattern: str
    movie_id: int
    rename_pattern: Optional[str] = None
    weight: int = Field(default=1)
    example_filename: str

class RegexReplace(BaseModel):
    regex_pattern: str
    new_regex_pattern: str
    example_filename: str
    # Absent keeps the current rename pattern; an explicit null removes it.
    rename_pattern: Optional[str] = None
    weight: Optional[int] = None
    movie_id: Optional[int] = None

class RegexDeleteRequest(BaseModel):
    regex_pattern: str

class IndexMovieRequest(BaseModel):
    movie_id: int

class FolderMove(BaseModel):
    destination_path: str
    new_destination_path: str

class ErrorDeleteRequest(BaseModel):
    error_file_path: str

class FusionRequest(BaseModel):
    error_file_path: str

class MetadataTagsRequest(BaseModel):
    apps: List[str]
    ids: Dict[str, str]

app = FastAPI(
    title="Gestionar Movie API",
    description="API for movies (films and standalone show episodes), their regex patterns and their rejected files",
    version="1.0.0"
)

def load_metadata_files():
    """Read and check both metadata files. A malformed file raises a ValueError
    naming the file, so the API refuses to start instead of serving wrong tags."""
    contents = []
    for file_path, label in ((providers_file, "providers"), (brackets_file, "brackets")):
        try:
            with open(file_path, encoding="utf-8") as config_file:
                content = json.load(config_file)
        except OSError as e:
            raise ValueError(f"Metadata {label} file {file_path} can't be read: {e}")
        except ValueError as e:
            raise ValueError(f"Metadata {label} file {file_path} is not valid JSON: {e}")
        if not isinstance(content, dict):
            raise ValueError(f"Metadata {label} file {file_path} must hold a JSON object")
        contents.append(content)
    providers, brackets = contents

    for app_name, app_brackets in brackets.items():
        if len(app_name.strip()) == 0:
            raise ValueError(f"Metadata brackets file {brackets_file}: an app name is empty")
        if not isinstance(app_brackets, dict) or not isinstance(app_brackets.get("begin"), str) or not isinstance(app_brackets.get("end"), str):
            raise ValueError(f"Metadata brackets file {brackets_file}: app `{app_name}` must be an object with string `begin` and `end`")
    for provider, app_tags in providers.items():
        if len(provider.strip()) == 0:
            raise ValueError(f"Metadata providers file {providers_file}: a provider name is empty")
        if not isinstance(app_tags, dict):
            raise ValueError(f"Metadata providers file {providers_file}: provider `{provider}` must be an object app -> tag")
        for app_name, tag in app_tags.items():
            if not isinstance(tag, str) or len(tag.strip()) == 0:
                raise ValueError(f"Metadata providers file {providers_file}: provider `{provider}` has an empty tag for app `{app_name}`")
            if app_name not in brackets:
                raise ValueError(f"Metadata providers file {providers_file}: provider `{provider}` names app `{app_name}`, which has no brackets in {brackets_file}")
    return providers, brackets

@app.on_event("startup")
def on_startup():
    global engine, metadata_providers, metadata_brackets
    metadata_providers, metadata_brackets = load_metadata_files()
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL, echo=False, connect_args=get_connect_args(settings.DATABASE_URL), pool_pre_ping=True)

def test_absolute_path(path, field_name):
    if path == None or len(path.strip()) == 0:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if '\0' in path or not os.path.isabs(path):
        raise HTTPException(status_code=400, detail=f"{field_name} must be an absolute path")
    return os.path.normpath(path)

def is_inside(path, bound):
    try:
        return os.path.commonpath([path, bound]) == bound
    except ValueError:
        return False

def is_video_file_name(file_name):
    """Video extension, case-insensitive. A merge in progress (`<name>.mkv.tmp`) is not one."""
    return file_name.lower().endswith(tools.video_extensions)

def search_regex(regex_pattern, file_name):
    """re.search, a stored pattern that does not compile matching nothing."""
    try:
        return re.search(regex_pattern, file_name)
    except re.error:
        return None

def get_show_folder_paths(session):
    """destination_path of every show folder, read with plain SQL (empty without show tables)."""
    if not inspect(session.connection()).has_table("folders"):
        return []
    return [os.path.normpath(row[0]) for row in session.execute(text("SELECT destination_path FROM folders")).all() if row[0]]

def test_regex(regex_pattern, example_filename, excluded_pattern, session):
    """The regex must compile and match the example; no other movie regex may
    match the example, and no show regex may take it as an episode: the daemon
    runs the movies first, so a movie regex catching an episode would steal it."""
    try:
        match = re.search(regex_pattern, example_filename)
    except re.error as e:
        raise HTTPException(status_code=400, detail=f"Regex does not compile: {e}")
    if match == None:
        raise HTTPException(status_code=400, detail="Regex does not match the example filename")

    for regex in get_all_regex(session):
        if regex.regex_pattern not in (regex_pattern, excluded_pattern) and search_regex(regex.regex_pattern, example_filename) != None:
            raise HTTPException(status_code=400, detail=f"Conflict with existing movie regex: `{regex.regex_pattern}` of movie {regex.movie_id}")

    if not inspect(session.connection()).has_table("regex_patterns"):
        return
    if inspect(session.connection()).has_table("folders"):
        query = "SELECT r.regex_pattern, r.folder_id, f.destination_path FROM regex_patterns r LEFT JOIN folders f ON f.id = r.folder_id"
    else:
        query = "SELECT regex_pattern, folder_id, NULL FROM regex_patterns"
    for show_regex_pattern, folder_id, destination_path in session.execute(text(query)).all():
        # The show daemon's contract: the `episode` group, else the first
        # group, must be a number >= 1 for the file to be taken as an episode.
        show_match = search_regex(show_regex_pattern, example_filename)
        if show_match == None:
            continue
        if 'episode' in show_match.groupdict():
            episode_number = show_match.group('episode')
        elif show_match.groups():
            episode_number = show_match.group(1)
        else:
            continue
        if episode_number != None and episode_number.isdigit() and int(episode_number) > 0:
            raise HTTPException(status_code=400, detail=f"Conflict with show regex `{show_regex_pattern}` of show folder {folder_id}"
                                + (f" ({destination_path})" if destination_path else "")
                                + ": it takes the example as an episode, and the movies run first so this regex would steal it")

def movie_to_infos(current_movie):
    return {
        "movie_id": current_movie.id,
        "title": current_movie.title,
        "destination_path": current_movie.destination_path,
        "suffix": current_movie.suffix,
        "original_language": current_movie.original_language,
        "number_cut": current_movie.number_cut,
        "cut_file_to_get_delay_second_method": current_movie.cut_file_to_get_delay_second_method,
        "metadata_ids": current_movie.metadata_ids or {},
        "file_path": current_movie.file_path,
        "file_weight": current_movie.file_weight
    }

def make_destination_directory(destination_path):
    """Create the destination directory if needed. Returns True when it was created."""
    if os.path.lexists(destination_path) and not os.path.isdir(destination_path):
        raise HTTPException(status_code=400, detail="A regular file already exists at this path")
    if os.path.isdir(destination_path):
        if not os.access(destination_path, os.W_OK):
            raise HTTPException(status_code=400, detail="Folder not writable")
        return False
    try:
        os.makedirs(destination_path, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Folder can't be created: {e}")
    return True

def remove_empty_directory(directory):
    """Undo a directory creation; a directory that is not empty any more stays."""
    try:
        os.rmdir(directory)
    except OSError:
        pass

@app.post("/movie/")
def create_movie(movie_in: Movie, session: Session = Depends(get_session)):
    """Create or update a movie, keyed by its title, with the full data each time.

    Database only, like the show's POST /folder/: no file is moved or renamed.
    A new suffix only names the next file integrated or merged. The destination
    directory is created when missing. The destination of a movie that has a
    file is changed with POST /movie/move/, which moves the file.
    """
    destination_path = test_absolute_path(movie_in.destination_path, "destination_path")
    try:
        title = test_conformity_movie_param(movie_in.title, destination_path, movie_in.suffix, movie_in.original_language, movie_in.number_cut,
                                            movie_in.cut_file_to_get_delay_second_method, movie_in.metadata_ids)[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    current_movie = get_movie_by_title(title, session)
    if current_movie != None:
        destination_changed = os.path.normpath(current_movie.destination_path) != destination_path
        if destination_changed and current_movie.file_path != None:
            raise HTTPException(status_code=400, detail=f"Movie {current_movie.id} has a file: use /movie/move/ to change its destination_path")
        if destination_changed:
            make_destination_directory(destination_path)
        try:
            update_movie_parameters(current_movie, destination_path, movie_in.suffix, movie_in.original_language, movie_in.number_cut,
                                    movie_in.cut_file_to_get_delay_second_method, movie_in.metadata_ids, session)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        infos = movie_to_infos(current_movie)
        infos["message"] = "Movie updated"
        return infos

    make_destination_directory(destination_path)
    try:
        new_movie = insert_movie(title, destination_path, movie_in.suffix, movie_in.original_language, movie_in.number_cut,
                                 movie_in.cut_file_to_get_delay_second_method, movie_in.metadata_ids, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        # Two concurrent creations of the same title: the unique constraint wins.
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Movie not created, it conflicts with an existing one: {e.orig}")
    infos = movie_to_infos(new_movie)
    infos["message"] = "Movie created"
    return infos

@app.post("/movie/move/")
def move_movie(move_request: MovieMove, session: Session = Depends(get_session)):
    """Change the destination directory of one movie, moving its file.

    The file keeps its name. The new directory is created if needed; the move
    is refused when the target file exists on disk or is another movie's file.
    The file is moved first, then the database is committed; a failed commit
    moves it back. Then the old directory is removed when it is left empty and
    is not a mount point, another movie's destination or a show folder (one
    level, like /folders/move/).
    """
    current_movie = get_movie_data(move_request.movie_id, session)
    if current_movie == None:
        raise HTTPException(status_code=404, detail=f"Movie {move_request.movie_id} not found")
    new_path = test_absolute_path(move_request.new_destination_path, "new_destination_path")
    old_path = os.path.normpath(current_movie.destination_path)
    if os.path.normcase(old_path) == os.path.normcase(new_path):
        raise HTTPException(status_code=400, detail="new_destination_path is the current destination of the movie")

    old_file_path = current_movie.file_path
    new_file_path = None
    if old_file_path != None:
        if not os.path.isfile(old_file_path):
            raise HTTPException(status_code=400, detail=f"The movie file {os.path.basename(old_file_path)} is not found on disk, nothing moved")
        new_file_path = os.path.join(new_path, os.path.basename(old_file_path))
        if os.path.lexists(new_file_path):
            raise HTTPException(status_code=400, detail=f"{os.path.basename(new_file_path)} already exists in {new_path}, nothing moved")
        other_movie = get_movie_by_file_path(new_file_path, session)
        if other_movie != None:
            raise HTTPException(status_code=400, detail=f"{os.path.basename(new_file_path)} is the file of movie {other_movie.id}, nothing moved")

    directory_created = make_destination_directory(new_path)
    if new_file_path != None:
        try:
            shutil.move(old_file_path, new_file_path)
        except OSError as e:
            if directory_created:
                remove_empty_directory(new_path)
            raise HTTPException(status_code=500, detail=f"File move failed, nothing changed: {e}")

    try:
        current_movie.destination_path = new_path
        current_movie.file_path = new_file_path
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        detail = f"Database update failed: {e}"
        if new_file_path != None:
            try:
                shutil.move(new_file_path, old_file_path)
                detail += ". The file was moved back"
            except OSError as move_error:
                detail += f". Could not move the file back: {move_error}"
        if directory_created:
            remove_empty_directory(new_path)
        raise HTTPException(status_code=500, detail=detail)

    removed_directories = []
    kept_directories = set(os.path.normpath(other_movie.destination_path) for other_movie in get_all_movies(session)) | set(get_show_folder_paths(session))
    if (old_path not in kept_directories and os.path.isdir(old_path) and not os.path.islink(old_path)
            and not os.path.ismount(old_path) and not os.listdir(old_path)):
        try:
            os.rmdir(old_path)
            removed_directories.append(old_path)
        except OSError:
            pass

    infos = movie_to_infos(current_movie)
    infos["message"] = "Movie moved"
    infos["file_moved"] = new_file_path != None
    infos["removed_directories"] = removed_directories
    return infos

@app.get("/movie/")
def get_movie(movie_id: Optional[int] = None, title: Optional[str] = None, session: Session = Depends(get_session)):
    """One movie, by movie_id or by its exact title (exactly one of them)."""
    if (movie_id == None) == (title == None):
        raise HTTPException(status_code=400, detail="Give exactly one of movie_id or title")
    if movie_id != None:
        current_movie = get_movie_data(movie_id, session)
        if current_movie == None:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    else:
        current_movie = get_movie_by_title(title, session)
        if current_movie == None:
            raise HTTPException(status_code=404, detail=f"Movie `{title}` not found")
    return movie_to_infos(current_movie)

@app.get("/movies/")
def get_movies_list(session: Session = Depends(get_session)):
    movies = get_all_movies(session)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found")
    return {"movies": [movie_to_infos(current_movie) for current_movie in movies]}

@app.get("/movies/destination_like/")
def get_movies_destination_like(destination_like: str, session: Session = Depends(get_session)):
    """Movies whose destination_path contains destination_like (SQL LIKE)."""
    movies = search_like_movie(destination_like, session)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found matching the criteria")
    return {"movies": [movie_to_infos(current_movie) for current_movie in movies]}

@app.get("/movies/title_like/")
def get_movies_title_like(title_like: str, session: Session = Depends(get_session)):
    """Movies whose title contains title_like, case-insensitive."""
    movies = search_title_like_movie(title_like, session)
    if not movies:
        raise HTTPException(status_code=404, detail="No movies found matching the criteria")
    return {"movies": [movie_to_infos(current_movie) for current_movie in movies]}

@app.post("/movies/delete/")
def delete_movie_entry(delete_request: MovieDeleteRequest, session: Session = Depends(get_session)):
    """Delete a movie that has no regex and no incompatible file. Its file, if
    any, stays on disk as an ordinary unregistered file."""
    current_movie = get_movie_data(delete_request.movie_id, session)
    if current_movie == None:
        raise HTTPException(status_code=404, detail=f"Movie {delete_request.movie_id} not found")
    regex_count = len(get_regex_by_movie_id(current_movie.id, session))
    incompatible_count = len(get_incompatible_files_by_movie_id(current_movie.id, session))
    if regex_count or incompatible_count:
        raise HTTPException(status_code=400, detail=f"Movie {current_movie.id} still has {regex_count} regex and {incompatible_count} incompatible file(s); delete them first")
    file_path = current_movie.file_path
    delete_movie(current_movie, session)
    return {
        "message": "Movie deleted" + (", its file was left on disk" if file_path != None else ""),
        "movie_id": delete_request.movie_id,
        "file_left_on_disk": file_path
    }

def test_regex_rename(rename_pattern):
    """Same rule as the show's: a rename pattern, when given, holds the placeholder."""
    if rename_pattern != None and suffix_pattern_insert not in rename_pattern:
        raise HTTPException(status_code=400, detail=f"Rename pattern must contain the suffix pattern: {suffix_pattern_insert}")

def get_final_file_name(rename_pattern, current_movie, file_name):
    """The name a file caught by the regex gets in the movie's directory."""
    try:
        return build_file_name(rename_pattern, current_movie.suffix, file_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/regex/")
def create_regex(regex_data: Regex, session: Session = Depends(get_session)):
    """Link a regex to a movie (checked by test_regex). An existing pattern is
    updated (movie, rename pattern and weight), like the show's /regex/."""
    if regex_data.weight < 1:
        raise HTTPException(status_code=400, detail="weight must be at least 1")
    rename_pattern = regex_data.rename_pattern if regex_data.rename_pattern else None
    test_regex_rename(rename_pattern)
    current_movie = get_movie_data(regex_data.movie_id, session)
    if current_movie == None:
        raise HTTPException(status_code=404, detail=f"Movie {regex_data.movie_id} not found")
    test_regex(regex_data.regex_pattern, regex_data.example_filename, None, session)
    final_file_name = get_final_file_name(rename_pattern, current_movie, regex_data.example_filename)

    existing = get_regex_data(regex_data.regex_pattern, session)
    try:
        if existing == None:
            insert_regex(regex_data.regex_pattern, current_movie.id, rename_pattern, regex_data.weight, session)
            message = "Regex added"
        else:
            update_regex(existing, current_movie.id, rename_pattern, regex_data.weight, session)
            message = "Regex updated"
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IntegrityError as e:
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Regex not saved: {e.orig}")

    return {
        "message": message,
        "regex_pattern": regex_data.regex_pattern,
        "movie_id": current_movie.id,
        "rename_pattern": rename_pattern,
        "weight": regex_data.weight,
        "final_file_name": final_file_name
    }

@app.post("/regex/replace/")
def replace_regex_pattern(replace_data: RegexReplace, session: Session = Depends(get_session)):
    """Edit an existing regex: pattern, rename pattern, weight and movie.

    Absent fields keep their current value; rename_pattern set to null removes
    the rename pattern. The result is checked like POST /regex/, the replaced
    regex excluded from the conflicts; the new pattern must differ and not
    exist yet. The pattern is the primary key: the new row is added and the
    old one deleted in a single commit.
    """
    current_regex = get_regex_data(replace_data.regex_pattern, session)
    if current_regex == None:
        raise HTTPException(status_code=404, detail=f"Regex `{replace_data.regex_pattern}` not found")
    if len(replace_data.new_regex_pattern) == 0:
        raise HTTPException(status_code=400, detail="new_regex_pattern cannot be empty")
    if replace_data.new_regex_pattern == replace_data.regex_pattern:
        raise HTTPException(status_code=400, detail="new_regex_pattern is the current pattern")
    if get_regex_data(replace_data.new_regex_pattern, session) != None:
        raise HTTPException(status_code=400, detail=f"Regex `{replace_data.new_regex_pattern}` already exists")

    movie_id = replace_data.movie_id if replace_data.movie_id != None else current_regex.movie_id
    rename_pattern = replace_data.rename_pattern if "rename_pattern" in replace_data.model_fields_set else current_regex.rename_pattern
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    weight = replace_data.weight if replace_data.weight != None else current_regex.weight
    if weight < 1:
        raise HTTPException(status_code=400, detail="weight must be at least 1")
    test_regex_rename(rename_pattern)

    target_movie = get_movie_data(movie_id, session)
    if target_movie == None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    test_regex(replace_data.new_regex_pattern, replace_data.example_filename, replace_data.regex_pattern, session)
    final_file_name = get_final_file_name(rename_pattern, target_movie, replace_data.example_filename)

    previous_regex_pattern = current_regex.regex_pattern
    try:
        new_regex = replace_regex(current_regex, replace_data.new_regex_pattern, movie_id, rename_pattern, weight, session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database update failed, regex unchanged: {e}")

    return {
        "message": "Regex replaced",
        "regex_pattern": new_regex.regex_pattern,
        "previous_regex_pattern": previous_regex_pattern,
        "movie_id": new_regex.movie_id,
        "rename_pattern": new_regex.rename_pattern,
        "weight": new_regex.weight,
        "final_file_name": final_file_name
    }

@app.post("/regex/delete/")
def delete_regex_pattern(delete_request: RegexDeleteRequest, session: Session = Depends(get_session)):
    current_regex = get_regex_data(delete_request.regex_pattern, session)
    if current_regex == None:
        raise HTTPException(status_code=404, detail=f"Regex `{delete_request.regex_pattern}` not found")
    try:
        delete_regex(current_regex, session)
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database delete failed, regex kept: {e}")
    return {"message": "Regex deleted", "regex_pattern": delete_request.regex_pattern}

@app.get("/regex_movie/")
def get_regex_by_movie(movie_id: int, session: Session = Depends(get_session)):
    if get_movie_data(movie_id, session) == None:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    regex_list = get_regex_by_movie_id(movie_id, session)
    if not regex_list:
        raise HTTPException(status_code=404, detail="No regex found for this movie")
    return {"regex_patterns": [{"regex_pattern": regex.regex_pattern, "rename_pattern": regex.rename_pattern, "weight": regex.weight} for regex in regex_list]}

@app.get("/regex_list/")
def get_regex_list(session: Session = Depends(get_session)):
    """Every movie regex, heaviest first, with its movie. An empty list is a valid answer."""
    return {
        "regex_patterns": [
            {
                "regex_pattern": regex.regex_pattern,
                "rename_pattern": regex.rename_pattern,
                "weight": regex.weight,
                "movie_id": regex.movie_id,
                "title": regex.movie.title,
                "suffix": regex.movie.suffix,
                "destination_path": regex.movie.destination_path
            }
            for regex in get_all_regex(session)
        ]
    }

@app.post("/index_movie/")
def index_movie(index_request: IndexMovieRequest, session: Session = Depends(get_session)):
    """Register, for one movie, a file already present in its destination directory.

    Candidates: the regular, non-hidden video files of the directory that are no
    movie's file and that one of the movie's regexes matches. At most one is
    taken (heaviest regex, then largest file), renamed with its regex's rename
    pattern and the movie's suffix, or kept under its own name when the regex
    has none (refused if the target exists), and registered with the regex
    weight; the others are listed in "skipped" with a reason. Nothing is merged
    or deleted.
    """
    current_movie = get_movie_data(index_request.movie_id, session)
    if current_movie == None:
        raise HTTPException(status_code=404, detail=f"Movie {index_request.movie_id} not found")
    destination_path = current_movie.destination_path
    if not os.path.isdir(destination_path):
        raise HTTPException(status_code=400, detail="Movie destination path is not a directory on this instance")
    regex_list = get_regex_by_movie_id(current_movie.id, session)
    if not regex_list:
        raise HTTPException(status_code=404, detail="No regex found for this movie")

    known_paths = set(os.path.normpath(other_movie.file_path) for other_movie in get_all_movies(session) if other_movie.file_path != None)
    candidates = []
    skipped = []
    unmatched = []
    already_indexed = 0
    for file_name in sorted(os.listdir(destination_path)):
        file_path = os.path.join(destination_path, file_name)
        if file_name.startswith('.') or not os.path.isfile(file_path):
            continue
        if os.path.normpath(file_path) in known_paths:
            already_indexed += 1
            continue
        matched_regex = next((regex for regex in regex_list if search_regex(regex.regex_pattern, file_name) != None), None)
        if matched_regex == None:
            unmatched.append(file_name)
        elif not is_video_file_name(file_name):
            skipped.append({"file_name": file_name, "reason": "not_a_video_file", "regex_pattern": matched_regex.regex_pattern})
        else:
            candidates.append((file_name, file_path, matched_regex, os.path.getsize(file_path)))

    indexed = []
    if candidates and current_movie.file_path != None:
        # Indexing never merges: with a registered file, every candidate is left untouched.
        for file_name, file_path, matched_regex, file_size in candidates:
            skipped.append({"file_name": file_name, "reason": "movie_already_has_a_file", "registered_file_path": current_movie.file_path})
    elif candidates:
        candidates.sort(key=lambda candidate: (candidate[2].weight, candidate[3]), reverse=True)
        file_name, file_path, matched_regex, file_size = candidates[0]
        for other_name, other_path, other_regex, other_size in candidates[1:]:
            skipped.append({"file_name": other_name, "reason": "not_selected", "selected_file_name": file_name,
                            "regex_pattern": other_regex.regex_pattern, "file_weight": other_regex.weight, "file_size": other_size})

        reason = None
        try:
            new_file_name = build_file_name(matched_regex.rename_pattern, current_movie.suffix, file_name)
        except ValueError as e:
            new_file_name = file_name
            reason = "invalid_rename_pattern"
            skipped.append({"file_name": file_name, "reason": reason, "regex_pattern": matched_regex.regex_pattern, "error": str(e)})
        new_file_path = os.path.join(destination_path, new_file_name)
        if reason == None and new_file_path != file_path:
            if os.path.lexists(new_file_path):
                reason = "target_exists"
            elif get_movie_by_file_path(new_file_path, session) != None:
                reason = "target_registered"
            else:
                try:
                    os.rename(file_path, new_file_path)
                except OSError as e:
                    skipped.append({"file_name": file_name, "reason": "rename_failed", "new_file_name": new_file_name, "error": str(e)})
                    reason = "rename_failed"
        if reason in ("target_exists", "target_registered"):
            skipped.append({"file_name": file_name, "reason": reason, "new_file_name": new_file_name})
        elif reason == None:
            try:
                set_movie_file(current_movie, new_file_path, matched_regex.weight, session)
            except SQLAlchemyError as e:
                session.rollback()
                raise HTTPException(status_code=500, detail=f"File {new_file_name} renamed but not registered: {e}")
            indexed.append({"file_name": file_name, "new_file_name": new_file_name, "file_weight": matched_regex.weight, "regex_pattern": matched_regex.regex_pattern})

    return {
        "message": f"{len(indexed)} file(s) indexed",
        "movie_id": current_movie.id,
        "destination_path": destination_path,
        "indexed": indexed,
        "skipped": skipped,
        "unmatched": unmatched,
        "already_indexed": already_indexed
    }

@app.get("/folders/")
def get_folders_list(session: Session = Depends(get_session)):
    """Distinct destination directories with their number of movies."""
    movies_count = {}
    for current_movie in get_all_movies(session):
        movies_count[current_movie.destination_path] = movies_count.get(current_movie.destination_path, 0) + 1
    if not movies_count:
        raise HTTPException(status_code=404, detail="No folders found")
    return {"folders": [{"destination_path": path, "movies_count": count} for path, count in movies_count.items()]}

@app.post("/folders/move/")
def move_folder(move_request: FolderMove, session: Session = Depends(get_session)):
    """Move (or rename) a movie destination directory on disk and in the database.

    Every movie whose destination is the directory or lies inside it is
    rewritten (destination_path and file_path). Refused when a path is not
    absolute or both are the same, no movie lies there (404), the new path
    already holds a movie or is inside the old one, the directory is or holds
    a show folder (move it from the show side), the new path exists on disk or
    the old one is not a directory. The directory is moved first, then the
    database is committed; a failed commit moves it back. No lock is taken, as
    on the show side: a move is an administrative action done while idle.

    Then, like the show's move, the parent of the old path is removed when it
    is empty, not a mount point, and not a movie destination or show folder.
    """
    old_path = test_absolute_path(move_request.destination_path, "destination_path")
    new_path = test_absolute_path(move_request.new_destination_path, "new_destination_path")
    if os.path.normcase(old_path) == os.path.normcase(new_path):
        raise HTTPException(status_code=400, detail="new_destination_path is the current path")

    all_movies = get_all_movies(session)
    moved_movies = [current_movie for current_movie in all_movies if is_inside(os.path.normpath(current_movie.destination_path), old_path)]
    if not moved_movies:
        raise HTTPException(status_code=404, detail=f"No movie registered in {old_path}")
    for current_movie in all_movies:
        if current_movie not in moved_movies and is_inside(os.path.normpath(current_movie.destination_path), new_path):
            raise HTTPException(status_code=400, detail=f"{new_path} already holds movie {current_movie.id}")
    if is_inside(new_path, old_path):
        raise HTTPException(status_code=400, detail="A folder cannot be moved inside itself")
    show_folder_paths = get_show_folder_paths(session)
    for show_folder_path in show_folder_paths:
        if is_inside(show_folder_path, old_path):
            raise HTTPException(status_code=400, detail=f"{old_path} is or contains the show folder {show_folder_path}, move it from the show side")
    if os.path.lexists(new_path):
        raise HTTPException(status_code=400, detail=f"{new_path} already exists on disk")
    if not os.path.isdir(old_path):
        raise HTTPException(status_code=400, detail="Current path is not a directory on this instance, nothing moved")

    try:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, new_path)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Folder can't be moved, database untouched: {e}")

    old_prefix = os.path.join(old_path, "")
    files_updated = 0
    try:
        for current_movie in moved_movies:
            current_destination = os.path.normpath(current_movie.destination_path)
            current_movie.destination_path = new_path if current_destination == old_path else os.path.join(new_path, current_destination[len(old_prefix):])
            if current_movie.file_path != None and os.path.normpath(current_movie.file_path).startswith(old_prefix):
                current_movie.file_path = os.path.join(new_path, os.path.normpath(current_movie.file_path)[len(old_prefix):])
                files_updated += 1
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        shutil.move(new_path, old_path)
        raise HTTPException(status_code=500, detail=f"Database update failed, the move was reverted: {e}")

    removed_folder = os.path.dirname(old_path)
    removed_directories = []
    kept_directories = set(os.path.normpath(current_movie.destination_path) for current_movie in all_movies) | set(show_folder_paths)
    if removed_folder not in kept_directories and not os.path.ismount(removed_folder) and not os.listdir(removed_folder):
        try:
            os.rmdir(removed_folder)
            removed_directories.append(removed_folder)
        except OSError:
            pass

    return {
        "message": "Folder moved",
        "old_destination_path": old_path,
        "new_destination_path": new_path,
        "movies_updated": [current_movie.id for current_movie in moved_movies],
        "files_updated": files_updated,
        "removed_directories": removed_directories
    }

@app.get("/errors")
def get_incompatible_files_list(session: Session = Depends(get_session)):
    """Files rejected by the movie pipeline, with their movie."""
    return {
        "incompatible_files": [
            {
                "id": incompatible_file.id,
                "movie_id": incompatible_file.movie_id,
                "title": incompatible_file.movie.title,
                "suffix": incompatible_file.movie.suffix,
                "destination_path": incompatible_file.movie.destination_path,
                "file_path": incompatible_file.file_path,
                "file_weight": incompatible_file.file_weight
            }
            for incompatible_file in get_all_incompatible_files(session)
        ]
    }

def holds_only_error_files(directory):
    """True for an existing directory (not a link) holding nothing but regular
    files named `*.error`; an empty directory counts too."""
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
    incompatible_file = get_incompatible_file_by_path(delete_request.error_file_path, session)
    if incompatible_file == None:
        raise HTTPException(status_code=404, detail=f"{delete_request.error_file_path} not found in movie_incompatible_files")

    file_path = incompatible_file.file_path
    if os.path.lexists(file_path) and not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail=f"{file_path} is not a regular file, nothing deleted")
    if os.path.islink(file_path):
        raise HTTPException(status_code=400, detail=f"{file_path} is a symbolic link, nothing deleted")
    if get_movie_by_file_path(file_path, session) != None:
        raise HTTPException(status_code=400, detail=f"{file_path} is also a registered movie file, nothing deleted")

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

@app.get("/fusion")
def get_fusion_queue_status():
    """Queue state, relayed from the internal movie instance."""
    try:
        status_code, payload = internal_client.call_internal_api("GET", "/internal/fusion")
    except (urllib.error.URLError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"Internal movie fusion worker unreachable: {e}")
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=payload.get("detail", "Internal movie fusion worker error"))
    return payload

@app.post("/fusion")
def create_fusion_job(fusion_request: FusionRequest, session: Session = Depends(get_session)):
    """Validate a merge request, then hand it to the internal movie worker."""
    incompatible_file = get_incompatible_file_by_path(fusion_request.error_file_path, session)
    if incompatible_file == None:
        raise HTTPException(status_code=404, detail=f"{fusion_request.error_file_path} not found in movie_incompatible_files")
    master_movie = incompatible_file.movie
    if master_movie.file_path == None:
        raise HTTPException(status_code=404, detail=f"Movie {master_movie.id} has no registered file, nothing to merge with")

    try:
        status_code, payload = internal_client.call_internal_api(
            "POST", "/internal/fusion", payload={"error_file_path": fusion_request.error_file_path}
        )
    except (urllib.error.URLError, OSError) as e:
        raise HTTPException(status_code=503, detail=f"Internal movie fusion worker unreachable: {e}")
    if status_code >= 400:
        raise HTTPException(status_code=status_code, detail=payload.get("detail", "Internal movie fusion worker error"))

    payload["master_file_path"] = master_movie.file_path
    return payload

@app.get("/health")
def get_health():
    """Deployment check: version, mode, and the internal movie instance state."""
    status = "ok"
    is_running = False
    # None means "the internal instance did not tell us", not False.
    fusion_enabled = None
    queue_length = None
    internal_status = None
    try:
        status_code, payload = internal_client.call_internal_api("GET", "/internal/health", timeout=5)
        internal_status = status_code
        if status_code == 200:
            is_running = bool(payload.get("is_running", False))
            queue_length = payload.get("queue_length")
            fusion_enabled = payload.get("fusion_enabled")
            if fusion_enabled is False:
                status = "degraded"
        else:
            status = "degraded"
    except (urllib.error.URLError, OSError):
        # The public instance keeps serving its own endpoints when the internal
        # worker is down, so this reports degraded rather than failing.
        status = "degraded"

    return {
        "status": status,
        "git_commit": tools.get_git_commit(),
        "mode": tools.get_execution_mode(),
        "dev": tools.get_dev_env_var(),
        "is_running": is_running,
        "fusion_enabled": fusion_enabled,
        "queue_length": queue_length,
        "internal_status": internal_status
    }

@app.get("/metadata_providers/")
def get_metadata_providers():
    """Both metadata configuration files, as loaded at startup."""
    return {
        "providers": metadata_providers,
        "brackets": metadata_brackets,
        "providers_file": providers_file,
        "brackets_file": brackets_file
    }

@app.post("/metadata_tags/")
def create_metadata_tags(tags_request: MetadataTagsRequest):
    """Tags `<begin><tag>-<id><end>` for the given ids and apps, e.g. plex
    `{tmdb-1255}`, jellyfin `[tmdbid-1255]`. Apps in the order given, providers
    in the order of the providers file, then the unknown ones in the order
    given. A provider an app has no tag for is listed in unsupported."""
    if not tags_request.apps:
        raise HTTPException(status_code=400, detail="apps cannot be empty")
    try:
        ids = clean_metadata_ids(tags_request.ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ordered_apps = []
    for app_name in tags_request.apps:
        if app_name not in metadata_brackets:
            raise HTTPException(status_code=400, detail=f"Unknown app `{app_name}`, known apps: {', '.join(metadata_brackets)}")
        if app_name not in ordered_apps:
            ordered_apps.append(app_name)
    ordered_providers = [provider for provider in metadata_providers if provider in ids] + [provider for provider in ids if provider not in metadata_providers]

    per_app = {}
    unsupported = []
    for app_name in ordered_apps:
        app_tags = []
        for provider in ordered_providers:
            tag = metadata_providers.get(provider, {}).get(app_name)
            if tag == None:
                unsupported.append([app_name, provider])
            else:
                app_tags.append(f"{metadata_brackets[app_name]['begin']}{tag}-{ids[provider]}{metadata_brackets[app_name]['end']}")
        per_app[app_name] = " ".join(app_tags)

    return {
        "tags": " ".join(per_app[app_name] for app_name in ordered_apps if per_app[app_name]),
        "per_app": per_app,
        "unsupported": unsupported
    }
