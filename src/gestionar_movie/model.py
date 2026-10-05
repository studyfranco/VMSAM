"""SQLAlchemy models and queries for the movie manager database."""

from sqlalchemy import Text, UniqueConstraint, ForeignKey, BigInteger, Index, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from typing import Optional,List
from typing_extensions import Annotated

int_big = Annotated[BigInteger, mapped_column(BigInteger)]

class Base(DeclarativeBase):
    """Declarative base for the movie models."""
    type_annotations = {
        'str': Text,
    }

class movie(Base):
    """A movie, keyed by its TMDB id, with its destination folder."""
    __tablename__ = 'movies'
    
    tmdb_id: Mapped[str] = mapped_column(primary_key=True)
    destination_path: Mapped[str] = mapped_column(index=True)
    original_language: Mapped[str]
    number_cut: Mapped[int]
    cut_file_to_get_delay_second_method: Mapped[float]
    file: Mapped[Optional[str]]
    file_weight: Mapped[Optional[int]]

    regex_patterns: Mapped[List["regexPattern"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan"
    )
    incompatible_files: Mapped[List["incompatibleFile"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan"
    )

class regexPattern(Base):
    """A filename regex that maps incoming files to a movie."""
    __tablename__ = 'movies_regex_patterns'
    
    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    tmdb_id: Mapped[str] = mapped_column(ForeignKey("movie.tmdb_id"), index=True)
    rename_pattern: Mapped[Optional[str]]
    weight: Mapped[int]

    movie: Mapped["movie"] = relationship(back_populates="regex_patterns")

class incompatibleFile(Base):
    """A file the pipeline could not merge with its movie."""
    __tablename__ = 'movies_incompatible_files'

    tmdb_id: Mapped[str] = mapped_column(ForeignKey("movie.tmdb_id"), primary_key=True)
    file_path: Mapped[str]
    file_weight: Mapped[int]

    movie: Mapped["movie"] = relationship(back_populates="incompatible_files")

def setup_database(database_url, create_tables=False):
    """Create an engine for `database_url` and return a new session.

    Args:
        database_url: SQLAlchemy URL (PostgreSQL or SQLite).
        create_tables: Create missing tables first when True.
    """

    connect_args = {}
    if database_url.startswith("postgresql"):
        connect_args['connect_timeout'] = 60
    elif database_url.startswith("sqlite"):
        connect_args['timeout'] = 61
    engine = create_engine(database_url, echo=False, connect_args=connect_args, pool_pre_ping=True)
    
    if create_tables:
        Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)

    return Session()

def get_all_movie(session):
    """Return all movies, sorted by path."""
    return session.query(movie).order_by(
        movie.destination_path.asc()
    ).all()

def get_movie_data(tmdb_id, session):
    """Return the movie with this TMDB id, or None."""
    return session.query(movie).filter(movie.tmdb_id == tmdb_id).first()

def get_movie_by_path(destination_path, session):
    """Return the movie at this destination path, or None."""
    return session.query(movie).filter(movie.destination_path == destination_path).first()

def search_like_movie(folder_name_part, session):
    """Return movies whose path contains `folder_name_part`."""
    return session.query(movie).filter(movie.destination_path.like(f"%{folder_name_part}%")).all()

def insert_movie(tmdb_id, destination_path, original_language, number_cut, cut_file_to_get_delay_second_method, session):
    """Validate and insert a new movie; raise ValueError on invalid input."""
    if tmdb_id == None and not len(tmdb_id):
        raise ValueError("tmdb_id cannot be empty")
    if number_cut == None:
        number_cut = 5
    elif number_cut < 1:
        raise ValueError("number_cut must be at least 1")
    if cut_file_to_get_delay_second_method == None:
        cut_file_to_get_delay_second_method = 2.5
    elif cut_file_to_get_delay_second_method <= 1:
        raise ValueError("cut_file_to_get_delay_second_method must be greater than 1")
    if destination_path == None and not len(destination_path):
        raise ValueError("destination_path cannot be empty")
    if original_language == None and not len(original_language):
        raise ValueError("original_language cannot be empty")
    
    new_movie = movie(
        tmdb_id=tmdb_id,
        destination_path=destination_path,
        original_language=original_language,
        number_cut=number_cut,
        cut_file_to_get_delay_second_method=cut_file_to_get_delay_second_method
    )
    session.add(new_movie)
    session.commit()
    return new_movie

def get_regex_data(regex, session):
    """Return the regex row for this exact pattern, or None."""
    return session.query(regexPattern).filter(
        regexPattern.regex_pattern == regex
    ).first()
    
def insert_regex(regex_pattern, tmdb_id, rename_pattern, weight, session):
    """Validate and insert a regex; raise ValueError on invalid input."""
    if regex_pattern == None or len(regex_pattern) == 0:
        raise ValueError("regex_pattern cannot be empty")
    if tmdb_id == None or len(tmdb_id) == 0:
        raise ValueError("tmdb_id cannot be empty")
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    if weight == None:
        weight = 1
    elif weight < 1:
        raise ValueError("weight must be at least 1")
    
    new_regex = regexPattern(
        regex_pattern=regex_pattern,
        tmdb_id=tmdb_id,
        rename_pattern=rename_pattern,
        weight=weight
    )
    session.add(new_regex)
    session.commit()
    return new_regex

def get_all_regex(session):
    """Return all regexes, heaviest first."""
    return session.query(regexPattern).order_by(
        regexPattern.weight.desc()
    ).all()

def get_regex_by_tmdb_id(tmdb_id, session):
    return session.query(regexPattern).filter(
        regexPattern.tmdb_id == tmdb_id
    ).all()

def get_incompatible_files_data(tmdb_id, session):
    """Return the incompatible files of one movie, heaviest first."""
    return session.query(incompatibleFile).filter(
        incompatibleFile.tmdb_id == tmdb_id
    ).order_by(
        incompatibleFile.file_weight.desc()
    ).all()

def insert_incompatible_file(tmdb_id, file_path, file_weight, session):
    new_incompatible = incompatibleFile(
        tmdb_id=tmdb_id,
        file_path=file_path,
        file_weight=file_weight
    )
    session.add(new_incompatible)
    session.commit()
    return new_incompatible