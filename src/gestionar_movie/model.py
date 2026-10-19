"""Database model of the movie side.

A movie is any single file to keep: a film, or a standalone show episode. Its
`title` is its name in the database and is unique; it is not the file name.
The file name comes from the regex that caught the file, like the show's: its
`rename_pattern` with `{<suffix>}` replaced by the movie's suffix, or the
file's own name when the regex has none. Several movies may share one
destination directory or one suffix.

The table names (`movie`, `movie_regex_patterns`, `movie_incompatible_files`)
differ from those of an older unused draft (`movies`, ...) on purpose: those
tables may still exist with another schema, and create_all never alters an
existing table.
"""

from sqlalchemy import Text, Integer, BigInteger, JSON, UniqueConstraint, ForeignKey, CheckConstraint, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from typing import Optional, List
import os
from gestionar_common.database import get_connect_args

# BigInteger primary keys do not autoincrement on SQLite (only INTEGER PRIMARY
# KEY is the rowid alias), so a SQLite database gets a plain Integer.
big_id_type = BigInteger().with_variant(Integer(), "sqlite")

class Base(DeclarativeBase):
    type_annotation_map = {
        str: Text,
    }

class movie(Base):
    __tablename__ = 'movie'

    id: Mapped[int] = mapped_column(primary_key=True)
    destination_path: Mapped[str] = mapped_column(index=True)
    title: Mapped[str]
    suffix: Mapped[str] = mapped_column(default='', server_default='')
    original_language: Mapped[str]
    number_cut: Mapped[int]
    cut_file_to_get_delay_second_method: Mapped[float]
    metadata_ids: Mapped[dict] = mapped_column(JSON, default=dict)
    file_path: Mapped[Optional[str]]
    file_weight: Mapped[Optional[int]]

    __table_args__ = (
        UniqueConstraint("title", name="uq_movie_title"),
        UniqueConstraint("file_path", name="uq_movie_file_path"),
    )

    regex_patterns: Mapped[List["movie_regex_pattern"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan", passive_deletes=True
    )
    incompatible_files: Mapped[List["movie_incompatible_file"]] = relationship(
        back_populates="movie", cascade="all, delete-orphan", passive_deletes=True
    )

class movie_regex_pattern(Base):
    __tablename__ = 'movie_regex_patterns'

    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movie.id", ondelete="CASCADE"), index=True)
    rename_pattern: Mapped[Optional[str]]
    weight: Mapped[int]

    __table_args__ = (
        CheckConstraint("weight >= 1", name="ck_movie_regex_patterns_weight"),
    )

    movie: Mapped["movie"] = relationship(back_populates="regex_patterns")

class movie_incompatible_file(Base):
    __tablename__ = 'movie_incompatible_files'

    id: Mapped[int] = mapped_column(big_id_type, primary_key=True, autoincrement=True)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movie.id", ondelete="CASCADE"), index=True)
    file_path: Mapped[str]
    file_weight: Mapped[int]

    __table_args__ = (
        UniqueConstraint("file_path", name="uq_movie_incompatible_files_file_path"),
    )

    movie: Mapped["movie"] = relationship(back_populates="incompatible_files")

def setup_database(database_url, create_tables=False):
    engine = create_engine(database_url, echo=False, connect_args=get_connect_args(database_url), pool_pre_ping=True)
    if create_tables:
        Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# Placeholder of a regex rename pattern, replaced by the movie's suffix.
suffix_pattern_insert = "{<suffix>}"

def get_all_movies(session):
    return session.query(movie).order_by(
        movie.title.asc()
    ).all()

def get_movie_data(movie_id, session):
    return session.query(movie).filter(movie.id == movie_id).first()

def get_movie_by_title(title, session):
    return session.query(movie).filter(movie.title == title).first()

def get_movie_by_file_path(file_path, session):
    return session.query(movie).filter(movie.file_path == file_path).first()

def search_like_movie(destination_part, session):
    return session.query(movie).filter(movie.destination_path.like(f"%{destination_part}%")).order_by(
        movie.title.asc()
    ).all()

def search_title_like_movie(title_part, session):
    """Movies whose title contains title_part, case-insensitive, `%` and `_` taken literally."""
    return session.query(movie).filter(func.lower(movie.title).contains(title_part.lower(), autoescape=True)).order_by(
        movie.title.asc()
    ).all()

def has_separator(value):
    return os.sep in value or (os.altsep != None and os.altsep in value) or '\0' in value

def clean_metadata_ids(metadata_ids):
    """Provider -> id, both non-empty strings without a path separator, stripped."""
    cleaned = {}
    for provider, value in (metadata_ids or {}).items():
        if provider == None or len(provider.strip()) == 0:
            raise ValueError("metadata_ids: a provider name is empty")
        if value == None or len(value.strip()) == 0:
            raise ValueError(f"metadata_ids: the id of `{provider}` is empty")
        if has_separator(value):
            raise ValueError(f"metadata_ids: the id of `{provider}` must not contain a path separator")
        cleaned[provider.strip()] = value.strip()
    return cleaned

def test_conformity_movie_param(title, destination_path, suffix, original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids):
    """Shared by insert_movie and update_movie_parameters.

    Returns (title, suffix, number_cut, cut_file_to_get_delay_second_method,
    metadata_ids) with the title stripped, the defaults applied and the
    metadata ids cleaned. The title names the movie's error directory, hence
    the separator rule; its uniqueness is the table's constraint.
    """
    if title == None or len(title.strip()) == 0:
        raise ValueError("title cannot be empty")
    title = title.strip()
    if has_separator(title) or title in ('.', '..'):
        raise ValueError("title must not contain a path separator")
    if destination_path == None or len(destination_path.strip()) == 0:
        raise ValueError("destination_path cannot be empty")
    if suffix == None:
        suffix = ''
    if has_separator(suffix):
        raise ValueError("suffix must not contain a path separator")

    if number_cut == None:
        number_cut = 5
    elif number_cut < 1:
        raise ValueError("number_cut must be at least 1")
    if cut_file_to_get_delay_second_method == None:
        cut_file_to_get_delay_second_method = 2.5
    elif cut_file_to_get_delay_second_method <= 1:
        raise ValueError("cut_file_to_get_delay_second_method must be greater than 1")
    if original_language == None or len(original_language.strip()) == 0:
        raise ValueError("original_language cannot be empty")
    metadata_ids = clean_metadata_ids(metadata_ids)

    return (title, suffix, number_cut, cut_file_to_get_delay_second_method, metadata_ids)

def insert_movie(title, destination_path, suffix, original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids, session):
    title, suffix, number_cut, cut_file_to_get_delay_second_method, metadata_ids = test_conformity_movie_param(
        title, destination_path, suffix, original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids)

    new_movie = movie(
        title=title,
        destination_path=destination_path,
        suffix=suffix,
        original_language=original_language,
        number_cut=number_cut,
        cut_file_to_get_delay_second_method=cut_file_to_get_delay_second_method,
        metadata_ids=metadata_ids
    )
    session.add(new_movie)
    session.commit()
    return new_movie

def update_movie_parameters(movie_data, destination_path, suffix, original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids, session):
    """Set every parameter but the title (the key), the file and its weight.
    Nothing on disk changes: a new suffix only applies to the next file named."""
    title, suffix, number_cut, cut_file_to_get_delay_second_method, metadata_ids = test_conformity_movie_param(
        movie_data.title, destination_path, suffix, original_language, number_cut, cut_file_to_get_delay_second_method, metadata_ids)

    movie_data.destination_path = destination_path
    movie_data.suffix = suffix
    movie_data.original_language = original_language
    movie_data.number_cut = number_cut
    movie_data.cut_file_to_get_delay_second_method = cut_file_to_get_delay_second_method
    movie_data.metadata_ids = metadata_ids
    session.commit()
    return movie_data

def set_movie_file(movie_data, file_path, file_weight, session):
    movie_data.file_path = file_path
    movie_data.file_weight = file_weight
    session.commit()
    return movie_data

def delete_movie(movie_data, session):
    session.delete(movie_data)
    session.commit()

def build_file_name(rename_pattern, suffix, file_name):
    """Name of a file caught by a regex: its rename pattern with the suffix
    inserted, or file_name itself when the regex has no rename pattern."""
    if rename_pattern == None or len(rename_pattern) == 0:
        return file_name
    new_file_name = rename_pattern.replace(suffix_pattern_insert, suffix or '')
    if len(new_file_name.strip()) == 0 or new_file_name in ('.', '..') or has_separator(new_file_name):
        raise ValueError(f"Rename pattern `{rename_pattern}` with suffix `{suffix}` does not give a file name")
    return new_file_name

def get_regex_data(regex, session):
    return session.query(movie_regex_pattern).filter(
        movie_regex_pattern.regex_pattern == regex
    ).first()

def get_all_regex(session):
    return session.query(movie_regex_pattern).order_by(
        movie_regex_pattern.weight.desc(), movie_regex_pattern.regex_pattern.asc()
    ).all()

def get_regex_by_movie_id(movie_id, session):
    return session.query(movie_regex_pattern).filter(
        movie_regex_pattern.movie_id == movie_id
    ).order_by(movie_regex_pattern.weight.desc(), movie_regex_pattern.regex_pattern.asc()).all()

def test_conformity_regex_param(movie_id, rename_pattern, weight):
    """Returns (rename_pattern, weight); an empty rename pattern becomes None."""
    if movie_id == None or movie_id <= 0:
        raise ValueError("movie_id must be a positive integer")
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    if rename_pattern != None:
        if suffix_pattern_insert not in rename_pattern:
            raise ValueError(f"Rename pattern must contain the suffix pattern: {suffix_pattern_insert}")
        if has_separator(rename_pattern):
            raise ValueError("Rename pattern must be a file name, not a path")
    if weight == None:
        weight = 1
    elif weight < 1:
        raise ValueError("weight must be at least 1")
    return (rename_pattern, weight)

def insert_regex_sub(regex_pattern, movie_id, rename_pattern, weight, session):
    """Add the row without committing."""
    if regex_pattern == None or len(regex_pattern) == 0:
        raise ValueError("regex_pattern cannot be empty")
    rename_pattern, weight = test_conformity_regex_param(movie_id, rename_pattern, weight)

    new_regex = movie_regex_pattern(regex_pattern=regex_pattern, movie_id=movie_id, rename_pattern=rename_pattern, weight=weight)
    session.add(new_regex)
    return new_regex

def insert_regex(regex_pattern, movie_id, rename_pattern, weight, session):
    new_regex = insert_regex_sub(regex_pattern, movie_id, rename_pattern, weight, session)
    session.commit()
    return new_regex

def update_regex(regex_data, movie_id, rename_pattern, weight, session):
    regex_data.rename_pattern, regex_data.weight = test_conformity_regex_param(movie_id, rename_pattern, weight)
    regex_data.movie_id = movie_id
    session.commit()
    return regex_data

def replace_regex(regex_data, new_regex_pattern, movie_id, rename_pattern, weight, session):
    """Change the pattern (the primary key) and the other fields of a regex.

    The new row is added and the old one deleted under a single commit, so
    either both happen or neither.
    """
    new_regex = insert_regex_sub(new_regex_pattern, movie_id, rename_pattern, weight, session)
    session.delete(regex_data)
    session.commit()
    return new_regex

def delete_regex(regex_data, session):
    session.delete(regex_data)
    session.commit()

def get_all_incompatible_files(session):
    return session.query(movie_incompatible_file).order_by(
        movie_incompatible_file.movie_id.asc(),
        movie_incompatible_file.file_weight.desc()
    ).all()

def get_incompatible_files_by_movie_id(movie_id, session):
    return session.query(movie_incompatible_file).filter(
        movie_incompatible_file.movie_id == movie_id
    ).order_by(movie_incompatible_file.file_weight.desc()).all()

def get_incompatible_file_by_path(file_path, session):
    return session.query(movie_incompatible_file).filter(
        movie_incompatible_file.file_path == file_path
    ).first()

def insert_incompatible_file(movie_id, file_path, file_weight, session):
    new_incompatible = movie_incompatible_file(movie_id=movie_id, file_path=file_path, file_weight=file_weight)
    session.add(new_incompatible)
    session.commit()
    return new_incompatible

def delete_incompatible_file(incompatible_file_data, session):
    session.delete(incompatible_file_data)
    session.commit()
