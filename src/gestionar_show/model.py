"""SQLAlchemy models and queries for the show manager database."""

from sqlalchemy import Text, UniqueConstraint, ForeignKey, BigInteger, Index, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from typing import Optional,List
from typing_extensions import Annotated

int_big = Annotated[BigInteger, mapped_column(BigInteger)]

class Base(DeclarativeBase):
    """Declarative base mapping `str` columns to TEXT."""
    type_annotation_map = {
        str: Text,
    }

class folder(Base):
    """A destination folder holding the episodes of one show."""
    __tablename__ = 'folders'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    destination_path: Mapped[str] = mapped_column(index=True)
    original_language: Mapped[str]
    number_cut: Mapped[int]
    cut_file_to_get_delay_second_method: Mapped[float]
    max_episode_number: Mapped[int]

    # Constraints must go through __table_args__ to reach the Table.
    __table_args__ = (
        UniqueConstraint("destination_path", name="uq_folders_destination_path"),
    )

    regex_patterns: Mapped[List["regexPattern"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )
    episodes: Mapped[List["episode"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )
    incompatible_files: Mapped[List["incompatibleFile"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )

class regexPattern(Base):
    """A filename regex that maps incoming files to a folder and an episode number."""
    __tablename__ = 'regex_patterns'
    
    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    rename_pattern: Mapped[Optional[str]]
    weight: Mapped[int]

    folder: Mapped["folder"] = relationship(back_populates="regex_patterns")

class incompatibleFile(Base):
    """A file the pipeline could not merge with its episode's master."""
    __tablename__ = 'incompatible_files'

    id: Mapped[int_big] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    episode_number: Mapped[int]
    file_path: Mapped[str]
    file_weight: Mapped[int]

    __table_args__ = (
        UniqueConstraint("file_path", name="uq_incompatible_files_file_path"),
    )

    folder: Mapped["folder"] = relationship(back_populates="incompatible_files")

class episode(Base):
    """The registered master file of one episode."""
    __tablename__ = 'episodes'

    id: Mapped[int_big] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    episode_number: Mapped[int]
    file_path: Mapped[str]
    file_weight: Mapped[int]

    __table_args__ = (
        Index("ix_folder_episode", "folder_id", "episode_number"),
    )

    folder: Mapped["folder"] = relationship(back_populates="episodes")

class incrementaller(Base):
    """A regex that renames files by shifting their episode number."""
    __tablename__ = 'incrementaller'

    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    rename_pattern: Mapped[str]
    episode_incremental: Mapped[int]

class special_rename(Base):
    """Exact-name rename applied by the daemon before any regex runs.

    Used for specials whose incoming name carries no usable episode number; the
    renamed file is then picked up by the folder's ordinary regex.
    """
    __tablename__ = 'special_renames'

    file_name: Mapped[str] = mapped_column(primary_key=True)
    new_file_name: Mapped[str]

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

def get_all_folder(session):
    """Return all folders, sorted by path."""
    return session.query(folder).order_by(
        folder.destination_path.asc()
    ).all()

def get_folder_data(folder_id, session):
    """Return the folder with this id, or None."""
    return session.query(folder).filter(folder.id == folder_id).first()

def get_folder_by_path(destination_path, session):
    """Return the folder at this destination path, or None."""
    return session.query(folder).filter(folder.destination_path == destination_path).first()

def search_like_folder(folder_name_part, session):
    """Return folders whose path contains `folder_name_part`."""
    return session.query(folder).filter(folder.destination_path.like(f"%{folder_name_part}%")).all()

def insert_folder(destination_path, original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number, session):
    """Validate and insert a new folder; raise ValueError on invalid input."""
    if max_episode_number == None:
        max_episode_number = 12
    elif max_episode_number < 1:
        raise ValueError("max_episode_number must be at least 1")
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
    
    new_folder = folder(
        destination_path=destination_path,
        original_language=original_language,
        number_cut=number_cut,
        cut_file_to_get_delay_second_method=cut_file_to_get_delay_second_method,
        max_episode_number=max_episode_number
    )
    session.add(new_folder)
    session.commit()
    return new_folder

def get_regex_data(regex, session):
    """Return the regex row for this exact pattern, or None."""
    return session.query(regexPattern).filter(
        regexPattern.regex_pattern == regex
    ).first()
    
def insert_regex(regex_pattern, folder_id, rename_pattern, weight, session):
    """Validate and insert a regex; raise ValueError on invalid input."""
    if regex_pattern == None or len(regex_pattern) == 0:
        raise ValueError("regex_pattern cannot be empty")
    if folder_id == None or folder_id <= 0:
        raise ValueError("folder_id must be a positive integer")
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    if weight == None:
        weight = 1
    elif weight < 1:
        raise ValueError("weight must be at least 1")
    
    new_regex = regexPattern(
        regex_pattern=regex_pattern,
        folder_id=folder_id,
        rename_pattern=rename_pattern,
        weight=weight
    )
    session.add(new_regex)
    session.commit()
    return new_regex

def update_regex(regex_data, folder_id, rename_pattern, weight, session):
    """Update a regex's folder, rename pattern and weight."""
    if folder_id == None or folder_id <= 0:
        raise ValueError("folder_id must be a positive integer")
    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None
    if weight == None:
        weight = 1
    elif weight < 1:
        raise ValueError("weight must be at least 1")
    
    regex_data.folder_id = folder_id
    regex_data.rename_pattern = rename_pattern
    regex_data.weight = weight
    session.commit()
    return regex_data

def get_all_regex(session):
    """Return all regexes, heaviest first."""
    return session.query(regexPattern).order_by(
        regexPattern.weight.desc()
    ).all()

def get_regex_by_folder_id(folder_id, session):
    return session.query(regexPattern).filter(
        regexPattern.folder_id == folder_id
    ).all()
    
def get_episode_data(folder_id, episode_number, session):
    """Return the registered episode of a folder, or None."""
    return session.query(episode).filter(
        episode.folder_id == folder_id,
        episode.episode_number == episode_number
    ).first()

def get_episode_by_path(file_path, session):
    """Return the episode registered at this file path, or None."""
    return session.query(episode).filter(
        episode.file_path == file_path
    ).first()

def get_episodes_by_folder_id(folder_id, session):
    """Return a folder's episodes sorted by number."""
    return session.query(episode).filter(
        episode.folder_id == folder_id
    ).order_by(
        episode.episode_number.asc()
    ).all()

def insert_episode(folder_id, episode_number, file_path, file_weight, session):
    new_episode = episode(
        folder_id=folder_id,
        episode_number=episode_number,
        file_path=file_path,
        file_weight=file_weight
    )
    session.add(new_episode)
    session.commit()
    return new_episode

def get_incompatible_files_data(folder_id, episode_number, session):
    """Return the incompatible files of one episode, heaviest first."""
    return session.query(incompatibleFile).filter(
        incompatibleFile.folder_id == folder_id,
        incompatibleFile.episode_number == episode_number
    ).order_by(
        incompatibleFile.file_weight.desc()
    ).all()

def insert_incompatible_file(folder_id, episode_number, file_path, file_weight, session):
    new_incompatible = incompatibleFile(
        folder_id=folder_id,
        episode_number=episode_number,
        file_path=file_path,
        file_weight=file_weight
    )
    session.add(new_incompatible)
    session.commit()
    return new_incompatible

def get_all_incompatible_files(session):
    """Return all incompatible files sorted by folder, episode and weight."""
    return session.query(incompatibleFile).order_by(
        incompatibleFile.folder_id.asc(),
        incompatibleFile.episode_number.asc(),
        incompatibleFile.file_weight.desc()
    ).all()

def get_incompatible_file_by_path(file_path, session):
    """Return the incompatible file at this path, or None."""
    return session.query(incompatibleFile).filter(
        incompatibleFile.file_path == file_path
    ).first()

def delete_incompatible_file(incompatible_file_data, session):
    session.delete(incompatible_file_data)
    session.commit()

def get_incrementaller_data(regex, session):
    """Return the incrementaller for this exact pattern, or None."""
    return session.query(incrementaller).filter(
        incrementaller.regex_pattern == regex
    ).first()

def get_all_incrementaller(session):
    return session.query(incrementaller).all()

def insert_incrementaller(regex_pattern, rename_pattern, episode_incremental, session):
    """Validate and insert an incrementaller; raise ValueError on invalid input."""
    if regex_pattern == None or len(regex_pattern) == 0:
        raise ValueError("regex_pattern cannot be empty")
    if rename_pattern != None and len(rename_pattern) == 0:
        raise ValueError("rename_pattern cannot be empty")
    if episode_incremental == None:
        raise ValueError("episode_incremental must be define")
    
    new_incremental = incrementaller(
        regex_pattern=regex_pattern,
        rename_pattern=rename_pattern,
        episode_incremental=episode_incremental
    )
    session.add(new_incremental)
    session.commit()
    return new_incremental

def update_incrementaller(incremental_data, rename_pattern, episode_incremental, session):
    """Update an incrementaller's rename pattern and episode shift."""
    if rename_pattern != None and len(rename_pattern) == 0:
        raise ValueError("rename_pattern cannot be empty")
    if episode_incremental == None:
        raise ValueError("episode_incremental must be define")

    incremental_data.rename_pattern = rename_pattern
    incremental_data.episode_incremental = episode_incremental
    session.commit()
    return incremental_data

def get_special_rename_data(file_name, session):
    """Return the special rename for this exact file name, or None."""
    return session.query(special_rename).filter(
        special_rename.file_name == file_name
    ).first()

def get_all_special_rename(session):
    """Return all special renames sorted by file name."""
    return session.query(special_rename).order_by(
        special_rename.file_name.asc()
    ).all()

def insert_special_rename(file_name, new_file_name, session):
    """Validate and insert a special rename; raise ValueError on invalid input."""
    if file_name == None or len(file_name) == 0:
        raise ValueError("file_name cannot be empty")
    if new_file_name == None or len(new_file_name) == 0:
        raise ValueError("new_file_name cannot be empty")
    if file_name == new_file_name:
        raise ValueError("new_file_name must differ from file_name")

    new_special = special_rename(
        file_name=file_name,
        new_file_name=new_file_name
    )
    session.add(new_special)
    session.commit()
    return new_special

def update_special_rename(special_data, new_file_name, session):
    if new_file_name == None or len(new_file_name) == 0:
        raise ValueError("new_file_name cannot be empty")
    if special_data.file_name == new_file_name:
        raise ValueError("new_file_name must differ from file_name")

    special_data.new_file_name = new_file_name
    session.commit()
    return special_data
