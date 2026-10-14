from sqlalchemy import Text, UniqueConstraint, ForeignKey, BigInteger, Index, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from typing import Optional,List
from typing_extensions import Annotated
from gestionar_common.database import get_connect_args

int_big = Annotated[BigInteger, mapped_column(BigInteger)]

class Base(DeclarativeBase):
    # SQLAlchemy 2.0 resolves Mapped[...] annotations through type_annotation_map,
    # keyed by the Python type itself. The previous `type_annotations = {'str': Text}`
    # was neither the expected attribute name nor the expected key type, so it was
    # silently ignored and every Mapped[str] fell back to the default VARCHAR.
    type_annotation_map = {
        str: Text,
    }

class folder(Base):
    __tablename__ = 'folders'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    destination_path: Mapped[str] = mapped_column(index=True)
    original_language: Mapped[str]
    number_cut: Mapped[int]
    cut_file_to_get_delay_second_method: Mapped[float]
    max_episode_number: Mapped[int]

    # A bare UniqueConstraint(...) in the class body is a free-standing expression
    # that never reaches the Table, so it has to go through __table_args__.
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
    __tablename__ = 'regex_patterns'
    
    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    rename_pattern: Mapped[Optional[str]]
    weight: Mapped[int]

    folder: Mapped["folder"] = relationship(back_populates="regex_patterns")

class incompatibleFile(Base):
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
    __tablename__ = 'incrementaller'

    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    rename_pattern: Mapped[str]
    episode_incremental: Mapped[int]

def setup_database(database_url, create_tables=False):
    """Configuration complète de la base de données"""

    connect_args = get_connect_args(database_url)
    # Créer l'engine
    engine = create_engine(database_url, echo=False, connect_args=connect_args, pool_pre_ping=True)
    
    # Créer les tables si demandé
    if create_tables:
        Base.metadata.create_all(engine)
    
    # Configurer la session
    Session = sessionmaker(bind=engine)

    return Session()

def get_all_folder(session):
    return session.query(folder).order_by(
        folder.destination_path.asc()
    ).all()

def get_folder_data(folder_id, session):
    return session.query(folder).filter(folder.id == folder_id).first()

def get_folder_by_path(destination_path, session):
    return session.query(folder).filter(folder.destination_path == destination_path).first()

def search_like_folder(folder_name_part, session):
    return session.query(folder).filter(folder.destination_path.like(f"%{folder_name_part}%")).all()

def test_conformity_folder_param(original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number):
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

    if original_language == None or not len(original_language.strip()):
        raise ValueError("original_language cannot be empty")

    return (max_episode_number, number_cut, cut_file_to_get_delay_second_method)

def insert_folder(destination_path, original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number, session):
    max_episode_number, number_cut, cut_file_to_get_delay_second_method = test_conformity_folder_param(original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number)
    if destination_path == None or not len(destination_path.strip()):
        raise ValueError("destination_path cannot be empty")
    
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

def update_folder_parameters(folder_data, original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number, session):
    folder_data.max_episode_number, folder_data.number_cut, folder_data.cut_file_to_get_delay_second_method = test_conformity_folder_param(original_language, number_cut, cut_file_to_get_delay_second_method, max_episode_number)

    folder_data.original_language = original_language
    session.commit()
    return folder_data

def get_regex_data(regex, session):
    return session.query(regexPattern).filter(
        regexPattern.regex_pattern == regex
    ).first()

def test_conformity_regex_param(folder_id, rename_pattern, weight):
    if folder_id == None or folder_id <= 0:
        raise ValueError("folder_id must be a positive integer")

    if rename_pattern != None and len(rename_pattern) == 0:
        rename_pattern = None

    if weight == None:
        weight = 1
    elif weight < 1:
        raise ValueError("weight must be at least 1")

    return (rename_pattern, weight)

def insert_regex_sub(regex_pattern, folder_id, rename_pattern, weight, session):
    if regex_pattern == None or len(regex_pattern) == 0:
        raise ValueError("regex_pattern cannot be empty")

    rename_pattern, weight = test_conformity_regex_param(folder_id, rename_pattern, weight)

    new_regex = regexPattern(
        regex_pattern=regex_pattern,
        folder_id=folder_id,
        rename_pattern=rename_pattern,
        weight=weight
    )
    session.add(new_regex)
    return new_regex

def insert_regex(regex_pattern, folder_id, rename_pattern, weight, session):
    new_regex = insert_regex_sub(regex_pattern, folder_id, rename_pattern, weight, session)
    session.commit()
    return new_regex

def update_regex(regex_data, folder_id, rename_pattern, weight, session):
    regex_data.rename_pattern, regex_data.weight = test_conformity_regex_param(folder_id, rename_pattern, weight)
    
    regex_data.folder_id = folder_id
    session.commit()
    return regex_data

def replace_regex(regex_data, new_regex_pattern, folder_id, rename_pattern, weight, session):
    """Change the pattern of a regex, which is the primary key of the table.

    A primary key is not updated in place: the new row is inserted and the old
    one deleted in the same transaction, so either both happen or neither.
    """
    new_regex = insert_regex_sub(new_regex_pattern, folder_id, rename_pattern, weight, session)

    session.delete(regex_data)
    session.commit()
    return new_regex

def delete_regex(regex_data, session):
    session.delete(regex_data)
    session.commit()

def get_all_regex(session):
    return session.query(regexPattern).order_by(
        regexPattern.weight.desc()
    ).all()

def get_regex_by_folder_id(folder_id, session):
    return session.query(regexPattern).filter(
        regexPattern.folder_id == folder_id
    ).all()
    
def get_episode_data(folder_id, episode_number, session):
    return session.query(episode).filter(
        episode.folder_id == folder_id,
        episode.episode_number == episode_number
    ).first()

def get_episode_by_path(file_path, session):
    return session.query(episode).filter(
        episode.file_path == file_path
    ).first()

def get_episodes_by_folder_id(folder_id, session):
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
    return session.query(incompatibleFile).order_by(
        incompatibleFile.folder_id.asc(),
        incompatibleFile.episode_number.asc(),
        incompatibleFile.file_weight.desc()
    ).all()

def get_incompatible_file_by_path(file_path, session):
    return session.query(incompatibleFile).filter(
        incompatibleFile.file_path == file_path
    ).first()

def delete_incompatible_file(incompatible_file_data, session):
    session.delete(incompatible_file_data)
    session.commit()

def get_incrementaller_data(regex, session):
    return session.query(incrementaller).filter(
        incrementaller.regex_pattern == regex
    ).first()

def get_all_incrementaller(session):
    return session.query(incrementaller).all()

def insert_incrementaller(regex_pattern, rename_pattern, episode_incremental, session):
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
    if rename_pattern != None and len(rename_pattern) == 0:
        raise ValueError("rename_pattern cannot be empty")
    if episode_incremental == None:
        raise ValueError("episode_incremental must be define")

    incremental_data.rename_pattern = rename_pattern
    incremental_data.episode_incremental = episode_incremental
    session.commit()
    return incremental_data
