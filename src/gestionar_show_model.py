from sqlalchemy import Text, UniqueConstraint, ForeignKey, BigInteger, Index, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from typing import Optional,List
from typing_extensions import Annotated

int_big = Annotated[BigInteger, mapped_column(BigInteger)]

class Base(DeclarativeBase):
    type_annotations = {
        'str': Text,
    }

class folder(Base):
    __tablename__ = 'folders'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    destination_path: Mapped[str] = mapped_column(index=True)
    original_language: Mapped[str]
    
    UniqueConstraint("destination_path")

    regex_patterns: Mapped[List["regexPattern"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )
    episodes: Mapped[List["episode"]] = relationship(
        back_populates="folder", cascade="all, delete-orphan"
    )

class regexPattern(Base):
    __tablename__ = 'regex_patterns'
    
    regex_pattern: Mapped[str] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"), index=True)
    rename_pattern: Mapped[Optional[str]]
    weight: Mapped[int]

    folder: Mapped["folder"] = relationship(back_populates="regex_patterns")

class episode(Base):
    __tablename__ = 'episodes'

    id: Mapped[int_big] = mapped_column(primary_key=True)
    folder_id: Mapped[int] = mapped_column(ForeignKey("folders.id"))
    episode_number: Mapped[int]
    file_path: Mapped[str]
    file_weight: Mapped[int]

    __table_args__ = (
        Index("ix_folder_episode", "folder_id", "episode_number"),
    )

    folder: Mapped["folder"] = relationship(back_populates="episodes")
    
def setup_database(database_url, create_tables=True):
    """Configuration complète de la base de données"""
    # Créer l'engine
    engine = create_engine(database_url, echo=True)
    
    # Créer les tables si demandé
    if create_tables:
        Base.metadata.create_all(engine)
    
    # Configurer la session
    Session = sessionmaker(bind=engine)

    return Session()

def get_folder_data(folder_id, session):
    return session.query(folder).filter(folder.id == folder_id).first()

def get_regex_data(regex, session):
    return session.query(regexPattern).filter(
        regexPattern.regex_pattern == regex
    ).first()

def get_all_regex(session):
    return session.query(regexPattern).order_by(
        regexPattern.weight.desc()
    ).all()
    
def get_episode_data(folder_id, episode_number, session):
    return session.query(episode).filter(
        episode.folder_id == folder_id,
        episode.episode_number == episode_number
    ).first()

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