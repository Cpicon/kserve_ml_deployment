"""Database configuration and session management."""

import logging
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from aiq_circular_detection.models import Base
from config import get_settings

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Ensure a data directory exists if using SQLite
if settings.database_url.startswith("sqlite:///"):
    # Extract a path from sqlite URL (sqlite:///path/to/db.sqlite3)
    db_path = settings.database_url.replace("sqlite:///", "")
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

# Create engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
    echo=settings.database_echo,  # Log SQL statements if configured
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Enable foreign key constraints for SQLite
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        """Enable foreign key constraints in SQLite."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Create sessionmaker
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session.
    
    Yields a database session and ensures it's closed after use.
    This should be used as a FastAPI dependency.
    
    Example:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize a database by creating all tables.
    
    This should be called on application startup.
    """
    logger.info(f"Creating database tables at {settings.database_url}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db() -> None:
    """Drop all database tables.
    
    WARNING: This will delete all data!
    This should only be used for testing or development.
    """
    logger.warning(f"Dropping all database tables at {settings.database_url}")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped") 