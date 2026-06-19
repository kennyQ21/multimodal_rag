import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import get_settings

settings = get_settings()

# Ensure model cache dirs are set in environment for HuggingFace libs
os.environ.setdefault("HF_HOME", settings.hf_home)
os.environ.setdefault("TRANSFORMERS_CACHE", settings.transformers_cache)
os.environ.setdefault("EASYOCR_MODULE_PATH", settings.easyocr_module_path)

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
