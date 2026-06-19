import datetime
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Text, Float
from sqlalchemy.orm import relationship
from .database import Base


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    total_pages = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    image_path = Column(String, nullable=True)
    document = relationship("Document", back_populates="pages")
    images = relationship("Image", back_populates="page", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, index=True)
    page_id = Column(String, ForeignKey("pages.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    bbox = Column(JSON)
    path = Column(String)
    figure_label = Column(String, nullable=True)   # "Fig 1", "Fig 2", etc.
    caption = Column(Text, nullable=True)
    image_summary = Column(Text, nullable=True)    # AI-generated visual description
    page = relationship("Page", back_populates="images")


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    procedure_id = Column(String, index=True)
    procedure_title = Column(String, nullable=True)
    page_start = Column(Integer, nullable=False)
    page_end = Column(Integer, nullable=False)
    # steps: [{"step": 1, "instruction": "...", "images": ["image_id_1", ...]}]
    steps = Column(JSON, nullable=False)
    retrieval_text = Column(Text, nullable=True)
    document = relationship("Document", back_populates="chunks")


class OCRCache(Base):
    __tablename__ = "ocr_cache"
    image_hash = Column(String, primary_key=True, index=True)
    ocr_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class QueryLog(Base):
    __tablename__ = "queries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    latency_sec = Column(Float, nullable=True)
    result_count = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
