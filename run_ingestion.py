import logging
import sys
from app.ingestion.pipeline import process_pdf
from app.storage.database import Base, engine

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "Dataset.pdf"
    print("Initializing Database tables...")
    Base.metadata.create_all(bind=engine)
    print(f"Starting ingestion for {file_path}...")
    try:
        doc_id = process_pdf(file_path)
        print(f"Ingestion successful! Document ID: {doc_id}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
