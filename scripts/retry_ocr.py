import logging
import cv2
import sys
from app.storage.database import SessionLocal
from app.storage.models import Image, OCRCache, Chunk
from app.ingestion.ocr import run_ocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_failed_ocr():
    db = SessionLocal()
    # Find images where caption is empty
    empty_images = db.query(Image).filter((Image.caption == "") | (Image.caption == None)).all()
    
    if not empty_images:
        logger.info("No empty images found. Everything is already OCR'd.")
        return

    logger.info(f"Found {len(empty_images)} images with missing OCR text. Retrying...")
    
    for img in empty_images:
        img_cv = cv2.imread(img.path)
        if img_cv is None:
            logger.warning(f"Could not read image file at {img.path}")
            continue
            
        try:
            logger.info(f"Running OCR on {img.path}...")
            ocr_text = run_ocr(img_cv)
            
            if ocr_text:
                img.caption = ocr_text[:200]
                
                # Update OCR cache
                cache = db.query(OCRCache).filter(OCRCache.image_hash == img.id).first()
                if cache:
                    cache.ocr_text = ocr_text
                else:
                    db.add(OCRCache(image_hash=img.id, ocr_text=ocr_text))
                    
                # Update chunks that use this image
                # For page-level chunking, instruction is the text of the page
                chunks = db.query(Chunk).filter(Chunk.steps.contains([{"images": [img.id]}])).all()
                for chunk in chunks:
                    # Very simple update assuming 1 step per chunk
                    new_steps = list(chunk.steps)
                    for step in new_steps:
                        if img.id in step.get("images", []):
                            step["instruction"] = ocr_text
                    chunk.steps = new_steps
                    chunk.retrieval_text = ocr_text
                    
                db.commit()
                logger.info(f"Successfully recovered text for {img.id}")
            else:
                logger.warning(f"OCR still returned empty for {img.id}")
                
        except Exception as e:
            logger.error(f"OCR failed for {img.id}: {e}")
            db.rollback()

if __name__ == "__main__":
    retry_failed_ocr()
