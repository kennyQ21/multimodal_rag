# Multimodal RAG for Image-Heavy Technical Manuals

A multimodal Retrieval-Augmented Generation (RAG) system built to answer how-to questions from scanned, image-heavy operating manuals with ordered, step-by-step illustrated procedures.

## Key Features
* **Illustrated Steps**: Interleaves extracted text instructions with exact matching screenshots from the manual.
* **Hybrid Retrieval**: Combines BM25 and Vector Search (ChromaDB + BGE Large) with Reciprocal Rank Fusion (RRF).
* **Cross-Encoder Reranking**: Uses `bge-reranker-v2-m3` to sort and identify the most relevant procedure chunks.
* **Neighbor Expansion**: Automatically pulls surrounding steps of a top-ranked procedure to ensure completeness and avoid partial answers.
* **Source Grounding**: Every instruction and image is strictly tied to a specific page and figure label.
* **Local Parsing**: Uses PyMuPDF + PaddleOCR for CPU-optimized layout detection and text extraction from screenshots/figures.
* **Groq Generation**: Uses `llama-3.3-70b-versatile` exclusively for answer composition and formatting (zero-hallucination instruction selection).

## Architecture & Tradeoffs
* **Vector Store / DB**: Uses pure SQLite for storing extracted chunks and image metadata, removing the need for heavy external vector databases or Docker.
* **Retrieval (BM25)**: Relies exclusively on local `rank_bm25` (Okapi BM25) for high-precision text retrieval, bypassing the severe Windows DLL compatibility issues caused by PyTorch/ONNX on Python 3.12.
* **Parsing (PyMuPDF + PaddleOCR)**: Uses PyMuPDF for layout detection and text extraction from screenshots/figures. PaddleOCR is retained for robust, high-accuracy text extraction from crops.
* **Unanswerable Questions**: The LLM evaluates retrieved chunks and if it cannot confidently format steps using only the provided context, the system cleanly reports it cannot answer the query.

## Evaluation
**Metrics implemented:**
1. **Step Completeness & Ordering**: The final generated output is compared against the source manual for sequence integrity.
2. **Image-Step Pairing**: Measures if the assigned image accurately matches the instruction context.
3. **Grounding Verification**: All steps must map directly to an explicit page and figure number citation.

*To evaluate more systematically, I would run semantic similarity (e.g., using RAGAS) on generated text against a gold-standard annotated manual dataset.*

## Quickstart

1. **Install Dependencies (Python 3.12 Recommended)**:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

2. **Configure Environment**:
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
DATABASE_URL=sqlite:///./multimodal_rag.db
```

3. **Run Ingestion**:
Provide the `Dataset.pdf` (e.g., PT-LT100 Leak Testing Instrument manual) in the root.
```bash
python run_ingestion.py
```
*(This extracts text, crops images, runs OCR, generates embeddings, and saves to ChromaDB & SQLite).*

4. **Query the System**:
```bash
python scripts/test_query.py
```
Or start the FastAPI server:
```bash
python run.py
```

## Sample Questions Answered
* **Routine operation**: "How do I run a leak test on a blister pack, step by step?"
* **Calibration**: "How do I calibrate the vacuum function on the PT-LT100?"
* **Accessory setup**: "How do I connect an external balance and weigh the sample before/after a test?"
* **Maintenance**: "How do I replace the vacuum pump filter?"
* **Configuration**: "How do I set up user access control / log in as a user?"
* **Unanswerable**: "How do I update the instrument firmware over WI-FI?" (Will cleanly reject).
