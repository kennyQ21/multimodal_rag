# Multimodal RAG for Image-Heavy Technical Manuals

A multimodal Retrieval-Augmented Generation (RAG) system built to answer how-to questions from scanned, image-heavy operating manuals with ordered, step-by-step illustrated procedures.

## Key Features
* **Illustrated Steps**: Interleaves extracted text instructions with exact matching screenshots from the manual.
* **Hybrid Retrieval**: Relies on Okapi BM25 and SQLite to bypass heavy vector database constraints and ensure maximum local environment compatibility.
* **Source Grounding**: Every instruction and image is strictly tied to a specific page and figure label.
* **Local Parsing**: Uses PyMuPDF for layout detection and PaddleOCR for local, CPU-optimized text extraction from screenshots/figures.
* **Groq Generation**: Uses `llama-3.3-70b-versatile` exclusively for answer composition and formatting (zero-hallucination instruction selection).

---

## 🏗️ Detailed Architecture

This application operates entirely locally (except for the LLM generation) using lightweight components designed for Windows compatibility without relying on Docker or complex GPU frameworks.

### 1. Ingestion Pipeline (`run_ingestion.py`)
* **Document Parsing (PyMuPDF)**: Reads the technical manual (`Dataset.pdf`), detects page dimensions, and strictly crops the visual contents. Each page is converted to an image and saved locally in `data/crops/`.
* **Text Extraction (PaddleOCR)**: Extracts raw text directly from the cropped images. To bypass severe Windows DLL dependency errors on modern Python versions (e.g. 3.12/3.13), it utilizes a specialized `sitecustomize.py` DLL loading fix and a stable CPU build of `paddlepaddle`.
* **Storage (SQLite)**: Saves the page metadata (page number, image path, OCR text) in a local relational database (`multimodal_rag.db`).

### 2. Retrieval & RAG Pipeline (`app/query/pipeline.py`)
* **Lexical Search (BM25)**: Because local PyTorch and ONNX models can crash unpredictably on Windows environments missing C++ dependencies, we strictly use `rank_bm25`. This retrieves the most textually relevant pages based on the user's query.
* **LangGraph Orchestration**: The AI workflow is managed by LangGraph. It compiles the retrieved text and passes it to Groq API.
* **Answer Composition (Groq)**: The LLM reads the retrieved text and structures a step-by-step guide. It is strictly prompted to **only** use the provided context. If the query cannot be answered using the extracted manual text, the LLM safely replies: *"There are no steps provided."*

---

## 🚀 Setup and Start Guide (Windows + CPU)

The project relies on a very specific setup to guarantee stable local execution on Windows laptops, bypassing common PaddleOCR DLL loading failures.

### 1. Create a Clean Virtual Environment

Open **PowerShell** and initialize the project:

```powershell
# Clone the repository
git clone https://github.com/kennyQ21/multimodal_rag.git
cd multimodal_rag

# Create a clean Python 3.12 environment
py -3.12 -m venv .venv
.venv\Scripts\activate

# Upgrade base tooling
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install PaddlePaddle (CPU)

Install the CPU build **first** using the stable Chinese mirror to prevent dependency conflicts:

```powershell
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Verify the installation:
```powershell
python -c "import paddle; print(paddle.__version__)"
# Expected output: 3.2.0
```

### 3. Install PaddleOCR and Requirements

```powershell
# Install PaddleOCR standalone (avoids heavy optional features)
python -m pip install paddleocr

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Ensure Windows DLL Loading Fix is Active

The repository includes a `sitecustomize.py` file in the root. Python automatically imports this at startup on Windows to force the environment to properly discover `libpaddle.pyd` C++ dependencies. **Do not delete this file.**

### 5. Configure Environment Variables

Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
DATABASE_URL=sqlite:///./multimodal_rag.db
```

### 6. Cache Weights and Run Ingestion

Place `Dataset.pdf` in the root folder, then run the ingestion pipeline. On the first run, PaddleOCR will download and cache the necessary inference weights.

```powershell
python run_ingestion.py
```
*(This will populate `data/crops/` with images and build `multimodal_rag.db`).*

### 7. Start the Application / Query

To query the terminal directly:
```powershell
python scripts/test_query.py
```

To start the FastAPI server:
```powershell
python run.py
```

---

## 🧪 Evaluation

System metrics, architectural tradeoffs, and testing outcomes have been extensively documented in `evaluation.md`.
