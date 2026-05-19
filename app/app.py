"""
FastAPI app for the RAG system.

Run through the main.py file.
"""

from pathlib import Path
from typing import List
from contextlib import asynccontextmanager
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag.rag import (
    count_uploaded_files,
    get_chain,
    get_uploaded_dir,
    index_chain,
    is_indexed,
    reset_chain,
    SUPPORTED_EXTENSIONS,
)
from src.llm.gemini import GeminiFlashLiteProvider

from app.schemas import (
    IndexResponse,
    IndexStatusResponse,
    QueryRequest,
    QueryResponse,
    Source,
)

ENABLE_PRINT_DEBUG = os.getenv("ENABLE_PRINT_DEBUG", "False").lower() == "true"
NUMBER_OF_SOURCES = int(os.getenv("RAG_NUMBER_OF_SOURCES", "4"))

# Compute repo root from this file location (`.../app/app.py` -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clear previously uploaded files so RAG starts fresh."""
    uploaded_dir = get_uploaded_dir()
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    # Remove contents (but keep the directory itself).
    for child in uploaded_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    # Start with no in-memory index; user must upload and index explicitly.
    reset_chain()
    yield
    # Ensure the in-memory index rebuilds on shutdown.
    reset_chain()


# Create the FastAPI app with lifespan handlers.
app = FastAPI(
    title="RAG Chatbot",
    description="LLM-powered chat application with retrieval over custom documents for grounded, context-aware responses.",
    lifespan=lifespan,
)

# Serve the frontend (HTML + JS) from repo-level `frontend/`.
_FRONTEND_DIR = _REPO_ROOT / "frontend"
app.mount(
    "/static",
    StaticFiles(directory=str(_FRONTEND_DIR / "static")),
    name="static",
)

# Frontend entrypoint
@app.get("/")
def index():
    """Frontend entrypoint."""
    return FileResponse(str(_FRONTEND_DIR / "templates/index.html"))

# Health check endpoint
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}

# Upload endpoint
@app.post("/upload")
def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple documents for indexing (supported: docx/pdf/txt/md)."""
    supported = SUPPORTED_EXTENSIONS
    uploaded_dir = get_uploaded_dir()
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    def _unique_dest(filename: str) -> Path:
        """
        Avoid overwriting when users upload multiple files with the same name.

        Browsers don't provide full folder paths for security, so collisions are
        possible (e.g. selecting `notes.txt` from two different folders).
        """
        dest = uploaded_dir / filename
        if not dest.exists():
            return dest

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        i = 1
        while True:
            candidate = uploaded_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate
            i += 1

    saved: List[str] = []
    skipped: List[str] = []

    for file in files:
        if not file.filename:
            continue

        filename = Path(file.filename).name  # prevent directory traversal
        suffix = Path(filename).suffix.lower()
        if suffix not in supported:
            skipped.append(filename)
            continue

        dest = _unique_dest(filename)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        saved.append(dest.name)

    if not saved:
        supported_str = ", ".join(sorted(supported))
        skipped_str = ", ".join(skipped) if skipped else "(no files saved)"
        raise HTTPException(
            status_code=400,
            detail=f"No supported files were uploaded. Supported: {supported_str}. Skipped: {skipped_str}.",
        )

    # New uploads invalidate the current index until the user re-indexes.
    reset_chain()
    return {"saved": saved, "skipped": skipped}


@app.get("/index/status", response_model=IndexStatusResponse)
def index_status() -> IndexStatusResponse:
    """Return whether documents are indexed and how many files are uploaded."""
    return IndexStatusResponse(
        indexed=is_indexed(),
        file_count=count_uploaded_files(),
    )


@app.post("/index", response_model=IndexResponse)
def index_documents() -> IndexResponse:
    """Build the in-memory vector store from uploaded files."""
    if count_uploaded_files() == 0:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"No supported files to index. Upload files first. Supported: {supported_str}.",
        )

    try:
        stats = index_chain(debug=ENABLE_PRINT_DEBUG, number_of_sources=NUMBER_OF_SOURCES)
        return IndexResponse(**stats)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query endpoint
## response_model tells FastAPI what Pydantic model the response should conform to.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Ask a question; returns an answer based on the indexed documents.

    Args:
        request: QueryRequest - The request containing the question and history.

    Returns:
        QueryResponse - The response containing the answer and sources.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        chain = get_chain()

        # Build the full messages list: prior turns + the current question.
        messages = [{"role": m.role, "content": m.content} for m in request.history]
        messages.append({"role": "user", "content": question})
        if ENABLE_PRINT_DEBUG:
            print("[DEBUG][app.py]: messages", messages)

        answer, raw_sources = chain.get_response(messages)
        sources = [Source(**s) for s in raw_sources]
        return QueryResponse(answer=answer, sources=sources)
    except RuntimeError as e:
        # Surface "no docs uploaded" type issues as a client error.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
