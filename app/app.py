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

from rag.rag import get_chain, reset_chain
from src.llm.gemini import GeminiFlashLiteProvider

from app.schemas import QueryRequest, QueryResponse

# Compute repo root from this file location (`.../app/app.py` -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Upload storage defaults to `<repo>/context_files/`.
def _uploaded_dir() -> Path:
    return Path(
        os.getenv(
            "RAG_UPLOADED_DIR",
            str(_REPO_ROOT / "context_files"),
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clear previously uploaded files so RAG starts fresh."""
    uploaded_dir = _uploaded_dir()
    uploaded_dir.mkdir(parents=True, exist_ok=True)

    # Remove contents (but keep the directory itself).
    for child in uploaded_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    # Ensure the in-memory index rebuilds on first `/query` after restart.
    reset_chain()
    yield
    # Ensure the in-memory index rebuilds on shutdown.
    reset_chain()


# Create the FastAPI app with lifespan handlers.
app = FastAPI(
    title="Locaria's Intelligence Layer",
    description="Query your documents using retrieval-augmented generation. Powered by LangChain and Google Vertex AI.",
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
    supported = {".docx", ".pdf", ".txt", ".md"}
    uploaded_dir = _uploaded_dir()
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

    # Next `/query` should rebuild the in-memory vector store.
    reset_chain()
    return {"saved": saved, "skipped": skipped}

# Query endpoint
## response_model tells FastAPI what Pydantic model the response should conform to.
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Ask a question; returns an answer based on the indexed documents."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        llm_provider = GeminiFlashLiteProvider() # this could be changed depending on the provider to use
        chain = get_chain(llm_provider)
        answer = chain.invoke(question)
        return QueryResponse(answer=answer)
    except RuntimeError as e:
        # Surface "no docs uploaded" type issues as a client error.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
