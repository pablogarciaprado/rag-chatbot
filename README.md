# RAG Chatbot

FastAPI-based Retrieval-Augmented Generation (RAG) application with:

- a document upload endpoint
- a question-answering endpoint over uploaded documents
- a minimal browser UI served by the API

The app builds an in-memory vector store from files in `context_files/` and injects retrieved context into the system prompt before generating answers.

## Project Structure

- `main.py`: app entrypoint (`uvicorn.run("app.app:app", ...)`)
- `app/`: FastAPI routes and request/response schemas
- `rag/`: RAG pipeline composition and chain lifecycle
- `src/llm/`: LLM provider interface and Gemini implementation
- `src/prompt/`: prompt middleware and `system_prompt.txt`
- `frontend/`: static UI assets and HTML template
- `context_files/`: uploaded/source documents used for retrieval

## Requirements

- Python 3.10+
- A `GOOGLE_API_KEY` available in environment variables or repo-root `.env`

## Installation

### Option A: Conda (recommended)

```bash
conda create -n rag-chatbot python=3.11 -y
conda activate rag-chatbot
pip install -e .
```

### Option B: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

- `GOOGLE_API_KEY`: required for Gemini model calls
- `RAG_UPLOADED_DIR` (optional): override the default document directory (`context_files/`)

Example `.env`:

```bash
GOOGLE_API_KEY=your_key_here
RAG_UPLOADED_DIR=context_files
```

## Run

From the repository root:

```bash
python3 main.py
```

Then open:

- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Frontend UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## API Endpoints

- `GET /health`
  - Returns service health status.
- `POST /upload`
  - Upload one or more files for indexing.
  - Supported extensions: `.docx`, `.pdf`, `.txt`, `.md`
- `POST /query`
  - Ask a question over the indexed document set.
  - Request body:

    ```json
    {
      "question": "Your question here"
    }
    ```

  - Response body:

    ```json
    {
      "answer": "Model response"
    }
    ```

## Typical Local Workflow

1. Start the server (`python3 main.py`)
2. Open `http://127.0.0.1:8000/`
3. Upload files in the UI (or place them in `context_files/`)
4. Ask questions from the UI or via `POST /query`

## Notes

- The vector store is in-memory and rebuilt when needed (`reset_chain()`), so this setup is best for local development and prototyping.
- On startup, the app clears the configured upload directory to start with a clean document set.
