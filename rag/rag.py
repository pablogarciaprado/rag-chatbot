"""
RAG pipeline.

Implements:
  - load supported files from `context_files/` (populated via the FastAPI upload endpoint)
  - chunking
  - embeddings + in-memory vector store
  - dynamic prompt injection (retrieval happens in middleware)
  - a thin wrapper so `app.app` can call `get_chain().invoke(question)`
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

# Repo-level absolute imports.
from src.llm.base import BaseLLMProvider
from src.llm.gemini import GeminiFlashLiteProvider
from src.prompt.prompt_manager import build_prompt_middleware

# Resolve repository root from `.../rag/rag.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load repo-root `.env` when present.
load_dotenv(dotenv_path=str(_REPO_ROOT / ".env"), override=False)

UPLOADED_DIR = os.getenv(
    "RAG_UPLOADED_DIR",
    str(_REPO_ROOT / "context_files"),
)

SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt", ".md", ".pptx"}


def _ensure_nltk_resource_from_error(error: LookupError) -> bool:
    """
    Attempt to download a missing NLTK (Natural Language Toolkit) resource inferred from a LookupError.

    Returns True if a download was attempted successfully.
    """
    try:
        import nltk
    except Exception:
        return False

    match = re.search(r"Attempted to load '([^']+)'", str(error))
    if not match:
        return False

    attempted_path = match.group(1).strip("/")
    resource_name = attempted_path.split("/")[-1] if attempted_path else ""
    if not resource_name:
        return False

    try:
        nltk.download(resource_name, quiet=True)
        return True
    except Exception:
        return False


def _load_documents() -> List[Any]:
    """Load all supported files under UPLOADED_DIR."""
    from langchain_community.document_loaders import (
        Docx2txtLoader, # For .docx files
        PyPDFLoader, # For .pdf files
        TextLoader, # For .txt and .md files
        UnstructuredPowerPointLoader, # For .pptx files
    )

    base_dir = Path(UPLOADED_DIR)
    if not base_dir.exists():
        return []

    documents: List[Any] = []
    for path in sorted(base_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        if suffix == ".docx":
            loader = Docx2txtLoader(str(path))
            documents.extend(loader.load())
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
            documents.extend(loader.load())
        elif suffix == ".pptx":
            loader = UnstructuredPowerPointLoader(str(path))
            try:
                documents.extend(loader.load())
            except LookupError as e:
                # Unstructured's PPTX processing can require NLTK tagger data.
                if _ensure_nltk_resource_from_error(e):
                    documents.extend(loader.load())
                else:
                    raise
        else:
            # Treat .txt and .md as plain text.
            loader = TextLoader(str(path))
            documents.extend(loader.load())

    return documents

def _split_documents(documents: List[Any]) -> List[Any]:
    """Split documents into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def _ensure_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Set it in your environment or in repo-root `.env`."
        )
    return api_key

def _build_embeddings() -> Any:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
    )

def _build_vectorstore(chunks: List[Any]) -> Any:
    from langchain_core.vectorstores import InMemoryVectorStore

    chunk_texts = [doc.page_content for doc in chunks]
    embeddings = _build_embeddings()
    return InMemoryVectorStore.from_texts(
        chunk_texts,
        embedding=embeddings,
    )

class RagWrapper:
    """Thin wrapper so callers can do `invoke(messages) -> str`."""

    def __init__(self, agent_: Any):
        self._agent = agent_

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Accept the full conversation as a list of ``{"role": ..., "content": ...}``
        dicts and return the assistant's reply as a plain string.
        """
        payload: Dict[str, Any] = {"messages": messages}
        state = self._agent.invoke(payload)

        # LangChain agents usually return a state dict with `messages`.
        if isinstance(state, dict) and "messages" in state:
            last = state["messages"][-1]
            content = getattr(last, "content", None)
            if content:
                return content

        # Fallback: best-effort string conversion
        return str(state)


def build_rag_chain(llm_provider: Optional[BaseLLMProvider] = None, debug: bool = False):
    """
    Build the agent-based RAG app (matches notebook's create_agent + dynamic_prompt).

    Returns an object with `.invoke(messages: list[dict]) -> str`.
    """
    from langchain.agents import create_agent

    _ensure_google_api_key()

    if llm_provider is None:
        llm_provider = GeminiFlashLiteProvider()

    # Load documents that will provide context for the RAG system.
    documents = _load_documents()
    if not documents:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise RuntimeError(
            "No supported files found under "
            f"UPLOADED_DIR={UPLOADED_DIR!r}. Upload files via the UI. "
            f"Supported extensions: {supported}."
        )

    # Split documents into overlapping chunks.
    chunks = _split_documents(documents)

    if debug:
        print("[DEBUG]: chunks", len(chunks))
        print("[DEBUG]: chunks", chunks)

    # Build the vector store.
    ## We are using an in-memory vector store to store the chunks,
    ## this is not scalable for large datasets, but it is convenient for a simple application.
    vectorstore = _build_vectorstore(chunks)

    # Build the LLM.
    llm = llm_provider.build_llm()

    # Build the prompt middleware.
    ## This will inject the retrieved context into the prompt.
    middleware = build_prompt_middleware(vectorstore)

    # Build the agent.
    agent = create_agent(model=llm, tools=[], middleware=[middleware])

    # Return the RAG wrapper.
    return RagWrapper(agent)


# Lazy singleton (build on first request)
_CHAIN: Optional[Any] = None


def get_chain(llm_provider: Optional[BaseLLMProvider] = None, debug: bool = False):
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = build_rag_chain(llm_provider, debug)
    return _CHAIN

def reset_chain() -> None:
    """Force the in-memory RAG index to be rebuilt on next request."""
    global _CHAIN
    _CHAIN = None
