"""
Pydantic models for the API
"""

from typing import List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """A single turn in the conversation history."""
    role: str # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    """
    Request model for the query endpoint.

    `history` contains all prior turns (user + assistant) in chronological
    order, *not* including the current `question`.  The backend appends
    the current question before invoking the LLM so the model sees the
    full conversation context.
    """
    question: str
    history: List[Message] = []


class Source(BaseModel):
    """A single document chunk that was retrieved to answer the question."""
    file: str
    path: str
    page: Optional[int] = None


class QueryResponse(BaseModel):
    """
    Response model for the query endpoint
    """
    answer: str
    sources: List[Source] = []


class IndexStatusResponse(BaseModel):
    """Whether documents are indexed and how many files are on disk."""
    indexed: bool
    file_count: int


class IndexResponse(BaseModel):
    """Result of building the in-memory index."""
    documents: int
    chunks: int