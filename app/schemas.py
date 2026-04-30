"""
Pydantic models for the API
"""

from typing import List

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


class QueryResponse(BaseModel):
    """
    Response model for the query endpoint
    """
    answer: str