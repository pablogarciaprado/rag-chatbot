"""
Pydantic models for the API
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """
    Request model for the query endpoint
    """
    question: str

class QueryResponse(BaseModel):
    """
    Response model for the query endpoint
    """
    answer: str