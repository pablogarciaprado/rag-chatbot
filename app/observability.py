"""
Logfire observability for the FastAPI RAG application.

Call `prepare_langchain_otel()` before any LangChain import.
Call `setup_observability(app)` after the FastAPI app is created.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import logfire

if TYPE_CHECKING:
    from fastapi import FastAPI


def _logfire_enabled() -> bool:
    return os.getenv("LOGFIRE_ENABLED", "true").lower() in ("true", "1", "yes")


def prepare_langchain_otel() -> None:
    """
    Enable LangChain/LangGraph OpenTelemetry export to Logfire.

    Must run before LangChain is imported (see app/app.py import order).
    """
    if not _logfire_enabled():
        return
    if os.getenv("LOGFIRE_INSTRUMENT_LANGCHAIN", "true").lower() not in ("true", "1", "yes"):
        return

    os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
    # LANGSMITH_OTEL_ONLY=true sends traces to Logfire via OTEL, not LangSmith’s hosted product
    os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")


def setup_observability(app: FastAPI) -> None:
    """Configure Logfire and instrument FastAPI and outbound HTTP (Gemini API)."""
    if not _logfire_enabled():
        return

    logfire.configure(
        service_name=os.getenv("LOGFIRE_SERVICE_NAME", "rag-chatbot"),
    )
    logfire.instrument_fastapi(
        app,
        excluded_urls=r"/health,/static/.*",
    )
    logfire.instrument_httpx()
