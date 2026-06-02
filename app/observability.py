"""
Logfire observability for the FastAPI RAG application.

Call `prepare_langchain_otel()` before any LangChain import.
Call `setup_observability(app)` after the FastAPI app is created.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import logfire
from starlette.requests import Request
from starlette.responses import Response

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


def install_user_context_middleware(
    app: FastAPI,
    get_user_id: Callable[[Request], str],
) -> None:
    """
    Attach user_id to every trace via Logfire baggage.

    Args:
        app: The FastAPI app.
        get_user_id: A function that returns the user id for the current request.
    """

    @app.middleware("http")
    async def _attach_user(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        with logfire.set_baggage(user_id=get_user_id(request)):
            return await call_next(request)


def setup_observability(app: FastAPI) -> None:
    """Configure Logfire and instrument FastAPI and outbound HTTP (Gemini API)."""
    if not _logfire_enabled():
        return

    from app.auth import get_current_user_id

    logfire.configure(
        service_name=os.getenv("LOGFIRE_SERVICE_NAME", "rag-chatbot"),
    )
    logfire.instrument_fastapi(
        app,
        excluded_urls=r"/health,/static/.*",
    )
    logfire.instrument_httpx()

    # Setup Logfire user context middleware.
    install_user_context_middleware(app, get_current_user_id)
