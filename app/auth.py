"""User identity for observability and (future) authorization."""

from __future__ import annotations

import os

from starlette.requests import Request

# Placeholder until real authentication is implemented.
DEFAULT_USER_ID = "dev-user"


def get_current_user_id(request: Request) -> str:
    """
    Return the user id for the current request.

    Override with LOGFIRE_USER_ID in the environment. Replace this with JWT,
    session, or API-key lookup when auth is added.
    """
    del request  # unused until auth is implemented
    return os.getenv("LOGFIRE_USER_ID", DEFAULT_USER_ID)
