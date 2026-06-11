"""Admin authentication using HMAC-SHA256 tokens.

Zero external dependencies — uses stdlib hmac + hashlib.
Tokens are HMAC-signed with the admin password + a server salt,
include an expiry timestamp, and are verified with timing-safe comparison.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import time

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")
_TOKEN_SALT = "bioagents_admin_salt_v1"  # nosec B105 - HMAC salt, not a password
_TOKEN_EXPIRY_SECONDS = 86400  # 24 hours

if ADMIN_PASSWORD == "changeme":  # nosec B105 - comparing against known default
    logger.warning(
        "ADMIN_PASSWORD is set to the default value 'changeme'. "
        "Set the ADMIN_PASSWORD environment variable for production."
    )

_bearer_scheme = HTTPBearer(auto_error=False)


class AdminLoginRequest(BaseModel):
    """Request body for admin login."""

    password: str


class AdminLoginResponse(BaseModel):
    """Response body for successful admin login."""

    token: str
    expires_at: float


def _compute_signature(payload: str) -> str:
    """Compute HMAC-SHA256 signature for a payload string."""
    key = f"{ADMIN_PASSWORD}:{_TOKEN_SALT}".encode()
    return hmac.new(key, payload.encode(), "sha256").hexdigest()


def create_admin_token(password: str) -> str:
    """Create an admin token if the password matches.

    Returns a base token string containing JSON payload + HMAC signature.
    Raises ValueError on incorrect password.
    """
    if not hmac.compare_digest(password, ADMIN_PASSWORD):
        raise ValueError("Invalid admin password")

    expires_at = time.time() + _TOKEN_EXPIRY_SECONDS
    payload = json.dumps({"exp": expires_at}, separators=(",", ":"))
    signature = _compute_signature(payload)
    return f"{payload}.{signature}"


def verify_admin_token(token: str) -> bool:
    """Verify an admin token's signature and expiry."""
    try:
        if "." not in token:
            return False
        payload_str, signature = token.rsplit(".", 1)
        expected_sig = _compute_signature(payload_str)
        if not hmac.compare_digest(signature, expected_sig):
            return False
        payload = json.loads(payload_str)
        return not time.time() > payload.get("exp", 0)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


async def get_admin_dependency(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency that validates admin Bearer tokens.

    Returns the token string on success, raises 401 on failure.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_admin_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
