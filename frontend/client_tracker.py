"""Client fingerprinting and identification.

Generates privacy-conscious client IDs from IP + User-Agent hashes
(never stores raw IPs). Extracts client metadata from FastAPI requests.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)


def generate_client_id(ip_address: str, user_agent: str) -> str:
    """Generate a deterministic 16-char client ID from IP + User-Agent.

    Only stores hashed values — never raw PII.
    """
    raw = f"{ip_address}:{user_agent}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_client_info(request: Request) -> dict[str, Any]:
    """Extract client metadata from a FastAPI request.

    Handles X-Forwarded-For for proxy environments.
    """
    # Handle proxied requests: X-Forwarded-For may contain a chain
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        # Take the first (original client) IP from the chain
        ip_address = forwarded_for.split(",")[0].strip()
    elif request.client:
        ip_address = request.client.host
    else:
        ip_address = "unknown"

    user_agent = request.headers.get("user-agent", "unknown")
    accept_language = request.headers.get("accept-language", "")

    return {
        "ip_address": ip_address,
        "user_agent": user_agent,
        "ip_hash": hashlib.sha256(ip_address.encode()).hexdigest()[:16],
        "user_agent_hash": hashlib.sha256(user_agent.encode()).hexdigest()[:16],
        "client_id": generate_client_id(ip_address, user_agent),
        "accept_language": accept_language,
    }


def get_or_create_client(
    db: Any,
    client_id: str,
    ip_hash: str,
    ua_hash: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Upsert a client record and return the current state."""
    db.upsert_client(
        client_id=client_id,
        ip_hash=ip_hash,
        user_agent_hash=ua_hash,
        metadata=metadata,
    )
    return db.get_client(client_id)
