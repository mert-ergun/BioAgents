"""Tests for client fingerprinting and identification."""

from __future__ import annotations

import hashlib
from unittest.mock import Mock

from frontend.client_tracker import extract_client_info, generate_client_id, get_or_create_client


class TestGenerateClientId:
    def test_consistent_ids(self) -> None:
        """Same inputs should produce the same client ID."""
        id1 = generate_client_id("192.168.1.1", "Mozilla/5.0")
        id2 = generate_client_id("192.168.1.1", "Mozilla/5.0")
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self) -> None:
        """Different inputs should produce different client IDs."""
        id1 = generate_client_id("192.168.1.1", "Mozilla/5.0")
        id2 = generate_client_id("10.0.0.1", "Chrome/100")
        assert id1 != id2

    def test_id_is_16_chars_hex(self) -> None:
        """Client ID should be a 16-char hex string."""
        cid = generate_client_id("127.0.0.1", "test-agent")
        assert len(cid) == 16
        assert all(c in "0123456789abcdef" for c in cid)

    def test_same_ip_different_ua(self) -> None:
        """Same IP but different User-Agent should produce different IDs."""
        id1 = generate_client_id("10.0.0.1", "Chrome")
        id2 = generate_client_id("10.0.0.1", "Firefox")
        assert id1 != id2

    def test_empty_inputs(self) -> None:
        """Empty strings should still produce a valid ID."""
        cid = generate_client_id("", "")
        assert len(cid) == 16

    def test_matches_hashlib_directly(self) -> None:
        """Verify the ID matches what hashlib.sha256 would produce."""
        raw = "127.0.0.1:Mozilla/5.0"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert generate_client_id("127.0.0.1", "Mozilla/5.0") == expected


def _make_request(
    ip: str = "127.0.0.1",
    user_agent: str = "TestAgent/1.0",
    forwarded_for: str | None = None,
    accept_language: str = "en-US",
) -> Mock:
    """Create a mock FastAPI Request with configurable headers."""
    request = Mock()
    request.client = Mock()
    request.client.host = ip
    headers: dict[str, str] = {"user-agent": user_agent}
    if forwarded_for is not None:
        headers["x-forwarded-for"] = forwarded_for
    if accept_language:
        headers["accept-language"] = accept_language
    request.headers = headers
    return request


class TestExtractClientInfo:
    def test_basic_extraction(self) -> None:
        """Extract info from a normal request."""
        request = _make_request()
        info = extract_client_info(request)
        assert info["ip_address"] == "127.0.0.1"
        assert info["user_agent"] == "TestAgent/1.0"
        assert info["accept_language"] == "en-US"
        assert len(info["client_id"]) == 16
        assert len(info["ip_hash"]) == 16
        assert len(info["user_agent_hash"]) == 16

    def test_x_forwarded_for_single(self) -> None:
        """X-Forwarded-For header should be used instead of client.host."""
        request = _make_request(ip="10.0.0.1", forwarded_for="203.0.113.1")
        info = extract_client_info(request)
        assert info["ip_address"] == "203.0.113.1"

    def test_x_forwarded_for_chain(self) -> None:
        """When X-Forwarded-For has multiple IPs, use the first (original client)."""
        request = _make_request(
            ip="10.0.0.1", forwarded_for="203.0.113.1, 70.41.3.18, 150.172.238.178"
        )
        info = extract_client_info(request)
        assert info["ip_address"] == "203.0.113.1"

    def test_x_forwarded_for_with_whitespace(self) -> None:
        """X-Forwarded-For entries with spaces should be trimmed."""
        request = _make_request(ip="10.0.0.1", forwarded_for="  203.0.113.1 , 70.41.3.18  ")
        info = extract_client_info(request)
        assert info["ip_address"] == "203.0.113.1"

    def test_no_user_agent_fallback(self) -> None:
        """Missing User-Agent should fall back to 'unknown'."""
        request = _make_request()
        del request.headers["user-agent"]
        info = extract_client_info(request)
        assert info["user_agent"] == "unknown"

    def test_no_client_object(self) -> None:
        """Missing request.client should fall back to 'unknown' IP."""
        request = Mock()
        request.client = None
        request.headers = {"user-agent": "Test"}
        info = extract_client_info(request)
        assert info["ip_address"] == "unknown"

    def test_hashes_are_deterministic(self) -> None:
        """Same inputs should produce same hashes across calls."""
        request1 = _make_request()
        request2 = _make_request()
        info1 = extract_client_info(request1)
        info2 = extract_client_info(request2)
        assert info1["client_id"] == info2["client_id"]
        assert info1["ip_hash"] == info2["ip_hash"]

    def test_hashes_differ_from_raw_ip(self) -> None:
        """Stored hashes should NOT equal the raw IP address."""
        request = _make_request(ip="192.168.1.1")
        info = extract_client_info(request)
        assert info["ip_hash"] != "192.168.1.1"


class TestGetOrCreateClient:
    def test_creates_new_client(self) -> None:
        """Should create and return a new client record."""
        from frontend.admin_database import AdminDatabase

        db = AdminDatabase(":memory:")
        result = get_or_create_client(db, "c1", "ip_hash", "ua_hash")
        assert result is not None
        assert result["client_id"] == "c1"

    def test_updates_existing_client(self) -> None:
        """Should update an existing client's last_seen and total_requests."""
        from frontend.admin_database import AdminDatabase

        db = AdminDatabase(":memory:")
        get_or_create_client(db, "c1", "ip_hash", "ua_hash")
        result = get_or_create_client(db, "c1", "ip_hash", "ua_hash", metadata={"key": "val"})
        assert result is not None
        assert result["total_requests"] == 2
