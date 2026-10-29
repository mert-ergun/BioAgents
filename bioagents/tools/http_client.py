"""Small HTTPS client for tools that contact a fixed set of public APIs."""

from __future__ import annotations

import math
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


def _validate_url(url: str, allowed_hosts: Collection[str]) -> None:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("API URL must use HTTPS and an explicitly allowed host")


class _AllowedRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: Collection[str]):
        super().__init__()
        self.allowed_hosts = frozenset(allowed_hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl, self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def read_https_text(
    url: str,
    *,
    allowed_hosts: Collection[str],
    timeout: float = 60,
    headers: Mapping[str, str] | None = None,
    data: bytes | None = None,
) -> str:
    """Read UTF-8 text, enforcing the same HTTPS/host policy on every redirect.

    Keep urllib's HTTP/URL exceptions so callers retain their retry behavior.
    Default TLS certificate verification remains enabled.
    """
    hosts = frozenset(allowed_hosts)
    _validate_url(url, hosts)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("API timeout must be finite and positive")
    request = urllib.request.Request(url, data=data, headers=dict(headers or {}))
    opener = urllib.request.build_opener(_AllowedRedirects(hosts))
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
    if not isinstance(body, bytes):
        raise ValueError("API response body must be bytes")
    return body.decode("utf-8", errors="replace")
