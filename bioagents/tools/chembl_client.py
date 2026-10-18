"""Robust client for the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data).

The EBI ChEMBL web services intermittently answer with ``500 Internal Server Error``
(an HTML error page) or stall for tens of seconds — the *same* URL can fail and then
succeed seconds later. A single-shot request with a short retry loop therefore fails
whole workflows for a transient outage. This client:

- uses split connect/read timeouts and an overall time budget per call;
- retries 429/5xx, connection errors, read timeouts and non-JSON (HTML error page)
  bodies with exponential backoff + jitter, honouring ``Retry-After``;
- fails fast on other 4xx (a malformed query will not fix itself), with the status,
  URL and server message in the error;
- follows ``page_meta.next`` for pagination (ChEMBL caps ``limit`` at 1000).

It also fixes two query mistakes that silently returned wrong data:

- range filters need a double underscore (``standard_value__lte``). ChEMBL silently
  ignores unknown parameters, so ``standard_value_lte`` returned *all* potencies;
- ``standard_type__in`` matching is case-sensitive: ``KI``/``KD`` match only a
  fraction of the ``Ki``/``Kd`` records, so types are normalised to ChEMBL's spelling.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import requests

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_CHEMBL_HOST = "https://www.ebi.ac.uk"

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 1.5
DEFAULT_BACKOFF_CAP = 30.0
DEFAULT_TIME_BUDGET = 180.0
MAX_PAGE_SIZE = 1000
MOLECULE_BATCH_SIZE = 50

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# ChEMBL's canonical spelling of common activity types (filters are case-sensitive).
_CANONICAL_TYPES = {
    t.upper(): t for t in ("IC50", "Ki", "Kd", "EC50", "AC50", "GI50", "Potency", "Inhibition")
}

# JSON field holding the records of each list endpoint.
_LIST_KEYS = {"activity": "activities", "molecule": "molecules", "target": "targets"}

_CHEMBL_ID_RE = re.compile(r"^CHEMBL\d+$")


class ChemblError(RuntimeError):
    """A ChEMBL request failed; the message says why and whether a retry may help."""

    def __init__(self, message: str, *, status: int | None = None, transient: bool = False):
        super().__init__(message)
        self.status = status
        self.transient = transient


def normalise_activity_types(activity_types: str | list[str]) -> list[str]:
    """Map user-supplied activity types to ChEMBL's case-sensitive spelling."""
    items = activity_types.split(",") if isinstance(activity_types, str) else activity_types
    out: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key:
            continue
        canonical = _CANONICAL_TYPES.get(key.upper(), key)
        if canonical not in out:
            out.append(canonical)
    return out


def is_chembl_id(value: str) -> bool:
    return bool(_CHEMBL_ID_RE.match(str(value).strip().upper()))


def _body_snippet(resp: requests.Response, limit: int = 200) -> str:
    text = resp.text or ""
    if "<html" in text[:500].lower():
        title = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
        return f"HTML error page: {title.group(1).strip()}" if title else "HTML error page"
    return re.sub(r"\s+", " ", text)[:limit]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    backoff_cap: float = DEFAULT_BACKOFF_CAP

    def delay(self, attempt: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_cap)
            except ValueError:
                pass
        raw = self.backoff_base * (2**attempt)
        return float(min(raw, self.backoff_cap) * (0.75 + random.random() / 2))  # nosec B311


class ChemblClient:
    """Thin, retrying wrapper over the ChEMBL REST API."""

    def __init__(
        self,
        base_url: str = CHEMBL_BASE_URL,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        retry: RetryPolicy | None = None,
        time_budget: float = DEFAULT_TIME_BUDGET,
        session: requests.Session | None = None,
        sleep: Any = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = (connect_timeout, read_timeout)
        self.retry = retry or RetryPolicy()
        self.time_budget = time_budget
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "BioAgents-ChEMBL-client/1.0"}
        )
        self._sleep = sleep

    def _url(self, path: str) -> str:
        if path.startswith("http"):
            return path
        if path.startswith("/chembl/"):
            return urljoin(_CHEMBL_HOST, path)
        return f"{self.base_url}/{path.lstrip('/')}"

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``path`` and return parsed JSON, retrying transient failures."""
        url = self._url(path)
        deadline = time.monotonic() + self.time_budget
        last_error = "no attempt made"
        attempts = self.retry.max_retries + 1
        for attempt in range(attempts):
            retry_after = None
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    try:
                        payload: dict[str, Any] = resp.json()
                        return payload
                    except ValueError:
                        last_error = f"HTTP 200 but non-JSON body ({_body_snippet(resp)})"
                elif resp.status_code in RETRYABLE_STATUS:
                    retry_after = resp.headers.get("Retry-After")
                    last_error = f"HTTP {resp.status_code} ({_body_snippet(resp)})"
                else:
                    raise ChemblError(
                        f"ChEMBL rejected the request: HTTP {resp.status_code} for {resp.url} "
                        f"({_body_snippet(resp)}). Check the query parameters / IDs.",
                        status=resp.status_code,
                    )
            if attempt == attempts - 1:
                break
            delay = self.retry.delay(attempt, retry_after)
            if time.monotonic() + delay >= deadline:
                last_error += f"; time budget of {self.time_budget:.0f}s exhausted"
                break
            logger.warning(
                "ChEMBL request %s failed (%s); retry %d/%d in %.1fs",
                url,
                last_error,
                attempt + 1,
                self.retry.max_retries,
                delay,
            )
            self._sleep(delay)
        raise ChemblError(
            f"ChEMBL API unavailable after {attempt + 1} attempt(s) for {url} "
            f"params={params or {}}: {last_error}. This is a server-side/network failure "
            "at EBI, not a query error — retry later.",
            transient=True,
        )

    def iter_pages(
        self, resource: str, params: dict[str, Any], *, max_records: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield records from a paginated list endpoint, following ``page_meta.next``."""
        key = _LIST_KEYS.get(resource, f"{resource}s")
        page_params: dict[str, Any] | None = {
            **params,
            "limit": min(int(params.get("limit", MAX_PAGE_SIZE)), MAX_PAGE_SIZE),
        }
        path: str | None = f"{resource}.json"
        seen = 0
        while path:
            data = self.get_json(path, page_params)
            records = data.get(key)
            if records is None:
                raise ChemblError(f"Unexpected ChEMBL response for {resource}: no '{key}' field")
            for rec in records:
                yield rec
                seen += 1
                if max_records is not None and seen >= max_records:
                    return
            path = (data.get("page_meta") or {}).get("next")
            page_params = None  # ``next`` already carries the query string

    def get_target(self, target_chembl_id: str) -> dict[str, Any] | None:
        """Return a target record, or None if ChEMBL has no such target."""
        try:
            return self.get_json(f"target/{target_chembl_id.strip().upper()}.json")
        except ChemblError as exc:
            if exc.status == 404:
                return None
            raise

    def search_targets(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self.get_json("target/search.json", {"q": query, "limit": limit})
        return list(data.get("targets") or [])

    def resolve_target(
        self, target_chembl_id: str | None, target_name: str | None, organism: str = "Homo sapiens"
    ) -> dict[str, Any]:
        """Resolve a target by ID, falling back to a name search (single protein first)."""
        if target_chembl_id and is_chembl_id(target_chembl_id):
            target = self.get_target(target_chembl_id)
            if target:
                return target
        if not target_name:
            raise ChemblError(f"ChEMBL target '{target_chembl_id}' not found and no name given")
        hits = self.search_targets(target_name)
        if not hits:
            raise ChemblError(f"No ChEMBL target matches '{target_name}'")

        def rank(t: dict[str, Any]) -> tuple[int, int]:
            return (
                0 if t.get("target_type") == "SINGLE PROTEIN" else 1,
                0 if t.get("organism") == organism else 1,
            )

        return sorted(hits, key=rank)[0]

    def get_activities(
        self,
        target_chembl_id: str,
        *,
        activity_types: str | list[str] = "IC50",
        max_value_nM: float | None = None,
        max_records: int | None = 5000,
    ) -> list[dict[str, Any]]:
        """Return exact/upper-bounded ('=', '<', '<=') nM activities for a target."""
        params: dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "standard_units": "nM",
            "standard_relation__in": "=,<,<=",
            "standard_value__isnull": "false",
        }
        types = normalise_activity_types(activity_types)
        if types:
            params["standard_type__in"] = ",".join(types)
        if max_value_nM is not None:
            params["standard_value__lte"] = float(max_value_nM)
        return list(self.iter_pages("activity", params, max_records=max_records))

    def get_molecules(self, chembl_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch-fetch molecule records (MOLECULE_BATCH_SIZE IDs per request)."""
        out: dict[str, dict[str, Any]] = {}
        ids = list(dict.fromkeys(chembl_ids))
        for i in range(0, len(ids), MOLECULE_BATCH_SIZE):
            chunk = ids[i : i + MOLECULE_BATCH_SIZE]
            params = {"molecule_chembl_id__in": ",".join(chunk), "limit": len(chunk)}
            for mol in self.iter_pages("molecule", params):
                mid = mol.get("molecule_chembl_id")
                if mid:
                    out[mid] = mol
        return out


def canonical_smiles(molecule: dict[str, Any]) -> str | None:
    structures = molecule.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    return smiles or None
