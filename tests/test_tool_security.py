"""Offline regressions for reviewed Bandit findings in custom tools."""

import sys
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from bioagents.tools import http_client
from bioagents.tools.custom_tools import esm1b_fitness_scores as esm
from bioagents.tools.custom_tools import mutation_literature_comparison as comparison
from bioagents.tools.custom_tools import ncbi_blastp as blast
from bioagents.tools.custom_tools import pdb_fetch_structure as pdb
from bioagents.tools.custom_tools import pubmed_search_abstracts as pubmed
from bioagents.tools.custom_tools import uniprot_get_features as uniprot


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://api.example/data",
        "ftp://api.example/data",
        "https://api.example.evil/data",
        "https://evil.example/api.example",
        "https://user:password@api.example/data",
        "https://api.example:444/data",
        "https://api.example:invalid/data",
        "https:///missing-host",
    ],
)
def test_api_client_rejects_unapproved_urls_before_opening_a_connection(monkeypatch, url):
    build = Mock()
    monkeypatch.setattr(http_client.urllib.request, "build_opener", build)
    with pytest.raises(ValueError):
        http_client.read_https_text(url, allowed_hosts={"api.example"})
    build.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_api_client_rejects_invalid_deadlines_before_connecting(monkeypatch, timeout):
    build = Mock()
    monkeypatch.setattr(http_client.urllib.request, "build_opener", build)
    with pytest.raises(ValueError, match="timeout"):
        http_client.read_https_text(
            "https://api.example/data", allowed_hosts={"api.example"}, timeout=timeout
        )
    build.assert_not_called()


def test_api_client_preserves_post_payload_headers_timeout_and_decoding(monkeypatch):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b"result \xff"
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(http_client.urllib.request, "build_opener", lambda handler: opener)
    result = http_client.read_https_text(
        "https://api.example/search",
        allowed_hosts={"api.example"},
        data=b"query=encoded",
        headers={"Accept": "application/json"},
        timeout=12,
    )
    assert result == "result \ufffd"
    request = opener.open.call_args.args[0]
    assert request.get_method() == "POST"
    assert request.data == b"query=encoded"
    assert request.get_header("Accept") == "application/json"
    assert opener.open.call_args.kwargs["timeout"] == 12
    response.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "target",
    [
        "http://api.example/next",
        "https://evil.example/next",
        "file:///etc/passwd",
        "https://api.example:444/next",
        "https://user@api.example/next",
    ],
)
def test_redirects_cannot_bypass_https_host_or_port_policy(target):
    handler = http_client._AllowedRedirects({"api.example"})
    request = urllib.request.Request("https://api.example/start")
    with pytest.raises(ValueError):
        handler.redirect_request(request, None, 302, "Found", {}, target)


def test_redirects_to_an_approved_https_host_still_work():
    handler = http_client._AllowedRedirects({"api.example"})
    request = urllib.request.Request("https://api.example/start")
    result = handler.redirect_request(request, None, 302, "Found", {}, "https://api.example/next")
    assert result.full_url == "https://api.example/next"


def test_pubmed_keeps_retryable_http_error_behavior(monkeypatch):
    get = Mock(
        side_effect=[
            urllib.error.HTTPError(
                "https://eutils.ncbi.nlm.nih.gov", 429, "rate limit", None, None
            ),
            '{"count": 1}',
        ]
    )
    monkeypatch.setattr(pubmed, "read_https_text", get)
    monkeypatch.setattr(pubmed.time, "sleep", lambda _: None)
    result = pubmed._http_get_with_retry(pubmed.EUTILS_BASE + "/esearch.fcgi", {"term": "x&y"})
    assert result == '{"count": 1}'
    assert get.call_count == 2
    assert "term=x%26y" in get.call_args.args[0]
    assert get.call_args.kwargs["allowed_hosts"] == {"eutils.ncbi.nlm.nih.gov"}


@pytest.mark.parametrize("module", [uniprot, comparison])
def test_remote_json_object_shape_is_checked_before_downstream_processing(monkeypatch, module):
    monkeypatch.setattr(module, "read_https_text", lambda *args, **kwargs: "[]")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises((ValueError, RuntimeError), match="object"):
        module._fetch_json("https://rest.uniprot.org/data", timeout=1, retries=1)


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE BlastOutput [<!ENTITY value "expanded">]><BlastOutput>&value;</BlastOutput>',
        '<!DOCTYPE BlastOutput [<!ENTITY value SYSTEM "file:///etc/passwd">]><BlastOutput>&value;</BlastOutput>',
        '<!DOCTYPE BlastOutput [<!ENTITY % value SYSTEM "https://evil.example/entity">%value;]><BlastOutput/>',
        "<BlastOutput>",
    ],
)
def test_blast_xml_rejects_entities_external_references_and_malformed_documents(xml):
    with pytest.raises(RuntimeError, match="Failed to parse NCBI BLAST XML"):
        blast._parse_blast_xml(xml, "test-rid")


def test_blast_public_dtd_declaration_and_normal_xml_remain_compatible():
    xml = """<?xml version="1.0"?>
<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "https://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">
<BlastOutput><BlastOutput_program>blastp</BlastOutput_program></BlastOutput>"""
    root = blast._parse_blast_xml(xml, "test-rid")
    assert root.findtext("BlastOutput_program") == "blastp"


def test_transformers_model_and_tokenizer_use_the_same_pinned_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(
        esm, "_diag", lambda: {"torch": "test", "transformers": "test", "fair_esm": None}
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    tokenizer = Mock()
    model = Mock(side_effect=RuntimeError("stop before loading model weights"))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=model),
        ),
    )
    result = esm.esm1b_fitness_scores(
        "ACD", [1], str(tmp_path), model_backend="transformers", device="cpu"
    )
    assert result["status"] == "capability_unavailable"
    for loader in (tokenizer, model):
        loader.assert_called_once_with(
            esm.HF_MODEL, revision=esm.HF_MODEL_REVISION, trust_remote_code=False
        )
    assert len(esm.HF_MODEL_REVISION) == 40


def test_pdb_mirror_fallback_does_not_hide_unexpected_programming_errors(tmp_path, monkeypatch):
    get = Mock(side_effect=RuntimeError("unexpected programming error"))
    monkeypatch.setattr(pdb.requests, "get", get)
    with pytest.raises(RuntimeError, match="unexpected programming error"):
        pdb.pdb_fetch_structure("4GV1", str(tmp_path))
    get.assert_called_once()
