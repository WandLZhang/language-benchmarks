"""Glossary-RAG context mode: retrieve Words.hk (粵典) entries from the Vertex AI RAG Engine
corpus and inject them into the prompt. This is the reusable version of convo-live's Method A
(bench/bench_colloquial.py). English queries hit the corpus via its English glosses, so it
works for EN->yue as well as zh->yue; it returns HK-canonical forms (e.g. Pikachu -> 比卡超)
that models otherwise miss.

Corpus: projects/wz-data-catalog-demo/locations/us-central1/ragCorpora/5764607523034234880 (display "wordshk").
Override via env LB_RAG_CORPUS / LB_RAG_PROJECT.
"""
import os
import time
import requests

RAG_PROJECT = os.getenv("LB_RAG_PROJECT", "wz-data-catalog-demo")
RAG_LOCATION = os.getenv("LB_RAG_LOCATION", "us-central1")
RAG_CORPUS = os.getenv("LB_RAG_CORPUS",
                       "projects/wz-data-catalog-demo/locations/us-central1/ragCorpora/5764607523034234880")

_tok = {"t": None, "exp": 0.0}


def _token():
    import google.auth
    from google.auth.transport.requests import Request
    now = time.time()
    if _tok["t"] and now < _tok["exp"]:
        return _tok["t"]
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    _tok["t"], _tok["exp"] = creds.token, now + 50 * 60
    return creds.token


def retrieve(text, k=8):
    """Return up to k Words.hk entry texts most similar to `text` (English or Chinese)."""
    if not text:
        return []
    url = (f"https://{RAG_LOCATION}-aiplatform.googleapis.com/v1/"
           f"projects/{RAG_PROJECT}/locations/{RAG_LOCATION}:retrieveContexts")
    body = {"vertexRagStore": {"ragResources": [{"ragCorpus": RAG_CORPUS}]},
            "query": {"text": text, "ragRetrievalConfig": {"topK": k}}}
    for attempt in range(3):
        try:
            r = requests.post(url, timeout=30, json=body, headers={
                "Authorization": f"Bearer {_token()}", "x-goog-user-project": RAG_PROJECT,
                "Content-Type": "application/json"})
            r.raise_for_status()
            return [c.get("text", "") for c in r.json().get("contexts", {}).get("contexts", [])]
        except Exception:
            if attempt < 2:
                time.sleep(2 * (attempt + 1)); continue
            raise


def context_block(text, k=8):
    """Formatted reference block to append to the system prompt (empty if no hits)."""
    entries = retrieve(text, k)
    if not entries:
        return ""
    lines = "\n".join(f"- {e.strip()[:320]}" for e in entries)
    return ("\n\n## REFERENCE — authentic Cantonese from Words.hk (粵典). Prefer these attested "
            "colloquial forms / HK names over your own guess:\n" + lines)
