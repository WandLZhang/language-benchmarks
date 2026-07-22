"""Homegrown web-fetch: retrieve CURRENT Hong Kong vernacular for a phrase via a grounded
retriever model, to inject into any generator (incl. Grok/Qwen, which can't self-search on
Vertex MaaS). Two retriever backends so we can bake off which finds better living slang:
  - gemini_fetch : gemini-3.6-flash + Google Search grounding
  - claude_fetch : claude + web_search (+ adaptive thinking)
Returns the retrieved text + citations; cache per (backend, phrase) so the main run reuses.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import providers  # noqa: E402

FETCH_SYS = (
    "You are a Hong Kong Cantonese usage researcher. Use web search to find how the given "
    "English (or Chinese) phrase is CURRENTLY expressed in everyday colloquial Hong Kong "
    "Cantonese (口語) — including up-to-date slang, internet/forum (LIHKG) usage, and the "
    "HK-canonical name for proper nouns. Return a SHORT factual note (<=80 words): the term(s) "
    "HK people actually use now, in Traditional characters, with a one-line gloss each. "
    "No preamble, no translation of the whole phrase — just the reference terms and how they're used.")

GEMINI = {"id": "gem-fetch", "provider": "gemini", "vertex_id": "gemini-3.6-flash", "region": "global"}
CLAUDE = {"id": "cla-fetch", "provider": "anthropic", "vertex_id": "claude-opus-4-8", "region": "global"}

_cache = {}


def _fetch(spec, phrase):
    key = (spec["id"], phrase)
    if key in _cache:
        return _cache[key]
    r = providers.call_model(spec, FETCH_SYS, f"Phrase: {phrase}", grounding=True, max_tokens=1500)
    out = {"text": r["text"] if r["ok"] else "", "citations": r["citations"],
           "latency_s": r["total_s"], "ok": r["ok"], "error": r["error"]}
    _cache[key] = out
    return out


def gemini_fetch(phrase):
    return _fetch(GEMINI, phrase)


def claude_fetch(phrase):
    return _fetch(CLAUDE, phrase)


def block(fetch_result):
    t = (fetch_result or {}).get("text", "").strip()
    if not t:
        return ""
    return ("\n\n## LIVE WEB CONTEXT — current Hong Kong usage (from web search). Prefer these "
            "up-to-date vernacular forms:\n" + t)
