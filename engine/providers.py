"""Native per-provider calling for Vertex AI Model Garden.

Three real code paths (no third-party model registry to lag behind brand-new IDs):
  - anthropic : AnthropicVertex SDK           (Claude family, region "global")
  - gemini    : google-genai SDK (Vertex)     (Gemini family)
  - maas      : OpenAI-compatible MaaS endpoint (Grok / Qwen / Kimi / DeepSeek), host
                aiplatform.googleapis.com for the "global" location.

Every call streams so we can measure time-to-first-token, and retries transient errors
(429 / RESOURCE_EXHAUSTED / 503 / overloaded) with exponential backoff. Auth is ADC
(`gcloud auth application-default login`) with quota project = LT_PROJECT.

Imported by run_bench.py / judge.py / pull_reference.py. Run directly to probe connectivity.
"""
import os
import time

PROJECT = os.environ.get("LT_PROJECT", "wz-cloud-claude")
MAX_RETRIES = 4

_anthropic_clients = {}
_genai_clients = {}
_token_cache = {"token": None, "exp": 0.0}


def _anthropic_client(region):
    from anthropic import AnthropicVertex
    if region not in _anthropic_clients:
        _anthropic_clients[region] = AnthropicVertex(region=region, project_id=PROJECT)
    return _anthropic_clients[region]


def _genai_client(location):
    from google import genai
    # Fresh client per call: the genai/httpx client is NOT safe to share across
    # threads for concurrent streaming (raises "client has been closed" when one
    # thread's stream context closes the shared transport). Cheap to construct.
    return genai.Client(vertexai=True, project=PROJECT, location=location)


def _access_token():
    """OAuth2 access token from ADC for the MaaS endpoint, cached until near expiry."""
    import google.auth
    from google.auth.transport.requests import Request
    now = time.time()
    if _token_cache["token"] and now < _token_cache["exp"]:
        return _token_cache["token"]
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    _token_cache["token"] = creds.token
    _token_cache["exp"] = now + 50 * 60
    return creds.token


def _maas_client(region):
    import openai
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    base = f"https://{host}/v1/projects/{PROJECT}/locations/{region}/endpoints/openapi"
    return openai.OpenAI(base_url=base, api_key=_access_token())


def _is_retryable(e):
    s = f"{type(e).__name__} {e}".lower()
    return any(x in s for x in ("429", "resource_exhausted", "rate", "503", "overloaded",
                                "unavailable", "timeout", "deadline"))


def _one_call(spec, system, user, grounding, max_tokens):
    """Single streaming attempt. Returns (text, ttft, usage, citations). Raises on error."""
    provider = spec["provider"]
    t0 = time.monotonic()
    ttft = [None]
    parts, usage, citations = [], {}, []

    def mark():
        if ttft[0] is None:
            ttft[0] = time.monotonic() - t0

    # Optional reasoning knobs, modelled as a MODEL VARIANT (not a context) so the thinking arm
    # still competes head-to-head inside the judge's (item, condition) group:
    #   effort:   "low"|"medium"|"high"  -> Opus 5+ API (adaptive thinking + output_config.effort)
    #   thinking: <token budget>         -> older Claude API (thinking.type.enabled)
    #   thinking: 0 on Gemini            -> disables its thinking
    raw_think = spec.get("thinking")
    off = (raw_think == "disabled") or (raw_think == 0)     # explicit no-thinking arm
    think = 0 if (raw_think is None or isinstance(raw_think, str)) else int(raw_think)
    effort = spec.get("effort")

    if provider == "anthropic":
        opts = dict(model=spec["vertex_id"], max_tokens=max_tokens,
                    system=[{"type": "text", "text": system}],
                    messages=[{"role": "user", "content": user}])
        if off:
            opts["thinking"] = {"type": "disabled"}
        elif effort:
            opts["thinking"] = {"type": "adaptive"}
            opts["output_config"] = {"effort": effort}
        elif think:
            opts["thinking"] = {"type": "enabled", "budget_tokens": think}
            opts["max_tokens"] = max(max_tokens, think + 2000)   # room for the answer after thinking
        if grounding:
            # FORCE a web fetch. Left to itself (even with thinking) Claude skips web_search
            # ~99% of the time; tool_choice="any" guarantees at least one search while still
            # letting it return the clean 2-line output (strict {"type":"tool"} makes it dump
            # search commentary). Needs token headroom for the tool round-trip.
            opts["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
            opts["tool_choice"] = {"type": "any"}
            opts["max_tokens"] = max(max_tokens, 8000)
        with _anthropic_client(spec.get("region", "global")).messages.stream(**opts) as stream:
            for ev in stream:
                if getattr(ev, "type", None) == "content_block_delta" and hasattr(ev, "delta"):
                    t = getattr(ev.delta, "text", None)
                    if t:
                        mark(); parts.append(t)
            fm = stream.current_message_snapshot
            usage = {"input_tokens": fm.usage.input_tokens, "output_tokens": fm.usage.output_tokens}
            for b in fm.content:
                if getattr(b, "type", None) == "text" and getattr(b, "citations", None):
                    for c in b.citations:
                        if getattr(c, "url", None):
                            citations.append({"url": c.url, "title": getattr(c, "title", "")})

    elif provider == "gemini":
        from google.genai import types
        sys_instr = system
        cfg = dict(max_output_tokens=max_tokens)
        if grounding:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
            # Gemini's google_search can't be forced via tool_choice/tool_config; a hard system
            # instruction is what reliably makes it actually search (verified: 0 -> 3 chunks).
            sys_instr = ("You MUST call the google_search tool to verify CURRENT Hong Kong usage "
                         "BEFORE answering. Never answer from memory.\n\n") + system
        cfg["system_instruction"] = sys_instr
        if "thinking" in spec:                      # budget; 0/"disabled" turns Gemini thinking off
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0 if off else think)
        gclient = _genai_client(spec.get("region", "global"))  # hold ref: must outlive the stream
        stream = gclient.models.generate_content_stream(
            model=spec["vertex_id"], contents=user, config=types.GenerateContentConfig(**cfg))
        for chunk in stream:
            t = getattr(chunk, "text", None)
            if t:
                mark(); parts.append(t)
            if getattr(chunk, "usage_metadata", None):
                um = chunk.usage_metadata
                usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                         "output_tokens": getattr(um, "candidates_token_count", None)}
            # Capture google_search grounding usage (was previously unmeasured for Gemini).
            for cand in (getattr(chunk, "candidates", None) or []):
                gm = getattr(cand, "grounding_metadata", None)
                for gc in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(gc, "web", None)
                    url = getattr(web, "uri", None) if web else None
                    if url and url not in {c["url"] for c in citations}:
                        citations.append({"url": url, "title": getattr(web, "title", "")})

    elif provider == "maas":
        model = f'{spec["publisher"]}/{spec["vertex_id"]}'
        stream = _maas_client(spec.get("region", "global")).chat.completions.create(
            model=model, stream=True, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                mark(); parts.append(chunk.choices[0].delta.content)
            if getattr(chunk, "usage", None):
                usage = {"input_tokens": chunk.usage.prompt_tokens,
                         "output_tokens": chunk.usage.completion_tokens}
    else:
        raise ValueError(f"unknown provider {provider!r}")

    return "".join(parts), ttft[0], usage, citations, (time.monotonic() - t0)


def call_model(spec, system, user, grounding=False, max_tokens=1024):
    """Streaming translation call with retry/backoff. Returns a result dict (never raises)."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            text, ttft, usage, citations, total = _one_call(spec, system, user, grounding, max_tokens)
            return {"ok": True, "text": text, "ttft_s": ttft, "total_s": total,
                    "usage": usage, "citations": citations, "error": None}
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1 and _is_retryable(e):
                time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s
                continue
            return {"ok": False, "text": "", "ttft_s": None, "total_s": None,
                    "usage": {}, "citations": [], "error": f"{type(e).__name__}: {e}"}


# ---- connectivity probe ----------------------------------------------------
_PROBE = [
    {"id": "claude-sonnet-5", "provider": "anthropic", "vertex_id": "claude-sonnet-5", "region": "global"},
    {"id": "claude-opus-4-8", "provider": "anthropic", "vertex_id": "claude-opus-4-8", "region": "global"},
    {"id": "gemini-3.6-flash", "provider": "gemini", "vertex_id": "gemini-3.6-flash", "region": "global"},
    {"id": "grok-4.20", "provider": "maas", "publisher": "xai", "vertex_id": "grok-4.20-non-reasoning", "region": "global"},
    {"id": "qwen3-235b", "provider": "maas", "publisher": "qwen", "vertex_id": "qwen3-235b-a22b-instruct-2507-maas", "region": "global"},
    {"id": "deepseek-v3.2", "provider": "maas", "publisher": "deepseek-ai", "vertex_id": "deepseek-v3.2-maas", "region": "global"},
]

if __name__ == "__main__":
    print(f"PROJECT={PROJECT}\n")
    sysmsg = "You are a translator. Reply with only the translation, no notes."
    for spec in _PROBE:
        r = call_model(spec, sysmsg, "Translate to French: hello", grounding=False, max_tokens=512)
        status = "OK " if r["ok"] else "FAIL"
        ttft = f'{r["ttft_s"]:.2f}s' if r["ttft_s"] is not None else "  -  "
        snippet = (r["text"] or r["error"] or "").replace("\n", " ")[:90]
        print(f'[{status}] {spec["id"]:<22} ttft={ttft}  {snippet}')
