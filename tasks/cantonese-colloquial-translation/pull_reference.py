"""Pull the user's REAL everyday-translation queries from cloud-claude Firestore.

The everyday chats (systemPrompt = "You are a Mandarin-to-Cantonese translation assistant...")
store a template-instruction message first, then the user's actual vocab as follow-up user
messages. This is the most authentic test data: real usage, not a curated guess.

Cleans with an LLM (house rule: LLMs parse, not regex) — drops meta/"System:" control
messages, complaints/rants, and long non-translation instructions; strips embedded
"(system: ...)" notes; dedupes. Writes benchmark/testset_real.jsonl (tier "real").

Usage:  python pull_reference.py
"""
import json
import pathlib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))
import firebase_admin
from firebase_admin import firestore

import providers

USER = "xoBY9nLz8ObwvIRPdJ855EBmAlv2"
SIG = "mandarin-to-cantonese translation assistant"
PREFIX = "translate to colloquial cantonese and mandarin"

CLEANER = {"id": "claude-sonnet-4-6", "provider": "anthropic",
           "vertex_id": "claude-sonnet-4-6", "region": "global"}

CLEAN_SYS = """You are cleaning a list of user messages sent to a translation tool. The user
was translating everyday English (occasionally Chinese) words/phrases into Cantonese+Mandarin.

Return ONLY the genuine things-to-translate, cleaned:
- DROP messages that are meta/control (e.g. start with "System"/"system", or are about the
  tool/verification), complaints or rants, and long non-translation instructions (prayers,
  essay/reflection requests, instructions about rhyming/formatting).
- STRIP embedded parenthetical notes like "(system: ...)" or "(use ABB)", keeping the core phrase.
- Otherwise keep the real word/phrase/sentence to translate, unchanged.
Output ONLY a JSON array of cleaned strings, no prose."""


def pull_raw():
    try:
        firebase_admin.initialize_app(options={"projectId": providers.PROJECT})
    except ValueError:
        pass
    db = firestore.client()
    convs = list(db.collection("chats").document(USER).collection("conversations").stream())
    out = []
    for c in convs:
        cd = c.to_dict() or {}
        if SIG not in (cd.get("systemPrompt") or "").lower():
            continue
        for m in c.reference.collection("messages").order_by("timestamp").stream():
            d = m.to_dict() or {}
            if d.get("role") != "user":
                continue
            t = (d.get("content") or "").strip()
            if t:
                out.append(t)
    return out


def dedupe(items):
    seen = {}
    for t in items:
        k = t.lower().strip()
        if k and k not in seen:
            seen[k] = t.strip()
    return list(seen.values())


def main():
    raw = pull_raw()
    cands = dedupe(raw)
    print(f"pulled {len(raw)} everyday user msgs; {len(cands)} unique. Cleaning with LLM...")

    cleaned = []
    CH = 60
    for i in range(0, len(cands), CH):
        chunk = cands[i:i + CH]
        r = providers.call_model(CLEANER, CLEAN_SYS,
                                 "Clean this list:\n" + json.dumps(chunk, ensure_ascii=False),
                                 grounding=False, max_tokens=6000)
        if not r["ok"]:
            print(f"  clean chunk {i} FAILED: {r['error']}")
            continue
        txt = r["text"]
        a, b = txt.find("["), txt.rfind("]")
        try:
            arr = json.loads(txt[a:b + 1])
            cleaned.extend(x.strip() for x in arr if isinstance(x, str) and x.strip())
        except Exception as e:
            print(f"  parse chunk {i} FAILED: {e}")

    final = dedupe(cleaned)
    outp = pathlib.Path(__file__).parent / "testset_real.jsonl"
    with open(outp, "w", encoding="utf-8") as fh:
        for i, t in enumerate(final):
            fh.write(json.dumps({"id": f"r{i+1:03d}", "tier": "real", "text": t},
                                ensure_ascii=False) + "\n")
    print(f"kept {len(final)} cleaned real queries -> {outp}")


if __name__ == "__main__":
    main()
