"""
Children's-book page translation benchmark — which model should translate a photographed book page
into colloquial Hong Kong Cantonese.

Self-contained on purpose: engine/run_bench.py drives providers.call_model, which is text-only (no
image part), so this task ships its own runner rather than refactoring the shared engine. It writes
results/<ts>/{raw.jsonl,judged.jsonl,summary.json,report.md} in the SAME shape engine/report.py
emits, so engine/leaderboard.py merges it into the wide table with no engine changes. Same
precedent as tasks/speech-to-text/bench_stt.py.

Context arms (the architectural question behind cloud-claude's page-mismatch bug)
  stateless : one page per call, no story history
  history   : prior pages + the model's own prior outputs — what cloud-claude does today, and what
              lets a model drift into writing the next page of a story it already knows
  glossary  : stateless + Words.hk (粵典) glossary-RAG (engine/rag.py), queried on the page's OCR

Scored at `stateless`; Web Δ in the leaderboard = glossary − stateless.

QUALITY-FIRST: this workload is latency-insensitive, so Claude arms run adaptive thinking at
effort=max and nothing is ranked on speed. Latency is recorded, never used to break a tie.

Run:
  source .venv/bin/activate
  python tasks/book-page-translation/pull_pages.py --pages 6
  python tasks/book-page-translation/bench_pages.py
"""
import argparse
import base64
import datetime
import json
import os
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict

import yaml
from anthropic import AnthropicVertex
from google import genai
from google.genai import types

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "engine"))
import rag  # noqa: E402  — Words.hk glossary retrieval, shared with the text task

PROJECT = os.getenv("LT_PROJECT", "wz-cloud-claude")
UID = os.getenv("LT_UID", "xoBY9nLz8ObwvIRPdJ855EBmAlv2")
MAX_TOKENS = 8000
MAX_RETRIES = 3

_genai = genai.Client(vertexai=True, project=PROJECT, location="global")
_ant = AnthropicVertex(region="global", project_id=PROJECT)

# Sent with EVERY page, in every arm. cloud-claude sends an empty user message on image turns, so
# the only instruction in a 70-page chat is the first turn's one-liner; restating it per page is
# part of what is under test. Keeping it identical across arms isolates `history` as the only
# variable between stateless and history.
PAGE_INSTRUCTION = ("The text is printed on the attached photo of a single page of a children's "
                    "book. Translate ONLY the text printed on THIS page — do not continue the "
                    "story, do not summarise, do not skip a line.")


def load_templates():
    """System prompts straight from Firestore, so the bench always tests what the app deploys."""
    import firebase_admin
    from firebase_admin import firestore
    if not firebase_admin._apps:
        firebase_admin.initialize_app(options={"projectId": PROJECT})
    db = firestore.client()
    out = {}
    for tid in {"g8QTqrl3O40ex8pmBSvf", "K731ZzMJXnlP85BNCFmY"}:
        d = db.collection("prompts").document(UID).collection("userPrompts").document(tid).get().to_dict()
        if not d:
            raise SystemExit(f"prompt template {tid} not found in Firestore")
        out[tid] = {"system": d.get("systemPrompt", ""), "content": d.get("content", "")}
        print(f"  template {tid[:8]} '{d.get('title')}' — system {len(out[tid]['system'])} chars")
    return out


# ---------- generation ----------
def _anthropic_call(spec, system, turns, max_tokens):
    """turns = [(image_bytes|None, text|None, role)] flattened into Anthropic message blocks."""
    msgs = []
    for img, text, role in turns:
        blocks = []
        if img:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                       "data": base64.b64encode(img).decode()}})
        if text:
            blocks.append({"type": "text", "text": text})
        msgs.append({"role": role, "content": blocks})

    opts = {"model": spec["vertex_id"], "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system}], "messages": msgs}
    if spec.get("effort"):
        opts["thinking"] = {"type": "adaptive"}
        opts["output_config"] = {"effort": spec["effort"]}

    for strip in (False, True):     # some models reject output_config.effort; say so, don't hide it
        if strip:
            opts.pop("thinking", None)
            opts.pop("output_config", None)
        try:
            r = _ant.messages.create(**opts)
            if strip:
                print(f"      note: {spec['id']} rejected output_config.effort — ran without it")
            return "".join(b.text for b in r.content if getattr(b, "type", None) == "text").strip()
        except Exception as e:  # noqa: BLE001
            if not strip and "effort" in str(e).lower():
                continue
            raise


def _gemini_call(spec, system, turns, max_tokens):
    contents = []
    for img, text, role in turns:
        parts = []
        if img:
            parts.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))
        if text:
            parts.append(types.Part.from_text(text=text))
        contents.append(types.Content(role="model" if role == "assistant" else "user", parts=parts))
    r = _genai.models.generate_content(
        model=spec["vertex_id"], contents=contents,
        config=types.GenerateContentConfig(system_instruction=system, temperature=1.0,
                                           max_output_tokens=max_tokens))
    return (r.text or "").strip()


def generate(spec, system, turns, max_tokens=MAX_TOKENS):
    """Returns (text, latency_s). Retries transient failures; raises on the last one."""
    fn = _anthropic_call if spec["provider"] == "anthropic" else _gemini_call
    last = None
    for attempt in range(MAX_RETRIES):
        t0 = time.monotonic()
        try:
            return fn(spec, system, turns, max_tokens), time.monotonic() - t0
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 2)
    raise last


# ---------- judging ----------
def judge_page(jspec, rubric, image, candidates):
    """One comparative, anonymised call per judge per page-arm. Sees the page photo itself."""
    order = list(candidates.items())
    random.Random(hash(tuple(sorted(candidates))) & 0xFFFFFFFF).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    blocks = "\n\n".join(f"### Candidate {lab}\n{(txt or '(empty)')}" for lab, (_, txt) in zip(labels, order))
    user = (f"{blocks}\n\nScore every candidate against the attached page photo. JSON only.")

    if jspec["provider"] == "anthropic":
        raw = _anthropic_call({"id": jspec["id"], "provider": "anthropic", "vertex_id": jspec["vertex_id"]},
                              rubric, [(image, user, "user")], 3000)
    else:
        raw = _gemini_call({"id": jspec["id"], "provider": "gemini", "vertex_id": jspec["vertex_id"]},
                           rubric, [(image, user, "user")], 3000)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        raise ValueError(f"judge returned no JSON: {raw[:120]}")
    parsed = json.loads(raw[a:b + 1])
    scores = parsed.get("scores", {})
    best_label = parsed.get("best")
    out = {mid: scores.get(lab) or {} for lab, (mid, _) in zip(labels, order)}
    best = next((mid for lab, (mid, _) in zip(labels, order) if lab == best_label), None)
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list of task.yaml model ids")
    ap.add_argument("--modes", default="", help="comma list of context modes")
    ap.add_argument("--limit", type=int, default=0, help="cap pages per book (debug)")
    ap.add_argument("--tag", default="", help="results dir name instead of a timestamp")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "task.yaml").read_text())
    models = cfg["models"]
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m["id"] in want]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()] or cfg["context_modes"]
    scored = cfg["scored_mode"]
    metrics = cfg["metrics"]

    items = [json.loads(l) for l in (HERE / cfg["testset_file"]).read_text().splitlines() if l.strip()]
    if args.limit:
        keep = defaultdict(int)
        out = []
        for it in items:
            if keep[it["chat_id"]] < args.limit:
                keep[it["chat_id"]] += 1
                out.append(it)
        items = out
    for it in items:
        it["_img"] = (HERE / "pages" / it["image_file"]).read_bytes()

    print("Loading system prompts from Firestore...")
    templates = load_templates()

    ts = args.tag or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = HERE / "results" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    books = sorted({it["book"] for it in items})
    print(f"\n{len(items)} pages · {len(books)} books · modes {modes} · "
          f"models {[m['id'] for m in models]}\n")

    # ---- generate --------------------------------------------------------------------------
    gen = {}            # (model_id, mode, item_id) -> text
    lats = defaultdict(list)
    errs = defaultdict(int)
    glossary_cache = {}
    rawf = open(outdir / "raw.jsonl", "w", encoding="utf-8")

    for m in models:
        for mode in modes:
            by_book = defaultdict(list)
            for it in items:
                by_book[it["chat_id"]].append(it)
            for chat_id, pages in by_book.items():
                pages.sort(key=lambda p: p["page_index"])
                hist = []
                for it in pages:
                    tpl = templates[it["template_id"]]
                    system = tpl["system"]
                    instr = f"{tpl['content']}\n\n{PAGE_INSTRUCTION}"

                    if mode == "glossary":
                        q = it["source_text"] or it["book"]
                        if q not in glossary_cache:
                            try:
                                glossary_cache[q] = rag.context_block(q, k=8)
                            except Exception as e:  # noqa: BLE001
                                print(f"    glossary retrieval failed ({str(e)[:60]}) — ungrounded")
                                glossary_cache[q] = ""
                        system = system + glossary_cache[q]

                    turns = (hist if mode == "history" else []) + [(it["_img"], instr, "user")]
                    print(f"  {m['id']:22} {mode:9} {it['id']} p{it['page_index']} "
                          f"[{len(turns)} turns, sys {len(system)}ch]", end=" ", flush=True)
                    try:
                        text, lat = generate(m, system, turns)
                        gen[(m["id"], mode, it["id"])] = text
                        lats[(m["id"], mode)].append(lat)
                        print(f"{lat:5.1f}s  {text[:48].replace(chr(10), ' ')}")
                    except Exception as e:  # noqa: BLE001
                        errs[(m["id"], mode)] += 1
                        text = ""
                        gen[(m["id"], mode, it["id"])] = ""
                        print(f"ERROR {str(e)[:70]}")
                        lat = None
                    rawf.write(json.dumps({"model_id": m["id"], "mode": mode, "item_id": it["id"],
                                           "book": it["book"], "page_index": it["page_index"],
                                           "source_lang": it["source_lang"],
                                           "system_len": len(system), "instruction": instr,
                                           "source_text": it["source_text"], "output": text,
                                           "lat_s": round(lat, 2) if lat else None},
                                          ensure_ascii=False) + "\n")
                    rawf.flush()
                    if mode == "history":
                        hist = hist + [(it["_img"], instr, "user"), (None, text or "(no output)", "assistant")]
    rawf.close()

    # ---- judge -----------------------------------------------------------------------------
    scores = defaultdict(lambda: defaultdict(list))   # (model,mode) -> metric -> [values]
    wins = defaultdict(int)
    judged_n = defaultdict(int)
    with open(outdir / "judged.jsonl", "w", encoding="utf-8") as jf:
        for mode in modes:
            for it in items:
                cands = {m["id"]: gen.get((m["id"], mode, it["id"]), "") for m in models}
                for j in cfg["judges"]:
                    try:
                        per, best = judge_page(j, cfg["rubric"], it["_img"], cands)
                    except Exception as e:  # noqa: BLE001
                        print(f"  judge {j['id']} {mode} {it['id']} FAILED: {str(e)[:70]}")
                        continue
                    for mid, sc in per.items():
                        for k in metrics:
                            if isinstance(sc.get(k), (int, float)):
                                scores[(mid, mode)][k].append(float(sc[k]))
                    if best:
                        wins[(best, mode)] += 1
                    judged_n[mode] += 1
                    jf.write(json.dumps({"mode": mode, "item_id": it["id"], "judge": j["id"],
                                         "best": best, "scores": per}, ensure_ascii=False) + "\n")
                    jf.flush()
                line = "  ".join(
                    f"{mid.replace('claude-','').replace('gemini-','g'):14}"
                    f"f{statistics.mean(scores[(mid, mode)]['fidelity'][-1:] or [0]):.0f}"
                    f"/o{statistics.mean(scores[(mid, mode)]['overall'][-1:] or [0]):.0f}"
                    f"/v{statistics.mean(scores[(mid, mode)]['vividness'][-1:] or [0]):.0f}"
                    for mid in cands)
                print(f"  judged {mode:9} {it['id']}  {line}")

    # ---- aggregate -------------------------------------------------------------------------
    def mean(xs):
        return statistics.mean(xs) if xs else None

    def p50(xs):
        return sorted(xs)[len(xs) // 2] if xs else None

    def rank_score(mid):
        vals = [mean(scores[(mid, scored)][k]) for k in metrics]
        vals = [v for v in vals if v is not None]
        return mean(vals) or 0

    summary = {"task": cfg["name"], "title": cfg["title"], "n_items": len(items),
               "judges": [j["id"] for j in cfg["judges"]], "tiers": [scored], "models": {}}
    n_judgements = max(1, judged_n[scored])
    for m in models:
        mid, key = m["id"], (m["id"], scored)
        off = mean(scores[key]["overall"])
        on = mean(scores[(mid, "glossary")]["overall"]) if "glossary" in modes else None
        summary["models"][mid] = {
            **{k: mean(scores[key][k]) for k in metrics},
            "rank_score": rank_score(mid),
            "win_pct": 100.0 * wins[key] / n_judgements,
            "p50_total_s": p50(lats[key]), "p95_total_s": None, "ttft_p50_s": None,
            "ok": len(lats[key]), "n": len(items), "errors": errs[key],
            "by_tier": {scored: off},
            "context_axis": {mo: {k: mean(scores[(mid, mo)][k]) for k in metrics} for mo in modes},
            "grounding": {"off": off, "on": on,
                          "delta": (on - off) if (on is not None and off is not None) else None},
        }

    # Winner: highest mean of the three quality metrics, but fidelity is a gate — a beautiful
    # translation of the wrong page is the failure this task exists to catch. Latency is ignored.
    eligible = [m["id"] for m in models if (summary["models"][m["id"]].get("fidelity") or 0) >= 4.0]
    ranked = sorted(summary["models"], key=rank_score, reverse=True)
    summary["winner"] = next((mid for mid in ranked if mid in eligible), ranked[0] if ranked else None)
    summary["winner_rule"] = "max mean(overall,fidelity,vividness) among models with fidelity >= 4.0; latency ignored"
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- report ----------------------------------------------------------------------------
    def f(v, nd=2):
        return "—" if v is None else f"{v:.{nd}f}"

    L = [f"# {cfg['title']} — {ts}\n",
         f"{len(items)} pages from {len(books)} books · project `{PROJECT}` · "
         f"judges {', '.join(j['id'] for j in cfg['judges'])}\n",
         "Quality-first: this workload is latency-insensitive, so Claude arms run adaptive thinking "
         "at `effort=max` and nothing is ranked on speed.\n",
         f"## Scored at context mode `{scored}`\n",
         "| Model | Fidelity | Colloquial | Vividness | Mean | Win% | p50 s | err |",
         "|---|---|---|---|---|---|---|---|"]
    for mid in ranked:
        d = summary["models"][mid]
        star = " 🏆" if mid == summary["winner"] else ""
        L.append(f"| {mid}{star} | {f(d.get('fidelity'))} | **{f(d.get('overall'))}** | "
                 f"{f(d.get('vividness'))} | {f(d['rank_score'])} | {d['win_pct']:.0f}% | "
                 f"{f(d['p50_total_s'], 1)} | {d['errors']} |")
    L += [f"\nWinner rule: {summary['winner_rule']}.\n",
          "\n## Context axis\n",
          "`history` is what cloud-claude does today — prior pages plus the model's own prior "
          "outputs in context. A fidelity drop from `stateless` to `history` is the model writing "
          "the next page of a story it already knows instead of the page in front of it.\n",
          "| Model | Mode | Fidelity | Colloquial | Vividness |", "|---|---|---|---|---|"]
    for mid in ranked:
        for mo, d in summary["models"][mid]["context_axis"].items():
            L.append(f"| {mid} | {mo} | {f(d.get('fidelity'))} | {f(d.get('overall'))} | "
                     f"{f(d.get('vividness'))} |")
    (outdir / "report.md").write_text("\n".join(L))

    print("\n=== SUMMARY (%s) ===" % scored)
    for mid in ranked:
        d = summary["models"][mid]
        print(f"  {mid:22} fidelity={f(d.get('fidelity'))} colloquial={f(d.get('overall'))} "
              f"vivid={f(d.get('vividness'))} mean={f(d['rank_score'])} "
              f"p50={f(d['p50_total_s'], 1)}s err={d['errors']}")
    print(f"\nwinner: {summary['winner']}  ({summary['winner_rule']})")
    print(f"-> {outdir}\nNext: python engine/leaderboard.py")


if __name__ == "__main__":
    main()
