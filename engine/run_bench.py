"""Generic benchmark runner with a CONTEXT AXIS. Reads a task dir's task.yaml and runs
candidate models on test items with REAL Vertex AI Model Garden calls (no mocks), under one or
more context modes:
  - none     : plain prompt (baseline)
  - web      : hosted web-grounding (Claude web_search + adaptive thinking; Gemini google_search).
               Capable models only.
  - glossary : inject retrieved Words.hk (粵典) entries into the system prompt (engine/rag.py).
               Works for every model, incl. Grok.

Measures TTFT + total latency; failures recorded ok=false (never fabricated).

Usage: python engine/run_bench.py tasks/<task> [--contexts none,web,glossary]
         [--tiers slang,culture] [--models a,b] [--workers N] [--limit N]
"""
import os
import sys
import json
import argparse
import datetime
import pathlib
import concurrent.futures as futures

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import providers  # noqa: E402
import rag        # noqa: E402
import webfetch   # noqa: E402


def load_task(task_dir, testset=None):
    t = yaml.safe_load((task_dir / "task.yaml").read_text())
    t["_system"] = (task_dir / t["prompt_file"]).read_text()
    # --testset overrides task.yaml, so a run can target the full local set without editing (and
    # silently mis-sizing) the committed config.
    tsf = testset or t["testset_file"]
    t["_items"] = [json.loads(l) for l in (task_dir / tsf).read_text().splitlines() if l.strip()]
    t["_testset"] = tsf
    return t


def run_one(task, model, item, context):
    system = task["_system"]
    grounding = False
    impl = context
    if context == "glossary":
        system = system + rag.context_block(item["text"], k=task.get("rag_top_k", 4))
    elif context == "web":
        # Uniform FORCED grounding for every model: inject a lean forced Gemini-google-search
        # fetch (verified to fire 100%). Native self-search is unreliable — Gemini won't fire
        # inside the translation prompt even when hard-prompted — so we never rely on it; this
        # keeps "web" identical across models (a clean, genuinely-grounded comparison).
        system = system + webfetch.block(webfetch.gemini_fetch(item["text"]))
        impl = "web-injected"
    r = providers.call_model(model, system, task["user_template"].format(text=item["text"]),
                             grounding=grounding, max_tokens=task.get("max_tokens", 2048))
    return {"model_id": model["id"], "provider": model["provider"], "vertex_id": model["vertex_id"],
            "item_id": item["id"], "tier": item.get("tier", "default"), "text": item["text"],
            "context": context, "context_impl": impl, "grounding": grounding, "ok": r["ok"],
            "ttft_s": r["ttft_s"], "total_s": r["total_s"], "output": r["text"], "usage": r["usage"],
            "citations": r["citations"], "error": r["error"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_dir")
    ap.add_argument("--contexts", default="none", help="comma list: none,web,glossary")
    ap.add_argument("--tiers", default="", help="only items in these tiers (comma list)")
    ap.add_argument("--models", default="")
    ap.add_argument("--testset", default="", help="override task.yaml testset_file")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    task_dir = pathlib.Path(args.task_dir)
    task = load_task(task_dir, args.testset or None)
    models = task["models"]
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m["id"] in want]
    items = task["_items"]
    if args.tiers:
        keep = {t.strip() for t in args.tiers.split(",")}
        items = [it for it in items if it.get("tier") in keep]
    if args.limit:
        items = items[: args.limit]
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]

    tasks = []
    for m in models:
        for it in items:
            for ctx in contexts:
                # web applies to ALL models: capable ones self-search (native), others get
                # an injected Gemini-google-search fetch (see run_one).
                tasks.append((m, it, ctx))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = task_dir / "results" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "meta.json").write_text(json.dumps({
        "task": task["name"], "project": providers.PROJECT, "timestamp": ts,
        "models": [m["id"] for m in models], "contexts": contexts, "tiers": args.tiers or "all",
        "testset": task.get("_testset"),
        "n_items": len(items), "n_tasks": len(tasks)}, indent=2, ensure_ascii=False))

    print(f"Task {task['name']}: {len(tasks)} calls ({len(models)} models x {len(items)} items x "
          f"contexts {contexts}), workers={args.workers}")
    done = ok = 0
    with open(outdir / "raw.jsonl", "w", encoding="utf-8") as fh, \
         futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut = {ex.submit(run_one, task, m, it, c): (m, it, c) for (m, it, c) in tasks}
        for f in futures.as_completed(fut):
            m, it, c = fut[f]
            try:
                row = f.result()
            except Exception as e:
                row = {"model_id": m["id"], "item_id": it["id"], "context": c, "grounding": c == "web",
                       "ok": False, "error": f"harness: {e}", "output": "", "ttft_s": None,
                       "total_s": None, "tier": it.get("tier", "default"), "text": it["text"],
                       "provider": m["provider"], "vertex_id": m["vertex_id"], "usage": {}, "citations": []}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n"); fh.flush()
            done += 1; ok += 1 if row.get("ok") else 0
            print(f'[{done}/{len(tasks)}] {"ok " if row.get("ok") else "ERR"} {m["id"]:<22}'
                  f'{c:<9} {it["id"]}')
    print(f"\n{ok}/{len(tasks)} ok -> {outdir}\nNext: python engine/judge.py {task_dir} {outdir}")


if __name__ == "__main__":
    main()
