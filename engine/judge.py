"""Generic LLM-judge grading (comparative, anonymized, multi-judge) for a task's results.

Reads the task's rubric + judge panel from task.yaml, groups candidate outputs per
(item, context-condition), and asks each judge to score all anonymized+shuffled candidates.
Grading is done entirely by the LLM (no regex/heuristics). Multiple judge families blunt
self-preference.

Usage: python engine/judge.py tasks/<task> tasks/<task>/results/<ts>
"""
import os
import sys
import json
import random
import pathlib
import concurrent.futures as futures

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import providers  # noqa: E402


def extract_json(text):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b == -1 or b < a:
        raise ValueError(f"no JSON in judge reply: {text[:150]!r}")
    return json.loads(text[a:b + 1])


def judge_group(judge_spec, rubric, item_text, candidates):
    order = list(candidates)
    random.Random(hash(item_text) & 0xFFFFFFFF).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    blocks = [f"### Candidate {lab}\n{(out.strip() or '(no output)')}"
              for lab, (_, out) in zip(labels, order)]
    user = f'English input: "{item_text}"\n\n' + "\n\n".join(blocks) + "\n\nGrade every candidate now. JSON only."
    r = providers.call_model(judge_spec, rubric, user, grounding=False, max_tokens=3000)
    if not r["ok"]:
        return [(mid, {"error": r["error"]}) for mid, _ in order]
    try:
        parsed = extract_json(r["text"])
        scores = parsed.get("scores", {})
    except Exception as e:
        return [(mid, {"error": f"parse: {e}"}) for mid, _ in order]
    best = parsed.get("best")
    rows = []
    for lab, (mid, _) in zip(labels, order):
        sc = scores.get(lab, {"error": "missing label"})
        if isinstance(sc, dict) and "error" not in sc:
            sc = dict(sc); sc["is_best"] = (lab == best)
        rows.append((mid, sc))
    return rows


def main():
    task_dir, res_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    task = yaml.safe_load((task_dir / "task.yaml").read_text())
    judges, rubric = task["judges"], task["rubric"]
    rows = [json.loads(l) for l in (res_dir / "raw.jsonl").read_text().splitlines() if l.strip()]

    groups = {}
    for r in rows:
        cond = r.get("context") or ("grounded" if r.get("grounding") else "none")
        g = groups.setdefault((r["item_id"], cond), {"text": r["text"], "tier": r["tier"], "cands": []})
        g["cands"].append((r["model_id"], r.get("output", "")))

    calls = [(iid, cond, gg["text"], gg["tier"], gg["cands"], j)
             for (iid, cond), gg in groups.items() for j in judges]
    print(f"Judging {len(groups)} groups x {len(judges)} judges = {len(calls)} calls")
    done = 0
    with open(res_dir / "judged.jsonl", "w", encoding="utf-8") as fh, \
         futures.ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(judge_group, j, rubric, text, cands): (iid, cond, tier, j["id"])
               for (iid, cond, text, tier, cands, j) in calls}
        for f in futures.as_completed(fut):
            iid, cond, tier, jid = fut[f]
            try:
                results = f.result()
            except Exception as e:
                results = []; print(f"  [ERR] {iid}/{cond}/{jid}: {e}")
            for mid, sc in results:
                fh.write(json.dumps({"item_id": iid, "condition": cond, "tier": tier,
                                     "judge_id": jid, "model_id": mid, "score": sc},
                                    ensure_ascii=False) + "\n")
            fh.flush(); done += 1
            print(f"[{done}/{len(calls)}] {iid}/{cond} by {jid}")
    print(f"\nDone -> {res_dir/'judged.jsonl'}\nNext: python engine/report.py {task_dir} {res_dir}")


if __name__ == "__main__":
    main()
