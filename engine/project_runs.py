"""Project an older run's models onto a newer run's scale, with no re-judging.

Judging is comparative, so the same output scores lower in a stronger pool. A model in both runs
(the anchor) measures that shift. Each model carried over from the old run gets the anchor's shift
added to every quality number. Its latency carries over as measured. Its Win% is dropped, because a
best-pick rate only means something inside the pool that produced it. Deltas inside the old run
(Web Δ) don't move, because both of their terms get the same shift.

The shift is the mean per-item score change (new run minus old run), averaged over the judges that
scored the item in both runs. If both runs hold byte-identical anchor outputs on at least
MIN_IDENTICAL items, only those items count, so the shift measures the pool alone. That also lets two
arms of one model serve as the anchor (thinking on in the old run, off in the new one): where they
wrote the same answer, any score change comes from the pool. With fewer identical outputs, every item
counts, which is valid only when both anchor rows are the same configuration.

The new run's own models stay as measured, and its winner stays the winner. Models the task no longer
lists are dropped (retired).

Usage: python engine/project_runs.py <task_dir> <new_run> <old_run> <old_anchor>[=<new_anchor>]
  e.g. python engine/project_runs.py tasks/cantonese-colloquial-translation 20261016-121121 FINAL3 \
           claude-opus-5=claude-opus-5-nothink
Writes <task_dir>/results/<new_run>+<old_run>/summary.json. leaderboard.py reads the newest summary.
"""
import copy
import json
import pathlib
import random
import statistics
import sys

import yaml

MIN_IDENTICAL = 20


def load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def item_rows(run, scored, metrics):
    """(model, item) -> output, and (model, item) -> {judge: {metric: score}}, for the scored condition.
    Reads both result layouts: engine/judge.py writes one judged row per candidate, and the
    self-contained runners (bench_pages.py) write one row per item holding every candidate's scores."""
    outputs, scores = {}, {}
    for r in load_jsonl(run / "raw.jsonl"):
        if r.get("context", r.get("mode")) == scored:
            outputs[(r["model_id"], r["item_id"])] = (r.get("output") or "").strip()
    for j in load_jsonl(run / "judged.jsonl"):
        if j.get("condition", j.get("mode")) != scored:
            continue
        judge = j.get("judge_id", j.get("judge"))
        candidates = j["scores"].items() if "scores" in j else [(j["model_id"], j.get("score") or {})]
        for mid, sc in candidates:
            vals = {c: sc[c] for c in metrics if isinstance(sc.get(c), (int, float))}
            if vals:
                scores.setdefault((mid, j["item_id"]), {})[judge] = vals
    return outputs, scores


def anchor_shift(old, new, old_anchor, new_anchor, metrics):
    (old_out, old_sc), (new_out, new_sc) = old, new
    items = sorted(i for (m, i) in old_sc if m == old_anchor and (new_anchor, i) in new_sc)
    same = [i for i in items if old_out.get((old_anchor, i)) and old_out[(old_anchor, i)] == new_out.get((new_anchor, i))]
    identical = len(same) >= MIN_IDENTICAL
    if not identical and old_anchor != new_anchor:
        sys.exit(f"{old_anchor} and {new_anchor} are different arms and share only {len(same)} identical "
                 f"outputs; the shift needs {MIN_IDENTICAL} to separate the pool from the arm.")
    shifts = {}
    for c in metrics:
        diffs = []
        for i in (same if identical else items):
            o, n = old_sc[(old_anchor, i)], new_sc[(new_anchor, i)]
            per_judge = [n[j][c] - o[j][c] for j in o if j in n and c in o[j] and c in n[j]]
            if per_judge:
                diffs.append(statistics.mean(per_judge))
        rng = random.Random(7)
        boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        shifts[c] = {"shift": statistics.mean(diffs), "ci95": [boots[250], boots[9749]], "items": len(diffs)}
    return shifts, {"identical_outputs_only": identical, "identical_items": len(same), "anchor_items": len(items)}


def projected(md, shifts, metrics, origin):
    md = copy.deepcopy(md)
    s = {c: v["shift"] for c, v in shifts.items()}
    for c in metrics:
        if isinstance(md.get(c), (int, float)):
            md[c] += s[c]
    for vals in (md.get("context_axis") or {}).values():
        for c in metrics:
            if isinstance(vals.get(c), (int, float)):
                vals[c] += s[c]
    for t, v in (md.get("by_tier") or {}).items():
        if isinstance(v, (int, float)):
            md["by_tier"][t] = v + s["overall"]
    for k in ("off", "on"):
        if isinstance((md.get("grounding") or {}).get(k), (int, float)):
            md["grounding"][k] += s["overall"]
    if md.get("rank_score") is not None:
        md["rank_score"] = statistics.mean(md[c] for c in metrics)
    md["win_pct"] = None
    md["projected"] = origin
    return md


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    task_dir = pathlib.Path(sys.argv[1])
    new_run, old_run = task_dir / "results" / sys.argv[2], task_dir / "results" / sys.argv[3]
    old_anchor, _, new_anchor = sys.argv[4].partition("=")
    new_anchor = new_anchor or old_anchor
    task = yaml.safe_load((task_dir / "task.yaml").read_text())
    metrics, scored = task["metrics"], task.get("scored_mode", "none")
    listed = {m["id"] for m in task["models"]}
    new_s = json.loads((new_run / "summary.json").read_text())
    old_s = json.loads((old_run / "summary.json").read_text())

    shifts, basis = anchor_shift(item_rows(old_run, scored, metrics), item_rows(new_run, scored, metrics),
                                 old_anchor, new_anchor, metrics)
    origin = {"run": old_run.name, "anchor": f"{old_anchor}={new_anchor}", **basis,
              "shift": {c: round(v["shift"], 3) for c, v in shifts.items()}}

    models = {m: md for m, md in new_s["models"].items() if m in listed}
    for m, md in old_s["models"].items():
        if m in listed and m not in models:
            models[m] = projected(md, shifts, metrics, origin)
    key = "rank_score" if any(md.get("rank_score") is not None for md in models.values()) else "overall"
    best = max(models, key=lambda m: models[m].get(key) or 0)
    if best != new_s["winner"]:
        print(f"!! projected {best} outranks the measured winner {new_s['winner']} on {key}; winner stays measured")

    summary = {**new_s, "models": models, "projection": {
        "new_run": new_run.name, "old_run": old_run.name, "anchor": {"old": old_anchor, "new": new_anchor},
        **basis, "shift": shifts,
        "retired": sorted(m for m in set(old_s["models"]) | set(new_s["models"]) if m not in listed)}}
    out = task_dir / "results" / f"{new_run.name}+{old_run.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nanchor {old_anchor} ({old_run.name}) -> {new_anchor} ({new_run.name}): "
          f"{basis['identical_items']} identical of {basis['anchor_items']} items, "
          f"{'identical outputs only' if basis['identical_outputs_only'] else 'all items'}")
    for c, v in shifts.items():
        print(f"  {c:10} shift {v['shift']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  over {v['items']} items")
    print(f"\n{'model':28} " + " ".join(f"{c:>10}" for c in metrics) + "   win%    p50 s  source")
    for m in sorted(models, key=lambda m: -(models[m].get("overall") or 0)):
        md = models[m]
        win = "—" if md.get("win_pct") is None else f"{md['win_pct']:.0f}"
        print(f"{m:28} " + " ".join(f"{md.get(c) or 0:10.2f}" for c in metrics)
              + f"  {win:>5}  {md.get('p50_total_s') or 0:7.2f}  "
              + (f"projected from {old_run.name}" if md.get("projected") else "measured"))
    print(f"\nWrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
