"""Generic per-task report. Aggregates raw.jsonl (latency) + judged.jsonl (quality) into
report.md, winner.json, and summary.json (machine-readable — consumed by leaderboard.py).

Quality winner = highest mean overall on the no-context (ungrounded) condition. Latency is a
separate axis. A same-item grounding sub-study shows the context-mode (web-grounding) delta.

Usage: python engine/report.py tasks/<task> tasks/<task>/results/<ts>
"""
import sys
import json
import pathlib
import statistics
from collections import defaultdict

import yaml


def pct(xs, p):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [v for v in xs if isinstance(v, (int, float))]
    return statistics.mean(xs) if xs else None


def fmt(x, nd=2):
    return "  -  " if x is None else f"{x:.{nd}f}"


def main():
    task_dir, d = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    task = yaml.safe_load((task_dir / "task.yaml").read_text())
    metrics = task["metrics"]
    specs = {m["id"]: m for m in task["models"]}
    raw = [json.loads(l) for l in (d / "raw.jsonl").read_text().splitlines() if l.strip()]
    judged = [json.loads(l) for l in (d / "judged.jsonl").read_text().splitlines() if l.strip()]

    lat = defaultdict(lambda: {"total": [], "ttft": [], "ok": 0, "n": 0})
    glat = defaultdict(lambda: {"total": []})
    for r in raw:
        if r.get("grounding"):
            if r.get("ok"):
                glat[r["model_id"]]["total"].append(r.get("total_s"))
            continue
        s = lat[r["model_id"]]; s["n"] += 1
        if r.get("ok"):
            s["ok"] += 1; s["total"].append(r.get("total_s")); s["ttft"].append(r.get("ttft_s"))

    q = defaultdict(lambda: defaultdict(list))     # (model,cond) -> metric -> [vals]
    q_tier = defaultdict(list)                       # (model,tier,cond) -> [overall]
    wins = defaultdict(lambda: [0, 0])
    for j in judged:
        sc = j.get("score") or {}
        if not isinstance(sc, dict) or "error" in sc:
            continue
        key = (j["model_id"], j["condition"])
        for c in metrics:
            if isinstance(sc.get(c), (int, float)):
                q[key][c].append(sc[c])
        if isinstance(sc.get("overall"), (int, float)):
            q_tier[(j["model_id"], j["tier"], j["condition"])].append(sc["overall"])
        wins[key][1] += 1
        if sc.get("is_best"):
            wins[key][0] += 1

    def qm(m, c):
        return mean(q[(m, "ungrounded")].get(c, []))

    tiers = sorted({j["tier"] for j in judged if j["condition"] == "ungrounded"})
    gtiers = sorted({t for (m, t, c) in q_tier if c == "grounded"})
    models = sorted([m for m in specs if (m, "ungrounded") in q], key=lambda m: (qm(m, "overall") or 0), reverse=True)

    # ---- machine-readable summary for the leaderboard ----
    summary = {"task": task["name"], "title": task["title"], "n_items": len({r["item_id"] for r in raw}),
               "judges": [j["id"] for j in task["judges"]], "tiers": tiers, "models": {}}
    for m in models:
        off = [v for t in gtiers for v in q_tier[(m, t, "ungrounded")]]
        on = [v for t in gtiers for v in q_tier[(m, t, "grounded")]]
        w = wins[(m, "ungrounded")]
        summary["models"][m] = {
            **{c: qm(m, c) for c in metrics},
            "win_pct": (100 * w[0] / w[1]) if w[1] else None,
            "p50_total_s": pct(lat[m]["total"], .5), "p95_total_s": pct(lat[m]["total"], .95),
            "ttft_p50_s": pct(lat[m]["ttft"], .5), "ok": lat[m]["ok"], "n": lat[m]["n"],
            "by_tier": {t: mean(q_tier[(m, t, "ungrounded")]) for t in tiers},
            "grounding": {"off": mean(off), "on": mean(on),
                          "delta": (mean(on) - mean(off)) if (mean(off) is not None and mean(on) is not None) else None},
        }
    winner = models[0]
    summary["winner"] = winner
    (d / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- human report ----
    L = [f"# {task['title']} — {d.name}\n",
         f"Project `{task.get('name')}` · {summary['n_items']} items · judges={summary['judges']}\n",
         "## Quality (no-context, mean of judges, 1-5)\n",
         "| Model | Overall | " + " | ".join(c.replace("_", " ") for c in metrics if c != "overall") + " | Win% |",
         "|---|---|" + "---|" * len(metrics)]
    for m in models:
        w = wins[(m, "ungrounded")]; wp = f"{100*w[0]/w[1]:.0f}%" if w[1] else "-"
        cells = " | ".join(fmt(qm(m, c)) for c in metrics if c != "overall")
        L.append(f"| {m} | **{fmt(qm(m,'overall'))}** | {cells} | {wp} |")
    L += ["\n## Quality by tier (no-context overall)\n", "| Model | " + " | ".join(tiers) + " |",
          "|---|" + "---|" * len(tiers)]
    for m in models:
        L.append(f"| {m} | " + " | ".join(fmt(mean(q_tier[(m, t, 'ungrounded')])) for t in tiers) + " |")
    L += ["\n## Latency (no-context)\n", "| Model | p50 total | p95 total | p50 TTFT | ok |",
          "|---|---|---|---|---|"]
    for m in models:
        s = lat[m]
        L.append(f"| {m} | {fmt(pct(s['total'],.5))}s | {fmt(pct(s['total'],.95))}s | {fmt(pct(s['ttft'],.5))}s | {s['ok']}/{s['n']} |")
    if gtiers:
        L += [f"\n## Context-mode delta — web-grounding on tiers {gtiers} (same items)\n",
              "| Model | off | on | delta |", "|---|---|---|---|"]
        for m in models:
            g = summary["models"][m]["grounding"]
            dl = g["delta"]
            L.append(f"| {m} | {fmt(g['off'])} | {fmt(g['on'])} | {('%+.2f'%dl) if dl is not None else '  -  '} |")
    L.append(f"\n## Winner (no-context quality): **{winner}** ({fmt(qm(winner,'overall'))}/5, "
             f"p50 {fmt(pct(lat[winner]['total'],.5))}s)\n")

    (d / "report.md").write_text("\n".join(L))
    # winner.json (compat with proxy deploy.sh)
    ws = specs[winner]
    (d / "winner.json").write_text(json.dumps({
        "id": winner, "provider": ws["provider"], "vertex_id": ws["vertex_id"],
        "region": ws.get("region", "global"), "publisher": ws.get("publisher"),
        "grounding_capable": bool(ws.get("grounding")),
        "grounding_recommended": bool((summary["models"][winner]["grounding"]["delta"] or -1) > 0),
        "mean_overall": round(qm(winner, "overall") or 0, 3),
        "p50_total_s": round(pct(lat[winner]["total"], .5) or 0, 2)}, indent=2, ensure_ascii=False))
    print("\n".join(L))
    print(f"\nWrote report.md, winner.json, summary.json in {d}")


if __name__ == "__main__":
    main()
