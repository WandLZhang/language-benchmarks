"""Generate README.md — the leaderboard IS the repo README.

Scans tasks/*/task.yaml + each task's latest results/*/summary.json and emits one HTML table
whose columns are grouped under a MERGED super-header per task (use case), with the actual
sub-metrics beneath. Add a new task folder -> it becomes a new column group automatically.

Usage: python engine/leaderboard.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"

# metric key -> short column header
SHORT = {"overall": "Overall", "cantonese_authenticity": "Canto", "mandarin_correctness": "Man",
         "meaning_fidelity": "Mean", "naturalness": "Nat", "format_validity": "Fmt"}

HEADER = """# language-benchmarks

Model comparisons for **language / linguistic / translation** tasks, with a first-class
**contextualization axis** (no-context vs web-grounding vs — roadmap — glossary/TM-RAG).
Real Vertex AI Model Garden calls, LLM-judge panel (anonymized + comparative), latency measured
separately from quality. **This README is the leaderboard** — regenerate with
`python engine/leaderboard.py`.

Each use case is a merged column group; sub-columns are quality (1-5, judge mean),
Win% (comparative best-pick rate), p50 latency (s), and **Ctx Δ** (web-grounding overall
delta on the hard tiers — positive = grounding helped).
"""

METHODOLOGY = """
## Methodology (why it's built this way)

- **Reference-free LLM-judge** for authenticity/register — the translation field's modern
  approach (GEMBA / GEMBA-MQM). FLORES `yue_Hant` is *written* Standard Chinese, so it's no
  gold set for colloquial Cantonese; we grade real + self-authored colloquial prompts instead.
- **Judge panel across families** (Claude + Gemini), anonymized & comparative, to blunt
  position/self-preference bias; small gaps ≈ ties. LLM-judges are weaker on dialects, so treat
  as directional and calibrate with humans for high-stakes calls.
- **Latency is a separate axis** (TTFT + total), never blended into quality.
- **Contextualization axis** — best practice for translation is to compare context modes
  (no-context · web-grounding · glossary/dictionary-RAG · TM few-shot). We ship no-context +
  web-grounding today; glossary-RAG (words.hk / Chain-of-Dictionary) and TM-RAG are roadmap.
- **Metrics roadmap**: add chrF++ / COMET / xCOMET / CometKiwi on any reference-based tier.
- **Data/licensing**: ship only the harness + self-authored/owned seed sets; pull third-party
  datasets (FLORES+, Tatoeba, Yue-TRANS) at runtime — several Cantonese corpora are CC-BY-NC.

Add a task: create `tasks/<name>/task.yaml` (models, judges, rubric, prompt, testset), then
`run_bench.py` → `judge.py` → `report.py` → `leaderboard.py`.
"""


def latest_summary(task_dir):
    res = sorted((task_dir / "results").glob("*/summary.json")) if (task_dir / "results").exists() else []
    if not res:
        return None
    return json.loads(res[-1].read_text())


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    import yaml
    tasks = []
    for td in sorted(p for p in TASKS.iterdir() if (p / "task.yaml").exists()):
        cfg = yaml.safe_load((td / "task.yaml").read_text())
        tasks.append((cfg, latest_summary(td)))   # summ may be None (skeleton / pending)

    if not tasks:
        (ROOT / "README.md").write_text(HEADER + "\n_No tasks defined yet._\n" + METHODOLOGY)
        return

    pending = all(s is None for _, s in tasks)

    # sub-columns per task = its metrics + Win% + p50 + Web Δ (did web-grounding help)
    def subcols(cfg):
        return [SHORT.get(c, c) for c in cfg["metrics"]] + ["Win%", "p50 s", "Web Δ"]

    # union of models: ranked by first task's overall if scored, else task.yaml order
    order, seen = [], set()
    for cfg, s in tasks:
        ids = (sorted(s["models"], key=lambda m: (s["models"][m].get("overall") or 0), reverse=True)
               if s else [m["id"] for m in cfg["models"]])
        for m in ids:
            if m not in seen:
                order.append(m); seen.add(m)

    h = ['<table>', '<thead>', '<tr>', '<th rowspan="2">Model</th>']
    for cfg, _ in tasks:
        h.append(f'<th colspan="{len(subcols(cfg))}" align="center">{cfg["title"]}</th>')
    h.append('</tr>\n<tr>')
    for cfg, _ in tasks:
        for sc in subcols(cfg):
            h.append(f'<th>{sc}</th>')
    h.append('</tr>\n</thead>\n<tbody>')

    for m in order:
        row = ['<tr>', f'<td><b>{m}</b></td>']
        for cfg, s in tasks:
            md = s["models"].get(m) if s else None
            if not md:
                row += ['<td align="center">—</td>'] * len(subcols(cfg))
                continue
            for c in cfg["metrics"]:
                v = md.get(c)
                cell = f'<b>{fmt(v)}</b>' if c == "overall" else fmt(v)
                row.append(f'<td align="center">{cell}</td>')
            wp = md.get("win_pct")
            row.append(f'<td align="center">{("%.0f%%" % wp) if wp is not None else "—"}</td>')
            row.append(f'<td align="center">{fmt(md.get("p50_total_s"))}</td>')
            row.append(f'<td align="center">{fmt((md.get("web") or md.get("grounding") or {}).get("delta"))}</td>')
        row.append('</tr>')
        h.append("".join(row))
    h.append('</tbody>\n</table>')

    banner = ("\n> ⏳ **Results pending** — benchmark run in progress; cells fill in once judged. "
              "Structure/factors are final.\n") if pending else ""
    notes = ["\n### Notes per use case\n"]
    for cfg, s in tasks:
        if s:
            notes.append(f"- **{cfg['title']}** — winner **{s['winner']}**; {s['n_items']} items; "
                         f"judges {', '.join(s['judges'])}.")
        else:
            notes.append(f"- **{cfg['title']}** — {len(cfg['models'])} models × context modes "
                         "(none · web-grounding · glossary-RAG); results pending.")

    (ROOT / "README.md").write_text(HEADER + "\n## Leaderboard\n" + banner + "\n" + "\n".join(h)
                                    + "\n" + "\n".join(notes) + "\n" + METHODOLOGY)
    print(f"Wrote {ROOT/'README.md'} ({'pending' if pending else 'scored'}), "
          f"{len(tasks)} task(s), {len(order)} models.")


if __name__ == "__main__":
    main()
