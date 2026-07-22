# language-benchmarks

Model comparisons for **language / linguistic / translation** tasks, with a first-class
**contextualization axis** (no-context vs web-grounding vs — roadmap — glossary/TM-RAG).
Real Vertex AI Model Garden calls, LLM-judge panel (anonymized + comparative), latency measured
separately from quality. **This README is the leaderboard** — regenerate with
`python engine/leaderboard.py`.

Each use case is a merged column group; sub-columns are quality (1-5, judge mean),
Win% (comparative best-pick rate), p50 latency (s), and **Ctx Δ** (web-grounding overall
delta on the hard tiers — positive = grounding helped).

## Leaderboard

> ⏳ **Results pending** — benchmark run in progress; cells fill in once judged. Structure/factors are final.

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4" align="center">Cantonese Colloquial Translation</th>
</tr>
<tr>
<th>Quality (1-5)</th>
<th>Win%</th>
<th>p50 s</th>
<th>Web Δ</th>
</tr>
</thead>
<tbody>
<tr><td><b>claude-opus-4-8</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-5</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-4-6</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.6-flash</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.5-flash</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.5-flash-lite</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>grok-4.20</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>grok-4.1-fast</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>qwen3-235b</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>deepseek-v3.2</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
</tbody>
</table>

### Notes per use case

- **Cantonese Colloquial Translation** — 10 models × context modes (none · web-grounding · glossary-RAG); results pending.

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
