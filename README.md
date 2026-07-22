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

> **Web Δ = FORCED web-grounding.** Hosted search barely fires on Vertex on its own, and Gemini's `google_search` can't be forced inside a task prompt at all — so `web` injects a lean **forced** Gemini google_search fetch (verified **100% firing over 30 trials**) into every model. **Finding: forced web lifts nearly every model, and most the weaker/cheaper ones** (grok-4.1-fast +0.79, qwen3-235b +0.69, sonnet-4-6 +0.54), compressing the field — while the leader (gemini-3.5-flash) is already at ceiling (−0.03). A fast model + forced web (grok-4.1-fast 3.96 @ 0.55s, sonnet-4-6 4.23 @ 1.7s) rivals the slow quality leaders.

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
<tr><td><b>gemini-3.5-flash</b></td><td align="center"><b>4.36</b></td><td align="center">25%</td><td align="center">6.51</td><td align="center">-0.03</td></tr>
<tr><td><b>gemini-3.6-flash</b></td><td align="center"><b>4.25</b></td><td align="center">16%</td><td align="center">4.72</td><td align="center">0.10</td></tr>
<tr><td><b>claude-opus-4-8</b></td><td align="center"><b>4.13</b></td><td align="center">15%</td><td align="center">1.80</td><td align="center">0.18</td></tr>
<tr><td><b>claude-sonnet-5</b></td><td align="center"><b>4.02</b></td><td align="center">12%</td><td align="center">2.06</td><td align="center">0.30</td></tr>
<tr><td><b>deepseek-v3.2</b></td><td align="center"><b>3.83</b></td><td align="center">7%</td><td align="center">2.11</td><td align="center">0.30</td></tr>
<tr><td><b>gemini-3.5-flash-lite</b></td><td align="center"><b>3.80</b></td><td align="center">6%</td><td align="center">2.12</td><td align="center">0.39</td></tr>
<tr><td><b>claude-sonnet-4-6</b></td><td align="center"><b>3.70</b></td><td align="center">8%</td><td align="center">1.70</td><td align="center">0.54</td></tr>
<tr><td><b>grok-4.20</b></td><td align="center"><b>3.54</b></td><td align="center">7%</td><td align="center">0.53</td><td align="center">0.41</td></tr>
<tr><td><b>qwen3-235b</b></td><td align="center"><b>3.30</b></td><td align="center">4%</td><td align="center">1.01</td><td align="center">0.69</td></tr>
<tr><td><b>grok-4.1-fast</b></td><td align="center"><b>3.17</b></td><td align="center">2%</td><td align="center">0.55</td><td align="center">0.79</td></tr>
</tbody>
</table>

### Notes per use case

- **Cantonese Colloquial Translation** — winner **gemini-3.5-flash**; 231 items; judges opus-judge, gemini-judge.

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
