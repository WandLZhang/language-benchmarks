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

> **⚠️ Web Δ is provisional this round.** Hosted web-grounding barely fires on Vertex unless forced. Claude's `web_search` is forcible via `tool_choice` (fires 100%); Gemini's `google_search` won't fire inside the translation prompt even when hard-prompted. The codified fix is a **two-step forced fetch** (lean forced Gemini google_search → inject → translate), verified **100% firing over 30 trials**. The web arm is being **re-run with that forced fetch for every model**. The **Quality (no-context) ranking is final.**

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
<tr><td><b>gemini-3.5-flash</b></td><td align="center"><b>4.28</b></td><td align="center">21%</td><td align="center">6.98</td><td align="center">0.01</td></tr>
<tr><td><b>gemini-3.6-flash</b></td><td align="center"><b>4.22</b></td><td align="center">17%</td><td align="center">5.20</td><td align="center">-0.02</td></tr>
<tr><td><b>claude-opus-4-8</b></td><td align="center"><b>4.09</b></td><td align="center">14%</td><td align="center">1.94</td><td align="center">-0.05</td></tr>
<tr><td><b>claude-sonnet-5</b></td><td align="center"><b>3.99</b></td><td align="center">12%</td><td align="center">3.19</td><td align="center">-0.01</td></tr>
<tr><td><b>deepseek-v3.2</b></td><td align="center"><b>3.86</b></td><td align="center">8%</td><td align="center">2.67</td><td align="center">0.13</td></tr>
<tr><td><b>gemini-3.5-flash-lite</b></td><td align="center"><b>3.78</b></td><td align="center">7%</td><td align="center">2.21</td><td align="center">-0.11</td></tr>
<tr><td><b>claude-sonnet-4-6</b></td><td align="center"><b>3.67</b></td><td align="center">8%</td><td align="center">1.81</td><td align="center">0.15</td></tr>
<tr><td><b>grok-4.20</b></td><td align="center"><b>3.50</b></td><td align="center">7%</td><td align="center">0.48</td><td align="center">0.37</td></tr>
<tr><td><b>qwen3-235b</b></td><td align="center"><b>3.26</b></td><td align="center">4%</td><td align="center">0.94</td><td align="center">0.62</td></tr>
<tr><td><b>grok-4.1-fast</b></td><td align="center"><b>3.22</b></td><td align="center">3%</td><td align="center">0.59</td><td align="center">0.51</td></tr>
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
