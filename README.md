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

> **Web Δ = FORCED web-grounding.** Hosted search barely fires on Vertex on its own, and Gemini's `google_search` can't be forced inside a task prompt at all — so `web` injects a lean **forced** Gemini google_search fetch (verified **100% firing over 30 trials**) into every model. **Finding: forced web lifts nearly every model, and most the weaker/cheaper ones** (grok-4.1-fast +0.71, qwen3-235b +0.70, sonnet-4-6 +0.57), compressing the field — while the leaders are already at ceiling (gemini-3.5-flash +0.00, claude-opus-5 +0.03). A fast model + forced web (sonnet-4-6 4.22 @ 1.7s, grok-4.1-fast 3.88 @ 0.55s) closes much of the gap to the slow quality leaders.

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4" align="center">Cantonese Colloquial Translation</th>
<th colspan="4" align="center">Chinese Sentence Generation (word + context)</th>
</tr>
<tr>
<th>Quality (1-5)</th>
<th>Win%</th>
<th>p50 s</th>
<th>Web Δ</th>
<th>Quality (1-5)</th>
<th>Win%</th>
<th>p50 s</th>
<th>Web Δ</th>
</tr>
</thead>
<tbody>
<tr><td><b>gemini-3.5-flash</b></td><td align="center"><b>4.28</b></td><td align="center">17%</td><td align="center">6.51</td><td align="center">0.00</td><td align="center"><b>1.07</b></td><td align="center">0%</td><td align="center">3.82</td><td align="center">—</td></tr>
<tr><td><b>claude-opus-5</b></td><td align="center"><b>4.21</b></td><td align="center">20%</td><td align="center">2.80</td><td align="center">0.03</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.6-flash</b></td><td align="center"><b>4.17</b></td><td align="center">13%</td><td align="center">4.72</td><td align="center">0.10</td><td align="center"><b>1.31</b></td><td align="center">0%</td><td align="center">4.26</td><td align="center">—</td></tr>
<tr><td><b>claude-opus-4-8</b></td><td align="center"><b>4.10</b></td><td align="center">11%</td><td align="center">1.80</td><td align="center">0.15</td><td align="center"><b>3.93</b></td><td align="center">17%</td><td align="center">2.25</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-5</b></td><td align="center"><b>3.95</b></td><td align="center">8%</td><td align="center">2.06</td><td align="center">0.29</td><td align="center"><b>4.21</b></td><td align="center">17%</td><td align="center">1.31</td><td align="center">—</td></tr>
<tr><td><b>deepseek-v3.2</b></td><td align="center"><b>3.80</b></td><td align="center">4%</td><td align="center">2.11</td><td align="center">0.26</td><td align="center"><b>3.45</b></td><td align="center">0%</td><td align="center">1.18</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.5-flash-lite</b></td><td align="center"><b>3.77</b></td><td align="center">6%</td><td align="center">2.12</td><td align="center">0.37</td><td align="center"><b>3.43</b></td><td align="center">14%</td><td align="center">1.95</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-4-6</b></td><td align="center"><b>3.65</b></td><td align="center">9%</td><td align="center">1.70</td><td align="center">0.57</td><td align="center"><b>3.69</b></td><td align="center">3%</td><td align="center">1.64</td><td align="center">—</td></tr>
<tr><td><b>grok-4.20</b></td><td align="center"><b>3.49</b></td><td align="center">6%</td><td align="center">0.53</td><td align="center">0.39</td><td align="center"><b>4.14</b></td><td align="center">31%</td><td align="center">0.54</td><td align="center">—</td></tr>
<tr><td><b>qwen3-235b</b></td><td align="center"><b>3.24</b></td><td align="center">3%</td><td align="center">1.01</td><td align="center">0.70</td><td align="center"><b>3.81</b></td><td align="center">14%</td><td align="center">0.66</td><td align="center">—</td></tr>
<tr><td><b>grok-4.1-fast</b></td><td align="center"><b>3.17</b></td><td align="center">3%</td><td align="center">0.55</td><td align="center">0.71</td><td align="center"><b>3.43</b></td><td align="center">3%</td><td align="center">0.49</td><td align="center">—</td></tr>
<tr><td><b>claude-haiku-4-5</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><b>1.00</b></td><td align="center">0%</td><td align="center">—</td><td align="center">—</td></tr>
</tbody>
</table>

### Notes per use case

- **Cantonese Colloquial Translation** — winner **gemini-3.5-flash**; 231 items; judges opus-judge, gemini-judge.
- **Chinese Sentence Generation (word + context)** — winner **claude-sonnet-5**; 15 items; judges opus-judge, gemini-judge.

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
