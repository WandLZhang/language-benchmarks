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
<th colspan="6" align="center">Children's Book Page Translation</th>
<th colspan="4" align="center">Cantonese Colloquial Translation</th>
<th colspan="3" align="center">Chinese Sentence Generation (word + context)</th>
<th colspan="2" align="center">Cantonese/Mandarin Speech-to-Text</th>
</tr>
<tr>
<th>Quality (1-5)</th>
<th>Page fidelity</th>
<th>Vividness</th>
<th>Win%</th>
<th>p50 s</th>
<th>Web Δ</th>
<th>Quality (1-5)</th>
<th>Win%</th>
<th>p50 s</th>
<th>Web Δ</th>
<th>Quality (1-5)</th>
<th>Win%</th>
<th>p50 s</th>
<th>Quality (1-5)</th>
<th>p50 s</th>
</tr>
</thead>
<tbody>
<tr><td><b>claude-opus-5-5</b></td><td align="center"><b>4.81</b></td><td align="center">4.90</td><td align="center">4.31</td><td align="center">52%</td><td align="center">88.35</td><td align="center">—</td><td align="center"><b>4.33</b></td><td align="center">27%</td><td align="center">4.44</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-4-argon</b></td><td align="center"><b>4.65</b></td><td align="center">4.77</td><td align="center">4.02</td><td align="center">27%</td><td align="center">4.23</td><td align="center">—</td><td align="center"><b>4.16</b></td><td align="center">25%</td><td align="center">4.17</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-opus-5</b></td><td align="center"><b>4.34</b></td><td align="center">4.81</td><td align="center">3.68</td><td align="center">15%</td><td align="center">4.71</td><td align="center">—</td><td align="center"><i><b>4.07</b></i></td><td align="center">—</td><td align="center">2.80</td><td align="center">0.03</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-opus-4-8</b></td><td align="center"><i><b>4.33</b></i></td><td align="center"><i>4.77</i></td><td align="center"><i>3.44</i></td><td align="center">—</td><td align="center">11.09</td><td align="center">0.01</td><td align="center"><i><b>3.96</b></i></td><td align="center">—</td><td align="center">1.80</td><td align="center">0.15</td><td align="center"><b>3.93</b></td><td align="center">17%</td><td align="center">2.25</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.5-flash</b></td><td align="center"><i><b>4.26</b></i></td><td align="center"><i>4.85</i></td><td align="center"><i>3.18</i></td><td align="center">—</td><td align="center">12.72</td><td align="center">-0.09</td><td align="center"><i><b>4.13</b></i></td><td align="center">—</td><td align="center">6.51</td><td align="center">0.00</td><td align="center"><b>1.07</b></td><td align="center">0%</td><td align="center">3.82</td><td align="center"><b>3.67</b></td><td align="center">4.47</td></tr>
<tr><td><b>claude-sonnet-5</b></td><td align="center"><i><b>4.22</b></i></td><td align="center"><i>4.78</i></td><td align="center"><i>3.48</i></td><td align="center">—</td><td align="center">70.62</td><td align="center">0.27</td><td align="center"><i><b>3.81</b></i></td><td align="center">—</td><td align="center">2.06</td><td align="center">0.29</td><td align="center"><b>4.21</b></td><td align="center">17%</td><td align="center">1.31</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-5-5</b></td><td align="center"><b>4.19</b></td><td align="center">4.90</td><td align="center">3.38</td><td align="center">6%</td><td align="center">42.42</td><td align="center">—</td><td align="center"><b>4.07</b></td><td align="center">16%</td><td align="center">3.70</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>gemini-3.6-flash</b></td><td align="center"><i><b>3.75</b></i></td><td align="center"><i>4.94</i></td><td align="center"><i>2.69</i></td><td align="center">—</td><td align="center">7.01</td><td align="center">-0.05</td><td align="center"><i><b>4.03</b></i></td><td align="center">—</td><td align="center">4.72</td><td align="center">0.10</td><td align="center"><b>1.31</b></td><td align="center">0%</td><td align="center">4.26</td><td align="center"><b>2.92</b></td><td align="center">3.49</td></tr>
<tr><td><b>claude-opus-5-5-effort-low</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><b>4.16</b></td><td align="center">19%</td><td align="center">1.45</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-opus-5-nothink</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><b>3.92</b></td><td align="center">13%</td><td align="center">2.04</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>claude-sonnet-4-6</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><i><b>3.51</b></i></td><td align="center">—</td><td align="center">1.70</td><td align="center">0.57</td><td align="center"><b>3.69</b></td><td align="center">3%</td><td align="center">1.64</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>grok-4.1-fast</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><i><b>3.03</b></i></td><td align="center">—</td><td align="center">0.55</td><td align="center">0.71</td><td align="center"><b>3.43</b></td><td align="center">3%</td><td align="center">0.49</td><td align="center">—</td><td align="center">—</td></tr>
<tr><td><b>sensevoice-ondevice</b></td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center">—</td><td align="center"><b>4.00</b></td><td align="center">0.56</td></tr>
</tbody>
</table>

### Notes per use case

- **Children's Book Page Translation** — winner **claude-opus-5-5**; 24 items; judges gemini-judge, opus-judge.
- **Cantonese Colloquial Translation** — winner **claude-opus-5-5**; 231 items; judges opus-judge, gemini-judge.
- **Chinese Sentence Generation (word + context)** — winner **claude-sonnet-5**; 15 items; judges opus-judge, gemini-judge.
- **Cantonese/Mandarin Speech-to-Text** — winner **sensevoice-ondevice**; 12 items; judges .

## Methodology (why it's built this way)

- **Reference-free LLM-judge** for authenticity/register — the translation field's modern
  approach (GEMBA / GEMBA-MQM). FLORES `yue_Hant` is *written* Standard Chinese, so it's no
  gold set for colloquial Cantonese; we grade real + self-authored colloquial prompts instead.
- **Judge panel across families** (Claude + Gemini), anonymized & comparative, to blunt
  position/self-preference bias; small gaps ≈ ties. LLM-judges are weaker on dialects, so treat
  as directional and calibrate with humans for high-stakes calls.
- **Latency is a separate axis** (TTFT + total), never blended into quality. Some tasks are
  explicitly latency-insensitive and rank on quality alone — book pages are translated once and
  read many times.
- **Truncation is not a score.** Adaptive thinking spends from the same budget as the answer, so a
  too-small `max_tokens` returns `stop_reason=max_tokens` with zero text — which a judge reads as a
  quality collapse. Measured: claude-sonnet-5 at effort=max produced empty output on 11 of 24 book
  pages at `max_tokens=8000`, scoring 2.77 against its real 4.4-class range. Runners set each
  model's ceiling and raise on empty-at-max-tokens instead of scoring it.
- **Contextualization axis** — best practice for translation is to compare context modes
  (no-context · web-grounding · glossary/dictionary-RAG · TM few-shot). We ship no-context +
  web-grounding today; glossary-RAG (words.hk / Chain-of-Dictionary) and TM-RAG are roadmap.
- **Metrics roadmap**: add chrF++ / COMET / xCOMET / CometKiwi on any reference-based tier.
- **Data/licensing**: ship only the harness + self-authored/owned seed sets; pull third-party
  datasets (FLORES+, Tatoeba, Yue-TRANS) at runtime — several Cantonese corpora are CC-BY-NC.

Add a task: create `tasks/<name>/task.yaml` (models, judges, rubric, prompt, testset), then
`run_bench.py` → `judge.py` → `report.py` → `leaderboard.py`.
