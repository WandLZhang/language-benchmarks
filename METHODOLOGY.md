# Methodology & references

How translation/language systems are evaluated and contextualized in the field, and the choices
this repo makes. (Condensed from a market survey; links are the primary sources.)

## Evaluation metrics

| Metric | Kind | Use here |
|---|---|---|
| **BLEU** (via [sacreBLEU](https://github.com/mjpost/sacrebleu)) | string overlap | legacy only; report with version signature |
| **chrF / chrF++** | char n-gram F | primary string metric — robust for Chinese/Cantonese (word tokenization is fragile). Roadmap on reference tiers |
| **COMET / xCOMET** | learned, source-aware | primary automatic metric where references exist ([COMET](https://github.com/Unbabel/COMET)). Roadmap |
| **CometKiwi** | learned, **reference-free** QE | scores with source+hypothesis only — fits our no-reference colloquial tiers. Roadmap |
| **MQM** | human error-span rubric | gold protocol (accuracy/fluency/terminology/style/locale); WMT standard |
| **GEMBA / GEMBA-MQM** | **LLM-as-judge** | what we use now — LLM emits scores/error-spans, reference-free ([GEMBA](https://arxiv.org/abs/2302.14520), [GEMBA-MQM](https://arxiv.org/abs/2310.13988)) |

Benchmarks worth modeling: [WMT shared tasks](https://www2.statmt.org/wmt25/) (Metrics/QE/Terminology),
[FLORES-200/FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus),
[lechmazur/translation](https://github.com/lechmazur/translation) (multi-judge, z-scored, bootstrap CIs — closest design analog),
IFMTBench (instruction-following MT).

**Cantonese caveat:** FLORES `yue_Hant` is *written* Standard Chinese in traditional script, **not**
colloquial Cantonese ([issue](https://github.com/facebookresearch/flores/issues/61)) — don't use it as an
authenticity gold set. Authentic-colloquial data: [Yue-Benchmark / Yue-TRANS](https://github.com/jiangjyjy/Yue-Benchmark)
(NAACL 2025), Tatoeba en–yue, [HKCanto-Eval](https://arxiv.org/abs/2503.12440).

## Contextualization axis (the differentiator)

Best practice for translation is to compare **context modes** and report the delta, because
retrieval helps unevenly:

- **no-context** — plain prompt (baseline).
- **web-grounding** — hosted search (Anthropic `web_search` / Gemini `google_search`); best for
  fresh **slang/neologisms** ([NEO-BENCH](https://aclanthology.org/2024.acl-long.749.pdf)). *Implemented.*
- **glossary / dictionary-RAG** — inject matched lexicon entries; **Chain-of-Dictionary** (up to
  +13× chrF++ low-resource, [CoD](https://arxiv.org/abs/2305.06575)); natural home for a words.hk/粵典
  termbase and **proper-noun/name dictionaries** (Pokémon/anime localization where HK≠Mandarin — a
  curated name dict beats grounding). *Roadmap.*
- **TM-RAG / fuzzy few-shot** — retrieve similar source+gold pairs ([T-Ragx](https://github.com/rayliuca/T-Ragx),
  Google [Adaptive Translation](https://docs.cloud.google.com/translate/docs/advanced/adaptive-translation)).
  Fuzzy/edit-distance retrieval often beats embeddings; example *quality* can matter more than similarity → tunable knobs. *Roadmap.*
- **long-context** — for document coherence, not glossary lookup; watch TTFT blow-up + lost-in-the-middle.

**When retrieval helps:** low-resource/dialect, terminology & brand/name consistency, register from
TM, fresh slang. **Marginal:** high-resource pairs with a strong base model (<0.5 COMET, can add noise).

## LLM-as-judge best practices (applied)

- Panel of **≥2 judge families** (Claude + Gemini), **anonymized + comparative**; don't judge a
  candidate with a model from its own family (self-preference) — the cross-family judge is the guard.
- Mitigate **position/verbosity bias** (order swap, length control); treat small gaps as ties.
- Headline via **pairwise Bradley-Terry + bootstrap CIs** ([Arena](https://arxiv.org/abs/2403.04132)); MQM-rubric for diagnostics. *(BT/CIs roadmap; current: comparative best-pick Win% + mean scores.)*
- LLM-judges are **less reliable on low-resource/dialects** (Fleiss' κ ≈ 0.3) — directional; calibrate with humans for high-stakes decisions.

## Rigor

Pin model versions/prompts/decoding params; bootstrap CIs; guard against test-set contamination
(FLORES devtest is public); pull third-party data at runtime and record licenses (several Cantonese
corpora are **CC-BY-NC** — not redistributable in a public repo).
