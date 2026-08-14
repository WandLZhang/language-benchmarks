# Cantonese/Mandarin Speech-to-Text

Which model should transcribe a short spoken Cantonese/Mandarin reply? The bar is **proper nouns and
rare vocabulary** — the words a learner needs back verbatim. Classic ASR tends to mangle exactly those;
audio-native LLMs tend not to.

Reference-based, so there is no judge panel for correctness — but the **headline metric is judged**,
because of the caption problem below.

## Why this task has its own runner

`engine/run_bench.py` drives `providers.call_model`, which is text-only — no audio part. Rather than
refactor the shared engine, `bench_stt.py` is self-contained and writes `results/<ts>/summary.json` in
the same shape `engine/report.py` emits, so `engine/leaderboard.py` merges it into the wide table
unchanged.

## Tiers

| Tier | Audio | Reference | What it is good for |
|---|---|---|---|
| **C** synthetic | Chirp 3 HD TTS of `testset.jsonl` | verbatim (we wrote it) | plants specific rare words + proper nouns; exact metrics are valid |
| **A** youtube | real speech, cut on caption cues | published **manual** captions | real accents/pace; the ranking here differs from C |
| **B** clips | your own recordings | you type it | the true product condition |

## Two traps this harness refuses to fall into

1. **ASR captions.** YouTube auto-captions are themselves ASR output — scoring against them measures
   agreement between two ASR systems. `pull_youtube_clips.py` refuses any language that is not in the
   **manual** subtitle list.
2. **Translated captions.** A channel can publish Cantonese subtitles over *English* narration. The
   track is "manual", but it is a translation, not a transcript. Caught in practice: a vlog captioned
   「今日嘅ootd靈感係博物館遊客」 whose audio says *"AND TODAY'S CONCEPT IS A MUSEUM VISITOR"*. The
   extractor now transcribes one clip locally and **refuses** the video if the caption is CJK but the
   audio comes back Latin (`--sherpa-dir` enables this check).

## Why `overall` is a judged meaning score

Published captions are **cleaned-up prose**, not verbatim speech — captioners drop fillers, repetitions
and colloquial particles, and tidy the grammar. Character metrics therefore punish a model for being
right: on tier A every model lands near CER ~0.95 regardless of quality. So `overall` is a **1-5 meaning
score** from a judge told the reference is cleaned-up (wording differences are free; a wrong name caps
the score at 3). `cer`, `exact_sim` and `coverage` remain as diagnostics.

The judge is `claude-opus-4-8`, which is neutral by construction here: no Claude model is an STT
candidate. Set `JUDGE_PROJECT` if Claude lives in a different project than the candidates.

## Biasing axis

Gemini candidates run three ways: `none`, `question`, `question+word` (leaderboard scores the last).
Biasing helps rare words, but a biased model can emit the target word that was **never spoken** — which
would wrongly satisfy `meaningful_usage` in a downstream grader. The `has_word:false` rows are negative
controls that measure this as `false_insert`. Early on it hit 0.25; hardening the prompt
("you are a transcription engine… never insert this word if it was not spoken") took it to 0.00.

## Run

```bash
source .venv/bin/activate
pip install sherpa-onnx yt-dlp numpy            # ffmpeg + bzip2 also required

# tier C (synthetic; needs a deployed Chirp 3 HD TTS endpoint via AUDIO_FN)
LT_PROJECT=your-project AUDIO_PROJECT=your-tts-project python tasks/speech-to-text/bench_stt.py --tier C

# tier A (real speech; manual captions only)
python tasks/speech-to-text/pull_youtube_clips.py --video <id> --lang yue-HK \
    --n 12 --min-sec 3 --max-sec 9 --sherpa-dir models/sherpa-onnx-sense-voice-* --out clips/<name>
python tasks/speech-to-text/bench_stt.py --tier A --clips clips/<name> --sherpa-dir models/sherpa-onnx-sense-voice-*

python engine/leaderboard.py
```

`testset.jsonl` is gitignored (it plants the owner's real names/places to test proper-noun retention);
`testset.example.jsonl` is the sanitised, reproducible version — copy it over to run the task.

## Result that decided the shipped model

On **real Cantonese speech** (meaning 1-5): SenseVoice on-device **4.00**, chirp_3 3.75,
gemini-3.5-flash 3.67, chirp_2 3.33, gemini-3.6-flash 2.92, gemini-3.5-flash-lite 2.58.

On **synthetic audio with planted proper nouns**, the ranking inverts: gemini-3.5-flash is best
(0.98 exact, **8/8** proper nouns) and SenseVoice keeps **0/8** ("Crunchyroll" → "CRUNCHY ROALD").

So the two tiers answer different questions, and the choice depends on the product. `chinese-convo-live`
ships **SenseVoice self-hosted** (sherpa-onnx on Cloud Run): best real-Cantonese meaning, RTF 0.083
(~0.4 s for a 5 s utterance, ~8x faster than the Gemini path), accepting garbled names because the UI
lets the user fix them before the answer is graded.
