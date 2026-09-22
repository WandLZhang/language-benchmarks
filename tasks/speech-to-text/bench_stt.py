"""
Speech-to-text benchmark — which model should transcribe a spoken Cantonese/Mandarin reply.

Self-contained on purpose: engine/run_bench.py drives providers.call_model, which is text-only (no
audio part), so this task ships its own runner rather than refactoring the shared engine. It writes
results/<ts>/{raw.jsonl,summary.json,report.md} in the SAME shape engine/report.py emits, so
engine/leaderboard.py merges it into the wide table with no engine changes.

Candidates
  - Gemini audio-native (3.6-flash / 3.5-flash / 3.5-flash-lite)
  - Chirp 2 / Chirp 3 via Cloud STT v2 — the classic-ASR baseline
  - SenseVoice on-device via sherpa-onnx (--sherpa-dir) — the model already running on the phone in
    subtitle-everything/mobile-audio; unreachable from a web page, but decides the native question

Biasing axis (Gemini only): none | question | question+word. Biasing helps rare words but can make the
model emit a target word that was never spoken — which would wrongly satisfy `meaningful_usage` when the
answer is graded downstream. The NEGATIVE-CONTROL rows (has_word=false) measure that as false_insert.
The leaderboard is scored with the shipped config (question+word); the axis lives in report.md.

Tiers
  C  synthetic — Chirp 3 HD TTS of the testset (default, fully automated; cleaner than a phone mic)
  A  youtube   — real speech, MANUAL zh-HK captions ONLY (never kind:"asr", that would be circular)
  B  clips     — real phone recordings (true condition)

Run:
  source .venv/bin/activate
  LT_PROJECT=your-gcp-project python tasks/speech-to-text/bench_stt.py --tier C
"""
import argparse
import base64
import datetime
import difflib
import glob
import json
import os
import pathlib
import statistics
import time
from collections import defaultdict

import google.auth
import google.auth.transport.requests
import random
import requests
import yaml
from anthropic import AnthropicVertex
from google import genai
from google.genai import types

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent          # repo root, for repo-relative model_dir in task.yaml
PROJECT = os.getenv("LT_PROJECT", "wz-cloud-claude")
AUDIO_PROJECT = os.getenv("AUDIO_PROJECT", "wz-convo-live")   # where the TTS function is deployed
STT_REGION = os.getenv("STT_REGION", "us-central1")           # Chirp is regional
AUDIO_FN = os.getenv("AUDIO_FN", f"https://us-east4-{AUDIO_PROJECT}.cloudfunctions.net/convo_live_generate_audio")

_creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
_genai = genai.Client(vertexai=True, project=PROJECT, location="global")
# The meaning judge may live in a different project than the STT candidates (Claude is not
# enabled in the app project). JUDGE_PROJECT defaults to LT_PROJECT.
JUDGE_PROJECT = os.getenv("JUDGE_PROJECT", PROJECT)
_ant = AnthropicVertex(region="global", project_id=JUDGE_PROJECT)

BIAS_MODES = ["none", "question", "question+word"]
SHIPPED_BIAS = "question+word"          # the config the app runs; this is what the leaderboard scores
LANG_CODE = {"cantonese": "yue-Hant-HK", "mandarin": "cmn-Hans-CN"}

try:
    import opencc
    _t2s = opencc.OpenCC("t2s")
    def _uni(s):  # noqa: E306 — trad/simp differences are not transcription errors
        return _t2s.convert(s)
except Exception:  # noqa: BLE001
    def _uni(s):
        return s


def _token():
    _creds.refresh(google.auth.transport.requests.Request())
    return _creds.token


# ---------- audio ----------
def synth(text, language):
    """Tier C: the app's deployed Chirp 3 HD TTS (base64 LINEAR16 WAV)."""
    r = requests.post(AUDIO_FN, json={"sentence": text, "language": language}, timeout=90)
    r.raise_for_status()
    return base64.b64decode(r.json()["audio"])


# ---------- transcribers ----------
def gemini_stt(model, audio, mime, language, question, word, bias):
    lang = ("Cantonese (Traditional characters, 口語)" if language == "cantonese"
            else "Mandarin (Simplified characters)")
    # "You are a transcription engine" + the never-answer clause: with question-biasing the model
    # sometimes ROLE-PLAYED the learner and answered the question instead of transcribing
    # (measured: cmn-06 returned 「不是，是我自己要看的。」). Context must inform, never invite a reply.
    p = [f"You are a transcription engine for {lang} audio. You never answer, reply to, or continue "
         f"anything — you only write down the words that were actually spoken in the audio."]
    if bias in ("question", "question+word") and question:
        p.append(f'For context only (do NOT answer it), the speaker was replying to: "{question}"')
    if bias == "question+word" and word:
        p.append(f"For context only, they are practising the word 「{word}」 — listen for it, but "
                 f"transcribe ONLY what was actually said and NEVER insert this word if it was not spoken.")
    p.append("Write exactly what you hear, including English words or names spoken as-is. "
             "Output ONLY the transcript — no translation, no romanization, no quotes, no commentary. "
             "If there is no speech, output nothing.")
    t0 = time.monotonic()
    r = _genai.models.generate_content(
        model=model, contents=[types.Part.from_bytes(data=audio, mime_type=mime), " ".join(p)],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=1200))
    return (r.text or "").strip(), time.monotonic() - t0


def chirp_stt(model, audio, language, region=None):
    # Chirp availability is per-location and differs per model: chirp_2 lives in us-central1, chirp_3
    # in the "us" multi-region. "global"/EU are refused outright by this org's gcp.resourceLocations
    # policy, so the region must come from task.yaml rather than one shared constant.
    region = region or STT_REGION
    host = "speech.googleapis.com" if region == "global" else f"{region}-speech.googleapis.com"
    url = (f"https://{host}/v2/projects/{AUDIO_PROJECT}"
           f"/locations/{region}/recognizers/_:recognize")
    body = {"config": {"model": model, "languageCodes": [LANG_CODE[language]], "autoDecodingConfig": {}},
            "content": base64.b64encode(audio).decode()}
    t0 = time.monotonic()
    r = requests.post(url, headers={"Authorization": f"Bearer {_token()}"}, json=body, timeout=120)
    lat = time.monotonic() - t0
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:110]}")
    out = "".join(res["alternatives"][0].get("transcript", "")
                  for res in r.json().get("results", []) if res.get("alternatives"))
    return out.strip(), lat


# One recognizer per model dir, so several on-device models can be compared in a single run.
_sherpa = {}


def sherpa_stt(model_dir, wav_bytes):
    import io
    import wave
    import numpy as np
    import sherpa_onnx
    if model_dir not in _sherpa:
        cand = [p for p in glob.glob(os.path.join(model_dir, "*.onnx")) if "int8" in p] or \
               glob.glob(os.path.join(model_dir, "*.onnx"))
        tokens = os.path.join(model_dir, "tokens.txt")
        # Two Fun-ASR-Nano packages exist and they are not interchangeable. The `sherpa-onnx-
        # sense-voice-funasr-nano-*` repos are a single-file CTC export of its encoder, so they load
        # through from_sense_voice and stay non-autoregressive. The `sherpa-onnx-funasr-nano-*` repos
        # are the Qwen3-decoder build and need from_funasr_nano plus four separate artifacts.
        name = os.path.basename(model_dir.rstrip("/"))
        if name.startswith("sherpa-onnx-funasr-nano"):
            _sherpa[model_dir] = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                encoder_adaptor=os.path.join(model_dir, "encoder_adaptor.onnx"),
                llm=os.path.join(model_dir, "llm.onnx"),
                embedding=os.path.join(model_dir, "embedding.onnx"),
                tokenizer=model_dir)
        else:
            _sherpa[model_dir] = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=cand[0], tokens=tokens, use_itn=True)
    rec = _sherpa[model_dir]
    wf = wave.open(io.BytesIO(wav_bytes), "rb")
    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.monotonic()
    s = rec.create_stream()
    s.accept_waveform(wf.getframerate(), pcm)
    rec.decode_stream(s)
    return s.result.text.strip(), time.monotonic() - t0


def sherpa_dir_for(model, default_dir):
    """A sherpa row may name its own `model_dir` in task.yaml; otherwise use --sherpa-dir."""
    d = model.get("model_dir")
    if not d:
        return default_dir
    return d if os.path.isabs(d) else str(ROOT / d)


MEANING_JUDGE = {"id": "opus48-judge", "vertex_id": "claude-opus-4-8"}

MEANING_RUBRIC = """You are scoring SPEECH-TO-TEXT transcripts of Cantonese/Mandarin audio.

CRITICAL: the reference is a published caption, which is a CLEANED-UP, slightly formalised version of
what was actually said — not a verbatim transcript. Speakers use fillers, repetitions, colloquial
particles and looser grammar that captioners tidy away. So do NOT punish wording, particle, or
word-order differences from the reference.

Score each candidate transcript 1-5 on whether it captures WHAT WAS MEANT:
  5 = same meaning and intent; any differences are wording/particles/tidying
  4 = same meaning, one minor detail off or missing
  3 = mostly right, but a noticeable detail is wrong or dropped
  2 = partially right, meaning materially distorted
  1 = wrong, garbled, hallucinated, empty, or in the wrong language
Names and proper nouns DO matter: getting a name/brand/place clearly wrong caps the score at 3.

Return ONLY JSON: {"scores":{"A":{"meaning":n,"why":"<=10 words"},...}}"""


def judge_meaning(ref, outputs):
    """Comparative + anonymised: one call scores every candidate for this clip."""
    order = list(outputs.items())
    random.Random(hash(ref) & 0xFFFFFFFF).shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    blocks = "\n\n".join(f"### Candidate {lab}\n{txt or '(empty)'}" for lab, (_, txt) in zip(labels, order))
    user = f"Reference caption (cleaned-up):\n{ref}\n\n{blocks}\n\nScore every candidate. JSON only."
    m = _ant.messages.create(model=MEANING_JUDGE["vertex_id"], max_tokens=1500,
                             system=MEANING_RUBRIC, messages=[{"role": "user", "content": user}])
    txt = "".join(b.text for b in m.content if getattr(b, "type", None) == "text")
    a, b = txt.find("{"), txt.rfind("}")
    scores = json.loads(txt[a:b + 1]).get("scores", {})
    return {mid: (scores.get(lab) or {}).get("meaning") for lab, (mid, _) in zip(labels, order)}


# ---------- scoring ----------
def _norm(s):
    return _uni("".join(c for c in s if c.isalnum())).lower()


def accuracy(ref, hyp):
    return difflib.SequenceMatcher(None, _norm(ref), _norm(hyp)).ratio()


def coverage(ref, hyp):
    """Share of the REFERENCE captured by the hypothesis, ignoring extra speech. Tier A cue text does
    not always cover everything audible in the cue's time window (e.g. an intro the captioner skipped),
    so plain similarity penalises a model for correctly transcribing real speech."""
    a, b = _norm(ref), _norm(hyp)
    if not a:
        return None
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b).get_matching_blocks()) / len(a)


def cer(ref, hyp):
    a, b = _norm(ref), _norm(hyp)
    if not a:
        return 1.0
    ops = difflib.SequenceMatcher(None, a, b).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in ops if t != "equal") / len(a)


def load_items(tier, clips_dir):
    if tier == "C":
        items = [json.loads(l) for l in (HERE / "testset.jsonl").read_text().splitlines() if l.strip()]
        for it in items:
            it["audio"], it["mime"], it["ref"] = synth(it["text"], it["lang"]), "audio/wav", it["text"]
            print(f"  synth {it['id']} {it['word']} ({len(it['audio'])//1024}KB)")
        return items
    items = []
    for wav in sorted(glob.glob(os.path.join(clips_dir, "*.wav"))):
        base = wav[:-4]
        if not os.path.exists(base + ".txt"):
            continue
        meta = json.load(open(base + ".json")) if os.path.exists(base + ".json") else {}
        items.append({"id": os.path.basename(base), "lang": meta.get("lang", "cantonese"),
                      "word": meta.get("word", ""), "has_word": meta.get("has_word", True),
                      "q": meta.get("q", ""), "audio": open(wav, "rb").read(), "mime": "audio/wav",
                      "ref": open(base + ".txt", encoding="utf-8").read().strip()})
    print(f"  loaded {len(items)} clips from {clips_dir}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="C", choices=["A", "B", "C"])
    ap.add_argument("--clips", default="", help="tier A/B: dir of <name>.wav + <name>.txt (+ optional .json)")
    ap.add_argument("--sherpa-dir", default=os.getenv("SHERPA_DIR", ""), help="SenseVoice model dir")
    ap.add_argument("--models", default="", help="comma list of task.yaml model ids")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "task.yaml").read_text())
    models = cfg["models"]
    if args.models:
        want = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m["id"] in want]
    if not args.sherpa_dir:
        models = [m for m in models if m["provider"] != "sherpa"]

    items = load_items(args.tier, args.clips)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = HERE / "results" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\ntier {args.tier} · {len(items)} items · models {[m['id'] for m in models]}\n")

    # per (model, bias) accumulators; non-gemini providers only run the shipped-bias slot
    acc = defaultdict(list)
    cers = defaultdict(list)
    covs = defaultdict(list)
    lats = defaultdict(list)
    fi = defaultdict(lambda: [0, 0])
    errs = defaultdict(int)

    meanings = defaultdict(list)
    with open(outdir / "raw.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            shipped_hyps = {}
            for m in models:
                modes = BIAS_MODES if m["provider"] == "gemini" else [SHIPPED_BIAS]
                for bias in modes:
                    key = (m["id"], bias)
                    try:
                        if m["provider"] == "gemini":
                            hyp, lat = gemini_stt(m["vertex_id"], it["audio"], it["mime"], it["lang"],
                                                  it["q"], it["word"], bias)
                        elif m["provider"] == "stt_v2":
                            hyp, lat = chirp_stt(m["vertex_id"], it["audio"], it["lang"], m.get("region"))
                        else:
                            hyp, lat = sherpa_stt(sherpa_dir_for(m, args.sherpa_dir), it["audio"])
                        a, c = accuracy(it["ref"], hyp), cer(it["ref"], hyp)
                        acc[key].append(a)
                        cers[key].append(c)
                        covs[key].append(coverage(it["ref"], hyp))
                        lats[key].append(lat)
                        if not it["has_word"] and it["word"]:
                            fi[key][1] += 1
                            if _norm(it["word"]) in _norm(hyp):
                                fi[key][0] += 1
                        f.write(json.dumps({"tier": args.tier, "item_id": it["id"], "model_id": m["id"],
                                            "bias": bias, "lang": it["lang"], "word": it["word"],
                                            "has_word": it["has_word"], "ref": it["ref"], "hyp": hyp,
                                            "acc": round(a, 3), "cer": round(c, 3), "lat_s": round(lat, 2)},
                                           ensure_ascii=False) + "\n")
                        f.flush()
                        if bias == SHIPPED_BIAS:
                            shipped_hyps[m["id"]] = hyp
                        print(f"  {m['id']:22} {bias:14} {it['id']} acc={a:.2f} {lat:5.2f}s  {hyp[:42]}")
                    except Exception as e:  # noqa: BLE001
                        errs[key] += 1
                        if bias == SHIPPED_BIAS:
                            shipped_hyps[m["id"]] = ""
                        print(f"  {m['id']:22} {bias:14} {it['id']} ERROR {str(e)[:60]}")
            # meaning score: the reference is a cleaned-up caption, so character metrics understate
            # correctness; a judge decides whether the MEANING survived.
            if shipped_hyps:
                try:
                    for mid, sc in judge_meaning(it["ref"], shipped_hyps).items():
                        if sc is not None:
                            meanings[mid].append(float(sc))
                    print(f"    meaning: " + "  ".join(
                        f"{k}={meanings[k][-1]:.0f}" for k in shipped_hyps if meanings.get(k)))
                except Exception as e:  # noqa: BLE001
                    print(f"    meaning judge failed: {str(e)[:60]}")

    def mean(xs):
        return statistics.mean(xs) if xs else None

    def p50(xs):
        return sorted(xs)[len(xs) // 2] if xs else None

    # summary.json — shape consumed by engine/leaderboard.py (scored at the shipped bias)
    summary = {"task": cfg["name"], "title": cfg["title"], "n_items": len(items),
               "judges": [], "tiers": [args.tier], "models": {}}
    for m in models:
        k = (m["id"], SHIPPED_BIAS)
        n_fi, d_fi = fi[k]
        summary["models"][m["id"]] = {
            "overall": mean(meanings[m["id"]]), "exact_sim": mean(acc[k]),
            "cer": mean(cers[k]), "coverage": mean(covs[k]),
            "false_insert": (n_fi / d_fi) if d_fi else None,
            "win_pct": None,                      # reference-based: no comparative judge
            "p50_total_s": p50(lats[k]), "p95_total_s": None, "ttft_p50_s": None,
            "ok": len(acc[k]), "n": len(items), "errors": errs[k],
            "by_tier": {args.tier: mean(acc[k])},
            "bias_axis": {b: {"overall": mean(acc[(m["id"], b)]), "cer": mean(cers[(m["id"], b)]),
                              "false_insert": (fi[(m['id'], b)][0] / fi[(m['id'], b)][1])
                              if fi[(m['id'], b)][1] else None}
                          for b in (BIAS_MODES if m["provider"] == "gemini" else [SHIPPED_BIAS])},
            "grounding": {"off": None, "on": None, "delta": None},  # n/a -> renders "—"
        }
    ranked = sorted(summary["models"], key=lambda x: (summary["models"][x]["overall"] or 0), reverse=True)
    summary["winner"] = ranked[0] if ranked else None
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # report.md — human view, incl. the bias axis the leaderboard doesn't show
    L = [f"# {cfg['title']} — {ts}\n", f"Tier **{args.tier}** · {len(items)} items · project `{PROJECT}`\n",
         "## Scored at the shipped biasing config (question + target word)\n",
         "| Model | Accuracy | CER | p50 s | False-insert | err |", "|---|---|---|---|---|---|"]
    def _f(v, nd=3):
        return "—" if v is None else f"{v:.{nd}f}"

    for mid in ranked:
        d = summary["models"][mid]
        if d["overall"] is None:          # every call failed (e.g. model unsupported for this language)
            L.append(f"| {mid} | — | — | — | — | {d['errors']} (all failed) |")
            continue
        L.append(f"| {mid} | **{_f(d['overall'])}** | {_f(d['cer'])} | "
                 f"{_f(d['p50_total_s'], 2)} | {_f(d['false_insert'], 2)} | {d['errors']} |")
    L += ["\n## Biasing axis (Gemini only)\n",
          "Biasing lifts rare-word accuracy but can insert a word that was never spoken — which would "
          "wrongly satisfy `meaningful_usage` downstream. `false_insert` is measured on the "
          "`has_word:false` negative controls.\n",
          "| Model | Bias | Accuracy | CER | False-insert |", "|---|---|---|---|---|"]
    for m in models:
        for b, d in summary["models"][m["id"]]["bias_axis"].items():
            if d["overall"] is None:
                continue
            L.append(f"| {m['id']} | {b} | {_f(d['overall'])} | {_f(d['cer'])} | {_f(d['false_insert'], 2)} |")
    if args.tier == "C":
        L.append("\n> Tier C is **synthesized** speech — cleaner than a phone mic, so these numbers are "
                 "optimistic. Run tier A (real YouTube speech, manual captions) and tier B (phone clips) "
                 "before locking a model.")
    (outdir / "report.md").write_text("\n".join(L))

    print("\n=== SUMMARY (shipped bias) ===")
    for mid in ranked:
        d = summary["models"][mid]
        if d["overall"] is None:
            print(f"  {mid:22} ALL FAILED ({d['errors']} errors)")
            continue
        print(f"  {mid:22} acc={_f(d['overall'])} cer={_f(d['cer'])} "
              f"p50={_f(d['p50_total_s'], 2)}s false-insert={_f(d['false_insert'], 2)} err={d['errors']}")
    print(f"\n-> {outdir}\nNext: python engine/leaderboard.py")


if __name__ == "__main__":
    main()
