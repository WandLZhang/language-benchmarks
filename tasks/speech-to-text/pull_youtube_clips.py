"""
Build a tier-A test set: real Cantonese speech with a HUMAN reference transcript.

Ground-truth rule (from subtitle-everything/youtube-w-subtitles/README.md): use ONLY videos whose
caption track is a MANUAL zh-HK track. YouTube's auto-captions are themselves ASR output, so scoring
an ASR model against them measures agreement between two ASR systems, not accuracy. yt-dlp lists
manual tracks under "Available subtitles" and ASR ones under "automatic captions" — this script
refuses to proceed unless the requested language is in the manual list.

Each caption cue becomes one (audio, reference) pair: the cue text is the reference, and the audio is
that cue's time range cut from the video's audio.

Usage:
  python tasks/speech-to-text/pull_youtube_clips.py --video CiGJ46C-Dw4 --n 12 --out clips/brian
  python tasks/speech-to-text/bench_stt.py --tier A --clips clips/brian
"""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

import yt_dlp


def srt_time(s):
    """'00:01:02,345' -> seconds."""
    hh, mm, rest = s.split(":")
    ss, ms = rest.replace(".", ",").split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt(path):
    cues, block = [], []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            block.append(line.strip())
            continue
        if len(block) >= 3 and "-->" in block[1]:
            a, b = block[1].split("-->")
            cues.append({"start": srt_time(a.strip()), "end": srt_time(b.strip()),
                         "text": " ".join(block[2:]).strip()})
        block = []
    if len(block) >= 3 and "-->" in block[1]:
        a, b = block[1].split("-->")
        cues.append({"start": srt_time(a.strip()), "end": srt_time(b.strip()),
                     "text": " ".join(block[2:]).strip()})
    return cues


def assert_manual(video_url, lang):
    """Refuse ASR tracks — scoring ASR against ASR would be circular."""
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(video_url, download=False)
    manual = set(info.get("subtitles") or {})
    auto = set(info.get("automatic_captions") or {})
    if lang not in manual:
        raise SystemExit(f"REFUSED: '{lang}' is not a MANUAL track for this video "
                         f"(manual={sorted(manual)[:6]}, in-auto={lang in auto}). "
                         f"Pick a video whose captions a human wrote.")
    print(f"  caption check: '{lang}' IS a manual track (auto-captions also exist: {lang in auto}) — OK")
    return info.get("title", "")


def audio_language_guard(wav_path, ref_text, sherpa_dir):
    """Refuse TRANSLATED subtitle tracks.

    A channel can publish Cantonese subtitles over ENGLISH narration — the track is 'manual', but it is
    a translation, not a transcript, so scoring ASR against it is meaningless. (Caught in practice: a
    travel vlog captioned 「今日嘅ootd靈感係博物館遊客」 whose audio says "AND TODAY'S CONCEPT IS A
    MUSEUM VISITOR".) Transcribe one clip locally and compare scripts: if the reference is CJK but the
    audio comes back Latin, the track is translated.
    """
    if not sherpa_dir:
        print("  ! no --sherpa-dir: SKIPPING the audio-language guard (translated subs would go unnoticed)")
        return
    import wave
    import numpy as np
    import sherpa_onnx
    cand = [p for p in glob_onnx(sherpa_dir) if "int8" in p] or glob_onnx(sherpa_dir)
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=cand[0], tokens=os.path.join(sherpa_dir, "tokens.txt"), use_itn=True)
    wf = wave.open(wav_path)
    pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    s = rec.create_stream()
    s.accept_waveform(wf.getframerate(), pcm)
    rec.decode_stream(s)
    heard = s.result.text.strip()
    cjk = lambda t: sum(1 for c in t if "㐀" <= c <= "鿿")  # noqa: E731
    ref_cjk, heard_cjk = cjk(ref_text), cjk(heard)
    latin = sum(1 for c in heard if c.isascii() and c.isalpha())
    print(f"  audio-language guard: caption CJK={ref_cjk}, heard CJK={heard_cjk}, heard latin={latin}")
    print(f"    heard: {heard[:70]}")
    if ref_cjk >= 4 and heard_cjk <= 1 and latin >= 8:
        raise SystemExit("REFUSED: the caption track is CJK but the AUDIO IS ENGLISH — these are "
                         "TRANSLATED subtitles, not a transcript. Pick a video whose narration is "
                         "actually in the caption's language.")


def glob_onnx(d):
    import glob as _g
    return _g.glob(os.path.join(d, "*.onnx"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="YouTube video id or URL")
    ap.add_argument("--lang", default="zh-HK")
    ap.add_argument("--speech-lang", default="cantonese", choices=["cantonese", "mandarin"])
    ap.add_argument("--n", type=int, default=12, help="how many cues to keep")
    ap.add_argument("--min-sec", type=float, default=1.5)
    ap.add_argument("--max-sec", type=float, default=9.0)
    ap.add_argument("--min-chars", type=int, default=8)
    # Caption timings tend to land ON the first syllable, so a raw cut clips the word onset. A short
    # lead-in/tail keeps the utterance whole without pulling in the neighbouring cue.
    ap.add_argument("--pad-start", type=float, default=0.45)
    ap.add_argument("--pad-end", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sherpa-dir", default=os.getenv("SHERPA_DIR", ""),
                    help="SenseVoice dir; enables the audio-language guard (translated-subs check)")
    args = ap.parse_args()

    url = args.video if args.video.startswith("http") else f"https://www.youtube.com/watch?v={args.video}"
    title = assert_manual(url, args.lang)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "src")
        # player_client=android: the default web client 403s from datacenter IPs (verified here);
        # android still serves the audio stream. Other clients fail differently (ios/mweb: "format
        # not available", tv: "DRM protected").
        opts = {"quiet": True, "format": "bestaudio/best", "outtmpl": base + ".%(ext)s",
                "writesubtitles": True, "writeautomaticsub": False,
                "subtitleslangs": [args.lang], "subtitlesformat": "srt",
                "extractor_args": {"youtube": {"player_client": ["android"]}},
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}]}
        print("  downloading audio + manual subs…")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        wavs = [p for p in os.listdir(tmp) if p.endswith(".wav")]
        subs = [p for p in os.listdir(tmp) if p.endswith(".srt") or p.endswith(".vtt")]
        if not wavs or not subs:
            raise SystemExit(f"download incomplete (wav={wavs}, subs={subs})")
        src_wav, src_sub = os.path.join(tmp, wavs[0]), os.path.join(tmp, subs[0])
        if src_sub.endswith(".vtt"):  # normalize to srt
            srt = src_sub[:-4] + ".srt"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_sub, srt], check=True)
            src_sub = srt

        cues = [c for c in parse_srt(src_sub)
                if args.min_sec <= (c["end"] - c["start"]) <= args.max_sec
                and len(c["text"]) >= args.min_chars and "\n" not in c["text"]]
        # spread the picks across the video rather than taking the first N (intros are atypical)
        step = max(1, len(cues) // max(args.n, 1))
        picks = cues[::step][:args.n]
        print(f"  {len(cues)} usable cues -> keeping {len(picks)}")

        for i, c in enumerate(picks):
            stem = outdir / f"{args.video[-6:]}-{i:02d}"
            cut_start = max(0.0, c["start"] - args.pad_start)
            cut_dur = (c["end"] + args.pad_end) - cut_start
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(round(cut_start, 3)),
                            "-t", str(round(cut_dur, 3)), "-i", src_wav,
                            "-ar", "16000", "-ac", "1", str(stem) + ".wav"], check=True)
            (stem.with_suffix(".txt")).write_text(c["text"], encoding="utf-8")
            (stem.with_suffix(".json")).write_text(json.dumps(
                {"lang": args.speech_lang, "word": "", "has_word": True, "q": "",
                 "source": url, "title": title, "cue_start": c["start"], "cue_end": c["end"],
                 "start": round(cut_start, 3), "end": round(cut_start + cut_dur, 3),
                 "pad_start": args.pad_start, "pad_end": args.pad_end,
                 "captions": "manual (cleaned-up prose, not verbatim speech)"}, ensure_ascii=False), encoding="utf-8")
            print(f"    {stem.name}  {c['end']-c['start']:.1f}s  {c['text'][:44]}")
            if i == 0:
                audio_language_guard(str(stem) + ".wav", c["text"], args.sherpa_dir)
    print(f"\n-> {outdir}  (run: bench_stt.py --tier A --clips {outdir})")


if __name__ == "__main__":
    main()
