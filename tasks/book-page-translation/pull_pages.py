"""Build the book-page testset from the REAL starred story chats in cloud-claude's Firestore.

Pulls the first N page photos of a handful of books, in the order they were shot, and OCRs each
page once so the glossary-RAG arm has a query string. The photos are the only uncorrupted record in
those chats — message ordering and the assistant replies are not trustworthy (see
cloud-claude/test_scripts/rebuild_book_chats.py), so nothing stored in the assistant messages is
used as a reference here. The judge scores against the page photo itself.

Pages are kept CONSECUTIVE from page 1 of each book on purpose: the `history` context arm needs a
real story to accumulate.

Writes:
  pages/<chatId>-<nn>.jpg   the page photos          (gitignored — third-party book content)
  testset.jsonl             one row per page          (gitignored — embeds Storage download tokens)

Usage:
    source .venv/bin/activate
    python tasks/book-page-translation/pull_pages.py --pages 6
"""
import argparse
import json
import pathlib
import time
import urllib.request

import firebase_admin
from firebase_admin import firestore
from google import genai
from google.genai import types

HERE = pathlib.Path(__file__).parent
PAGES_DIR = HERE / "pages"
PROJECT = "wz-cloud-claude"
UID = "xoBY9nLz8ObwvIRPdJ855EBmAlv2"

# Books to sample: two Chinese fairy tales and two English picture books, so both translation
# directions and both registers (folk tale / modern picture book) are represented.
# 灰姑娘 is first on purpose — it carries the known page-3 loss this whole effort started from.
BOOKS = [
    {"chat_id": "QjYqcD7epRfGcIs96t40", "book": "灰姑娘",        "source_lang": "zh"},
    {"chat_id": "tqE2KK5PAERzbLNtT6Lu", "book": "小紅帽",        "source_lang": "zh"},
    {"chat_id": "KWfYUOTik7y4d03fPnp1", "book": "Grumpy Monkey", "source_lang": "en"},
    {"chat_id": "jFzkbSUuu4G7xgPG6iuz", "book": "The garden, the curtain and the cross", "source_lang": "en"},
]

# System prompts live in Firestore (prompts/<uid>/userPrompts/<id>) and are the same ones the app
# sends. zh pages use 中→粵, en pages use EN→普粵 (books).
TEMPLATE = {"zh": "g8QTqrl3O40ex8pmBSvf", "en": "K731ZzMJXnlP85BNCFmY"}

OCR_MODEL = "gemini-3.5-flash"
OCR_SYS = ("Transcribe the printed body text of this children's book page, exactly as printed. "
           "Ignore pinyin/jyutping romanization guides printed above or below the characters, "
           "ignore page numbers, and ignore anything drawn rather than typeset. "
           "Output ONLY the transcribed text, no commentary.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=6, help="pages per book, from page 1")
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR pass (glossary arm needs it)")
    args = ap.parse_args()

    if not firebase_admin._apps:
        firebase_admin.initialize_app(options={"projectId": PROJECT})
    db = firestore.client()
    gc = genai.Client(vertexai=True, project=PROJECT, location="global")
    PAGES_DIR.mkdir(exist_ok=True)

    rows = []
    for b in BOOKS:
        ref = (db.collection("chats").document(UID)
               .collection("conversations").document(b["chat_id"]))
        title = (ref.get().to_dict() or {}).get("title", b["book"])
        photos = [(m.id, m.to_dict()) for m in ref.collection("messages").order_by("timestamp").stream()
                  if (m.to_dict() or {}).get("role") == "user" and (m.to_dict() or {}).get("image")]
        print(f"\n{b['book']} ({title}) — {len(photos)} photos, taking first {args.pages}")

        for idx, (msg_id, d) in enumerate(photos[:args.pages], start=1):
            dest = PAGES_DIR / f"{b['chat_id']}-{idx:02d}.jpg"
            if not dest.exists():
                dest.write_bytes(urllib.request.urlopen(d["image"]["url"], timeout=90).read())

            source_text = ""
            if not args.no_ocr:
                for attempt in range(3):
                    try:
                        r = gc.models.generate_content(
                            model=OCR_MODEL,
                            contents=[types.Part.from_bytes(data=dest.read_bytes(),
                                                            mime_type="image/jpeg"), OCR_SYS],
                            config=types.GenerateContentConfig(temperature=0, max_output_tokens=2000))
                        source_text = (r.text or "").strip()
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            print(f"    OCR failed for page {idx}: {e}")
                        else:
                            time.sleep(2 * (attempt + 1))

            rows.append({
                "id": f"{b['chat_id'][:6]}-{idx:02d}",
                "chat_id": b["chat_id"], "book": b["book"], "source_lang": b["source_lang"],
                "template_id": TEMPLATE[b["source_lang"]],
                "page_index": idx, "message_id": msg_id,
                "image_file": dest.name, "image_url": d["image"]["url"],
                "source_text": source_text,
            })
            print(f"  page {idx:2d}  {dest.name}  {len(dest.read_bytes())//1024:4d}KB  "
                  f"ocr={source_text[:56].replace(chr(10), ' ')!r}")

    out = HERE / "testset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} pages -> {out}")
    if any(not r["source_text"] for r in rows):
        n = sum(1 for r in rows if not r["source_text"])
        print(f"WARNING: {n} page(s) have no OCR text — the glossary arm will run ungrounded on those.")


if __name__ == "__main__":
    main()
