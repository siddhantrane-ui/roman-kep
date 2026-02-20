"""
Dataset preparation for LLaMA humanizer fine-tuning.

Pipeline:
  1. Sample N human-written texts from AI_Human.csv (generated=0)
  2. Use local Ollama model to rewrite each as formal AI-style text (the input side)
  3. Save pairs as train.jsonl:  { "input": AI-style, "output": human original }

Run:  python prepare_dataset.py
Output: train.jsonl  (ready for fine-tuning)
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "AI_Human.csv"
OUTPUT_PATH   = "train.jsonl"
OLLAMA_MODEL  = "qwen3:8b"
OLLAMA_URL    = "http://localhost:11434/api/generate"
SAMPLE_SIZE   = 5000   # how many pairs to generate
MIN_WORDS     = 80     # skip texts shorter than this
MAX_WORDS     = 600    # skip texts longer than this (avoid token limits)
RANDOM_SEED   = 42
# ─────────────────────────────────────────────────────────────────────────────

AI_STYLE_PROMPT = """Rewrite the following text to sound like it was written by an AI language model. Make it formal, structured, and polished:
- Use clear topic sentences for each paragraph
- Use transition words: Furthermore, Additionally, Moreover, In conclusion
- Remove all contractions (it's → it is, don't → do not)
- Use passive voice occasionally
- Make sentence lengths uniform and medium-length
- Remove any personal opinions, hedges, or casual phrasing
- Sound comprehensive and well-organized

Return ONLY the rewritten text. No labels or explanation.

Text to rewrite:
"""


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"  [ollama error] {e}")
        return None


def load_human_texts(csv_path: str, min_words: int, max_words: int) -> list[str]:
    texts = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["generated"]) != 0.0:
                continue
            wc = len(row["text"].split())
            if min_words <= wc <= max_words:
                texts.append(row["text"].strip())
    return texts


def already_done(output_path: str) -> int:
    p = Path(output_path)
    if not p.exists():
        return 0
    return sum(1 for _ in p.open(encoding="utf-8"))


def main():
    print(f"Loading human texts from {CSV_PATH}...")
    human_texts = load_human_texts(CSV_PATH, MIN_WORDS, MAX_WORDS)
    print(f"  Found {len(human_texts):,} qualifying human texts")

    random.seed(RANDOM_SEED)
    random.shuffle(human_texts)
    selected = human_texts[:SAMPLE_SIZE]

    done = already_done(OUTPUT_PATH)
    if done:
        print(f"  Resuming — {done} pairs already written, skipping those")
    selected = selected[done:]

    print(f"\nGenerating {len(selected)} AI-style versions with {OLLAMA_MODEL}...")
    print(f"Output → {OUTPUT_PATH}\n")

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for i, human_text in enumerate(selected, start=done + 1):
            print(f"[{i}/{SAMPLE_SIZE}] ", end="", flush=True)

            ai_text = ollama_generate(AI_STYLE_PROMPT + human_text)
            if not ai_text:
                print("skipped (no response)")
                continue

            record = {
                "input": ai_text,
                "output": human_text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"done ({len(ai_text.split())} words out)")

    total = already_done(OUTPUT_PATH)
    print(f"\nDone. {total} pairs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
