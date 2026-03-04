import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt: matches fine-tune training format ─────────────────────────────────
HUMANIZE_SYSTEM = (
    "Make this less robotic and more conversational, natural and human. "
    "Keep the content and context exactly the same as the original — do not generate new content, do not change the topic, do not add new ideas, and do not remove any existing information. "
    "Every fact, name, date, and idea from the original must appear in the output unchanged. "
    "Do NOT open with generic AI-style sentences like 'X was a great person' or 'X is known as a leader' — these are robotic. "
    "Instead, open in a way that reflects the actual point or argument of the original text. "
    "Do not oversimplify or water down the content. "
    "All words must be spelled correctly — no typos or spelling mistakes of any kind."
)

AVAILABLE_MODELS = [
    "ft:gpt-4.1-mini-2025-04-14:quantal::DDpmoxZI",
    "ft:gpt-4.1-nano-2025-04-14:quantal::DDpkniYG",
]

DEFAULT_MODEL = AVAILABLE_MODELS[0]


def humanize_text(ai_text: str, model: str = DEFAULT_MODEL) -> str:
    """Rewrite AI text as human-written, then fix spelling."""
    if not ai_text or not ai_text.strip():
        raise ValueError("Input text cannot be empty.")

    # ── Pass 1: humanize ──────────────────────────────────────────────────────
    r1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": HUMANIZE_SYSTEM},
            {"role": "user", "content": ai_text},
        ],
    )
    humanized = r1.choices[0].message.content
    if not humanized:
        finish_reason = r1.choices[0].finish_reason
        raise ValueError(f"Model returned empty response. Finish reason: {finish_reason}")

    # ── Pass 2: spell-check only ──────────────────────────────────────────────
    r2 = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Fix only spelling mistakes in the text. Do not change any words, grammar, punctuation, style, or content. Return the corrected text only."},
            {"role": "user", "content": humanized},
        ],
        temperature=0,
    )
    spellchecked = r2.choices[0].message.content
    return (spellchecked or humanized).strip()


def get_available_models() -> list[str]:
    """Returns the supported OpenAI models for this tool."""
    return AVAILABLE_MODELS
