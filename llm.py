import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt: matches fine-tune training format ─────────────────────────────────
HUMANIZE_SYSTEM = (
    "Make this less robotic and more conversational, natural and human. "

    "STRICT CONTENT RULES — breaking any of these is a failure:\n"
    "1. Do NOT add anything. No new facts, no new opinions, no new examples, no new sentences, no personal commentary.\n"
    "2. Do NOT remove anything. Every fact, name, date, event, and idea from the original must appear in the output.\n"
    "3. Do NOT change the topic or drift from the subject. If the text is about X, your output must stay about X.\n"
    "4. Do NOT open with generic phrases like 'X was a great person' or 'X is widely known as' — these are robotic fillers.\n"
    "5. Every word must be spelled correctly. No typos. No spelling mistakes of any kind.\n"

    "STYLE RULES — apply these to sound human:\n"
    "- Mix short punchy sentences with longer ones.\n"
    "- Avoid AI words: comprehensive, foster, delve, transformative, tapestry, pivotal, unlocked.\n"
    "- Write conversationally — like a real person explaining something they know well."
)

AVAILABLE_MODELS = [
    "ft:gpt-4.1-mini-2025-04-14:quantal::DDpmoxZI",
    "ft:gpt-4.1-nano-2025-04-14:quantal::DDpkniYG",
]

DEFAULT_MODEL = AVAILABLE_MODELS[0]


def humanize_text(ai_text: str, model: str = DEFAULT_MODEL) -> str:
    """Rewrite AI text as human-written."""
    if not ai_text or not ai_text.strip():
        raise ValueError("Input text cannot be empty.")

    input_words = len(ai_text.split())
    max_words = int(input_words * 1.4)
    system_msg = HUMANIZE_SYSTEM + f"\n6. LENGTH: Your output must not exceed {max_words} words. The input is {input_words} words — keep within 40% of that."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": ai_text},
        ],
    )
    content = response.choices[0].message.content
    if not content:
        finish_reason = response.choices[0].finish_reason
        raise ValueError(f"Model returned empty response. Finish reason: {finish_reason}")

    # ── Post layer: spelling fix only ─────────────────────────────────────────
    spell_check = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a spell checker. Fix ONLY misspelled words. Do not change anything else — not the wording, not the grammar, not the sentence structure, not the style. Return the text exactly as given except for corrected spelling."},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    fixed = spell_check.choices[0].message.content
    return (fixed or content).strip()


def get_available_models() -> list[str]:
    """Returns the supported OpenAI models for this tool."""
    return AVAILABLE_MODELS
