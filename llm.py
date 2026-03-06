import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt: matches fine-tune training format ─────────────────────────────────
HUMANIZE_SYSTEM = (
    "Make this less robotic and more conversational, natural and human. "
    "Stay on the same topic and do not introduce new content that is not in the original text. "
    "Keep the key terms and subject words exactly as they are — do not replace them with synonyms. "
    "Do not add questions or interrogative sentences unless they already exist in the original text."
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
    system_msg = HUMANIZE_SYSTEM + f" Keep the output within {max_words} words."

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

    # ── Hard word cap: truncate at last sentence within limit ─────────────────
    words = content.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        # cut back to the last sentence boundary
        for punct in (".", "!", "?"):
            idx = truncated.rfind(punct)
            if idx != -1:
                truncated = truncated[:idx + 1]
                break
        content = truncated

    # ── Post layer: spelling + grammar fix only ───────────────────────────────
    spell_check = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Fix ONLY spelling mistakes and grammatical errors. Do not change the wording, the style, the sentence structure, or the content in any way. Do not add or remove anything. Return the text exactly as given, with only spelling and grammar corrected."},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    fixed = (spell_check.choices[0].message.content or content).strip()

    # ── Final word cap after spell check ──────────────────────────────────────
    words = fixed.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        for punct in (".", "!", "?"):
            idx = truncated.rfind(punct)
            if idx != -1:
                truncated = truncated[:idx + 1]
                break
        fixed = truncated

    return fixed


def get_available_models() -> list[str]:
    """Returns the supported OpenAI models for this tool."""
    return AVAILABLE_MODELS
