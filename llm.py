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

    # ── Post layer: fact restore + spelling fix (minimal touch) ─────────────
    cleanup = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fact checker. You are given an ORIGINAL text and a REWRITTEN version.\n"
                    "Your job — do ONLY these three things, nothing else:\n"
                    "1. If any proper nouns, names, dates, numbers, or specific terms from the ORIGINAL are missing from the REWRITTEN version, insert them into the nearest relevant sentence. Do not rewrite the sentence — just insert the missing word or phrase.\n"
                    "2. Remove any sentence from the REWRITTEN version that contains an idea, claim, or detail that does not exist in the ORIGINAL. Delete the whole sentence — do not rewrite it.\n"
                    "3. Fix spelling mistakes only. Do not fix grammar, do not change sentence structure, do not change wording.\n"
                    "Do NOT rewrite, restructure, or rephrase anything. Return the REWRITTEN text with only the minimum changes above."
                ),
            },
            {
                "role": "user",
                "content": f"ORIGINAL:\n{ai_text}\n\nREWRITTEN:\n{content}",
            },
        ],
        temperature=0,
    )
    fixed = (cleanup.choices[0].message.content or content).strip()

    # ── Spelling pass ─────────────────────────────────────────────────────────
    spell = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Fix ONLY spelling mistakes, punctuation errors (missing or wrong full stops, commas, apostrophes), and basic grammar errors. Do not change the wording, the sentence structure, or the style in any way. Do not add or remove any content. Return the text exactly as given with only those corrections applied."},
            {"role": "user", "content": fixed},
        ],
        temperature=0,
    )
    fixed = (spell.choices[0].message.content or fixed).strip()

    # ── Final word cap after cleanup ──────────────────────────────────────────
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
