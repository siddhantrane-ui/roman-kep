import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt 1: ESL Humanizer ───────────────────────────────────────────────────
HUMANIZE_PROMPT = """Role: You are an expert human writer with a background in narrative journalism. Your goal is to rewrite the provided text to pass deep linguistic analysis.

Core Directives:

Variable Sentence Dynamics: Mix very short, punchy sentences (3–5 words) with long, complex, multi-clausal sentences. This creates "burstiness."

Anti-AI Vocabulary: Strictly avoid "AI-isms" like comprehensive, foster, delve, transformative, tapestry, pivotal, or unlocked. Use visceral, gritty, or common-sense language instead.

Non-Linear Flow: Humans often circle back to an idea or use "sentence fragments" for emphasis. Don't be too organized. Avoid the "First, Second, Third, In Conclusion" structure.

Subjective Tone: Inject a sense of opinion or a "human lens." Use subtle idioms or metaphors that aren't overused clichés.

Punctuation Variety: Use em-dashes (—), semicolons (;), and parentheses to break the visual flow of the text.

Imperfection: Occasionally use a slightly less "optimized" word choice if it sounds more natural.

Constraint: Retain all the factual information of the input, but completely destroy the "AI footprint." Do not be overly polite or formal."""


def humanize_text(ai_text: str, model: str = "gpt-5", temperature: float = 1.0) -> str:
    """Rewrite AI text as human-written."""
    if not ai_text or not ai_text.strip():
        raise ValueError("Input text cannot be empty.")

    # Reasoning models (o-series, gpt-5) don't support temperature
    REASONING_MODELS = {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-5"}
    is_reasoning = any(model.startswith(m) for m in REASONING_MODELS)

    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": HUMANIZE_PROMPT},
            {"role": "user", "content": f"Rewrite this AI-generated text so it reads as genuinely human-written. Apply everything from your instructions — break the sentence patterns, vary lengths aggressively, use unexpected word choices, remove all AI structural tells. Keep all the facts and meaning intact. The input is {len(ai_text.split())} words — your output must stay within ±10% of that count:\n\n{ai_text}"},
        ],
        "max_completion_tokens": 8000,
    }
    if not is_reasoning:
        params["temperature"] = temperature

    response = client.chat.completions.create(**params)
    content = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason
    if not content:
        raise ValueError(f"Model returned empty response. Finish reason: {finish_reason}")
    if finish_reason == "length":
        raise ValueError("Response was cut off — input text may be too long. Try a shorter passage.")
    return content.strip()


def get_available_models() -> list[str]:
    """Returns the supported OpenAI models for this tool."""
    return [
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-4.1-mini",
    ]
