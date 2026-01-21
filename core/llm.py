from __future__ import annotations

from typing import Dict, List, Tuple, Any

from anthropic import Anthropic


def language_instruction(lang_code: str) -> str:
    if lang_code == "es":
        return "Responde en español, claro y profesional. Usa listas cuando ayuden."
    if lang_code == "pt":
        return "Responda em português, claro e profissional. Use listas quando ajudar."
    if lang_code == "en":
        return "Answer in English, clear and professional. Use bullets when helpful."
    return "Respond in the user's language (Spanish or Portuguese)."


def build_system_prompt(answer_lang: str) -> str:
    return (
        "You are the Democracia+ assistant.\n"
        "Answer using ONLY the provided Democracia+ materials excerpts.\n"
        "If the info is not present, say so and ask what is missing.\n"
        "Prefer structured, actionable outputs.\n\n"
        + language_instruction(answer_lang)
    )


def format_excerpts(retrieved: List[Tuple[Dict[str, Any], float]]) -> str:
    if not retrieved:
        return "No relevant excerpts were retrieved."
    parts = []
    for i, (sec, score) in enumerate(retrieved, start=1):
        parts.append(
            f"[Excerpt {i} | {sec.get('path','')} | pages {sec.get('page_start')}–{sec.get('page_end')} | sim {score:.3f}]\n"
            f"{sec.get('text','')}"
        )
    return "\n\n".join(parts)


def build_user_turn(query: str, retrieved: List[Tuple[Dict[str, Any], float]], persona_hint: str) -> str:
    ctx = format_excerpts(retrieved)
    persona_block = f"\n\nConversation focus:\n{persona_hint}\n" if persona_hint else ""
    return (
        "Use the following Democracia+ excerpts to answer the question.\n"
        "Rules:\n"
        "- Use only the excerpts as factual basis.\n"
        "- If excerpts are insufficient, say what is missing.\n"
        "- Provide a structured, actionable answer.\n\n"
        f"EXCERPTS:\n{ctx}"
        f"{persona_block}\n\n"
        f"QUESTION:\n{query}"
    )


def call_claude(api_key: str, model: str, temperature: float, max_tokens: int, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    return "".join([b.text for b in resp.content if getattr(b, 'type', None) == 'text'])
