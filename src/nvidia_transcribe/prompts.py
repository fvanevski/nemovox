"""Official prompt presets and generation-loop guardrail for Nemotron 3.5 Transcribe."""

from __future__ import annotations

import re

PROMPT_PRESETS: dict[str, str] = {
    "verbatim": (
        "Audio transcription. Language: as spoken. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization. Output: transcript only."
    ),
    "readable": (
        "Audio transcription. Language: as spoken. "
        "Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. "
        "Formatting: punctuation and capitalization. Output: transcript only."
    ),
    "speaker": (
        "Transcribe this audio with punctuation and capitalization. "
        "Include all spoken words and verbal fillers. "
        "Write numbers in their standard written form. "
        "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
    ),
    "multispeaker_sot": (
        "Transcribe this audio. Include all spoken words and verbal fillers. "
        "Write numbers as spoken. Use no punctuation and no capitalization. "
        "Start every speaker turn with a speaker tag such as <spk:0>, <spk:1>, <spk:2>, etc.; "
        "assign one stable tag per speaker and keep speaker identities consistent. "
        "Output only the speaker-tagged transcript."
    ),
    "rich": (
        "Transcribe this audio with punctuation and capitalization. "
        "Include all spoken words and verbal fillers. "
        "Write numbers in their standard written form. "
        "Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims and speaking style."
    ),
    "fleurs": (
        "Audio transcription. Language: as spoken, in its standard writing system. "
        "Wording: verbatim, including fillers and repetitions. "
        "Formatting: punctuation and capitalization; numbers in their standard written form. "
        "Output: transcript only."
    ),
    "ascend": (
        "Transcribe the audio word for word in the original language. Preserve repetitions, "
        "hesitations, false starts, incomplete words, and the speaker's grammar. Do not translate, "
        "summarize, correct the wording, or complete an unfinished phrase. Return only the "
        "transcript and stop when the speech ends. Write Mandarin in simplified Chinese characters "
        "and English in the Latin alphabet. Use a plain transcript without sentence punctuation."
    ),
    "translate": "Please translate the given speech to {target_language}.",
    "lid": (
        "Identify the spoken language. Respond with exactly one of the following "
        "language labels and nothing else: {labels}."
    ),
}

DEFAULT_LID_LABELS = "ar, da, de, el, en, es, et, fi, fr, hr, hu, it, ja, lt, lv, nl, pl, pt, ru, sl, sv, tr, uk, zh"

_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def build_instruction(
    mode: str = "verbatim",
    *,
    source_language: str | None = None,
    target_language: str = "English",
    context_terms: str | None = None,
    lid_labels: str = DEFAULT_LID_LABELS,
) -> str:
    """Build a Nemotron 3.5 Transcribe user prompt without assuming a fixed source language.

    Args:
        mode: Preset key in ``PROMPT_PRESETS`` or ``"contextual"``.
        source_language: Optional explicit language name/code; ``None`` or ``"auto"`` uses
            language-as-spoken auto detection.
        target_language: Target language for ``mode == "translate"``.
        context_terms: Comma-separated biasing terms when ``mode == "contextual"``.
        lid_labels: Accepted language labels for ``mode == "lid"``.
    """
    if mode == "contextual":
        if not context_terms or not context_terms.strip():
            raise ValueError("context_terms is required when mode='contextual'")
        prompt = f"Transcribe with punctuation and capitalization. Context terms: {context_terms.strip()}."
    elif mode == "translate":
        prompt = PROMPT_PRESETS["translate"].format(target_language=target_language.strip() or "English")
    elif mode == "lid":
        prompt = PROMPT_PRESETS["lid"].format(labels=lid_labels)
    elif mode in PROMPT_PRESETS:
        prompt = PROMPT_PRESETS[mode]
    else:
        raise ValueError(f"Unknown prompt mode {mode!r}; expected one of {sorted([*PROMPT_PRESETS, 'contextual'])}")

    lang = (source_language or "").strip()
    if lang and lang.lower() != "auto" and mode not in ("translate", "lid"):
        prompt = f"{prompt} Spoken language: {lang}."
    return prompt


def cleanup_generation_loops(text: str) -> str:
    """Collapse pathological >=21 char runs and >=20 block repeats (model card guardrail)."""
    text = _CHAR_RUN.sub(lambda m: m.group(1), text)
    return _BLOCK_RUN.sub(lambda m: m.group(1), text)
