"""
LLM-based post-processing of raw OCR transcripts via the Google Gemini API.

Requires GEMINI_API_KEY in your .env (or environment).

Install:
    pip install google-genai python-dotenv
"""

from __future__ import annotations
import os
import re
import time
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a world-class expert in early-modern Spanish language, palaeography, and
historical document transcription (15th–19th century). You have deep knowledge of:

  • Old Spanish orthography, including archaic spellings such as 'v' for 'b',
    'x' for 'j'/'sh', 'ss'/'ç' for 's'/'z', and the use of 'u' as 'v'.
  • Latin phrases, abbreviations, and formulaic legal and ecclesiastical language
    commonly embedded in Spanish historical documents.
  • Period-typical vocabulary: legal (fuero, merced, escribano, alguacil),
    religious (convento, obispo, bula), administrative (regidor, corregidor,
    cabildo), and noble titles.
  • Common OCR failure modes in scanned historical manuscripts and printed books:
    confused character pairs (rn/m, cl/d, li/h, 0/o, 1/l/I, 6/b, u/n, f/ſ long-s),
    missing ligatures, broken diacritics, and fragmented tokens.

Your task is to correct OCR errors in the full transcript the user provides.
Reading the transcript as a complete document gives you the contextual knowledge
to make confident, coherent corrections throughout.

CORRECTION RULES
────────────────
1. CHARACTER FIXES  — Correct glyph confusions caused by the OCR scanner:
   • 'rn' → 'm',  '0' → 'o',  '1'/'I' → 'l' or 'I' depending on context,
     'ſ' (long-s) → 's',  'cf' → 'd',  'li' → 'h'.

2. DIACRITICS  — Restore accent marks, tildes and cedillas where Spanish grammar
   and lexicon require them: á, é, í, ó, ú, ñ, ü, ç.
   Use your linguistic knowledge confidently — if the context clearly calls for
   'á' (preposition), 'qué', 'cómo', etc., correct them.

3. WORD RECONSTRUCTION  — If an OCR fragment is clearly an incomplete word
   (broken hyphenation, garbled stem, partially scanned character), reconstruct
   the full intended word using surrounding context.  You MAY draw on your
   knowledge of period vocabulary and the document's subject matter to complete
   partial tokens — prioritise the most plausible historical reading.

4. TOKENISATION  — Merge tokens incorrectly split by the OCR and split
   run-together words only when clearly erroneous.

5. PRESERVE LANGUAGE  — Do NOT translate, paraphrase, modernise, summarise,
   or add ANY content not inferable from the raw text.  This is a transcription
   correction task, not a creative rewrite.

6. NUMBERED LINE FORMAT — The input is given as a numbered list:
     1: <raw line>
     2: <raw line>
     ...
   Your output MUST use the EXACT same format:
     1: <corrected line>
     2: <corrected line>
     ...
   Every input line number must appear in the output.  Never skip, merge, or
   split line numbers.  If a line is already correct, repeat it unchanged.

7. NO EXTRA OUTPUT  — Return only the numbered corrected lines.  No preamble,
   no explanations, no summaries, no blank lines between entries.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file:\n"
            "  GEMINI_API_KEY=AIza..."
        )
    return genai.Client(api_key=api_key)


def _retry_generate(
    client: genai.Client,
    model: str,
    prompt: str,
    config: types.GenerateContentConfig,
    retries: int = 3,
) -> str:
    """Call client.models.generate_content with exponential-backoff retries."""
    delay = 2
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    [Gemini] Attempt {attempt + 1} failed ({exc}), retrying in {delay}s ...")
            time.sleep(delay)
            delay *= 2
    return ""


def _parse_numbered_lines(output: str, expected_len: int) -> List[str]:
    """
    Parse the numbered-line output returned by Gemini back into a plain list.

    Expected format:  "1: text\\n2: text\\n..."
    Falls back to splitting by newlines if numbering is missing.
    """
    numbered: dict[int, str] = {}
    for match in re.finditer(r"^\s*(\d+)\s*[:.)]\s*(.*)", output, re.MULTILINE):
        idx  = int(match.group(1))
        text = match.group(2).strip()
        numbered[idx] = text

    if numbered and max(numbered.keys()) >= expected_len:
        return [numbered.get(i + 1, "") for i in range(expected_len)]

    # Fallback: plain split
    lines = [ln for ln in output.splitlines() if ln.strip()]
    if len(lines) < expected_len:
        lines += [""] * (expected_len - len(lines))
    return lines[:expected_len]


# ──────────────────────────────────────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────────────────────────────────────
def gemini_postprocess_transcript(
    lines: List[str],
    model: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    max_tokens: int = 8192,
) -> List[str]:
    """
    Post-process a list of OCR lines by sending the **full transcript** to Gemini.

    The entire transcript is sent in a single API call so that the model can
    read the document as a whole and use cross-line context to make more
    accurate, coherent corrections (e.g. reconstructing a word that is split
    across lines, inferring document topic, resolving ambiguous abbreviations).

    Lines are submitted as a numbered list and the model returns a numbered
    list, which is then parsed back into a plain list of the same length.

    Parameters
    ----------
    lines       : raw OCR output — one string per detected text line.
    model       : Gemini model ID  (e.g. 'gemini-2.0-flash', 'gemini-1.5-pro').
    temperature : 0.0–1.0 — keep low (≤ 0.2) for deterministic corrections.
    max_tokens  : max output tokens for the API call.

    Returns
    -------
    list[str]  — corrected lines, same length as input.
    """
    if not lines:
        return []

    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    # Build numbered transcript
    numbered_input = "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))

    user_prompt = (
        f"Below is the full OCR transcript of an early-modern Spanish document "
        f"({len(lines)} lines).  Read it as a complete text, then return the "
        f"corrected version in the EXACT same numbered format.\n\n"
        "--- RAW OCR TRANSCRIPT ---\n"
        f"{numbered_input}\n"
        "--- END OF TRANSCRIPT ---\n\n"
        f"Return all {len(lines)} corrected lines numbered 1 through {len(lines)}."
    )

    print(f"    Sending full transcript ({len(lines)} lines) to Gemini ({model}) ...")
    raw_output = _retry_generate(client, model, user_prompt, config)
    corrected  = _parse_numbered_lines(raw_output, len(lines))
    changed    = sum(1 for r, c in zip(lines, corrected) if r.strip() != c.strip())
    print(f"    Done — {changed} / {len(lines)} lines corrected.")
    return corrected


# ──────────────────────────────────────────────────────────────────────────────
# Debug utility
# ──────────────────────────────────────────────────────────────────────────────
def print_diff(
    raw_lines: List[str],
    corrected_lines: List[str],
    title: str = "",
) -> None:
    """Print a side-by-side comparison of raw vs. corrected lines."""
    if title:
        print(f"\n{'='*70}\n  {title}\n{'='*70}")

    col = 68
    print(f"{'RAW':<{col}}  {'CORRECTED':<{col}}")
    print("-" * (col * 2 + 2))

    changed = 0
    for r, c in zip(raw_lines, corrected_lines):
        marker = " " if r.strip() == c.strip() else "✎"
        if r.strip() != c.strip():
            changed += 1
        print(f"{r:<{col}}  {c:<{col}}  {marker}")

    print(f"\n{changed}/{len(raw_lines)} lines changed by LLM post-processing.")