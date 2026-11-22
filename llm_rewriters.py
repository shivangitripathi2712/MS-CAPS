"""
API-based caption rewriters (GPT-4o and Gemini) for the compressor ablation.

They use exactly the same prompt, JSON parsing, and de-duplication as the
default Qwen2.5-VL rewriter in src/compressor.py, so only the model changes.
"""

import json
import os
import re
from typing import List, Optional

from src.compressor import CaptionRewriter, clean_caption_spacing, count_tokens


# ---------------------------------------------------------------------------
# Shared prompt + parsing (verbatim copies of QwenVLRewriter's pure methods, so
# the API models get the IDENTICAL prompt and identical candidate parsing).
# Kept here as standalone functions to avoid constructing a GPU-backed rewriter.
# ---------------------------------------------------------------------------

def build_rewrite_prompt(original_caption, strict_anchors, soft_anchors,
                         target_token_budgets, num_candidates) -> str:
    original_tokens = count_tokens(original_caption)
    strict_block = "\n".join(f'  ({i}) "{a}"' for i, a in enumerate(strict_anchors, 1)) or "  (none)"
    soft_block = "\n".join(f"  ({i}) {a}" for i, a in enumerate(soft_anchors, 1)) or "  (none)"
    budgets = list(target_token_budgets)
    min_budget = min(budgets) if budgets else max(8, original_tokens // 2)
    budget_lines = "\n".join(f"  - Variant {i+1}: about {b} tokens"
                             for i, b in enumerate(budgets)) or f"  - about {min_budget} tokens"
    return f"""You compress an image caption while preserving the visual information needed to faithfully reconstruct the image. Removing a visually important detail makes the reconstructed image look different from the original - avoid that.

ORIGINAL CAPTION ({original_tokens} tokens):
{original_caption}

REQUIREMENT 1 - KEEP THESE EXACT PHRASES (verbatim, unchanged):
These name distinctive or text-bearing elements (signs, plates, brands, framing
devices). Every variant MUST contain each of them word-for-word.
{strict_block}

REQUIREMENT 2 - PRESERVE THESE FACTS (you may rephrase, but the meaning must remain):
These are important visual facts. Every variant must still convey each one.
{soft_block}

REQUIREMENT 2b - ALWAYS KEEP VIEWPOINT, ORIENTATION, AND DISTINCTIVE MARKINGS:
These few words cost almost nothing but decide whether a reconstruction looks like
the original. Whenever the original caption states any of them, you MUST keep them:
  - Camera viewpoint: "side view", "front view", "rear view", "from above",
    "elevated perspective", "close-up", "wide shot".
  - Object orientation / pose: "facing right", "facing left", "parked diagonally",
    "side and rear visible", "front-facing", "angled".
  - Distinctive colors and markings: stripes, body color, painted text bands,
    decorative patterns (e.g. "blue stripes", "green roof", "red trim").
  - Spatial layout: foreground / midground / background placement and left/right
    position of the main objects.
Dropping a viewpoint or a distinctive marking counts as losing a key visual fact -
do not drop them to save length.

REQUIREMENT 3 - GRAMMATICAL SENTENCES:
Each variant is one or more complete, fluent English sentences. Never a list of
keywords, never a fragment, never comma-separated noun phrases.

REQUIREMENT 4 - SHORTER THAN THE ORIGINAL:
Every variant must be clearly shorter than {original_tokens} tokens. Produce
{num_candidates} variants of increasing length, shortest first:
{budget_lines}

Drop only redundant or non-visual wording. When in doubt about a distinctive or
spatial detail, keep it.

OUTPUT - return ONLY a JSON array of strings, shortest first. No markdown, no
numbering, no commentary:
["shortest variant", "...", "longest variant"]
"""


def parse_json_array(text: str) -> List[str]:
    """Identical to QwenVLRewriter._parse_json_array."""
    if not isinstance(text, str):
        return []
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    for candidate in (text, match.group(0) if match else None):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
        except Exception:
            continue
    return [s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text) if s.strip()]


def _dedup(cands: List[str]) -> List[str]:
    """Identical dedup to QwenVLRewriter.rewrite."""
    cands = [clean_caption_spacing(c) for c in cands if c]
    seen, unique = set(), []
    for c in cands:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# GPT-4o rewriter
# ---------------------------------------------------------------------------

class GPT4oRewriter(CaptionRewriter):
    """Caption compressor backend using the OpenAI API (text-only generation)."""

    def __init__(self, model: str = "gpt-4o-2024-08-06",
                 temperature: float = 0.7, max_tokens: int = 700):
        from openai import OpenAI
        self.client = OpenAI()  # OPENAI_API_KEY from env
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def rewrite(self, original_caption, strict_anchors, soft_anchors,
                target_token_budgets, num_candidates, seed: Optional[int] = None) -> List[str]:
        prompt = build_rewrite_prompt(original_caption, strict_anchors, soft_anchors,
                                      target_token_budgets, num_candidates)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=int(seed) if seed is not None else None,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"[GPT4oRewriter] generation failed: {exc}")
            return []
        return _dedup(parse_json_array(raw))


# ---------------------------------------------------------------------------
# Gemini rewriter
# ---------------------------------------------------------------------------

class GeminiRewriter(CaptionRewriter):
    """Caption compressor backend using the Google Gemini API (text-only generation)."""

    def __init__(self, model: str = "gemini-2.5-pro",
                 temperature: float = 0.7, max_tokens: int = 700):
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for GeminiRewriter.")
        genai.configure(api_key=key)
        self._genai = genai
        self.model_name = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.model = genai.GenerativeModel(model)

    def rewrite(self, original_caption, strict_anchors, soft_anchors,
                target_token_budgets, num_candidates, seed: Optional[int] = None) -> List[str]:
        prompt = build_rewrite_prompt(original_caption, strict_anchors, soft_anchors,
                                      target_token_budgets, num_candidates)
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=self._genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            raw = resp.text or ""
        except Exception as exc:
            print(f"[GeminiRewriter] generation failed: {exc}")
            return []
        return _dedup(parse_json_array(raw))