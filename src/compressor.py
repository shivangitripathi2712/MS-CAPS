"""
MS-CAPS compressor (KG-guided, reconstruction-based rate-distortion).

For each (image I, caption c):
  1. Extract the reference knowledge graph KG_I from I with a VLM.
  2. Reconstruct G(c) from the full caption to get the baseline distortion D_0
     and the distortion budget delta = (1 + epsilon) * D_0 (optionally capped by absolute_threshold).
     Samples whose full caption already reconstructs poorly are skipped.
  3. Derive strict anchors (verbatim phrases, e.g. visible text) and soft anchors
     (salient KG facts) that every compressed caption must keep.
  4. Ask the rewriter for candidates over a ladder of token budgets.
  5. Reconstruct each candidate, score Q_img / Q_cap / Q_KG and D, and apply the
     gate: C1 Q_img >= tau_img, C2 Q_cap >= tau_cap, C3 Q_KG >= tau_kg, C4 D <= delta.
  6. Keep the shortest passing caption (within quality_margin of the best
     passer's quality). If none pass, optionally try gentler budgets
     (escalation_ratios), then apply the fallback strategy: keep the original
     caption (default, as in the paper) or the lowest-D shorter candidate.
     Repeat on the winner up to max_iterations times.
"""
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from .distortion import DistortionMetric
from .parse_tree import extract_strict_anchors, _shared_pruner
from .scene_graph import (
    VLMSceneGraphExtractor,
    canonicalize_for_matching,
    compare_scene_graphs,
    derive_anchors_from_missing,
    derive_soft_anchors_from_kg,
    normalize_label,
    semantic_text_similarity,
    tuple_to_text,
)
from .t2i_reconstruct import T2IReconstructor, build_reconstructor


# Default budget ladders (fractions of ORIGINAL caption tokens).
_PRIMARY_RATIOS = [0.20, 0.30, 0.45, 0.55, 0.65]
_ESCALATION_RATIOS = [0.70, 0.78, 0.86, 0.94]


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class RewriteCandidate:
    index: int
    text: str
    target_token_budget: int
    token_count: int
    missing_anchors: List[str] = field(default_factory=list)
    passed_anchor_filter: bool = False


@dataclass
class CompressionStep:
    iteration: int
    stage: str
    candidate_index: int
    candidate_caption: str
    target_token_budget: int
    caption_before: str
    tokens_before: int
    tokens_after: int
    token_reduction: int
    distortion: float
    threshold: float
    passed_threshold: bool
    selected: bool
    reason: str
    gate_result: Dict[str, Any]
    distortion_details: Dict[str, Any]
    missing_anchors: List[str]
    time_seconds: float


@dataclass
class CompressionResult:
    image_id: str
    status: str
    original_caption: str
    compressed_caption: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    baseline_distortion: float
    threshold: float
    final_distortion: float
    final_distortion_details: Dict[str, Any]
    baseline_kg_comparison: Dict[str, Any]
    final_kg_comparison: Dict[str, Any]
    strict_anchors: List[str]
    soft_anchors: List[str]
    steps: List[CompressionStep]
    total_time: float
    final_reconstruction_path: Optional[str] = None
    baseline_reconstruction_path: Optional[str] = None
    original_image_path: Optional[str] = None
    kg_I_counts: Dict[str, int] = field(default_factory=dict)
    kg_final_counts: Dict[str, int] = field(default_factory=dict)
    # Backward-compatible aliases (older readers expect these names):
    protected_anchors: List[str] = field(default_factory=list)   # == soft_anchors
    quoted_anchors: List[str] = field(default_factory=list)       # == strict_anchors


# =====================================================================
# Helpers
# =====================================================================

def count_tokens(text: str) -> int:
    return len(text.split())


def clean_caption_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text.strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", str(name))[:180]


def _dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


def make_json_serializable(obj: Any) -> Any:
    if _dataclass_instance(obj):
        return make_json_serializable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, set):
        return [make_json_serializable(x) for x in sorted(obj, key=str)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_id": kg.get("image_id"),
        "objects": kg.get("objects", []),
        "attributes": kg.get("attributes", []),
        "relations": kg.get("relations", []),
        "scene_context": kg.get("scene_context", {}),
        "tuples": [list(t) for t in sorted(list(kg.get("tuples", set())), key=str)],
        "raw_vlm_output": kg.get("raw_vlm_output", ""),
    }


def _token_set(s: str) -> Set[str]:
    return {w.lower().strip(".,;:!?\"'()[]{}") for w in s.split() if w.strip()}


def is_substantive_rewrite(original: str, candidate: str, min_diff_ratio: float = 0.15) -> bool:
    a, b = _token_set(original), _token_set(candidate)
    if not a or not b:
        return True
    union, inter = a | b, a & b
    return (1.0 - len(inter) / len(union)) >= min_diff_ratio if union else True


def _looks_like_sentence(text: str) -> bool:
    """Reject bag-of-words: require at least one verb/aux (via shared spaCy)."""
    nlp = _shared_pruner().nlp
    if nlp is None:
        return True  # cannot check; do not reject
    doc = nlp(text)
    return any(t.pos_ in ("VERB", "AUX") for t in doc)


_QUOTE_RE = r'"([^"]+)"|\u201c([^\u201c\u201d]+)\u201d|`([^`]+)`'


def is_text_bearing_anchor(anchor: str, visible_text_set: Optional[Set[str]] = None) -> bool:
    """
    True only for DISTINCTIVE, text-bearing anchors that a compressed caption must
    keep VERBATIM because they cannot be recovered semantically: detected visible
    text (plates/brands/signage), alphanumerics (license plates), and ALL-CAPS
    brand tokens. Descriptive phrases ("green traffic light", "elevated
    perspective") are NOT text-bearing - they belong in soft anchors, where their
    preservation is enforced via reconstruction-side KG recall (Q_KG / R_obj),
    not literal string matching.
    """
    a = (anchor or "").strip()
    if not a:
        return False
    al = a.lower()
    vt = visible_text_set or set()
    # matches a detected visible-text string (plate / brand / slogan / sign)
    for t in vt:
        t = (t or "").strip().lower()
        if t and (t in al or al in t):
            return True
    # contains a digit -> license plate / alphanumeric code
    if any(ch.isdigit() for ch in a):
        return True
    # contains an ALL-CAPS alphabetic token of length >= 2 -> brand / sign
    for tok in a.split():
        t = tok.strip(".,;:!?'\"()[]{}")
        if len(t) >= 2 and t.isalpha() and t.isupper():
            return True
    return False


def _extract_quoted_substrings(text: str) -> List[str]:
    out = []
    for m in re.finditer(_QUOTE_RE, text or ""):
        s = next((g for g in m.groups() if g), "").strip()
        if s:
            out.append(s)
    return out


def candidate_preserves_anchors(
    candidate: str,
    strict_anchors: Sequence[str],
    soft_anchors: Sequence[str],
    kg_match_threshold: float = 0.55,
    max_soft_miss_count: int = 0,
    max_strict_miss_count: int = 0,
) -> Tuple[bool, List[str]]:
    """
    Strict anchors should be present (substring / verbatim / token-overlap); up to
    ``max_strict_miss_count`` may be missing (graduated tolerance - a heavily
    compressed caption cannot carry every plate/sign/phrase). Soft anchors should
    be present semantically; up to ``max_soft_miss_count`` may be missing.
    Returns (passed, missing_anchor_list).
    """
    if not candidate:
        return False, list(strict_anchors) + list(soft_anchors)

    cand_low = candidate.lower()
    cand_norm = canonicalize_for_matching(candidate)
    cand_norm_words = set(cand_norm.split())

    strict_missing: List[str] = []
    for a in strict_anchors:
        a_low = a.strip().lower()
        if not a_low:
            continue
        if a_low in cand_low:
            continue
        # token-overlap leniency for multi-word strict anchors
        a_norm = canonicalize_for_matching(a)
        a_words = [w for w in a_norm.split() if len(w) > 2]
        if a_words and all(w in cand_norm_words for w in a_words):
            continue
        strict_missing.append(a)

    soft_missing: List[str] = []
    for a in soft_anchors:
        a_norm = canonicalize_for_matching(a)
        if not a_norm:
            continue
        if a_norm in cand_norm:
            continue
        a_words = [w for w in a_norm.split() if len(w) > 2]
        if a_words and any(w in cand_norm_words for w in a_words):
            continue
        if semantic_text_similarity(a_norm, cand_norm) >= kg_match_threshold:
            continue
        soft_missing.append(a)

    if len(strict_missing) > max_strict_miss_count:
        return False, strict_missing + soft_missing
    if len(soft_missing) > max_soft_miss_count:
        return False, strict_missing + soft_missing
    return True, strict_missing + soft_missing


def save_side_by_side(left_img: Image.Image, right_img: Image.Image,
                      left_label: str, right_label: str, path: Path,
                      caption: str = "") -> None:
    left = left_img.convert("RGB").resize((512, 512))
    right = right_img.convert("RGB").resize((512, 512))
    canvas = Image.new("RGB", (1024, 580), color="white")
    canvas.paste(left, (0, 48))
    canvas.paste(right, (512, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((170, 15), left_label, fill="black")
    draw.text((565, 15), right_label, fill="black")
    if caption:
        preview = caption[:100] + "..." if len(caption) > 100 else caption
        draw.text((10, 532), preview, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# =====================================================================
# Rewriter
# =====================================================================

class CaptionRewriter(ABC):
    @abstractmethod
    def rewrite(self, original_caption: str, strict_anchors: Sequence[str],
                soft_anchors: Sequence[str], target_token_budgets: Sequence[int],
                num_candidates: int, seed: Optional[int] = None) -> List[str]:
        ...


class QwenVLRewriter(CaptionRewriter):
    """Text-only caption rewriter sharing the Qwen2.5-VL model (no extra GPU load)."""

    def __init__(self, vlm_extractor: VLMSceneGraphExtractor,
                 max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
        self.model = vlm_extractor.model
        self.processor = vlm_extractor.processor
        self.device = vlm_extractor.device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def _build_prompt(self, original_caption, strict_anchors, soft_anchors,
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

    def _parse_json_array(self, text: str) -> List[str]:
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
        # Last resort: pull quoted strings.
        return [s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text) if s.strip()]

    @torch.no_grad()
    def _generate_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = self.processor(text=[text_input], padding=True, return_tensors="pt")
        except Exception:
            inputs = self.processor.tokenizer(text_input, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device) if hasattr(inputs, "to") else {k: v.to(self.model.device) for k, v in inputs.items()}
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens,
            do_sample=True, temperature=self.temperature, top_p=self.top_p,
        )
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]
        decoder = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
        return decoder(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def rewrite(self, original_caption, strict_anchors, soft_anchors,
                target_token_budgets, num_candidates, seed: Optional[int] = None) -> List[str]:
        if seed is not None:
            torch.manual_seed(int(seed))
        prompt = self._build_prompt(original_caption, strict_anchors, soft_anchors,
                                    target_token_budgets, num_candidates)
        try:
            raw = self._generate_text(prompt)
        except Exception as exc:
            print(f"[Rewriter] Generation failed: {exc}")
            return []
        cands = [clean_caption_spacing(c) for c in self._parse_json_array(raw) if c]
        seen, unique = set(), []
        for c in cands:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)
        return unique


# =====================================================================
# Main compressor
# =====================================================================

class KGRDCompressor:
    """KG-guided Rate-Distortion caption compressor (see module docstring)."""

    def __init__(
        self,
        scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        scene_graph_device: str = "auto",
        scene_graph_dtype: str = "float16",
        scene_graph_max_new_tokens: int = 1024,
        scene_graph_temperature: float = 0.0,
        scene_graph_do_sample: bool = False,
        max_objects: int = 50,
        max_relations: int = 80,
        t2i_model: str = "black-forest-labs/FLUX.1-dev",
        t2i_device: str = "auto",
        t2i_dtype: str = "bfloat16",
        t2i_steps: int = 20,
        t2i_guidance: float = 3.5,
        t2i_image_size: int = 512,
        t2i_seeds: Optional[List[int]] = None,
        alpha: float = 0.4, beta: float = 0.2, gamma: float = 0.4,
        clip_model: str = "ViT-L-14", clip_pretrained: str = "openai",
        dino_model: str = "dinov2_vitb14", lpips_net: str = "alex",
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        metric_device: str = "auto",
        kg_object_weight: float = 0.4, kg_attribute_weight: float = 0.3, kg_relation_weight: float = 0.3,
        use_soft_kg: bool = True,
        tau_img: float = 0.50, tau_cap: float = 0.55, tau_kg: float = 0.45,
        use_relative_threshold: bool = True, epsilon: float = 0.10, absolute_threshold: Optional[float] = 0.25,
        min_caption_tokens: int = 5,
        min_target_ratio: float = 0.0,
        min_target_tokens: int = 0,
        max_iterations: int = 2,
        num_candidates_per_iteration: int = 5,
        budget_ratios: Optional[List[float]] = None,
        escalation_ratios: Optional[List[float]] = None,
        require_strict_compression: bool = True,
        min_baseline_kg_f1: float = 0.05, min_baseline_object_recall: float = 0.10,
        kg_f1_floor_drop: float = 0.08, object_recall_floor_drop: float = 0.05,
        attribute_recall_floor_drop: float = 0.12, attribute_recall_min_floor: float = 0.10,
        max_strict_anchors: int = 8, max_soft_anchors: int = 6,
        kg_match_threshold: float = 0.55, soft_anchor_max_miss_fraction: float = 0.40,
        min_rewrite_diff_ratio: float = 0.15,
        select_by_quality: bool = True, quality_metric: str = "Q_img", quality_margin: float = 0.05,
        fallback_strategy: str = "best_candidate",
        rewriter: Optional[CaptionRewriter] = None,
        rewriter_max_new_tokens: int = 512, rewriter_temperature: float = 0.7, rewriter_top_p: float = 0.9,
        save_intermediate: bool = True, output_dir: str = "./outputs", seed: int = 42,
        t2i_reconstructor=None,
    ):
        self.t2i_seeds = t2i_seeds or [42]
        self.use_relative_threshold = bool(use_relative_threshold)
        self.epsilon = float(epsilon)
        self.absolute_threshold = None if absolute_threshold is None else float(absolute_threshold)
        self.use_soft_kg = bool(use_soft_kg)
        self.tau_img, self.tau_cap, self.tau_kg = float(tau_img), float(tau_cap), float(tau_kg)
        self.min_caption_tokens = int(min_caption_tokens)
        self.min_target_ratio = float(min_target_ratio)
        self.min_target_tokens = int(min_target_tokens)
        self.max_iterations = int(max_iterations)
        self.num_candidates_per_iteration = int(num_candidates_per_iteration)
        self.budget_ratios = list(budget_ratios) if budget_ratios else list(_PRIMARY_RATIOS)
        self.escalation_ratios = list(escalation_ratios) if escalation_ratios else list(_ESCALATION_RATIOS)
        self.require_strict_compression = bool(require_strict_compression)
        self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
        self.min_baseline_object_recall = float(min_baseline_object_recall)
        self.kg_f1_floor_drop = float(kg_f1_floor_drop)
        self.object_recall_floor_drop = float(object_recall_floor_drop)
        self.attribute_recall_floor_drop = float(attribute_recall_floor_drop)
        self.attribute_recall_min_floor = float(attribute_recall_min_floor)
        self.max_strict_anchors = int(max_strict_anchors)
        self.max_soft_anchors = int(max_soft_anchors)
        self.kg_match_threshold = float(kg_match_threshold)
        self.soft_anchor_max_miss_fraction = float(soft_anchor_max_miss_fraction)
        self.min_rewrite_diff_ratio = float(min_rewrite_diff_ratio)
        self.select_by_quality = bool(select_by_quality)
        self.quality_metric = str(quality_metric)
        self.quality_margin = float(quality_margin)
        if fallback_strategy not in {"keep_original", "best_candidate"}:
            raise ValueError(f"Invalid fallback_strategy: {fallback_strategy}")
        self.fallback_strategy = fallback_strategy
        self.save_intermediate = bool(save_intermediate)
        self.seed = int(seed)

        self.output_dir = Path(output_dir)
        self.reconstruction_dir = self.output_dir / "reconstructions"
        self.result_dir = self.output_dir / "results"
        self.kg_debug_dir = self.output_dir / "kg_debug"
        self.candidate_debug_dir = self.output_dir / "candidates"
        self.debug_dir = self.output_dir / "debug_samples"
        for d in (self.result_dir, self.kg_debug_dir, self.candidate_debug_dir, self.debug_dir):
            d.mkdir(parents=True, exist_ok=True)

        print("=" * 70 + "\nKG-RD Compressor: initializing\n" + "=" * 70)
        print("\n[1/3] Loading VLM scene-graph extractor...")
        self.sg_extractor = VLMSceneGraphExtractor(
            model_name=scene_graph_model, device=scene_graph_device, dtype=scene_graph_dtype,
            max_new_tokens=scene_graph_max_new_tokens, temperature=scene_graph_temperature,
            do_sample=scene_graph_do_sample, max_objects=max_objects, max_relations=max_relations,
        )
        print("\n[2/3] Loading T2I reconstructor...")
        if t2i_reconstructor is not None:
            self.t2i = t2i_reconstructor
        else:
            self.t2i = T2IReconstructor(
                model_name=t2i_model, device=t2i_device, dtype=t2i_dtype,
                num_inference_steps=t2i_steps, guidance_scale=t2i_guidance,
                image_size=t2i_image_size, img2img_mode=False,
            )
        print("\n[3/3] Loading distortion metric (CLIP + DINO + LPIPS + BERTScore)...")
        self.distortion = DistortionMetric(
            alpha=alpha, beta=beta, gamma=gamma,
            clip_model_name=clip_model, clip_pretrained=clip_pretrained,
            dino_model_name=dino_model, lpips_net=lpips_net,
            bertscore_model=bertscore_model, device=metric_device,
            kg_object_weight=kg_object_weight, kg_attribute_weight=kg_attribute_weight,
            kg_relation_weight=kg_relation_weight,
        )
        self.rewriter = rewriter or QwenVLRewriter(
            vlm_extractor=self.sg_extractor, max_new_tokens=rewriter_max_new_tokens,
            temperature=rewriter_temperature, top_p=rewriter_top_p,
        )
        print("\n" + "=" * 70)
        print(f"Ready. taus: img={self.tau_img} cap={self.tau_cap} kg={self.tau_kg} | "
              f"eps={self.epsilon} abs<= {self.absolute_threshold} | use_soft_kg={self.use_soft_kg}")
        print(f"Budget ladder: {self.budget_ratios}  escalation: {self.escalation_ratios}")
        print(f"Length floor: min_target_ratio={self.min_target_ratio} "
              f"min_target_tokens={self.min_target_tokens} (min_caption_tokens={self.min_caption_tokens})")
        print(f"candidates/iter={self.num_candidates_per_iteration} max_iters={self.max_iterations} "
              f"fallback={self.fallback_strategy}")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Dict[str, Any], rewriter: Optional[CaptionRewriter] = None) -> "KGRDCompressor":
        sg = config["scene_graph"]["vlm"]
        t = config["t2i"]
        d = config["distortion"]; w = d["weights"]; kg = d.get("kg", {})
        g = config["gate"]; comp = config["compression"]; rw = config.get("rewriter", {})
        bg = config.get("baseline_gate", {}); ff = config.get("fidelity_floors", {})
        anc = config.get("anchors", {}); fb = config.get("fallback", {})
        paths = config.get("paths", {}); ev = config.get("evaluation", {})
        # Text-only vs structure-conditioned (ControlNet) selected by config.
        reconstructor = build_reconstructor(config)
        return cls(
            scene_graph_model=sg["model"], scene_graph_device=sg["device"], scene_graph_dtype=sg["dtype"],
            scene_graph_max_new_tokens=sg.get("max_new_tokens", 1024),
            scene_graph_temperature=sg.get("temperature", 0.0), scene_graph_do_sample=sg.get("do_sample", False),
            max_objects=config["scene_graph"].get("max_objects", 50),
            max_relations=config["scene_graph"].get("max_relations", 80),
            t2i_model=t["model"], t2i_device=t["device"], t2i_dtype=t["dtype"],
            t2i_steps=t.get("num_inference_steps", 20), t2i_guidance=t.get("guidance_scale", 3.5),
            t2i_image_size=t.get("image_size", 512), t2i_seeds=t.get("seeds", [42]),
            alpha=w["alpha_img"], beta=w["beta_cap"], gamma=w["gamma_kg"],
            clip_model=d.get("clip", {}).get("model", "ViT-L-14"),
            clip_pretrained=d.get("clip", {}).get("pretrained", "openai"),
            dino_model=d.get("dino", {}).get("model", "dinov2_vitb14"),
            lpips_net=d.get("lpips", {}).get("net", "alex"),
            bertscore_model=d.get("bertscore", {}).get("model", "microsoft/deberta-xlarge-mnli"),
            metric_device=d.get("clip", {}).get("device", "cuda:0"),
            kg_object_weight=kg.get("object_weight", 0.4), kg_attribute_weight=kg.get("attribute_weight", 0.3),
            kg_relation_weight=kg.get("relation_weight", 0.3),
            use_soft_kg=d.get("use_soft_kg", True),
            tau_img=g["tau_img"], tau_cap=g["tau_cap"], tau_kg=g["tau_kg"],
            use_relative_threshold=d.get("use_relative_threshold", True),
            epsilon=d.get("epsilon", 0.10), absolute_threshold=d.get("absolute_threshold", 0.25),
            min_caption_tokens=comp.get("min_caption_tokens", 5),
            min_target_ratio=comp.get("min_target_ratio", 0.0),
            min_target_tokens=comp.get("min_target_tokens", 0),
            max_iterations=comp.get("max_iterations", 2),
            num_candidates_per_iteration=comp.get("num_candidates_per_iteration", 5),
            budget_ratios=comp.get("budget_ratios", _PRIMARY_RATIOS),
            escalation_ratios=comp.get("escalation_ratios", _ESCALATION_RATIOS),
            require_strict_compression=comp.get("require_strict_compression", True),
            min_baseline_kg_f1=bg.get("min_baseline_kg_f1", 0.05),
            min_baseline_object_recall=bg.get("min_baseline_object_recall", 0.10),
            kg_f1_floor_drop=ff.get("kg_f1_floor_drop", 0.08),
            object_recall_floor_drop=ff.get("object_recall_floor_drop", 0.05),
            attribute_recall_floor_drop=ff.get("attribute_recall_floor_drop", 0.12),
            attribute_recall_min_floor=ff.get("attribute_recall_min_floor", 0.10),
            max_soft_anchors=anc.get("max_kg_anchors", 6),
            kg_match_threshold=anc.get("kg_match_threshold", 0.55),
            soft_anchor_max_miss_fraction=anc.get("soft_anchor_max_miss_fraction", 0.40),
            select_by_quality=comp.get("select_by_quality", True),
            quality_metric=comp.get("quality_metric", "Q_img"),
            quality_margin=comp.get("quality_margin", 0.05),
            fallback_strategy=fb.get("strategy", "best_candidate"),
            rewriter=rewriter,
            rewriter_max_new_tokens=rw.get("max_new_tokens", 512),
            rewriter_temperature=rw.get("temperature", 0.7), rewriter_top_p=rw.get("top_p", 0.9),
            save_intermediate=ev.get("save_intermediate", True),
            output_dir=paths.get("output_dir", "./outputs"),
            seed=config.get("seed", 42),
            t2i_reconstructor=reconstructor,
        )

    # ================================================================== #
    # Public API
    # ================================================================== #
    def compress_single(self, image: Image.Image, caption: str, image_id: str = "unknown") -> CompressionResult:
        start = time.time()
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image.")
        image = image.convert("RGB") if image.mode != "RGB" else image
        caption = clean_caption_spacing(caption)
        original_tokens = count_tokens(caption)
        if original_tokens <= self.min_caption_tokens:
            raise ValueError(f"Caption has only {original_tokens} tokens (min {self.min_caption_tokens}).")
        sid = safe_filename(image_id)
        steps: List[CompressionStep] = []

        print("\n" + "-" * 70)
        print(f"Compressing image_id={image_id} | tokens={original_tokens}")
        print("-" * 70)

        # 1. KG_I + visible text
        print("[1] Extracting KG_I from original image...")
        # Structure-conditioned reconstructor: cache the control map from I ONCE,
        # so baseline and every candidate share the same layout (no-op for text-only).
        if hasattr(self.t2i, "set_structure_reference"):
            self.t2i.set_structure_reference(image)
        kg_original = self.sg_extractor.extract_with_relation_retry(image, image_id=image_id)
        visible_text = self.sg_extractor.extract_visible_text(image)
        print(f"    KG_I: {len(kg_original.get('objects', []))} obj / "
              f"{len(kg_original.get('attributes', []))} attr / {len(kg_original.get('relations', []))} rel; "
              f"visible_text={[v['text'] for v in visible_text]}")
        if self.save_intermediate:
            self._save_kg(kg_original, sid, "kg_original_image")
        kg_I_counts = self._kg_counts(kg_original)
        # Save the original image for the Excel report.
        original_image_path = None
        if self.save_intermediate:
            (self.debug_dir / sid).mkdir(parents=True, exist_ok=True)
            original_image_path = self.debug_dir / sid / "original_image.png"
            image.save(original_image_path)

        # 2. Baseline reconstruction + D_0 + threshold
        print("[2] Baseline reconstruction G(c)...")
        baseline_images, baseline_kgs = self._reconstruct_and_extract(caption, sid, "baseline")
        baseline_recon_path = (self.reconstruction_dir / sid / f"baseline_seed{self.t2i_seeds[0]}.png")
        baseline_recon_path = baseline_recon_path if baseline_recon_path.exists() else None
        baseline_dist = self.distortion.compute_averaged_distortion(
            image_orig=image, images_recon=baseline_images,
            kg_original=kg_original, kgs_reconstructed=baseline_kgs,
        )
        baseline_D = float(baseline_dist.get("D", 1.0))
        threshold = (self.distortion.compute_relative_threshold(
            baseline_D, self.epsilon, self.absolute_threshold)
            if self.use_relative_threshold else self.absolute_threshold)
        baseline_cmp = self._aggregate_comparisons(kg_original, baseline_kgs)
        print(f"    D_0={baseline_D:.4f}  delta={threshold:.4f}  "
              f"(baseline f1={baseline_cmp['f1_mean']:.3f} obj_rec={baseline_cmp['object_recall_mean']:.3f})")
        if self.save_intermediate and baseline_images:
            save_side_by_side(image, baseline_images[0], "Original (I)", "Baseline G(c)",
                              self.debug_dir / sid / "side_by_side_original_vs_baseline.png", caption)
            self._save_comparison(baseline_cmp, sid, "baseline_kg_comparison")

        # 3. Baseline gate
        if (baseline_cmp["f1_mean"] < self.min_baseline_kg_f1
                or baseline_cmp["object_recall_mean"] < self.min_baseline_object_recall):
            print("[SKIP] Baseline fidelity below threshold.")
            return self._skipped_result(image_id, caption, original_tokens, baseline_D,
                                        threshold, baseline_dist, baseline_cmp, sid, start)

        # 4. Anchors
        strict_anchors = extract_strict_anchors(
            caption, kg=kg_original, visible_text=visible_text,
            kg_match_threshold=self.kg_match_threshold, max_anchors=self.max_strict_anchors,
        )
        soft_anchors = self._build_soft_anchors(kg_original, baseline_cmp)
        # Strict = VERBATIM-required text only (plates/brands/signage). Descriptive
        # phrases ("green traffic light", "elevated perspective") are NOT verbatim-
        # recoverable obligations; demote them to soft, where the rewriter is told
        # to keep them and the reconstruction-side KG recall (Q_KG / R_obj floor)
        # actually enforces important objects/relations. This keeps "strict" small
        # and satisfiable so faithful candidates are not rejected wholesale.
        vt_set = {v.get("text", "") for v in (visible_text or [])}
        distinctive = [a for a in strict_anchors if is_text_bearing_anchor(a, vt_set)]
        demoted = [a for a in strict_anchors if a not in distinctive]
        if demoted:
            seen, merged = set(), []
            for a in demoted + list(soft_anchors):
                if a and a.lower() not in seen:
                    seen.add(a.lower())
                    merged.append(a)
            soft_anchors = merged[: self.max_soft_anchors + 4]
        # Sanitize strict anchors: strip stray quotes/labels like "brand name 'FORCE"
        # that the phrase extractor occasionally emits, so the verbatim-match filter
        # isn't handed an unsatisfiable string.
        _clean_strict = []
        for a in distinctive:
            s = re.sub(r"^(brand name|sign|text|label|plate)\s*[:'\"]*\s*", "", a.strip(), flags=re.IGNORECASE)
            s = s.strip().strip("'\"`").strip()
            if s:
                _clean_strict.append(s)
        strict_anchors = _clean_strict
        print(f"[3] Strict anchors S_c: {strict_anchors}")
        print(f"    Soft  anchors G_c: {soft_anchors}")

        # 5/6. Outer loop: find shortest faithful, then compress further
        current_caption = caption
        winner: Optional[Dict[str, Any]] = None
        global_best_below: Optional[Dict[str, Any]] = None

        for iteration in range(1, self.max_iterations + 1):
            current_tokens = count_tokens(current_caption)
            if current_tokens <= self.min_caption_tokens:
                break
            print(f"\n[Iter {iteration}] from {current_tokens} tokens")
            iw, best_below, iter_steps = self._find_shortest_faithful(
                image=image, original_caption=caption, kg_original=kg_original,
                source_caption=current_caption, original_tokens=original_tokens,
                threshold=threshold, baseline_cmp=baseline_cmp,
                strict_anchors=strict_anchors, soft_anchors=soft_anchors,
                iteration=iteration, sid=sid,
            )
            steps.extend(iter_steps)
            if best_below and (global_best_below is None
                               or best_below["scores"]["D"] < global_best_below["scores"]["D"]):
                global_best_below = best_below
            if iw is None:
                print(f"[Iter {iteration}] no faithful candidate.")
                break
            if self.require_strict_compression and iw["tokens"] >= current_tokens:
                print(f"[Iter {iteration}] winner not shorter; stopping.")
                break
            winner = iw
            current_caption = iw["caption"]
            # re-inject the winner's own missing facts for the next iteration
            soft_anchors = self._build_soft_anchors(kg_original, iw["comparison"])
            print(f"[Iter {iteration}] winner: {iw['tokens']}t  D={iw['scores']['D']:.4f}")

        # 7. Resolve final: shortest passing candidate; otherwise follow the
        #    fallback strategy ("keep_original" returns c, "best_candidate"
        #    returns the closest shorter candidate).
        if winner is not None:
            final = winner
            status = "compressed"
        elif self.fallback_strategy == "best_candidate" and global_best_below is not None:
            final = global_best_below
            status = "compressed_via_fallback"
            print(f"[FALLBACK] best below-threshold candidate D={final['scores']['D']:.4f}")
        else:
            final = None
            if global_best_below is not None:
                status = "kept_original_none_passed"
                print("[KEEP] No candidate passed all checks; returning original caption.")
            else:
                status = "kept_original_rewriter_failed"
                print("[WARN] No usable candidate produced; returning original.")

        if final is not None:
            final_caption = final["caption"]
            final_images = final["images"]
            final_scores = final["scores"]
            final_cmp = self._single_comparison(kg_original, final["kg_recon"])
            kg_final_counts = self._kg_counts(final.get("kg_recon", {}))
        else:
            final_caption = caption
            final_images = baseline_images
            final_scores = dict(baseline_dist)
            final_cmp = baseline_cmp
            kg_final_counts = self._kg_counts(baseline_kgs[0]) if baseline_kgs else {}

        compressed_tokens = count_tokens(final_caption)
        cr = 1.0 - compressed_tokens / max(original_tokens, 1)
        recon_path = self._save_final(image, final_caption, final_images, sid)
        final_scores = dict(final_scores)
        final_scores["status"] = status

        result = CompressionResult(
            image_id=image_id, status=status,
            original_caption=caption, compressed_caption=final_caption,
            original_tokens=original_tokens, compressed_tokens=compressed_tokens,
            compression_ratio=cr, baseline_distortion=baseline_D, threshold=threshold,
            final_distortion=float(final_scores.get("D", 0.0)),
            final_distortion_details=final_scores,
            baseline_kg_comparison=baseline_cmp, final_kg_comparison=final_cmp,
            strict_anchors=list(strict_anchors), soft_anchors=list(soft_anchors),
            quoted_anchors=list(strict_anchors), protected_anchors=list(soft_anchors),
            steps=steps, total_time=time.time() - start,
            final_reconstruction_path=str(recon_path) if recon_path else None,
            baseline_reconstruction_path=str(baseline_recon_path) if baseline_recon_path else None,
            original_image_path=str(original_image_path) if original_image_path else None,
            kg_I_counts=kg_I_counts, kg_final_counts=kg_final_counts,
        )
        print("\n" + "-" * 70)
        print(f"Result [{image_id}]: {original_tokens} -> {compressed_tokens} tokens | "
              f"CR={cr:.2%} | D={result.final_distortion:.4f} <= delta={threshold:.4f} | {status}")
        print(f"c*: {final_caption}")
        print("-" * 70)
        if self.save_intermediate:
            self._save_result(result)
        return result

    def compress_dataset(self, dataset: Any, max_samples: Optional[int] = None) -> List[CompressionResult]:
        n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        results: List[CompressionResult] = []
        print("\n" + "=" * 70 + f"\nProcessing {n} samples\n" + "=" * 70)
        for idx in tqdm(range(n), desc="KG-RD compression"):
            sample = dataset[idx]
            try:
                results.append(self.compress_single(
                    image=sample["image"], caption=sample["caption"],
                    image_id=sample.get("image_id", f"sample_{idx:06d}"),
                ))
            except Exception as exc:
                print(f"[ERROR] idx={idx}: {exc}")
                continue
        self._save_summary(results)
        return results

    # ================================================================== #
    # Core: find the shortest faithful compression for one source caption
    # ================================================================== #
    def _find_shortest_faithful(
        self, image, original_caption, kg_original, source_caption, original_tokens,
        threshold, baseline_cmp, strict_anchors, soft_anchors, iteration, sid,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[CompressionStep]]:
        source_tokens = count_tokens(source_caption)
        steps: List[CompressionStep] = []
        best_below: Optional[Dict[str, Any]] = None
        # Soft anchors SHAPE generation (the rewriter is told to keep them) and are
        # ENFORCED downstream by reconstruction-side KG recall (the fidelity floor:
        # Q_KG / R_obj / R_attr). They are NOT a hard survival filter — gating
        # candidate survival on semantic presence of many descriptive phrases kills
        # every short candidate. So we allow MOST soft anchors to be missing here and
        # let the floor + gate do the real faithfulness enforcement on reconstructions.
        n_soft = len([a for a in soft_anchors if a and str(a).strip()])
        # tolerate up to ~all-but-two soft anchors missing at the filter stage
        max_soft_miss = max(int(round(0.75 * n_soft)), n_soft - 2) if n_soft else 0
        # Minimum acceptable compressed length (blocks fragment captions on EVERY
        # path: gate, structural relaxation, and fallback).
        min_required = max(self.min_caption_tokens, self.min_target_tokens,
                           int(round(self.min_target_ratio * original_tokens)))

        stages = [("primary", self.budget_ratios), ("escalation", self.escalation_ratios)]
        for stage_name, ratios in stages:
            budgets = self._budgets_from_ratios(ratios, original_tokens, source_tokens)
            if not budgets:
                continue
            print(f"    [{stage_name}] budgets={budgets}")
            raw = self.rewriter.rewrite(
                original_caption=source_caption, strict_anchors=strict_anchors,
                soft_anchors=soft_anchors, target_token_budgets=budgets,
                num_candidates=self.num_candidates_per_iteration,
                seed=self.seed + iteration * 100 + (0 if stage_name == "primary" else 50),
            )
            if not raw:
                continue
            candidates = self._filter_candidates(
                raw, strict_anchors, soft_anchors, source_tokens, budgets,
                original_caption, max_soft_miss, min_required,
            )
            survivors = sorted([c for c in candidates if c.passed_anchor_filter],
                               key=lambda c: c.token_count)
            if not survivors:
                # Last resort: anchors already shaped generation. Let every
                # structurally-valid candidate through so they get scored and
                # the best_candidate fallback can choose by distortion. Candidates
                # marked with a "__...__" reason (too short / below floor / not a
                # sentence / not shorter) stay rejected.
                for c in candidates:
                    if any(m.startswith("__") for m in c.missing_anchors):
                        continue
                    c.passed_anchor_filter = True
                survivors = sorted([c for c in candidates if c.passed_anchor_filter],
                                   key=lambda c: c.token_count)
                if survivors:
                    print(f"    [{stage_name}] anchor filter relaxed "
                          f"(structural-only): {len(survivors)} candidate(s) will be scored")
            if self.save_intermediate:
                self._save_candidates(sid, iteration, stage_name, raw, candidates, strict_anchors, soft_anchors)
            print(f"    [{stage_name}] anchor filter: {len(survivors)}/{len(candidates)} survived")

            stage_passers: List[Dict[str, Any]] = []  # gate-passers this stage (quality-aware select)
            for cand in survivors:  # shortest-first
                t0 = time.time()
                cand_images, cand_kgs = self._reconstruct_and_extract(
                    cand.text, sid, f"iter{iteration}_{stage_name}_cand{cand.index}")
                per_seed = [
                    self.distortion.compute_candidate_distortion(
                        image_orig=image, image_recon=img_r, caption_orig=original_caption,
                        caption_compressed=cand.text, kg_image_orig=kg_original,
                        kg_image_recon=kg_r, use_soft_kg=self.use_soft_kg,
                    )
                    for img_r, kg_r in zip(cand_images, cand_kgs)
                ]
                scores = self._average_scores(per_seed)
                gate = self.distortion.evaluate_faithfulness_gate(
                    scores, delta=threshold, tau_img=self.tau_img,
                    tau_cap=self.tau_cap, tau_kg=self.tau_kg)
                floor_ok, floor_msg = self._fidelity_floor_check(scores, baseline_cmp)
                passed = gate["passed"] and floor_ok
                reason = gate["gate_log"] if floor_ok else f"{gate['gate_log']} | FLOOR: {floor_msg}"
                print(f"      [{'PASS' if passed else 'FAIL'}] #{cand.index} ({cand.token_count}t) {gate['gate_log']}")
                if not floor_ok:
                    print(f"        FLOOR: {floor_msg}")

                comparison = self._single_comparison(kg_original, cand_kgs[0]) if cand_kgs else {}
                step = CompressionStep(
                    iteration=iteration, stage=stage_name, candidate_index=cand.index,
                    candidate_caption=cand.text, target_token_budget=cand.target_token_budget,
                    caption_before=source_caption, tokens_before=source_tokens,
                    tokens_after=cand.token_count, token_reduction=source_tokens - cand.token_count,
                    distortion=float(scores.get("D", 1.0)), threshold=threshold,
                    passed_threshold=passed, selected=False, reason=reason,
                    gate_result=gate, distortion_details=scores,
                    missing_anchors=cand.missing_anchors, time_seconds=time.time() - t0,
                )
                steps.append(step)
                winner_obj = {"caption": cand.text, "tokens": cand.token_count,
                              "images": cand_images, "kg_recon": cand_kgs[0] if cand_kgs else {},
                              "scores": scores, "comparison": comparison, "step": step}
                if passed:
                    if not self.select_by_quality:
                        # Legacy behaviour: first gate-pass (shortest) wins immediately.
                        step.selected = True
                        print(f"      [SELECTED shortest faithful] #{cand.index}")
                        return winner_obj, best_below, steps
                    # Quality-aware: collect ALL gate-passing candidates this stage,
                    # decide after scoring them (shortest among the high-quality ones).
                    stage_passers.append(winner_obj)
                    print(f"      [PASS-collected] #{cand.index} "
                          f"({self.quality_metric}={scores.get(self.quality_metric, float('nan')):.4f})")
                # track best below-threshold (lowest D), within [min_required, original)
                if min_required <= cand.token_count < original_tokens:
                    if best_below is None or scores.get("D", 1.0) < best_below["scores"]["D"]:
                        best_below = winner_obj

            # Quality-aware selection among this stage's gate-passers:
            # choose the SHORTEST caption whose reconstruction quality is within
            # quality_margin of the BEST passer's quality. This implements
            # "shortest caption whose CLIP/Q_img score is high" - it avoids picking
            # a barely-passing short caption when a slightly longer one reconstructs
            # much better, while still preferring brevity among the good ones.
            if self.select_by_quality and stage_passers:
                qm = self.quality_metric
                best_q = max(p["scores"].get(qm, 0.0) for p in stage_passers)
                eligible = [p for p in stage_passers
                            if p["scores"].get(qm, 0.0) >= best_q - self.quality_margin]
                chosen = min(eligible, key=lambda p: p["tokens"])
                chosen["step"].selected = True
                print(f"    [{stage_name}] quality-aware select: best {qm}={best_q:.4f}; "
                      f"{len(eligible)}/{len(stage_passers)} within margin {self.quality_margin}; "
                      f"chose {chosen['tokens']}t ({qm}={chosen['scores'].get(qm, 0.0):.4f})")
                return chosen, best_below, steps

            print(f"    [{stage_name}] no candidate passed; "
                  f"{'escalating' if stage_name == 'primary' else 'exhausted'}.")
        return None, best_below, steps

    # ================================================================== #
    # Reconstruction + KG (shared by baseline and candidates)
    # ================================================================== #
    def _reconstruct_and_extract(self, caption: str, sid: str, prefix: str
                                 ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
        out_dir = (self.reconstruction_dir / sid) if self.save_intermediate else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        images = self.t2i.reconstruct(caption=caption, seeds=self.t2i_seeds,
                                      output_dir=out_dir, prefix=prefix)
        kgs = []
        for seed, img in zip(self.t2i_seeds, images):
            kg = self.sg_extractor.extract_with_relation_retry(img, image_id=f"{sid}_{prefix}_seed{seed}")
            kgs.append(kg)
            if self.save_intermediate:
                self._save_kg(kg, sid, f"{prefix}_seed{seed}_recon_kg")
        return images, kgs

    # ================================================================== #
    # Anchors
    # ================================================================== #
    def _build_soft_anchors(self, kg_original: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
        soft = derive_soft_anchors_from_kg(kg_original, max_anchors=self.max_soft_anchors)
        # re-inject distinctive facts the reconstruction MISSED
        missing = derive_anchors_from_missing(
            {"missing": comparison.get("missing", set())},
            max_anchors=self.max_soft_anchors, include_relations=True,
        )
        seen, merged = set(), []
        for a in soft + missing:
            al = a.lower().strip()
            if not al or al in seen:
                continue
            # Drop long descriptive/POV phrases that a compressed caption cannot be
            # expected to carry verbatim (e.g. "slightly elevated roadside
            # perspective", "visible road markings"). These are framing prose, not
            # scene-defining facts; the rewriter still tends to keep salient ones,
            # and KG recall on the reconstruction is what actually enforces content.
            wc = len(al.split())
            if wc >= 4:
                continue
            merged.append(a)
        # Keep soft-anchor count modest so it informs rather than dominates.
        return merged[: self.max_soft_anchors]

    # ================================================================== #
    # Candidate filtering
    # ================================================================== #
    def _budgets_from_ratios(self, ratios: Sequence[float], original_tokens: int, source_tokens: int) -> List[int]:
        budgets, seen = [], set()
        cap = max(self.min_caption_tokens + 1, source_tokens - 1)
        for r in ratios:
            b = int(round(r * original_tokens))
            b = max(self.min_caption_tokens, min(b, cap))
            if b not in seen:
                seen.add(b)
                budgets.append(b)
        return sorted(budgets)

    def _filter_candidates(self, raw_candidates, strict_anchors, soft_anchors,
                           source_tokens, budgets, original_caption, max_soft_miss,
                           min_required_tokens: int = 0) -> List[RewriteCandidate]:
        candidates, seen = [], set()
        budgets = list(budgets) or [source_tokens]
        # Graduated strict-anchor tolerance: a 24-word caption cannot carry 8
        # plates/signs/phrases. Require keeping most when few, ~half when many.
        n_strict = len([a for a in strict_anchors if a and str(a).strip()])
        if n_strict == 0:
            max_strict_miss = 0
        elif n_strict <= 2:
            max_strict_miss = 1
        elif n_strict <= 5:
            max_strict_miss = 2
        else:
            max_strict_miss = n_strict // 2
        for idx, raw in enumerate(raw_candidates):
            text = clean_caption_spacing(raw)
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            tok = count_tokens(text)
            target = min(budgets, key=lambda b: abs(b - tok))
            cand = RewriteCandidate(idx, text, int(target), tok)

            if self.require_strict_compression and tok >= source_tokens:
                cand.missing_anchors = ["__not_shorter_than_source__"]
            elif tok < max(self.min_caption_tokens, min_required_tokens):
                _floor = max(self.min_caption_tokens, min_required_tokens)
                cand.missing_anchors = [f"__below_floor({tok}<{_floor})__"]
            elif not _looks_like_sentence(text):
                cand.missing_anchors = ["__not_a_sentence__"]
            elif original_caption and not is_substantive_rewrite(
                    original_caption, text, self.min_rewrite_diff_ratio):
                cand.missing_anchors = ["__too_similar_to_original__"]
            else:
                ok, missing = candidate_preserves_anchors(
                    text, strict_anchors, soft_anchors,
                    self.kg_match_threshold, max_soft_miss, max_strict_miss)
                cand.passed_anchor_filter = ok
                cand.missing_anchors = missing
            candidates.append(cand)
        return candidates

    def _fidelity_floor_check(self, cand_scores: Dict[str, float],
                              baseline_cmp: Dict[str, Any]) -> Tuple[bool, str]:
        b_f1 = float(baseline_cmp.get("f1_mean", 0.0))
        b_obj = float(baseline_cmp.get("object_recall_mean", 0.0))
        b_attr = float(baseline_cmp.get("attribute_recall_mean", 0.0))
        c_obj = float(cand_scores.get("R_obj", 0.0))
        c_attr = float(cand_scores.get("R_attr", 0.0))
        c_kg = float(cand_scores.get("Q_KG", 0.0))
        min_f1 = max(self.min_baseline_kg_f1, b_f1 - self.kg_f1_floor_drop)
        min_obj = max(self.min_baseline_object_recall, b_obj - self.object_recall_floor_drop)
        min_attr = max(self.attribute_recall_min_floor, b_attr - self.attribute_recall_floor_drop)
        if c_kg < min_f1:
            return False, f"Q_KG={c_kg:.4f} < floor {min_f1:.4f}"
        if c_obj < min_obj:
            return False, f"R_obj={c_obj:.4f} < floor {min_obj:.4f}"
        if c_attr < min_attr:
            return False, f"R_attr={c_attr:.4f} < floor {min_attr:.4f}"
        return True, "floors_passed"

    # ================================================================== #
    # KG comparison aggregation
    # ================================================================== #
    def _aggregate_comparisons(self, kg_original, reconstructed_kgs) -> Dict[str, Any]:
        comps = [compare_scene_graphs(kg_original, kg_r) for kg_r in reconstructed_kgs]
        preserved, missing, extra = set(), set(), set()
        f1s, prec, rec, obj_r, attr_r, rel_r = [], [], [], [], [], []
        for c in comps:
            preserved |= set(c.get("preserved", set()))
            missing |= set(c.get("missing", set()))
            extra |= set(c.get("extra", set()))
            f1s.append(c.get("f1", 0.0)); prec.append(c.get("precision", 0.0)); rec.append(c.get("recall", 0.0))
            obj_r.append(c.get("object_recall", 0.0)); attr_r.append(c.get("attribute_recall", 0.0))
            rel_r.append(c.get("relation_recall", 0.0))
        m = lambda xs: float(np.mean(xs)) if xs else 0.0
        return {
            "preserved": preserved, "missing": missing, "extra": extra,
            "f1_mean": m(f1s), "f1_std": float(np.std(f1s)) if f1s else 0.0,
            "precision_mean": m(prec), "recall_mean": m(rec),
            "object_recall_mean": m(obj_r), "attribute_recall_mean": m(attr_r), "relation_recall_mean": m(rel_r),
            "num_preserved": len(preserved), "num_missing": len(missing), "num_extra": len(extra),
            "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
            "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
            "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
        }

    def _single_comparison(self, kg_original, kg_candidate) -> Dict[str, Any]:
        c = compare_scene_graphs(kg_original, kg_candidate)
        return {
            "preserved": c.get("preserved", set()), "missing": c.get("missing", set()), "extra": c.get("extra", set()),
            "f1_mean": float(c.get("f1", 0.0)), "precision_mean": float(c.get("precision", 0.0)),
            "recall_mean": float(c.get("recall", 0.0)),
            "object_recall_mean": float(c.get("object_recall", 0.0)),
            "attribute_recall_mean": float(c.get("attribute_recall", 0.0)),
            "relation_recall_mean": float(c.get("relation_recall", 0.0)),
            "weighted_kg_f1_hard": float(c.get("weighted_kg_f1_hard", 0.0)),
            "weighted_kg_f1_soft": float(c.get("weighted_kg_f1_soft", 0.0)),
            "chair_i": float(c.get("chair_i", 0.0)),
            "num_preserved": c.get("num_preserved", 0), "num_missing": c.get("num_missing", 0),
            "num_extra": c.get("num_extra", 0),
            "preserved_text": c.get("preserved_text", []), "missing_text": c.get("missing_text", []),
            "extra_text": c.get("extra_text", []),
        }

    @staticmethod
    def _kg_counts(kg: Dict[str, Any]) -> Dict[str, int]:
        return {
            "obj": len(kg.get("objects", []) or []),
            "attr": len(kg.get("attributes", []) or []),
            "rel": len(kg.get("relations", []) or []),
        }

    def _average_scores(self, per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not per_seed:
            return {}
        avg: Dict[str, Any] = {}
        for key in per_seed[0]:
            if key.startswith("_"):
                avg[key] = per_seed[-1][key]
                continue
            vals = [s[key] for s in per_seed
                    if isinstance(s.get(key), (int, float, np.integer, np.floating))]
            if vals:
                avg[key] = float(np.mean(vals))
                avg[f"{key}_std"] = float(np.std(vals))
        return avg

    # ================================================================== #
    # Output / persistence
    # ================================================================== #
    def _save_final(self, image, final_caption, final_images, sid) -> Optional[Path]:
        d = self.debug_dir / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "final_caption.txt").write_text(final_caption, encoding="utf-8")
        if not final_images:
            return None
        recon = final_images[0].convert("RGB")
        path = d / "final_reconstruction_I_star.png"
        recon.save(path)
        save_side_by_side(image, recon, "Original (I)", "Final Reconstruction (I*)",
                          d / "side_by_side_original_vs_I_star.png", final_caption)
        print(f"[Output] I* saved: {path}")
        return path

    def _skipped_result(self, image_id, caption, original_tokens, baseline_D, threshold,
                        baseline_dist, baseline_cmp, sid, start) -> CompressionResult:
        details = dict(baseline_dist); details["status"] = "skipped_low_baseline_fidelity"
        result = CompressionResult(
            image_id=image_id, status="skipped_low_baseline_fidelity",
            original_caption=caption, compressed_caption=caption,
            original_tokens=original_tokens, compressed_tokens=original_tokens,
            compression_ratio=0.0, baseline_distortion=baseline_D, threshold=threshold,
            final_distortion=baseline_D, final_distortion_details=details,
            baseline_kg_comparison=baseline_cmp, final_kg_comparison=baseline_cmp,
            strict_anchors=[], soft_anchors=[], quoted_anchors=[], protected_anchors=[],
            steps=[], total_time=time.time() - start, final_reconstruction_path=None,
        )
        if self.save_intermediate:
            self._save_result(result)
        return result

    def _save_kg(self, kg, image_id, name) -> None:
        d = self.kg_debug_dir / safe_filename(image_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{safe_filename(name)}.json").write_text(
            json.dumps(make_json_serializable(kg_to_readable_dict(kg)), indent=2, ensure_ascii=False),
            encoding="utf-8")

    def _save_comparison(self, comparison, image_id, name) -> None:
        d = self.kg_debug_dir / safe_filename(image_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{safe_filename(name)}.json").write_text(
            json.dumps(make_json_serializable(comparison), indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_candidates(self, image_id, iteration, stage, raw, filtered, strict, soft) -> None:
        d = self.candidate_debug_dir / safe_filename(image_id)
        d.mkdir(parents=True, exist_ok=True)
        payload = {"iteration": iteration, "stage": stage, "strict_anchors": list(strict),
                   "soft_anchors": list(soft), "raw_candidates": list(raw),
                   "filtered_candidates": [asdict(c) for c in filtered]}
        (d / f"iter{iteration}_{stage}_candidates.json").write_text(
            json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_result(self, result: CompressionResult) -> None:
        path = self.result_dir / f"{safe_filename(result.image_id)}.json"
        path.write_text(json.dumps(make_json_serializable(result), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Saved] {path}")

    def _save_summary(self, results: List[CompressionResult]) -> None:
        if not results:
            return
        statuses: Dict[str, int] = {}
        for r in results:
            statuses[r.status] = statuses.get(r.status, 0) + 1
        summary = {
            "num_samples": len(results), "status_counts": statuses,
            "avg_compression_ratio": float(np.mean([r.compression_ratio for r in results])),
            "avg_baseline_distortion": float(np.mean([r.baseline_distortion for r in results])),
            "avg_final_distortion": float(np.mean([r.final_distortion for r in results])),
            "avg_original_tokens": float(np.mean([r.original_tokens for r in results])),
            "avg_compressed_tokens": float(np.mean([r.compressed_tokens for r in results])),
            "avg_time_seconds": float(np.mean([r.total_time for r in results])),
            "total_time_hours": float(np.sum([r.total_time for r in results]) / 3600.0),
            "use_relative_threshold": self.use_relative_threshold, "epsilon": self.epsilon,
            "tau_img": self.tau_img, "tau_cap": self.tau_cap, "tau_kg": self.tau_kg,
        }
        (self.result_dir / "summary.json").write_text(
            json.dumps(make_json_serializable(summary), indent=2), encoding="utf-8")
        print("\n" + "=" * 70)
        print(f"SUMMARY - {len(results)} samples | {statuses}")
        print(f"  Avg CR={summary['avg_compression_ratio']:.2%} | "
              f"D0={summary['avg_baseline_distortion']:.4f} -> D={summary['avg_final_distortion']:.4f}")
        print(f"  Tokens {summary['avg_original_tokens']:.1f} -> {summary['avg_compressed_tokens']:.1f} | "
              f"{summary['total_time_hours']:.2f}h")
        print("=" * 70)
