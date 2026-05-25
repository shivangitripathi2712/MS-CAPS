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
        use_relative_threshold: bool = True, epsilon: float = 0.10, absolute_threshold: float = 0.25,
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
        self.absolute_threshold = float(absolute_threshold)
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

        # 7. Resolve final (never the original unless rewriter produced nothing)
        if winner is not None:
            final = winner
            status = "compressed"
        elif self.fallback_strategy == "best_candidate" and global_best_below is not None:
            final = global_best_below
            status = "compressed_via_fallback"
            print(f"[FALLBACK] best below-threshold candidate D={final['scores']['D']:.4f}")
        else:
            # Total rewriter failure: only case where original is returned.
            final = None
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





# """
# KG-RD Compressor
# ================

# KG-guided Rate-Distortion caption compressor (reconstruction-guided).

# Per the locked framework, BOTH the baseline and every candidate are scored
# against FLUX reconstructions, and Q_KG is computed from the reconstruction's KG
# (NOT from caption text). The decoder G is caption-only, so distortion isolates
# the caption's information content.

# Pipeline (per image)
# ---------------------
# 1. KG_I  = KG from the original image I (ground truth).
# 2. visible_text = readable in-image text (plate/brand/sign) for strict anchors.
# 3. Baseline: I_c = G(c); KG_{I_c}; D_0 = alpha*(1-Q_img_base)+gamma*(1-Q_KG_base);
#    delta = min((1+epsilon)*D_0, absolute_threshold).
# 4. Baseline gate: skip samples whose full caption already reconstructs poorly.
# 5. Anchors:
#      strict S_c = noun-phrases (KG-cross-validated) + visible text  (substring-enforced)
#      soft  G_c  = salient KG facts (framing/reflection first) + missing-from-baseline
# 6. Find the SHORTEST faithful compression:
#      - budget ladder [20,30,45,55,65]% of original tokens (one rewrite call)
#      - evaluate survivors shortest-first; first gate-pass wins (shortest faithful)
#      - if none pass, ESCALATE to higher budgets [70,78,86,94]%
#      - if still none, fall back to the best (lowest-D) shorter-than-original
#        candidate (NEVER the original caption) that also clears the length floor
# 7. Outer loop (max_iterations): try to compress the winner further; re-inject the
#    winner's missing facts as soft anchors.
# 8. Save I* = G(c*) and the result.

# Gate (C1-C4, all must pass; logged per condition):
#     C1 Q_img>=tau_img  C2 Q_cap>=tau_cap  C3 Q_KG>=tau_kg  C4 D<=delta

# Minimum-length floor
# --------------------
# A minimum acceptable compressed length is enforced on EVERY selection path
# (gate, structural relaxation, and fallback) so that no path can ever return a
# fragment caption. It is a declared, frozen hyperparameter:
#     min_required = max(min_caption_tokens, min_target_tokens,
#                        round(min_target_ratio * original_tokens))
# Configure via compression.min_target_ratio / compression.min_target_tokens.
# Defaults are 0 (disabled) so the floor is opt-in via config, not a hidden number.
# """

# import json
# import re
# import time
# from abc import ABC, abstractmethod
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .parse_tree import extract_strict_anchors, _shared_pruner
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     canonicalize_for_matching,
#     compare_scene_graphs,
#     derive_anchors_from_missing,
#     derive_soft_anchors_from_kg,
#     normalize_label,
#     semantic_text_similarity,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor, build_reconstructor


# # Default budget ladders (fractions of ORIGINAL caption tokens).
# _PRIMARY_RATIOS = [0.20, 0.30, 0.45, 0.55, 0.65]
# _ESCALATION_RATIOS = [0.70, 0.78, 0.86, 0.94]


# # =====================================================================
# # Data classes
# # =====================================================================

# @dataclass
# class RewriteCandidate:
#     index: int
#     text: str
#     target_token_budget: int
#     token_count: int
#     missing_anchors: List[str] = field(default_factory=list)
#     passed_anchor_filter: bool = False


# @dataclass
# class CompressionStep:
#     iteration: int
#     stage: str
#     candidate_index: int
#     candidate_caption: str
#     target_token_budget: int
#     caption_before: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     gate_result: Dict[str, Any]
#     distortion_details: Dict[str, Any]
#     missing_anchors: List[str]
#     time_seconds: float


# @dataclass
# class CompressionResult:
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     strict_anchors: List[str]
#     soft_anchors: List[str]
#     steps: List[CompressionStep]
#     total_time: float
#     final_reconstruction_path: Optional[str] = None
#     baseline_reconstruction_path: Optional[str] = None
#     original_image_path: Optional[str] = None
#     kg_I_counts: Dict[str, int] = field(default_factory=dict)
#     kg_final_counts: Dict[str, int] = field(default_factory=dict)
#     # Backward-compatible aliases (older readers expect these names):
#     protected_anchors: List[str] = field(default_factory=list)   # == soft_anchors
#     quoted_anchors: List[str] = field(default_factory=list)       # == strict_anchors


# # =====================================================================
# # Helpers
# # =====================================================================

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"\(\s+", "(", text)
#     text = re.sub(r"\s+\)", ")", text)
#     text = text.replace(" ,", ",").replace(" .", ".")
#     return text.strip()


# def safe_filename(name: str) -> str:
#     return re.sub(r"[^\w\-.]+", "_", str(name))[:180]


# def _dataclass_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if _dataclass_instance(obj):
#         return make_json_serializable(asdict(obj))
#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]
#     if isinstance(obj, tuple):
#         return list(obj)
#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "scene_context": kg.get("scene_context", {}),
#         "tuples": [list(t) for t in sorted(list(kg.get("tuples", set())), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def _token_set(s: str) -> Set[str]:
#     return {w.lower().strip(".,;:!?\"'()[]{}") for w in s.split() if w.strip()}


# def is_substantive_rewrite(original: str, candidate: str, min_diff_ratio: float = 0.15) -> bool:
#     a, b = _token_set(original), _token_set(candidate)
#     if not a or not b:
#         return True
#     union, inter = a | b, a & b
#     return (1.0 - len(inter) / len(union)) >= min_diff_ratio if union else True


# def _looks_like_sentence(text: str) -> bool:
#     """Reject bag-of-words: require at least one verb/aux (via shared spaCy)."""
#     nlp = _shared_pruner().nlp
#     if nlp is None:
#         return True  # cannot check; do not reject
#     doc = nlp(text)
#     return any(t.pos_ in ("VERB", "AUX") for t in doc)


# _QUOTE_RE = r'"([^"]+)"|\u201c([^\u201c\u201d]+)\u201d|`([^`]+)`'


# def is_text_bearing_anchor(anchor: str, visible_text_set: Optional[Set[str]] = None) -> bool:
#     """
#     True only for DISTINCTIVE, text-bearing anchors that a compressed caption must
#     keep VERBATIM because they cannot be recovered semantically: detected visible
#     text (plates/brands/signage), alphanumerics (license plates), and ALL-CAPS
#     brand tokens. Descriptive phrases ("green traffic light", "elevated
#     perspective") are NOT text-bearing - they belong in soft anchors, where their
#     preservation is enforced via reconstruction-side KG recall (Q_KG / R_obj),
#     not literal string matching.
#     """
#     a = (anchor or "").strip()
#     if not a:
#         return False
#     al = a.lower()
#     vt = visible_text_set or set()
#     # matches a detected visible-text string (plate / brand / slogan / sign)
#     for t in vt:
#         t = (t or "").strip().lower()
#         if t and (t in al or al in t):
#             return True
#     # contains a digit -> license plate / alphanumeric code
#     if any(ch.isdigit() for ch in a):
#         return True
#     # contains an ALL-CAPS alphabetic token of length >= 2 -> brand / sign
#     for tok in a.split():
#         t = tok.strip(".,;:!?'\"()[]{}")
#         if len(t) >= 2 and t.isalpha() and t.isupper():
#             return True
#     return False


# def _extract_quoted_substrings(text: str) -> List[str]:
#     out = []
#     for m in re.finditer(_QUOTE_RE, text or ""):
#         s = next((g for g in m.groups() if g), "").strip()
#         if s:
#             out.append(s)
#     return out


# def candidate_preserves_anchors(
#     candidate: str,
#     strict_anchors: Sequence[str],
#     soft_anchors: Sequence[str],
#     kg_match_threshold: float = 0.55,
#     max_soft_miss_count: int = 0,
#     max_strict_miss_count: int = 0,
# ) -> Tuple[bool, List[str]]:
#     """
#     Strict anchors should be present (substring / verbatim / token-overlap); up to
#     ``max_strict_miss_count`` may be missing (graduated tolerance - a heavily
#     compressed caption cannot carry every plate/sign/phrase). Soft anchors should
#     be present semantically; up to ``max_soft_miss_count`` may be missing.
#     Returns (passed, missing_anchor_list).
#     """
#     if not candidate:
#         return False, list(strict_anchors) + list(soft_anchors)

#     cand_low = candidate.lower()
#     cand_norm = canonicalize_for_matching(candidate)
#     cand_norm_words = set(cand_norm.split())

#     strict_missing: List[str] = []
#     for a in strict_anchors:
#         a_low = a.strip().lower()
#         if not a_low:
#             continue
#         if a_low in cand_low:
#             continue
#         # token-overlap leniency for multi-word strict anchors
#         a_norm = canonicalize_for_matching(a)
#         a_words = [w for w in a_norm.split() if len(w) > 2]
#         if a_words and all(w in cand_norm_words for w in a_words):
#             continue
#         strict_missing.append(a)

#     soft_missing: List[str] = []
#     for a in soft_anchors:
#         a_norm = canonicalize_for_matching(a)
#         if not a_norm:
#             continue
#         if a_norm in cand_norm:
#             continue
#         a_words = [w for w in a_norm.split() if len(w) > 2]
#         if a_words and any(w in cand_norm_words for w in a_words):
#             continue
#         if semantic_text_similarity(a_norm, cand_norm) >= kg_match_threshold:
#             continue
#         soft_missing.append(a)

#     if len(strict_missing) > max_strict_miss_count:
#         return False, strict_missing + soft_missing
#     if len(soft_missing) > max_soft_miss_count:
#         return False, strict_missing + soft_missing
#     return True, strict_missing + soft_missing


# def save_side_by_side(left_img: Image.Image, right_img: Image.Image,
#                       left_label: str, right_label: str, path: Path,
#                       caption: str = "") -> None:
#     left = left_img.convert("RGB").resize((512, 512))
#     right = right_img.convert("RGB").resize((512, 512))
#     canvas = Image.new("RGB", (1024, 580), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))
#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), left_label, fill="black")
#     draw.text((565, 15), right_label, fill="black")
#     if caption:
#         preview = caption[:100] + "..." if len(caption) > 100 else caption
#         draw.text((10, 532), preview, fill="black")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     canvas.save(path)


# # =====================================================================
# # Rewriter
# # =====================================================================

# class CaptionRewriter(ABC):
#     @abstractmethod
#     def rewrite(self, original_caption: str, strict_anchors: Sequence[str],
#                 soft_anchors: Sequence[str], target_token_budgets: Sequence[int],
#                 num_candidates: int, seed: Optional[int] = None) -> List[str]:
#         ...


# class QwenVLRewriter(CaptionRewriter):
#     """Text-only caption rewriter sharing the Qwen2.5-VL model (no extra GPU load)."""

#     def __init__(self, vlm_extractor: VLMSceneGraphExtractor,
#                  max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
#         self.model = vlm_extractor.model
#         self.processor = vlm_extractor.processor
#         self.device = vlm_extractor.device
#         self.max_new_tokens = int(max_new_tokens)
#         self.temperature = float(temperature)
#         self.top_p = float(top_p)

#     def _build_prompt(self, original_caption, strict_anchors, soft_anchors,
#                       target_token_budgets, num_candidates) -> str:
#         original_tokens = count_tokens(original_caption)
#         strict_block = "\n".join(f'  ({i}) "{a}"' for i, a in enumerate(strict_anchors, 1)) or "  (none)"
#         soft_block = "\n".join(f"  ({i}) {a}" for i, a in enumerate(soft_anchors, 1)) or "  (none)"
#         budgets = list(target_token_budgets)
#         min_budget = min(budgets) if budgets else max(8, original_tokens // 2)
#         budget_lines = "\n".join(f"  - Variant {i+1}: about {b} tokens"
#                                  for i, b in enumerate(budgets)) or f"  - about {min_budget} tokens"
#         return f"""You compress an image caption while preserving the visual information needed to faithfully reconstruct the image. Removing a visually important detail makes the reconstructed image look different from the original - avoid that.

# ORIGINAL CAPTION ({original_tokens} tokens):
# {original_caption}

# REQUIREMENT 1 - KEEP THESE EXACT PHRASES (verbatim, unchanged):
# These name distinctive or text-bearing elements (signs, plates, brands, framing
# devices). Every variant MUST contain each of them word-for-word.
# {strict_block}

# REQUIREMENT 2 - PRESERVE THESE FACTS (you may rephrase, but the meaning must remain):
# These are important visual facts. Every variant must still convey each one.
# {soft_block}

# REQUIREMENT 2b - ALWAYS KEEP VIEWPOINT, ORIENTATION, AND DISTINCTIVE MARKINGS:
# These few words cost almost nothing but decide whether a reconstruction looks like
# the original. Whenever the original caption states any of them, you MUST keep them:
#   - Camera viewpoint: "side view", "front view", "rear view", "from above",
#     "elevated perspective", "close-up", "wide shot".
#   - Object orientation / pose: "facing right", "facing left", "parked diagonally",
#     "side and rear visible", "front-facing", "angled".
#   - Distinctive colors and markings: stripes, body color, painted text bands,
#     decorative patterns (e.g. "blue stripes", "green roof", "red trim").
#   - Spatial layout: foreground / midground / background placement and left/right
#     position of the main objects.
# Dropping a viewpoint or a distinctive marking counts as losing a key visual fact -
# do not drop them to save length.

# REQUIREMENT 3 - GRAMMATICAL SENTENCES:
# Each variant is one or more complete, fluent English sentences. Never a list of
# keywords, never a fragment, never comma-separated noun phrases.

# REQUIREMENT 4 - SHORTER THAN THE ORIGINAL:
# Every variant must be clearly shorter than {original_tokens} tokens. Produce
# {num_candidates} variants of increasing length, shortest first:
# {budget_lines}

# Drop only redundant or non-visual wording. When in doubt about a distinctive or
# spatial detail, keep it.

# OUTPUT - return ONLY a JSON array of strings, shortest first. No markdown, no
# numbering, no commentary:
# ["shortest variant", "...", "longest variant"]
# """

#     def _parse_json_array(self, text: str) -> List[str]:
#         if not isinstance(text, str):
#             return []
#         text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"```", "", text).strip()
#         match = re.search(r"\[.*\]", text, flags=re.DOTALL)
#         for candidate in (text, match.group(0) if match else None):
#             if not candidate:
#                 continue
#             try:
#                 parsed = json.loads(candidate)
#                 if isinstance(parsed, list):
#                     return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#             except Exception:
#                 continue
#         # Last resort: pull quoted strings.
#         return [s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text) if s.strip()]

#     @torch.no_grad()
#     def _generate_text(self, prompt: str) -> str:
#         messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
#         text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         try:
#             inputs = self.processor(text=[text_input], padding=True, return_tensors="pt")
#         except Exception:
#             inputs = self.processor.tokenizer(text_input, padding=True, return_tensors="pt")
#         inputs = inputs.to(self.model.device) if hasattr(inputs, "to") else {k: v.to(self.model.device) for k, v in inputs.items()}
#         generated_ids = self.model.generate(
#             **inputs, max_new_tokens=self.max_new_tokens,
#             do_sample=True, temperature=self.temperature, top_p=self.top_p,
#         )
#         input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
#         trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]
#         decoder = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
#         return decoder(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

#     def rewrite(self, original_caption, strict_anchors, soft_anchors,
#                 target_token_budgets, num_candidates, seed: Optional[int] = None) -> List[str]:
#         if seed is not None:
#             torch.manual_seed(int(seed))
#         prompt = self._build_prompt(original_caption, strict_anchors, soft_anchors,
#                                     target_token_budgets, num_candidates)
#         try:
#             raw = self._generate_text(prompt)
#         except Exception as exc:
#             print(f"[Rewriter] Generation failed: {exc}")
#             return []
#         cands = [clean_caption_spacing(c) for c in self._parse_json_array(raw) if c]
#         seen, unique = set(), []
#         for c in cands:
#             if c.lower() not in seen:
#                 seen.add(c.lower())
#                 unique.append(c)
#         return unique


# # =====================================================================
# # Main compressor
# # =====================================================================

# class KGRDCompressor:
#     """KG-guided Rate-Distortion caption compressor (see module docstring)."""

#     def __init__(
#         self,
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,
#         t2i_model: str = "black-forest-labs/FLUX.1-dev",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "bfloat16",
#         t2i_steps: int = 20,
#         t2i_guidance: float = 3.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,
#         alpha: float = 0.4, beta: float = 0.2, gamma: float = 0.4,
#         clip_model: str = "ViT-L-14", clip_pretrained: str = "openai",
#         dino_model: str = "dinov2_vitb14", lpips_net: str = "alex",
#         bertscore_model: str = "microsoft/deberta-xlarge-mnli",
#         metric_device: str = "auto",
#         kg_object_weight: float = 0.4, kg_attribute_weight: float = 0.3, kg_relation_weight: float = 0.3,
#         use_soft_kg: bool = True,
#         tau_img: float = 0.50, tau_cap: float = 0.55, tau_kg: float = 0.45,
#         use_relative_threshold: bool = True, epsilon: float = 0.10, absolute_threshold: float = 0.25,
#         min_caption_tokens: int = 5,
#         min_target_ratio: float = 0.0,
#         min_target_tokens: int = 0,
#         max_iterations: int = 2,
#         num_candidates_per_iteration: int = 5,
#         budget_ratios: Optional[List[float]] = None,
#         escalation_ratios: Optional[List[float]] = None,
#         require_strict_compression: bool = True,
#         min_baseline_kg_f1: float = 0.05, min_baseline_object_recall: float = 0.10,
#         kg_f1_floor_drop: float = 0.08, object_recall_floor_drop: float = 0.05,
#         attribute_recall_floor_drop: float = 0.12, attribute_recall_min_floor: float = 0.10,
#         max_strict_anchors: int = 8, max_soft_anchors: int = 6,
#         kg_match_threshold: float = 0.55, soft_anchor_max_miss_fraction: float = 0.40,
#         min_rewrite_diff_ratio: float = 0.15,
#         fallback_strategy: str = "best_candidate",
#         rewriter: Optional[CaptionRewriter] = None,
#         rewriter_max_new_tokens: int = 512, rewriter_temperature: float = 0.7, rewriter_top_p: float = 0.9,
#         save_intermediate: bool = True, output_dir: str = "./outputs", seed: int = 42,
#         t2i_reconstructor=None,
#     ):
#         self.t2i_seeds = t2i_seeds or [42]
#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)
#         self.use_soft_kg = bool(use_soft_kg)
#         self.tau_img, self.tau_cap, self.tau_kg = float(tau_img), float(tau_cap), float(tau_kg)
#         self.min_caption_tokens = int(min_caption_tokens)
#         self.min_target_ratio = float(min_target_ratio)
#         self.min_target_tokens = int(min_target_tokens)
#         self.max_iterations = int(max_iterations)
#         self.num_candidates_per_iteration = int(num_candidates_per_iteration)
#         self.budget_ratios = list(budget_ratios) if budget_ratios else list(_PRIMARY_RATIOS)
#         self.escalation_ratios = list(escalation_ratios) if escalation_ratios else list(_ESCALATION_RATIOS)
#         self.require_strict_compression = bool(require_strict_compression)
#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)
#         self.kg_f1_floor_drop = float(kg_f1_floor_drop)
#         self.object_recall_floor_drop = float(object_recall_floor_drop)
#         self.attribute_recall_floor_drop = float(attribute_recall_floor_drop)
#         self.attribute_recall_min_floor = float(attribute_recall_min_floor)
#         self.max_strict_anchors = int(max_strict_anchors)
#         self.max_soft_anchors = int(max_soft_anchors)
#         self.kg_match_threshold = float(kg_match_threshold)
#         self.soft_anchor_max_miss_fraction = float(soft_anchor_max_miss_fraction)
#         self.min_rewrite_diff_ratio = float(min_rewrite_diff_ratio)
#         if fallback_strategy not in {"keep_original", "best_candidate"}:
#             raise ValueError(f"Invalid fallback_strategy: {fallback_strategy}")
#         self.fallback_strategy = fallback_strategy
#         self.save_intermediate = bool(save_intermediate)
#         self.seed = int(seed)

#         self.output_dir = Path(output_dir)
#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"
#         self.candidate_debug_dir = self.output_dir / "candidates"
#         self.debug_dir = self.output_dir / "debug_samples"
#         for d in (self.result_dir, self.kg_debug_dir, self.candidate_debug_dir, self.debug_dir):
#             d.mkdir(parents=True, exist_ok=True)

#         print("=" * 70 + "\nKG-RD Compressor: initializing\n" + "=" * 70)
#         print("\n[1/3] Loading VLM scene-graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model, device=scene_graph_device, dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens, temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample, max_objects=max_objects, max_relations=max_relations,
#         )
#         print("\n[2/3] Loading T2I reconstructor...")
#         if t2i_reconstructor is not None:
#             self.t2i = t2i_reconstructor
#         else:
#             self.t2i = T2IReconstructor(
#                 model_name=t2i_model, device=t2i_device, dtype=t2i_dtype,
#                 num_inference_steps=t2i_steps, guidance_scale=t2i_guidance,
#                 image_size=t2i_image_size, img2img_mode=False,
#             )
#         print("\n[3/3] Loading distortion metric (CLIP + DINO + LPIPS + BERTScore)...")
#         self.distortion = DistortionMetric(
#             alpha=alpha, beta=beta, gamma=gamma,
#             clip_model_name=clip_model, clip_pretrained=clip_pretrained,
#             dino_model_name=dino_model, lpips_net=lpips_net,
#             bertscore_model=bertscore_model, device=metric_device,
#             kg_object_weight=kg_object_weight, kg_attribute_weight=kg_attribute_weight,
#             kg_relation_weight=kg_relation_weight,
#         )
#         self.rewriter = rewriter or QwenVLRewriter(
#             vlm_extractor=self.sg_extractor, max_new_tokens=rewriter_max_new_tokens,
#             temperature=rewriter_temperature, top_p=rewriter_top_p,
#         )
#         print("\n" + "=" * 70)
#         print(f"Ready. taus: img={self.tau_img} cap={self.tau_cap} kg={self.tau_kg} | "
#               f"eps={self.epsilon} abs<= {self.absolute_threshold} | use_soft_kg={self.use_soft_kg}")
#         print(f"Budget ladder: {self.budget_ratios}  escalation: {self.escalation_ratios}")
#         print(f"Length floor: min_target_ratio={self.min_target_ratio} "
#               f"min_target_tokens={self.min_target_tokens} (min_caption_tokens={self.min_caption_tokens})")
#         print(f"candidates/iter={self.num_candidates_per_iteration} max_iters={self.max_iterations} "
#               f"fallback={self.fallback_strategy}")
#         print("=" * 70 + "\n")

#     # ------------------------------------------------------------------ #
#     @classmethod
#     def from_config(cls, config: Dict[str, Any], rewriter: Optional[CaptionRewriter] = None) -> "KGRDCompressor":
#         sg = config["scene_graph"]["vlm"]
#         t = config["t2i"]
#         d = config["distortion"]; w = d["weights"]; kg = d.get("kg", {})
#         g = config["gate"]; comp = config["compression"]; rw = config.get("rewriter", {})
#         bg = config.get("baseline_gate", {}); ff = config.get("fidelity_floors", {})
#         anc = config.get("anchors", {}); fb = config.get("fallback", {})
#         paths = config.get("paths", {}); ev = config.get("evaluation", {})
#         # Text-only vs structure-conditioned (ControlNet) selected by config.
#         reconstructor = build_reconstructor(config)
#         return cls(
#             scene_graph_model=sg["model"], scene_graph_device=sg["device"], scene_graph_dtype=sg["dtype"],
#             scene_graph_max_new_tokens=sg.get("max_new_tokens", 1024),
#             scene_graph_temperature=sg.get("temperature", 0.0), scene_graph_do_sample=sg.get("do_sample", False),
#             max_objects=config["scene_graph"].get("max_objects", 50),
#             max_relations=config["scene_graph"].get("max_relations", 80),
#             t2i_model=t["model"], t2i_device=t["device"], t2i_dtype=t["dtype"],
#             t2i_steps=t.get("num_inference_steps", 20), t2i_guidance=t.get("guidance_scale", 3.5),
#             t2i_image_size=t.get("image_size", 512), t2i_seeds=t.get("seeds", [42]),
#             alpha=w["alpha_img"], beta=w["beta_cap"], gamma=w["gamma_kg"],
#             clip_model=d.get("clip", {}).get("model", "ViT-L-14"),
#             clip_pretrained=d.get("clip", {}).get("pretrained", "openai"),
#             dino_model=d.get("dino", {}).get("model", "dinov2_vitb14"),
#             lpips_net=d.get("lpips", {}).get("net", "alex"),
#             bertscore_model=d.get("bertscore", {}).get("model", "microsoft/deberta-xlarge-mnli"),
#             metric_device=d.get("clip", {}).get("device", "cuda:0"),
#             kg_object_weight=kg.get("object_weight", 0.4), kg_attribute_weight=kg.get("attribute_weight", 0.3),
#             kg_relation_weight=kg.get("relation_weight", 0.3),
#             use_soft_kg=d.get("use_soft_kg", True),
#             tau_img=g["tau_img"], tau_cap=g["tau_cap"], tau_kg=g["tau_kg"],
#             use_relative_threshold=d.get("use_relative_threshold", True),
#             epsilon=d.get("epsilon", 0.10), absolute_threshold=d.get("absolute_threshold", 0.25),
#             min_caption_tokens=comp.get("min_caption_tokens", 5),
#             min_target_ratio=comp.get("min_target_ratio", 0.0),
#             min_target_tokens=comp.get("min_target_tokens", 0),
#             max_iterations=comp.get("max_iterations", 2),
#             num_candidates_per_iteration=comp.get("num_candidates_per_iteration", 5),
#             budget_ratios=comp.get("budget_ratios", _PRIMARY_RATIOS),
#             escalation_ratios=comp.get("escalation_ratios", _ESCALATION_RATIOS),
#             require_strict_compression=comp.get("require_strict_compression", True),
#             min_baseline_kg_f1=bg.get("min_baseline_kg_f1", 0.05),
#             min_baseline_object_recall=bg.get("min_baseline_object_recall", 0.10),
#             kg_f1_floor_drop=ff.get("kg_f1_floor_drop", 0.08),
#             object_recall_floor_drop=ff.get("object_recall_floor_drop", 0.05),
#             attribute_recall_floor_drop=ff.get("attribute_recall_floor_drop", 0.12),
#             attribute_recall_min_floor=ff.get("attribute_recall_min_floor", 0.10),
#             max_soft_anchors=anc.get("max_kg_anchors", 6),
#             kg_match_threshold=anc.get("kg_match_threshold", 0.55),
#             soft_anchor_max_miss_fraction=anc.get("soft_anchor_max_miss_fraction", 0.40),
#             fallback_strategy=fb.get("strategy", "best_candidate"),
#             rewriter=rewriter,
#             rewriter_max_new_tokens=rw.get("max_new_tokens", 512),
#             rewriter_temperature=rw.get("temperature", 0.7), rewriter_top_p=rw.get("top_p", 0.9),
#             save_intermediate=ev.get("save_intermediate", True),
#             output_dir=paths.get("output_dir", "./outputs"),
#             seed=config.get("seed", 42),
#             t2i_reconstructor=reconstructor,
#         )

#     # ================================================================== #
#     # Public API
#     # ================================================================== #
#     def compress_single(self, image: Image.Image, caption: str, image_id: str = "unknown") -> CompressionResult:
#         start = time.time()
#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")
#         image = image.convert("RGB") if image.mode != "RGB" else image
#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)
#         if original_tokens <= self.min_caption_tokens:
#             raise ValueError(f"Caption has only {original_tokens} tokens (min {self.min_caption_tokens}).")
#         sid = safe_filename(image_id)
#         steps: List[CompressionStep] = []

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | tokens={original_tokens}")
#         print("-" * 70)

#         # 1. KG_I + visible text
#         print("[1] Extracting KG_I from original image...")
#         # Structure-conditioned reconstructor: cache the control map from I ONCE,
#         # so baseline and every candidate share the same layout (no-op for text-only).
#         if hasattr(self.t2i, "set_structure_reference"):
#             self.t2i.set_structure_reference(image)
#         kg_original = self.sg_extractor.extract_with_relation_retry(image, image_id=image_id)
#         visible_text = self.sg_extractor.extract_visible_text(image)
#         print(f"    KG_I: {len(kg_original.get('objects', []))} obj / "
#               f"{len(kg_original.get('attributes', []))} attr / {len(kg_original.get('relations', []))} rel; "
#               f"visible_text={[v['text'] for v in visible_text]}")
#         if self.save_intermediate:
#             self._save_kg(kg_original, sid, "kg_original_image")
#         kg_I_counts = self._kg_counts(kg_original)
#         # Save the original image for the Excel report.
#         original_image_path = None
#         if self.save_intermediate:
#             (self.debug_dir / sid).mkdir(parents=True, exist_ok=True)
#             original_image_path = self.debug_dir / sid / "original_image.png"
#             image.save(original_image_path)

#         # 2. Baseline reconstruction + D_0 + threshold
#         print("[2] Baseline reconstruction G(c)...")
#         baseline_images, baseline_kgs = self._reconstruct_and_extract(caption, sid, "baseline")
#         baseline_recon_path = (self.reconstruction_dir / sid / f"baseline_seed{self.t2i_seeds[0]}.png")
#         baseline_recon_path = baseline_recon_path if baseline_recon_path.exists() else None
#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image, images_recon=baseline_images,
#             kg_original=kg_original, kgs_reconstructed=baseline_kgs,
#         )
#         baseline_D = float(baseline_dist.get("D", 1.0))
#         threshold = (self.distortion.compute_relative_threshold(
#             baseline_D, self.epsilon, self.absolute_threshold)
#             if self.use_relative_threshold else self.absolute_threshold)
#         baseline_cmp = self._aggregate_comparisons(kg_original, baseline_kgs)
#         print(f"    D_0={baseline_D:.4f}  delta={threshold:.4f}  "
#               f"(baseline f1={baseline_cmp['f1_mean']:.3f} obj_rec={baseline_cmp['object_recall_mean']:.3f})")
#         if self.save_intermediate and baseline_images:
#             save_side_by_side(image, baseline_images[0], "Original (I)", "Baseline G(c)",
#                               self.debug_dir / sid / "side_by_side_original_vs_baseline.png", caption)
#             self._save_comparison(baseline_cmp, sid, "baseline_kg_comparison")

#         # 3. Baseline gate
#         if (baseline_cmp["f1_mean"] < self.min_baseline_kg_f1
#                 or baseline_cmp["object_recall_mean"] < self.min_baseline_object_recall):
#             print("[SKIP] Baseline fidelity below threshold.")
#             return self._skipped_result(image_id, caption, original_tokens, baseline_D,
#                                         threshold, baseline_dist, baseline_cmp, sid, start)

#         # 4. Anchors
#         strict_anchors = extract_strict_anchors(
#             caption, kg=kg_original, visible_text=visible_text,
#             kg_match_threshold=self.kg_match_threshold, max_anchors=self.max_strict_anchors,
#         )
#         soft_anchors = self._build_soft_anchors(kg_original, baseline_cmp)
#         # Strict = VERBATIM-required text only (plates/brands/signage). Descriptive
#         # phrases ("green traffic light", "elevated perspective") are NOT verbatim-
#         # recoverable obligations; demote them to soft, where the rewriter is told
#         # to keep them and the reconstruction-side KG recall (Q_KG / R_obj floor)
#         # actually enforces important objects/relations. This keeps "strict" small
#         # and satisfiable so faithful candidates are not rejected wholesale.
#         vt_set = {v.get("text", "") for v in (visible_text or [])}
#         distinctive = [a for a in strict_anchors if is_text_bearing_anchor(a, vt_set)]
#         demoted = [a for a in strict_anchors if a not in distinctive]
#         if demoted:
#             seen, merged = set(), []
#             for a in demoted + list(soft_anchors):
#                 if a and a.lower() not in seen:
#                     seen.add(a.lower())
#                     merged.append(a)
#             soft_anchors = merged[: self.max_soft_anchors + 4]
#         # Sanitize strict anchors: strip stray quotes/labels like "brand name 'FORCE"
#         # that the phrase extractor occasionally emits, so the verbatim-match filter
#         # isn't handed an unsatisfiable string.
#         _clean_strict = []
#         for a in distinctive:
#             s = re.sub(r"^(brand name|sign|text|label|plate)\s*[:'\"]*\s*", "", a.strip(), flags=re.IGNORECASE)
#             s = s.strip().strip("'\"`").strip()
#             if s:
#                 _clean_strict.append(s)
#         strict_anchors = _clean_strict
#         print(f"[3] Strict anchors S_c: {strict_anchors}")
#         print(f"    Soft  anchors G_c: {soft_anchors}")

#         # 5/6. Outer loop: find shortest faithful, then compress further
#         current_caption = caption
#         winner: Optional[Dict[str, Any]] = None
#         global_best_below: Optional[Dict[str, Any]] = None

#         for iteration in range(1, self.max_iterations + 1):
#             current_tokens = count_tokens(current_caption)
#             if current_tokens <= self.min_caption_tokens:
#                 break
#             print(f"\n[Iter {iteration}] from {current_tokens} tokens")
#             iw, best_below, iter_steps = self._find_shortest_faithful(
#                 image=image, original_caption=caption, kg_original=kg_original,
#                 source_caption=current_caption, original_tokens=original_tokens,
#                 threshold=threshold, baseline_cmp=baseline_cmp,
#                 strict_anchors=strict_anchors, soft_anchors=soft_anchors,
#                 iteration=iteration, sid=sid,
#             )
#             steps.extend(iter_steps)
#             if best_below and (global_best_below is None
#                                or best_below["scores"]["D"] < global_best_below["scores"]["D"]):
#                 global_best_below = best_below
#             if iw is None:
#                 print(f"[Iter {iteration}] no faithful candidate.")
#                 break
#             if self.require_strict_compression and iw["tokens"] >= current_tokens:
#                 print(f"[Iter {iteration}] winner not shorter; stopping.")
#                 break
#             winner = iw
#             current_caption = iw["caption"]
#             # re-inject the winner's own missing facts for the next iteration
#             soft_anchors = self._build_soft_anchors(kg_original, iw["comparison"])
#             print(f"[Iter {iteration}] winner: {iw['tokens']}t  D={iw['scores']['D']:.4f}")

#         # 7. Resolve final (never the original unless rewriter produced nothing)
#         if winner is not None:
#             final = winner
#             status = "compressed"
#         elif self.fallback_strategy == "best_candidate" and global_best_below is not None:
#             final = global_best_below
#             status = "compressed_via_fallback"
#             print(f"[FALLBACK] best below-threshold candidate D={final['scores']['D']:.4f}")
#         else:
#             # Total rewriter failure: only case where original is returned.
#             final = None
#             status = "kept_original_rewriter_failed"
#             print("[WARN] No usable candidate produced; returning original.")

#         if final is not None:
#             final_caption = final["caption"]
#             final_images = final["images"]
#             final_scores = final["scores"]
#             final_cmp = self._single_comparison(kg_original, final["kg_recon"])
#             kg_final_counts = self._kg_counts(final.get("kg_recon", {}))
#         else:
#             final_caption = caption
#             final_images = baseline_images
#             final_scores = dict(baseline_dist)
#             final_cmp = baseline_cmp
#             kg_final_counts = self._kg_counts(baseline_kgs[0]) if baseline_kgs else {}

#         compressed_tokens = count_tokens(final_caption)
#         cr = 1.0 - compressed_tokens / max(original_tokens, 1)
#         recon_path = self._save_final(image, final_caption, final_images, sid)
#         final_scores = dict(final_scores)
#         final_scores["status"] = status

#         result = CompressionResult(
#             image_id=image_id, status=status,
#             original_caption=caption, compressed_caption=final_caption,
#             original_tokens=original_tokens, compressed_tokens=compressed_tokens,
#             compression_ratio=cr, baseline_distortion=baseline_D, threshold=threshold,
#             final_distortion=float(final_scores.get("D", 0.0)),
#             final_distortion_details=final_scores,
#             baseline_kg_comparison=baseline_cmp, final_kg_comparison=final_cmp,
#             strict_anchors=list(strict_anchors), soft_anchors=list(soft_anchors),
#             quoted_anchors=list(strict_anchors), protected_anchors=list(soft_anchors),
#             steps=steps, total_time=time.time() - start,
#             final_reconstruction_path=str(recon_path) if recon_path else None,
#             baseline_reconstruction_path=str(baseline_recon_path) if baseline_recon_path else None,
#             original_image_path=str(original_image_path) if original_image_path else None,
#             kg_I_counts=kg_I_counts, kg_final_counts=kg_final_counts,
#         )
#         print("\n" + "-" * 70)
#         print(f"Result [{image_id}]: {original_tokens} -> {compressed_tokens} tokens | "
#               f"CR={cr:.2%} | D={result.final_distortion:.4f} <= delta={threshold:.4f} | {status}")
#         print(f"c*: {final_caption}")
#         print("-" * 70)
#         if self.save_intermediate:
#             self._save_result(result)
#         return result

#     def compress_dataset(self, dataset: Any, max_samples: Optional[int] = None) -> List[CompressionResult]:
#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []
#         print("\n" + "=" * 70 + f"\nProcessing {n} samples\n" + "=" * 70)
#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]
#             try:
#                 results.append(self.compress_single(
#                     image=sample["image"], caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 ))
#             except Exception as exc:
#                 print(f"[ERROR] idx={idx}: {exc}")
#                 continue
#         self._save_summary(results)
#         return results

#     # ================================================================== #
#     # Core: find the shortest faithful compression for one source caption
#     # ================================================================== #
#     def _find_shortest_faithful(
#         self, image, original_caption, kg_original, source_caption, original_tokens,
#         threshold, baseline_cmp, strict_anchors, soft_anchors, iteration, sid,
#     ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[CompressionStep]]:
#         source_tokens = count_tokens(source_caption)
#         steps: List[CompressionStep] = []
#         best_below: Optional[Dict[str, Any]] = None
#         # Soft anchors SHAPE generation (the rewriter is told to keep them) and are
#         # ENFORCED downstream by reconstruction-side KG recall (the fidelity floor:
#         # Q_KG / R_obj / R_attr). They are NOT a hard survival filter — gating
#         # candidate survival on semantic presence of many descriptive phrases kills
#         # every short candidate. So we allow MOST soft anchors to be missing here and
#         # let the floor + gate do the real faithfulness enforcement on reconstructions.
#         n_soft = len([a for a in soft_anchors if a and str(a).strip()])
#         # tolerate up to ~all-but-two soft anchors missing at the filter stage
#         max_soft_miss = max(int(round(0.75 * n_soft)), n_soft - 2) if n_soft else 0
#         # Minimum acceptable compressed length (blocks fragment captions on EVERY
#         # path: gate, structural relaxation, and fallback).
#         min_required = max(self.min_caption_tokens, self.min_target_tokens,
#                            int(round(self.min_target_ratio * original_tokens)))

#         stages = [("primary", self.budget_ratios), ("escalation", self.escalation_ratios)]
#         for stage_name, ratios in stages:
#             budgets = self._budgets_from_ratios(ratios, original_tokens, source_tokens)
#             if not budgets:
#                 continue
#             print(f"    [{stage_name}] budgets={budgets}")
#             raw = self.rewriter.rewrite(
#                 original_caption=source_caption, strict_anchors=strict_anchors,
#                 soft_anchors=soft_anchors, target_token_budgets=budgets,
#                 num_candidates=self.num_candidates_per_iteration,
#                 seed=self.seed + iteration * 100 + (0 if stage_name == "primary" else 50),
#             )
#             if not raw:
#                 continue
#             candidates = self._filter_candidates(
#                 raw, strict_anchors, soft_anchors, source_tokens, budgets,
#                 original_caption, max_soft_miss, min_required,
#             )
#             survivors = sorted([c for c in candidates if c.passed_anchor_filter],
#                                key=lambda c: c.token_count)
#             if not survivors:
#                 # Last resort: anchors already shaped generation. Let every
#                 # structurally-valid candidate through so they get scored and
#                 # the best_candidate fallback can choose by distortion. Candidates
#                 # marked with a "__...__" reason (too short / below floor / not a
#                 # sentence / not shorter) stay rejected.
#                 for c in candidates:
#                     if any(m.startswith("__") for m in c.missing_anchors):
#                         continue
#                     c.passed_anchor_filter = True
#                 survivors = sorted([c for c in candidates if c.passed_anchor_filter],
#                                    key=lambda c: c.token_count)
#                 if survivors:
#                     print(f"    [{stage_name}] anchor filter relaxed "
#                           f"(structural-only): {len(survivors)} candidate(s) will be scored")
#             if self.save_intermediate:
#                 self._save_candidates(sid, iteration, stage_name, raw, candidates, strict_anchors, soft_anchors)
#             print(f"    [{stage_name}] anchor filter: {len(survivors)}/{len(candidates)} survived")

#             for cand in survivors:  # shortest-first
#                 t0 = time.time()
#                 cand_images, cand_kgs = self._reconstruct_and_extract(
#                     cand.text, sid, f"iter{iteration}_{stage_name}_cand{cand.index}")
#                 per_seed = [
#                     self.distortion.compute_candidate_distortion(
#                         image_orig=image, image_recon=img_r, caption_orig=original_caption,
#                         caption_compressed=cand.text, kg_image_orig=kg_original,
#                         kg_image_recon=kg_r, use_soft_kg=self.use_soft_kg,
#                     )
#                     for img_r, kg_r in zip(cand_images, cand_kgs)
#                 ]
#                 scores = self._average_scores(per_seed)
#                 gate = self.distortion.evaluate_faithfulness_gate(
#                     scores, delta=threshold, tau_img=self.tau_img,
#                     tau_cap=self.tau_cap, tau_kg=self.tau_kg)
#                 floor_ok, floor_msg = self._fidelity_floor_check(scores, baseline_cmp)
#                 passed = gate["passed"] and floor_ok
#                 reason = gate["gate_log"] if floor_ok else f"{gate['gate_log']} | FLOOR: {floor_msg}"
#                 print(f"      [{'PASS' if passed else 'FAIL'}] #{cand.index} ({cand.token_count}t) {gate['gate_log']}")
#                 if not floor_ok:
#                     print(f"        FLOOR: {floor_msg}")

#                 comparison = self._single_comparison(kg_original, cand_kgs[0]) if cand_kgs else {}
#                 step = CompressionStep(
#                     iteration=iteration, stage=stage_name, candidate_index=cand.index,
#                     candidate_caption=cand.text, target_token_budget=cand.target_token_budget,
#                     caption_before=source_caption, tokens_before=source_tokens,
#                     tokens_after=cand.token_count, token_reduction=source_tokens - cand.token_count,
#                     distortion=float(scores.get("D", 1.0)), threshold=threshold,
#                     passed_threshold=passed, selected=False, reason=reason,
#                     gate_result=gate, distortion_details=scores,
#                     missing_anchors=cand.missing_anchors, time_seconds=time.time() - t0,
#                 )
#                 steps.append(step)
#                 winner_obj = {"caption": cand.text, "tokens": cand.token_count,
#                               "images": cand_images, "kg_recon": cand_kgs[0] if cand_kgs else {},
#                               "scores": scores, "comparison": comparison, "step": step}
#                 if passed:
#                     step.selected = True
#                     print(f"      [SELECTED shortest faithful] #{cand.index}")
#                     return winner_obj, best_below, steps
#                 # track best below-threshold (lowest D), within [min_required, original)
#                 if min_required <= cand.token_count < original_tokens:
#                     if best_below is None or scores.get("D", 1.0) < best_below["scores"]["D"]:
#                         best_below = winner_obj
#             print(f"    [{stage_name}] no candidate passed; "
#                   f"{'escalating' if stage_name == 'primary' else 'exhausted'}.")
#         return None, best_below, steps

#     # ================================================================== #
#     # Reconstruction + KG (shared by baseline and candidates)
#     # ================================================================== #
#     def _reconstruct_and_extract(self, caption: str, sid: str, prefix: str
#                                  ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
#         out_dir = (self.reconstruction_dir / sid) if self.save_intermediate else None
#         if out_dir:
#             out_dir.mkdir(parents=True, exist_ok=True)
#         images = self.t2i.reconstruct(caption=caption, seeds=self.t2i_seeds,
#                                       output_dir=out_dir, prefix=prefix)
#         kgs = []
#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract_with_relation_retry(img, image_id=f"{sid}_{prefix}_seed{seed}")
#             kgs.append(kg)
#             if self.save_intermediate:
#                 self._save_kg(kg, sid, f"{prefix}_seed{seed}_recon_kg")
#         return images, kgs

#     # ================================================================== #
#     # Anchors
#     # ================================================================== #
#     def _build_soft_anchors(self, kg_original: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
#         soft = derive_soft_anchors_from_kg(kg_original, max_anchors=self.max_soft_anchors)
#         # re-inject distinctive facts the reconstruction MISSED
#         missing = derive_anchors_from_missing(
#             {"missing": comparison.get("missing", set())},
#             max_anchors=self.max_soft_anchors, include_relations=True,
#         )
#         seen, merged = set(), []
#         for a in soft + missing:
#             al = a.lower().strip()
#             if not al or al in seen:
#                 continue
#             # Drop long descriptive/POV phrases that a compressed caption cannot be
#             # expected to carry verbatim (e.g. "slightly elevated roadside
#             # perspective", "visible road markings"). These are framing prose, not
#             # scene-defining facts; the rewriter still tends to keep salient ones,
#             # and KG recall on the reconstruction is what actually enforces content.
#             wc = len(al.split())
#             if wc >= 4:
#                 continue
#             merged.append(a)
#         # Keep soft-anchor count modest so it informs rather than dominates.
#         return merged[: self.max_soft_anchors]

#     # ================================================================== #
#     # Candidate filtering
#     # ================================================================== #
#     def _budgets_from_ratios(self, ratios: Sequence[float], original_tokens: int, source_tokens: int) -> List[int]:
#         budgets, seen = [], set()
#         cap = max(self.min_caption_tokens + 1, source_tokens - 1)
#         for r in ratios:
#             b = int(round(r * original_tokens))
#             b = max(self.min_caption_tokens, min(b, cap))
#             if b not in seen:
#                 seen.add(b)
#                 budgets.append(b)
#         return sorted(budgets)

#     def _filter_candidates(self, raw_candidates, strict_anchors, soft_anchors,
#                            source_tokens, budgets, original_caption, max_soft_miss,
#                            min_required_tokens: int = 0) -> List[RewriteCandidate]:
#         candidates, seen = [], set()
#         budgets = list(budgets) or [source_tokens]
#         # Graduated strict-anchor tolerance: a 24-word caption cannot carry 8
#         # plates/signs/phrases. Require keeping most when few, ~half when many.
#         n_strict = len([a for a in strict_anchors if a and str(a).strip()])
#         if n_strict == 0:
#             max_strict_miss = 0
#         elif n_strict <= 2:
#             max_strict_miss = 1
#         elif n_strict <= 5:
#             max_strict_miss = 2
#         else:
#             max_strict_miss = n_strict // 2
#         for idx, raw in enumerate(raw_candidates):
#             text = clean_caption_spacing(raw)
#             if not text or text.lower() in seen:
#                 continue
#             seen.add(text.lower())
#             tok = count_tokens(text)
#             target = min(budgets, key=lambda b: abs(b - tok))
#             cand = RewriteCandidate(idx, text, int(target), tok)

#             if self.require_strict_compression and tok >= source_tokens:
#                 cand.missing_anchors = ["__not_shorter_than_source__"]
#             elif tok < max(self.min_caption_tokens, min_required_tokens):
#                 _floor = max(self.min_caption_tokens, min_required_tokens)
#                 cand.missing_anchors = [f"__below_floor({tok}<{_floor})__"]
#             elif not _looks_like_sentence(text):
#                 cand.missing_anchors = ["__not_a_sentence__"]
#             elif original_caption and not is_substantive_rewrite(
#                     original_caption, text, self.min_rewrite_diff_ratio):
#                 cand.missing_anchors = ["__too_similar_to_original__"]
#             else:
#                 ok, missing = candidate_preserves_anchors(
#                     text, strict_anchors, soft_anchors,
#                     self.kg_match_threshold, max_soft_miss, max_strict_miss)
#                 cand.passed_anchor_filter = ok
#                 cand.missing_anchors = missing
#             candidates.append(cand)
#         return candidates

#     def _fidelity_floor_check(self, cand_scores: Dict[str, float],
#                               baseline_cmp: Dict[str, Any]) -> Tuple[bool, str]:
#         b_f1 = float(baseline_cmp.get("f1_mean", 0.0))
#         b_obj = float(baseline_cmp.get("object_recall_mean", 0.0))
#         b_attr = float(baseline_cmp.get("attribute_recall_mean", 0.0))
#         c_obj = float(cand_scores.get("R_obj", 0.0))
#         c_attr = float(cand_scores.get("R_attr", 0.0))
#         c_kg = float(cand_scores.get("Q_KG", 0.0))
#         min_f1 = max(self.min_baseline_kg_f1, b_f1 - self.kg_f1_floor_drop)
#         min_obj = max(self.min_baseline_object_recall, b_obj - self.object_recall_floor_drop)
#         min_attr = max(self.attribute_recall_min_floor, b_attr - self.attribute_recall_floor_drop)
#         if c_kg < min_f1:
#             return False, f"Q_KG={c_kg:.4f} < floor {min_f1:.4f}"
#         if c_obj < min_obj:
#             return False, f"R_obj={c_obj:.4f} < floor {min_obj:.4f}"
#         if c_attr < min_attr:
#             return False, f"R_attr={c_attr:.4f} < floor {min_attr:.4f}"
#         return True, "floors_passed"

#     # ================================================================== #
#     # KG comparison aggregation
#     # ================================================================== #
#     def _aggregate_comparisons(self, kg_original, reconstructed_kgs) -> Dict[str, Any]:
#         comps = [compare_scene_graphs(kg_original, kg_r) for kg_r in reconstructed_kgs]
#         preserved, missing, extra = set(), set(), set()
#         f1s, prec, rec, obj_r, attr_r, rel_r = [], [], [], [], [], []
#         for c in comps:
#             preserved |= set(c.get("preserved", set()))
#             missing |= set(c.get("missing", set()))
#             extra |= set(c.get("extra", set()))
#             f1s.append(c.get("f1", 0.0)); prec.append(c.get("precision", 0.0)); rec.append(c.get("recall", 0.0))
#             obj_r.append(c.get("object_recall", 0.0)); attr_r.append(c.get("attribute_recall", 0.0))
#             rel_r.append(c.get("relation_recall", 0.0))
#         m = lambda xs: float(np.mean(xs)) if xs else 0.0
#         return {
#             "preserved": preserved, "missing": missing, "extra": extra,
#             "f1_mean": m(f1s), "f1_std": float(np.std(f1s)) if f1s else 0.0,
#             "precision_mean": m(prec), "recall_mean": m(rec),
#             "object_recall_mean": m(obj_r), "attribute_recall_mean": m(attr_r), "relation_recall_mean": m(rel_r),
#             "num_preserved": len(preserved), "num_missing": len(missing), "num_extra": len(extra),
#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     def _single_comparison(self, kg_original, kg_candidate) -> Dict[str, Any]:
#         c = compare_scene_graphs(kg_original, kg_candidate)
#         return {
#             "preserved": c.get("preserved", set()), "missing": c.get("missing", set()), "extra": c.get("extra", set()),
#             "f1_mean": float(c.get("f1", 0.0)), "precision_mean": float(c.get("precision", 0.0)),
#             "recall_mean": float(c.get("recall", 0.0)),
#             "object_recall_mean": float(c.get("object_recall", 0.0)),
#             "attribute_recall_mean": float(c.get("attribute_recall", 0.0)),
#             "relation_recall_mean": float(c.get("relation_recall", 0.0)),
#             "weighted_kg_f1_hard": float(c.get("weighted_kg_f1_hard", 0.0)),
#             "weighted_kg_f1_soft": float(c.get("weighted_kg_f1_soft", 0.0)),
#             "chair_i": float(c.get("chair_i", 0.0)),
#             "num_preserved": c.get("num_preserved", 0), "num_missing": c.get("num_missing", 0),
#             "num_extra": c.get("num_extra", 0),
#             "preserved_text": c.get("preserved_text", []), "missing_text": c.get("missing_text", []),
#             "extra_text": c.get("extra_text", []),
#         }

#     @staticmethod
#     def _kg_counts(kg: Dict[str, Any]) -> Dict[str, int]:
#         return {
#             "obj": len(kg.get("objects", []) or []),
#             "attr": len(kg.get("attributes", []) or []),
#             "rel": len(kg.get("relations", []) or []),
#         }

#     def _average_scores(self, per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
#         if not per_seed:
#             return {}
#         avg: Dict[str, Any] = {}
#         for key in per_seed[0]:
#             if key.startswith("_"):
#                 avg[key] = per_seed[-1][key]
#                 continue
#             vals = [s[key] for s in per_seed
#                     if isinstance(s.get(key), (int, float, np.integer, np.floating))]
#             if vals:
#                 avg[key] = float(np.mean(vals))
#                 avg[f"{key}_std"] = float(np.std(vals))
#         return avg

#     # ================================================================== #
#     # Output / persistence
#     # ================================================================== #
#     def _save_final(self, image, final_caption, final_images, sid) -> Optional[Path]:
#         d = self.debug_dir / sid
#         d.mkdir(parents=True, exist_ok=True)
#         (d / "final_caption.txt").write_text(final_caption, encoding="utf-8")
#         if not final_images:
#             return None
#         recon = final_images[0].convert("RGB")
#         path = d / "final_reconstruction_I_star.png"
#         recon.save(path)
#         save_side_by_side(image, recon, "Original (I)", "Final Reconstruction (I*)",
#                           d / "side_by_side_original_vs_I_star.png", final_caption)
#         print(f"[Output] I* saved: {path}")
#         return path

#     def _skipped_result(self, image_id, caption, original_tokens, baseline_D, threshold,
#                         baseline_dist, baseline_cmp, sid, start) -> CompressionResult:
#         details = dict(baseline_dist); details["status"] = "skipped_low_baseline_fidelity"
#         result = CompressionResult(
#             image_id=image_id, status="skipped_low_baseline_fidelity",
#             original_caption=caption, compressed_caption=caption,
#             original_tokens=original_tokens, compressed_tokens=original_tokens,
#             compression_ratio=0.0, baseline_distortion=baseline_D, threshold=threshold,
#             final_distortion=baseline_D, final_distortion_details=details,
#             baseline_kg_comparison=baseline_cmp, final_kg_comparison=baseline_cmp,
#             strict_anchors=[], soft_anchors=[], quoted_anchors=[], protected_anchors=[],
#             steps=[], total_time=time.time() - start, final_reconstruction_path=None,
#         )
#         if self.save_intermediate:
#             self._save_result(result)
#         return result

#     def _save_kg(self, kg, image_id, name) -> None:
#         d = self.kg_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         (d / f"{safe_filename(name)}.json").write_text(
#             json.dumps(make_json_serializable(kg_to_readable_dict(kg)), indent=2, ensure_ascii=False),
#             encoding="utf-8")

#     def _save_comparison(self, comparison, image_id, name) -> None:
#         d = self.kg_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         (d / f"{safe_filename(name)}.json").write_text(
#             json.dumps(make_json_serializable(comparison), indent=2, ensure_ascii=False), encoding="utf-8")

#     def _save_candidates(self, image_id, iteration, stage, raw, filtered, strict, soft) -> None:
#         d = self.candidate_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         payload = {"iteration": iteration, "stage": stage, "strict_anchors": list(strict),
#                    "soft_anchors": list(soft), "raw_candidates": list(raw),
#                    "filtered_candidates": [asdict(c) for c in filtered]}
#         (d / f"iter{iteration}_{stage}_candidates.json").write_text(
#             json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"
#         path.write_text(json.dumps(make_json_serializable(result), indent=2, ensure_ascii=False), encoding="utf-8")
#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             return
#         statuses: Dict[str, int] = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1
#         summary = {
#             "num_samples": len(results), "status_counts": statuses,
#             "avg_compression_ratio": float(np.mean([r.compression_ratio for r in results])),
#             "avg_baseline_distortion": float(np.mean([r.baseline_distortion for r in results])),
#             "avg_final_distortion": float(np.mean([r.final_distortion for r in results])),
#             "avg_original_tokens": float(np.mean([r.original_tokens for r in results])),
#             "avg_compressed_tokens": float(np.mean([r.compressed_tokens for r in results])),
#             "avg_time_seconds": float(np.mean([r.total_time for r in results])),
#             "total_time_hours": float(np.sum([r.total_time for r in results]) / 3600.0),
#             "use_relative_threshold": self.use_relative_threshold, "epsilon": self.epsilon,
#             "tau_img": self.tau_img, "tau_cap": self.tau_cap, "tau_kg": self.tau_kg,
#         }
#         (self.result_dir / "summary.json").write_text(
#             json.dumps(make_json_serializable(summary), indent=2), encoding="utf-8")
#         print("\n" + "=" * 70)
#         print(f"SUMMARY - {len(results)} samples | {statuses}")
#         print(f"  Avg CR={summary['avg_compression_ratio']:.2%} | "
#               f"D0={summary['avg_baseline_distortion']:.4f} -> D={summary['avg_final_distortion']:.4f}")
#         print(f"  Tokens {summary['avg_original_tokens']:.1f} -> {summary['avg_compressed_tokens']:.1f} | "
#               f"{summary['total_time_hours']:.2f}h")
#         print("=" * 70)



























# """
# KG-RD Compressor
# ================

# KG-guided Rate-Distortion caption compressor (reconstruction-guided).

# Per the locked framework, BOTH the baseline and every candidate are scored
# against FLUX reconstructions, and Q_KG is computed from the reconstruction's KG
# (NOT from caption text). The decoder G is caption-only, so distortion isolates
# the caption's information content.

# Pipeline (per image)
# ---------------------
# 1. KG_I  = KG from the original image I (ground truth).
# 2. visible_text = readable in-image text (plate/brand/sign) for strict anchors.
# 3. Baseline: I_c = G(c); KG_{I_c}; D_0 = alpha*(1-Q_img_base)+gamma*(1-Q_KG_base);
#    delta = min((1+epsilon)*D_0, absolute_threshold).
# 4. Baseline gate: skip samples whose full caption already reconstructs poorly.
# 5. Anchors:
#      strict S_c = noun-phrases (KG-cross-validated) + visible text  (substring-enforced)
#      soft  G_c  = salient KG facts (framing/reflection first) + missing-from-baseline
# 6. Find the SHORTEST faithful compression:
#      - budget ladder [20,30,45,55,65]% of original tokens (one rewrite call)
#      - evaluate survivors shortest-first; first gate-pass wins (shortest faithful)
#      - if none pass, ESCALATE to higher budgets [70,78,86,94]%
#      - if still none, fall back to the best (lowest-D) shorter-than-original
#        candidate — NEVER the original caption
# 7. Outer loop (max_iterations): try to compress the winner further; re-inject the
#    winner's missing facts as soft anchors.
# 8. Save I* = G(c*) and the result.

# Gate (C1–C4, all must pass; logged per condition):
#     C1 Q_img>=tau_img  C2 Q_cap>=tau_cap  C3 Q_KG>=tau_kg  C4 D<=delta
# """

# import json
# import re
# import time
# from abc import ABC, abstractmethod
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .parse_tree import extract_strict_anchors, _shared_pruner
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     canonicalize_for_matching,
#     compare_scene_graphs,
#     derive_anchors_from_missing,
#     derive_soft_anchors_from_kg,
#     normalize_label,
#     semantic_text_similarity,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor


# # Default budget ladders (fractions of ORIGINAL caption tokens).
# _PRIMARY_RATIOS = [0.20, 0.30, 0.45, 0.55, 0.65]
# _ESCALATION_RATIOS = [0.70, 0.78, 0.86, 0.94]


# # =====================================================================
# # Data classes
# # =====================================================================

# @dataclass
# class RewriteCandidate:
#     index: int
#     text: str
#     target_token_budget: int
#     token_count: int
#     missing_anchors: List[str] = field(default_factory=list)
#     passed_anchor_filter: bool = False


# @dataclass
# class CompressionStep:
#     iteration: int
#     stage: str
#     candidate_index: int
#     candidate_caption: str
#     target_token_budget: int
#     caption_before: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     gate_result: Dict[str, Any]
#     distortion_details: Dict[str, Any]
#     missing_anchors: List[str]
#     time_seconds: float


# @dataclass
# class CompressionResult:
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     strict_anchors: List[str]
#     soft_anchors: List[str]
#     steps: List[CompressionStep]
#     total_time: float
#     final_reconstruction_path: Optional[str] = None
#     baseline_reconstruction_path: Optional[str] = None
#     original_image_path: Optional[str] = None
#     kg_I_counts: Dict[str, int] = field(default_factory=dict)
#     kg_final_counts: Dict[str, int] = field(default_factory=dict)
#     # Backward-compatible aliases (older readers expect these names):
#     protected_anchors: List[str] = field(default_factory=list)   # == soft_anchors
#     quoted_anchors: List[str] = field(default_factory=list)       # == strict_anchors


# # =====================================================================
# # Helpers
# # =====================================================================

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"\(\s+", "(", text)
#     text = re.sub(r"\s+\)", ")", text)
#     text = text.replace(" ,", ",").replace(" .", ".")
#     return text.strip()


# def safe_filename(name: str) -> str:
#     return re.sub(r"[^\w\-.]+", "_", str(name))[:180]


# def _dataclass_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if _dataclass_instance(obj):
#         return make_json_serializable(asdict(obj))
#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]
#     if isinstance(obj, tuple):
#         return list(obj)
#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "scene_context": kg.get("scene_context", {}),
#         "tuples": [list(t) for t in sorted(list(kg.get("tuples", set())), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def _token_set(s: str) -> Set[str]:
#     return {w.lower().strip(".,;:!?\"'()[]{}") for w in s.split() if w.strip()}


# def is_substantive_rewrite(original: str, candidate: str, min_diff_ratio: float = 0.15) -> bool:
#     a, b = _token_set(original), _token_set(candidate)
#     if not a or not b:
#         return True
#     union, inter = a | b, a & b
#     return (1.0 - len(inter) / len(union)) >= min_diff_ratio if union else True


# def _looks_like_sentence(text: str) -> bool:
#     """Reject bag-of-words: require at least one verb/aux (via shared spaCy)."""
#     nlp = _shared_pruner().nlp
#     if nlp is None:
#         return True  # cannot check; do not reject
#     doc = nlp(text)
#     return any(t.pos_ in ("VERB", "AUX") for t in doc)


# _QUOTE_RE = r'"([^"]+)"|\u201c([^\u201c\u201d]+)\u201d|`([^`]+)`'


# def _extract_quoted_substrings(text: str) -> List[str]:
#     out = []
#     for m in re.finditer(_QUOTE_RE, text or ""):
#         s = next((g for g in m.groups() if g), "").strip()
#         if s:
#             out.append(s)
#     return out


# def candidate_preserves_anchors(
#     candidate: str,
#     strict_anchors: Sequence[str],
#     soft_anchors: Sequence[str],
#     kg_match_threshold: float = 0.55,
#     max_soft_miss_count: int = 0,
#     max_strict_miss_count: int = 0,
# ) -> Tuple[bool, List[str]]:
#     """
#     Strict anchors should be present (substring / verbatim / token-overlap); up to
#     ``max_strict_miss_count`` may be missing (graduated tolerance — a heavily
#     compressed caption cannot carry every plate/sign/phrase). Soft anchors should
#     be present semantically; up to ``max_soft_miss_count`` may be missing.
#     Returns (passed, missing_anchor_list).
#     """
#     if not candidate:
#         return False, list(strict_anchors) + list(soft_anchors)

#     cand_low = candidate.lower()
#     cand_norm = canonicalize_for_matching(candidate)
#     cand_norm_words = set(cand_norm.split())

#     strict_missing: List[str] = []
#     for a in strict_anchors:
#         a_low = a.strip().lower()
#         if not a_low:
#             continue
#         if a_low in cand_low:
#             continue
#         # token-overlap leniency for multi-word strict anchors
#         a_norm = canonicalize_for_matching(a)
#         a_words = [w for w in a_norm.split() if len(w) > 2]
#         if a_words and all(w in cand_norm_words for w in a_words):
#             continue
#         strict_missing.append(a)

#     soft_missing: List[str] = []
#     for a in soft_anchors:
#         a_norm = canonicalize_for_matching(a)
#         if not a_norm:
#             continue
#         if a_norm in cand_norm:
#             continue
#         a_words = [w for w in a_norm.split() if len(w) > 2]
#         if a_words and any(w in cand_norm_words for w in a_words):
#             continue
#         if semantic_text_similarity(a_norm, cand_norm) >= kg_match_threshold:
#             continue
#         soft_missing.append(a)

#     if len(strict_missing) > max_strict_miss_count:
#         return False, strict_missing + soft_missing
#     if len(soft_missing) > max_soft_miss_count:
#         return False, strict_missing + soft_missing
#     return True, strict_missing + soft_missing


# def save_side_by_side(left_img: Image.Image, right_img: Image.Image,
#                       left_label: str, right_label: str, path: Path,
#                       caption: str = "") -> None:
#     left = left_img.convert("RGB").resize((512, 512))
#     right = right_img.convert("RGB").resize((512, 512))
#     canvas = Image.new("RGB", (1024, 580), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))
#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), left_label, fill="black")
#     draw.text((565, 15), right_label, fill="black")
#     if caption:
#         preview = caption[:100] + "..." if len(caption) > 100 else caption
#         draw.text((10, 532), preview, fill="black")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     canvas.save(path)


# # =====================================================================
# # Rewriter
# # =====================================================================

# class CaptionRewriter(ABC):
#     @abstractmethod
#     def rewrite(self, original_caption: str, strict_anchors: Sequence[str],
#                 soft_anchors: Sequence[str], target_token_budgets: Sequence[int],
#                 num_candidates: int, seed: Optional[int] = None) -> List[str]:
#         ...


# class QwenVLRewriter(CaptionRewriter):
#     """Text-only caption rewriter sharing the Qwen2.5-VL model (no extra GPU load)."""

#     def __init__(self, vlm_extractor: VLMSceneGraphExtractor,
#                  max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
#         self.model = vlm_extractor.model
#         self.processor = vlm_extractor.processor
#         self.device = vlm_extractor.device
#         self.max_new_tokens = int(max_new_tokens)
#         self.temperature = float(temperature)
#         self.top_p = float(top_p)

#     def _build_prompt(self, original_caption, strict_anchors, soft_anchors,
#                       target_token_budgets, num_candidates) -> str:
#         original_tokens = count_tokens(original_caption)
#         strict_block = "\n".join(f'  ({i}) "{a}"' for i, a in enumerate(strict_anchors, 1)) or "  (none)"
#         soft_block = "\n".join(f"  ({i}) {a}" for i, a in enumerate(soft_anchors, 1)) or "  (none)"
#         budgets = list(target_token_budgets)
#         min_budget = min(budgets) if budgets else max(8, original_tokens // 2)
#         budget_lines = "\n".join(f"  - Variant {i+1}: about {b} tokens"
#                                  for i, b in enumerate(budgets)) or f"  - about {min_budget} tokens"
#         return f"""You compress an image caption while preserving the visual information needed to faithfully reconstruct the image. Removing a visually important detail makes the reconstructed image look different from the original — avoid that.

# ORIGINAL CAPTION ({original_tokens} tokens):
# {original_caption}

# REQUIREMENT 1 — KEEP THESE EXACT PHRASES (verbatim, unchanged):
# These name distinctive or text-bearing elements (signs, plates, brands, framing
# devices). Every variant MUST contain each of them word-for-word.
# {strict_block}

# REQUIREMENT 2 — PRESERVE THESE FACTS (you may rephrase, but the meaning must remain):
# These are important visual facts, including how things are positioned, oriented,
# or seen (e.g. something seen through or reflected in a mirror). Every variant must
# still convey each one.
# {soft_block}

# REQUIREMENT 3 — GRAMMATICAL SENTENCES:
# Each variant is one or more complete, fluent English sentences. Never a list of
# keywords, never a fragment, never comma-separated noun phrases.

# REQUIREMENT 4 — SHORTER THAN THE ORIGINAL:
# Every variant must be clearly shorter than {original_tokens} tokens. Produce
# {num_candidates} variants of increasing length, shortest first:
# {budget_lines}

# Drop only redundant or non-visual wording. When in doubt about a distinctive or
# spatial detail, keep it.

# OUTPUT — return ONLY a JSON array of strings, shortest first. No markdown, no
# numbering, no commentary:
# ["shortest variant", "...", "longest variant"]
# """

#     def _parse_json_array(self, text: str) -> List[str]:
#         if not isinstance(text, str):
#             return []
#         text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"```", "", text).strip()
#         match = re.search(r"\[.*\]", text, flags=re.DOTALL)
#         for candidate in (text, match.group(0) if match else None):
#             if not candidate:
#                 continue
#             try:
#                 parsed = json.loads(candidate)
#                 if isinstance(parsed, list):
#                     return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#             except Exception:
#                 continue
#         # Last resort: pull quoted strings.
#         return [s.strip() for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text) if s.strip()]

#     @torch.no_grad()
#     def _generate_text(self, prompt: str) -> str:
#         messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
#         text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         try:
#             inputs = self.processor(text=[text_input], padding=True, return_tensors="pt")
#         except Exception:
#             inputs = self.processor.tokenizer(text_input, padding=True, return_tensors="pt")
#         inputs = inputs.to(self.model.device) if hasattr(inputs, "to") else {k: v.to(self.model.device) for k, v in inputs.items()}
#         generated_ids = self.model.generate(
#             **inputs, max_new_tokens=self.max_new_tokens,
#             do_sample=True, temperature=self.temperature, top_p=self.top_p,
#         )
#         input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
#         trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]
#         decoder = getattr(self.processor, "batch_decode", None) or self.processor.tokenizer.batch_decode
#         return decoder(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

#     def rewrite(self, original_caption, strict_anchors, soft_anchors,
#                 target_token_budgets, num_candidates, seed: Optional[int] = None) -> List[str]:
#         if seed is not None:
#             torch.manual_seed(int(seed))
#         prompt = self._build_prompt(original_caption, strict_anchors, soft_anchors,
#                                     target_token_budgets, num_candidates)
#         try:
#             raw = self._generate_text(prompt)
#         except Exception as exc:
#             print(f"[Rewriter] Generation failed: {exc}")
#             return []
#         cands = [clean_caption_spacing(c) for c in self._parse_json_array(raw) if c]
#         seen, unique = set(), []
#         for c in cands:
#             if c.lower() not in seen:
#                 seen.add(c.lower())
#                 unique.append(c)
#         return unique


# # =====================================================================
# # Main compressor
# # =====================================================================

# class KGRDCompressor:
#     """KG-guided Rate-Distortion caption compressor (see module docstring)."""

#     def __init__(
#         self,
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,
#         t2i_model: str = "black-forest-labs/FLUX.1-dev",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "bfloat16",
#         t2i_steps: int = 20,
#         t2i_guidance: float = 3.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,
#         alpha: float = 0.4, beta: float = 0.2, gamma: float = 0.4,
#         clip_model: str = "ViT-L-14", clip_pretrained: str = "openai",
#         dino_model: str = "dinov2_vitb14", lpips_net: str = "alex",
#         bertscore_model: str = "microsoft/deberta-xlarge-mnli",
#         metric_device: str = "auto",
#         kg_object_weight: float = 0.4, kg_attribute_weight: float = 0.3, kg_relation_weight: float = 0.3,
#         use_soft_kg: bool = True,
#         tau_img: float = 0.50, tau_cap: float = 0.55, tau_kg: float = 0.45,
#         use_relative_threshold: bool = True, epsilon: float = 0.10, absolute_threshold: float = 0.25,
#         min_caption_tokens: int = 5,
#         max_iterations: int = 2,
#         num_candidates_per_iteration: int = 5,
#         budget_ratios: Optional[List[float]] = None,
#         escalation_ratios: Optional[List[float]] = None,
#         require_strict_compression: bool = True,
#         min_baseline_kg_f1: float = 0.05, min_baseline_object_recall: float = 0.10,
#         kg_f1_floor_drop: float = 0.08, object_recall_floor_drop: float = 0.05,
#         attribute_recall_floor_drop: float = 0.12, attribute_recall_min_floor: float = 0.10,
#         max_strict_anchors: int = 8, max_soft_anchors: int = 6,
#         kg_match_threshold: float = 0.55, soft_anchor_max_miss_fraction: float = 0.40,
#         min_rewrite_diff_ratio: float = 0.15,
#         fallback_strategy: str = "best_candidate",
#         rewriter: Optional[CaptionRewriter] = None,
#         rewriter_max_new_tokens: int = 512, rewriter_temperature: float = 0.7, rewriter_top_p: float = 0.9,
#         save_intermediate: bool = True, output_dir: str = "./outputs", seed: int = 42,
#     ):
#         self.t2i_seeds = t2i_seeds or [42]
#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)
#         self.use_soft_kg = bool(use_soft_kg)
#         self.tau_img, self.tau_cap, self.tau_kg = float(tau_img), float(tau_cap), float(tau_kg)
#         self.min_caption_tokens = int(min_caption_tokens)
#         self.max_iterations = int(max_iterations)
#         self.num_candidates_per_iteration = int(num_candidates_per_iteration)
#         self.budget_ratios = list(budget_ratios) if budget_ratios else list(_PRIMARY_RATIOS)
#         self.escalation_ratios = list(escalation_ratios) if escalation_ratios else list(_ESCALATION_RATIOS)
#         self.require_strict_compression = bool(require_strict_compression)
#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)
#         self.kg_f1_floor_drop = float(kg_f1_floor_drop)
#         self.object_recall_floor_drop = float(object_recall_floor_drop)
#         self.attribute_recall_floor_drop = float(attribute_recall_floor_drop)
#         self.attribute_recall_min_floor = float(attribute_recall_min_floor)
#         self.max_strict_anchors = int(max_strict_anchors)
#         self.max_soft_anchors = int(max_soft_anchors)
#         self.kg_match_threshold = float(kg_match_threshold)
#         self.soft_anchor_max_miss_fraction = float(soft_anchor_max_miss_fraction)
#         self.min_rewrite_diff_ratio = float(min_rewrite_diff_ratio)
#         if fallback_strategy not in {"keep_original", "best_candidate"}:
#             raise ValueError(f"Invalid fallback_strategy: {fallback_strategy}")
#         self.fallback_strategy = fallback_strategy
#         self.save_intermediate = bool(save_intermediate)
#         self.seed = int(seed)

#         self.output_dir = Path(output_dir)
#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"
#         self.candidate_debug_dir = self.output_dir / "candidates"
#         self.debug_dir = self.output_dir / "debug_samples"
#         for d in (self.result_dir, self.kg_debug_dir, self.candidate_debug_dir, self.debug_dir):
#             d.mkdir(parents=True, exist_ok=True)

#         print("=" * 70 + "\nKG-RD Compressor: initializing\n" + "=" * 70)
#         print("\n[1/3] Loading VLM scene-graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model, device=scene_graph_device, dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens, temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample, max_objects=max_objects, max_relations=max_relations,
#         )
#         print("\n[2/3] Loading T2I reconstructor (text-only)...")
#         self.t2i = T2IReconstructor(
#             model_name=t2i_model, device=t2i_device, dtype=t2i_dtype,
#             num_inference_steps=t2i_steps, guidance_scale=t2i_guidance,
#             image_size=t2i_image_size, img2img_mode=False,
#         )
#         print("\n[3/3] Loading distortion metric (CLIP + DINO + LPIPS + BERTScore)...")
#         self.distortion = DistortionMetric(
#             alpha=alpha, beta=beta, gamma=gamma,
#             clip_model_name=clip_model, clip_pretrained=clip_pretrained,
#             dino_model_name=dino_model, lpips_net=lpips_net,
#             bertscore_model=bertscore_model, device=metric_device,
#             kg_object_weight=kg_object_weight, kg_attribute_weight=kg_attribute_weight,
#             kg_relation_weight=kg_relation_weight,
#         )
#         self.rewriter = rewriter or QwenVLRewriter(
#             vlm_extractor=self.sg_extractor, max_new_tokens=rewriter_max_new_tokens,
#             temperature=rewriter_temperature, top_p=rewriter_top_p,
#         )
#         print("\n" + "=" * 70)
#         print(f"Ready. taus: img={self.tau_img} cap={self.tau_cap} kg={self.tau_kg} | "
#               f"eps={self.epsilon} abs<= {self.absolute_threshold} | use_soft_kg={self.use_soft_kg}")
#         print(f"Budget ladder: {self.budget_ratios}  escalation: {self.escalation_ratios}")
#         print(f"candidates/iter={self.num_candidates_per_iteration} max_iters={self.max_iterations} "
#               f"fallback={self.fallback_strategy}")
#         print("=" * 70 + "\n")

#     # ------------------------------------------------------------------ #
#     @classmethod
#     def from_config(cls, config: Dict[str, Any], rewriter: Optional[CaptionRewriter] = None) -> "KGRDCompressor":
#         sg = config["scene_graph"]["vlm"]
#         t = config["t2i"]
#         d = config["distortion"]; w = d["weights"]; kg = d.get("kg", {})
#         g = config["gate"]; comp = config["compression"]; rw = config.get("rewriter", {})
#         bg = config.get("baseline_gate", {}); ff = config.get("fidelity_floors", {})
#         anc = config.get("anchors", {}); fb = config.get("fallback", {})
#         paths = config.get("paths", {}); ev = config.get("evaluation", {})
#         return cls(
#             scene_graph_model=sg["model"], scene_graph_device=sg["device"], scene_graph_dtype=sg["dtype"],
#             scene_graph_max_new_tokens=sg.get("max_new_tokens", 1024),
#             scene_graph_temperature=sg.get("temperature", 0.0), scene_graph_do_sample=sg.get("do_sample", False),
#             max_objects=config["scene_graph"].get("max_objects", 50),
#             max_relations=config["scene_graph"].get("max_relations", 80),
#             t2i_model=t["model"], t2i_device=t["device"], t2i_dtype=t["dtype"],
#             t2i_steps=t.get("num_inference_steps", 20), t2i_guidance=t.get("guidance_scale", 3.5),
#             t2i_image_size=t.get("image_size", 512), t2i_seeds=t.get("seeds", [42]),
#             alpha=w["alpha_img"], beta=w["beta_cap"], gamma=w["gamma_kg"],
#             clip_model=d.get("clip", {}).get("model", "ViT-L-14"),
#             clip_pretrained=d.get("clip", {}).get("pretrained", "openai"),
#             dino_model=d.get("dino", {}).get("model", "dinov2_vitb14"),
#             lpips_net=d.get("lpips", {}).get("net", "alex"),
#             bertscore_model=d.get("bertscore", {}).get("model", "microsoft/deberta-xlarge-mnli"),
#             metric_device=d.get("clip", {}).get("device", "cuda:0"),
#             kg_object_weight=kg.get("object_weight", 0.4), kg_attribute_weight=kg.get("attribute_weight", 0.3),
#             kg_relation_weight=kg.get("relation_weight", 0.3),
#             use_soft_kg=d.get("use_soft_kg", True),
#             tau_img=g["tau_img"], tau_cap=g["tau_cap"], tau_kg=g["tau_kg"],
#             use_relative_threshold=d.get("use_relative_threshold", True),
#             epsilon=d.get("epsilon", 0.10), absolute_threshold=d.get("absolute_threshold", 0.25),
#             min_caption_tokens=comp.get("min_caption_tokens", 5),
#             max_iterations=comp.get("max_iterations", 2),
#             num_candidates_per_iteration=comp.get("num_candidates_per_iteration", 5),
#             budget_ratios=comp.get("budget_ratios", _PRIMARY_RATIOS),
#             escalation_ratios=comp.get("escalation_ratios", _ESCALATION_RATIOS),
#             require_strict_compression=comp.get("require_strict_compression", True),
#             min_baseline_kg_f1=bg.get("min_baseline_kg_f1", 0.05),
#             min_baseline_object_recall=bg.get("min_baseline_object_recall", 0.10),
#             kg_f1_floor_drop=ff.get("kg_f1_floor_drop", 0.08),
#             object_recall_floor_drop=ff.get("object_recall_floor_drop", 0.05),
#             attribute_recall_floor_drop=ff.get("attribute_recall_floor_drop", 0.12),
#             attribute_recall_min_floor=ff.get("attribute_recall_min_floor", 0.10),
#             max_soft_anchors=anc.get("max_kg_anchors", 6),
#             kg_match_threshold=anc.get("kg_match_threshold", 0.55),
#             fallback_strategy=fb.get("strategy", "best_candidate"),
#             rewriter=rewriter,
#             rewriter_max_new_tokens=rw.get("max_new_tokens", 512),
#             rewriter_temperature=rw.get("temperature", 0.7), rewriter_top_p=rw.get("top_p", 0.9),
#             save_intermediate=ev.get("save_intermediate", True),
#             output_dir=paths.get("output_dir", "./outputs"),
#             seed=config.get("seed", 42),
#         )

#     # ================================================================== #
#     # Public API
#     # ================================================================== #
#     def compress_single(self, image: Image.Image, caption: str, image_id: str = "unknown") -> CompressionResult:
#         start = time.time()
#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")
#         image = image.convert("RGB") if image.mode != "RGB" else image
#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)
#         if original_tokens <= self.min_caption_tokens:
#             raise ValueError(f"Caption has only {original_tokens} tokens (min {self.min_caption_tokens}).")
#         sid = safe_filename(image_id)
#         steps: List[CompressionStep] = []

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | tokens={original_tokens}")
#         print("-" * 70)

#         # 1. KG_I + visible text
#         print("[1] Extracting KG_I from original image...")
#         kg_original = self.sg_extractor.extract_with_relation_retry(image, image_id=image_id)
#         visible_text = self.sg_extractor.extract_visible_text(image)
#         print(f"    KG_I: {len(kg_original.get('objects', []))} obj / "
#               f"{len(kg_original.get('attributes', []))} attr / {len(kg_original.get('relations', []))} rel; "
#               f"visible_text={[v['text'] for v in visible_text]}")
#         if self.save_intermediate:
#             self._save_kg(kg_original, sid, "kg_original_image")
#         kg_I_counts = self._kg_counts(kg_original)
#         # Save the original image for the Excel report.
#         original_image_path = None
#         if self.save_intermediate:
#             (self.debug_dir / sid).mkdir(parents=True, exist_ok=True)
#             original_image_path = self.debug_dir / sid / "original_image.png"
#             image.save(original_image_path)

#         # 2. Baseline reconstruction + D_0 + threshold
#         print("[2] Baseline reconstruction G(c)...")
#         baseline_images, baseline_kgs = self._reconstruct_and_extract(caption, sid, "baseline")
#         baseline_recon_path = (self.reconstruction_dir / sid / f"baseline_seed{self.t2i_seeds[0]}.png")
#         baseline_recon_path = baseline_recon_path if baseline_recon_path.exists() else None
#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image, images_recon=baseline_images,
#             kg_original=kg_original, kgs_reconstructed=baseline_kgs,
#         )
#         baseline_D = float(baseline_dist.get("D", 1.0))
#         threshold = (self.distortion.compute_relative_threshold(
#             baseline_D, self.epsilon, self.absolute_threshold)
#             if self.use_relative_threshold else self.absolute_threshold)
#         baseline_cmp = self._aggregate_comparisons(kg_original, baseline_kgs)
#         print(f"    D_0={baseline_D:.4f}  delta={threshold:.4f}  "
#               f"(baseline f1={baseline_cmp['f1_mean']:.3f} obj_rec={baseline_cmp['object_recall_mean']:.3f})")
#         if self.save_intermediate and baseline_images:
#             save_side_by_side(image, baseline_images[0], "Original (I)", "Baseline G(c)",
#                               self.debug_dir / sid / "side_by_side_original_vs_baseline.png", caption)
#             self._save_comparison(baseline_cmp, sid, "baseline_kg_comparison")

#         # 3. Baseline gate
#         if (baseline_cmp["f1_mean"] < self.min_baseline_kg_f1
#                 or baseline_cmp["object_recall_mean"] < self.min_baseline_object_recall):
#             print("[SKIP] Baseline fidelity below threshold.")
#             return self._skipped_result(image_id, caption, original_tokens, baseline_D,
#                                         threshold, baseline_dist, baseline_cmp, sid, start)

#         # 4. Anchors
#         strict_anchors = extract_strict_anchors(
#             caption, kg=kg_original, visible_text=visible_text,
#             kg_match_threshold=self.kg_match_threshold, max_anchors=self.max_strict_anchors,
#         )
#         soft_anchors = self._build_soft_anchors(kg_original, baseline_cmp)
#         print(f"[3] Strict anchors S_c: {strict_anchors}")
#         print(f"    Soft  anchors G_c: {soft_anchors}")

#         # 5/6. Outer loop: find shortest faithful, then compress further
#         current_caption = caption
#         winner: Optional[Dict[str, Any]] = None
#         global_best_below: Optional[Dict[str, Any]] = None

#         for iteration in range(1, self.max_iterations + 1):
#             current_tokens = count_tokens(current_caption)
#             if current_tokens <= self.min_caption_tokens:
#                 break
#             print(f"\n[Iter {iteration}] from {current_tokens} tokens")
#             iw, best_below, iter_steps = self._find_shortest_faithful(
#                 image=image, original_caption=caption, kg_original=kg_original,
#                 source_caption=current_caption, original_tokens=original_tokens,
#                 threshold=threshold, baseline_cmp=baseline_cmp,
#                 strict_anchors=strict_anchors, soft_anchors=soft_anchors,
#                 iteration=iteration, sid=sid,
#             )
#             steps.extend(iter_steps)
#             if best_below and (global_best_below is None
#                                or best_below["scores"]["D"] < global_best_below["scores"]["D"]):
#                 global_best_below = best_below
#             if iw is None:
#                 print(f"[Iter {iteration}] no faithful candidate.")
#                 break
#             if self.require_strict_compression and iw["tokens"] >= current_tokens:
#                 print(f"[Iter {iteration}] winner not shorter; stopping.")
#                 break
#             winner = iw
#             current_caption = iw["caption"]
#             # re-inject the winner's own missing facts for the next iteration
#             soft_anchors = self._build_soft_anchors(kg_original, iw["comparison"])
#             print(f"[Iter {iteration}] winner: {iw['tokens']}t  D={iw['scores']['D']:.4f}")

#         # 7. Resolve final (never the original unless rewriter produced nothing)
#         if winner is not None:
#             final = winner
#             status = "compressed"
#         elif self.fallback_strategy == "best_candidate" and global_best_below is not None:
#             final = global_best_below
#             status = "compressed_via_fallback"
#             print(f"[FALLBACK] best below-threshold candidate D={final['scores']['D']:.4f}")
#         else:
#             # Total rewriter failure: only case where original is returned.
#             final = None
#             status = "kept_original_rewriter_failed"
#             print("[WARN] No usable candidate produced; returning original.")

#         if final is not None:
#             final_caption = final["caption"]
#             final_images = final["images"]
#             final_scores = final["scores"]
#             final_cmp = self._single_comparison(kg_original, final["kg_recon"])
#             kg_final_counts = self._kg_counts(final.get("kg_recon", {}))
#         else:
#             final_caption = caption
#             final_images = baseline_images
#             final_scores = dict(baseline_dist)
#             final_cmp = baseline_cmp
#             kg_final_counts = self._kg_counts(baseline_kgs[0]) if baseline_kgs else {}

#         compressed_tokens = count_tokens(final_caption)
#         cr = 1.0 - compressed_tokens / max(original_tokens, 1)
#         recon_path = self._save_final(image, final_caption, final_images, sid)
#         final_scores = dict(final_scores)
#         final_scores["status"] = status

#         result = CompressionResult(
#             image_id=image_id, status=status,
#             original_caption=caption, compressed_caption=final_caption,
#             original_tokens=original_tokens, compressed_tokens=compressed_tokens,
#             compression_ratio=cr, baseline_distortion=baseline_D, threshold=threshold,
#             final_distortion=float(final_scores.get("D", 0.0)),
#             final_distortion_details=final_scores,
#             baseline_kg_comparison=baseline_cmp, final_kg_comparison=final_cmp,
#             strict_anchors=list(strict_anchors), soft_anchors=list(soft_anchors),
#             quoted_anchors=list(strict_anchors), protected_anchors=list(soft_anchors),
#             steps=steps, total_time=time.time() - start,
#             final_reconstruction_path=str(recon_path) if recon_path else None,
#             baseline_reconstruction_path=str(baseline_recon_path) if baseline_recon_path else None,
#             original_image_path=str(original_image_path) if original_image_path else None,
#             kg_I_counts=kg_I_counts, kg_final_counts=kg_final_counts,
#         )
#         print("\n" + "-" * 70)
#         print(f"Result [{image_id}]: {original_tokens} -> {compressed_tokens} tokens | "
#               f"CR={cr:.2%} | D={result.final_distortion:.4f} <= delta={threshold:.4f} | {status}")
#         print(f"c*: {final_caption}")
#         print("-" * 70)
#         if self.save_intermediate:
#             self._save_result(result)
#         return result

#     def compress_dataset(self, dataset: Any, max_samples: Optional[int] = None) -> List[CompressionResult]:
#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []
#         print("\n" + "=" * 70 + f"\nProcessing {n} samples\n" + "=" * 70)
#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]
#             try:
#                 results.append(self.compress_single(
#                     image=sample["image"], caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 ))
#             except Exception as exc:
#                 print(f"[ERROR] idx={idx}: {exc}")
#                 continue
#         self._save_summary(results)
#         return results

#     # ================================================================== #
#     # Core: find the shortest faithful compression for one source caption
#     # ================================================================== #
#     def _find_shortest_faithful(
#         self, image, original_caption, kg_original, source_caption, original_tokens,
#         threshold, baseline_cmp, strict_anchors, soft_anchors, iteration, sid,
#     ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[CompressionStep]]:
#         source_tokens = count_tokens(source_caption)
#         steps: List[CompressionStep] = []
#         best_below: Optional[Dict[str, Any]] = None
#         max_soft_miss = int(round(self.soft_anchor_max_miss_fraction * len(soft_anchors)))

#         stages = [("primary", self.budget_ratios), ("escalation", self.escalation_ratios)]
#         for stage_name, ratios in stages:
#             budgets = self._budgets_from_ratios(ratios, original_tokens, source_tokens)
#             if not budgets:
#                 continue
#             print(f"    [{stage_name}] budgets={budgets}")
#             raw = self.rewriter.rewrite(
#                 original_caption=source_caption, strict_anchors=strict_anchors,
#                 soft_anchors=soft_anchors, target_token_budgets=budgets,
#                 num_candidates=self.num_candidates_per_iteration,
#                 seed=self.seed + iteration * 100 + (0 if stage_name == "primary" else 50),
#             )
#             if not raw:
#                 continue
#             candidates = self._filter_candidates(
#                 raw, strict_anchors, soft_anchors, source_tokens, budgets,
#                 original_caption, max_soft_miss,
#             )
#             survivors = sorted([c for c in candidates if c.passed_anchor_filter],
#                                key=lambda c: c.token_count)
#             if not survivors:
#                 # Last resort: anchors already shaped generation. Let every
#                 # structurally-valid candidate through so they get scored and
#                 # the best_candidate fallback can choose by distortion.
#                 for c in candidates:
#                     if any(m.startswith("__") for m in c.missing_anchors):
#                         continue  # structurally invalid (too short / not a sentence / not shorter)
#                     c.passed_anchor_filter = True
#                 survivors = sorted([c for c in candidates if c.passed_anchor_filter],
#                                    key=lambda c: c.token_count)
#                 if survivors:
#                     print(f"    [{stage_name}] anchor filter relaxed "
#                           f"(structural-only): {len(survivors)} candidate(s) will be scored")
#             if self.save_intermediate:
#                 self._save_candidates(sid, iteration, stage_name, raw, candidates, strict_anchors, soft_anchors)
#             print(f"    [{stage_name}] anchor filter: {len(survivors)}/{len(candidates)} survived")

#             for cand in survivors:  # shortest-first
#                 t0 = time.time()
#                 cand_images, cand_kgs = self._reconstruct_and_extract(
#                     cand.text, sid, f"iter{iteration}_{stage_name}_cand{cand.index}")
#                 per_seed = [
#                     self.distortion.compute_candidate_distortion(
#                         image_orig=image, image_recon=img_r, caption_orig=original_caption,
#                         caption_compressed=cand.text, kg_image_orig=kg_original,
#                         kg_image_recon=kg_r, use_soft_kg=self.use_soft_kg,
#                     )
#                     for img_r, kg_r in zip(cand_images, cand_kgs)
#                 ]
#                 scores = self._average_scores(per_seed)
#                 gate = self.distortion.evaluate_faithfulness_gate(
#                     scores, delta=threshold, tau_img=self.tau_img,
#                     tau_cap=self.tau_cap, tau_kg=self.tau_kg)
#                 floor_ok, floor_msg = self._fidelity_floor_check(scores, baseline_cmp)
#                 passed = gate["passed"] and floor_ok
#                 reason = gate["gate_log"] if floor_ok else f"{gate['gate_log']} | FLOOR: {floor_msg}"
#                 print(f"      [{'PASS' if passed else 'FAIL'}] #{cand.index} ({cand.token_count}t) {gate['gate_log']}")
#                 if not floor_ok:
#                     print(f"        FLOOR: {floor_msg}")

#                 comparison = self._single_comparison(kg_original, cand_kgs[0]) if cand_kgs else {}
#                 step = CompressionStep(
#                     iteration=iteration, stage=stage_name, candidate_index=cand.index,
#                     candidate_caption=cand.text, target_token_budget=cand.target_token_budget,
#                     caption_before=source_caption, tokens_before=source_tokens,
#                     tokens_after=cand.token_count, token_reduction=source_tokens - cand.token_count,
#                     distortion=float(scores.get("D", 1.0)), threshold=threshold,
#                     passed_threshold=passed, selected=False, reason=reason,
#                     gate_result=gate, distortion_details=scores,
#                     missing_anchors=cand.missing_anchors, time_seconds=time.time() - t0,
#                 )
#                 steps.append(step)
#                 winner_obj = {"caption": cand.text, "tokens": cand.token_count,
#                               "images": cand_images, "kg_recon": cand_kgs[0] if cand_kgs else {},
#                               "scores": scores, "comparison": comparison, "step": step}
#                 if passed:
#                     step.selected = True
#                     print(f"      [SELECTED shortest faithful] #{cand.index}")
#                     return winner_obj, best_below, steps
#                 # track best below-threshold (lowest D), shorter than original
#                 if cand.token_count < original_tokens:
#                     if best_below is None or scores.get("D", 1.0) < best_below["scores"]["D"]:
#                         best_below = winner_obj
#             print(f"    [{stage_name}] no candidate passed; "
#                   f"{'escalating' if stage_name == 'primary' else 'exhausted'}.")
#         return None, best_below, steps

#     # ================================================================== #
#     # Reconstruction + KG (shared by baseline and candidates)
#     # ================================================================== #
#     def _reconstruct_and_extract(self, caption: str, sid: str, prefix: str
#                                  ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
#         out_dir = (self.reconstruction_dir / sid) if self.save_intermediate else None
#         if out_dir:
#             out_dir.mkdir(parents=True, exist_ok=True)
#         images = self.t2i.reconstruct(caption=caption, seeds=self.t2i_seeds,
#                                       output_dir=out_dir, prefix=prefix)
#         kgs = []
#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract_with_relation_retry(img, image_id=f"{sid}_{prefix}_seed{seed}")
#             kgs.append(kg)
#             if self.save_intermediate:
#                 self._save_kg(kg, sid, f"{prefix}_seed{seed}_recon_kg")
#         return images, kgs

#     # ================================================================== #
#     # Anchors
#     # ================================================================== #
#     def _build_soft_anchors(self, kg_original: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
#         soft = derive_soft_anchors_from_kg(kg_original, max_anchors=self.max_soft_anchors)
#         # re-inject distinctive facts the reconstruction MISSED
#         missing = derive_anchors_from_missing(
#             {"missing": comparison.get("missing", set())},
#             max_anchors=self.max_soft_anchors, include_relations=True,
#         )
#         seen, merged = set(), []
#         for a in soft + missing:
#             if a.lower() not in seen:
#                 seen.add(a.lower())
#                 merged.append(a)
#         return merged[: self.max_soft_anchors + 2]

#     # ================================================================== #
#     # Candidate filtering
#     # ================================================================== #
#     def _budgets_from_ratios(self, ratios: Sequence[float], original_tokens: int, source_tokens: int) -> List[int]:
#         budgets, seen = [], set()
#         cap = max(self.min_caption_tokens + 1, source_tokens - 1)
#         for r in ratios:
#             b = int(round(r * original_tokens))
#             b = max(self.min_caption_tokens, min(b, cap))
#             if b not in seen:
#                 seen.add(b)
#                 budgets.append(b)
#         return sorted(budgets)

#     def _filter_candidates(self, raw_candidates, strict_anchors, soft_anchors,
#                            source_tokens, budgets, original_caption, max_soft_miss) -> List[RewriteCandidate]:
#         candidates, seen = [], set()
#         budgets = list(budgets) or [source_tokens]
#         # Graduated strict-anchor tolerance: a 24-word caption cannot carry 8
#         # plates/signs/phrases. Require keeping most when few, ~half when many.
#         n_strict = len([a for a in strict_anchors if a and str(a).strip()])
#         if n_strict == 0:
#             max_strict_miss = 0
#         elif n_strict <= 2:
#             max_strict_miss = 1
#         elif n_strict <= 5:
#             max_strict_miss = 2
#         else:
#             max_strict_miss = n_strict // 2
#         for idx, raw in enumerate(raw_candidates):
#             text = clean_caption_spacing(raw)
#             if not text or text.lower() in seen:
#                 continue
#             seen.add(text.lower())
#             tok = count_tokens(text)
#             target = min(budgets, key=lambda b: abs(b - tok))
#             cand = RewriteCandidate(idx, text, int(target), tok)

#             if self.require_strict_compression and tok >= source_tokens:
#                 cand.missing_anchors = ["__not_shorter_than_source__"]
#             elif tok < self.min_caption_tokens:
#                 cand.missing_anchors = [f"__below_min({tok})__"]
#             elif not _looks_like_sentence(text):
#                 cand.missing_anchors = ["__not_a_sentence__"]
#             elif original_caption and not is_substantive_rewrite(
#                     original_caption, text, self.min_rewrite_diff_ratio):
#                 cand.missing_anchors = ["__too_similar_to_original__"]
#             else:
#                 ok, missing = candidate_preserves_anchors(
#                     text, strict_anchors, soft_anchors,
#                     self.kg_match_threshold, max_soft_miss, max_strict_miss)
#                 cand.passed_anchor_filter = ok
#                 cand.missing_anchors = missing
#             candidates.append(cand)
#         return candidates

#     def _fidelity_floor_check(self, cand_scores: Dict[str, float],
#                               baseline_cmp: Dict[str, Any]) -> Tuple[bool, str]:
#         b_f1 = float(baseline_cmp.get("f1_mean", 0.0))
#         b_obj = float(baseline_cmp.get("object_recall_mean", 0.0))
#         b_attr = float(baseline_cmp.get("attribute_recall_mean", 0.0))
#         c_obj = float(cand_scores.get("R_obj", 0.0))
#         c_attr = float(cand_scores.get("R_attr", 0.0))
#         c_kg = float(cand_scores.get("Q_KG", 0.0))
#         min_f1 = max(self.min_baseline_kg_f1, b_f1 - self.kg_f1_floor_drop)
#         min_obj = max(self.min_baseline_object_recall, b_obj - self.object_recall_floor_drop)
#         min_attr = max(self.attribute_recall_min_floor, b_attr - self.attribute_recall_floor_drop)
#         if c_kg < min_f1:
#             return False, f"Q_KG={c_kg:.4f} < floor {min_f1:.4f}"
#         if c_obj < min_obj:
#             return False, f"R_obj={c_obj:.4f} < floor {min_obj:.4f}"
#         if c_attr < min_attr:
#             return False, f"R_attr={c_attr:.4f} < floor {min_attr:.4f}"
#         return True, "floors_passed"

#     # ================================================================== #
#     # KG comparison aggregation
#     # ================================================================== #
#     def _aggregate_comparisons(self, kg_original, reconstructed_kgs) -> Dict[str, Any]:
#         comps = [compare_scene_graphs(kg_original, kg_r) for kg_r in reconstructed_kgs]
#         preserved, missing, extra = set(), set(), set()
#         f1s, prec, rec, obj_r, attr_r, rel_r = [], [], [], [], [], []
#         for c in comps:
#             preserved |= set(c.get("preserved", set()))
#             missing |= set(c.get("missing", set()))
#             extra |= set(c.get("extra", set()))
#             f1s.append(c.get("f1", 0.0)); prec.append(c.get("precision", 0.0)); rec.append(c.get("recall", 0.0))
#             obj_r.append(c.get("object_recall", 0.0)); attr_r.append(c.get("attribute_recall", 0.0))
#             rel_r.append(c.get("relation_recall", 0.0))
#         m = lambda xs: float(np.mean(xs)) if xs else 0.0
#         return {
#             "preserved": preserved, "missing": missing, "extra": extra,
#             "f1_mean": m(f1s), "f1_std": float(np.std(f1s)) if f1s else 0.0,
#             "precision_mean": m(prec), "recall_mean": m(rec),
#             "object_recall_mean": m(obj_r), "attribute_recall_mean": m(attr_r), "relation_recall_mean": m(rel_r),
#             "num_preserved": len(preserved), "num_missing": len(missing), "num_extra": len(extra),
#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     def _single_comparison(self, kg_original, kg_candidate) -> Dict[str, Any]:
#         c = compare_scene_graphs(kg_original, kg_candidate)
#         return {
#             "preserved": c.get("preserved", set()), "missing": c.get("missing", set()), "extra": c.get("extra", set()),
#             "f1_mean": float(c.get("f1", 0.0)), "precision_mean": float(c.get("precision", 0.0)),
#             "recall_mean": float(c.get("recall", 0.0)),
#             "object_recall_mean": float(c.get("object_recall", 0.0)),
#             "attribute_recall_mean": float(c.get("attribute_recall", 0.0)),
#             "relation_recall_mean": float(c.get("relation_recall", 0.0)),
#             "weighted_kg_f1_hard": float(c.get("weighted_kg_f1_hard", 0.0)),
#             "weighted_kg_f1_soft": float(c.get("weighted_kg_f1_soft", 0.0)),
#             "chair_i": float(c.get("chair_i", 0.0)),
#             "num_preserved": c.get("num_preserved", 0), "num_missing": c.get("num_missing", 0),
#             "num_extra": c.get("num_extra", 0),
#             "preserved_text": c.get("preserved_text", []), "missing_text": c.get("missing_text", []),
#             "extra_text": c.get("extra_text", []),
#         }

#     @staticmethod
#     def _kg_counts(kg: Dict[str, Any]) -> Dict[str, int]:
#         return {
#             "obj": len(kg.get("objects", []) or []),
#             "attr": len(kg.get("attributes", []) or []),
#             "rel": len(kg.get("relations", []) or []),
#         }

#     def _average_scores(self, per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
#         if not per_seed:
#             return {}
#         avg: Dict[str, Any] = {}
#         for key in per_seed[0]:
#             if key.startswith("_"):
#                 avg[key] = per_seed[-1][key]
#                 continue
#             vals = [s[key] for s in per_seed
#                     if isinstance(s.get(key), (int, float, np.integer, np.floating))]
#             if vals:
#                 avg[key] = float(np.mean(vals))
#                 avg[f"{key}_std"] = float(np.std(vals))
#         return avg

#     # ================================================================== #
#     # Output / persistence
#     # ================================================================== #
#     def _save_final(self, image, final_caption, final_images, sid) -> Optional[Path]:
#         d = self.debug_dir / sid
#         d.mkdir(parents=True, exist_ok=True)
#         (d / "final_caption.txt").write_text(final_caption, encoding="utf-8")
#         if not final_images:
#             return None
#         recon = final_images[0].convert("RGB")
#         path = d / "final_reconstruction_I_star.png"
#         recon.save(path)
#         save_side_by_side(image, recon, "Original (I)", "Final Reconstruction (I*)",
#                           d / "side_by_side_original_vs_I_star.png", final_caption)
#         print(f"[Output] I* saved: {path}")
#         return path

#     def _skipped_result(self, image_id, caption, original_tokens, baseline_D, threshold,
#                         baseline_dist, baseline_cmp, sid, start) -> CompressionResult:
#         details = dict(baseline_dist); details["status"] = "skipped_low_baseline_fidelity"
#         result = CompressionResult(
#             image_id=image_id, status="skipped_low_baseline_fidelity",
#             original_caption=caption, compressed_caption=caption,
#             original_tokens=original_tokens, compressed_tokens=original_tokens,
#             compression_ratio=0.0, baseline_distortion=baseline_D, threshold=threshold,
#             final_distortion=baseline_D, final_distortion_details=details,
#             baseline_kg_comparison=baseline_cmp, final_kg_comparison=baseline_cmp,
#             strict_anchors=[], soft_anchors=[], quoted_anchors=[], protected_anchors=[],
#             steps=[], total_time=time.time() - start, final_reconstruction_path=None,
#         )
#         if self.save_intermediate:
#             self._save_result(result)
#         return result

#     def _save_kg(self, kg, image_id, name) -> None:
#         d = self.kg_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         (d / f"{safe_filename(name)}.json").write_text(
#             json.dumps(make_json_serializable(kg_to_readable_dict(kg)), indent=2, ensure_ascii=False),
#             encoding="utf-8")

#     def _save_comparison(self, comparison, image_id, name) -> None:
#         d = self.kg_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         (d / f"{safe_filename(name)}.json").write_text(
#             json.dumps(make_json_serializable(comparison), indent=2, ensure_ascii=False), encoding="utf-8")

#     def _save_candidates(self, image_id, iteration, stage, raw, filtered, strict, soft) -> None:
#         d = self.candidate_debug_dir / safe_filename(image_id)
#         d.mkdir(parents=True, exist_ok=True)
#         payload = {"iteration": iteration, "stage": stage, "strict_anchors": list(strict),
#                    "soft_anchors": list(soft), "raw_candidates": list(raw),
#                    "filtered_candidates": [asdict(c) for c in filtered]}
#         (d / f"iter{iteration}_{stage}_candidates.json").write_text(
#             json.dumps(make_json_serializable(payload), indent=2, ensure_ascii=False), encoding="utf-8")

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"
#         path.write_text(json.dumps(make_json_serializable(result), indent=2, ensure_ascii=False), encoding="utf-8")
#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             return
#         statuses: Dict[str, int] = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1
#         summary = {
#             "num_samples": len(results), "status_counts": statuses,
#             "avg_compression_ratio": float(np.mean([r.compression_ratio for r in results])),
#             "avg_baseline_distortion": float(np.mean([r.baseline_distortion for r in results])),
#             "avg_final_distortion": float(np.mean([r.final_distortion for r in results])),
#             "avg_original_tokens": float(np.mean([r.original_tokens for r in results])),
#             "avg_compressed_tokens": float(np.mean([r.compressed_tokens for r in results])),
#             "avg_time_seconds": float(np.mean([r.total_time for r in results])),
#             "total_time_hours": float(np.sum([r.total_time for r in results]) / 3600.0),
#             "use_relative_threshold": self.use_relative_threshold, "epsilon": self.epsilon,
#             "tau_img": self.tau_img, "tau_cap": self.tau_cap, "tau_kg": self.tau_kg,
#         }
#         (self.result_dir / "summary.json").write_text(
#             json.dumps(make_json_serializable(summary), indent=2), encoding="utf-8")
#         print("\n" + "=" * 70)
#         print(f"SUMMARY — {len(results)} samples | {statuses}")
#         print(f"  Avg CR={summary['avg_compression_ratio']:.2%} | "
#               f"D0={summary['avg_baseline_distortion']:.4f} -> D={summary['avg_final_distortion']:.4f}")
#         print(f"  Tokens {summary['avg_original_tokens']:.1f} -> {summary['avg_compressed_tokens']:.1f} | "
#               f"{summary['total_time_hours']:.2f}h")
#         print("=" * 70)



# """
# KG-RD Compressor - Grammar-Aware Rewrite Compression
# ====================================================

# Replaces greedy span deletion with rewrite-and-validate compression.

# Goal:
#     Maximum compression under acceptable reconstruction fidelity.

# Pipeline:
#     1. Extract KG from the original image.
#     2. Reconstruct image from full caption -> baseline distortion + threshold.
#     3. Apply baseline gate (skip if reconstruction is already poor).
#     4. Extract protected anchors:
#         - Quoted strings from caption (must appear verbatim).
#         - Key object/attribute terms from the original-image KG.
#     5. For up to max_iterations rounds:
#         a. Generate K grammatical candidate rewrites at varied token budgets
#            via a CaptionRewriter (text-only LLM).
#         b. Filter candidates that drop any protected anchor.
#         c. For each surviving candidate (shortest first):
#             - Reconstruct via FLUX.
#             - Extract KG via Qwen2.5-VL.
#             - Compute distortion + KG comparison.
#             - Check fidelity floors.
#         d. Select the shortest candidate that passes the fidelity floor.
#            If shorter than current, replace current and continue.
#            Else stop.
#     6. If no candidate ever passes, apply fallback_strategy.
# """

# import json
# import re
# import time
# from abc import ABC, abstractmethod
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     canonicalize_for_matching,
#     compare_scene_graphs,
#     normalize_label,
#     semantic_text_similarity,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor


# # =====================================================================
# # Data classes
# # =====================================================================

# @dataclass
# class RewriteCandidate:
#     """One candidate rewrite produced by the rewriter."""
#     index: int
#     text: str
#     target_token_budget: int
#     token_count: int
#     missing_anchors: List[str] = field(default_factory=list)
#     passed_anchor_filter: bool = False


# @dataclass
# class CompressionStep:
#     """One validated candidate during the compression loop."""
#     iteration: int
#     candidate_index: int
#     candidate_caption: str
#     target_token_budget: int
#     caption_before: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     distortion_details: Dict[str, Any]
#     missing_anchors: List[str]
#     time_seconds: float


# @dataclass
# class CompressionResult:
#     """Final result for one image-caption pair."""
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     protected_anchors: List[str]
#     quoted_anchors: List[str]
#     steps: List[CompressionStep]
#     total_time: float


# # =====================================================================
# # Helpers
# # =====================================================================

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"\(\s+", "(", text)
#     text = re.sub(r"\s+\)", ")", text)
#     text = text.replace(" ,", ",").replace(" .", ".")
#     return text.strip()


# def safe_filename(name: str) -> str:
#     name = str(name)
#     name = re.sub(r"[^\w\-.]+", "_", name)
#     return name[:180]


# def dataclass_is_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if dataclass_is_instance(obj):
#         return make_json_serializable(asdict(obj))
#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]
#     if isinstance(obj, tuple):
#         return list(obj)
#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     tuples = kg.get("tuples", set())
#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "tuples": [list(t) for t in sorted(list(tuples), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def print_kg(title: str, kg: Dict[str, Any], max_items: int = 40) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)

#     print("\nOBJECTS:")
#     for obj in kg.get("objects", [])[:max_items]:
#         label = obj.get("label", obj.get("name", ""))
#         attrs = obj.get("attributes", [])
#         print(f"  - {label} | attributes={attrs}")

#     print("\nATTRIBUTES:")
#     for attr in kg.get("attributes", [])[:max_items]:
#         print(f"  - ({attr.get('object', '')}, has_attribute, {attr.get('attribute', '')})")

#     print("\nRELATIONS:")
#     for rel in kg.get("relations", [])[:max_items]:
#         pred = rel.get("predicate", rel.get("relation", ""))
#         print(f"  - ({rel.get('subject', '')}, {pred}, {rel.get('object', '')})")

#     print("=" * 100)


# def print_kg_comparison(title: str, comparison: Dict[str, Any], max_items: int = 60) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)
#     print(
#         f"F1 mean={comparison.get('f1_mean', 0.0):.4f} | "
#         f"Recall mean={comparison.get('recall_mean', 0.0):.4f} | "
#         f"Object recall={comparison.get('object_recall_mean', 0.0):.4f} | "
#         f"Attribute recall={comparison.get('attribute_recall_mean', 0.0):.4f} | "
#         f"Relation recall={comparison.get('relation_recall_mean', 0.0):.4f}"
#     )

#     print("\nPRESERVED:")
#     for x in comparison.get("preserved_text", [])[:max_items]:
#         print("  +", x)

#     print("\nMISSING:")
#     for x in comparison.get("missing_text", [])[:max_items]:
#         print("  -", x)

#     print("\nEXTRA / HALLUCINATED:")
#     for x in comparison.get("extra_text", [])[:max_items]:
#         print("  *", x)

#     print("=" * 100)


# def save_debug_sample_visuals(
#     image_original: Image.Image,
#     image_reconstructed: Image.Image,
#     caption: str,
#     image_id: str,
#     output_dir: Path,
# ) -> None:
#     debug_dir = output_dir / "debug_samples" / safe_filename(image_id)
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     original_path = debug_dir / "original_image.png"
#     reconstructed_path = debug_dir / "reconstructed_from_caption.png"
#     caption_path = debug_dir / "caption.txt"
#     side_by_side_path = debug_dir / "side_by_side_original_vs_reconstructed.png"

#     image_original_rgb = image_original.convert("RGB")
#     image_reconstructed_rgb = image_reconstructed.convert("RGB")

#     image_original_rgb.save(original_path)
#     image_reconstructed_rgb.save(reconstructed_path)

#     with open(caption_path, "w", encoding="utf-8") as f:
#         f.write(caption)

#     left = image_original_rgb.resize((512, 512))
#     right = image_reconstructed_rgb.resize((512, 512))
#     canvas = Image.new("RGB", (1024, 560), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))

#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), "Original Image", fill="black")
#     draw.text((655, 15), "Reconstructed from Caption", fill="black")
#     canvas.save(side_by_side_path)

#     print(f"[Debug] Saved: {original_path}")
#     print(f"[Debug] Saved: {reconstructed_path}")
#     print(f"[Debug] Saved: {side_by_side_path}")


# # =====================================================================
# # Anchor extraction & verification
# # =====================================================================

# # All quote-pair styles we accept around an anchor. The first element of
# # each tuple is the opener, the second is the closer.
# _QUOTE_PAIRS = [
#     ('"', '"'),
#     ("\u201c", "\u201d"),  # smart double
#     ("'", "'"),
#     ("\u2018", "\u2019"),  # smart single
#     ("`", "`"),
# ]


# def _extract_all_quoted_substrings(text: str) -> List[str]:
#     """
#     Return every quoted substring in `text`, regardless of quote style.

#     Includes single and double quotes (straight + smart) and backticks.
#     Returned strings are stripped but case-preserved.
#     """
#     if not text:
#         return []

#     pattern = (
#         r'"([^"]+)"'
#         r"|\u201c([^\u201c\u201d]+)\u201d"
#         r"|`([^`]+)`"
#         r"|'([^']+)'"
#         r"|\u2018([^\u2018\u2019]+)\u2019"
#     )

#     out: List[str] = []
#     for m in re.finditer(pattern, text):
#         for g in m.groups():
#             if g is not None:
#                 s = g.strip()
#                 if s:
#                     out.append(s)
#                 break
#     return out


# def extract_quoted_anchors(caption: str) -> List[str]:
#     """
#     Extract quoted strings from a caption. These must be preserved as
#     quoted strings (any quote style) in any candidate rewrite.

#     Originals nearly always use double quotes; we restrict extraction to
#     double / smart-double here to avoid possessive false positives like
#     "the dog's bone". Candidate matching, however, accepts ANY quote style
#     (see candidate_preserves_anchors).
#     """
#     if not caption:
#         return []

#     raw = re.findall(r'"([^"]+)"|\u201c([^\u201c\u201d]+)\u201d', caption)

#     anchors: List[str] = []
#     seen: Set[str] = set()
#     for groups in raw:
#         s = next((g for g in groups if g), "")
#         s_stripped = s.strip()
#         if not s_stripped:
#             continue
#         key = s_stripped.lower()
#         if key in seen:
#             continue
#         seen.add(key)
#         anchors.append(s_stripped)

#     return anchors


# # Visual-marker keywords used to up-rank KG anchors that refer to text or
# # painted markings, since these are usually the most reconstruction-critical.
# _VISUAL_MARKER_KEYWORDS = {
#     "text", "word", "words", "letter", "letters", "writing",
#     "marking", "markings", "sign", "signs", "label", "paint",
#     "painted", "graffiti",
# }

# _HIGH_VALUE_ATTRIBUTES = {
#     "red", "orange", "yellow", "blue", "green", "white", "black",
#     "gray", "grey", "brown", "purple", "pink",
#     "faded", "old", "new", "bright", "dark",
#     "horizontal", "vertical", "diagonal",
#     "large", "small", "tall", "short", "long",
# }


# def extract_kg_anchors(
#     kg: Dict[str, Any],
#     max_anchors: int = 6,
# ) -> List[str]:
#     """
#     Extract the most important visual terms from the original-image KG.
#     These guide the rewriter and serve as soft preservation constraints.

#     Heuristic:
#       1. Score each object by:
#          - +2 if it is a visual-marker term (text/word/marking/sign/...)
#          - +1 per attribute attached to it
#          - +1 per relation it participates in
#       2. Take the top-K objects by score.
#       3. For each chosen object, also include its single most distinctive
#          attribute (color/state) when it falls in a high-value vocabulary.
#     """
#     objects = kg.get("objects", []) or []
#     relations = kg.get("relations", []) or []
#     attributes = kg.get("attributes", []) or []

#     if not objects:
#         return []

#     rel_count: Dict[str, int] = {}
#     for rel in relations:
#         s = normalize_label(rel.get("subject", ""))
#         o = normalize_label(rel.get("object", ""))
#         if s:
#             rel_count[s] = rel_count.get(s, 0) + 1
#         if o:
#             rel_count[o] = rel_count.get(o, 0) + 1

#     attrs_by_obj: Dict[str, List[str]] = {}
#     for attr in attributes:
#         obj_name = normalize_label(attr.get("object", ""))
#         attr_val = normalize_label(attr.get("attribute", ""))
#         if obj_name and attr_val:
#             attrs_by_obj.setdefault(obj_name, []).append(attr_val)

#     scored = []
#     seen_labels: Set[str] = set()

#     for obj in objects:
#         label = normalize_label(obj.get("label", obj.get("name", "")))
#         if not label or label in seen_labels:
#             continue
#         seen_labels.add(label)

#         score = 0
#         if any(kw in label.split() for kw in _VISUAL_MARKER_KEYWORDS):
#             score += 2

#         embedded_attrs = obj.get("attributes", []) or []
#         score += min(len(embedded_attrs), 3)
#         score += min(len(attrs_by_obj.get(label, [])), 3)
#         score += min(rel_count.get(label, 0), 3)

#         all_attrs = list(dict.fromkeys(
#             [normalize_label(a) for a in embedded_attrs] +
#             attrs_by_obj.get(label, [])
#         ))

#         scored.append((score, label, all_attrs))

#     scored.sort(key=lambda x: (-x[0], x[1]))

#     anchors: List[str] = []
#     used: Set[str] = set()

#     for score, label, attrs in scored:
#         if len(anchors) >= max_anchors:
#             break
#         if label in used:
#             continue

#         # Pair the object with its most distinctive high-value attribute.
#         best_attr = None
#         for a in attrs:
#             if a in _HIGH_VALUE_ATTRIBUTES:
#                 best_attr = a
#                 break

#         anchor = f"{best_attr} {label}" if best_attr else label
#         anchors.append(anchor)
#         used.add(label)

#     return anchors


# def candidate_preserves_anchors(
#     candidate: str,
#     quoted_anchors: Sequence[str],
#     kg_anchors: Sequence[str],
#     kg_match_threshold: float = 0.55,
#     max_kg_miss_count: int = 0,
# ) -> Tuple[bool, List[str]]:
#     """
#     Check whether a candidate preserves required anchors.

#     Quoted anchors are STRICT: every quoted anchor must appear inside a
#     quoted substring of the candidate (any quote style accepted), or for
#     multi-character anchors as a bare substring.

#     KG anchors are SOFT: up to `max_kg_miss_count` anchors may be missing
#     and the candidate still passes. Default 0 keeps the previous strict
#     behaviour. Anchor presence is checked via canonical-form substring,
#     content-word overlap, or semantic similarity >= kg_match_threshold.

#     Returns (passed, missing_anchors_list).
#     """
#     if not candidate:
#         return False, [f'"{a}"' for a in quoted_anchors] + list(kg_anchors)

#     candidate_norm = canonicalize_for_matching(candidate)

#     candidate_quoted_lower = {
#         s.strip().lower() for s in _extract_all_quoted_substrings(candidate)
#     }

#     quoted_missing: List[str] = []
#     kg_missing: List[str] = []

#     # 1. Quoted anchors (strict).
#     for anchor in quoted_anchors:
#         anchor_key = anchor.strip().lower()
#         if anchor_key in candidate_quoted_lower:
#             continue
#         if len(anchor_key) >= 3 and anchor_key in candidate.lower():
#             continue
#         quoted_missing.append(f'"{anchor}"')

#     # 2. KG anchors (soft).
#     for anchor in kg_anchors:
#         anchor_norm = canonicalize_for_matching(anchor)
#         if not anchor_norm:
#             continue

#         if anchor_norm in candidate_norm:
#             continue

#         anchor_words = [w for w in anchor_norm.split() if len(w) > 2]
#         if anchor_words and any(w in candidate_norm.split() for w in anchor_words):
#             continue

#         sim = semantic_text_similarity(anchor_norm, candidate_norm)
#         if sim >= kg_match_threshold:
#             continue

#         kg_missing.append(anchor)

#     # Decide pass/fail.
#     if quoted_missing:
#         return False, quoted_missing + kg_missing
#     if len(kg_missing) > max_kg_miss_count:
#         return False, kg_missing

#     return True, kg_missing  # passed; report soft KG misses for logging only


# def _token_set(s: str) -> Set[str]:
#     return {w.lower().strip(".,;:!?\"'()[]{}") for w in s.split() if w.strip()}


# def is_substantive_rewrite(
#     original: str,
#     candidate: str,
#     min_diff_ratio: float = 0.15,
# ) -> bool:
#     """
#     True if the candidate differs from the original enough to count as a
#     real rewrite (not just a near-echo with a punctuation tweak).

#     Uses token-set Jaccard distance; "different enough" means at least
#     `min_diff_ratio` of the union tokens are unique to one side.
#     """
#     a = _token_set(original)
#     b = _token_set(candidate)
#     if not a or not b:
#         return True

#     union = a | b
#     intersect = a & b
#     if not union:
#         return True

#     jaccard = len(intersect) / len(union)
#     return (1.0 - jaccard) >= min_diff_ratio


# # =====================================================================
# # Rewriter interface
# # =====================================================================

# class CaptionRewriter(ABC):
#     """
#     Generator interface used by KGRDCompressor.

#     Any rewriter (Qwen-VL, small text LLM, API) just needs to implement
#     `rewrite(...)` and return a list of grammatical candidate strings.
#     """

#     @abstractmethod
#     def rewrite(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> List[str]:
#         ...


# class QwenVLRewriter(CaptionRewriter):
#     """
#     Text-only rewriter that reuses the Qwen2.5-VL model already loaded
#     inside VLMSceneGraphExtractor. No additional GPU memory is consumed.
#     """

#     def __init__(
#         self,
#         vlm_extractor: VLMSceneGraphExtractor,
#         max_new_tokens: int = 512,
#         temperature: float = 0.7,
#         top_p: float = 0.9,
#     ):
#         self.model = vlm_extractor.model
#         self.processor = vlm_extractor.processor
#         self.device = vlm_extractor.device
#         self.max_new_tokens = int(max_new_tokens)
#         self.temperature = float(temperature)
#         self.top_p = float(top_p)

#     def _build_prompt(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> str:
#         original_tokens = count_tokens(original_caption)

#         if quoted_anchors:
#             quoted_lines = []
#             for i, a in enumerate(quoted_anchors, 1):
#                 note = "  <- single character, but it IS a real anchor; preserve it" if len(a) == 1 else ""
#                 quoted_lines.append(f'  ({i}) "{a}"{note}')
#             quoted_block = "\n".join(quoted_lines)
#             n_quoted = len(quoted_anchors)
#         else:
#             quoted_block = "  (none)"
#             n_quoted = 0

#         if kg_anchors:
#             kg_block = "\n".join(f"  ({i}) {a}" for i, a in enumerate(kg_anchors, 1))
#             n_kg = len(kg_anchors)
#         else:
#             kg_block = "  (none)"
#             n_kg = 0

#         budgets = list(target_token_budgets)
#         if budgets:
#             min_budget = min(budgets)
#             max_budget = max(budgets)
#         else:
#             min_budget = max_budget = max(8, original_tokens // 2)

#         budget_lines = "\n".join(
#             f"  - Variant {i + 1}: ~{b} tokens"
#             for i, b in enumerate(budgets)
#         ) if budgets else f"  - Variants: ~{min_budget} to ~{max_budget} tokens"

#         return f"""You compress image captions while preserving the visual content needed to reconstruct the image with a text-to-image model.

# ORIGINAL CAPTION ({original_tokens} tokens):
# {original_caption}

# ==========================================================
# HARD REQUIREMENT 1 - QUOTED STRINGS (visible text/markings)
# ==========================================================
# The caption contains {n_quoted} quoted string(s). Each names visible TEXT or
# MARKINGS in the image. EVERY variant MUST include EVERY quoted string,
# each surrounded by quotation marks. Use straight double quotes ".." (preferred)
# or single quotes '..' - either is accepted.

# {quoted_block}

# A variant that omits ANY of these quoted strings is INVALID, even the
# single-character ones. Repeat: every variant must contain all {n_quoted}.

# ==========================================================
# HARD REQUIREMENT 2 - VISUAL ANCHORS
# ==========================================================
# Every variant must also reference each of these visual elements (verbatim
# or close paraphrase):

# {kg_block}

# ==========================================================
# HARD REQUIREMENT 3 - WRITE FULL SENTENCES, NEVER FRAGMENTS
# ==========================================================
# Each variant MUST be a meaningful, grammatical English sentence that a
# human could read aloud naturally. Never produce a comma-separated list of
# nouns or noun phrases. A short fragment is WORSE than a longer sentence.

# GOOD (natural sentence):
#   A faded street corner with orange "ROW" sprayed in the middle, a horizontal "i" above, and a faded red "FIRE LANE" line on the curb.

# BAD (fragment - INVALID):
#   Orange "ROW", "i", red "FIRE LANE", curb.
#   Faded corner with orange "ROW" and "i", red "FIRE LANE" on curb.

# ==========================================================
# TASK
# ==========================================================
# Produce {num_candidates} variants. Variant 1 should be the shortest. Each
# variant should target approximately the following length:

# {budget_lines}

# CRITICAL LENGTH CONSTRAINTS:
#   * NO variant may be shorter than {min_budget} tokens. Variants below this
#     length are INVALID and will be discarded. Aim for full sentences.
#   * Every variant must be SHORTER than the original ({original_tokens} tokens).
#   * Do not introduce objects or details not present in the original caption.

# ==========================================================
# SELF-CHECK before output - verify EACH variant
# ==========================================================
# For each variant, confirm ALL of:
#   [ ] Reads as a complete English sentence with subject + verb
#   [ ] Does NOT start with a comma-separated list of nouns
#   [ ] Contains at least one main verb (is, are, shows, has, contains, etc.)

# GOOD (full sentence): "An overcast intersection at Cicero Ave shows a white truck stopped at a red light while vehicles cross past Boston Market and Pizza Hut signs."

# BAD (fragment - REWRITE as full sentence): "Cicero Ave, white truck, red light, vehicles, Boston Market."
# BAD (semicolon list - REWRITE as full sentence): "Busy intersection; white truck; red light; vehicles; Boston Market."

# If a variant is a comma- or semicolon-separated list, REWRITE it as a real sentence
# before output. Token budget is a SOFT TARGET; grammatical sentence is HARD requirement.

# ==========================================================
# OUTPUT FORMAT
# ==========================================================
# Return ONLY a JSON array of strings, ordered shortest first.
# No markdown fences. No numbering. No explanation. Just the JSON array.

# Example output (illustrative, ignore content):
# ["shortest variant here", "medium variant here", "longest variant here"]
# """

#     def _parse_json_array(self, text: str) -> List[str]:
#         """
#         Robustly parse a JSON array of strings from LLM output.
#         """
#         if not isinstance(text, str):
#             return []

#         text = text.strip()
#         text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"```", "", text).strip()

#         # Try direct parse.
#         try:
#             parsed = json.loads(text)
#             if isinstance(parsed, list):
#                 return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#         except Exception:
#             pass

#         # Find the first [...] block.
#         match = re.search(r"\[.*\]", text, flags=re.DOTALL)
#         if match:
#             try:
#                 parsed = json.loads(match.group(0))
#                 if isinstance(parsed, list):
#                     return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#             except Exception:
#                 pass

#         # Last resort: extract quoted lines.
#         lines = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
#         return [s.strip() for s in lines if s.strip()]

#     @torch.no_grad()
#     def _generate_text(self, prompt: str) -> str:
#         messages = [
#             {"role": "user", "content": [{"type": "text", "text": prompt}]}
#         ]

#         text_input = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         # Try the multimodal processor in text-only mode first.
#         # Fall back to the underlying tokenizer if the processor requires images.
#         try:
#             inputs = self.processor(
#                 text=[text_input],
#                 padding=True,
#                 return_tensors="pt",
#             )
#         except Exception:
#             inputs = self.processor.tokenizer(
#                 text_input,
#                 padding=True,
#                 return_tensors="pt",
#             )

#         if hasattr(inputs, "to"):
#             inputs = inputs.to(self.model.device)
#         else:
#             inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         generated_ids = self.model.generate(
#             **inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             temperature=self.temperature,
#             top_p=self.top_p,
#         )

#         input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
#         trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]

#         if hasattr(self.processor, "batch_decode"):
#             output_text = self.processor.batch_decode(
#                 trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )[0]
#         else:
#             output_text = self.processor.tokenizer.batch_decode(
#                 trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )[0]

#         return output_text.strip()

#     def rewrite(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> List[str]:
#         prompt = self._build_prompt(
#             original_caption=original_caption,
#             quoted_anchors=quoted_anchors,
#             kg_anchors=kg_anchors,
#             target_token_budgets=target_token_budgets,
#             num_candidates=num_candidates,
#         )

#         try:
#             raw_output = self._generate_text(prompt)
#         except Exception as exc:
#             print(f"[Rewriter] Generation failed: {exc}")
#             return []

#         candidates = self._parse_json_array(raw_output)
#         candidates = [clean_caption_spacing(c) for c in candidates]
#         candidates = [c for c in candidates if c]

#         # Deduplicate while preserving order.
#         seen: Set[str] = set()
#         unique: List[str] = []
#         for c in candidates:
#             key = c.lower()
#             if key in seen:
#                 continue
#             seen.add(key)
#             unique.append(c)

#         return unique


# # =====================================================================
# # Main compressor
# # =====================================================================

# class KGRDCompressor:
#     """
#     KG-guided Rate-Distortion Caption Compressor (rewrite-based).

#     Replaces greedy span deletion with grammar-aware rewrite candidates
#     validated through the FLUX -> Qwen2.5-VL -> distortion pipeline.
#     """

#     def __init__(
#         self,
#         # --- Scene graph extractor (Qwen2.5-VL) ---
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,

#         # --- T2I (FLUX) ---
#         t2i_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "float16",
#         t2i_steps: int = 30,
#         t2i_guidance: float = 7.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,

#         # --- Distortion metric ---
#         alpha: float = 0.4,
#         beta: float = 0.4,
#         gamma: float = 0.2,
#         clip_model: str = "ViT-L-14",
#         clip_pretrained: str = "openai",
#         lpips_net: str = "alex",
#         metric_device: str = "auto",

#         # --- Threshold ---
#         use_relative_threshold: bool = True,
#         epsilon: float = 0.10,
#         absolute_threshold: float = 0.25,

#         # --- Compression loop ---
#         min_caption_tokens: int = 5,
#         max_iterations: int = 1,
#         num_candidates_per_iteration: int = 5,
#         min_target_token_ratio: float = 0.30,
#         max_target_token_ratio: float = 0.80,
#         require_strict_compression: bool = True,

#         # Preferred caption length for the FINAL compressed caption.
#         # Selection prefers the SHORTEST passing candidate >= this value.
#         # If no passing candidate reaches this length, the LONGEST passing
#         # candidate is selected instead. This avoids picking 10-token
#         # fragments when 25-token meaningful captions are also available.
#         preferred_target_tokens: int = 25,

#         # --- Baseline gate ---
#         min_baseline_kg_f1: float = 0.05,
#         min_baseline_object_recall: float = 0.10,

#         # --- Fidelity floors for accepting a candidate ---
#         kg_f1_floor_drop: float = 0.08,
#         object_recall_floor_drop: float = 0.05,
#         attribute_recall_floor_drop: float = 0.12,
#         attribute_recall_min_floor: float = 0.10,

#         # --- Anchor extraction ---
#         max_kg_anchors: int = 6,

#         # --- Anchor filter strictness ---
#         # Soft KG-anchor matching: allow this fraction of KG anchors to be
#         # missing in a candidate and still pass the pre-filter.
#         # Quoted anchors remain strict regardless.
#         kg_anchor_max_miss_fraction: float = 0.40,

#         # Reject candidates whose token-set Jaccard with the original is
#         # too high (i.e., the LLM basically echoed the original). Below
#         # this difference ratio, the candidate is treated as a non-rewrite.
#         min_rewrite_diff_ratio: float = 0.15,

#         # --- Fallback if no candidate passes ---
#         # Default to "best_candidate" so that even when fidelity floors
#         # reject every candidate, we still return the closest acceptable
#         # rewrite rather than the unmodified original.
#         fallback_strategy: str = "best_candidate",   # keep_original | best_candidate

#         # --- Rewriter ---
#         rewriter: Optional[CaptionRewriter] = None,
#         rewriter_max_new_tokens: int = 512,
#         rewriter_temperature: float = 0.7,
#         rewriter_top_p: float = 0.9,

#         # --- I/O ---
#         save_intermediate: bool = True,
#         output_dir: str = "./outputs",
#     ):
#         self.t2i_seeds = t2i_seeds or [42]

#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)

#         self.min_caption_tokens = int(min_caption_tokens)
#         self.max_iterations = int(max_iterations)
#         self.num_candidates_per_iteration = int(num_candidates_per_iteration)
#         self.min_target_token_ratio = float(min_target_token_ratio)
#         self.max_target_token_ratio = float(max_target_token_ratio)
#         self.require_strict_compression = bool(require_strict_compression)
#         self.preferred_target_tokens = int(preferred_target_tokens)

#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)

#         self.kg_f1_floor_drop = float(kg_f1_floor_drop)
#         self.object_recall_floor_drop = float(object_recall_floor_drop)
#         self.attribute_recall_floor_drop = float(attribute_recall_floor_drop)
#         self.attribute_recall_min_floor = float(attribute_recall_min_floor)

#         self.max_kg_anchors = int(max_kg_anchors)
#         self.kg_anchor_max_miss_fraction = float(kg_anchor_max_miss_fraction)
#         self.min_rewrite_diff_ratio = float(min_rewrite_diff_ratio)

#         if fallback_strategy not in {"keep_original", "best_candidate"}:
#             raise ValueError(
#                 f"fallback_strategy must be one of "
#                 f"'keep_original' | 'best_candidate', got: {fallback_strategy}"
#             )
#         self.fallback_strategy = fallback_strategy

#         self.save_intermediate = bool(save_intermediate)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"
#         self.candidate_debug_dir = self.output_dir / "candidates"
#         self.result_dir.mkdir(parents=True, exist_ok=True)
#         self.kg_debug_dir.mkdir(parents=True, exist_ok=True)
#         self.candidate_debug_dir.mkdir(parents=True, exist_ok=True)

#         print("=" * 70)
#         print("KG-RD Compressor: Initializing rewrite-based pipeline")
#         print("=" * 70)

#         print("\n[1/3] Loading VLM scene graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model,
#             device=scene_graph_device,
#             dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens,
#             temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample,
#             max_objects=max_objects,
#             max_relations=max_relations,
#         )

#         print("\n[2/3] Loading text-to-image reconstructor...")
#         self.t2i = T2IReconstructor(
#             model_name=t2i_model,
#             device=t2i_device,
#             dtype=t2i_dtype,
#             num_inference_steps=t2i_steps,
#             guidance_scale=t2i_guidance,
#             image_size=t2i_image_size,
#         )

#         print("\n[3/3] Loading distortion metric...")
#         self.distortion = DistortionMetric(
#             alpha=alpha,
#             beta=beta,
#             gamma=gamma,
#             clip_model_name=clip_model,
#             clip_pretrained=clip_pretrained,
#             lpips_net=lpips_net,
#             device=metric_device,
#         )

#         # Default rewriter shares Qwen2.5-VL with the scene graph extractor.
#         if rewriter is None:
#             print("\n[Rewriter] Using QwenVLRewriter (shared with scene graph VLM).")
#             self.rewriter: CaptionRewriter = QwenVLRewriter(
#                 vlm_extractor=self.sg_extractor,
#                 max_new_tokens=rewriter_max_new_tokens,
#                 temperature=rewriter_temperature,
#                 top_p=rewriter_top_p,
#             )
#         else:
#             print(f"\n[Rewriter] Using injected rewriter: {type(rewriter).__name__}")
#             self.rewriter = rewriter

#         print("\n" + "=" * 70)
#         if self.use_relative_threshold:
#             print(f"Pipeline ready | relative threshold: delta = D0 * (1 + {self.epsilon})")
#         else:
#             print(f"Pipeline ready | absolute threshold: delta = {self.absolute_threshold}")
#         print(
#             f"Baseline gate | min KG-F1={self.min_baseline_kg_f1}, "
#             f"min object recall={self.min_baseline_object_recall}"
#         )
#         print(
#             f"Rewrite | candidates/iter={self.num_candidates_per_iteration}, "
#             f"max_iters={self.max_iterations}, "
#             f"target_ratio=[{self.min_target_token_ratio:.0%}..{self.max_target_token_ratio:.0%}], "
#             f"fallback={self.fallback_strategy}"
#         )
#         print("=" * 70 + "\n")

#     # -----------------------------------------------------------------
#     # Public API
#     # -----------------------------------------------------------------

#     def compress_single(
#         self,
#         image: Image.Image,
#         caption: str,
#         image_id: str = "unknown",
#     ) -> CompressionResult:
#         start_time = time.time()

#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)

#         if original_tokens <= self.min_caption_tokens:
#             raise ValueError(
#                 f"Caption has only {original_tokens} tokens, "
#                 f"min_caption_tokens={self.min_caption_tokens}."
#             )

#         image_id_safe = safe_filename(image_id)

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | original_tokens={original_tokens}")
#         print("-" * 70)

#         steps: List[CompressionStep] = []

#         # --- 1. Original-image KG ---
#         print("[1] Extracting KG from original image...")
#         kg_original = self.sg_extractor.extract(image, image_id=image_id)
#         print(
#             f"    Original KG: {len(kg_original.get('objects', []))} objects, "
#             f"{len(kg_original.get('attributes', []))} attributes, "
#             f"{len(kg_original.get('relations', []))} relations"
#         )
#         if self.save_intermediate:
#             self._save_kg_debug(kg_original, image_id_safe, "original_image_kg")
#         print_kg("ORIGINAL IMAGE KG", kg_original, max_items=40)

#         # --- 2. Baseline reconstruction & threshold ---
#         print("[2] Generating baseline reconstruction from full caption...")
#         baseline_images, baseline_kgs = self._reconstruct_and_extract_kgs(
#             caption=caption,
#             image_id=image_id_safe,
#             prefix="baseline_full_caption",
#         )

#         if self.save_intermediate and baseline_images:
#             save_debug_sample_visuals(
#                 image_original=image,
#                 image_reconstructed=baseline_images[0],
#                 caption=caption,
#                 image_id=image_id_safe,
#                 output_dir=self.output_dir,
#             )
#         if baseline_kgs:
#             print_kg("BASELINE RECONSTRUCTED KG", baseline_kgs[0], max_items=40)

#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image,
#             images_recon=baseline_images,
#             kg_original=kg_original,
#             kgs_reconstructed=baseline_kgs,
#         )
#         baseline_D = float(baseline_dist["D"])

#         if self.use_relative_threshold:
#             threshold = self.distortion.compute_relative_threshold(
#                 baseline_distortion=baseline_D,
#                 epsilon=self.epsilon,
#             )
#         else:
#             threshold = self.absolute_threshold

#         baseline_comparison = self._aggregate_kg_comparisons(kg_original, baseline_kgs)
#         if self.save_intermediate:
#             self._save_kg_comparison_debug(
#                 baseline_comparison, image_id_safe, "baseline_kg_comparison"
#             )

#         print(f"    Baseline distortion D0 = {baseline_D:.4f}")
#         print(f"    Threshold delta       = {threshold:.4f}")
#         print_kg_comparison("BASELINE KG COMPARISON", baseline_comparison)

#         # --- 3. Baseline gate ---
#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))

#         if baseline_f1 < self.min_baseline_kg_f1 or baseline_obj_recall < self.min_baseline_object_recall:
#             print("\n[SKIP] Baseline reconstruction fidelity is too low.")
#             return self._make_skipped_result(
#                 image_id=image_id,
#                 caption=caption,
#                 original_tokens=original_tokens,
#                 baseline_D=baseline_D,
#                 threshold=threshold,
#                 baseline_dist=baseline_dist,
#                 baseline_comparison=baseline_comparison,
#                 baseline_kgs=baseline_kgs,
#                 image_id_safe=image_id_safe,
#                 start_time=start_time,
#             )

#         # --- 4. Anchor extraction ---
#         quoted_anchors = extract_quoted_anchors(caption)
#         kg_anchors = extract_kg_anchors(kg_original, max_anchors=self.max_kg_anchors)

#         print(f"\n[4] Protected anchors:")
#         print(f"    Quoted: {quoted_anchors if quoted_anchors else '(none)'}")
#         print(f"    KG    : {kg_anchors if kg_anchors else '(none)'}")

#         # --- 5. Iterative rewrite-and-validate loop ---
#         current_caption = caption
#         current_kgs = baseline_kgs
#         current_dist = baseline_dist
#         current_images = baseline_images
#         best_below_floor: Optional[Dict[str, Any]] = None  # for best_candidate fallback

#         for iteration in range(1, self.max_iterations + 1):
#             current_tokens = count_tokens(current_caption)
#             if current_tokens <= self.min_caption_tokens:
#                 print(f"[Iter {iteration}] Already at minimum length. Stopping.")
#                 break

#             print(f"\n[Iter {iteration}] Current tokens: {current_tokens}")

#             target_budgets = self._compute_target_budgets(
#                 source_tokens=current_tokens,
#                 original_tokens=original_tokens,
#             )
#             print(f"    Target token budgets: {target_budgets}")

#             # 5a. Generate candidates.
#             t_gen = time.time()
#             raw_candidates = self.rewriter.rewrite(
#                 original_caption=current_caption,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#                 target_token_budgets=target_budgets,
#                 num_candidates=self.num_candidates_per_iteration,
#             )
#             print(f"    Generated {len(raw_candidates)} candidates in {time.time() - t_gen:.1f}s")

#             if not raw_candidates:
#                 print(f"[Iter {iteration}] Rewriter returned no candidates. Stopping.")
#                 break

#             # 5b. Anchor-filter and dedupe.
#             candidates = self._filter_candidates(
#                 raw_candidates=raw_candidates,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#                 source_tokens=current_tokens,
#                 target_budgets=target_budgets,
#                 original_caption=caption,
#             )

#             survivors = [c for c in candidates if c.passed_anchor_filter]

#             # If the soft anchor filter still rejects every candidate, retry
#             # once with permissive matching: only quoted anchors are required,
#             # KG anchors become optional. Better to test a noisy candidate
#             # than to fall through to keep_original.
#             if not survivors and candidates:
#                 print("    [Relax] Anchor filter rejected all; retrying with KG anchors optional.")
#                 for c in candidates:
#                     # Skip ones that already failed for non-anchor reasons.
#                     if any(m.startswith("__") for m in c.missing_anchors):
#                         continue
#                     passed, missing = candidate_preserves_anchors(
#                         candidate=c.text,
#                         quoted_anchors=quoted_anchors,
#                         kg_anchors=[],
#                         kg_match_threshold=0.55,
#                         max_kg_miss_count=0,
#                     )
#                     if passed:
#                         c.passed_anchor_filter = True
#                         # Keep the original miss list visible for debugging.
#                 survivors = [c for c in candidates if c.passed_anchor_filter]

#             if self.save_intermediate:
#                 self._save_candidates_debug(
#                     image_id=image_id_safe,
#                     iteration=iteration,
#                     raw_candidates=raw_candidates,
#                     filtered_candidates=candidates,
#                     quoted_anchors=quoted_anchors,
#                     kg_anchors=kg_anchors,
#                 )

#             print(f"    Anchor filter: {len(survivors)}/{len(candidates)} survived")

#             for c in candidates:
#                 tag = "OK " if c.passed_anchor_filter else "MISS"
#                 miss_str = f" missing={c.missing_anchors}" if c.missing_anchors else ""
#                 print(f"      [{tag}] ({c.token_count} tok) {c.text[:90]}{'...' if len(c.text) > 90 else ''}{miss_str}")

#             if not survivors:
#                 print(f"[Iter {iteration}] No candidate passed even relaxed anchor filter. Stopping.")
#                 break

#             # 5c. Validation order:
#             #     prefer the SHORTEST candidate >= preferred_target_tokens
#             #     (so we land near 25-30 tokens rather than 10-13 fragments).
#             #     If no candidate reaches the preferred length, fall back to
#             #     the LONGEST candidate (closest to preferred from below).
#             #     The first candidate to PASS FLUX validation in this order
#             #     is selected -- this minimizes wasted FLUX calls while still
#             #     reaching the preferred length when possible.
#             preferred_target = self.preferred_target_tokens

#             def _selection_order(c: RewriteCandidate) -> Tuple[int, int, int]:
#                 if c.token_count >= preferred_target:
#                     return (0, c.token_count, c.index)         # >= target: ascending
#                 return (1, -c.token_count, c.index)            # < target: descending

#             survivors.sort(key=_selection_order)

#             print(
#                 f"    Validation order (prefer >= {preferred_target} tok, "
#                 f"else longest): "
#                 + ", ".join(f"#{c.index}({c.token_count}t)" for c in survivors)
#             )

#             iteration_winner: Optional[Dict[str, Any]] = None

#             for cand in survivors:
#                 step_start = time.time()
#                 print(f"    [Validate] candidate#{cand.index} ({cand.token_count} tok): {cand.text}")

#                 cand_images, cand_kgs = self._reconstruct_and_extract_kgs(
#                     caption=cand.text,
#                     image_id=image_id_safe,
#                     prefix=f"iter{iteration}_cand{cand.index}",
#                 )

#                 cand_dist = self.distortion.compute_averaged_distortion(
#                     image_orig=image,
#                     images_recon=cand_images,
#                     kg_original=kg_original,
#                     kgs_reconstructed=cand_kgs,
#                 )
#                 cand_D = float(cand_dist["D"])
#                 cand_comparison = self._aggregate_kg_comparisons(kg_original, cand_kgs)

#                 passed, reason, accept_details = self._candidate_acceptance_decision(
#                     candidate_D=cand_D,
#                     threshold=threshold,
#                     candidate_comparison=cand_comparison,
#                     baseline_comparison=baseline_comparison,
#                 )

#                 details = {
#                     k: v for k, v in cand_dist.items() if not str(k).endswith("_std")
#                 }
#                 details.update({
#                     "candidate_kg_f1": cand_comparison.get("f1_mean", 0.0),
#                     "candidate_object_recall": cand_comparison.get("object_recall_mean", 0.0),
#                     "candidate_attribute_recall": cand_comparison.get("attribute_recall_mean", 0.0),
#                     "candidate_relation_recall": cand_comparison.get("relation_recall_mean", 0.0),
#                     "acceptance_reason": reason,
#                     "acceptance_details": accept_details,
#                 })

#                 step = CompressionStep(
#                     iteration=iteration,
#                     candidate_index=cand.index,
#                     candidate_caption=cand.text,
#                     target_token_budget=cand.target_token_budget,
#                     caption_before=current_caption,
#                     tokens_before=current_tokens,
#                     tokens_after=cand.token_count,
#                     token_reduction=current_tokens - cand.token_count,
#                     distortion=cand_D,
#                     threshold=threshold,
#                     passed_threshold=passed,
#                     selected=False,
#                     reason=reason,
#                     distortion_details=details,
#                     missing_anchors=cand.missing_anchors,
#                     time_seconds=time.time() - step_start,
#                 )
#                 steps.append(step)

#                 status = "PASS" if passed else "FAIL"
#                 print(
#                     f"      [{status}] D={cand_D:.4f} | "
#                     f"KG-F1={cand_comparison.get('f1_mean', 0.0):.4f} | "
#                     f"obj_rec={cand_comparison.get('object_recall_mean', 0.0):.4f} | "
#                     f"reason={reason}"
#                 )

#                 # Track best below-floor candidate for fallback.
#                 if not passed:
#                     score = (cand_D - threshold)  # smaller (more negative) = better
#                     if best_below_floor is None or score < best_below_floor["score"]:
#                         best_below_floor = {
#                             "score": score,
#                             "caption": cand.text,
#                             "tokens": cand.token_count,
#                             "D": cand_D,
#                             "kgs": cand_kgs,
#                             "images": cand_images,
#                             "dist": cand_dist,
#                             "comparison": cand_comparison,
#                             "step": step,
#                         }
#                     continue

#                 # First passing candidate (shortest, since we sorted) wins this iteration.
#                 step.selected = True
#                 iteration_winner = {
#                     "caption": cand.text,
#                     "tokens": cand.token_count,
#                     "D": cand_D,
#                     "kgs": cand_kgs,
#                     "images": cand_images,
#                     "dist": cand_dist,
#                     "comparison": cand_comparison,
#                     "step": step,
#                 }
#                 print(f"      [SELECTED] candidate#{cand.index}")
#                 break

#             if iteration_winner is None:
#                 print(f"[Iter {iteration}] No candidate passed fidelity floors. Stopping.")
#                 break

#             if self.require_strict_compression and iteration_winner["tokens"] >= current_tokens:
#                 print(f"[Iter {iteration}] Winner not shorter than current. Stopping.")
#                 break

#             current_caption = iteration_winner["caption"]
#             current_kgs = iteration_winner["kgs"]
#             current_dist = iteration_winner["dist"]
#             current_images = iteration_winner["images"]
#             print(
#                 f"[Iter {iteration}] Updated current caption | "
#                 f"tokens={iteration_winner['tokens']} | D={iteration_winner['D']:.4f}"
#             )

#         # --- 6. Resolve final result (with fallback) ---
#         compressed_tokens = count_tokens(current_caption)
#         no_compression = (compressed_tokens >= original_tokens)

#         if no_compression and self.fallback_strategy == "best_candidate" and best_below_floor is not None:
#             print(
#                 f"\n[FALLBACK] Using best below-floor candidate "
#                 f"(D={best_below_floor['D']:.4f}, threshold={threshold:.4f})"
#             )
#             current_caption = best_below_floor["caption"]
#             current_kgs = best_below_floor["kgs"]
#             current_dist = best_below_floor["dist"]
#             current_images = best_below_floor["images"]
#             best_below_floor["step"].selected = True
#             compressed_tokens = best_below_floor["tokens"]
#             no_compression = False
#             status = "compressed_via_fallback"
#         elif no_compression:
#             status = "kept_original_no_passing_candidate"
#         else:
#             status = "compressed"

#         compression_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))
#         total_time = time.time() - start_time

#         final_comparison = self._aggregate_kg_comparisons(kg_original, current_kgs)
#         if self.save_intermediate:
#             self._save_final_kgs_alias(current_kgs, image_id_safe)
#             self._save_kg_comparison_debug(final_comparison, image_id_safe, "final_kg_comparison")
#             self._save_final_reconstruction_visuals(
#                 image_original=image,
#                 image_final_recon=current_images[0] if current_images else None,
#                 final_caption=current_caption,
#                 image_id_safe=image_id_safe,
#             )

#         print_kg_comparison("FINAL KG COMPARISON", final_comparison)

#         final_details = {k: v for k, v in current_dist.items() if not str(k).endswith("_std")}
#         final_details["status"] = status

#         result = CompressionResult(
#             image_id=image_id,
#             status=status,
#             original_caption=caption,
#             compressed_caption=current_caption,
#             original_tokens=original_tokens,
#             compressed_tokens=compressed_tokens,
#             compression_ratio=compression_ratio,
#             baseline_distortion=baseline_D,
#             threshold=threshold,
#             final_distortion=float(current_dist["D"]),
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=final_comparison,
#             protected_anchors=list(kg_anchors),
#             quoted_anchors=list(quoted_anchors),
#             steps=steps,
#             total_time=total_time,
#         )

#         print("\n" + "-" * 70)
#         print(
#             f"Result for {image_id}: "
#             f"{original_tokens} -> {compressed_tokens} tokens | "
#             f"CR={compression_ratio:.2%} | "
#             f"Final D={result.final_distortion:.4f} | "
#             f"delta={threshold:.4f} | status={status}"
#         )
#         print("-" * 70)
#         print("\nORIGINAL CAPTION:")
#         print(caption)
#         print("\nFINAL COMPRESSED CAPTION:")
#         print(current_caption)

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     def compress_dataset(
#         self,
#         dataset: Any,
#         max_samples: Optional[int] = None,
#     ) -> List[CompressionResult]:
#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []

#         print("\n" + "=" * 70)
#         print(f"Processing {n} image-caption pairs")
#         print("=" * 70)

#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]
#             try:
#                 result = self.compress_single(
#                     image=sample["image"],
#                     caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 )
#                 results.append(result)
#             except Exception as exc:
#                 print(
#                     f"[ERROR] Failed on idx={idx}, "
#                     f"image_id={sample.get('image_id', 'unknown')}: {exc}"
#                 )
#                 continue

#         self._save_summary(results)
#         return results

#     # -----------------------------------------------------------------
#     # Reconstruction + KG extraction
#     # -----------------------------------------------------------------

#     def _reconstruct_and_extract_kgs(
#         self,
#         caption: str,
#         image_id: str,
#         prefix: str,
#     ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
#         output_dir = None
#         if self.save_intermediate:
#             output_dir = self.reconstruction_dir / safe_filename(image_id)
#             output_dir.mkdir(parents=True, exist_ok=True)

#         images = self.t2i.reconstruct(
#             caption=caption,
#             seeds=self.t2i_seeds,
#             output_dir=output_dir,
#             prefix=prefix,
#         )

#         kgs = []
#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract(img, image_id=f"{image_id}_{prefix}_seed{seed}")
#             kgs.append(kg)
#             if self.save_intermediate:
#                 self._save_kg_debug(kg, image_id, f"{prefix}_seed{seed}_reconstructed_kg")

#         return images, kgs

#     # -----------------------------------------------------------------
#     # KG comparison aggregation across seeds
#     # -----------------------------------------------------------------

#     def _aggregate_kg_comparisons(
#         self,
#         kg_original: Dict[str, Any],
#         reconstructed_kgs: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         comparisons = [compare_scene_graphs(kg_original, kg_recon) for kg_recon in reconstructed_kgs]

#         preserved: Set[Tuple[str, ...]] = set()
#         missing: Set[Tuple[str, ...]] = set()
#         extra: Set[Tuple[str, ...]] = set()

#         f1s, precisions, recalls = [], [], []
#         object_recalls, attribute_recalls, relation_recalls = [], [], []

#         for comp in comparisons:
#             preserved |= set(comp.get("preserved", set()))
#             missing |= set(comp.get("missing", set()))
#             extra |= set(comp.get("extra", set()))

#             f1s.append(comp.get("f1", 0.0))
#             precisions.append(comp.get("precision", 0.0))
#             recalls.append(comp.get("recall", 0.0))
#             object_recalls.append(comp.get("object_recall", 0.0))
#             attribute_recalls.append(comp.get("attribute_recall", 0.0))
#             relation_recalls.append(comp.get("relation_recall", 0.0))

#         return {
#             "preserved": preserved,
#             "missing": missing,
#             "extra": extra,
#             "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
#             "f1_std": float(np.std(f1s)) if f1s else 0.0,
#             "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
#             "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
#             "object_recall_mean": float(np.mean(object_recalls)) if object_recalls else 0.0,
#             "attribute_recall_mean": float(np.mean(attribute_recalls)) if attribute_recalls else 0.0,
#             "relation_recall_mean": float(np.mean(relation_recalls)) if relation_recalls else 0.0,
#             "num_preserved": len(preserved),
#             "num_missing": len(missing),
#             "num_extra": len(extra),
#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     # -----------------------------------------------------------------
#     # Candidate handling
#     # -----------------------------------------------------------------

#     def _compute_target_budgets(
#         self,
#         source_tokens: int,
#         original_tokens: int,
#     ) -> List[int]:
#         """
#         Build N target token budgets ranging from aggressive to conservative,
#         relative to the ORIGINAL caption (not the current one) so the loop
#         doesn't drift toward longer captions across iterations.
#         """
#         n = self.num_candidates_per_iteration
#         if n <= 0:
#             return []

#         lo = max(self.min_caption_tokens, int(round(self.min_target_token_ratio * original_tokens)))
#         hi = max(lo + 1, int(round(self.max_target_token_ratio * original_tokens)))

#         # Cap upper budget below the current caption to encourage compression.
#         hi = min(hi, max(lo + 1, source_tokens - 1))

#         if n == 1:
#             return [lo]

#         step = (hi - lo) / (n - 1)
#         budgets = [int(round(lo + i * step)) for i in range(n)]

#         # Deduplicate and ensure ascending order.
#         seen: Set[int] = set()
#         unique = []
#         for b in budgets:
#             if b not in seen:
#                 seen.add(b)
#                 unique.append(b)
#         return unique

#     def _filter_candidates(
#         self,
#         raw_candidates: Sequence[str],
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         source_tokens: int,
#         target_budgets: Sequence[int],
#         original_caption: str = "",
#     ) -> List[RewriteCandidate]:
#         """
#         Apply cheap pre-filters before expensive FLUX validation:
#           - Drop empty / duplicate candidates.
#           - Drop candidates that are not strictly shorter than the source
#             (when require_strict_compression).
#           - Drop candidates below the absolute token floor (`min_caption_tokens`).
#             The per-iteration target budgets are SUGGESTIONS sent to the LLM
#             for variety; they are NOT used as rejection thresholds, since the
#             objective is "shortest grammatical caption under acceptable
#             fidelity" -- aggressive compression is desired, and quality is
#             enforced by the anchor filter, the substantive-rewrite check,
#             and the post-FLUX fidelity floor.
#           - Drop near-echo candidates that don't differ enough from the original.
#           - Mark anchor preservation status (soft KG, strict quoted).
#         """
#         candidates: List[RewriteCandidate] = []
#         seen_text: Set[str] = set()

#         budgets = list(target_budgets) if target_budgets else [source_tokens]

#         # Absolute minimum -- below this a caption can no longer carry useful
#         # visual content for reconstruction.
#         min_acceptable_tokens = self.min_caption_tokens

#         # Soft KG anchor strictness: allow up to this many to be missing.
#         max_kg_miss = int(round(self.kg_anchor_max_miss_fraction * len(kg_anchors)))

#         for idx, raw in enumerate(raw_candidates):
#             text = clean_caption_spacing(raw)
#             if not text:
#                 continue

#             key = text.lower()
#             if key in seen_text:
#                 continue
#             seen_text.add(key)

#             tok = count_tokens(text)

#             target_budget = min(budgets, key=lambda b: abs(b - tok)) if budgets else tok

#             cand = RewriteCandidate(
#                 index=idx,
#                 text=text,
#                 target_token_budget=int(target_budget),
#                 token_count=tok,
#             )

#             if self.require_strict_compression and tok >= source_tokens:
#                 cand.passed_anchor_filter = False
#                 cand.missing_anchors = ["__not_shorter_than_source__"]
#                 candidates.append(cand)
#                 continue

#             if tok < min_acceptable_tokens:
#                 cand.passed_anchor_filter = False
#                 cand.missing_anchors = [
#                     f"__below_min_acceptable_tokens__({tok}<{min_acceptable_tokens})"
#                 ]
#                 candidates.append(cand)
#                 continue

#             # Reject candidates that are essentially the original caption.
#             if original_caption and not is_substantive_rewrite(
#                 original_caption,
#                 text,
#                 min_diff_ratio=self.min_rewrite_diff_ratio,
#             ):
#                 cand.passed_anchor_filter = False
#                 cand.missing_anchors = ["__too_similar_to_original__"]
#                 candidates.append(cand)
#                 continue

#             passed, missing = candidate_preserves_anchors(
#                 candidate=text,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#                 max_kg_miss_count=max_kg_miss,
#             )
#             cand.passed_anchor_filter = passed
#             cand.missing_anchors = missing
#             candidates.append(cand)

#         return candidates

#     def _candidate_acceptance_decision(
#         self,
#         candidate_D: float,
#         threshold: float,
#         candidate_comparison: Dict[str, Any],
#         baseline_comparison: Dict[str, Any],
#     ) -> Tuple[bool, str, Dict[str, Any]]:
#         """
#         Decide whether a candidate is acceptable, using floors derived
#         from the baseline (full-caption) reconstruction. Quoted/KG anchor
#         preservation is already enforced by the pre-filter, so this checks
#         only the reconstruction-side fidelity floors.
#         """
#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))
#         baseline_attr_recall = float(baseline_comparison.get("attribute_recall_mean", 0.0))

#         candidate_f1 = float(candidate_comparison.get("f1_mean", 0.0))
#         candidate_obj_recall = float(candidate_comparison.get("object_recall_mean", 0.0))
#         candidate_attr_recall = float(candidate_comparison.get("attribute_recall_mean", 0.0))

#         min_f1_allowed = max(self.min_baseline_kg_f1, baseline_f1 - self.kg_f1_floor_drop)
#         min_obj_recall_allowed = max(
#             self.min_baseline_object_recall,
#             baseline_obj_recall - self.object_recall_floor_drop,
#         )
#         min_attr_recall_allowed = max(
#             self.attribute_recall_min_floor,
#             baseline_attr_recall - self.attribute_recall_floor_drop,
#         )

#         details = {
#             "candidate_D": candidate_D,
#             "threshold": threshold,
#             "baseline_f1": baseline_f1,
#             "candidate_f1": candidate_f1,
#             "min_f1_allowed": min_f1_allowed,
#             "baseline_object_recall": baseline_obj_recall,
#             "candidate_object_recall": candidate_obj_recall,
#             "min_object_recall_allowed": min_obj_recall_allowed,
#             "baseline_attribute_recall": baseline_attr_recall,
#             "candidate_attribute_recall": candidate_attr_recall,
#             "min_attribute_recall_allowed": min_attr_recall_allowed,
#         }

#         if candidate_D > threshold:
#             return False, "rejected_above_global_threshold", details
#         if candidate_f1 < min_f1_allowed:
#             return False, "rejected_below_acceptable_kg_f1", details
#         if candidate_obj_recall < min_obj_recall_allowed:
#             return False, "rejected_below_acceptable_object_recall", details
#         if candidate_attr_recall < min_attr_recall_allowed:
#             return False, "rejected_below_acceptable_attribute_recall", details

#         return True, "accepted_under_acceptable_fidelity", details

#     # -----------------------------------------------------------------
#     # Skip path
#     # -----------------------------------------------------------------

#     def _make_skipped_result(
#         self,
#         image_id: str,
#         caption: str,
#         original_tokens: int,
#         baseline_D: float,
#         threshold: float,
#         baseline_dist: Dict[str, Any],
#         baseline_comparison: Dict[str, Any],
#         baseline_kgs: List[Dict[str, Any]],
#         image_id_safe: str,
#         start_time: float,
#     ) -> CompressionResult:
#         final_details = {k: v for k, v in baseline_dist.items() if not str(k).endswith("_std")}
#         final_details["status"] = "skipped_low_baseline_fidelity"

#         if self.save_intermediate:
#             self._save_final_kgs_alias(baseline_kgs, image_id_safe)

#         result = CompressionResult(
#             image_id=image_id,
#             status="skipped_low_baseline_fidelity",
#             original_caption=caption,
#             compressed_caption=caption,
#             original_tokens=original_tokens,
#             compressed_tokens=original_tokens,
#             compression_ratio=0.0,
#             baseline_distortion=baseline_D,
#             threshold=threshold,
#             final_distortion=baseline_D,
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=baseline_comparison,
#             protected_anchors=[],
#             quoted_anchors=[],
#             steps=[],
#             total_time=time.time() - start_time,
#         )

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     # -----------------------------------------------------------------
#     # Persistence
#     # -----------------------------------------------------------------

#     def _save_kg_debug(self, kg: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(kg_to_readable_dict(kg)), f, indent=2, ensure_ascii=False)
#         print(f"[KG Saved] {path}")

#     def _save_kg_comparison_debug(self, comparison: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(comparison), f, indent=2, ensure_ascii=False)
#         print(f"[KG Comparison Saved] {path}")

#     def _save_final_kgs_alias(self, kgs: List[Dict[str, Any]], image_id: str) -> None:
#         for idx, kg in enumerate(kgs):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             self._save_kg_debug(kg, image_id, f"final_reconstructed_kg_seed{seed}")

#     def _save_candidates_debug(
#         self,
#         image_id: str,
#         iteration: int,
#         raw_candidates: Sequence[str],
#         filtered_candidates: Sequence[RewriteCandidate],
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#     ) -> None:
#         debug_dir = self.candidate_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"iter{iteration}_candidates.json"

#         payload = {
#             "iteration": iteration,
#             "quoted_anchors": list(quoted_anchors),
#             "kg_anchors": list(kg_anchors),
#             "raw_candidates": list(raw_candidates),
#             "filtered_candidates": [asdict(c) for c in filtered_candidates],
#         }
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(payload), f, indent=2, ensure_ascii=False)
#         print(f"[Candidates Saved] {path}")

#     def _save_final_reconstruction_visuals(
#         self,
#         image_original: Image.Image,
#         image_final_recon: Optional[Image.Image],
#         final_caption: str,
#         image_id_safe: str,
#     ) -> None:
#         """
#         Save the final reconstructed image (from the COMPRESSED caption) plus
#         a side-by-side comparison. Used by downstream reporting (e.g. the
#         Excel comparison sheet). Does not raise if the final reconstruction
#         is missing.
#         """
#         debug_dir = self.output_dir / "debug_samples" / safe_filename(image_id_safe)
#         debug_dir.mkdir(parents=True, exist_ok=True)

#         final_caption_path = debug_dir / "final_caption.txt"
#         with open(final_caption_path, "w", encoding="utf-8") as f:
#             f.write(final_caption)
#         print(f"[Debug] Saved: {final_caption_path}")

#         if image_final_recon is None:
#             return

#         final_recon_path = debug_dir / "final_reconstructed.png"
#         image_final_recon.convert("RGB").save(final_recon_path)
#         print(f"[Debug] Saved: {final_recon_path}")

#         # Side-by-side: original vs final compressed reconstruction.
#         left = image_original.convert("RGB").resize((512, 512))
#         right = image_final_recon.convert("RGB").resize((512, 512))
#         canvas = Image.new("RGB", (1024, 560), color="white")
#         canvas.paste(left, (0, 48))
#         canvas.paste(right, (512, 48))
#         draw = ImageDraw.Draw(canvas)
#         draw.text((170, 15), "Original Image", fill="black")
#         draw.text((595, 15), "Reconstructed from Compressed Caption", fill="black")
#         side_path = debug_dir / "side_by_side_original_vs_final.png"
#         canvas.save(side_path)
#         print(f"[Debug] Saved: {side_path}")

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(result), f, indent=2, ensure_ascii=False)
#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             print("[Summary] No results to summarize.")
#             return

#         compression_ratios = [r.compression_ratio for r in results]
#         final_distortions = [r.final_distortion for r in results]
#         baseline_distortions = [r.baseline_distortion for r in results]
#         original_tokens = [r.original_tokens for r in results]
#         compressed_tokens = [r.compressed_tokens for r in results]
#         times = [r.total_time for r in results]

#         statuses: Dict[str, int] = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1

#         summary = {
#             "num_samples": len(results),
#             "status_counts": statuses,
#             "avg_compression_ratio": float(np.mean(compression_ratios)),
#             "std_compression_ratio": float(np.std(compression_ratios)),
#             "avg_baseline_distortion": float(np.mean(baseline_distortions)),
#             "avg_final_distortion": float(np.mean(final_distortions)),
#             "avg_original_tokens": float(np.mean(original_tokens)),
#             "avg_compressed_tokens": float(np.mean(compressed_tokens)),
#             "avg_time_seconds": float(np.mean(times)),
#             "total_time_hours": float(np.sum(times) / 3600.0),
#             "use_relative_threshold": self.use_relative_threshold,
#             "epsilon": self.epsilon,
#             "absolute_threshold": self.absolute_threshold,
#             "num_seeds": len(self.t2i_seeds),
#             "seeds": self.t2i_seeds,
#             "min_baseline_kg_f1": self.min_baseline_kg_f1,
#             "min_baseline_object_recall": self.min_baseline_object_recall,
#             "num_candidates_per_iteration": self.num_candidates_per_iteration,
#             "max_iterations": self.max_iterations,
#             "min_target_token_ratio": self.min_target_token_ratio,
#             "max_target_token_ratio": self.max_target_token_ratio,
#             "fallback_strategy": self.fallback_strategy,
#         }

#         summary_path = self.result_dir / "summary.json"
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(summary), f, indent=2)

#         print("\n" + "=" * 70)
#         print(f"SUMMARY over {len(results)} samples")
#         print(f"Status counts:          {statuses}")
#         print(f"Avg compression ratio:  {summary['avg_compression_ratio']:.2%}")
#         print(f"Avg baseline D:         {summary['avg_baseline_distortion']:.4f}")
#         print(f"Avg final D:            {summary['avg_final_distortion']:.4f}")
#         print(
#             f"Avg tokens:             "
#             f"{summary['avg_original_tokens']:.1f} -> "
#             f"{summary['avg_compressed_tokens']:.1f}"
#         )
#         print(f"Total time:             {summary['total_time_hours']:.2f} hours")
#         print(f"Saved summary:          {summary_path}")
#         print("=" * 70)

# """
# KG-RD Compressor - Grammar-Aware Rewrite Compression
# ====================================================

# Replaces greedy span deletion with rewrite-and-validate compression.

# Goal:
#     Maximum compression under acceptable reconstruction fidelity.

# Pipeline:
#     1. Extract KG from the original image.
#     2. Reconstruct image from full caption -> baseline distortion + threshold.
#     3. Apply baseline gate (skip if reconstruction is already poor).
#     4. Extract protected anchors:
#         - Quoted strings from caption (must appear verbatim).
#         - Key object/attribute terms from the original-image KG.
#     5. For up to max_iterations rounds:
#         a. Generate K grammatical candidate rewrites at varied token budgets
#            via a CaptionRewriter (text-only LLM).
#         b. Filter candidates that drop any protected anchor.
#         c. For each surviving candidate (shortest first):
#             - Reconstruct via FLUX.
#             - Extract KG via Qwen2.5-VL.
#             - Compute distortion + KG comparison.
#             - Check fidelity floors.
#         d. Select the shortest candidate that passes the fidelity floor.
#            If shorter than current, replace current and continue.
#            Else stop.
#     6. If no candidate ever passes, apply fallback_strategy.
# """

# import json
# import re
# import time
# from abc import ABC, abstractmethod
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# import torch
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     canonicalize_for_matching,
#     compare_scene_graphs,
#     normalize_label,
#     semantic_text_similarity,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor


# # =====================================================================
# # Data classes
# # =====================================================================

# @dataclass
# class RewriteCandidate:
#     """One candidate rewrite produced by the rewriter."""
#     index: int
#     text: str
#     target_token_budget: int
#     token_count: int
#     missing_anchors: List[str] = field(default_factory=list)
#     passed_anchor_filter: bool = False


# @dataclass
# class CompressionStep:
#     """One validated candidate during the compression loop."""
#     iteration: int
#     candidate_index: int
#     candidate_caption: str
#     target_token_budget: int
#     caption_before: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     distortion_details: Dict[str, Any]
#     missing_anchors: List[str]
#     time_seconds: float


# @dataclass
# class CompressionResult:
#     """Final result for one image-caption pair."""
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     protected_anchors: List[str]
#     quoted_anchors: List[str]
#     steps: List[CompressionStep]
#     total_time: float


# # =====================================================================
# # Helpers
# # =====================================================================

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"\(\s+", "(", text)
#     text = re.sub(r"\s+\)", ")", text)
#     text = text.replace(" ,", ",").replace(" .", ".")
#     return text.strip()


# def safe_filename(name: str) -> str:
#     name = str(name)
#     name = re.sub(r"[^\w\-.]+", "_", name)
#     return name[:180]


# def dataclass_is_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if dataclass_is_instance(obj):
#         return make_json_serializable(asdict(obj))
#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]
#     if isinstance(obj, tuple):
#         return list(obj)
#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]
#     if isinstance(obj, np.integer):
#         return int(obj)
#     if isinstance(obj, np.floating):
#         return float(obj)
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     return obj


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     tuples = kg.get("tuples", set())
#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "tuples": [list(t) for t in sorted(list(tuples), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def print_kg(title: str, kg: Dict[str, Any], max_items: int = 40) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)

#     print("\nOBJECTS:")
#     for obj in kg.get("objects", [])[:max_items]:
#         label = obj.get("label", obj.get("name", ""))
#         attrs = obj.get("attributes", [])
#         print(f"  - {label} | attributes={attrs}")

#     print("\nATTRIBUTES:")
#     for attr in kg.get("attributes", [])[:max_items]:
#         print(f"  - ({attr.get('object', '')}, has_attribute, {attr.get('attribute', '')})")

#     print("\nRELATIONS:")
#     for rel in kg.get("relations", [])[:max_items]:
#         pred = rel.get("predicate", rel.get("relation", ""))
#         print(f"  - ({rel.get('subject', '')}, {pred}, {rel.get('object', '')})")

#     print("=" * 100)


# def print_kg_comparison(title: str, comparison: Dict[str, Any], max_items: int = 60) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)
#     print(
#         f"F1 mean={comparison.get('f1_mean', 0.0):.4f} | "
#         f"Recall mean={comparison.get('recall_mean', 0.0):.4f} | "
#         f"Object recall={comparison.get('object_recall_mean', 0.0):.4f} | "
#         f"Attribute recall={comparison.get('attribute_recall_mean', 0.0):.4f} | "
#         f"Relation recall={comparison.get('relation_recall_mean', 0.0):.4f}"
#     )

#     print("\nPRESERVED:")
#     for x in comparison.get("preserved_text", [])[:max_items]:
#         print("  +", x)

#     print("\nMISSING:")
#     for x in comparison.get("missing_text", [])[:max_items]:
#         print("  -", x)

#     print("\nEXTRA / HALLUCINATED:")
#     for x in comparison.get("extra_text", [])[:max_items]:
#         print("  *", x)

#     print("=" * 100)


# def save_debug_sample_visuals(
#     image_original: Image.Image,
#     image_reconstructed: Image.Image,
#     caption: str,
#     image_id: str,
#     output_dir: Path,
# ) -> None:
#     debug_dir = output_dir / "debug_samples" / safe_filename(image_id)
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     original_path = debug_dir / "original_image.png"
#     reconstructed_path = debug_dir / "reconstructed_from_caption.png"
#     caption_path = debug_dir / "caption.txt"
#     side_by_side_path = debug_dir / "side_by_side_original_vs_reconstructed.png"

#     image_original_rgb = image_original.convert("RGB")
#     image_reconstructed_rgb = image_reconstructed.convert("RGB")

#     image_original_rgb.save(original_path)
#     image_reconstructed_rgb.save(reconstructed_path)

#     with open(caption_path, "w", encoding="utf-8") as f:
#         f.write(caption)

#     left = image_original_rgb.resize((512, 512))
#     right = image_reconstructed_rgb.resize((512, 512))
#     canvas = Image.new("RGB", (1024, 560), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))

#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), "Original Image", fill="black")
#     draw.text((655, 15), "Reconstructed from Caption", fill="black")
#     canvas.save(side_by_side_path)

#     print(f"[Debug] Saved: {original_path}")
#     print(f"[Debug] Saved: {reconstructed_path}")
#     print(f"[Debug] Saved: {side_by_side_path}")


# # =====================================================================
# # Anchor extraction & verification
# # =====================================================================

# # All quote-pair styles we accept around an anchor. The first element of
# # each tuple is the opener, the second is the closer.
# _QUOTE_PAIRS = [
#     ('"', '"'),
#     ("\u201c", "\u201d"),  # smart double
#     ("'", "'"),
#     ("\u2018", "\u2019"),  # smart single
#     ("`", "`"),
# ]


# def _extract_all_quoted_substrings(text: str) -> List[str]:
#     """
#     Return every quoted substring in `text`, regardless of quote style.

#     Includes single and double quotes (straight + smart) and backticks.
#     Returned strings are stripped but case-preserved.
#     """
#     if not text:
#         return []

#     pattern = (
#         r'"([^"]+)"'
#         r"|\u201c([^\u201c\u201d]+)\u201d"
#         r"|`([^`]+)`"
#         r"|'([^']+)'"
#         r"|\u2018([^\u2018\u2019]+)\u2019"
#     )

#     out: List[str] = []
#     for m in re.finditer(pattern, text):
#         for g in m.groups():
#             if g is not None:
#                 s = g.strip()
#                 if s:
#                     out.append(s)
#                 break
#     return out


# def extract_quoted_anchors(caption: str) -> List[str]:
#     """
#     Extract quoted strings from a caption. These must be preserved as
#     quoted strings (any quote style) in any candidate rewrite.

#     Originals nearly always use double quotes; we restrict extraction to
#     double / smart-double here to avoid possessive false positives like
#     "the dog's bone". Candidate matching, however, accepts ANY quote style
#     (see candidate_preserves_anchors).
#     """
#     if not caption:
#         return []

#     raw = re.findall(r'"([^"]+)"|\u201c([^\u201c\u201d]+)\u201d', caption)

#     anchors: List[str] = []
#     seen: Set[str] = set()
#     for groups in raw:
#         s = next((g for g in groups if g), "")
#         s_stripped = s.strip()
#         if not s_stripped:
#             continue
#         key = s_stripped.lower()
#         if key in seen:
#             continue
#         seen.add(key)
#         anchors.append(s_stripped)

#     return anchors


# # Visual-marker keywords used to up-rank KG anchors that refer to text or
# # painted markings, since these are usually the most reconstruction-critical.
# _VISUAL_MARKER_KEYWORDS = {
#     "text", "word", "words", "letter", "letters", "writing",
#     "marking", "markings", "sign", "signs", "label", "paint",
#     "painted", "graffiti",
# }

# _HIGH_VALUE_ATTRIBUTES = {
#     "red", "orange", "yellow", "blue", "green", "white", "black",
#     "gray", "grey", "brown", "purple", "pink",
#     "faded", "old", "new", "bright", "dark",
#     "horizontal", "vertical", "diagonal",
#     "large", "small", "tall", "short", "long",
# }


# def extract_kg_anchors(
#     kg: Dict[str, Any],
#     max_anchors: int = 6,
# ) -> List[str]:
#     """
#     Extract the most important visual terms from the original-image KG.
#     These guide the rewriter and serve as soft preservation constraints.

#     Heuristic:
#       1. Score each object by:
#          - +2 if it is a visual-marker term (text/word/marking/sign/...)
#          - +1 per attribute attached to it
#          - +1 per relation it participates in
#       2. Take the top-K objects by score.
#       3. For each chosen object, also include its single most distinctive
#          attribute (color/state) when it falls in a high-value vocabulary.
#     """
#     objects = kg.get("objects", []) or []
#     relations = kg.get("relations", []) or []
#     attributes = kg.get("attributes", []) or []

#     if not objects:
#         return []

#     rel_count: Dict[str, int] = {}
#     for rel in relations:
#         s = normalize_label(rel.get("subject", ""))
#         o = normalize_label(rel.get("object", ""))
#         if s:
#             rel_count[s] = rel_count.get(s, 0) + 1
#         if o:
#             rel_count[o] = rel_count.get(o, 0) + 1

#     attrs_by_obj: Dict[str, List[str]] = {}
#     for attr in attributes:
#         obj_name = normalize_label(attr.get("object", ""))
#         attr_val = normalize_label(attr.get("attribute", ""))
#         if obj_name and attr_val:
#             attrs_by_obj.setdefault(obj_name, []).append(attr_val)

#     scored = []
#     seen_labels: Set[str] = set()

#     for obj in objects:
#         label = normalize_label(obj.get("label", obj.get("name", "")))
#         if not label or label in seen_labels:
#             continue
#         seen_labels.add(label)

#         score = 0
#         if any(kw in label.split() for kw in _VISUAL_MARKER_KEYWORDS):
#             score += 2

#         embedded_attrs = obj.get("attributes", []) or []
#         score += min(len(embedded_attrs), 3)
#         score += min(len(attrs_by_obj.get(label, [])), 3)
#         score += min(rel_count.get(label, 0), 3)

#         all_attrs = list(dict.fromkeys(
#             [normalize_label(a) for a in embedded_attrs] +
#             attrs_by_obj.get(label, [])
#         ))

#         scored.append((score, label, all_attrs))

#     scored.sort(key=lambda x: (-x[0], x[1]))

#     anchors: List[str] = []
#     used: Set[str] = set()

#     for score, label, attrs in scored:
#         if len(anchors) >= max_anchors:
#             break
#         if label in used:
#             continue

#         # Pair the object with its most distinctive high-value attribute.
#         best_attr = None
#         for a in attrs:
#             if a in _HIGH_VALUE_ATTRIBUTES:
#                 best_attr = a
#                 break

#         anchor = f"{best_attr} {label}" if best_attr else label
#         anchors.append(anchor)
#         used.add(label)

#     return anchors


# def candidate_preserves_anchors(
#     candidate: str,
#     quoted_anchors: Sequence[str],
#     kg_anchors: Sequence[str],
#     kg_match_threshold: float = 0.55,
# ) -> Tuple[bool, List[str]]:
#     """
#     Check whether a candidate preserves required anchors.

#     - Quoted anchors: the anchor text must appear inside a quoted substring
#       of the candidate. Any quote style is accepted ("...", '...', smart
#       quotes, backticks) so the LLM's choice of quote characters doesn't
#       cause false rejections.
#     - KG anchors: at least one content-word overlap with the candidate, or
#       semantic similarity >= kg_match_threshold against the candidate text.

#     Returns (passed, missing_anchors_list).
#     """
#     if not candidate:
#         return False, [f'"{a}"' for a in quoted_anchors] + list(kg_anchors)

#     candidate_norm = canonicalize_for_matching(candidate)

#     # Collect every quoted substring in the candidate, normalized.
#     candidate_quoted_lower = {
#         s.strip().lower() for s in _extract_all_quoted_substrings(candidate)
#     }

#     missing: List[str] = []

#     # 1. Each quoted anchor must appear as a quoted substring (any style).
#     for anchor in quoted_anchors:
#         anchor_key = anchor.strip().lower()
#         if anchor_key in candidate_quoted_lower:
#             continue
#         # Final fallback: bare appearance (no quotes) is still better than
#         # nothing for short multi-word anchors like FIRE LANE that are
#         # visually distinctive even unquoted. Single-character anchors must
#         # be quoted to count.
#         if len(anchor_key) >= 3 and anchor_key in candidate.lower():
#             continue
#         missing.append(f'"{anchor}"')

#     # 2. KG anchors: lexical or semantic preservation.
#     for anchor in kg_anchors:
#         anchor_norm = canonicalize_for_matching(anchor)
#         if not anchor_norm:
#             continue

#         if anchor_norm in candidate_norm:
#             continue

#         anchor_words = [w for w in anchor_norm.split() if len(w) > 2]
#         if anchor_words and any(w in candidate_norm.split() for w in anchor_words):
#             continue

#         sim = semantic_text_similarity(anchor_norm, candidate_norm)
#         if sim >= kg_match_threshold:
#             continue

#         missing.append(anchor)

#     return (len(missing) == 0), missing


# # =====================================================================
# # Rewriter interface
# # =====================================================================

# class CaptionRewriter(ABC):
#     """
#     Generator interface used by KGRDCompressor.

#     Any rewriter (Qwen-VL, small text LLM, API) just needs to implement
#     `rewrite(...)` and return a list of grammatical candidate strings.
#     """

#     @abstractmethod
#     def rewrite(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> List[str]:
#         ...


# class QwenVLRewriter(CaptionRewriter):
#     """
#     Text-only rewriter that reuses the Qwen2.5-VL model already loaded
#     inside VLMSceneGraphExtractor. No additional GPU memory is consumed.
#     """

#     def __init__(
#         self,
#         vlm_extractor: VLMSceneGraphExtractor,
#         max_new_tokens: int = 512,
#         temperature: float = 0.7,
#         top_p: float = 0.9,
#     ):
#         self.model = vlm_extractor.model
#         self.processor = vlm_extractor.processor
#         self.device = vlm_extractor.device
#         self.max_new_tokens = int(max_new_tokens)
#         self.temperature = float(temperature)
#         self.top_p = float(top_p)

#     def _build_prompt(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> str:
#         original_tokens = count_tokens(original_caption)

#         if quoted_anchors:
#             quoted_lines = []
#             for i, a in enumerate(quoted_anchors, 1):
#                 note = "  <- single character, but it IS a real anchor; preserve it" if len(a) == 1 else ""
#                 quoted_lines.append(f'  ({i}) "{a}"{note}')
#             quoted_block = "\n".join(quoted_lines)
#             n_quoted = len(quoted_anchors)
#         else:
#             quoted_block = "  (none)"
#             n_quoted = 0

#         if kg_anchors:
#             kg_block = "\n".join(f"  ({i}) {a}" for i, a in enumerate(kg_anchors, 1))
#             n_kg = len(kg_anchors)
#         else:
#             kg_block = "  (none)"
#             n_kg = 0

#         budgets = list(target_token_budgets)
#         if budgets:
#             min_budget = min(budgets)
#             max_budget = max(budgets)
#         else:
#             min_budget = max_budget = max(8, original_tokens // 2)

#         budget_lines = "\n".join(
#             f"  - Variant {i + 1}: ~{b} tokens"
#             for i, b in enumerate(budgets)
#         ) if budgets else f"  - Variants: ~{min_budget} to ~{max_budget} tokens"

#         return f"""You compress image captions while preserving the visual content needed to reconstruct the image with a text-to-image model.

# ORIGINAL CAPTION ({original_tokens} tokens):
# {original_caption}

# ==========================================================
# HARD REQUIREMENT 1 - QUOTED STRINGS (visible text/markings)
# ==========================================================
# The caption contains {n_quoted} quoted string(s). Each names visible TEXT or
# MARKINGS in the image. EVERY variant MUST include EVERY quoted string,
# each surrounded by quotation marks. Use straight double quotes ".." (preferred)
# or single quotes '..' - either is accepted.

# {quoted_block}

# A variant that omits ANY of these quoted strings is INVALID, even the
# single-character ones. Repeat: every variant must contain all {n_quoted}.

# ==========================================================
# HARD REQUIREMENT 2 - VISUAL ANCHORS
# ==========================================================
# Every variant must also reference each of these visual elements (verbatim
# or close paraphrase):

# {kg_block}

# ==========================================================
# HARD REQUIREMENT 3 - WRITE FULL PHRASES, NOT WORD LISTS
# ==========================================================
# Variants must read as natural English: a sentence, or a noun phrase with
# proper modifiers / clauses. They MUST NOT be a comma-separated word list.

# GOOD (reads naturally):
#   Faded street corner with orange "ROW" sprayed in the middle, a horizontal "i" above, and a faded red "FIRE LANE" line on the curb.

# BAD (telegraphic word list - do NOT produce this):
#   Orange "ROW", red 'FIRE LANE', curb, sidewalk.

# ==========================================================
# TASK
# ==========================================================
# Produce {num_candidates} variants. Variant 1 should be the shortest. Each
# variant should target approximately the following length:

# {budget_lines}

# Hard floor: NO variant may be shorter than {min_budget} tokens.
# Hard ceiling: every variant must be SHORTER than the original ({original_tokens} tokens).
# Do not introduce objects or details not present in the original caption.

# ==========================================================
# OUTPUT FORMAT
# ==========================================================
# Return ONLY a JSON array of strings, ordered shortest first.
# No markdown fences. No numbering. No explanation. Just the JSON array.

# Example output (illustrative, ignore content):
# ["shortest variant here", "medium variant here", "longest variant here"]
# """

#     def _parse_json_array(self, text: str) -> List[str]:
#         """
#         Robustly parse a JSON array of strings from LLM output.
#         """
#         if not isinstance(text, str):
#             return []

#         text = text.strip()
#         text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"```", "", text).strip()

#         # Try direct parse.
#         try:
#             parsed = json.loads(text)
#             if isinstance(parsed, list):
#                 return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#         except Exception:
#             pass

#         # Find the first [...] block.
#         match = re.search(r"\[.*\]", text, flags=re.DOTALL)
#         if match:
#             try:
#                 parsed = json.loads(match.group(0))
#                 if isinstance(parsed, list):
#                     return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
#             except Exception:
#                 pass

#         # Last resort: extract quoted lines.
#         lines = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
#         return [s.strip() for s in lines if s.strip()]

#     @torch.no_grad()
#     def _generate_text(self, prompt: str) -> str:
#         messages = [
#             {"role": "user", "content": [{"type": "text", "text": prompt}]}
#         ]

#         text_input = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         # Try the multimodal processor in text-only mode first.
#         # Fall back to the underlying tokenizer if the processor requires images.
#         try:
#             inputs = self.processor(
#                 text=[text_input],
#                 padding=True,
#                 return_tensors="pt",
#             )
#         except Exception:
#             inputs = self.processor.tokenizer(
#                 text_input,
#                 padding=True,
#                 return_tensors="pt",
#             )

#         if hasattr(inputs, "to"):
#             inputs = inputs.to(self.model.device)
#         else:
#             inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

#         generated_ids = self.model.generate(
#             **inputs,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=True,
#             temperature=self.temperature,
#             top_p=self.top_p,
#         )

#         input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
#         trimmed = [out[len(inp):] for inp, out in zip(input_ids, generated_ids)]

#         if hasattr(self.processor, "batch_decode"):
#             output_text = self.processor.batch_decode(
#                 trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )[0]
#         else:
#             output_text = self.processor.tokenizer.batch_decode(
#                 trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )[0]

#         return output_text.strip()

#     def rewrite(
#         self,
#         original_caption: str,
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         target_token_budgets: Sequence[int],
#         num_candidates: int,
#     ) -> List[str]:
#         prompt = self._build_prompt(
#             original_caption=original_caption,
#             quoted_anchors=quoted_anchors,
#             kg_anchors=kg_anchors,
#             target_token_budgets=target_token_budgets,
#             num_candidates=num_candidates,
#         )

#         try:
#             raw_output = self._generate_text(prompt)
#         except Exception as exc:
#             print(f"[Rewriter] Generation failed: {exc}")
#             return []

#         candidates = self._parse_json_array(raw_output)
#         candidates = [clean_caption_spacing(c) for c in candidates]
#         candidates = [c for c in candidates if c]

#         # Deduplicate while preserving order.
#         seen: Set[str] = set()
#         unique: List[str] = []
#         for c in candidates:
#             key = c.lower()
#             if key in seen:
#                 continue
#             seen.add(key)
#             unique.append(c)

#         return unique


# # =====================================================================
# # Main compressor
# # =====================================================================

# class KGRDCompressor:
#     """
#     KG-guided Rate-Distortion Caption Compressor (rewrite-based).

#     Replaces greedy span deletion with grammar-aware rewrite candidates
#     validated through the FLUX -> Qwen2.5-VL -> distortion pipeline.
#     """

#     def __init__(
#         self,
#         # --- Scene graph extractor (Qwen2.5-VL) ---
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,

#         # --- T2I (FLUX) ---
#         t2i_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "float16",
#         t2i_steps: int = 30,
#         t2i_guidance: float = 7.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,

#         # --- Distortion metric ---
#         alpha: float = 0.4,
#         beta: float = 0.4,
#         gamma: float = 0.2,
#         clip_model: str = "ViT-L-14",
#         clip_pretrained: str = "openai",
#         lpips_net: str = "alex",
#         metric_device: str = "auto",

#         # --- Threshold ---
#         use_relative_threshold: bool = True,
#         epsilon: float = 0.10,
#         absolute_threshold: float = 0.25,

#         # --- Compression loop ---
#         min_caption_tokens: int = 5,
#         max_iterations: int = 2,
#         num_candidates_per_iteration: int = 5,
#         min_target_token_ratio: float = 0.30,
#         max_target_token_ratio: float = 0.80,
#         require_strict_compression: bool = True,

#         # --- Baseline gate ---
#         min_baseline_kg_f1: float = 0.05,
#         min_baseline_object_recall: float = 0.10,

#         # --- Fidelity floors for accepting a candidate ---
#         kg_f1_floor_drop: float = 0.08,
#         object_recall_floor_drop: float = 0.05,
#         attribute_recall_floor_drop: float = 0.12,
#         attribute_recall_min_floor: float = 0.10,

#         # --- Anchor extraction ---
#         max_kg_anchors: int = 6,

#         # --- Fallback if no candidate passes ---
#         fallback_strategy: str = "keep_original",   # keep_original | best_candidate

#         # --- Rewriter ---
#         rewriter: Optional[CaptionRewriter] = None,
#         rewriter_max_new_tokens: int = 512,
#         rewriter_temperature: float = 0.7,
#         rewriter_top_p: float = 0.9,

#         # --- I/O ---
#         save_intermediate: bool = True,
#         output_dir: str = "./outputs",
#     ):
#         self.t2i_seeds = t2i_seeds or [42]

#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)

#         self.min_caption_tokens = int(min_caption_tokens)
#         self.max_iterations = int(max_iterations)
#         self.num_candidates_per_iteration = int(num_candidates_per_iteration)
#         self.min_target_token_ratio = float(min_target_token_ratio)
#         self.max_target_token_ratio = float(max_target_token_ratio)
#         self.require_strict_compression = bool(require_strict_compression)

#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)

#         self.kg_f1_floor_drop = float(kg_f1_floor_drop)
#         self.object_recall_floor_drop = float(object_recall_floor_drop)
#         self.attribute_recall_floor_drop = float(attribute_recall_floor_drop)
#         self.attribute_recall_min_floor = float(attribute_recall_min_floor)

#         self.max_kg_anchors = int(max_kg_anchors)

#         if fallback_strategy not in {"keep_original", "best_candidate"}:
#             raise ValueError(
#                 f"fallback_strategy must be one of "
#                 f"'keep_original' | 'best_candidate', got: {fallback_strategy}"
#             )
#         self.fallback_strategy = fallback_strategy

#         self.save_intermediate = bool(save_intermediate)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"
#         self.candidate_debug_dir = self.output_dir / "candidates"
#         self.result_dir.mkdir(parents=True, exist_ok=True)
#         self.kg_debug_dir.mkdir(parents=True, exist_ok=True)
#         self.candidate_debug_dir.mkdir(parents=True, exist_ok=True)

#         print("=" * 70)
#         print("KG-RD Compressor: Initializing rewrite-based pipeline")
#         print("=" * 70)

#         print("\n[1/3] Loading VLM scene graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model,
#             device=scene_graph_device,
#             dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens,
#             temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample,
#             max_objects=max_objects,
#             max_relations=max_relations,
#         )

#         print("\n[2/3] Loading text-to-image reconstructor...")
#         self.t2i = T2IReconstructor(
#             model_name=t2i_model,
#             device=t2i_device,
#             dtype=t2i_dtype,
#             num_inference_steps=t2i_steps,
#             guidance_scale=t2i_guidance,
#             image_size=t2i_image_size,
#         )

#         print("\n[3/3] Loading distortion metric...")
#         self.distortion = DistortionMetric(
#             alpha=alpha,
#             beta=beta,
#             gamma=gamma,
#             clip_model_name=clip_model,
#             clip_pretrained=clip_pretrained,
#             lpips_net=lpips_net,
#             device=metric_device,
#         )

#         # Default rewriter shares Qwen2.5-VL with the scene graph extractor.
#         if rewriter is None:
#             print("\n[Rewriter] Using QwenVLRewriter (shared with scene graph VLM).")
#             self.rewriter: CaptionRewriter = QwenVLRewriter(
#                 vlm_extractor=self.sg_extractor,
#                 max_new_tokens=rewriter_max_new_tokens,
#                 temperature=rewriter_temperature,
#                 top_p=rewriter_top_p,
#             )
#         else:
#             print(f"\n[Rewriter] Using injected rewriter: {type(rewriter).__name__}")
#             self.rewriter = rewriter

#         print("\n" + "=" * 70)
#         if self.use_relative_threshold:
#             print(f"Pipeline ready | relative threshold: delta = D0 * (1 + {self.epsilon})")
#         else:
#             print(f"Pipeline ready | absolute threshold: delta = {self.absolute_threshold}")
#         print(
#             f"Baseline gate | min KG-F1={self.min_baseline_kg_f1}, "
#             f"min object recall={self.min_baseline_object_recall}"
#         )
#         print(
#             f"Rewrite | candidates/iter={self.num_candidates_per_iteration}, "
#             f"max_iters={self.max_iterations}, "
#             f"target_ratio=[{self.min_target_token_ratio:.0%}..{self.max_target_token_ratio:.0%}], "
#             f"fallback={self.fallback_strategy}"
#         )
#         print("=" * 70 + "\n")

#     # -----------------------------------------------------------------
#     # Public API
#     # -----------------------------------------------------------------

#     def compress_single(
#         self,
#         image: Image.Image,
#         caption: str,
#         image_id: str = "unknown",
#     ) -> CompressionResult:
#         start_time = time.time()

#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)

#         if original_tokens <= self.min_caption_tokens:
#             raise ValueError(
#                 f"Caption has only {original_tokens} tokens, "
#                 f"min_caption_tokens={self.min_caption_tokens}."
#             )

#         image_id_safe = safe_filename(image_id)

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | original_tokens={original_tokens}")
#         print("-" * 70)

#         steps: List[CompressionStep] = []

#         # --- 1. Original-image KG ---
#         print("[1] Extracting KG from original image...")
#         kg_original = self.sg_extractor.extract(image, image_id=image_id)
#         print(
#             f"    Original KG: {len(kg_original.get('objects', []))} objects, "
#             f"{len(kg_original.get('attributes', []))} attributes, "
#             f"{len(kg_original.get('relations', []))} relations"
#         )
#         if self.save_intermediate:
#             self._save_kg_debug(kg_original, image_id_safe, "original_image_kg")
#         print_kg("ORIGINAL IMAGE KG", kg_original, max_items=40)

#         # --- 2. Baseline reconstruction & threshold ---
#         print("[2] Generating baseline reconstruction from full caption...")
#         baseline_images, baseline_kgs = self._reconstruct_and_extract_kgs(
#             caption=caption,
#             image_id=image_id_safe,
#             prefix="baseline_full_caption",
#         )

#         if self.save_intermediate and baseline_images:
#             save_debug_sample_visuals(
#                 image_original=image,
#                 image_reconstructed=baseline_images[0],
#                 caption=caption,
#                 image_id=image_id_safe,
#                 output_dir=self.output_dir,
#             )
#         if baseline_kgs:
#             print_kg("BASELINE RECONSTRUCTED KG", baseline_kgs[0], max_items=40)

#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image,
#             images_recon=baseline_images,
#             kg_original=kg_original,
#             kgs_reconstructed=baseline_kgs,
#         )
#         baseline_D = float(baseline_dist["D"])

#         if self.use_relative_threshold:
#             threshold = self.distortion.compute_relative_threshold(
#                 baseline_distortion=baseline_D,
#                 epsilon=self.epsilon,
#             )
#         else:
#             threshold = self.absolute_threshold

#         baseline_comparison = self._aggregate_kg_comparisons(kg_original, baseline_kgs)
#         if self.save_intermediate:
#             self._save_kg_comparison_debug(
#                 baseline_comparison, image_id_safe, "baseline_kg_comparison"
#             )

#         print(f"    Baseline distortion D0 = {baseline_D:.4f}")
#         print(f"    Threshold delta       = {threshold:.4f}")
#         print_kg_comparison("BASELINE KG COMPARISON", baseline_comparison)

#         # --- 3. Baseline gate ---
#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))

#         if baseline_f1 < self.min_baseline_kg_f1 or baseline_obj_recall < self.min_baseline_object_recall:
#             print("\n[SKIP] Baseline reconstruction fidelity is too low.")
#             return self._make_skipped_result(
#                 image_id=image_id,
#                 caption=caption,
#                 original_tokens=original_tokens,
#                 baseline_D=baseline_D,
#                 threshold=threshold,
#                 baseline_dist=baseline_dist,
#                 baseline_comparison=baseline_comparison,
#                 baseline_kgs=baseline_kgs,
#                 image_id_safe=image_id_safe,
#                 start_time=start_time,
#             )

#         # --- 4. Anchor extraction ---
#         quoted_anchors = extract_quoted_anchors(caption)
#         kg_anchors = extract_kg_anchors(kg_original, max_anchors=self.max_kg_anchors)

#         print(f"\n[4] Protected anchors:")
#         print(f"    Quoted: {quoted_anchors if quoted_anchors else '(none)'}")
#         print(f"    KG    : {kg_anchors if kg_anchors else '(none)'}")

#         # --- 5. Iterative rewrite-and-validate loop ---
#         current_caption = caption
#         current_kgs = baseline_kgs
#         current_dist = baseline_dist
#         best_below_floor: Optional[Dict[str, Any]] = None  # for best_candidate fallback

#         for iteration in range(1, self.max_iterations + 1):
#             current_tokens = count_tokens(current_caption)
#             if current_tokens <= self.min_caption_tokens:
#                 print(f"[Iter {iteration}] Already at minimum length. Stopping.")
#                 break

#             print(f"\n[Iter {iteration}] Current tokens: {current_tokens}")

#             target_budgets = self._compute_target_budgets(
#                 source_tokens=current_tokens,
#                 original_tokens=original_tokens,
#             )
#             print(f"    Target token budgets: {target_budgets}")

#             # 5a. Generate candidates.
#             t_gen = time.time()
#             raw_candidates = self.rewriter.rewrite(
#                 original_caption=current_caption,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#                 target_token_budgets=target_budgets,
#                 num_candidates=self.num_candidates_per_iteration,
#             )
#             print(f"    Generated {len(raw_candidates)} candidates in {time.time() - t_gen:.1f}s")

#             if not raw_candidates:
#                 print(f"[Iter {iteration}] Rewriter returned no candidates. Stopping.")
#                 break

#             # 5b. Anchor-filter and dedupe.
#             candidates = self._filter_candidates(
#                 raw_candidates=raw_candidates,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#                 source_tokens=current_tokens,
#                 target_budgets=target_budgets,
#             )

#             if self.save_intermediate:
#                 self._save_candidates_debug(
#                     image_id=image_id_safe,
#                     iteration=iteration,
#                     raw_candidates=raw_candidates,
#                     filtered_candidates=candidates,
#                     quoted_anchors=quoted_anchors,
#                     kg_anchors=kg_anchors,
#                 )

#             survivors = [c for c in candidates if c.passed_anchor_filter]
#             print(f"    Anchor filter: {len(survivors)}/{len(candidates)} survived")

#             for c in candidates:
#                 tag = "OK " if c.passed_anchor_filter else "MISS"
#                 miss_str = f" missing={c.missing_anchors}" if c.missing_anchors else ""
#                 print(f"      [{tag}] ({c.token_count} tok) {c.text[:90]}{'...' if len(c.text) > 90 else ''}{miss_str}")

#             if not survivors:
#                 print(f"[Iter {iteration}] No candidate preserved all anchors. Stopping.")
#                 break

#             # 5c. Validate shortest first.
#             survivors.sort(key=lambda c: (c.token_count, c.index))

#             iteration_winner: Optional[Dict[str, Any]] = None

#             for cand in survivors:
#                 step_start = time.time()
#                 print(f"    [Validate] candidate#{cand.index} ({cand.token_count} tok): {cand.text}")

#                 cand_images, cand_kgs = self._reconstruct_and_extract_kgs(
#                     caption=cand.text,
#                     image_id=image_id_safe,
#                     prefix=f"iter{iteration}_cand{cand.index}",
#                 )

#                 cand_dist = self.distortion.compute_averaged_distortion(
#                     image_orig=image,
#                     images_recon=cand_images,
#                     kg_original=kg_original,
#                     kgs_reconstructed=cand_kgs,
#                 )
#                 cand_D = float(cand_dist["D"])
#                 cand_comparison = self._aggregate_kg_comparisons(kg_original, cand_kgs)

#                 passed, reason, accept_details = self._candidate_acceptance_decision(
#                     candidate_D=cand_D,
#                     threshold=threshold,
#                     candidate_comparison=cand_comparison,
#                     baseline_comparison=baseline_comparison,
#                 )

#                 details = {
#                     k: v for k, v in cand_dist.items() if not str(k).endswith("_std")
#                 }
#                 details.update({
#                     "candidate_kg_f1": cand_comparison.get("f1_mean", 0.0),
#                     "candidate_object_recall": cand_comparison.get("object_recall_mean", 0.0),
#                     "candidate_attribute_recall": cand_comparison.get("attribute_recall_mean", 0.0),
#                     "candidate_relation_recall": cand_comparison.get("relation_recall_mean", 0.0),
#                     "acceptance_reason": reason,
#                     "acceptance_details": accept_details,
#                 })

#                 step = CompressionStep(
#                     iteration=iteration,
#                     candidate_index=cand.index,
#                     candidate_caption=cand.text,
#                     target_token_budget=cand.target_token_budget,
#                     caption_before=current_caption,
#                     tokens_before=current_tokens,
#                     tokens_after=cand.token_count,
#                     token_reduction=current_tokens - cand.token_count,
#                     distortion=cand_D,
#                     threshold=threshold,
#                     passed_threshold=passed,
#                     selected=False,
#                     reason=reason,
#                     distortion_details=details,
#                     missing_anchors=cand.missing_anchors,
#                     time_seconds=time.time() - step_start,
#                 )
#                 steps.append(step)

#                 status = "PASS" if passed else "FAIL"
#                 print(
#                     f"      [{status}] D={cand_D:.4f} | "
#                     f"KG-F1={cand_comparison.get('f1_mean', 0.0):.4f} | "
#                     f"obj_rec={cand_comparison.get('object_recall_mean', 0.0):.4f} | "
#                     f"reason={reason}"
#                 )

#                 # Track best below-floor candidate for fallback.
#                 if not passed:
#                     score = (cand_D - threshold)  # smaller (more negative) = better
#                     if best_below_floor is None or score < best_below_floor["score"]:
#                         best_below_floor = {
#                             "score": score,
#                             "caption": cand.text,
#                             "tokens": cand.token_count,
#                             "D": cand_D,
#                             "kgs": cand_kgs,
#                             "dist": cand_dist,
#                             "comparison": cand_comparison,
#                             "step": step,
#                         }
#                     continue

#                 # First passing candidate (shortest, since we sorted) wins this iteration.
#                 step.selected = True
#                 iteration_winner = {
#                     "caption": cand.text,
#                     "tokens": cand.token_count,
#                     "D": cand_D,
#                     "kgs": cand_kgs,
#                     "dist": cand_dist,
#                     "comparison": cand_comparison,
#                     "step": step,
#                 }
#                 print(f"      [SELECTED] candidate#{cand.index}")
#                 break

#             if iteration_winner is None:
#                 print(f"[Iter {iteration}] No candidate passed fidelity floors. Stopping.")
#                 break

#             if self.require_strict_compression and iteration_winner["tokens"] >= current_tokens:
#                 print(f"[Iter {iteration}] Winner not shorter than current. Stopping.")
#                 break

#             current_caption = iteration_winner["caption"]
#             current_kgs = iteration_winner["kgs"]
#             current_dist = iteration_winner["dist"]
#             print(
#                 f"[Iter {iteration}] Updated current caption | "
#                 f"tokens={iteration_winner['tokens']} | D={iteration_winner['D']:.4f}"
#             )

#         # --- 6. Resolve final result (with fallback) ---
#         compressed_tokens = count_tokens(current_caption)
#         no_compression = (compressed_tokens >= original_tokens)

#         if no_compression and self.fallback_strategy == "best_candidate" and best_below_floor is not None:
#             print(
#                 f"\n[FALLBACK] Using best below-floor candidate "
#                 f"(D={best_below_floor['D']:.4f}, threshold={threshold:.4f})"
#             )
#             current_caption = best_below_floor["caption"]
#             current_kgs = best_below_floor["kgs"]
#             current_dist = best_below_floor["dist"]
#             best_below_floor["step"].selected = True
#             compressed_tokens = best_below_floor["tokens"]
#             no_compression = False
#             status = "compressed_via_fallback"
#         elif no_compression:
#             status = "kept_original_no_passing_candidate"
#         else:
#             status = "compressed"

#         compression_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))
#         total_time = time.time() - start_time

#         final_comparison = self._aggregate_kg_comparisons(kg_original, current_kgs)
#         if self.save_intermediate:
#             self._save_final_kgs_alias(current_kgs, image_id_safe)
#             self._save_kg_comparison_debug(final_comparison, image_id_safe, "final_kg_comparison")

#         print_kg_comparison("FINAL KG COMPARISON", final_comparison)

#         final_details = {k: v for k, v in current_dist.items() if not str(k).endswith("_std")}
#         final_details["status"] = status

#         result = CompressionResult(
#             image_id=image_id,
#             status=status,
#             original_caption=caption,
#             compressed_caption=current_caption,
#             original_tokens=original_tokens,
#             compressed_tokens=compressed_tokens,
#             compression_ratio=compression_ratio,
#             baseline_distortion=baseline_D,
#             threshold=threshold,
#             final_distortion=float(current_dist["D"]),
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=final_comparison,
#             protected_anchors=list(kg_anchors),
#             quoted_anchors=list(quoted_anchors),
#             steps=steps,
#             total_time=total_time,
#         )

#         print("\n" + "-" * 70)
#         print(
#             f"Result for {image_id}: "
#             f"{original_tokens} -> {compressed_tokens} tokens | "
#             f"CR={compression_ratio:.2%} | "
#             f"Final D={result.final_distortion:.4f} | "
#             f"delta={threshold:.4f} | status={status}"
#         )
#         print("-" * 70)
#         print("\nORIGINAL CAPTION:")
#         print(caption)
#         print("\nFINAL COMPRESSED CAPTION:")
#         print(current_caption)

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     def compress_dataset(
#         self,
#         dataset: Any,
#         max_samples: Optional[int] = None,
#     ) -> List[CompressionResult]:
#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []

#         print("\n" + "=" * 70)
#         print(f"Processing {n} image-caption pairs")
#         print("=" * 70)

#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]
#             try:
#                 result = self.compress_single(
#                     image=sample["image"],
#                     caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 )
#                 results.append(result)
#             except Exception as exc:
#                 print(
#                     f"[ERROR] Failed on idx={idx}, "
#                     f"image_id={sample.get('image_id', 'unknown')}: {exc}"
#                 )
#                 continue

#         self._save_summary(results)
#         return results

#     # -----------------------------------------------------------------
#     # Reconstruction + KG extraction
#     # -----------------------------------------------------------------

#     def _reconstruct_and_extract_kgs(
#         self,
#         caption: str,
#         image_id: str,
#         prefix: str,
#     ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
#         output_dir = None
#         if self.save_intermediate:
#             output_dir = self.reconstruction_dir / safe_filename(image_id)
#             output_dir.mkdir(parents=True, exist_ok=True)

#         images = self.t2i.reconstruct(
#             caption=caption,
#             seeds=self.t2i_seeds,
#             output_dir=output_dir,
#             prefix=prefix,
#         )

#         kgs = []
#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract(img, image_id=f"{image_id}_{prefix}_seed{seed}")
#             kgs.append(kg)
#             if self.save_intermediate:
#                 self._save_kg_debug(kg, image_id, f"{prefix}_seed{seed}_reconstructed_kg")

#         return images, kgs

#     # -----------------------------------------------------------------
#     # KG comparison aggregation across seeds
#     # -----------------------------------------------------------------

#     def _aggregate_kg_comparisons(
#         self,
#         kg_original: Dict[str, Any],
#         reconstructed_kgs: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         comparisons = [compare_scene_graphs(kg_original, kg_recon) for kg_recon in reconstructed_kgs]

#         preserved: Set[Tuple[str, ...]] = set()
#         missing: Set[Tuple[str, ...]] = set()
#         extra: Set[Tuple[str, ...]] = set()

#         f1s, precisions, recalls = [], [], []
#         object_recalls, attribute_recalls, relation_recalls = [], [], []

#         for comp in comparisons:
#             preserved |= set(comp.get("preserved", set()))
#             missing |= set(comp.get("missing", set()))
#             extra |= set(comp.get("extra", set()))

#             f1s.append(comp.get("f1", 0.0))
#             precisions.append(comp.get("precision", 0.0))
#             recalls.append(comp.get("recall", 0.0))
#             object_recalls.append(comp.get("object_recall", 0.0))
#             attribute_recalls.append(comp.get("attribute_recall", 0.0))
#             relation_recalls.append(comp.get("relation_recall", 0.0))

#         return {
#             "preserved": preserved,
#             "missing": missing,
#             "extra": extra,
#             "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
#             "f1_std": float(np.std(f1s)) if f1s else 0.0,
#             "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
#             "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
#             "object_recall_mean": float(np.mean(object_recalls)) if object_recalls else 0.0,
#             "attribute_recall_mean": float(np.mean(attribute_recalls)) if attribute_recalls else 0.0,
#             "relation_recall_mean": float(np.mean(relation_recalls)) if relation_recalls else 0.0,
#             "num_preserved": len(preserved),
#             "num_missing": len(missing),
#             "num_extra": len(extra),
#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     # -----------------------------------------------------------------
#     # Candidate handling
#     # -----------------------------------------------------------------

#     def _compute_target_budgets(
#         self,
#         source_tokens: int,
#         original_tokens: int,
#     ) -> List[int]:
#         """
#         Build N target token budgets ranging from aggressive to conservative,
#         relative to the ORIGINAL caption (not the current one) so the loop
#         doesn't drift toward longer captions across iterations.
#         """
#         n = self.num_candidates_per_iteration
#         if n <= 0:
#             return []

#         lo = max(self.min_caption_tokens, int(round(self.min_target_token_ratio * original_tokens)))
#         hi = max(lo + 1, int(round(self.max_target_token_ratio * original_tokens)))

#         # Cap upper budget below the current caption to encourage compression.
#         hi = min(hi, max(lo + 1, source_tokens - 1))

#         if n == 1:
#             return [lo]

#         step = (hi - lo) / (n - 1)
#         budgets = [int(round(lo + i * step)) for i in range(n)]

#         # Deduplicate and ensure ascending order.
#         seen: Set[int] = set()
#         unique = []
#         for b in budgets:
#             if b not in seen:
#                 seen.add(b)
#                 unique.append(b)
#         return unique

#     def _filter_candidates(
#         self,
#         raw_candidates: Sequence[str],
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#         source_tokens: int,
#         target_budgets: Sequence[int],
#     ) -> List[RewriteCandidate]:
#         """
#         Apply cheap pre-filters before expensive FLUX validation:
#           - Drop empty / overly long candidates.
#           - Drop candidates that are not strictly shorter (when required).
#           - Drop candidates well below the smallest budget (likely fragments).
#           - Mark anchor-preservation status.
#         """
#         candidates: List[RewriteCandidate] = []
#         seen_text: Set[str] = set()

#         budgets = list(target_budgets) if target_budgets else [source_tokens]

#         # Hard lower bound: reject candidates more than `slack` tokens below
#         # the smallest target budget. Keeps the LLM honest when it ignores
#         # the budget and emits 6-token comma-list fragments.
#         budget_slack = 4
#         min_acceptable_tokens = max(
#             self.min_caption_tokens,
#             (min(budgets) if budgets else self.min_caption_tokens) - budget_slack,
#         )

#         for idx, raw in enumerate(raw_candidates):
#             text = clean_caption_spacing(raw)
#             if not text:
#                 continue

#             key = text.lower()
#             if key in seen_text:
#                 continue
#             seen_text.add(key)

#             tok = count_tokens(text)

#             # Match to the closest budget for reporting.
#             target_budget = min(budgets, key=lambda b: abs(b - tok)) if budgets else tok

#             cand = RewriteCandidate(
#                 index=idx,
#                 text=text,
#                 target_token_budget=int(target_budget),
#                 token_count=tok,
#             )

#             # Strict-compression check.
#             if self.require_strict_compression and tok >= source_tokens:
#                 cand.passed_anchor_filter = False
#                 cand.missing_anchors = ["__not_shorter_than_source__"]
#                 candidates.append(cand)
#                 continue

#             if tok < min_acceptable_tokens:
#                 cand.passed_anchor_filter = False
#                 cand.missing_anchors = [
#                     f"__below_min_acceptable_tokens__({tok}<{min_acceptable_tokens})"
#                 ]
#                 candidates.append(cand)
#                 continue

#             passed, missing = candidate_preserves_anchors(
#                 candidate=text,
#                 quoted_anchors=quoted_anchors,
#                 kg_anchors=kg_anchors,
#             )
#             cand.passed_anchor_filter = passed
#             cand.missing_anchors = missing
#             candidates.append(cand)

#         return candidates

#     def _candidate_acceptance_decision(
#         self,
#         candidate_D: float,
#         threshold: float,
#         candidate_comparison: Dict[str, Any],
#         baseline_comparison: Dict[str, Any],
#     ) -> Tuple[bool, str, Dict[str, Any]]:
#         """
#         Decide whether a candidate is acceptable, using floors derived
#         from the baseline (full-caption) reconstruction. Quoted/KG anchor
#         preservation is already enforced by the pre-filter, so this checks
#         only the reconstruction-side fidelity floors.
#         """
#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))
#         baseline_attr_recall = float(baseline_comparison.get("attribute_recall_mean", 0.0))

#         candidate_f1 = float(candidate_comparison.get("f1_mean", 0.0))
#         candidate_obj_recall = float(candidate_comparison.get("object_recall_mean", 0.0))
#         candidate_attr_recall = float(candidate_comparison.get("attribute_recall_mean", 0.0))

#         min_f1_allowed = max(self.min_baseline_kg_f1, baseline_f1 - self.kg_f1_floor_drop)
#         min_obj_recall_allowed = max(
#             self.min_baseline_object_recall,
#             baseline_obj_recall - self.object_recall_floor_drop,
#         )
#         min_attr_recall_allowed = max(
#             self.attribute_recall_min_floor,
#             baseline_attr_recall - self.attribute_recall_floor_drop,
#         )

#         details = {
#             "candidate_D": candidate_D,
#             "threshold": threshold,
#             "baseline_f1": baseline_f1,
#             "candidate_f1": candidate_f1,
#             "min_f1_allowed": min_f1_allowed,
#             "baseline_object_recall": baseline_obj_recall,
#             "candidate_object_recall": candidate_obj_recall,
#             "min_object_recall_allowed": min_obj_recall_allowed,
#             "baseline_attribute_recall": baseline_attr_recall,
#             "candidate_attribute_recall": candidate_attr_recall,
#             "min_attribute_recall_allowed": min_attr_recall_allowed,
#         }

#         if candidate_D > threshold:
#             return False, "rejected_above_global_threshold", details
#         if candidate_f1 < min_f1_allowed:
#             return False, "rejected_below_acceptable_kg_f1", details
#         if candidate_obj_recall < min_obj_recall_allowed:
#             return False, "rejected_below_acceptable_object_recall", details
#         if candidate_attr_recall < min_attr_recall_allowed:
#             return False, "rejected_below_acceptable_attribute_recall", details

#         return True, "accepted_under_acceptable_fidelity", details

#     # -----------------------------------------------------------------
#     # Skip path
#     # -----------------------------------------------------------------

#     def _make_skipped_result(
#         self,
#         image_id: str,
#         caption: str,
#         original_tokens: int,
#         baseline_D: float,
#         threshold: float,
#         baseline_dist: Dict[str, Any],
#         baseline_comparison: Dict[str, Any],
#         baseline_kgs: List[Dict[str, Any]],
#         image_id_safe: str,
#         start_time: float,
#     ) -> CompressionResult:
#         final_details = {k: v for k, v in baseline_dist.items() if not str(k).endswith("_std")}
#         final_details["status"] = "skipped_low_baseline_fidelity"

#         if self.save_intermediate:
#             self._save_final_kgs_alias(baseline_kgs, image_id_safe)

#         result = CompressionResult(
#             image_id=image_id,
#             status="skipped_low_baseline_fidelity",
#             original_caption=caption,
#             compressed_caption=caption,
#             original_tokens=original_tokens,
#             compressed_tokens=original_tokens,
#             compression_ratio=0.0,
#             baseline_distortion=baseline_D,
#             threshold=threshold,
#             final_distortion=baseline_D,
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=baseline_comparison,
#             protected_anchors=[],
#             quoted_anchors=[],
#             steps=[],
#             total_time=time.time() - start_time,
#         )

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     # -----------------------------------------------------------------
#     # Persistence
#     # -----------------------------------------------------------------

#     def _save_kg_debug(self, kg: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(kg_to_readable_dict(kg)), f, indent=2, ensure_ascii=False)
#         print(f"[KG Saved] {path}")

#     def _save_kg_comparison_debug(self, comparison: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(comparison), f, indent=2, ensure_ascii=False)
#         print(f"[KG Comparison Saved] {path}")

#     def _save_final_kgs_alias(self, kgs: List[Dict[str, Any]], image_id: str) -> None:
#         for idx, kg in enumerate(kgs):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             self._save_kg_debug(kg, image_id, f"final_reconstructed_kg_seed{seed}")

#     def _save_candidates_debug(
#         self,
#         image_id: str,
#         iteration: int,
#         raw_candidates: Sequence[str],
#         filtered_candidates: Sequence[RewriteCandidate],
#         quoted_anchors: Sequence[str],
#         kg_anchors: Sequence[str],
#     ) -> None:
#         debug_dir = self.candidate_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"iter{iteration}_candidates.json"

#         payload = {
#             "iteration": iteration,
#             "quoted_anchors": list(quoted_anchors),
#             "kg_anchors": list(kg_anchors),
#             "raw_candidates": list(raw_candidates),
#             "filtered_candidates": [asdict(c) for c in filtered_candidates],
#         }
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(payload), f, indent=2, ensure_ascii=False)
#         print(f"[Candidates Saved] {path}")

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(result), f, indent=2, ensure_ascii=False)
#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             print("[Summary] No results to summarize.")
#             return

#         compression_ratios = [r.compression_ratio for r in results]
#         final_distortions = [r.final_distortion for r in results]
#         baseline_distortions = [r.baseline_distortion for r in results]
#         original_tokens = [r.original_tokens for r in results]
#         compressed_tokens = [r.compressed_tokens for r in results]
#         times = [r.total_time for r in results]

#         statuses: Dict[str, int] = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1

#         summary = {
#             "num_samples": len(results),
#             "status_counts": statuses,
#             "avg_compression_ratio": float(np.mean(compression_ratios)),
#             "std_compression_ratio": float(np.std(compression_ratios)),
#             "avg_baseline_distortion": float(np.mean(baseline_distortions)),
#             "avg_final_distortion": float(np.mean(final_distortions)),
#             "avg_original_tokens": float(np.mean(original_tokens)),
#             "avg_compressed_tokens": float(np.mean(compressed_tokens)),
#             "avg_time_seconds": float(np.mean(times)),
#             "total_time_hours": float(np.sum(times) / 3600.0),
#             "use_relative_threshold": self.use_relative_threshold,
#             "epsilon": self.epsilon,
#             "absolute_threshold": self.absolute_threshold,
#             "num_seeds": len(self.t2i_seeds),
#             "seeds": self.t2i_seeds,
#             "min_baseline_kg_f1": self.min_baseline_kg_f1,
#             "min_baseline_object_recall": self.min_baseline_object_recall,
#             "num_candidates_per_iteration": self.num_candidates_per_iteration,
#             "max_iterations": self.max_iterations,
#             "min_target_token_ratio": self.min_target_token_ratio,
#             "max_target_token_ratio": self.max_target_token_ratio,
#             "fallback_strategy": self.fallback_strategy,
#         }

#         summary_path = self.result_dir / "summary.json"
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(summary), f, indent=2)

#         print("\n" + "=" * 70)
#         print(f"SUMMARY over {len(results)} samples")
#         print(f"Status counts:          {statuses}")
#         print(f"Avg compression ratio:  {summary['avg_compression_ratio']:.2%}")
#         print(f"Avg baseline D:         {summary['avg_baseline_distortion']:.4f}")
#         print(f"Avg final D:            {summary['avg_final_distortion']:.4f}")
#         print(
#             f"Avg tokens:             "
#             f"{summary['avg_original_tokens']:.1f} -> "
#             f"{summary['avg_compressed_tokens']:.1f}"
#         )
#         print(f"Total time:             {summary['total_time_hours']:.2f} hours")
#         print(f"Saved summary:          {summary_path}")
#         print("=" * 70)





# """
# KG-RD Compressor: Grammar-Aware Rewrite Compression
# ====================================================

# This version replaces greedy span deletion with rewrite-based compression.

# Goal:
#     Find the shortest grammatical caption that preserves important visual
#     anchors and passes reconstruction-fidelity validation.

# High-level pipeline:
#     1. Extract KG from the original image.
#     2. Reconstruct the image from the original caption using T2I.
#     3. Extract KG from the baseline reconstruction.
#     4. Compute baseline distortion and KG comparison.
#     5. Extract protected anchors from quoted text + original KG + caption.
#     6. Generate 3-5 grammatical compressed rewrite candidates.
#     7. Cheaply reject candidates that lose anchors or break grammar.
#     8. Validate remaining candidates through:
#            caption -> FLUX reconstruction -> Qwen2.5-VL KG -> distortion.
#     9. Select the shortest candidate satisfying the fidelity threshold.

# This avoids the previous failure mode where deletion-only compression produced
# broken text such as "On of is a and faded line".
# """

# import json
# import re
# import string
# import time
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     compare_scene_graphs,
#     normalize_label,
#     normalize_relation,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor


# # ---------------------------------------------------------------------
# # Data classes
# # ---------------------------------------------------------------------

# @dataclass
# class CompressionStep:
#     iteration: int
#     deleted_span: str
#     deleted_label: str
#     caption_before: str
#     caption_after: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     distortion_details: Dict[str, Any]
#     aligned_preserved: List[str] = field(default_factory=list)
#     aligned_missing: List[str] = field(default_factory=list)
#     time_seconds: float = 0.0


# @dataclass
# class CompressionResult:
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     steps: List[CompressionStep]
#     total_time: float


# # ---------------------------------------------------------------------
# # General helpers
# # ---------------------------------------------------------------------

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", str(text)).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"([([{])\s+", r"\1", text)
#     text = re.sub(r"\s+([)\]}])", r"\1", text)
#     text = text.replace(" ,", ",").replace(" .", ".")
#     text = text.replace(",.", ".").replace(".,", ".")
#     text = re.sub(r"\.\s*\.", ".", text)
#     return text.strip()


# def safe_filename(name: str) -> str:
#     name = str(name)
#     name = re.sub(r"[^\w\-.]+", "_", name)
#     return name[:180]


# def dataclass_is_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if dataclass_is_instance(obj):
#         return make_json_serializable(asdict(obj))

#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]

#     if isinstance(obj, tuple):
#         return list(obj)

#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]

#     if isinstance(obj, (np.integer,)):
#         return int(obj)

#     if isinstance(obj, (np.floating,)):
#         return float(obj)

#     if isinstance(obj, np.ndarray):
#         return obj.tolist()

#     return obj


# def content_words(text: str) -> Set[str]:
#     text = normalize_label(text)
#     words = set(text.split())

#     stopwords = {
#         "a", "an", "the", "of", "and", "or", "to", "in", "on", "at",
#         "by", "with", "for", "from", "as", "is", "are", "was", "were",
#         "be", "being", "been", "this", "that", "these", "those",
#         "image", "photo", "picture", "scene", "view", "showing", "shows",
#     }

#     return {w for w in words if len(w) > 2 and w not in stopwords}


# def normalize_for_anchor(text: str) -> str:
#     text = str(text).lower().strip()
#     text = text.replace("_", " ").replace("-", " ")
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def get_quoted_texts(text: str) -> List[str]:
#     values = []
#     for m in re.finditer(r'"([^"]+)"', text):
#         q = m.group(1).strip()
#         if q:
#             values.append(q)
#     return values


# def has_balanced_quotes(text: str) -> bool:
#     return text.count('"') % 2 == 0


# def is_caption_text_valid(text: str) -> bool:
#     """
#     Conservative readability check for rewritten captions.
#     This does not guarantee perfect grammar, but catches common broken outputs.
#     """
#     if not isinstance(text, str) or not text.strip():
#         return False

#     text = clean_caption_spacing(text)

#     if count_tokens(text) < 5:
#         return False

#     if not has_balanced_quotes(text):
#         return False

#     padded = f" {text} "

#     bad_literals = [
#         " ,", " .", ",.", ".,", "..", " ;", " :",
#         " In is ", " On is ", " At is ", " With is ",
#         " On of ", " In of ", " is with ", " is of ",
#         " a and ", " an and ", " the and ",
#         " is a and ", " is an and ", " is the and ",
#         " On of is ", " In of is ",
#         "partially and faded",
#     ]
#     for pat in bad_literals:
#         if pat in padded:
#             return False

#     bad_regex = [
#         r"\bA\s+[aeiouAEIOU]\w*",          # A old -> bad, should be An old
#         r"\ban\s+[^aeiouAEIOU\s]\w*",      # an red -> bad
#         r"(^|[.!?]\s+)(In|On|At|With|Of)\s+(is|are|was|were|with|of)\b",
#         r"\b(is|are|was|were)\s+(a|an|the)?\s*(and|with|of)\b",
#         r"\b(of|in|on|with)\s+(of|in|on|with|is|are|was|were)\b",
#         r"\b(the|a|an)\s+(of|in|on|with|is|are|was|were|and)\b",
#     ]
#     for pattern in bad_regex:
#         if re.search(pattern, text):
#             return False

#     return True


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     tuples = kg.get("tuples", set())
#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "tuples": [list(t) for t in sorted(list(tuples), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def print_kg(title: str, kg: Dict[str, Any], max_items: int = 40) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)

#     print("\nOBJECTS:")
#     for obj in kg.get("objects", [])[:max_items]:
#         label = obj.get("label", obj.get("name", ""))
#         attrs = obj.get("attributes", [])
#         print(f"  - {label} | attributes={attrs}")

#     print("\nATTRIBUTES:")
#     for attr in kg.get("attributes", [])[:max_items]:
#         obj = attr.get("object", "")
#         attribute = attr.get("attribute", "")
#         print(f"  - ({obj}, has_attribute, {attribute})")

#     print("\nRELATIONS:")
#     for rel in kg.get("relations", [])[:max_items]:
#         subj = rel.get("subject", "")
#         pred = rel.get("predicate", rel.get("relation", ""))
#         obj = rel.get("object", "")
#         print(f"  - ({subj}, {pred}, {obj})")

#     print("\nKG TUPLES:")
#     for t in list(kg.get("tuples", set()))[:max_items]:
#         print(f"  - {t}")

#     print("=" * 100)


# def print_kg_comparison(title: str, comparison: Dict[str, Any], max_items: int = 60) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)
#     print(
#         f"F1 mean={comparison.get('f1_mean', 0.0):.4f} | "
#         f"Recall mean={comparison.get('recall_mean', 0.0):.4f} | "
#         f"Object recall={comparison.get('object_recall_mean', 0.0):.4f} | "
#         f"Attribute recall={comparison.get('attribute_recall_mean', 0.0):.4f} | "
#         f"Relation recall={comparison.get('relation_recall_mean', 0.0):.4f}"
#     )

#     print("\nPRESERVED ELEMENTS:")
#     for x in comparison.get("preserved_text", [])[:max_items]:
#         print("  +", x)

#     print("\nMISSING ELEMENTS:")
#     for x in comparison.get("missing_text", [])[:max_items]:
#         print("  -", x)

#     print("\nEXTRA / HALLUCINATED ELEMENTS:")
#     for x in comparison.get("extra_text", [])[:max_items]:
#         print("  *", x)

#     print("=" * 100)


# def save_debug_sample_visuals(
#     image_original: Image.Image,
#     image_reconstructed: Image.Image,
#     caption: str,
#     image_id: str,
#     output_dir: Path,
#     name_prefix: str = "baseline",
# ) -> None:
#     debug_dir = output_dir / "debug_samples" / safe_filename(image_id)
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     original_path = debug_dir / "original_image.png"
#     reconstructed_path = debug_dir / f"{safe_filename(name_prefix)}_reconstructed.png"
#     caption_path = debug_dir / f"{safe_filename(name_prefix)}_caption.txt"
#     side_by_side_path = debug_dir / f"{safe_filename(name_prefix)}_side_by_side.png"

#     image_original_rgb = image_original.convert("RGB")
#     image_reconstructed_rgb = image_reconstructed.convert("RGB")

#     image_original_rgb.save(original_path)
#     image_reconstructed_rgb.save(reconstructed_path)

#     with open(caption_path, "w", encoding="utf-8") as f:
#         f.write(caption)

#     left = image_original_rgb.resize((512, 512))
#     right = image_reconstructed_rgb.resize((512, 512))

#     canvas = Image.new("RGB", (1024, 560), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))

#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), "Original Image", fill="black")
#     draw.text((650, 15), f"Reconstructed: {name_prefix}", fill="black")
#     canvas.save(side_by_side_path)

#     print(f"[Debug Image Saved] {original_path}")
#     print(f"[Debug Image Saved] {reconstructed_path}")
#     print(f"[Debug Caption Saved] {caption_path}")
#     print(f"[Debug Side-by-Side Saved] {side_by_side_path}")


# # ---------------------------------------------------------------------
# # Main compressor
# # ---------------------------------------------------------------------

# class KGRDCompressor:
#     """
#     Knowledge-Graph-guided Rate-Distortion Caption Compressor.

#     This implementation uses rewrite candidates instead of greedy deletion.
#     """

#     def __init__(
#         self,
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,

#         t2i_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "float16",
#         t2i_steps: int = 30,
#         t2i_guidance: float = 7.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,

#         alpha: float = 0.4,
#         beta: float = 0.4,
#         gamma: float = 0.2,
#         clip_model: str = "ViT-L-14",
#         clip_pretrained: str = "openai",
#         lpips_net: str = "alex",
#         metric_device: str = "auto",

#         use_relative_threshold: bool = True,
#         epsilon: float = 0.10,
#         absolute_threshold: float = 0.25,

#         min_caption_tokens: int = 5,
#         max_iterations: int = 50,
#         max_candidates_per_iteration: int = 10,
#         allow_unaligned_candidates: bool = False,

#         target_compression_ratio: float = 0.35,
#         max_compression_ratio: float = 0.60,
#         stop_when_target_reached: bool = True,

#         min_baseline_kg_f1: float = 0.05,
#         min_baseline_object_recall: float = 0.10,

#         # Rewrite-specific controls.
#         rewrite_mode: str = "qwen_text_only",
#         max_rewrite_candidates: int = 5,
#         validate_top_k: int = 5,
#         save_rejected_candidates: bool = False,

#         save_intermediate: bool = True,
#         output_dir: str = "./outputs",
#     ):
#         self.t2i_seeds = t2i_seeds or [42]

#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)

#         self.min_caption_tokens = int(min_caption_tokens)
#         self.max_iterations = int(max_iterations)  # kept for config compatibility
#         self.max_candidates_per_iteration = int(max_candidates_per_iteration)  # compatibility
#         self.allow_unaligned_candidates = bool(allow_unaligned_candidates)  # compatibility

#         self.target_compression_ratio = float(target_compression_ratio)
#         self.max_compression_ratio = float(max_compression_ratio)
#         self.stop_when_target_reached = bool(stop_when_target_reached)

#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)

#         self.rewrite_mode = rewrite_mode
#         self.max_rewrite_candidates = int(max_rewrite_candidates)
#         self.validate_top_k = int(validate_top_k)
#         self.save_rejected_candidates = bool(save_rejected_candidates)

#         self.save_intermediate = bool(save_intermediate)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"
#         self.result_dir.mkdir(parents=True, exist_ok=True)
#         self.kg_debug_dir.mkdir(parents=True, exist_ok=True)

#         print("=" * 70)
#         print("KG-RD Compressor: Initializing rewrite-based pipeline")
#         print("=" * 70)

#         print("\n[1/3] Loading VLM scene graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model,
#             device=scene_graph_device,
#             dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens,
#             temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample,
#             max_objects=max_objects,
#             max_relations=max_relations,
#         )

#         print("\n[2/3] Loading text-to-image reconstructor...")
#         self.t2i = T2IReconstructor(
#             model_name=t2i_model,
#             device=t2i_device,
#             dtype=t2i_dtype,
#             num_inference_steps=t2i_steps,
#             guidance_scale=t2i_guidance,
#             image_size=t2i_image_size,
#         )

#         print("\n[3/3] Loading distortion metric...")
#         self.distortion = DistortionMetric(
#             alpha=alpha,
#             beta=beta,
#             gamma=gamma,
#             clip_model_name=clip_model,
#             clip_pretrained=clip_pretrained,
#             lpips_net=lpips_net,
#             device=metric_device,
#         )

#         print("\n" + "=" * 70)
#         if self.use_relative_threshold:
#             print(f"Pipeline ready | relative threshold: delta = D0 * (1 + {self.epsilon})")
#         else:
#             print(f"Pipeline ready | absolute threshold: delta = {self.absolute_threshold}")
#         print(
#             f"Rewrite mode={self.rewrite_mode} | max candidates={self.max_rewrite_candidates} | "
#             f"validate_top_k={self.validate_top_k}"
#         )
#         print(
#             f"Compression target={self.target_compression_ratio:.0%} | "
#             f"max={self.max_compression_ratio:.0%}"
#         )
#         print("=" * 70 + "\n")

#     # -----------------------------------------------------------------
#     # Public API
#     # -----------------------------------------------------------------

#     def compress_single(
#         self,
#         image: Image.Image,
#         caption: str,
#         image_id: str = "unknown",
#     ) -> CompressionResult:
#         start_time = time.time()

#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)
#         image_id_safe = safe_filename(image_id)

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | original_tokens={original_tokens}")
#         print("-" * 70)

#         # -------------------------------------------------------------
#         # 1. Original KG
#         # -------------------------------------------------------------
#         print("[1] Extracting KG from original image...")
#         kg_original = self.sg_extractor.extract(image, image_id=image_id)
#         print(
#             f"    Original KG: "
#             f"{len(kg_original.get('objects', []))} objects, "
#             f"{len(kg_original.get('attributes', []))} attributes, "
#             f"{len(kg_original.get('relations', []))} relations"
#         )
#         if self.save_intermediate:
#             self._save_kg_debug(kg_original, image_id_safe, "original_image_kg")
#         print_kg("ORIGINAL IMAGE KG", kg_original, max_items=40)

#         # -------------------------------------------------------------
#         # 2. Baseline reconstruction
#         # -------------------------------------------------------------
#         print("[2] Generating baseline reconstruction from full caption...")
#         baseline_images, baseline_kgs = self._reconstruct_and_extract_kgs(
#             caption=caption,
#             image_id=image_id_safe,
#             prefix="baseline_full_caption",
#             save_outputs=True,
#         )

#         if self.save_intermediate and baseline_images:
#             save_debug_sample_visuals(
#                 image_original=image,
#                 image_reconstructed=baseline_images[0],
#                 caption=caption,
#                 image_id=image_id_safe,
#                 output_dir=self.output_dir,
#                 name_prefix="baseline_full_caption",
#             )

#         if baseline_kgs:
#             print_kg("BASELINE RECONSTRUCTED IMAGE KG", baseline_kgs[0], max_items=40)

#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image,
#             images_recon=baseline_images,
#             kg_original=kg_original,
#             kgs_reconstructed=baseline_kgs,
#         )
#         baseline_D = float(baseline_dist["D"])

#         if self.use_relative_threshold:
#             threshold = self.distortion.compute_relative_threshold(
#                 baseline_distortion=baseline_D,
#                 epsilon=self.epsilon,
#             )
#         else:
#             threshold = self.absolute_threshold

#         baseline_comparison = self._aggregate_kg_comparisons(
#             kg_original=kg_original,
#             reconstructed_kgs=baseline_kgs,
#         )

#         if self.save_intermediate:
#             self._save_kg_comparison_debug(
#                 comparison=baseline_comparison,
#                 image_id=image_id_safe,
#                 name="baseline_kg_comparison",
#             )

#         print(f"    Baseline distortion D0 = {baseline_D:.4f}")
#         print(f"    Compression threshold delta = {threshold:.4f}")
#         print_kg_comparison("BASELINE KG COMPARISON", baseline_comparison)

#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))

#         if baseline_f1 < self.min_baseline_kg_f1 or baseline_obj_recall < self.min_baseline_object_recall:
#             print("\n[SKIP] Baseline reconstruction fidelity is too low.")
#             result = self._make_result(
#                 image_id=image_id,
#                 status="skipped_low_baseline_fidelity",
#                 original_caption=caption,
#                 compressed_caption=caption,
#                 baseline_D=baseline_D,
#                 threshold=threshold,
#                 final_dist=baseline_dist,
#                 baseline_comparison=baseline_comparison,
#                 final_comparison=baseline_comparison,
#                 steps=[],
#                 start_time=start_time,
#             )
#             if self.save_intermediate:
#                 self._save_result(result)
#             return result

#         # -------------------------------------------------------------
#         # 3. Protected anchors
#         # -------------------------------------------------------------
#         print("[3] Extracting protected anchors...")
#         anchors = self._extract_protected_anchors(caption, kg_original)
#         self._print_anchors(anchors)

#         # -------------------------------------------------------------
#         # 4. Generate rewrite candidates
#         # -------------------------------------------------------------
#         print("[4] Generating grammar-aware rewrite candidates...")
#         candidates = self._generate_rewrite_candidates(
#             original_caption=caption,
#             anchors=anchors,
#             original_tokens=original_tokens,
#         )

#         candidates = self._deduplicate_candidates(candidates, original_caption=caption)
#         candidates = self._cheap_filter_candidates(
#             candidates=candidates,
#             anchors=anchors,
#             original_caption=caption,
#             original_tokens=original_tokens,
#         )

#         # Prefer shorter candidates, but avoid validating too many.
#         candidates.sort(key=lambda c: (count_tokens(c), c))
#         candidates = candidates[: self.validate_top_k]

#         print(f"    Candidates after cheap filtering: {len(candidates)}")
#         for i, c in enumerate(candidates, start=1):
#             print(f"    Candidate {i}: {count_tokens(c)} tokens | {c}")

#         # -------------------------------------------------------------
#         # 5. Validate candidates through FLUX -> KG -> distortion
#         # -------------------------------------------------------------
#         print("[5] Validating rewrite candidates with reconstruction loop...")
#         steps: List[CompressionStep] = []
#         accepted: List[Dict[str, Any]] = []

#         for idx, candidate_caption in enumerate(candidates, start=1):
#             step_start = time.time()
#             candidate_tokens = count_tokens(candidate_caption)
#             token_reduction = original_tokens - candidate_tokens
#             cr = 1.0 - (candidate_tokens / max(original_tokens, 1))

#             print(
#                 f"    [VALIDATE {idx}] tokens {original_tokens}->{candidate_tokens} "
#                 f"| CR={cr:.2%}"
#             )

#             candidate_images, candidate_kgs = self._reconstruct_and_extract_kgs(
#                 caption=candidate_caption,
#                 image_id=image_id_safe,
#                 prefix=f"rewrite_candidate_{idx}",
#                 save_outputs=self.save_rejected_candidates,
#             )

#             candidate_dist = self.distortion.compute_averaged_distortion(
#                 image_orig=image,
#                 images_recon=candidate_images,
#                 kg_original=kg_original,
#                 kgs_reconstructed=candidate_kgs,
#             )
#             candidate_D = float(candidate_dist["D"])

#             candidate_comparison = self._aggregate_kg_comparisons(
#                 kg_original=kg_original,
#                 reconstructed_kgs=candidate_kgs,
#             )

#             passed, reason, details = self._rewrite_acceptance_decision(
#                 candidate_caption=candidate_caption,
#                 candidate_D=candidate_D,
#                 threshold=threshold,
#                 baseline_comparison=baseline_comparison,
#                 candidate_comparison=candidate_comparison,
#                 anchors=anchors,
#             )

#             distortion_details = {
#                 k: v for k, v in candidate_dist.items()
#                 if not str(k).endswith("_std")
#             }
#             distortion_details.update(
#                 {
#                     "candidate_kg_f1": candidate_comparison.get("f1_mean", 0.0),
#                     "candidate_object_recall": candidate_comparison.get("object_recall_mean", 0.0),
#                     "candidate_attribute_recall": candidate_comparison.get("attribute_recall_mean", 0.0),
#                     "candidate_relation_recall": candidate_comparison.get("relation_recall_mean", 0.0),
#                     "acceptance_reason": reason,
#                     "acceptance_details": details,
#                 }
#             )

#             step = CompressionStep(
#                 iteration=1,
#                 deleted_span=f"rewrite_candidate_{idx}",
#                 deleted_label="REWRITE",
#                 caption_before=caption,
#                 caption_after=candidate_caption,
#                 tokens_before=original_tokens,
#                 tokens_after=candidate_tokens,
#                 token_reduction=token_reduction,
#                 distortion=candidate_D,
#                 threshold=threshold,
#                 passed_threshold=passed,
#                 selected=False,
#                 reason=reason,
#                 distortion_details=distortion_details,
#                 aligned_preserved=[],
#                 aligned_missing=[],
#                 time_seconds=time.time() - step_start,
#             )
#             steps.append(step)

#             status = "PASS" if passed else "FAIL"
#             print(
#                 f"        [{status}] D={candidate_D:.4f} | "
#                 f"KG-F1={candidate_comparison.get('f1_mean', 0.0):.4f} | "
#                 f"reason={reason}"
#             )

#             if passed:
#                 accepted.append(
#                     {
#                         "idx": idx,
#                         "caption": candidate_caption,
#                         "tokens": candidate_tokens,
#                         "compression_ratio": cr,
#                         "distortion": candidate_dist,
#                         "D": candidate_D,
#                         "comparison": candidate_comparison,
#                         "images": candidate_images,
#                         "kgs": candidate_kgs,
#                         "step": step,
#                     }
#                 )

#         # -------------------------------------------------------------
#         # 6. Select shortest accepted candidate
#         # -------------------------------------------------------------
#         if accepted:
#             accepted.sort(key=lambda x: (x["tokens"], x["D"]))
#             best = accepted[0]
#             best["step"].selected = True

#             final_caption = best["caption"]
#             final_dist = best["distortion"]
#             final_comparison = best["comparison"]
#             final_images = best["images"]
#             final_kgs = best["kgs"]
#             status = "compressed"

#             print(
#                 f"[SELECTED] Candidate {best['idx']} | "
#                 f"tokens={best['tokens']} | CR={best['compression_ratio']:.2%} | "
#                 f"D={best['D']:.4f}"
#             )

#             if self.save_intermediate:
#                 self._save_selected_candidate_outputs(
#                     images=final_images,
#                     kgs=final_kgs,
#                     caption=final_caption,
#                     image=image,
#                     image_id=image_id_safe,
#                     prefix="final_rewrite",
#                 )
#                 self._save_kg_comparison_debug(
#                     comparison=final_comparison,
#                     image_id=image_id_safe,
#                     name="final_kg_comparison",
#                 )
#         else:
#             print("[NO ACCEPTED REWRITE] Falling back to original caption.")
#             final_caption = caption
#             final_dist = baseline_dist
#             final_comparison = baseline_comparison
#             status = "not_compressed"

#             if self.save_intermediate:
#                 self._save_final_kgs_alias(baseline_kgs, image_id_safe)
#                 self._save_kg_comparison_debug(
#                     comparison=final_comparison,
#                     image_id=image_id_safe,
#                     name="final_kg_comparison",
#                 )

#         result = self._make_result(
#             image_id=image_id,
#             status=status,
#             original_caption=caption,
#             compressed_caption=final_caption,
#             baseline_D=baseline_D,
#             threshold=threshold,
#             final_dist=final_dist,
#             baseline_comparison=baseline_comparison,
#             final_comparison=final_comparison,
#             steps=steps,
#             start_time=start_time,
#         )

#         print_kg_comparison("FINAL KG COMPARISON", final_comparison)
#         print("\n" + "-" * 70)
#         print(
#             f"Result for {image_id}: "
#             f"{result.original_tokens} -> {result.compressed_tokens} tokens | "
#             f"CR={result.compression_ratio:.2%} | "
#             f"Final D={result.final_distortion:.4f} | "
#             f"delta={threshold:.4f} | status={result.status}"
#         )
#         print("-" * 70)
#         print("\nORIGINAL CAPTION:")
#         print(caption)
#         print("\nFINAL COMPRESSED CAPTION:")
#         print(final_caption)

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     def compress_dataset(
#         self,
#         dataset: Any,
#         max_samples: Optional[int] = None,
#     ) -> List[CompressionResult]:
#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []

#         print("\n" + "=" * 70)
#         print(f"Processing {n} image-caption pairs")
#         print("=" * 70)

#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]
#             try:
#                 result = self.compress_single(
#                     image=sample["image"],
#                     caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 )
#                 results.append(result)
#             except Exception as exc:
#                 print(
#                     f"[ERROR] Failed on sample index={idx}, "
#                     f"image_id={sample.get('image_id', 'unknown')}: {exc}"
#                 )
#                 continue

#         self._save_summary(results)
#         return results

#     # -----------------------------------------------------------------
#     # Rewrite candidate generation
#     # -----------------------------------------------------------------

#     def _extract_protected_anchors(self, caption: str, kg: Dict[str, Any]) -> Dict[str, List[str]]:
#         quoted = []
#         for q in get_quoted_texts(caption):
#             q_clean = q.strip()
#             if q_clean:
#                 quoted.append(q_clean)

#         objects: List[str] = []
#         attributes: List[str] = []
#         visual_phrases: List[str] = []

#         for obj in kg.get("objects", []):
#             label = obj.get("label", obj.get("name", ""))
#             label = normalize_label(label)
#             if label:
#                 objects.append(label)
#             for attr in obj.get("attributes", []) or []:
#                 attr_norm = normalize_label(attr)
#                 if attr_norm:
#                     attributes.append(attr_norm)

#         for attr in kg.get("attributes", []):
#             obj = normalize_label(attr.get("object", ""))
#             value = normalize_label(attr.get("attribute", ""))
#             if obj:
#                 objects.append(obj)
#             if value:
#                 attributes.append(value)

#         # Caption-derived anchors. These are intentionally generic.
#         caption_lower = caption.lower()
#         phrase_patterns = [
#             r"street corner",
#             r"spray painted word",
#             r"painted word",
#             r"horizontal letter",
#             r"red line",
#             r"fire lane",
#             r"curb",
#             r"sidewalk",
#             r"pavement",
#             r"road",
#             r"marking",
#             r"paint marking",
#             r"white paint",
#         ]
#         for pat in phrase_patterns:
#             if re.search(pat, caption_lower):
#                 visual_phrases.append(pat.replace(r"\s+", " "))

#         color_words = []
#         for color in ["orange", "red", "white", "gray", "grey", "black", "yellow", "blue", "green", "brown"]:
#             if re.search(rf"\b{color}\b", caption_lower):
#                 color_words.append(color)

#         # Deduplicate while preserving order.
#         return {
#             "quoted_text": self._unique_strings(quoted),
#             "objects": self._unique_strings(objects),
#             "attributes": self._unique_strings(attributes),
#             "visual_phrases": self._unique_strings(visual_phrases),
#             "colors": self._unique_strings(color_words),
#         }

#     def _print_anchors(self, anchors: Dict[str, List[str]]) -> None:
#         for key, values in anchors.items():
#             print(f"    {key}: {values[:12]}")

#     def _generate_rewrite_candidates(
#         self,
#         original_caption: str,
#         anchors: Dict[str, List[str]],
#         original_tokens: int,
#     ) -> List[str]:
#         candidates: List[str] = []

#         # 1. LLM-generated candidates.
#         if self.rewrite_mode in {"qwen_text_only", "qwen", "auto"}:
#             try:
#                 candidates.extend(
#                     self._generate_qwen_text_rewrites(
#                         original_caption=original_caption,
#                         anchors=anchors,
#                         original_tokens=original_tokens,
#                     )
#                 )
#             except Exception as exc:
#                 print(f"    [Rewrite Warning] Qwen text rewrite failed: {exc}")

#         # 2. Deterministic fallback candidates.
#         candidates.extend(
#             self._generate_rule_based_rewrites(
#                 original_caption=original_caption,
#                 anchors=anchors,
#                 original_tokens=original_tokens,
#             )
#         )

#         return candidates

#     def _generate_qwen_text_rewrites(
#         self,
#         original_caption: str,
#         anchors: Dict[str, List[str]],
#         original_tokens: int,
#     ) -> List[str]:
#         """
#         Reuse the already-loaded Qwen2.5-VL model as a text-only rewrite model.
#         """
#         quoted = anchors.get("quoted_text", [])
#         visual = anchors.get("visual_phrases", [])
#         colors = anchors.get("colors", [])
#         objects = anchors.get("objects", [])[:10]
#         attributes = anchors.get("attributes", [])[:15]

#         target_20 = max(self.min_caption_tokens, int(original_tokens * 0.80))
#         target_35 = max(self.min_caption_tokens, int(original_tokens * 0.65))
#         target_50 = max(self.min_caption_tokens, int(original_tokens * 0.50))

#         prompt = f"""
# You are compressing a dense image caption for text-to-image reconstruction.

# Task:
# Generate {self.max_rewrite_candidates} grammatical compressed captions.
# Each candidate must preserve the important visual anchors and quoted text.
# Return ONLY a valid JSON list of strings. No markdown. No explanation.

# Original caption:
# {original_caption}

# Important quoted text that must remain exactly visible:
# {quoted}

# Important visual phrases:
# {visual}

# Important colors:
# {colors}

# Important KG objects:
# {objects}

# Important KG attributes:
# {attributes}

# Requirements:
# 1. Every candidate must be grammatical and readable.
# 2. Keep quoted text such as "ROW", "FIRE LANE", and "i" if present.
# 3. Preserve objects, colors, markings, visible text, and spatial layout.
# 4. Do not write broken fragments.
# 5. Produce different lengths:
#    - one conservative caption around {target_20} tokens
#    - one moderate caption around {target_35} tokens
#    - one aggressive caption around {target_50} tokens
#    - one shortest possible caption that still preserves anchors
# 6. Do not add new objects not in the caption.
# """.strip()

#         processor = self.sg_extractor.processor
#         model = self.sg_extractor.model

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]

#         text = processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         inputs = processor(
#             text=[text],
#             padding=True,
#             return_tensors="pt",
#         )

#         # Use the scene-graph device when available. If the model was loaded
#         # with device_map='auto', cuda:0 is still usually correct for inputs.
#         device = getattr(self.sg_extractor, "device", "cuda:0")
#         try:
#             inputs = inputs.to(device)
#         except Exception:
#             pass

#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             do_sample=True,
#             temperature=0.35,
#             top_p=0.9,
#         )

#         generated_ids_trimmed = []
#         for input_ids, output_ids in zip(inputs.input_ids, generated_ids):
#             generated_ids_trimmed.append(output_ids[len(input_ids):])

#         output = processor.batch_decode(
#             generated_ids_trimmed,
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=False,
#         )[0].strip()

#         return self._parse_candidate_list(output)

#     def _parse_candidate_list(self, text: str) -> List[str]:
#         text = text.strip()
#         text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
#         text = text.replace("```", "").strip()

#         # Try JSON list directly.
#         try:
#             data = json.loads(text)
#             if isinstance(data, list):
#                 return [clean_caption_spacing(str(x)) for x in data if str(x).strip()]
#             if isinstance(data, dict):
#                 values = []
#                 for v in data.values():
#                     if isinstance(v, str):
#                         values.append(v)
#                     elif isinstance(v, list):
#                         values.extend(str(x) for x in v)
#                 return [clean_caption_spacing(x) for x in values if x.strip()]
#         except Exception:
#             pass

#         # Extract a JSON list from surrounding text.
#         m = re.search(r"\[.*\]", text, flags=re.DOTALL)
#         if m:
#             try:
#                 data = json.loads(m.group(0))
#                 if isinstance(data, list):
#                     return [clean_caption_spacing(str(x)) for x in data if str(x).strip()]
#             except Exception:
#                 pass

#         # Fallback: parse numbered/bulleted lines.
#         candidates = []
#         for line in text.splitlines():
#             line = line.strip()
#             line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
#             line = line.strip('"')
#             if len(line.split()) >= 5:
#                 candidates.append(clean_caption_spacing(line))
#         return candidates

#     def _generate_rule_based_rewrites(
#         self,
#         original_caption: str,
#         anchors: Dict[str, List[str]],
#         original_tokens: int,
#     ) -> List[str]:
#         """
#         Safe deterministic fallback rewrites.
#         These are generic, but also work well for the current DOCCI example.
#         """
#         c = clean_caption_spacing(original_caption)
#         candidates = []

#         # Conservative cleanup.
#         conservative = c
#         replacements = [
#             (r"\bA high angle view of\b", ""),
#             (r"\bIn the middle of the view is\b", ""),
#             (r"\bOn the right side of the image is\b", ""),
#             (r"\bpartially visible and\b", ""),
#             (r"\bheavily faded in white paint\b", "faded"),
#             (r"\bon the street corner\b", ""),
#         ]
#         for pat, repl in replacements:
#             conservative = re.sub(pat, repl, conservative, flags=re.IGNORECASE)
#         conservative = clean_caption_spacing(conservative)
#         candidates.append(conservative)

#         # Anchor-composed fallback.
#         quoted = [q.strip() for q in anchors.get("quoted_text", [])]
#         q_norm = {normalize_for_anchor(q): q for q in quoted}
#         has_row = any("row" in k for k in q_norm)
#         has_fire_lane = any("fire lane" in k for k in q_norm)
#         has_i = any(k == "i" for k in q_norm)

#         lower = original_caption.lower()
#         scene = "faded street corner" if "street corner" in lower else "scene"
#         old_prefix = "Old " if "old" in lower else ""

#         parts = [f"{old_prefix}{scene}".strip()]

#         if has_row:
#             row_text = next(v for k, v in q_norm.items() if "row" in k)
#             if "orange" in lower:
#                 parts.append(f'with orange spray-painted "{row_text.strip()}"')
#             else:
#                 parts.append(f'with spray-painted "{row_text.strip()}"')

#         if has_i:
#             i_text = q_norm.get("i", "i")
#             if "horizontal" in lower:
#                 parts.append(f'and a horizontal "{i_text.strip()}"')
#             else:
#                 parts.append(f'and "{i_text.strip()}"')

#         if has_fire_lane:
#             fl_text = next(v for k, v in q_norm.items() if "fire lane" in k)
#             fire_phrase = f'faded red "{fl_text.strip()}" marking'
#             if "curb" in lower:
#                 fire_phrase += " on the curb"
#             elif "street" in lower or "road" in lower:
#                 fire_phrase += " on the street"
#             parts.append(fire_phrase)

#         if len(parts) > 1:
#             composed = parts[0] + " " + " ".join(parts[1:]) + "."
#             candidates.append(clean_caption_spacing(composed))

#         # Slightly more detailed composed caption.
#         if has_row or has_fire_lane:
#             detail = []
#             detail.append(f"{old_prefix}{scene}".strip())
#             if has_row:
#                 row_text = next(v for k, v in q_norm.items() if "row" in k)
#                 row_phrase = f'orange spray-painted "{row_text.strip()}"'
#                 if has_i:
#                     i_text = q_norm.get("i", "i")
#                     row_phrase += f' with a horizontal "{i_text.strip()}"'
#                 detail.append(row_phrase)
#             if has_fire_lane:
#                 fl_text = next(v for k, v in q_norm.items() if "fire lane" in k)
#                 detail.append(f'faded red "{fl_text.strip()}" line/marking')
#             candidates.append(clean_caption_spacing("; ".join(detail) + "."))

#         return candidates

#     def _deduplicate_candidates(self, candidates: Sequence[str], original_caption: str) -> List[str]:
#         seen: Set[str] = set()
#         output: List[str] = []
#         original_norm = normalize_for_anchor(original_caption)

#         for cand in candidates:
#             cand = clean_caption_spacing(cand)
#             if not cand:
#                 continue
#             if normalize_for_anchor(cand) == original_norm:
#                 continue
#             key = normalize_for_anchor(cand)
#             if key in seen:
#                 continue
#             seen.add(key)
#             output.append(cand)
#         return output

#     def _cheap_filter_candidates(
#         self,
#         candidates: Sequence[str],
#         anchors: Dict[str, List[str]],
#         original_caption: str,
#         original_tokens: int,
#     ) -> List[str]:
#         filtered: List[str] = []
#         for cand in candidates:
#             cand = clean_caption_spacing(cand)
#             tokens = count_tokens(cand)
#             if tokens >= original_tokens:
#                 continue
#             if tokens < self.min_caption_tokens:
#                 continue
#             cr = 1.0 - (tokens / max(original_tokens, 1))
#             if cr > self.max_compression_ratio:
#                 continue
#             if not is_caption_text_valid(cand):
#                 continue
#             ok, _missing = self._candidate_preserves_required_anchors(cand, anchors)
#             if not ok:
#                 continue
#             filtered.append(cand)
#         return filtered

#     def _candidate_preserves_required_anchors(
#         self,
#         candidate_caption: str,
#         anchors: Dict[str, List[str]],
#     ) -> Tuple[bool, List[str]]:
#         cand_norm = normalize_for_anchor(candidate_caption)
#         missing: List[str] = []

#         # Quoted text is mandatory. The candidate may include it with or
#         # without quotes, but the text itself must appear.
#         for q in anchors.get("quoted_text", []):
#             q_norm = normalize_for_anchor(q)
#             if q_norm and q_norm not in cand_norm:
#                 missing.append(f"quoted:{q}")

#         # Very important visual phrases from the caption/KG. We do not require
#         # all KG attributes because some can be too fine-grained, but require
#         # visible text and major markings if present.
#         hard_keywords = []
#         for phrase in anchors.get("visual_phrases", []):
#             if phrase in {"fire lane", "red line", "curb", "street corner", "marking", "paint marking"}:
#                 hard_keywords.append(phrase)

#         # If the original had colors, require the main visual colors when tied
#         # to text/markings. This keeps orange/red for the current example.
#         for color in anchors.get("colors", []):
#             if color in {"orange", "red"}:
#                 hard_keywords.append(color)

#         for kw in self._unique_strings(hard_keywords):
#             kw_norm = normalize_for_anchor(kw)
#             if kw_norm and kw_norm not in cand_norm:
#                 # Allow red line and marking to be represented as red + fire lane.
#                 if kw_norm == "red line" and "red" in cand_norm:
#                     continue
#                 if kw_norm in {"marking", "paint marking"} and ("marking" in cand_norm or "line" in cand_norm):
#                     continue
#                 missing.append(f"anchor:{kw}")

#         return len(missing) == 0, missing

#     # -----------------------------------------------------------------
#     # Candidate validation
#     # -----------------------------------------------------------------

#     def _rewrite_acceptance_decision(
#         self,
#         candidate_caption: str,
#         candidate_D: float,
#         threshold: float,
#         baseline_comparison: Dict[str, Any],
#         candidate_comparison: Dict[str, Any],
#         anchors: Dict[str, List[str]],
#     ) -> Tuple[bool, str, Dict[str, Any]]:
#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj = float(baseline_comparison.get("object_recall_mean", 0.0))
#         baseline_attr = float(baseline_comparison.get("attribute_recall_mean", 0.0))

#         cand_f1 = float(candidate_comparison.get("f1_mean", 0.0))
#         cand_obj = float(candidate_comparison.get("object_recall_mean", 0.0))
#         cand_attr = float(candidate_comparison.get("attribute_recall_mean", 0.0))

#         # Floors are relative to the full-caption baseline. This lets the
#         # compressor find short captions without requiring every candidate to be
#         # better than the previous one.
#         min_f1 = max(self.min_baseline_kg_f1, baseline_f1 - 0.10)
#         min_obj = max(self.min_baseline_object_recall, baseline_obj - 0.10)
#         min_attr = max(0.10, baseline_attr - 0.15)

#         anchors_ok, missing_anchors = self._candidate_preserves_required_anchors(candidate_caption, anchors)

#         details = {
#             "candidate_D": candidate_D,
#             "threshold": threshold,
#             "baseline_f1": baseline_f1,
#             "candidate_f1": cand_f1,
#             "min_f1": min_f1,
#             "baseline_object_recall": baseline_obj,
#             "candidate_object_recall": cand_obj,
#             "min_object_recall": min_obj,
#             "baseline_attribute_recall": baseline_attr,
#             "candidate_attribute_recall": cand_attr,
#             "min_attribute_recall": min_attr,
#             "missing_anchors": missing_anchors,
#         }

#         if not anchors_ok:
#             return False, "rejected_missing_required_anchors", details
#         if not is_caption_text_valid(candidate_caption):
#             return False, "rejected_bad_caption_text", details
#         if candidate_D > threshold:
#             return False, "rejected_above_global_threshold", details
#         if cand_f1 < min_f1:
#             return False, "rejected_below_acceptable_kg_f1", details
#         if cand_obj < min_obj:
#             return False, "rejected_below_acceptable_object_recall", details
#         if cand_attr < min_attr:
#             return False, "rejected_below_acceptable_attribute_recall", details
#         return True, "accepted_rewrite_under_acceptable_fidelity", details

#     # -----------------------------------------------------------------
#     # Reconstruction + KG extraction
#     # -----------------------------------------------------------------

#     def _reconstruct_and_extract_kgs(
#         self,
#         caption: str,
#         image_id: str,
#         prefix: str,
#         save_outputs: bool = False,
#     ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
#         output_dir = None
#         if self.save_intermediate and save_outputs:
#             output_dir = self.reconstruction_dir / safe_filename(image_id)
#             output_dir.mkdir(parents=True, exist_ok=True)

#         images = self.t2i.reconstruct(
#             caption=caption,
#             seeds=self.t2i_seeds,
#             output_dir=output_dir,
#             prefix=prefix,
#         )

#         kgs = []
#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract(
#                 img,
#                 image_id=f"{image_id}_{prefix}_seed{seed}",
#             )
#             kgs.append(kg)

#             if self.save_intermediate and save_outputs:
#                 self._save_kg_debug(
#                     kg=kg,
#                     image_id=image_id,
#                     name=f"{prefix}_seed{seed}_reconstructed_kg",
#                 )

#         return images, kgs

#     def _save_selected_candidate_outputs(
#         self,
#         images: List[Image.Image],
#         kgs: List[Dict[str, Any]],
#         caption: str,
#         image: Image.Image,
#         image_id: str,
#         prefix: str,
#     ) -> None:
#         recon_dir = self.reconstruction_dir / safe_filename(image_id)
#         recon_dir.mkdir(parents=True, exist_ok=True)

#         for idx, img in enumerate(images):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             img_path = recon_dir / f"{safe_filename(prefix)}_seed{seed}.png"
#             img.save(img_path)
#             print(f"[Final Reconstructed Image Saved] {img_path}")

#         for idx, kg in enumerate(kgs):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             self._save_kg_debug(
#                 kg=kg,
#                 image_id=image_id,
#                 name=f"{prefix}_seed{seed}_reconstructed_kg",
#             )

#         if images:
#             save_debug_sample_visuals(
#                 image_original=image,
#                 image_reconstructed=images[0],
#                 caption=caption,
#                 image_id=image_id,
#                 output_dir=self.output_dir,
#                 name_prefix=prefix,
#             )

#     # -----------------------------------------------------------------
#     # KG comparison
#     # -----------------------------------------------------------------

#     def _aggregate_kg_comparisons(
#         self,
#         kg_original: Dict[str, Any],
#         reconstructed_kgs: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:
#         comparisons = [
#             compare_scene_graphs(kg_original, kg_recon)
#             for kg_recon in reconstructed_kgs
#         ]

#         preserved: Set[Tuple[str, ...]] = set()
#         missing: Set[Tuple[str, ...]] = set()
#         extra: Set[Tuple[str, ...]] = set()

#         f1s, precisions, recalls = [], [], []
#         object_recalls, attribute_recalls, relation_recalls = [], [], []

#         for comp in comparisons:
#             preserved |= set(comp.get("preserved", set()))
#             missing |= set(comp.get("missing", set()))
#             extra |= set(comp.get("extra", set()))
#             f1s.append(comp.get("f1", 0.0))
#             precisions.append(comp.get("precision", 0.0))
#             recalls.append(comp.get("recall", 0.0))
#             object_recalls.append(comp.get("object_recall", 0.0))
#             attribute_recalls.append(comp.get("attribute_recall", 0.0))
#             relation_recalls.append(comp.get("relation_recall", 0.0))

#         return {
#             "preserved": preserved,
#             "missing": missing,
#             "extra": extra,
#             "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
#             "f1_std": float(np.std(f1s)) if f1s else 0.0,
#             "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
#             "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
#             "object_recall_mean": float(np.mean(object_recalls)) if object_recalls else 0.0,
#             "attribute_recall_mean": float(np.mean(attribute_recalls)) if attribute_recalls else 0.0,
#             "relation_recall_mean": float(np.mean(relation_recalls)) if relation_recalls else 0.0,
#             "num_preserved": len(preserved),
#             "num_missing": len(missing),
#             "num_extra": len(extra),
#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     # -----------------------------------------------------------------
#     # Saving / result helpers
#     # -----------------------------------------------------------------

#     def _make_result(
#         self,
#         image_id: str,
#         status: str,
#         original_caption: str,
#         compressed_caption: str,
#         baseline_D: float,
#         threshold: float,
#         final_dist: Dict[str, Any],
#         baseline_comparison: Dict[str, Any],
#         final_comparison: Dict[str, Any],
#         steps: List[CompressionStep],
#         start_time: float,
#     ) -> CompressionResult:
#         original_tokens = count_tokens(original_caption)
#         compressed_tokens = count_tokens(compressed_caption)
#         compression_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))
#         final_details = {
#             k: v for k, v in final_dist.items()
#             if not str(k).endswith("_std")
#         }
#         final_details["status"] = status

#         return CompressionResult(
#             image_id=image_id,
#             status=status,
#             original_caption=original_caption,
#             compressed_caption=compressed_caption,
#             original_tokens=original_tokens,
#             compressed_tokens=compressed_tokens,
#             compression_ratio=compression_ratio,
#             baseline_distortion=float(baseline_D),
#             threshold=float(threshold),
#             final_distortion=float(final_dist.get("D", baseline_D)),
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=final_comparison,
#             steps=steps,
#             total_time=time.time() - start_time,
#         )

#     def _save_kg_debug(self, kg: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(kg_to_readable_dict(kg)), f, indent=2, ensure_ascii=False)
#         print(f"[KG Saved] {path}")

#     def _save_kg_comparison_debug(self, comparison: Dict[str, Any], image_id: str, name: str) -> None:
#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)
#         path = debug_dir / f"{safe_filename(name)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(comparison), f, indent=2, ensure_ascii=False)
#         print(f"[KG Comparison Saved] {path}")

#     def _save_final_kgs_alias(self, kgs: List[Dict[str, Any]], image_id: str) -> None:
#         for idx, kg in enumerate(kgs):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             self._save_kg_debug(kg=kg, image_id=image_id, name=f"final_reconstructed_kg_seed{seed}")

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(result), f, indent=2, ensure_ascii=False)
#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             print("[Summary] No successful results to summarize.")
#             return

#         compression_ratios = [r.compression_ratio for r in results]
#         final_distortions = [r.final_distortion for r in results]
#         baseline_distortions = [r.baseline_distortion for r in results]
#         original_tokens = [r.original_tokens for r in results]
#         compressed_tokens = [r.compressed_tokens for r in results]
#         times = [r.total_time for r in results]

#         statuses: Dict[str, int] = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1

#         summary = {
#             "num_samples": len(results),
#             "status_counts": statuses,
#             "avg_compression_ratio": float(np.mean(compression_ratios)),
#             "std_compression_ratio": float(np.std(compression_ratios)),
#             "avg_baseline_distortion": float(np.mean(baseline_distortions)),
#             "std_baseline_distortion": float(np.std(baseline_distortions)),
#             "avg_final_distortion": float(np.mean(final_distortions)),
#             "std_final_distortion": float(np.std(final_distortions)),
#             "avg_original_tokens": float(np.mean(original_tokens)),
#             "avg_compressed_tokens": float(np.mean(compressed_tokens)),
#             "avg_time_seconds": float(np.mean(times)),
#             "total_time_hours": float(np.sum(times) / 3600.0),
#             "use_relative_threshold": self.use_relative_threshold,
#             "epsilon": self.epsilon,
#             "absolute_threshold": self.absolute_threshold,
#             "num_seeds": len(self.t2i_seeds),
#             "seeds": self.t2i_seeds,
#             "rewrite_mode": self.rewrite_mode,
#             "max_rewrite_candidates": self.max_rewrite_candidates,
#             "validate_top_k": self.validate_top_k,
#             "min_baseline_kg_f1": self.min_baseline_kg_f1,
#             "min_baseline_object_recall": self.min_baseline_object_recall,
#         }

#         summary_path = self.result_dir / "summary.json"
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(summary), f, indent=2)

#         print("\n" + "=" * 70)
#         print(f"SUMMARY over {len(results)} samples")
#         print(f"Status counts:          {statuses}")
#         print(f"Avg compression ratio:  {summary['avg_compression_ratio']:.2%}")
#         print(f"Avg baseline D:         {summary['avg_baseline_distortion']:.4f}")
#         print(f"Avg final D:            {summary['avg_final_distortion']:.4f}")
#         print(
#             f"Avg tokens:             "
#             f"{summary['avg_original_tokens']:.1f} -> "
#             f"{summary['avg_compressed_tokens']:.1f}"
#         )
#         print(f"Total time:             {summary['total_time_hours']:.2f} hours")
#         print(f"Saved summary:          {summary_path}")
#         print("=" * 70)

#     @staticmethod
#     def _unique_strings(values: Sequence[str]) -> List[str]:
#         seen: Set[str] = set()
#         out: List[str] = []
#         for v in values:
#             v = str(v).strip()
#             if not v:
#                 continue
#             key = normalize_for_anchor(v)
#             if key in seen:
#                 continue
#             seen.add(key)
#             out.append(v)
#         return out














# """
# KG-RD Compressor
# ================

# Main iterative compression loop for KG-guided reconstruction-compatible
# caption compression.

# Goal:
#     Compress the caption safely under acceptable reconstruction fidelity,
#     while preserving grammar, quoted text, and visually central noun chunks.

# Pipeline:
#     1. Extract KG from the original image.
#     2. Reconstruct image from caption using T2I.
#     3. Extract KG from reconstructed image.
#     4. Compare original KG and reconstructed KG.
#     5. Iteratively delete caption spans.
#     6. Accept deletion only if:
#         - distortion stays below the allowed threshold,
#         - distortion does not get worse than current caption beyond tolerance,
#         - KG-F1 does not drop too much,
#         - object recall does not drop,
#         - attribute recall does not drop too much,
#         - KG elements aligned to the deleted span are still preserved.
# """

# import json
# import re
# import string
# import time
# from dataclasses import asdict, dataclass, field
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# import numpy as np
# from PIL import Image, ImageDraw
# from tqdm import tqdm

# from .distortion import DistortionMetric
# from .scene_graph import (
#     VLMSceneGraphExtractor,
#     compare_scene_graphs,
#     normalize_label,
#     normalize_relation,
#     tuple_to_text,
# )
# from .t2i_reconstruct import T2IReconstructor


# # ---------------------------------------------------------------------
# # Data classes
# # ---------------------------------------------------------------------

# @dataclass
# class CaptionSpan:
#     text: str
#     start_char: int
#     end_char: int
#     label: str
#     token_count: int
#     aligned_preserved: List[str] = field(default_factory=list)
#     aligned_missing: List[str] = field(default_factory=list)
#     aligned_extra: List[str] = field(default_factory=list)

#     @property
#     def is_protected(self) -> bool:
#         return len(self.aligned_missing) > 0

#     @property
#     def is_visual_candidate(self) -> bool:
#         return len(self.aligned_preserved) > 0 and not self.is_protected


# @dataclass
# class CompressionStep:
#     iteration: int
#     deleted_span: str
#     deleted_label: str
#     caption_before: str
#     caption_after: str
#     tokens_before: int
#     tokens_after: int
#     token_reduction: int
#     distortion: float
#     threshold: float
#     passed_threshold: bool
#     selected: bool
#     reason: str
#     distortion_details: Dict[str, Any]
#     aligned_preserved: List[str]
#     aligned_missing: List[str]
#     time_seconds: float


# @dataclass
# class CompressionResult:
#     image_id: str
#     status: str
#     original_caption: str
#     compressed_caption: str
#     original_tokens: int
#     compressed_tokens: int
#     compression_ratio: float
#     baseline_distortion: float
#     threshold: float
#     final_distortion: float
#     final_distortion_details: Dict[str, Any]
#     baseline_kg_comparison: Dict[str, Any]
#     final_kg_comparison: Dict[str, Any]
#     steps: List[CompressionStep]
#     total_time: float


# # ---------------------------------------------------------------------
# # Helper functions
# # ---------------------------------------------------------------------

# def count_tokens(text: str) -> int:
#     return len(text.split())


# def clean_caption_spacing(text: str) -> str:
#     text = re.sub(r"\s+", " ", text).strip()
#     text = re.sub(r"\s+([,.!?;:])", r"\1", text)
#     text = re.sub(r"\(\s+", "(", text)
#     text = re.sub(r"\s+\)", ")", text)

#     text = text.replace(",.", ".")
#     text = text.replace(".,", ".")
#     text = text.replace("..", ".")
#     text = text.replace(" ,", ",")
#     text = text.replace(" .", ".")

#     return text.strip()


# def remove_char_span(text: str, start: int, end: int) -> str:
#     new_text = text[:start] + " " + text[end:]
#     return clean_caption_spacing(new_text)


# def safe_filename(name: str) -> str:
#     name = str(name)
#     name = re.sub(r"[^\w\-.]+", "_", name)
#     return name[:180]


# def dataclass_is_instance(obj: Any) -> bool:
#     return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)


# def make_json_serializable(obj: Any) -> Any:
#     if dataclass_is_instance(obj):
#         return make_json_serializable(asdict(obj))

#     if isinstance(obj, dict):
#         return {str(k): make_json_serializable(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [make_json_serializable(x) for x in obj]

#     if isinstance(obj, tuple):
#         return list(obj)

#     if isinstance(obj, set):
#         return [make_json_serializable(x) for x in sorted(obj, key=str)]

#     if isinstance(obj, (np.integer,)):
#         return int(obj)

#     if isinstance(obj, (np.floating,)):
#         return float(obj)

#     if isinstance(obj, np.ndarray):
#         return obj.tolist()

#     return obj


# def content_words(text: str) -> Set[str]:
#     text = normalize_label(text)
#     words = set(text.split())

#     stopwords = {
#         "a", "an", "the", "of", "and", "or", "to", "in", "on", "at",
#         "by", "with", "for", "from", "as", "is", "are", "was", "were",
#         "be", "being", "been", "this", "that", "these", "those",
#         "image", "photo", "picture", "scene", "view", "showing", "shows",
#     }

#     return {w for w in words if len(w) > 2 and w not in stopwords}


# def tuple_components(kg_tuple: Tuple[str, ...]) -> List[str]:
#     if not kg_tuple:
#         return []

#     if kg_tuple[0] == "object" and len(kg_tuple) == 2:
#         return [kg_tuple[1]]

#     if kg_tuple[0] == "attribute" and len(kg_tuple) == 3:
#         return [kg_tuple[1], kg_tuple[2]]

#     if kg_tuple[0] == "relation" and len(kg_tuple) == 4:
#         return [kg_tuple[1], normalize_relation(kg_tuple[2]), kg_tuple[3]]

#     return [str(x) for x in kg_tuple[1:]]


# def span_aligns_with_tuple(span_text: str, kg_tuple: Tuple[str, ...]) -> bool:
#     span_norm = normalize_label(span_text)
#     span_words = content_words(span_norm)

#     components = tuple_components(kg_tuple)

#     for comp in components:
#         comp_norm = normalize_label(comp)

#         if not comp_norm:
#             continue

#         if len(comp_norm) > 2 and comp_norm in span_norm:
#             return True

#         comp_words = content_words(comp_norm)
#         if not comp_words:
#             continue

#         overlap = span_words & comp_words
#         overlap_ratio = len(overlap) / max(len(comp_words), 1)

#         if overlap_ratio >= 0.6:
#             return True

#     return False


# def get_quoted_ranges(text: str) -> List[Tuple[int, int]]:
#     """
#     Return character ranges covered by double-quoted text.

#     Example:
#         The word "ROW" is visible -> range covering "ROW"
#     """
#     if not text:
#         return []

#     return [(m.start(), m.end()) for m in re.finditer(r'"[^"]*"', text)]


# def span_overlaps_ranges(start: int, end: int, ranges: Sequence[Tuple[int, int]]) -> bool:
#     """
#     True if [start, end) overlaps any protected range.
#     """
#     for r_start, r_end in ranges:
#         if start < r_end and end > r_start:
#             return True
#     return False


# def has_balanced_quotes(text: str) -> bool:
#     """
#     Very simple quote-balance check for captions.
#     """
#     return text.count('"') % 2 == 0


# def is_caption_text_valid(text: str) -> bool:
#     """
#     Conservative grammar/readability safety check after deleting a span.

#     This prevents broken captions such as:
#         "A angle view ..."
#         "In is \"ROW\" ..."
#         "On of is a ..."
#         "On is with the words ..."
#         "a and faded line ..."
#     """
#     if not text or len(text.split()) < 3:
#         return False

#     if not has_balanced_quotes(text):
#         return False

#     text = clean_caption_spacing(text)
#     padded = f" {text} "

#     bad_literal_patterns = [
#         ",.",
#         ".,",
#         "..",
#         " ,",
#         " .",
#         " ;",
#         " :",
#         "\".",
#         ". is ",
#         ". are ",
#         " In is ",
#         " On is ",
#         " At is ",
#         " With is ",
#         " On with ",
#         " In with ",
#         " On of ",
#         " In of ",
#         " is with ",
#         " is a and ",
#         " is an and ",
#         " is the and ",
#         " a and ",
#         " an and ",
#         " the and ",
#     ]

#     for pat in bad_literal_patterns:
#         if pat in padded:
#             return False

#     bad_regex_patterns = [
#         # Article agreement / broken noun phrase.
#         r"\bA\s+[aeiouAEIOU]\w*",
#         r"\ban\s+[^aeiouAEIOU\s]\w*",

#         # Broken sentence openings after deleting location phrases.
#         r"(^|[.!?]\s+)(In|On|At|With|Of)\s+(is|are|was|were|with|of)\b",
#         r"(^|[.!?]\s+)(In|On|At|With|Of)\s*[,.;:]",
#         r"(^|[.!?]\s+)(In|On|At|With|Of)\s+of\s+(is|are|was|were)\b",

#         # Broken copular frame.
#         r"\b(is|are|was|were)\s+(a|an|the)?\s*(and|with|[,.;:])\b",
#         r"\b(is|are|was|were)\s+(with|of)\b",

#         # Broken quoted-text context.
#         r"\bwith\s+the\s+words\s*[,.;:]",
#         r"\bwith\s*[,.;:]",
#         r"\bword\s*[,.;:]",
#         r"\bletter\s*\"[^\"]+\"\s+placed\s*[\.;]",
#         r"\bplaced\s*[\.;]\s*(On|In|At|With)\b",

#         # Repeated function-word fragments.
#         r"\b(of|in|on|with)\s+(of|in|on|with|is|are|was|were)\b",
#         r"\b(the|a|an)\s+(of|in|on|with|is|are|was|were|and)\b",
#     ]

#     for pattern in bad_regex_patterns:
#         if re.search(pattern, text):
#             return False

#     sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
#     for sent in sentences:
#         if re.match(r"^(In|On|At|With|Of)\s+(is|are|was|were|with|of)\b", sent):
#             return False
#         if re.match(r"^(In|On|At|With|Of)\s*$", sent):
#             return False
#         if re.search(r"\b(is|are|was|were)\s+(a|an|the)?\s*(and|with|of)\b", sent):
#             return False

#     return True



# def get_grammar_anchor_ranges(caption: str) -> List[Tuple[int, int]]:
#     """
#     Ranges that should not be deleted because they form the grammatical
#     skeleton of common visual captions.

#     These anchors are dataset-agnostic enough for image captions:
#         - "In the middle of the view ..."
#         - "On the right side of the image ..."
#         - quoted/OCR text anchors
#         - "with the words \"...\""
#     """
#     ranges: List[Tuple[int, int]] = []

#     patterns = [
#         r"\bIn\s+the\s+middle(?:\s+of\s+the\s+\w+)?",
#         r"\bOn\s+the\s+(?:left|right)\s+side(?:\s+of\s+the\s+\w+)?",
#         r"\bAt\s+the\s+(?:top|bottom|center|centre)(?:\s+of\s+the\s+\w+)?",
#         r"\bwith\s+the\s+words\s+\"[^\"]+\"",
#         r"\bword\s+\"[^\"]+\"",
#         r"\bletter\s+\"[^\"]+\"(?:\s+placed\s+\w+\s+\w+)?",
#     ]

#     for pattern in patterns:
#         for m in re.finditer(pattern, caption, flags=re.IGNORECASE):
#             ranges.append((m.start(), m.end()))

#     ranges.extend(get_quoted_ranges(caption))

#     return merge_ranges(ranges)


# def merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
#     if not ranges:
#         return []

#     sorted_ranges = sorted((int(s), int(e)) for s, e in ranges if s < e)
#     merged: List[Tuple[int, int]] = []

#     for start, end in sorted_ranges:
#         if not merged or start > merged[-1][1]:
#             merged.append((start, end))
#         else:
#             old_start, old_end = merged[-1]
#             merged[-1] = (old_start, max(old_end, end))

#     return merged


# def span_is_inside_ranges(start: int, end: int, ranges: Sequence[Tuple[int, int]]) -> bool:
#     return any(start >= r_start and end <= r_end for r_start, r_end in ranges)


# def kg_to_readable_dict(kg: Dict[str, Any]) -> Dict[str, Any]:
#     tuples = kg.get("tuples", set())

#     return {
#         "image_id": kg.get("image_id"),
#         "objects": kg.get("objects", []),
#         "attributes": kg.get("attributes", []),
#         "relations": kg.get("relations", []),
#         "tuples": [list(t) for t in sorted(list(tuples), key=str)],
#         "raw_vlm_output": kg.get("raw_vlm_output", ""),
#     }


# def print_kg(title: str, kg: Dict[str, Any], max_items: int = 40) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)

#     print("\nOBJECTS:")
#     for obj in kg.get("objects", [])[:max_items]:
#         label = obj.get("label", obj.get("name", ""))
#         attrs = obj.get("attributes", [])
#         print(f"  - {label} | attributes={attrs}")

#     print("\nATTRIBUTES:")
#     for attr in kg.get("attributes", [])[:max_items]:
#         obj = attr.get("object", "")
#         attribute = attr.get("attribute", "")
#         print(f"  - ({obj}, has_attribute, {attribute})")

#     print("\nRELATIONS:")
#     for rel in kg.get("relations", [])[:max_items]:
#         subj = rel.get("subject", "")
#         pred = rel.get("predicate", rel.get("relation", ""))
#         obj = rel.get("object", "")
#         print(f"  - ({subj}, {pred}, {obj})")

#     print("\nKG TUPLES:")
#     for t in list(kg.get("tuples", set()))[:max_items]:
#         print(f"  - {t}")

#     print("=" * 100)


# def print_kg_comparison(title: str, comparison: Dict[str, Any], max_items: int = 60) -> None:
#     print("\n" + "=" * 100)
#     print(title)
#     print("=" * 100)

#     print(
#         f"F1 mean={comparison.get('f1_mean', 0.0):.4f} | "
#         f"Recall mean={comparison.get('recall_mean', 0.0):.4f} | "
#         f"Object recall={comparison.get('object_recall_mean', 0.0):.4f} | "
#         f"Attribute recall={comparison.get('attribute_recall_mean', 0.0):.4f} | "
#         f"Relation recall={comparison.get('relation_recall_mean', 0.0):.4f}"
#     )

#     print("\nPRESERVED ELEMENTS:")
#     for x in comparison.get("preserved_text", [])[:max_items]:
#         print("  +", x)

#     print("\nMISSING ELEMENTS:")
#     for x in comparison.get("missing_text", [])[:max_items]:
#         print("  -", x)

#     print("\nEXTRA / HALLUCINATED ELEMENTS:")
#     for x in comparison.get("extra_text", [])[:max_items]:
#         print("  *", x)

#     print("=" * 100)


# def save_debug_sample_visuals(
#     image_original: Image.Image,
#     image_reconstructed: Image.Image,
#     caption: str,
#     image_id: str,
#     output_dir: Path,
# ) -> None:
#     debug_dir = output_dir / "debug_samples" / safe_filename(image_id)
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     original_path = debug_dir / "original_image.png"
#     reconstructed_path = debug_dir / "reconstructed_from_caption.png"
#     caption_path = debug_dir / "caption.txt"
#     side_by_side_path = debug_dir / "side_by_side_original_vs_reconstructed.png"

#     image_original_rgb = image_original.convert("RGB")
#     image_reconstructed_rgb = image_reconstructed.convert("RGB")

#     image_original_rgb.save(original_path)
#     image_reconstructed_rgb.save(reconstructed_path)

#     with open(caption_path, "w", encoding="utf-8") as f:
#         f.write(caption)

#     left = image_original_rgb.resize((512, 512))
#     right = image_reconstructed_rgb.resize((512, 512))

#     canvas = Image.new("RGB", (1024, 560), color="white")
#     canvas.paste(left, (0, 48))
#     canvas.paste(right, (512, 48))

#     draw = ImageDraw.Draw(canvas)
#     draw.text((170, 15), "Original Image", fill="black")
#     draw.text((655, 15), "Reconstructed from Caption", fill="black")

#     canvas.save(side_by_side_path)

#     print(f"[Debug Image Saved] {original_path}")
#     print(f"[Debug Image Saved] {reconstructed_path}")
#     print(f"[Debug Caption Saved] {caption_path}")
#     print(f"[Debug Side-by-Side Saved] {side_by_side_path}")


# # ---------------------------------------------------------------------
# # Main compressor
# # ---------------------------------------------------------------------

# class KGRDCompressor:
#     """
#     Knowledge-Graph-guided Rate-Distortion Caption Compressor.
#     """

#     def __init__(
#         self,
#         scene_graph_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
#         scene_graph_device: str = "auto",
#         scene_graph_dtype: str = "float16",
#         scene_graph_max_new_tokens: int = 1024,
#         scene_graph_temperature: float = 0.0,
#         scene_graph_do_sample: bool = False,
#         max_objects: int = 50,
#         max_relations: int = 80,

#         t2i_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
#         t2i_device: str = "auto",
#         t2i_dtype: str = "float16",
#         t2i_steps: int = 30,
#         t2i_guidance: float = 7.5,
#         t2i_image_size: int = 512,
#         t2i_seeds: Optional[List[int]] = None,

#         alpha: float = 0.4,
#         beta: float = 0.4,
#         gamma: float = 0.2,
#         clip_model: str = "ViT-L-14",
#         clip_pretrained: str = "openai",
#         lpips_net: str = "alex",
#         metric_device: str = "auto",

#         use_relative_threshold: bool = True,
#         epsilon: float = 0.10,
#         absolute_threshold: float = 0.25,

#         min_caption_tokens: int = 5,
#         max_iterations: int = 50,
#         max_candidates_per_iteration: int = 10,
#         allow_unaligned_candidates: bool = False,

#         # Stop/quality controls.
#         # These defaults prevent over-compression while still allowing
#         # meaningful compression.
#         target_compression_ratio: float = 0.35,
#         max_compression_ratio: float = 0.45,
#         stop_when_target_reached: bool = True,

#         min_baseline_kg_f1: float = 0.05,
#         min_baseline_object_recall: float = 0.10,

#         save_intermediate: bool = True,
#         output_dir: str = "./outputs",
#     ):
#         self.t2i_seeds = t2i_seeds or [42]

#         self.use_relative_threshold = bool(use_relative_threshold)
#         self.epsilon = float(epsilon)
#         self.absolute_threshold = float(absolute_threshold)

#         self.min_caption_tokens = int(min_caption_tokens)
#         self.max_iterations = int(max_iterations)
#         self.max_candidates_per_iteration = int(max_candidates_per_iteration)
#         self.allow_unaligned_candidates = bool(allow_unaligned_candidates)

#         self.target_compression_ratio = float(target_compression_ratio)
#         self.max_compression_ratio = float(max_compression_ratio)
#         self.stop_when_target_reached = bool(stop_when_target_reached)

#         self.min_baseline_kg_f1 = float(min_baseline_kg_f1)
#         self.min_baseline_object_recall = float(min_baseline_object_recall)

#         self.save_intermediate = bool(save_intermediate)
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.reconstruction_dir = self.output_dir / "reconstructions"
#         self.result_dir = self.output_dir / "results"
#         self.kg_debug_dir = self.output_dir / "kg_debug"

#         self.result_dir.mkdir(parents=True, exist_ok=True)
#         self.kg_debug_dir.mkdir(parents=True, exist_ok=True)

#         self._spacy_nlp = None

#         print("=" * 70)
#         print("KG-RD Compressor: Initializing VLM-based pipeline")
#         print("=" * 70)

#         print("\n[1/3] Loading VLM scene graph extractor...")
#         self.sg_extractor = VLMSceneGraphExtractor(
#             model_name=scene_graph_model,
#             device=scene_graph_device,
#             dtype=scene_graph_dtype,
#             max_new_tokens=scene_graph_max_new_tokens,
#             temperature=scene_graph_temperature,
#             do_sample=scene_graph_do_sample,
#             max_objects=max_objects,
#             max_relations=max_relations,
#         )

#         print("\n[2/3] Loading text-to-image reconstructor...")
#         self.t2i = T2IReconstructor(
#             model_name=t2i_model,
#             device=t2i_device,
#             dtype=t2i_dtype,
#             num_inference_steps=t2i_steps,
#             guidance_scale=t2i_guidance,
#             image_size=t2i_image_size,
#         )

#         print("\n[3/3] Loading distortion metric...")
#         self.distortion = DistortionMetric(
#             alpha=alpha,
#             beta=beta,
#             gamma=gamma,
#             clip_model_name=clip_model,
#             clip_pretrained=clip_pretrained,
#             lpips_net=lpips_net,
#             device=metric_device,
#         )

#         print("\n" + "=" * 70)
#         if self.use_relative_threshold:
#             print(f"Pipeline ready | relative threshold: delta = D0 * (1 + {self.epsilon})")
#         else:
#             print(f"Pipeline ready | absolute threshold: delta = {self.absolute_threshold}")
#         print(
#             f"Baseline gate | min KG-F1={self.min_baseline_kg_f1}, "
#             f"min object recall={self.min_baseline_object_recall}"
#         )
#         print(
#             f"Compression target | target={self.target_compression_ratio:.0%}, "
#             f"max={self.max_compression_ratio:.0%}, "
#             f"stop_when_target_reached={self.stop_when_target_reached}"
#         )
#         print("=" * 70 + "\n")

#     # -----------------------------------------------------------------
#     # Public API
#     # -----------------------------------------------------------------

#     def compress_single(
#         self,
#         image: Image.Image,
#         caption: str,
#         image_id: str = "unknown",
#     ) -> CompressionResult:

#         start_time = time.time()

#         if not isinstance(image, Image.Image):
#             raise TypeError("image must be a PIL.Image.Image.")

#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         caption = clean_caption_spacing(caption)
#         original_tokens = count_tokens(caption)

#         if original_tokens <= self.min_caption_tokens:
#             raise ValueError(
#                 f"Caption has only {original_tokens} tokens, "
#                 f"min_caption_tokens={self.min_caption_tokens}."
#             )

#         image_id_safe = safe_filename(image_id)

#         print("\n" + "-" * 70)
#         print(f"Compressing image_id={image_id} | original_tokens={original_tokens}")
#         print("-" * 70)

#         steps: List[CompressionStep] = []

#         print("[1] Extracting KG from original image...")
#         kg_original = self.sg_extractor.extract(image, image_id=image_id)

#         print(
#             f"    Original KG: "
#             f"{len(kg_original.get('objects', []))} objects, "
#             f"{len(kg_original.get('attributes', []))} attributes, "
#             f"{len(kg_original.get('relations', []))} relations"
#         )

#         if self.save_intermediate:
#             self._save_kg_debug(kg_original, image_id_safe, "original_image_kg")

#         print_kg("ORIGINAL IMAGE KG", kg_original, max_items=40)

#         print("[2] Generating baseline reconstruction from full caption...")

#         baseline_images, baseline_kgs = self._reconstruct_and_extract_kgs(
#             caption=caption,
#             image_id=image_id_safe,
#             prefix="baseline_full_caption",
#         )

#         if self.save_intermediate and baseline_images:
#             save_debug_sample_visuals(
#                 image_original=image,
#                 image_reconstructed=baseline_images[0],
#                 caption=caption,
#                 image_id=image_id_safe,
#                 output_dir=self.output_dir,
#             )

#         if baseline_kgs:
#             print_kg("BASELINE RECONSTRUCTED IMAGE KG", baseline_kgs[0], max_items=40)

#         baseline_dist = self.distortion.compute_averaged_distortion(
#             image_orig=image,
#             images_recon=baseline_images,
#             kg_original=kg_original,
#             kgs_reconstructed=baseline_kgs,
#         )

#         baseline_D = float(baseline_dist["D"])

#         if self.use_relative_threshold:
#             threshold = self.distortion.compute_relative_threshold(
#                 baseline_distortion=baseline_D,
#                 epsilon=self.epsilon,
#             )
#         else:
#             threshold = self.absolute_threshold

#         baseline_comparison = self._aggregate_kg_comparisons(
#             kg_original=kg_original,
#             reconstructed_kgs=baseline_kgs,
#         )

#         if self.save_intermediate:
#             self._save_kg_comparison_debug(
#                 comparison=baseline_comparison,
#                 image_id=image_id_safe,
#                 name="baseline_kg_comparison",
#             )

#         print(f"    Baseline distortion D0 = {baseline_D:.4f}")
#         print(f"    Compression threshold delta = {threshold:.4f}")

#         print_kg_comparison("BASELINE KG COMPARISON", baseline_comparison)

#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))

#         if baseline_f1 < self.min_baseline_kg_f1 or baseline_obj_recall < self.min_baseline_object_recall:
#             print(
#                 "\n[SKIP] Baseline reconstruction fidelity is too low. "
#                 "Compression is not meaningful for this sample."
#             )
#             print(f"       Baseline KG-F1: {baseline_f1:.4f}")
#             print(f"       Baseline object recall: {baseline_obj_recall:.4f}")

#             final_details = {
#                 k: v for k, v in baseline_dist.items()
#                 if not str(k).endswith("_std")
#             }
#             final_details["status"] = "skipped_low_baseline_fidelity"

#             if self.save_intermediate:
#                 self._save_final_kgs_alias(baseline_kgs, image_id_safe)

#             result = CompressionResult(
#                 image_id=image_id,
#                 status="skipped_low_baseline_fidelity",
#                 original_caption=caption,
#                 compressed_caption=caption,
#                 original_tokens=original_tokens,
#                 compressed_tokens=original_tokens,
#                 compression_ratio=0.0,
#                 baseline_distortion=baseline_D,
#                 threshold=threshold,
#                 final_distortion=baseline_D,
#                 final_distortion_details=final_details,
#                 baseline_kg_comparison=baseline_comparison,
#                 final_kg_comparison=baseline_comparison,
#                 steps=[],
#                 total_time=time.time() - start_time,
#             )

#             if self.save_intermediate:
#                 self._save_result(result)

#             return result

#         current_caption = caption
#         current_kgs = baseline_kgs
#         current_dist = baseline_dist

#         iteration = 0

#         while iteration < self.max_iterations:
#             iteration += 1
#             current_tokens = count_tokens(current_caption)
#             current_compression_ratio = 1.0 - (current_tokens / max(original_tokens, 1))

#             if current_tokens <= self.min_caption_tokens:
#                 print(f"[Iter {iteration}] Minimum caption length reached. Stopping.")
#                 break

#             if (
#                 self.stop_when_target_reached
#                 and current_compression_ratio >= self.target_compression_ratio
#             ):
#                 print(
#                     f"[Iter {iteration}] Target compression reached "
#                     f"({current_compression_ratio:.2%} >= "
#                     f"{self.target_compression_ratio:.2%}). Stopping."
#                 )
#                 break

#             print(
#                 f"\n[Iter {iteration}] Current tokens: {current_tokens} | "
#                 f"CR={current_compression_ratio:.2%}"
#             )

#             current_comparison = self._aggregate_kg_comparisons(
#                 kg_original=kg_original,
#                 reconstructed_kgs=current_kgs,
#             )

#             spans = self._extract_caption_spans(current_caption)

#             protected_spans, candidate_spans = self._build_protected_and_candidate_spans(
#                 spans=spans,
#                 preserved=current_comparison["preserved"],
#                 missing=current_comparison["missing"],
#                 extra=current_comparison["extra"],
#                 caption=current_caption,
#             )

#             print(
#                 f"    Spans: {len(spans)} | "
#                 f"Protected: {len(protected_spans)} | "
#                 f"Candidates: {len(candidate_spans)}"
#             )

#             print("    Top candidates:")
#             for cand in candidate_spans[: min(10, len(candidate_spans))]:
#                 print(
#                     f"      - '{cand.text}' | label={cand.label} | "
#                     f"tokens={cand.token_count} | "
#                     f"preserved={cand.aligned_preserved} | "
#                     f"missing={cand.aligned_missing} | "
#                     f"extra={cand.aligned_extra}"
#                 )

#             if not candidate_spans:
#                 print(f"[Iter {iteration}] No removable candidates. Stopping.")
#                 break

#             candidate_spans = candidate_spans[: self.max_candidates_per_iteration]
#             accepted_candidates = []

#             # ---------------------------------------------------------
#             # Fast path: validate one batch of low-risk deletions.
#             # If the batch passes, accept it immediately and avoid many
#             # expensive FLUX+KG calls for small individual deletions.
#             # ---------------------------------------------------------
#             batch_candidate = self._make_batch_deletion_candidate(
#                 caption=current_caption,
#                 candidate_spans=candidate_spans,
#                 original_tokens=original_tokens,
#                 max_batch_spans=3,
#                 max_batch_tokens=6,
#             )

#             if batch_candidate is not None:
#                 batch_span, batch_caption, batch_token_reduction = batch_candidate
#                 batch_step_start = time.time()

#                 batch_before_tokens = count_tokens(current_caption)
#                 batch_after_tokens = count_tokens(batch_caption)

#                 print(
#                     f"    [BATCH TEST] remove '{batch_span.text}' | "
#                     f"tokens {batch_before_tokens}->{batch_after_tokens}"
#                 )

#                 batch_images, batch_kgs = self._reconstruct_and_extract_kgs(
#                     caption=batch_caption,
#                     image_id=image_id_safe,
#                     prefix=f"iter{iteration}_batch_delete_{safe_filename(batch_span.text)}",
#                 )

#                 batch_dist = self.distortion.compute_averaged_distortion(
#                     image_orig=image,
#                     images_recon=batch_images,
#                     kg_original=kg_original,
#                     kgs_reconstructed=batch_kgs,
#                 )

#                 batch_D = float(batch_dist["D"])
#                 current_D = float(current_dist["D"])

#                 batch_comparison = self._aggregate_kg_comparisons(
#                     kg_original=kg_original,
#                     reconstructed_kgs=batch_kgs,
#                 )

#                 batch_passed, batch_reason, batch_acceptance_details = self._candidate_acceptance_decision(
#                     span=batch_span,
#                     current_D=current_D,
#                     candidate_D=batch_D,
#                     threshold=threshold,
#                     current_comparison=current_comparison,
#                     candidate_comparison=batch_comparison,
#                     baseline_comparison=baseline_comparison,
#                 )

#                 batch_details = {
#                     k: v for k, v in batch_dist.items()
#                     if not str(k).endswith("_std")
#                 }
#                 batch_details.update(
#                     {
#                         "candidate_kg_f1": batch_comparison.get("f1_mean", 0.0),
#                         "candidate_object_recall": batch_comparison.get("object_recall_mean", 0.0),
#                         "candidate_attribute_recall": batch_comparison.get("attribute_recall_mean", 0.0),
#                         "candidate_relation_recall": batch_comparison.get("relation_recall_mean", 0.0),
#                         "acceptance_reason": batch_reason,
#                         "acceptance_details": batch_acceptance_details,
#                     }
#                 )

#                 batch_step = CompressionStep(
#                     iteration=iteration,
#                     deleted_span=batch_span.text,
#                     deleted_label=batch_span.label,
#                     caption_before=current_caption,
#                     caption_after=batch_caption,
#                     tokens_before=batch_before_tokens,
#                     tokens_after=batch_after_tokens,
#                     token_reduction=batch_token_reduction,
#                     distortion=batch_D,
#                     threshold=threshold,
#                     passed_threshold=batch_passed,
#                     selected=batch_passed,
#                     reason=batch_reason,
#                     distortion_details=batch_details,
#                     aligned_preserved=batch_span.aligned_preserved,
#                     aligned_missing=batch_span.aligned_missing,
#                     time_seconds=time.time() - batch_step_start,
#                 )
#                 steps.append(batch_step)

#                 batch_status = "PASS" if batch_passed else "FAIL"
#                 print(
#                     f"    [{batch_status} BATCH] D={batch_D:.4f} | "
#                     f"reason={batch_reason}"
#                 )

#                 if batch_passed:
#                     current_caption = batch_caption
#                     current_kgs = batch_kgs
#                     current_dist = batch_dist

#                     print(
#                         f"[Iter {iteration}] SELECTED batch deletion: "
#                         f"'{batch_span.text}' | D={batch_D:.4f} | "
#                         f"tokens={count_tokens(current_caption)}"
#                     )
#                     continue

#             for span in candidate_spans:
#                 step_start = time.time()

#                 candidate_caption = remove_char_span(
#                     current_caption,
#                     span.start_char,
#                     span.end_char,
#                 )

#                 before_tokens = count_tokens(current_caption)
#                 after_tokens = count_tokens(candidate_caption)
#                 token_reduction = before_tokens - after_tokens

#                 if after_tokens < self.min_caption_tokens:
#                     steps.append(
#                         self._make_rejected_step(
#                             iteration, span, current_caption, candidate_caption,
#                             before_tokens, after_tokens, token_reduction,
#                             threshold, "rejected_min_caption_tokens",
#                             time.time() - step_start,
#                         )
#                     )
#                     continue

#                 if token_reduction <= 0:
#                     steps.append(
#                         self._make_rejected_step(
#                             iteration, span, current_caption, candidate_caption,
#                             before_tokens, after_tokens, token_reduction,
#                             threshold, "rejected_no_token_reduction",
#                             time.time() - step_start,
#                         )
#                     )
#                     continue

#                 if not self._is_candidate_deletion_grammar_safe(
#                     current_caption,
#                     span,
#                     candidate_caption,
#                 ):
#                     steps.append(
#                         self._make_rejected_step(
#                             iteration, span, current_caption, candidate_caption,
#                             before_tokens, after_tokens, token_reduction,
#                             threshold, "rejected_bad_caption_text",
#                             time.time() - step_start,
#                         )
#                     )
#                     print(f"    [FAIL] remove '{span.text}' rejected due to bad caption text")
#                     continue

#                 candidate_compression_ratio = 1.0 - (after_tokens / max(original_tokens, 1))
#                 if candidate_compression_ratio > self.max_compression_ratio:
#                     steps.append(
#                         self._make_rejected_step(
#                             iteration, span, current_caption, candidate_caption,
#                             before_tokens, after_tokens, token_reduction,
#                             threshold, "rejected_above_max_compression_ratio",
#                             time.time() - step_start,
#                         )
#                     )
#                     print(
#                         f"    [FAIL] remove '{span.text}' rejected because "
#                         f"CR={candidate_compression_ratio:.2%} exceeds "
#                         f"max={self.max_compression_ratio:.2%}"
#                     )
#                     continue

#                 candidate_images, candidate_kgs = self._reconstruct_and_extract_kgs(
#                     caption=candidate_caption,
#                     image_id=image_id_safe,
#                     prefix=f"iter{iteration}_delete_{safe_filename(span.text)}",
#                 )

#                 candidate_dist = self.distortion.compute_averaged_distortion(
#                     image_orig=image,
#                     images_recon=candidate_images,
#                     kg_original=kg_original,
#                     kgs_reconstructed=candidate_kgs,
#                 )

#                 candidate_D = float(candidate_dist["D"])
#                 current_D = float(current_dist["D"])

#                 candidate_comparison = self._aggregate_kg_comparisons(
#                     kg_original=kg_original,
#                     reconstructed_kgs=candidate_kgs,
#                 )

#                 passed, reason, acceptance_details = self._candidate_acceptance_decision(
#                     span=span,
#                     current_D=current_D,
#                     candidate_D=candidate_D,
#                     threshold=threshold,
#                     current_comparison=current_comparison,
#                     candidate_comparison=candidate_comparison,
#                     baseline_comparison=baseline_comparison,
#                 )

#                 distortion_details = {
#                     k: v for k, v in candidate_dist.items()
#                     if not str(k).endswith("_std")
#                 }

#                 distortion_details.update(
#                     {
#                         "candidate_kg_f1": candidate_comparison.get("f1_mean", 0.0),
#                         "candidate_object_recall": candidate_comparison.get("object_recall_mean", 0.0),
#                         "candidate_attribute_recall": candidate_comparison.get("attribute_recall_mean", 0.0),
#                         "candidate_relation_recall": candidate_comparison.get("relation_recall_mean", 0.0),
#                         "acceptance_reason": reason,
#                         "acceptance_details": acceptance_details,
#                     }
#                 )

#                 step = CompressionStep(
#                     iteration=iteration,
#                     deleted_span=span.text,
#                     deleted_label=span.label,
#                     caption_before=current_caption,
#                     caption_after=candidate_caption,
#                     tokens_before=before_tokens,
#                     tokens_after=after_tokens,
#                     token_reduction=token_reduction,
#                     distortion=candidate_D,
#                     threshold=threshold,
#                     passed_threshold=passed,
#                     selected=False,
#                     reason=reason,
#                     distortion_details=distortion_details,
#                     aligned_preserved=span.aligned_preserved,
#                     aligned_missing=span.aligned_missing,
#                     time_seconds=time.time() - step_start,
#                 )

#                 steps.append(step)

#                 status = "PASS" if passed else "FAIL"
#                 print(
#                     f"    [{status}] remove '{span.text}' "
#                     f"({span.label}) | D={candidate_D:.4f} | "
#                     f"tokens {before_tokens}->{after_tokens} | reason={reason}"
#                 )

#                 if not passed and acceptance_details.get("lost_critical_tuples"):
#                     print("        Lost critical KG elements:")
#                     for lost in acceptance_details["lost_critical_tuples"]:
#                         print(f"          - {lost}")

#                 if passed:
#                     accepted_candidates.append(
#                         {
#                             "span": span,
#                             "step": step,
#                             "caption": candidate_caption,
#                             "kgs": candidate_kgs,
#                             "distortion": candidate_dist,
#                             "comparison": candidate_comparison,
#                             "D": candidate_D,
#                             "token_reduction": token_reduction,
#                             "tokens_after": after_tokens,
#                         }
#                     )

#             if not accepted_candidates:
#                 print(f"[Iter {iteration}] All tested deletions rejected. Stopping.")
#                 break

#             # Maximize compression among safe candidates, then choose better fidelity.
#             # Maximum compression under acceptable fidelity:
#             # choose the largest token reduction first; if tied, choose lower distortion.
#             accepted_candidates.sort(
#                 key=lambda x: (
#                     -x["token_reduction"],
#                     x["D"],
#                     x["tokens_after"],
#                 )
#             )

#             best = accepted_candidates[0]
#             best["step"].selected = True

#             current_caption = best["caption"]
#             current_kgs = best["kgs"]
#             current_dist = best["distortion"]

#             print(
#                 f"[Iter {iteration}] SELECTED deletion: '{best['span'].text}' | "
#                 f"D={best['D']:.4f} | tokens={count_tokens(current_caption)}"
#             )

#         compressed_tokens = count_tokens(current_caption)
#         compression_ratio = 1.0 - (compressed_tokens / max(original_tokens, 1))
#         total_time = time.time() - start_time

#         final_comparison = self._aggregate_kg_comparisons(
#             kg_original=kg_original,
#             reconstructed_kgs=current_kgs,
#         )

#         if self.save_intermediate:
#             self._save_final_kgs_alias(current_kgs, image_id_safe)
#             self._save_kg_comparison_debug(
#                 comparison=final_comparison,
#                 image_id=image_id_safe,
#                 name="final_kg_comparison",
#             )

#         print_kg_comparison("FINAL KG COMPARISON", final_comparison)

#         final_details = {
#             k: v for k, v in current_dist.items()
#             if not str(k).endswith("_std")
#         }
#         final_details["status"] = "compressed" if compression_ratio > 0 else "not_compressed"

#         result = CompressionResult(
#             image_id=image_id,
#             status=final_details["status"],
#             original_caption=caption,
#             compressed_caption=current_caption,
#             original_tokens=original_tokens,
#             compressed_tokens=compressed_tokens,
#             compression_ratio=compression_ratio,
#             baseline_distortion=baseline_D,
#             threshold=threshold,
#             final_distortion=float(current_dist["D"]),
#             final_distortion_details=final_details,
#             baseline_kg_comparison=baseline_comparison,
#             final_kg_comparison=final_comparison,
#             steps=steps,
#             total_time=total_time,
#         )

#         print("\n" + "-" * 70)
#         print(
#             f"Result for {image_id}: "
#             f"{original_tokens} -> {compressed_tokens} tokens | "
#             f"CR={compression_ratio:.2%} | "
#             f"Final D={result.final_distortion:.4f} | "
#             f"delta={threshold:.4f} | "
#             f"status={result.status}"
#         )
#         print("-" * 70)
#         print("\nORIGINAL CAPTION:")
#         print(caption)
#         print("\nFINAL COMPRESSED CAPTION:")
#         print(current_caption)

#         if self.save_intermediate:
#             self._save_result(result)

#         return result

#     def compress_dataset(
#         self,
#         dataset: Any,
#         max_samples: Optional[int] = None,
#     ) -> List[CompressionResult]:

#         n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
#         results: List[CompressionResult] = []

#         print("\n" + "=" * 70)
#         print(f"Processing {n} image-caption pairs")
#         print("=" * 70)

#         for idx in tqdm(range(n), desc="KG-RD compression"):
#             sample = dataset[idx]

#             try:
#                 result = self.compress_single(
#                     image=sample["image"],
#                     caption=sample["caption"],
#                     image_id=sample.get("image_id", f"sample_{idx:06d}"),
#                 )
#                 results.append(result)

#             except Exception as exc:
#                 print(
#                     f"[ERROR] Failed on sample index={idx}, "
#                     f"image_id={sample.get('image_id', 'unknown')}: {exc}"
#                 )
#                 continue

#         self._save_summary(results)
#         return results

#     # -----------------------------------------------------------------
#     # Reconstruction + KG extraction
#     # -----------------------------------------------------------------

#     def _reconstruct_and_extract_kgs(
#         self,
#         caption: str,
#         image_id: str,
#         prefix: str,
#     ) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:

#         output_dir = None

#         if self.save_intermediate:
#             output_dir = self.reconstruction_dir / safe_filename(image_id)
#             output_dir.mkdir(parents=True, exist_ok=True)

#         images = self.t2i.reconstruct(
#             caption=caption,
#             seeds=self.t2i_seeds,
#             output_dir=output_dir,
#             prefix=prefix,
#         )

#         kgs = []

#         for seed, img in zip(self.t2i_seeds, images):
#             kg = self.sg_extractor.extract(
#                 img,
#                 image_id=f"{image_id}_{prefix}_seed{seed}",
#             )
#             kgs.append(kg)

#             if self.save_intermediate:
#                 self._save_kg_debug(
#                     kg=kg,
#                     image_id=image_id,
#                     name=f"{prefix}_seed{seed}_reconstructed_kg",
#                 )

#         return images, kgs

#     # -----------------------------------------------------------------
#     # KG comparison
#     # -----------------------------------------------------------------

#     def _aggregate_kg_comparisons(
#         self,
#         kg_original: Dict[str, Any],
#         reconstructed_kgs: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:

#         comparisons = [
#             compare_scene_graphs(kg_original, kg_recon)
#             for kg_recon in reconstructed_kgs
#         ]

#         preserved: Set[Tuple[str, ...]] = set()
#         missing: Set[Tuple[str, ...]] = set()
#         extra: Set[Tuple[str, ...]] = set()

#         f1s = []
#         precisions = []
#         recalls = []
#         object_recalls = []
#         attribute_recalls = []
#         relation_recalls = []

#         for comp in comparisons:
#             preserved |= set(comp.get("preserved", set()))
#             missing |= set(comp.get("missing", set()))
#             extra |= set(comp.get("extra", set()))

#             f1s.append(comp.get("f1", 0.0))
#             precisions.append(comp.get("precision", 0.0))
#             recalls.append(comp.get("recall", 0.0))
#             object_recalls.append(comp.get("object_recall", 0.0))
#             attribute_recalls.append(comp.get("attribute_recall", 0.0))
#             relation_recalls.append(comp.get("relation_recall", 0.0))

#         return {
#             "preserved": preserved,
#             "missing": missing,
#             "extra": extra,

#             "f1_mean": float(np.mean(f1s)) if f1s else 0.0,
#             "f1_std": float(np.std(f1s)) if f1s else 0.0,

#             "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
#             "recall_mean": float(np.mean(recalls)) if recalls else 0.0,

#             "object_recall_mean": float(np.mean(object_recalls)) if object_recalls else 0.0,
#             "attribute_recall_mean": float(np.mean(attribute_recalls)) if attribute_recalls else 0.0,
#             "relation_recall_mean": float(np.mean(relation_recalls)) if relation_recalls else 0.0,

#             "num_preserved": len(preserved),
#             "num_missing": len(missing),
#             "num_extra": len(extra),

#             "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
#             "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
#             "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
#         }

#     # -----------------------------------------------------------------
#     # Caption span extraction
#     # -----------------------------------------------------------------

#     def _get_spacy_nlp(self):
#         if self._spacy_nlp is None:
#             import spacy
#             self._spacy_nlp = spacy.load("en_core_web_sm")
#         return self._spacy_nlp

#     def _extract_caption_spans(self, caption: str) -> List[CaptionSpan]:
#         try:
#             return self._extract_spacy_spans(caption)
#         except Exception as exc:
#             print(f"    [Warning] spaCy span extraction failed: {exc}")
#             print("    [Warning] Using regex fallback span extraction.")
#             return self._extract_regex_spans(caption)

#     def _extract_spacy_spans(self, caption: str) -> List[CaptionSpan]:
#         nlp = self._get_spacy_nlp()
#         doc = nlp(caption)

#         raw_spans = []

#         for chunk in doc.noun_chunks:
#             raw_spans.append((chunk.start_char, chunk.end_char, "NOUN_CHUNK"))

#         for token in doc:
#             if token.dep_ in {"prep", "relcl", "advcl", "acl"}:
#                 subtree = list(token.subtree)
#                 start = min(t.idx for t in subtree)
#                 end = max(t.idx + len(t.text) for t in subtree)
#                 label = token.dep_.upper()
#                 raw_spans.append((start, end, label))

#             if token.pos_ in {"ADJ", "ADV"} and len(token.text) > 2:
#                 raw_spans.append((token.idx, token.idx + len(token.text), token.pos_))

#         return self._deduplicate_and_filter_spans(caption, raw_spans)

#     def _extract_regex_spans(self, caption: str) -> List[CaptionSpan]:
#         raw_spans = []

#         prep_pattern = (
#             r"\b(in|on|under|over|near|behind|beside|with|without|around|"
#             r"through|across|inside|outside|next to|in front of)\b[^,.!?;:]*"
#         )

#         for match in re.finditer(prep_pattern, caption, flags=re.IGNORECASE):
#             raw_spans.append((match.start(), match.end(), "PP_REGEX"))

#         color_size_words = [
#             "red", "blue", "green", "yellow", "black", "white", "brown",
#             "gray", "grey", "small", "large", "big", "little", "tall",
#             "short", "old", "young", "bright", "dark", "cloudy", "sunny",
#             "orange", "faded", "visible", "horizontal", "vertical",
#             "partially", "heavily",
#         ]

#         for word in color_size_words:
#             for match in re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE):
#                 raw_spans.append((match.start(), match.end(), "ATTR_REGEX"))

#         return self._deduplicate_and_filter_spans(caption, raw_spans)

#     def _deduplicate_and_filter_spans(
#         self,
#         caption: str,
#         raw_spans: Sequence[Tuple[int, int, str]],
#     ) -> List[CaptionSpan]:

#         seen = set()
#         spans: List[CaptionSpan] = []
#         caption_len = len(caption)
#         caption_token_count = count_tokens(caption)

#         for start, end, label in raw_spans:
#             start = int(start)
#             end = int(end)

#             if start < 0 or end > caption_len or start >= end:
#                 continue

#             text = caption[start:end].strip()
#             text = text.strip(string.whitespace + string.punctuation)

#             if not text:
#                 continue

#             token_count = count_tokens(text)

#             if token_count >= caption_token_count:
#                 continue

#             if token_count > 12:
#                 continue

#             if len(content_words(text)) == 0 and token_count <= 1:
#                 continue

#             key = (start, end, label)
#             if key in seen:
#                 continue

#             seen.add(key)

#             spans.append(
#                 CaptionSpan(
#                     text=caption[start:end].strip(),
#                     start_char=start,
#                     end_char=end,
#                     label=label,
#                     token_count=token_count,
#                 )
#             )

#         spans.sort(key=lambda s: (-s.token_count, s.start_char))
#         return spans

#     # -----------------------------------------------------------------
#     # KG-to-caption alignment
#     # -----------------------------------------------------------------

#     def _is_low_risk_unaligned_candidate(self, span: CaptionSpan) -> bool:
#         """
#         Low-risk modifier spans can be tested even when they are not clearly
#         aligned to a KG node.

#         Do NOT include structural words like middle/right/side/view/angle here.
#         Those words often form sentence skeletons and caused outputs such as
#         "On of is ...".
#         """
#         text = span.text.strip()
#         text_lower = text.lower()

#         if not text_lower:
#             return False

#         if '"' in text or "'" in text:
#             return False

#         if span.token_count > 3:
#             return False

#         # Important color or visual attributes that are already aligned to KG
#         # should not be treated as low-risk.
#         visual_colors = {
#             "red", "orange", "white", "black", "yellow", "gray", "grey",
#             "blue", "green", "brown",
#         }
#         if content_words(text_lower) & visual_colors and span.aligned_preserved:
#             return False

#         low_risk_labels = {
#             "ADV",
#             "ADVP",
#             "ATTR_REGEX",
#         }

#         if span.label in low_risk_labels:
#             return True

#         # Adjectives are only low-risk if they are not key visual colors and
#         # not part of a protected visual noun phrase.
#         if span.label in {"ADJ", "ADJP"}:
#             words = content_words(text_lower)
#             blocked_words = {
#                 "red", "orange", "white", "black", "yellow", "gray", "grey",
#                 "horizontal", "vertical",
#             }
#             return not bool(words & blocked_words)

#         low_risk_words = {
#             "old", "faded", "partially", "visible", "heavily",
#         }

#         if span.token_count <= 2 and content_words(text_lower).issubset(low_risk_words):
#             return True

#         return False

#     def _is_protected_visual_span(self, span: CaptionSpan) -> bool:
#         """
#         Protect visually central noun chunks, quoted/OCR text, and KG-aligned
#         visual attributes.

#         This prevents over-compression such as deleting:
#             - "the orange spray painted word"
#             - "a partially visible and faded red line"
#             - "ROW"
#             - "FIRE LANE"
#             - "red" when it preserves a KG red marking
#         """
#         text = span.text.strip()
#         text_lower = text.lower()
#         words = content_words(text_lower)

#         if not text_lower:
#             return True

#         if '"' in text or "'" in text:
#             return True

#         if re.fullmatch(r"[A-Z0-9][A-Z0-9\s\-]{1,}", text.strip()):
#             return True

#         if re.search(r"\b(row|fire lane|fire|lane)\b", text_lower):
#             return True

#         visual_nouns = {
#             "word", "words", "letter", "letters", "text", "sign",
#             "marking", "markings", "line", "paint", "writing",
#             "corner", "street", "sidewalk", "curb", "road", "pavement",
#         }

#         visual_colors = {
#             "red", "orange", "white", "black", "yellow", "gray", "grey",
#             "blue", "green", "brown",
#         }

#         # Protect standalone KG-aligned colors and visual attributes.
#         if span.aligned_preserved and (words & visual_colors):
#             return True

#         # Protect important spatial relation phrases.
#         if text_lower in {"above it", "below it", "under it", "over it"}:
#             return True

#         if span.label == "NOUN_CHUNK" and words & visual_nouns:
#             return True

#         if span.label == "NOUN_CHUNK" and (words & visual_colors) and (words & visual_nouns):
#             return True

#         if span.label == "NOUN_CHUNK" and span.aligned_preserved and span.token_count >= 2:
#             return True

#         return False

#     def _is_candidate_deletion_grammar_safe(
#         self,
#         caption_before: str,
#         span: CaptionSpan,
#         caption_after: str,
#     ) -> bool:
#         """
#         Check whether deleting a span keeps the caption readable.

#         This is checked before running expensive FLUX reconstruction.
#         """
#         if not is_caption_text_valid(caption_after):
#             return False

#         quoted_ranges = get_quoted_ranges(caption_before)
#         grammar_anchor_ranges = get_grammar_anchor_ranges(caption_before)

#         # Never directly delete quoted/OCR text.
#         if span_overlaps_ranges(span.start_char, span.end_char, quoted_ranges):
#             return False

#         # Do not delete inside grammar anchors such as:
#         # "On the right side of the image" or "In the middle of the view".
#         if span_overlaps_ranges(span.start_char, span.end_char, grammar_anchor_ranges):
#             # Allow deleting only if the span is a very small non-structural modifier.
#             if not (span.label in {"ADV", "ADVP"} and span.token_count == 1):
#                 return False

#         if self._is_protected_visual_span(span):
#             return False

#         # Sentence-local safety.
#         sentence_start = max(
#             caption_before.rfind(".", 0, span.start_char),
#             caption_before.rfind("!", 0, span.start_char),
#             caption_before.rfind("?", 0, span.start_char),
#         ) + 1

#         sentence_end_candidates = [
#             pos for pos in [
#                 caption_before.find(".", span.end_char),
#                 caption_before.find("!", span.end_char),
#                 caption_before.find("?", span.end_char),
#             ]
#             if pos != -1
#         ]
#         sentence_end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(caption_before)

#         after_sentence = caption_after[sentence_start:sentence_end].strip()

#         broken_local_patterns = [
#             r"^(In|On|At|With|Of)\s+(is|are|was|were|with|of)\b",
#             r"\b(On|In|At|With)\s+of\s+(is|are|was|were)\b",
#             r"\b(is|are|was|were)\s+(a|an|the)?\s*(and|with|of)\b",
#             r"\ba\s+and\b",
#             r"\ban\s+and\b",
#         ]

#         for pattern in broken_local_patterns:
#             if re.search(pattern, after_sentence):
#                 return False

#         # Prevent deleting core structural spans from locative sentences.
#         lower_span = span.text.lower().strip()
#         structural_spans = {
#             "the right side", "right side", "the left side", "left side",
#             "the middle", "middle", "in the middle", "on the right side",
#             "on the left side", "above it",
#         }
#         if lower_span in structural_spans:
#             return False

#         return True

#     def _remove_multiple_spans(
#         self,
#         caption: str,
#         spans: Sequence[CaptionSpan],
#     ) -> str:
#         """
#         Remove multiple non-overlapping spans from a caption.
#         """
#         if not spans:
#             return caption

#         spans_sorted = sorted(spans, key=lambda s: s.start_char, reverse=True)
#         new_caption = caption

#         for span in spans_sorted:
#             new_caption = new_caption[: span.start_char] + " " + new_caption[span.end_char :]

#         return clean_caption_spacing(new_caption)

#     def _make_batch_deletion_candidate(
#         self,
#         caption: str,
#         candidate_spans: Sequence[CaptionSpan],
#         original_tokens: int,
#         max_batch_spans: int = 3,
#         max_batch_tokens: int = 6,
#     ) -> Optional[Tuple[CaptionSpan, str, int]]:
#         """
#         Build one batch-deletion candidate from low-risk spans.

#         This reduces runtime because we validate several safe deletions with
#         one FLUX+KG pass instead of validating every small deletion separately.
#         """
#         selected: List[CaptionSpan] = []
#         occupied: List[Tuple[int, int]] = []
#         deleted_tokens = 0

#         for span in candidate_spans:
#             if not self._is_low_risk_unaligned_candidate(span):
#                 continue

#             if self._is_protected_visual_span(span):
#                 continue

#             if span.aligned_missing or span.aligned_preserved:
#                 continue

#             if span_overlaps_ranges(span.start_char, span.end_char, get_quoted_ranges(caption)):
#                 continue

#             if span_overlaps_ranges(span.start_char, span.end_char, get_grammar_anchor_ranges(caption)):
#                 continue

#             structural_text = span.text.lower().strip()
#             if structural_text in {
#                 "the right side", "right side", "the left side", "left side",
#                 "the middle", "middle", "in the middle", "on the right side",
#                 "on the left side", "above it",
#             }:
#                 continue

#             overlaps_existing = any(
#                 span.start_char < end and span.end_char > start
#                 for start, end in occupied
#             )
#             if overlaps_existing:
#                 continue

#             candidate_deleted_tokens = deleted_tokens + span.token_count
#             if candidate_deleted_tokens > max_batch_tokens:
#                 continue

#             selected.append(span)
#             occupied.append((span.start_char, span.end_char))
#             deleted_tokens = candidate_deleted_tokens

#             if len(selected) >= max_batch_spans:
#                 break

#         if not selected:
#             return None

#         batch_caption = self._remove_multiple_spans(caption, selected)

#         before_tokens = count_tokens(caption)
#         after_tokens = count_tokens(batch_caption)
#         token_reduction = before_tokens - after_tokens

#         if token_reduction <= 1:
#             return None

#         candidate_cr = 1.0 - (after_tokens / max(original_tokens, 1))
#         if candidate_cr > self.max_compression_ratio:
#             return None

#         deleted_text = " + ".join([s.text for s in selected])

#         batch_span = CaptionSpan(
#             text=deleted_text,
#             start_char=min(s.start_char for s in selected),
#             end_char=max(s.end_char for s in selected),
#             label="BATCH_LOW_RISK",
#             token_count=token_reduction,
#             aligned_preserved=[],
#             aligned_missing=[],
#             aligned_extra=[],
#         )

#         if not self._is_candidate_deletion_grammar_safe(caption, batch_span, batch_caption):
#             return None

#         return batch_span, batch_caption, token_reduction

#     def _build_protected_and_candidate_spans(
#         self,
#         spans: List[CaptionSpan],
#         preserved: Set[Tuple[str, ...]],
#         missing: Set[Tuple[str, ...]],
#         extra: Set[Tuple[str, ...]],
#         caption: str = "",
#     ) -> Tuple[List[CaptionSpan], List[CaptionSpan]]:

#         protected_spans: List[CaptionSpan] = []
#         candidate_spans: List[CaptionSpan] = []
#         grammar_anchor_ranges = get_grammar_anchor_ranges(caption) if caption else []

#         for span in spans:
#             span.aligned_missing = [
#                 tuple_to_text(t)
#                 for t in missing
#                 if span_aligns_with_tuple(span.text, t)
#             ]

#             span.aligned_preserved = [
#                 tuple_to_text(t)
#                 for t in preserved
#                 if span_aligns_with_tuple(span.text, t)
#             ]

#             span.aligned_extra = [
#                 tuple_to_text(t)
#                 for t in extra
#                 if span_aligns_with_tuple(span.text, t)
#             ]

#             # Protect grammar anchors such as "On the right side of the image"
#             # and "In the middle of the view".
#             if grammar_anchor_ranges and span_overlaps_ranges(
#                 span.start_char,
#                 span.end_char,
#                 grammar_anchor_ranges,
#             ):
#                 protected_spans.append(span)
#                 continue

#             # Protect missing visual information because the reconstruction
#             # already struggles with it.
#             if span.aligned_missing:
#                 protected_spans.append(span)
#                 continue

#             # Protect central visual noun chunks and quoted/OCR text.
#             if self._is_protected_visual_span(span):
#                 protected_spans.append(span)
#                 continue

#             # Extra-aligned spans may reduce hallucinated content.
#             if span.aligned_extra:
#                 candidate_spans.append(span)
#                 continue

#             # Low-risk modifier deletion.
#             if self._is_low_risk_unaligned_candidate(span):
#                 candidate_spans.append(span)
#                 continue

#             # Preserved spans are only candidates if they are not protected
#             # visual noun chunks. Acceptance will still reject them if KG drops.
#             if span.aligned_preserved:
#                 candidate_spans.append(span)
#                 continue

#             if self.allow_unaligned_candidates:
#                 candidate_spans.append(span)

#         def candidate_priority(span: CaptionSpan):
#             aligns_extra = len(span.aligned_extra) > 0
#             aligns_preserved = len(span.aligned_preserved) > 0
#             low_risk_unaligned = (
#                 not aligns_extra
#                 and not aligns_preserved
#                 and self._is_low_risk_unaligned_candidate(span)
#             )

#             if aligns_extra:
#                 group = 0
#             elif low_risk_unaligned:
#                 group = 1
#             elif aligns_preserved:
#                 group = 2
#             else:
#                 group = 3

#             return (
#                 group,
#                 -span.token_count,
#                 span.start_char,
#             )

#         candidate_spans.sort(key=candidate_priority)

#         return protected_spans, candidate_spans

#     # -----------------------------------------------------------------
#     # Candidate acceptance constraints
#     # -----------------------------------------------------------------

#     def _get_span_aligned_preserved_tuples(
#         self,
#         span: CaptionSpan,
#         preserved_tuples: Set[Tuple[str, ...]],
#     ) -> Set[Tuple[str, ...]]:
#         return {
#             t for t in preserved_tuples
#             if span_aligns_with_tuple(span.text, t)
#         }

#     def _candidate_acceptance_decision(
#         self,
#         span: CaptionSpan,
#         current_D: float,
#         candidate_D: float,
#         threshold: float,
#         current_comparison: Dict[str, Any],
#         candidate_comparison: Dict[str, Any],
#         baseline_comparison: Optional[Dict[str, Any]] = None,
#     ) -> Tuple[bool, str, Dict[str, Any]]:
#         """
#         Decide whether a candidate deletion is acceptable.

#         This version is compression-first under acceptable fidelity.

#         Important:
#             We do NOT reject a candidate simply because it is slightly worse
#             than the current caption. That was fidelity-first behavior and
#             caused very small compression.

#         A deletion is accepted if:
#             1. candidate_D <= global threshold
#             2. candidate KG quality stays above baseline-derived floors
#             3. KG nodes directly tied to the deleted span are not lost
#         """

#         if baseline_comparison is None:
#             baseline_comparison = current_comparison

#         current_preserved = set(current_comparison.get("preserved", set()))
#         candidate_preserved = set(candidate_comparison.get("preserved", set()))

#         critical_tuples = self._get_span_aligned_preserved_tuples(
#             span=span,
#             preserved_tuples=current_preserved,
#         )

#         lost_critical_tuples = critical_tuples - candidate_preserved

#         baseline_f1 = float(baseline_comparison.get("f1_mean", 0.0))
#         baseline_obj_recall = float(baseline_comparison.get("object_recall_mean", 0.0))
#         baseline_attr_recall = float(baseline_comparison.get("attribute_recall_mean", 0.0))

#         current_f1 = float(current_comparison.get("f1_mean", 0.0))
#         current_obj_recall = float(current_comparison.get("object_recall_mean", 0.0))
#         current_attr_recall = float(current_comparison.get("attribute_recall_mean", 0.0))

#         candidate_f1 = float(candidate_comparison.get("f1_mean", 0.0))
#         candidate_obj_recall = float(candidate_comparison.get("object_recall_mean", 0.0))
#         candidate_attr_recall = float(candidate_comparison.get("attribute_recall_mean", 0.0))

#         # Baseline-derived acceptable floors.
#         # These allow compression even if a candidate is not better than the
#         # current caption, as long as the final reconstruction remains acceptable.
#         min_f1_allowed = max(self.min_baseline_kg_f1, baseline_f1 - 0.08)
#         min_obj_recall_allowed = max(self.min_baseline_object_recall, baseline_obj_recall - 0.05)
#         min_attr_recall_allowed = max(0.10, baseline_attr_recall - 0.12)

#         details = {
#             "current_D": current_D,
#             "candidate_D": candidate_D,
#             "threshold": threshold,

#             "baseline_f1": baseline_f1,
#             "current_f1": current_f1,
#             "candidate_f1": candidate_f1,
#             "min_f1_allowed": min_f1_allowed,

#             "baseline_object_recall": baseline_obj_recall,
#             "current_object_recall": current_obj_recall,
#             "candidate_object_recall": candidate_obj_recall,
#             "min_object_recall_allowed": min_obj_recall_allowed,

#             "baseline_attribute_recall": baseline_attr_recall,
#             "current_attribute_recall": current_attr_recall,
#             "candidate_attribute_recall": candidate_attr_recall,
#             "min_attribute_recall_allowed": min_attr_recall_allowed,

#             "critical_tuples": [
#                 tuple_to_text(t) for t in sorted(critical_tuples, key=str)
#             ],
#             "lost_critical_tuples": [
#                 tuple_to_text(t) for t in sorted(lost_critical_tuples, key=str)
#             ],
#         }

#         # 1. Global acceptable reconstruction fidelity.
#         if candidate_D > threshold:
#             return False, "rejected_above_global_threshold", details

#         # 2. KG fidelity floors relative to baseline full-caption reconstruction.
#         if candidate_f1 < min_f1_allowed:
#             return False, "rejected_below_acceptable_kg_f1", details

#         if candidate_obj_recall < min_obj_recall_allowed:
#             return False, "rejected_below_acceptable_object_recall", details

#         if candidate_attr_recall < min_attr_recall_allowed:
#             return False, "rejected_below_acceptable_attribute_recall", details

#         # 3. Do not remove KG elements directly tied to the deleted span.
#         # Example: deleting "FIRE LANE" should not be accepted if
#         # "paint marking has fire lane" disappears.
#         if lost_critical_tuples:
#             return False, "rejected_lost_deleted_span_kg_nodes", details

#         return True, "accepted_under_acceptable_fidelity", details

#     # -----------------------------------------------------------------
#     # Saving and debug utilities
#     # -----------------------------------------------------------------

#     def _make_rejected_step(
#         self,
#         iteration: int,
#         span: CaptionSpan,
#         caption_before: str,
#         caption_after: str,
#         tokens_before: int,
#         tokens_after: int,
#         token_reduction: int,
#         threshold: float,
#         reason: str,
#         time_seconds: float,
#     ) -> CompressionStep:

#         return CompressionStep(
#             iteration=iteration,
#             deleted_span=span.text,
#             deleted_label=span.label,
#             caption_before=caption_before,
#             caption_after=caption_after,
#             tokens_before=tokens_before,
#             tokens_after=tokens_after,
#             token_reduction=token_reduction,
#             distortion=float("inf"),
#             threshold=threshold,
#             passed_threshold=False,
#             selected=False,
#             reason=reason,
#             distortion_details={},
#             aligned_preserved=span.aligned_preserved,
#             aligned_missing=span.aligned_missing,
#             time_seconds=time_seconds,
#         )

#     def _save_kg_debug(
#         self,
#         kg: Dict[str, Any],
#         image_id: str,
#         name: str,
#     ) -> None:

#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)

#         path = debug_dir / f"{safe_filename(name)}.json"

#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(
#                 make_json_serializable(kg_to_readable_dict(kg)),
#                 f,
#                 indent=2,
#                 ensure_ascii=False,
#             )

#         print(f"[KG Saved] {path}")

#     def _save_kg_comparison_debug(
#         self,
#         comparison: Dict[str, Any],
#         image_id: str,
#         name: str,
#     ) -> None:

#         debug_dir = self.kg_debug_dir / safe_filename(image_id)
#         debug_dir.mkdir(parents=True, exist_ok=True)

#         path = debug_dir / f"{safe_filename(name)}.json"

#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(
#                 make_json_serializable(comparison),
#                 f,
#                 indent=2,
#                 ensure_ascii=False,
#             )

#         print(f"[KG Comparison Saved] {path}")

#     def _save_final_kgs_alias(
#         self,
#         kgs: List[Dict[str, Any]],
#         image_id: str,
#     ) -> None:

#         for idx, kg in enumerate(kgs):
#             seed = self.t2i_seeds[idx] if idx < len(self.t2i_seeds) else idx
#             self._save_kg_debug(
#                 kg=kg,
#                 image_id=image_id,
#                 name=f"final_reconstructed_kg_seed{seed}",
#             )

#     def _save_result(self, result: CompressionResult) -> None:
#         path = self.result_dir / f"{safe_filename(result.image_id)}.json"

#         data = make_json_serializable(result)

#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

#         print(f"[Saved] {path}")

#     def _save_summary(self, results: List[CompressionResult]) -> None:
#         if not results:
#             print("[Summary] No successful results to summarize.")
#             return

#         compression_ratios = [r.compression_ratio for r in results]
#         final_distortions = [r.final_distortion for r in results]
#         baseline_distortions = [r.baseline_distortion for r in results]
#         original_tokens = [r.original_tokens for r in results]
#         compressed_tokens = [r.compressed_tokens for r in results]
#         times = [r.total_time for r in results]

#         statuses = {}
#         for r in results:
#             statuses[r.status] = statuses.get(r.status, 0) + 1

#         summary = {
#             "num_samples": len(results),
#             "status_counts": statuses,

#             "avg_compression_ratio": float(np.mean(compression_ratios)),
#             "std_compression_ratio": float(np.std(compression_ratios)),

#             "avg_baseline_distortion": float(np.mean(baseline_distortions)),
#             "std_baseline_distortion": float(np.std(baseline_distortions)),

#             "avg_final_distortion": float(np.mean(final_distortions)),
#             "std_final_distortion": float(np.std(final_distortions)),

#             "avg_original_tokens": float(np.mean(original_tokens)),
#             "avg_compressed_tokens": float(np.mean(compressed_tokens)),

#             "avg_time_seconds": float(np.mean(times)),
#             "total_time_hours": float(np.sum(times) / 3600.0),

#             "use_relative_threshold": self.use_relative_threshold,
#             "epsilon": self.epsilon,
#             "absolute_threshold": self.absolute_threshold,
#             "num_seeds": len(self.t2i_seeds),
#             "seeds": self.t2i_seeds,
#             "min_baseline_kg_f1": self.min_baseline_kg_f1,
#             "min_baseline_object_recall": self.min_baseline_object_recall,
#             "target_compression_ratio": self.target_compression_ratio,
#             "max_compression_ratio": self.max_compression_ratio,
#             "stop_when_target_reached": self.stop_when_target_reached,
#         }

#         summary_path = self.result_dir / "summary.json"
#         with open(summary_path, "w", encoding="utf-8") as f:
#             json.dump(make_json_serializable(summary), f, indent=2)

#         print("\n" + "=" * 70)
#         print(f"SUMMARY over {len(results)} samples")
#         print(f"Status counts:          {statuses}")
#         print(f"Avg compression ratio:  {summary['avg_compression_ratio']:.2%}")
#         print(f"Avg baseline D:         {summary['avg_baseline_distortion']:.4f}")
#         print(f"Avg final D:            {summary['avg_final_distortion']:.4f}")
#         print(
#             f"Avg tokens:             "
#             f"{summary['avg_original_tokens']:.1f} -> "
#             f"{summary['avg_compressed_tokens']:.1f}"
#         )
#         print(f"Total time:             {summary['total_time_hours']:.2f} hours")
#         print(f"Saved summary:          {summary_path}")
#         print("=" * 70)