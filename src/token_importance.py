import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_CLIP_EOT_ID = 49407  # open_clip / CLIP end-of-text token id


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[TokenImportance] CUDA requested but unavailable. Using CPU.")
            return "cpu"
        if ":" in device:
            gpu_id = int(device.split(":")[-1])
            if gpu_id >= torch.cuda.device_count():
                print(f"[TokenImportance] {device} unavailable; using cuda:0.")
                return "cuda:0"
    return device


def ensure_rgb(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected PIL.Image.Image.")
    return image.convert("RGB") if image.mode != "RGB" else image


def clean_caption_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def normalize_scores(scores: Sequence[float]) -> List[float]:
    if len(scores) == 0:
        return []
    arr = np.array(scores, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-8:
        return [0.0 for _ in arr]
    return [float(x) for x in (arr - lo) / (hi - lo)]


def remove_word_by_index(words: List[str], index: int) -> str:
    new_words = words[:index] + words[index + 1:]
    return clean_caption_spacing(" ".join(new_words))


def remove_char_span(text: str, start_char: int, end_char: int) -> str:
    return clean_caption_spacing(text[:start_char] + " " + text[end_char:])


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class ImportanceResult:
    words: List[str]
    scores: List[float]
    raw_scores: List[float]
    full_similarity: float
    method: str


@dataclass
class SpanImportance:
    text: str
    start_char: int
    end_char: int
    label: str
    score: float
    raw_score: float


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------

class CLIPTokenImportance:
    """
    CLIP leave-one-out token/span importance (ablation tool).

    Importance(word) = max(0, sim(image, window) - sim(image, window_without_word))
    computed within the <=context-length window containing the word.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "auto",
        clip_model=None,
        clip_preprocess=None,
        clip_tokenizer=None,
        context_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)

        if clip_model is not None and clip_preprocess is not None and clip_tokenizer is not None:
            # Share an already-loaded CLIP (e.g. distortion.py's instance).
            self.model = clip_model
            self.preprocess = clip_preprocess
            self.tokenizer = clip_tokenizer
            print("[TokenImportance] Using shared CLIP instance.")
        else:
            import open_clip
            print(f"[TokenImportance] Loading CLIP '{model_name}/{pretrained}' on '{self.device}'")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            print("[TokenImportance] CLIP loaded.")

        self.context_length = int(
            context_length or getattr(self.model, "context_length", 77) or 77
        )

    @classmethod
    def from_shared(cls, clip_model, clip_preprocess, clip_tokenizer, device="auto",
                    context_length: Optional[int] = None) -> "CLIPTokenImportance":
        return cls(device=device, clip_model=clip_model, clip_preprocess=clip_preprocess,
                   clip_tokenizer=clip_tokenizer, context_length=context_length)

    # ----------------------------------------------------------------- #
    # Encoding
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image = ensure_rgb(image)
        t = self.preprocess(image).unsqueeze(0).to(self.device)
        return F.normalize(self.model.encode_image(t), dim=-1)

    @torch.no_grad()
    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer([text.strip()]).to(self.device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        return F.normalize(self.model.encode_text(self._tokenize(text)), dim=-1)

    @torch.no_grad()
    def clip_similarity(self, image_features: torch.Tensor, caption: str) -> float:
        tf = self.encode_text(caption)
        return float(np.clip((image_features * tf).sum(dim=-1).item(), -1.0, 1.0))

    def _fits(self, text: str) -> bool:
        """True if text fits within the CLIP context (EOT token survives)."""
        toks = self._tokenize(text)[0]
        return bool((toks == _CLIP_EOT_ID).any().item())

    def _windows(self, words: List[str]) -> List[Tuple[int, int]]:
        """Greedily pack words into <=context-length windows; return (start, end) word indices."""
        windows: List[Tuple[int, int]] = []
        start = 0
        i = 0
        cur: List[str] = []
        while i < len(words):
            trial = cur + [words[i]]
            if self._fits(" ".join(trial)):
                cur = trial
                i += 1
            else:
                if cur:
                    windows.append((start, i))  # [start, i)
                    cur = []
                    start = i
                else:
                    # Single word longer than context (rare) — emit it alone.
                    windows.append((i, i + 1))
                    i += 1
                    start = i
        if cur:
            windows.append((start, len(words)))
        return windows

    # ----------------------------------------------------------------- #
    # Word importance (chunked leave-one-out)
    # ----------------------------------------------------------------- #
    def compute_importance(
        self, image: Image.Image, caption: str,
        normalize: bool = True, max_words: Optional[int] = None,
    ) -> Dict[str, Any]:
        caption = clean_caption_spacing(caption)
        words = caption.split()
        if not words:
            return {"words": [], "scores": [], "raw_scores": [],
                    "full_similarity": 0.0, "method": "leave_one_out_chunked"}

        image_features = self.encode_image(image)
        windows = self._windows(words)

        raw_scores = [0.0] * len(words)
        weighted_full = 0.0
        total_words = 0
        scored_cap = len(words) if max_words is None else min(len(words), int(max_words))

        for (ws, we) in windows:
            window_words = words[ws:we]
            window_text = " ".join(window_words)
            full_sim = self.clip_similarity(image_features, window_text)
            weighted_full += full_sim * len(window_words)
            total_words += len(window_words)
            for local_i, _w in enumerate(window_words):
                gi = ws + local_i
                if gi >= scored_cap:
                    continue
                reduced = remove_word_by_index(window_words, local_i)
                if not reduced:
                    raw_scores[gi] = max(0.0, full_sim)
                    continue
                reduced_sim = self.clip_similarity(image_features, reduced)
                raw_scores[gi] = float(max(0.0, full_sim - reduced_sim))

        full_similarity = (weighted_full / total_words) if total_words else 0.0
        scores = normalize_scores(raw_scores) if normalize else [float(x) for x in raw_scores]
        return {
            "words": words, "scores": scores,
            "raw_scores": [float(x) for x in raw_scores],
            "full_similarity": float(full_similarity),
            "method": "leave_one_out_chunked",
        }

    # ----------------------------------------------------------------- #
    # Span importance (derived from word importance — consistent & cheap)
    # ----------------------------------------------------------------- #
    def compute_span_importance(
        self, image: Image.Image, caption: str,
        spans: Sequence[Any], normalize: bool = True,
    ) -> List[SpanImportance]:
        caption = clean_caption_spacing(caption)
        word_imp = self.compute_importance(image, caption, normalize=False)
        raw = word_imp["raw_scores"]

        def _word_idx_range(s: int, e: int) -> Tuple[Optional[int], Optional[int]]:
            sw = ew = None
            for idx, m in enumerate(re.finditer(r"\S+", caption)):
                if m.end() > s and m.start() < e:
                    if sw is None:
                        sw = idx
                    ew = idx
            return sw, ew

        results: List[SpanImportance] = []
        raw_scores: List[float] = []
        for span in spans:
            text, s, e, label = self._parse_span(span, caption)
            sw, ew = _word_idx_range(s, e)
            if sw is None:
                rs = 0.0
            else:
                vals = raw[sw:ew + 1]
                rs = float(sum(vals) / max(len(vals), 1))
            raw_scores.append(rs)
            results.append(SpanImportance(text, s, e, label, 0.0, rs))

        norm = normalize_scores(raw_scores) if normalize else raw_scores
        for item, sc in zip(results, norm):
            item.score = float(sc)
        results.sort(key=lambda x: (x.score, -(x.end_char - x.start_char), x.start_char))
        return results

    def compute_importance_gradient(self, image: Image.Image, caption: str) -> Dict[str, Any]:
        print("[TokenImportance] gradient attribution mapped to chunked leave-one-out.")
        return self.compute_importance(image=image, caption=caption)

    def _parse_span(self, span: Any, caption: str) -> Tuple[str, int, int, str]:
        if isinstance(span, dict):
            s, e = int(span["start_char"]), int(span["end_char"])
            return str(span.get("text", caption[s:e])), s, e, str(span.get("label", "SPAN"))
        if hasattr(span, "start_char") and hasattr(span, "end_char"):
            s, e = int(span.start_char), int(span.end_char)
            return str(getattr(span, "text", caption[s:e])), s, e, str(getattr(span, "label", "SPAN"))
        if isinstance(span, tuple) and len(span) >= 2:
            s, e = int(span[0]), int(span[1])
            label = str(span[2]) if len(span) >= 3 else "SPAN"
            return caption[s:e], s, e, label
        raise TypeError(f"Unsupported span format: {type(span)}")


# ---------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------

def rank_words_by_importance(importance_result: Dict[str, Any], ascending: bool = True):
    words = importance_result.get("words", [])
    scores = importance_result.get("scores", [])
    ranked = [(i, w, float(s)) for i, (w, s) in enumerate(zip(words, scores))]
    ranked.sort(key=lambda x: x[2], reverse=not ascending)
    return ranked


def rank_spans_by_importance(span_results: Sequence[SpanImportance], ascending: bool = True):
    return sorted(list(span_results), key=lambda x: x.score, reverse=not ascending)