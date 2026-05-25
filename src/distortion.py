from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import lpips
import open_clip


# =====================================================================
# Device helper
# =====================================================================

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[Distortion] CUDA requested but not available. Using CPU.")
            return "cpu"
        if ":" in device:
            gpu_id = int(device.split(":")[-1])
            if gpu_id >= torch.cuda.device_count():
                print(f"[Distortion] {device} unavailable; using cuda:0.")
                return "cuda:0"
    return device


# =====================================================================
# DistortionMetric
# =====================================================================

class DistortionMetric:
    """Combined distortion metric for KG-RD (see module docstring)."""

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.2,
        gamma: float = 0.4,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        dino_model_name: str = "dinov2_vitb14",
        lpips_net: str = "alex",
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        device: str = "auto",
        kg_object_weight: float = 0.4,
        kg_attribute_weight: float = 0.3,
        kg_relation_weight: float = 0.3,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.device = resolve_device(device)

        total_kg = kg_object_weight + kg_attribute_weight + kg_relation_weight
        if total_kg <= 0:
            raise ValueError("KG sub-weights must sum to a positive number.")
        self.kg_obj_w = float(kg_object_weight) / total_kg
        self.kg_attr_w = float(kg_attribute_weight) / total_kg
        self.kg_rel_w = float(kg_relation_weight) / total_kg

        self.bertscore_model = bertscore_model

        print(f"[Distortion] Weights: alpha_img={self.alpha}, beta_cap={self.beta}, gamma_kg={self.gamma}")
        print(f"[Distortion] KG sub-weights: obj={self.kg_obj_w:.3f}, attr={self.kg_attr_w:.3f}, rel={self.kg_rel_w:.3f}")

        # CLIP (image encoder + text encoder for CLIPScore). Exposed for sharing.
        print("[Distortion] Loading CLIP model...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.clip_model = self.clip_model.to(self.device).eval()

        # DINOv2
        print(f"[Distortion] Loading DINOv2 ({dino_model_name})...")
        self.dino_model = torch.hub.load("facebookresearch/dinov2", dino_model_name).to(self.device).eval()
        self.dino_transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # LPIPS
        print("[Distortion] Loading LPIPS model...")
        self.lpips_model = lpips.LPIPS(net=lpips_net).to(self.device).eval()
        self.lpips_transform = T.Compose([
            T.Resize((256, 256)), T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # BERTScore (lazy)
        self._bertscore_scorer = None
        print("[Distortion] All metrics loaded successfully.")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DistortionMetric":
        """Build from the project YAML ``distortion`` block."""
        d = config["distortion"]
        w = d["weights"]
        clip = d.get("clip", {})
        dino = d.get("dino", {})
        lp = d.get("lpips", {})
        bs = d.get("bertscore", {})
        kg = d.get("kg", {})
        return cls(
            alpha=w["alpha_img"], beta=w["beta_cap"], gamma=w["gamma_kg"],
            clip_model_name=clip.get("model", "ViT-L-14"),
            clip_pretrained=clip.get("pretrained", "openai"),
            dino_model_name=dino.get("model", "dinov2_vitb14"),
            lpips_net=lp.get("net", "alex"),
            bertscore_model=bs.get("model", "microsoft/deberta-xlarge-mnli"),
            device=clip.get("device", "cuda:0"),
            kg_object_weight=kg.get("object_weight", 0.4),
            kg_attribute_weight=kg.get("attribute_weight", 0.3),
            kg_relation_weight=kg.get("relation_weight", 0.3),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image.Image.")
        return image.convert("RGB") if image.mode != "RGB" else image

    def _get_bertscore(self):
        if self._bertscore_scorer is None:
            try:
                from bert_score import BERTScorer
                self._bertscore_scorer = BERTScorer(
                    model_type=self.bertscore_model, lang="en",
                    rescale_with_baseline=True, device=self.device,
                )
                print(f"[Distortion] BERTScore loaded (model={self.bertscore_model}).")
            except Exception as exc:
                print(f"[Distortion] BERTScore unavailable ({exc}); using 0.0 for BERT term.")
                self._bertscore_scorer = False
        return self._bertscore_scorer

    # ================================================================== #
    # Q_img components
    # ================================================================== #
    @torch.no_grad()
    def compute_clip_image_similarity(self, image_orig, image_recon) -> float:
        i1 = self.clip_preprocess(self._ensure_rgb(image_orig)).unsqueeze(0).to(self.device)
        i2 = self.clip_preprocess(self._ensure_rgb(image_recon)).unsqueeze(0).to(self.device)
        f1 = F.normalize(self.clip_model.encode_image(i1), dim=-1)
        f2 = F.normalize(self.clip_model.encode_image(i2), dim=-1)
        return float(np.clip(float((f1 * f2).sum().item()), 0.0, 1.0))

    @torch.no_grad()
    def compute_dino_similarity(self, image_orig, image_recon) -> float:
        t1 = self.dino_transform(self._ensure_rgb(image_orig)).unsqueeze(0).to(self.device)
        t2 = self.dino_transform(self._ensure_rgb(image_recon)).unsqueeze(0).to(self.device)
        f1 = F.normalize(self.dino_model(t1), dim=-1)
        f2 = F.normalize(self.dino_model(t2), dim=-1)
        return float(np.clip(float((f1 * f2).sum().item()), 0.0, 1.0))

    @torch.no_grad()
    def compute_lpips_distance(self, image_orig, image_recon) -> float:
        i1 = self.lpips_transform(self._ensure_rgb(image_orig)).unsqueeze(0).to(self.device)
        i2 = self.lpips_transform(self._ensure_rgb(image_recon)).unsqueeze(0).to(self.device)
        return float(self.lpips_model(i1, i2).item())

    def compute_q_img(self, image_orig, image_recon) -> Dict[str, float]:
        """Q_img = (CLIP_sim + DINO_sim + (1 - LPIPS)) / 3, all in [0,1]."""
        clip_sim = self.compute_clip_image_similarity(image_orig, image_recon)
        dino_sim = self.compute_dino_similarity(image_orig, image_recon)
        lpips_d = self.compute_lpips_distance(image_orig, image_recon)
        lpips_sim = float(np.clip(1.0 - lpips_d, 0.0, 1.0))
        q_img = (clip_sim + dino_sim + lpips_sim) / 3.0
        return {"Q_img": float(q_img), "CLIP_sim": float(clip_sim), "DINO_sim": float(dino_sim),
                "LPIPS": float(lpips_d), "LPIPS_sim": float(lpips_sim)}

    # ================================================================== #
    # Q_cap components
    # ================================================================== #
    def compute_bertscore(self, caption_ref: str, caption_hyp: str) -> float:
        scorer = self._get_bertscore()
        if scorer is False:
            return 0.0
        try:
            _, _, Fs = scorer.score([caption_hyp], [caption_ref])
            return float(np.clip(float(Fs[0].item()), 0.0, 1.0))
        except Exception as exc:
            print(f"[Distortion] BERTScore failed: {exc}")
            return 0.0

    @torch.no_grad()
    def compute_clipscore(self, caption: str, image: Image.Image) -> float:
        try:
            tokens = self.clip_tokenizer([caption]).to(self.device)  # truncates at 77
            img_t = self.clip_preprocess(self._ensure_rgb(image)).unsqueeze(0).to(self.device)
            t_feat = F.normalize(self.clip_model.encode_text(tokens), dim=-1)
            i_feat = F.normalize(self.clip_model.encode_image(img_t), dim=-1)
            return float(np.clip(float((t_feat * i_feat).sum().item()), 0.0, 1.0))
        except Exception as exc:
            print(f"[Distortion] CLIPScore failed: {exc}")
            return 0.0

    def compute_q_cap(self, caption_orig, caption_compressed, image_orig) -> Dict[str, float]:
        """Q_cap = 0.5*BERTScore(c, c'_k) + 0.5*CLIPScore(c'_k, I)."""
        bert = self.compute_bertscore(caption_orig, caption_compressed)
        clips = self.compute_clipscore(caption_compressed, image_orig)
        return {"Q_cap": float(0.5 * bert + 0.5 * clips), "BERTScore": float(bert), "CLIPScore": float(clips)}

    # ================================================================== #
    # Q_KG
    # ================================================================== #
    def compute_q_kg(self, kg_original, kg_reconstructed, use_soft: bool = True) -> Dict[str, float]:
        """
        Q_KG = obj_w*R_obj + attr_w*R_attr + rel_w*R_rel.

        kg_original      : KG from the original image (ground truth).
        kg_reconstructed : KG from the reconstruction (baseline I_c, or candidate I_{c'_k}).
        use_soft         : soft (mpnet) recall by default; hard set-recall if False.
        """
        from .scene_graph import compare_scene_graphs
        comp = compare_scene_graphs(kg_original=kg_original, kg_reconstructed=kg_reconstructed)

        if use_soft:
            r_obj = float(comp.get("object_recall_soft", 0.0))
            r_attr = float(comp.get("attribute_recall_soft", 0.0))
            r_rel = float(comp.get("relation_recall_soft", 0.0))
            label = "soft"
        else:
            r_obj = float(comp.get("object_recall", 0.0))
            r_attr = float(comp.get("attribute_recall", 0.0))
            r_rel = float(comp.get("relation_recall", 0.0))
            label = "hard"

        q_kg = self.kg_obj_w * r_obj + self.kg_attr_w * r_attr + self.kg_rel_w * r_rel
        return {
            "Q_KG": float(q_kg), "Q_KG_matching": label,
            "R_obj": r_obj, "R_attr": r_attr, "R_rel": r_rel,
            "KG_F1_hard": float(comp.get("f1", 0.0)),
            "weighted_KG_F1_hard": float(comp.get("weighted_kg_f1_hard", 0.0)),
            "weighted_KG_F1_soft": float(comp.get("weighted_kg_f1_soft", 0.0)),
            "num_preserved": int(comp.get("num_preserved", 0)),
            "num_missing": int(comp.get("num_missing", 0)),
            "num_extra": int(comp.get("num_extra", 0)),
            "chair_i": float(comp.get("chair_i", 0.0)),
            "_comparison": comp,
        }

    # ================================================================== #
    # Combined distortion
    # ================================================================== #
    def compute_candidate_distortion(
        self,
        image_orig: Image.Image,
        image_recon: Image.Image,
        caption_orig: str,
        caption_compressed: str,
        kg_image_orig: Dict[str, Any],
        kg_image_recon: Optional[Dict[str, Any]] = None,
        use_soft_kg: bool = True,
        *,
        kg_caption_compressed: Optional[Dict[str, Any]] = None,  # deprecated alias
    ) -> Dict[str, float]:
        """
        Distortion for a compressed candidate c'_k.

        Q_img : I vs I_{c'_k}            (CLIP + DINO + LPIPS)
        Q_cap : c vs c'_k, with I        (BERTScore + CLIPScore)
        Q_KG  : KG_I vs KG_{I_{c'_k}}    (KG from the RECONSTRUCTED image)

        D(c'_k) = alpha*(1-Q_img) + beta*(1-Q_cap) + gamma*(1-Q_KG)

        ``kg_image_recon`` is the KG extracted from G(c'_k). The legacy keyword
        ``kg_caption_compressed`` is accepted as a deprecated alias.
        """
        if kg_image_recon is None:
            kg_image_recon = kg_caption_compressed
        if kg_image_recon is None:
            raise ValueError(
                "compute_candidate_distortion needs kg_image_recon — the KG extracted "
                "from the candidate reconstruction G(c'_k)."
            )

        img_scores = self.compute_q_img(image_orig, image_recon)
        cap_scores = self.compute_q_cap(caption_orig, caption_compressed, image_orig)
        kg_scores = self.compute_q_kg(kg_image_orig, kg_image_recon, use_soft=use_soft_kg)

        q_img, q_cap, q_kg = img_scores["Q_img"], cap_scores["Q_cap"], kg_scores["Q_KG"]
        d_total = self.alpha * (1.0 - q_img) + self.beta * (1.0 - q_cap) + self.gamma * (1.0 - q_kg)

        result: Dict[str, float] = {"D": float(d_total), "Q_img": float(q_img),
                                    "Q_cap": float(q_cap), "Q_KG": float(q_kg)}
        result.update(img_scores)
        result.update(cap_scores)
        result.update({k: v for k, v in kg_scores.items() if k != "_comparison"})
        result["_kg_comparison"] = kg_scores.get("_comparison", {})
        return result

    def compute_baseline_distortion(
        self, image_orig, image_recon, kg_image_orig, kg_image_recon, use_soft_kg: bool = True,
    ) -> Dict[str, float]:
        """
        Baseline D_0 for the full caption c.

        Q_img_base : I vs G(c);  Q_KG_base : KG_I vs KG_{G(c)}.
        D_0 = alpha*(1-Q_img_base) + gamma*(1-Q_KG_base)   (no beta: Q_cap(c,c)=1).
        """
        img_scores = self.compute_q_img(image_orig, image_recon)
        kg_scores = self.compute_q_kg(kg_image_orig, kg_image_recon, use_soft=use_soft_kg)
        q_img_base, q_kg_base = img_scores["Q_img"], kg_scores["Q_KG"]
        d0 = self.alpha * (1.0 - q_img_base) + self.gamma * (1.0 - q_kg_base)

        result: Dict[str, float] = {"D": float(d0), "Q_img_base": float(q_img_base), "Q_KG_base": float(q_kg_base)}
        result.update({f"base_{k}": v for k, v in img_scores.items()})
        result.update({f"base_{k}": v for k, v in kg_scores.items() if k != "_comparison"})
        result["_base_kg_comparison"] = kg_scores.get("_comparison", {})
        return result

    # ================================================================== #
    # Averaging across seeds (baseline)
    # ================================================================== #
    def compute_averaged_distortion(
        self, image_orig, images_recon: List[Image.Image],
        kg_original: Dict[str, Any], kgs_reconstructed: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Average baseline distortion over multiple reconstruction seeds."""
        if not images_recon:
            raise ValueError("images_recon cannot be empty.")
        if len(images_recon) != len(kgs_reconstructed):
            raise ValueError("images_recon and kgs_reconstructed must have the same length.")

        all_results = [
            self.compute_baseline_distortion(image_orig, img_r, kg_original, kg_r)
            for img_r, kg_r in zip(images_recon, kgs_reconstructed)
        ]
        averaged: Dict[str, Any] = {}
        for key in all_results[0]:
            if key.startswith("_"):
                continue
            vals = [r[key] for r in all_results
                    if isinstance(r.get(key), (int, float, np.integer, np.floating))]
            if vals:
                averaged[key] = float(np.mean(vals))
                averaged[f"{key}_std"] = float(np.std(vals))
        averaged["_base_kg_comparison"] = all_results[-1].get("_base_kg_comparison", {})
        return averaged

    # ================================================================== #
    # Adaptive threshold
    # ================================================================== #
    def compute_relative_threshold(
        self, baseline_distortion: float, epsilon: float = 0.10,
        absolute_threshold: Optional[float] = None,
    ) -> float:
        """
        delta = (1 + epsilon) * D_0, capped at absolute_threshold if given:
            delta = min( (1 + epsilon) * D_0 , absolute_threshold )
        """
        if baseline_distortion < 0:
            raise ValueError("baseline_distortion must be non-negative.")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        delta = baseline_distortion * (1.0 + epsilon)
        if absolute_threshold is not None:
            delta = min(delta, float(absolute_threshold))
        return float(delta)

    # ================================================================== #
    # Faithfulness gate
    # ================================================================== #
    def evaluate_faithfulness_gate(
        self, candidate_scores: Dict[str, float], delta: float,
        tau_img: float = 0.50, tau_cap: float = 0.55, tau_kg: float = 0.45,
    ) -> Dict[str, Any]:
        """Explicit per-condition gate (C1–C4). All four must pass; each is logged."""
        q_img = float(candidate_scores.get("Q_img", 0.0))
        q_cap = float(candidate_scores.get("Q_cap", 0.0))
        q_kg = float(candidate_scores.get("Q_KG", 0.0))
        d = float(candidate_scores.get("D", 1.0))

        c1, c2, c3, c4 = q_img >= tau_img, q_cap >= tau_cap, q_kg >= tau_kg, d <= delta
        passed = c1 and c2 and c3 and c4
        conditions = {
            "C1_Q_img": {"pass": c1, "value": round(q_img, 4), "threshold": tau_img,
                         "label": "visual fidelity (CLIP+DINO+LPIPS)"},
            "C2_Q_cap": {"pass": c2, "value": round(q_cap, 4), "threshold": tau_cap,
                         "label": "caption similarity (BERT+CLIPScore)"},
            "C3_Q_KG": {"pass": c3, "value": round(q_kg, 4), "threshold": tau_kg,
                        "label": "KG preservation (obj/attr/rel recall)"},
            "C4_D_delta": {"pass": c4, "value": round(d, 4), "threshold": round(delta, 4),
                           "label": "adaptive distortion budget"},
        }
        failed = [k for k, v in conditions.items() if not v["pass"]]

        if passed:
            gate_log = (f"GATE PASSED | Q_img={q_img:.4f}>={tau_img} | Q_cap={q_cap:.4f}>={tau_cap} | "
                        f"Q_KG={q_kg:.4f}>={tau_kg} | D={d:.4f}<={delta:.4f}")
        else:
            parts = []
            for k in failed:
                v = conditions[k]
                op = "<=" if k == "C4_D_delta" else ">="
                parts.append(f"{k}: {v['value']} FAILS {op} {v['threshold']} ({v['label']})")
            gate_log = "GATE FAILED | " + " | ".join(parts)

        return {"passed": passed, "conditions": conditions, "gate_log": gate_log, "failed_conditions": failed}

    # ================================================================== #
    # Legacy compatibility
    # ================================================================== #
    def compute_full_distortion(self, image_orig, image_recon, kg_original, kg_reconstructed) -> Dict[str, float]:
        """Legacy: returns baseline-style distortion."""
        return self.compute_baseline_distortion(image_orig, image_recon, kg_original, kg_reconstructed)

    def is_acceptable(self, candidate_distortion: float, threshold: float) -> bool:
        return float(candidate_distortion) <= float(threshold)


