import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STATUS_COMPRESSED = {"compressed", "compressed_via_fallback"}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_results(output_dir: str) -> List[Dict[str, Any]]:
    """Load per-image result JSONs from ``outputs/results`` (or a given dir)."""
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path}")

    candidate_files = []
    results_dir = output_path / "results"
    if results_dir.exists():
        candidate_files.extend(sorted(results_dir.glob("*.json")))
    candidate_files.extend(sorted(output_path.glob("*.json")))
    candidate_files = sorted(set(candidate_files))

    results: List[Dict[str, Any]] = []
    for path in candidate_files:
        if path.name in {"summary.json", "config.json", "config_used.yaml"}:
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item, dict) and "image_id" in item and "compressed_caption" in item:
                results.append(item)
        except Exception as exc:
            print(f"[Warning] Could not load {path}: {exc}")

    print(f"[Evaluate] Loaded {len(results)} result files from {output_path}")
    return results


def load_summary(output_dir: str) -> Optional[Dict[str, Any]]:
    output_path = Path(output_dir)
    for path in (output_path / "summary.json", output_path / "results" / "summary.json"):
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------------
# Safe access
# ---------------------------------------------------------------------

def metric_get(details: Dict[str, Any], keys: Sequence[str], default: Any = np.nan) -> Any:
    """First present key wins; tolerates non-dict input."""
    if not isinstance(details, dict):
        return default
    for key in keys:
        if key in details and details[key] is not None:
            return details[key]
    return default


def count_selected_steps(steps: List[Dict[str, Any]]) -> int:
    if not isinstance(steps, list):
        return 0
    return sum(1 for s in steps if s.get("selected", False) or s.get("accepted", False))


def _get_anchors(result: Dict[str, Any], kind: str) -> List[str]:
    if kind == "strict":
        return result.get("strict_anchors") or result.get("quoted_anchors") or []
    return result.get("soft_anchors") or result.get("protected_anchors") or []


def _anchor_preservation_rate(result: Dict[str, Any]) -> float:
    """
    Fraction of anchors retained in c*. Strict anchors: substring match.
    Soft anchors: content-word overlap. Dependency-light (no mpnet here).
    """
    strict = _get_anchors(result, "strict")
    soft = _get_anchors(result, "soft")
    cap = (result.get("compressed_caption") or "").lower()
    total = len(strict) + len(soft)
    if total == 0:
        return np.nan
    cap_words = set(re.sub(r"[^\w\s]", " ", cap).split())
    kept = 0
    for a in strict:
        if a.strip().lower() in cap:
            kept += 1
    for a in soft:
        words = [w for w in re.sub(r"[^\w\s]", " ", a.lower()).split() if len(w) > 2]
        if words and any(w in cap_words for w in words):
            kept += 1
    return kept / total


# ---------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------

def compute_dataset_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per image, flattening scores + comparison + anchors + steps."""
    rows = []
    for r in results:
        fd = r.get("final_distortion_details", {}) or {}      # gate/distortion scores
        fk = r.get("final_kg_comparison", {}) or {}            # comparison means
        steps = r.get("steps", []) or []

        q_img = metric_get(fd, ["Q_img", "Q_img_base", "base_Q_img"])
        q_cap = metric_get(fd, ["Q_cap"])                       # absent for skipped
        q_kg = metric_get(fd, ["Q_KG", "Q_KG_base", "base_Q_KG"])
        final_D = r.get("final_distortion", np.nan)

        clip_sim = metric_get(fd, ["CLIP_sim", "base_CLIP_sim", "CLIP_similarity"])
        dino_sim = metric_get(fd, ["DINO_sim", "base_DINO_sim"])
        lpips_d = metric_get(fd, ["LPIPS", "base_LPIPS"])
        bertscore = metric_get(fd, ["BERTScore"])
        clipscore = metric_get(fd, ["CLIPScore"])

        original_tokens = r.get("original_tokens", np.nan)
        compressed_tokens = r.get("compressed_tokens", np.nan)
        token_reduction = (
            original_tokens - compressed_tokens
            if isinstance(original_tokens, (int, float)) and isinstance(compressed_tokens, (int, float))
            else np.nan
        )

        rows.append({
            "image_id": r.get("image_id", ""),
            "status": r.get("status", ""),

            "original_caption": r.get("original_caption", ""),
            "compressed_caption": r.get("compressed_caption", ""),

            # Rate / length
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_reduction": token_reduction,
            "compression_ratio": r.get("compression_ratio", np.nan),

            # Distortion (overall + components)
            "baseline_distortion": r.get("baseline_distortion", np.nan),
            "threshold": r.get("threshold", np.nan),
            "final_distortion": final_D,
            "overall_distortion": final_D,
            "q_img": q_img,
            "q_cap": q_cap,
            "q_kg": q_kg,
            # semantic distortion = caption-side distortion term (1 - Q_cap)
            "semantic_distortion": (1.0 - q_cap) if isinstance(q_cap, (int, float)) and not _isnan(q_cap) else np.nan,

            # Q_img sub-components
            "clip_similarity": clip_sim,
            "dino_similarity": dino_sim,
            "lpips": lpips_d,
            # Q_cap sub-components
            "bertscore": bertscore,
            "clipscore": clipscore,

            # KG comparison (hard recall/F1 means from the comparison)
            "kg_f1": metric_get(fk, ["f1_mean"], default=metric_get(fd, ["KG_F1_hard"])),
            "kg_precision": metric_get(fk, ["precision_mean"]),
            "kg_recall": metric_get(fk, ["recall_mean"]),
            "object_recall": metric_get(fk, ["object_recall_mean"]),
            "attribute_recall": metric_get(fk, ["attribute_recall_mean"]),
            "relation_recall": metric_get(fk, ["relation_recall_mean"]),
            "weighted_kg_f1_hard": metric_get(fk, ["weighted_kg_f1_hard"], default=metric_get(fd, ["weighted_KG_F1_hard"])),
            "weighted_kg_f1_soft": metric_get(fk, ["weighted_kg_f1_soft"], default=metric_get(fd, ["weighted_KG_F1_soft"])),
            "chair_i": metric_get(fk, ["chair_i"], default=metric_get(fd, ["chair_i"])),

            # Gate-side recall (soft or hard depending on use_soft_kg)
            "gate_R_obj": metric_get(fd, ["R_obj", "base_R_obj"]),
            "gate_R_attr": metric_get(fd, ["R_attr", "base_R_attr"]),
            "gate_R_rel": metric_get(fd, ["R_rel", "base_R_rel"]),

            # Fact counts
            "num_preserved": metric_get(fk, ["num_preserved"], default=metric_get(fd, ["num_preserved"])),
            "num_missing": metric_get(fk, ["num_missing"], default=metric_get(fd, ["num_missing"])),
            "num_extra": metric_get(fk, ["num_extra"], default=metric_get(fd, ["num_extra"])),

            # Anchors
            "num_strict_anchors": len(_get_anchors(r, "strict")),
            "num_soft_anchors": len(_get_anchors(r, "soft")),
            "anchor_preservation_rate": _anchor_preservation_rate(r),

            # Steps / runtime
            "num_tested_steps": len(steps),
            "num_selected_steps": count_selected_steps(steps),
            "total_time": r.get("total_time", np.nan),
        })
    return pd.DataFrame(rows)


def _isnan(x: Any) -> bool:
    try:
        return np.isnan(x)
    except Exception:
        return False


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

_SUMMARY_COLUMNS = [
    "original_tokens", "compressed_tokens", "token_reduction", "compression_ratio",
    "baseline_distortion", "threshold", "final_distortion",
    "q_img", "q_cap", "q_kg", "semantic_distortion",
    "clip_similarity", "dino_similarity", "lpips", "bertscore", "clipscore",
    "kg_f1", "kg_precision", "kg_recall",
    "object_recall", "attribute_recall", "relation_recall",
    "weighted_kg_f1_hard", "weighted_kg_f1_soft", "chair_i",
    "anchor_preservation_rate", "num_selected_steps", "total_time",
]


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in _SUMMARY_COLUMNS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        rows.append({
            "metric": col, "mean": float(series.mean()), "std": float(series.std()),
            "min": float(series.min()), "max": float(series.max()),
            "n": int(series.notna().sum()),
        })
    return pd.DataFrame(rows)


def filter_compressed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only successfully-compressed rows (drops skipped / failed)."""
    if "status" not in df.columns:
        return df
    return df[df["status"].isin(STATUS_COMPRESSED)].copy()


def print_summary_table(df: pd.DataFrame, label: str = "") -> None:
    print("\n" + "=" * 70)
    if label:
        print(label)
        print("-" * 70)
    if "status" in df.columns:
        counts = df["status"].value_counts().to_dict()
        print(f"status counts: {counts}")
        print("-" * 70)
    display = [
        ("Compression Ratio", "compression_ratio"), ("Original Tokens", "original_tokens"),
        ("Compressed Tokens", "compressed_tokens"), ("Baseline Distortion", "baseline_distortion"),
        ("Final Distortion", "final_distortion"), ("Q_img", "q_img"), ("Q_cap", "q_cap"),
        ("Q_KG", "q_kg"), ("CLIP-I sim", "clip_similarity"), ("DINO sim", "dino_similarity"),
        ("LPIPS", "lpips"), ("BERTScore", "bertscore"), ("CLIPScore", "clipscore"),
        ("KG F1", "kg_f1"), ("KG Recall", "kg_recall"), ("Object Recall", "object_recall"),
        ("Attribute Recall", "attribute_recall"), ("Relation Recall", "relation_recall"),
        ("Weighted KG-F1 (hard)", "weighted_kg_f1_hard"), ("Weighted KG-F1 (soft)", "weighted_kg_f1_soft"),
        ("CHAIR_i", "chair_i"), ("Anchor Preservation", "anchor_preservation_rate"),
        ("Selected Steps", "num_selected_steps"), ("Total Time (s)", "total_time"),
    ]
    for name, col in display:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        print(f"{name:24s}: {series.mean():.4f} +/- {series.std():.4f}")
    print("=" * 70 + "\n")


def save_metrics_csv(df: pd.DataFrame, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Evaluate] Metrics CSV saved to {output_path}")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_rate_distortion_curve(results_by_setting: Dict[Union[float, str], List[Dict[str, Any]]],
                               output_path: str = "rd_curve.pdf", setting_label: str = "epsilon") -> None:
    settings = sorted(results_by_setting.keys())
    crs, ds = [], []
    for s in settings:
        df = filter_compressed(compute_dataset_metrics(results_by_setting[s]))
        crs.append(pd.to_numeric(df["compression_ratio"], errors="coerce").mean())
        ds.append(pd.to_numeric(df["final_distortion"], errors="coerce").mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(crs, ds, marker="o", linewidth=2)
    for i, s in enumerate(settings):
        ax.annotate(f"{setting_label}={s}", (crs[i], ds[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.set_xlabel("Compression Ratio"); ax.set_ylabel("Final Distortion")
    ax.set_title("Rate-Distortion Curve"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Plot] Rate-distortion curve saved to {output_path}")


def plot_ablation_distortion_components(results_by_config: Dict[str, List[Dict[str, Any]]],
                                        output_path: str = "ablation_distortion.pdf") -> None:
    configs = list(results_by_config.keys())
    crs, kg_f1s, obj_r, rel_r = [], [], [], []
    for c in configs:
        df = filter_compressed(compute_dataset_metrics(results_by_config[c]))
        crs.append(pd.to_numeric(df["compression_ratio"], errors="coerce").mean())
        kg_f1s.append(pd.to_numeric(df["kg_f1"], errors="coerce").mean())
        obj_r.append(pd.to_numeric(df["object_recall"], errors="coerce").mean())
        rel_r.append(pd.to_numeric(df["relation_recall"], errors="coerce").mean())
    x = np.arange(len(configs)); width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * width, crs, width, label="Compression Ratio")
    ax.bar(x - 0.5 * width, kg_f1s, width, label="KG F1")
    ax.bar(x + 0.5 * width, obj_r, width, label="Object Recall")
    ax.bar(x + 1.5 * width, rel_r, width, label="Relation Recall")
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("Score"); ax.set_title("Ablation: Distortion Components")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Plot] Ablation plot saved to {output_path}")


def plot_compression_trace(result: Dict[str, Any], output_path: str = "compression_trace.pdf") -> None:
    steps = result.get("steps", [])
    if not isinstance(steps, list) or not steps:
        print("[Plot] No steps; trace not created.")
        return
    tokens = [result.get("original_tokens", np.nan)]
    distortions = [result.get("baseline_distortion", np.nan)]
    for step in steps:
        if not (step.get("selected", False) or step.get("accepted", False)):
            continue
        tokens.append(step.get("tokens_after", np.nan))
        distortions.append(step.get("distortion", np.nan))
    if len(tokens) <= 1:
        print("[Plot] No selected steps.")
        return
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = list(range(len(tokens)))
    ax1.plot(x, tokens, marker="o", linewidth=2, label="Token Count")
    ax1.set_xlabel("Selected Compression Step"); ax1.set_ylabel("Token Count"); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, distortions, marker="s", linestyle="--", linewidth=2, label="Distortion")
    ax2.set_ylabel("Distortion")
    thr = result.get("threshold")
    if thr is not None:
        ax2.axhline(float(thr), linestyle=":", linewidth=1.5, label="Threshold (delta)")
    ax1.set_title(f"Compression Trace: {result.get('image_id', 'unknown')}")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="best")
    plt.tight_layout(); plt.savefig(output_path, dpi=300, bbox_inches="tight"); plt.close()
    print(f"[Plot] Compression trace saved to {output_path}")


# ---------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------

def generate_latex_table(results_dict: Dict[str, pd.DataFrame], output_path: str = "results_table.tex") -> None:
    def _m(df, col):
        return pd.to_numeric(df[col], errors="coerce").mean() if col in df.columns else float("nan")
    rows = []
    for method_name, df in results_dict.items():
        rows.append({
            "Method": method_name,
            "CR": f"{_m(df, 'compression_ratio'):.1%}",
            "D": f"{_m(df, 'final_distortion'):.3f}",
            "CLIP-I": f"{_m(df, 'clip_similarity'):.3f}",
            "DINO": f"{_m(df, 'dino_similarity'):.3f}",
            "LPIPS": f"{_m(df, 'lpips'):.3f}",
            "KG-F1": f"{_m(df, 'kg_f1'):.3f}",
            "Obj-R": f"{_m(df, 'object_recall'):.3f}",
            "Rel-R": f"{_m(df, 'relation_recall'):.3f}",
            "CHAIR": f"{_m(df, 'chair_i'):.3f}",
        })
    latex = pd.DataFrame(rows).to_latex(index=False, escape=False, column_format="l" + "c" * 9)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"[LaTeX] Table saved to {output_path}")


# ---------------------------------------------------------------------
# Debug entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    results = load_results("./outputs")
    df = compute_dataset_metrics(results)

    print_summary_table(df, label="KG-RD Compression — ALL samples")
    compressed = filter_compressed(df)
    if len(compressed) and len(compressed) != len(df):
        print_summary_table(compressed, label="KG-RD Compression — COMPRESSED only")

    save_metrics_csv(df, "./outputs/results/metrics.csv")
    save_metrics_csv(summarize_dataframe(df), "./outputs/results/metrics_summary.csv")
    print(summarize_dataframe(df).to_string(index=False))