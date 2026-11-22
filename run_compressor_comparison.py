"""
Compressor ablation: swap the caption rewriter (Qwen2.5-VL / GPT-4o / Gemini)
while keeping everything else fixed (KG extractor, FLUX reconstructor, metrics,
gate, and budgets).

Models are loaded once. For each rewriter, results are written to
<output_dir>/compressor_<model>/ so runs do not overwrite each other.

Usage:
    export OPENAI_API_KEY=...      # for gpt4o
    export GEMINI_API_KEY=...      # for gemini
    python run_compressor_comparison.py --config config.yaml \
        --max-samples 50 --models qwen gpt4o gemini
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from src.compressor import KGRDCompressor, QwenVLRewriter, make_json_serializable
from src.dataset import VehicleCaptionDataset


def build_rewriter(name: str, compressor: KGRDCompressor, cfg: Dict[str, Any], args):
    name = name.lower()
    rw = cfg.get("rewriter", {})
    if name == "qwen":
        # Reuse the already-loaded Qwen2.5-VL model (no second GPU copy).
        return QwenVLRewriter(vlm_extractor=compressor.sg_extractor,
                              max_new_tokens=rw.get("max_new_tokens", 512),
                              temperature=rw.get("temperature", 0.7),
                              top_p=rw.get("top_p", 0.9))
    if name == "gpt4o":
        from llm_rewriters import GPT4oRewriter
        return GPT4oRewriter(model=args.gpt4o_model)
    if name == "gemini":
        from llm_rewriters import GeminiRewriter
        return GeminiRewriter(model=args.gemini_model)
    raise ValueError(f"Unknown model '{name}' (use: qwen, gpt4o, gemini)")


def set_output_dir(compressor: KGRDCompressor, out_dir: Path) -> None:
    """Point all of the compressor's output folders at out_dir."""
    compressor.output_dir = out_dir
    compressor.reconstruction_dir = out_dir / "reconstructions"
    compressor.result_dir = out_dir / "results"
    compressor.kg_debug_dir = out_dir / "kg_debug"
    compressor.candidate_debug_dir = out_dir / "candidates"
    compressor.debug_dir = out_dir / "debug_samples"
    for d in (compressor.result_dir, compressor.kg_debug_dir,
              compressor.candidate_debug_dir, compressor.debug_dir):
        d.mkdir(parents=True, exist_ok=True)


def run_one_model(name: str, dataset, n: int, compressor: KGRDCompressor) -> List[Dict[str, Any]]:
    rows = []
    for idx in range(n):
        sample = dataset[idx]
        image_id = sample["image_id"]
        print(f"\n[{name}] [{idx + 1}/{n}] {image_id}")
        try:
            result = compressor.compress_single(sample["image"], sample["caption"], image_id=image_id)
        except Exception as exc:
            print(f"[ERROR] {image_id}: {exc}")
            continue
        r = make_json_serializable(asdict(result))
        fd = r.get("final_distortion_details", {}) or {}
        row = {
            "model": name,
            "image_id": image_id,
            "status": r["status"],
            "compression_ratio": r["compression_ratio"],
            "Q_img": fd.get("Q_img"),
            "Q_cap": fd.get("Q_cap"),
            "Q_KG": fd.get("Q_KG"),
            "D_final": r["final_distortion"],
            "passed": int(str(r["status"]).startswith("compressed")),
        }
        rows.append(row)
        print(f"    CR={row['compression_ratio']} Q_img={row['Q_img']} Q_cap={row['Q_cap']} "
              f"Q_KG={row['Q_KG']} D={row['D_final']} status={row['status']}")
    return rows


def aggregate(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    """Mean/std over samples that were compressed (passed or via fallback)."""
    all_m = [r for r in rows if r["model"] == model]
    sub = [r for r in all_m if r["passed"] == 1]
    out = {"model": model, "n_total": len(all_m), "n_passed": len(sub),
           "pass_rate": len(sub) / max(len(all_m), 1)}
    for k in ("compression_ratio", "Q_img", "Q_cap", "Q_KG", "D_final"):
        vals = [r[k] for r in sub if r[k] is not None]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals)) if vals else None
    return out


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[compare] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="MS-CAPS compressor ablation")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["qwen", "gpt4o", "gemini"],
                    help="Any of: qwen gpt4o gemini")
    ap.add_argument("--gpt4o-model", default="gpt-4o-2024-08-06")
    ap.add_argument("--gemini-model", default="gemini-2.5-pro")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("dataset", {})["max_samples"] = args.max_samples

    dataset = VehicleCaptionDataset.from_config(config)
    n = min(args.max_samples, len(dataset))
    print(f"[compare] {n} samples; models={args.models}")

    # Loads the fixed parts once: KG extractor, FLUX, metrics.
    compressor = KGRDCompressor.from_config(config)
    base_out = Path(config.get("paths", {}).get("output_dir", "./outputs"))

    all_rows: List[Dict[str, Any]] = []
    for name in args.models:
        compressor.rewriter = build_rewriter(name, compressor, config, args)
        set_output_dir(compressor, base_out / f"compressor_{name}")
        all_rows.extend(run_one_model(name, dataset, n, compressor))

    if not all_rows:
        print("[compare] no results produced.")
        return

    out = base_out / "compressor_comparison"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(all_rows, out / "per_sample.csv")
    aggs = [aggregate(all_rows, name) for name in args.models]
    write_csv(aggs, out / "summary.csv")
    print("\n==== Compressor comparison (compressed samples) ====")
    for a in aggs:
        print(a)


if __name__ == "__main__":
    main()
