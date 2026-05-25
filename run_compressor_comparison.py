import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

from src.compressor import KGRDCompressor
from src.scene_graph import VLMSceneGraphExtractor


def load_dataset(config: Dict[str, Any], max_samples: Optional[int]) -> List[Dict[str, str]]:
    """
    Return [{image_id, image_path, caption}, ...].

    NOTE: adapt this to however the MAIN run reads its 50 (image, dense caption)
    pairs. The placeholder reconstructs them from the saved debug_samples + the
    original captions stored in outputs/results/dc_*.json (which record the
    original caption and original_image_path). If your main loader reads from a
    manifest/CSV/JSON dataset file instead, replace this body to read that.
    """
    items = []
    for jf in sorted(glob.glob("outputs/results/dc_*.json")):
        d = json.load(open(jf))
        img = d.get("original_image_path")
        cap = d.get("original_caption") or d.get("dense_caption")
        sid = d.get("image_id", Path(jf).stem)
        if img and cap and os.path.exists(img):
            items.append({"image_id": sid, "image_path": img, "caption": cap})
        if max_samples and len(items) >= max_samples:
            break
    return items


def build_rewriter(name: str, sg_extractor, cfg: Dict[str, Any]):
    name = name.lower()
    if name == "qwen":
        from src.compressor import QwenVLRewriter
        rw = cfg.get("rewriter", {})
        return QwenVLRewriter(vlm_extractor=sg_extractor,
                              max_new_tokens=rw.get("max_new_tokens", 512),
                              temperature=rw.get("temperature", 0.7),
                              top_p=rw.get("top_p", 0.9))
    if name == "gpt4o":
        from llm_rewriters import GPT4oRewriter
        return GPT4oRewriter()
    if name == "gemini":
        from llm_rewriters import GeminiRewriter
        return GeminiRewriter()
    raise ValueError(f"unknown model '{name}' (use qwen / gpt4o / gemini)")


def run_one_model(name: str, dataset, config, sg_extractor) -> List[Dict[str, Any]]:
    rewriter = build_rewriter(name, sg_extractor, config)
    # Same pipeline, only the rewriter differs.
    comp = KGRDCompressor.from_config(config, rewriter=rewriter)
    rows = []
    for i, item in enumerate(dataset, 1):
        img = Image.open(item["image_path"]).convert("RGB")
        print(f"\n[{name}] [{i}/{len(dataset)}] {item['image_id']}")
        try:
            result = comp.compress_single(img, item["caption"], image_id=item["image_id"])
        except TypeError:
            # fall back if the method name/signature differs — adapt to your API
            result = comp.compress(img, item["caption"])
        # Pull metrics from the result dict; key names follow the main-run schema.
        rows.append({
            "model": name,
            "image_id": item["image_id"],
            "status": result.get("status", ""),
            "compression_ratio": result.get("compression_ratio"),
            "Q_img": result.get("Q_img"),
            "Q_cap": result.get("Q_cap"),
            "Q_KG": result.get("Q_KG"),
            "D_final": result.get("D_final"),
            "passed": 1 if result.get("status", "").startswith("compressed") else 0,
        })
        r = rows[-1]
        print(f"    CR={r['compression_ratio']} Q_img={r['Q_img']} Q_cap={r['Q_cap']} "
              f"Q_KG={r['Q_KG']} D_final={r['D_final']} status={r['status']}")
    return rows


def aggregate(rows: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    sub = [r for r in rows if r["model"] == model and r["passed"] == 1]
    if not sub:
        return {"model": model, "n_passed": 0}
    def m(k):
        vals = [r[k] for r in sub if r[k] is not None]
        return float(np.mean(vals)) if vals else None
    def s(k):
        vals = [r[k] for r in sub if r[k] is not None]
        return float(np.std(vals)) if vals else None
    n_total = len([r for r in rows if r["model"] == model])
    return {
        "model": model, "n_passed": len(sub), "n_total": n_total,
        "pass_rate": len(sub) / max(n_total, 1),
        "CR_mean": m("compression_ratio"), "CR_std": s("compression_ratio"),
        "Q_img": m("Q_img"), "Q_cap": m("Q_cap"), "Q_KG": m("Q_KG"),
        "D_final": m("D_final"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--models", nargs="+", default=["qwen", "gpt4o"],
                    help="subset of: qwen gpt4o gemini")
    ap.add_argument("--output-dir", default="outputs")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    dataset = load_dataset(config, args.max_samples)
    print(f"[compare] {len(dataset)} samples; models={args.models}")
    if not dataset:
        print("[compare] no samples loaded — adapt load_dataset() to your dataset loader."); return

    # One shared KG extractor (the FIXED scorer); rewriters are swapped per model.
    sg_extractor = VLMSceneGraphExtractor(
        model_name=config.get("scene_graph", {}).get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        device=config.get("scene_graph", {}).get("device", "cuda:0"),
    )

    all_rows = []
    for name in args.models:
        all_rows.extend(run_one_model(name, dataset, config, sg_extractor))

    out = Path(args.output_dir) / "results"
    out.mkdir(parents=True, exist_ok=True)
    per_csv = out / "compressor_comparison_per_sample.csv"
    with open(per_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    print(f"\n[compare] wrote {per_csv}")

    print("\n==== Compressor comparison (passed/compressed subset) ====")
    aggs = [aggregate(all_rows, name) for name in args.models]
    agg_csv = out / "compressor_comparison_summary.csv"
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(aggs[0].keys())); w.writeheader(); w.writerows(aggs)
    for a in aggs:
        print(a)
    print(f"\n[compare] wrote {agg_csv}")


if __name__ == "__main__":
    main()