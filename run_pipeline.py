import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import VehicleCaptionDataset
from src.compressor import KGRDCompressor, CompressionResult, make_json_serializable
from src import evaluate as ev

import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")
SELECTED_FILL = PatternFill("solid", fgColor="FFEB9C")
SUMMARY_FILL = PatternFill("solid", fgColor="2E75B6")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
BOLD_FONT = Font(bold=True, size=10)
NORMAL_FONT = Font(size=10)
SUMMARY_FONT = Font(bold=True, color="FFFFFF", size=10)
THIN_BORDER = Border(left=Side(style="thin"), right=Side(style="thin"),
                     top=Side(style="thin"), bottom=Side(style="thin"))

_COMPRESSED_STATUSES = {"compressed", "compressed_via_fallback"}

def _fmt(v, decimals=4):
    return round(v, decimals) if isinstance(v, float) else v

def _md(details: Dict[str, Any], keys, default=0.0):
    """First present key wins (handles candidate vs base_-prefixed schemas)."""
    if not isinstance(details, dict):
        return default
    for k in keys:
        if k in details and details[k] is not None:
            return details[k]
    return default

def _classify_tuples(comparison: Dict[str, Any], kind: str):
    """Split preserved/missing/extra tuple-sets into objects or relations."""
    def fmt(t):
        t = list(t)
        if kind == "object" and len(t) >= 2 and t[0] == "object":
            return t[1]
        if kind == "relation" and len(t) == 4 and t[0] == "relation":
            return f"({t[1]}, {t[2]}, {t[3]})"
        return None

    def pull(key):
        out = []
        for t in comparison.get(key, []) or []:
            s = fmt(t)
            if s:
                out.append(s)
        return sorted(out)

    return pull("preserved"), pull("missing"), pull("extra")

def _embed_image(ws, cell_ref, image_path, target_width=160):
    from openpyxl.drawing.image import Image as XLImage
    from PIL import Image as PILImage
    if not image_path or not Path(image_path).exists():
        return
    try:
        pil = PILImage.open(image_path).convert("RGB")
        w, h = pil.size
        new_h = int(h * (target_width / w))
        pil.thumbnail((target_width, new_h), PILImage.LANCZOS)
        thumb = Path(image_path).parent / f".xlthumb_{Path(image_path).stem}.png"
        pil.save(thumb)
        img = XLImage(str(thumb))
        img.anchor = cell_ref
        ws.add_image(img)
    except Exception as e:
        print(f"   [Excel] could not embed {image_path}: {e}")

def write_excel(results: List[Dict[str, Any]], config: Dict[str, Any], output_path: Path):
    """One row per sample + a summary row. results are CompressionResult dicts."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "KG-RD Analysis"

    columns = [
        ("Sample ID", 28), ("Original Image", 26), ("Baseline G(c)", 26), ("Final I*", 26),
        ("Original Caption", 50), ("Compressed Caption (c*)", 45),
        ("Original Tokens", 10), ("Compressed Tokens", 12), ("Compression Ratio", 12),
        ("Total Time (s)", 10), ("Status", 22),
        ("D_0", 10), ("Delta", 10), ("D_final", 10), ("D_final <= Delta?", 14),
        ("Q_img", 10), ("Q_cap", 10), ("Q_KG", 10),
        ("CLIP-I sim", 11), ("DINOv2 sim", 11), ("LPIPS dist", 11),
        ("BERTScore", 11), ("CLIPScore", 11),
        ("R_obj", 10), ("R_attr", 10), ("R_rel", 10),
        ("KG-F1 hard", 11), ("KG-F1 soft", 11), ("CHAIR_i", 10),
        ("Strict Anchors", 30), ("Soft Anchors", 30),
        ("Supported Objects (I*)", 28), ("Missing Objects (I*)", 28), ("Hallucinated Objects (I*)", 28),
        ("Supported Relations (I*)", 30), ("Missing Relations (I*)", 30), ("Extra Relations (I*)", 30),
        ("KG_I Counts", 16), ("KG_{I*} Counts", 16), ("Model Stack", 24),
    ]
    for i, (name, width) in enumerate(columns, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.fill = HEADER_FILL
        c.font = HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = THIN_BORDER
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.row_dimensions[1].height = 45
    ws.freeze_panes = "B2"

    results = sorted(results, key=lambda r: r.get("image_id", ""))
    agg: Dict[str, List[float]] = {k: [] for k in (
        "D_0", "delta", "D_final", "Q_img", "Q_cap", "Q_KG", "CLIP_sim", "DINO_sim",
        "LPIPS", "BERTScore", "CLIPScore", "R_obj", "R_attr", "R_rel",
        "KG_F1_hard", "KG_F1_soft", "CHAIR_i", "compression_ratio", "total_time",
        "original_tokens", "compressed_tokens")}
    status_counts: Dict[str, int] = {}

    sg = config.get("scene_graph", {}).get("vlm", {})
    t2i = config.get("t2i", {})
    model_stack = (f"{sg.get('model', 'Qwen2.5-VL')}\n{t2i.get('model', 'FLUX.1-dev')}\n"
                   f"text-only, steps={t2i.get('num_inference_steps', 20)}")

    for r_idx, r in enumerate(results, 2):
        ws.row_dimensions[r_idx].height = 140
        fd = r.get("final_distortion_details", {}) or {}
        fk = r.get("final_kg_comparison", {}) or {}
        status = r.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

        d0, delta, dfin = r.get("baseline_distortion", 0.0), r.get("threshold", 0.0), r.get("final_distortion", 0.0)
        q_img = _md(fd, ["Q_img", "Q_img_base", "base_Q_img"])
        q_cap = _md(fd, ["Q_cap"])
        q_kg = _md(fd, ["Q_KG", "Q_KG_base", "base_Q_KG"])
        clip_sim = _md(fd, ["CLIP_sim", "base_CLIP_sim"])
        dino_sim = _md(fd, ["DINO_sim", "base_DINO_sim"])
        lpips_d = _md(fd, ["LPIPS", "base_LPIPS"])
        bert = _md(fd, ["BERTScore"])
        clipscore = _md(fd, ["CLIPScore"])
        r_obj = _md(fd, ["R_obj", "base_R_obj"])
        r_attr = _md(fd, ["R_attr", "base_R_attr"])
        r_rel = _md(fd, ["R_rel", "base_R_rel"])
        kg_f1_hard = _md(fk, ["weighted_kg_f1_hard"], default=_md(fd, ["weighted_KG_F1_hard"]))
        kg_f1_soft = _md(fk, ["weighted_kg_f1_soft"], default=_md(fd, ["weighted_KG_F1_soft"]))
        chair = _md(fk, ["chair_i"], default=_md(fd, ["chair_i"]))

        supp_o, miss_o, hall_o = _classify_tuples(fk, "object")
        supp_r, miss_r, extra_r = _classify_tuples(fk, "relation")

        for key, val in (("D_0", d0), ("delta", delta), ("D_final", dfin), ("Q_img", q_img),
                         ("Q_cap", q_cap), ("Q_KG", q_kg), ("CLIP_sim", clip_sim), ("DINO_sim", dino_sim),
                         ("LPIPS", lpips_d), ("BERTScore", bert), ("CLIPScore", clipscore),
                         ("R_obj", r_obj), ("R_attr", r_attr), ("R_rel", r_rel),
                         ("KG_F1_hard", kg_f1_hard), ("KG_F1_soft", kg_f1_soft), ("CHAIR_i", chair),
                         ("compression_ratio", r.get("compression_ratio", 0.0)),
                         ("total_time", r.get("total_time", 0.0)),
                         ("original_tokens", r.get("original_tokens", 0)),
                         ("compressed_tokens", r.get("compressed_tokens", 0))):
            if isinstance(val, (int, float)):
                agg[key].append(float(val))

        kgI = r.get("kg_I_counts", {})
        kgF = r.get("kg_final_counts", {})
        pf = "YES" if dfin <= delta else "NO"
        status_fill = (PASS_FILL if status == "compressed"
                       else SELECTED_FILL if status in _COMPRESSED_STATUSES
                       else FAIL_FILL)

        row = [
            r.get("image_id", ""), "", "", "",
            r.get("original_caption", ""), r.get("compressed_caption", ""),
            r.get("original_tokens", 0), r.get("compressed_tokens", 0),
            f"{r.get('compression_ratio', 0.0):.1%}", f"{r.get('total_time', 0.0):.1f}", status,
            _fmt(d0), _fmt(delta), _fmt(dfin), pf,
            _fmt(q_img), _fmt(q_cap), _fmt(q_kg),
            _fmt(clip_sim), _fmt(dino_sim), _fmt(lpips_d), _fmt(bert), _fmt(clipscore),
            _fmt(r_obj), _fmt(r_attr), _fmt(r_rel), _fmt(kg_f1_hard), _fmt(kg_f1_soft), _fmt(chair),
            " | ".join(r.get("strict_anchors", [])) or "(none)",
            " | ".join(r.get("soft_anchors", [])) or "(none)",
            ", ".join(supp_o) or "(none)", ", ".join(miss_o) or "(none)", ", ".join(hall_o) or "(none)",
            "\n".join(supp_r) or "(none)", "\n".join(miss_r) or "(none)", "\n".join(extra_r) or "(none)",
            f"obj:{kgI.get('obj', 0)} attr:{kgI.get('attr', 0)} rel:{kgI.get('rel', 0)}",
            f"obj:{kgF.get('obj', 0)} attr:{kgF.get('attr', 0)} rel:{kgF.get('rel', 0)}",
            model_stack,
        ]
        for i, val in enumerate(row, 1):
            cell = ws.cell(row=r_idx, column=i, value=val)
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            cell.border = THIN_BORDER
            cell.font = NORMAL_FONT
            if isinstance(val, float):
                cell.number_format = "0.0000"
        ws.cell(row=r_idx, column=11).fill = status_fill
        ws.cell(row=r_idx, column=11).font = BOLD_FONT
        ws.cell(row=r_idx, column=15).fill = PASS_FILL if pf == "YES" else FAIL_FILL
        ws.cell(row=r_idx, column=15).font = BOLD_FONT

        _embed_image(ws, f"B{r_idx}", r.get("original_image_path"))
        _embed_image(ws, f"C{r_idx}", r.get("baseline_reconstruction_path"))
        _embed_image(ws, f"D{r_idx}", r.get("final_reconstruction_path"))

    n = len(results)
    summary_row = n + 2
    ws.row_dimensions[summary_row].height = 30
    avg = lambda k: float(np.mean(agg[k])) if agg[k] else 0.0
    n_compressed = sum(1 for r in results if r.get("status") in _COMPRESSED_STATUSES)
    pass_rate = n_compressed / max(n, 1)

    summary = [
        f"SUMMARY ({n} samples)", "", "", "", "",
        f"compressed {n_compressed}/{n} ({pass_rate:.1%})",
        round(avg("original_tokens"), 1), round(avg("compressed_tokens"), 1),
        f"{avg('compression_ratio'):.1%}", f"{avg('total_time'):.1f}", str(status_counts),
        _fmt(avg("D_0")), _fmt(avg("delta")), _fmt(avg("D_final")), f"{pass_rate:.1%}",
        _fmt(avg("Q_img")), _fmt(avg("Q_cap")), _fmt(avg("Q_KG")),
        _fmt(avg("CLIP_sim")), _fmt(avg("DINO_sim")), _fmt(avg("LPIPS")),
        _fmt(avg("BERTScore")), _fmt(avg("CLIPScore")),
        _fmt(avg("R_obj")), _fmt(avg("R_attr")), _fmt(avg("R_rel")),
        _fmt(avg("KG_F1_hard")), _fmt(avg("KG_F1_soft")), _fmt(avg("CHAIR_i")),
        "", "", "", "", "", "", "", "", "", "",
        f"Total {sum(agg['total_time']) / 60:.1f} min",
    ]
    for i, val in enumerate(summary, 1):
        cell = ws.cell(row=summary_row, column=i, value=val)
        cell.fill = SUMMARY_FILL
        cell.font = SUMMARY_FONT
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER
        if isinstance(val, float):
            cell.number_format = "0.0000"

    ws2 = wb.create_sheet("Formulas & Weights")
    ws2.column_dimensions["A"].width = 30
    ws2.column_dimensions["B"].width = 80
    d = config.get("distortion", {})
    w = d.get("weights", {})
    kg = d.get("kg", {})
    g = config.get("gate", {})
    comp = config.get("compression", {})
    ref = [
        ("FRAMEWORK", "KG-RD (reconstruction-guided; image-side KG both sides)"),
        ("Q_img", "(CLIP_sim + DINOv2_sim + (1 - LPIPS)) / 3"),
        ("Q_cap", "0.5 * BERTScore(c, c*) + 0.5 * CLIPScore(c*, I)"),
        ("Q_KG", f"{kg.get('object_weight')}*R_obj + {kg.get('attribute_weight')}*R_attr + {kg.get('relation_weight')}*R_rel"),
        ("D", f"{w.get('alpha_img')}*(1-Q_img) + {w.get('beta_cap')}*(1-Q_cap) + {w.get('gamma_kg')}*(1-Q_KG)"),
        ("D_0", f"{w.get('alpha_img')}*(1-Q_img_base) + {w.get('gamma_kg')}*(1-Q_KG_base)"),
        ("Delta", f"min((1 + {d.get('epsilon')})*D_0, {d.get('absolute_threshold')})"),
        ("Gate", f"Q_img>={g.get('tau_img')} AND Q_cap>={g.get('tau_cap')} AND Q_KG>={g.get('tau_kg')} AND D<=Delta"),
        ("Selection", "shortest faithful (budget ladder, escalate, never original)"),
        ("Q_KG recall", f"soft={d.get('use_soft_kg', True)} (mpnet); both hard+soft reported"),
        ("", ""),
        ("Budget ladder", str(comp.get("budget_ratios", [0.20, 0.30, 0.45, 0.55, 0.65]))),
        ("Escalation ladder", str(comp.get("escalation_ratios", [0.70, 0.78, 0.86, 0.94]))),
        ("Candidates / iter", str(comp.get("num_candidates_per_iteration"))),
        ("Max iterations", str(comp.get("max_iterations"))),
        ("Fallback", str(config.get("fallback", {}).get("strategy"))),
        ("", ""),
        ("Scene-graph VLM", str(sg.get("model"))),
        ("T2I model", str(t2i.get("model"))),
        ("CLIP", f"{d.get('clip', {}).get('model')}/{d.get('clip', {}).get('pretrained')}"),
        ("DINOv2", str(d.get("dino", {}).get("model"))),
        ("LPIPS net", str(d.get("lpips", {}).get("net"))),
        ("BERTScore", str(d.get("bertscore", {}).get("model"))),
    ]
    for i, (k, v) in enumerate(ref, 1):
        ws2.cell(row=i, column=1, value=k).font = BOLD_FONT
        ws2.cell(row=i, column=2, value=v).font = NORMAL_FONT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"\n[Excel] Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="KG-RD pipeline runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override dataset.max_samples (None = use config).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute even if a result JSON already exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.max_samples is not None:
        config.setdefault("dataset", {})["max_samples"] = args.max_samples
    max_samples = config.get("dataset", {}).get("max_samples")

    output_dir = Path(config.get("paths", {}).get("output_dir", "./outputs"))
    results_dir = output_dir / "results"
    excel_path = output_dir / "kg_rd_report.xlsx"

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print("=" * 70 + "\nKG-RD PIPELINE\n" + "=" * 70)
    dataset = VehicleCaptionDataset.from_config(config)
    n = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    print(f"Dataset: {len(dataset)} samples; processing {n}.")

    compressor = KGRDCompressor.from_config(config)

    results: List[Dict[str, Any]] = []
    for idx in range(n):
        sample = dataset[idx]
        image_id = sample.get("image_id", f"sample_{idx:06d}")
        result_json = results_dir / f"{image_id}.json"

        if not args.no_resume and result_json.exists():
            try:
                import json
                cached = json.loads(result_json.read_text(encoding="utf-8"))
                results.append(cached)
                print(f"[{idx+1}/{n}] {image_id} (cached, skipped)")
                continue
            except Exception:
                pass

        print(f"\n[{idx+1}/{n}] {image_id}")
        try:
            result = compressor.compress_single(
                image=sample["image"], caption=sample["caption"], image_id=image_id)
            results.append(make_json_serializable(asdict(result)))
        except Exception as exc:
            import traceback
            print(f"[ERROR] {image_id}: {exc}")
            traceback.print_exc()
            continue

    if not results:
        print("No results produced.")
        return

    write_excel(results, config, excel_path)

    df = ev.compute_dataset_metrics(results)
    ev.save_metrics_csv(df, str(results_dir / "metrics.csv"))
    ev.save_metrics_csv(ev.summarize_dataframe(df), str(results_dir / "metrics_summary.csv"))
    ev.print_summary_table(df, label="KG-RD — ALL samples")
    compressed = ev.filter_compressed(df)
    if len(compressed) and len(compressed) != len(df):
        ev.print_summary_table(compressed, label="KG-RD — COMPRESSED only")

    print("\n" + "=" * 70)
    print(f"DONE. {len(results)} results | Excel: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

