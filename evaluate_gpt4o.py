"""
Evaluator ablation: re-score the main pipeline's outputs with GPT-4o as the
scene-graph (KG) extractor instead of Qwen2.5-VL.

For every result JSON from run_pipeline.py, GPT-4o extracts a KG from the
original image I and from the final reconstruction G(c*), and the two KGs are
compared with the same compare_scene_graphs() used in the main pipeline.

Run run_pipeline.py first, then:
    export OPENAI_API_KEY=...
    python evaluate_gpt4o.py --output-dir outputs
"""

import argparse
import base64
import csv
import glob
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

# Reuse the project's comparison + post-processing UNCHANGED -> fair comparison.
from src.scene_graph import (
    compare_scene_graphs,
    build_tuple_set,
    build_networkx_graph,
    normalize_label,
    normalize_relation,
    _VLM_KG_PROMPT_TEMPLATE,
)

MODEL = "gpt-4o-2024-08-06"   # pinned snapshot — cite this in the paper

# GPT-4o pricing (USD per token). VERIFY against platform.openai.com/pricing —
# rates change; these are used only for the convenience cost print-out.
_PRICE_IN_PER_TOK = 2.50 / 1_000_000
_PRICE_OUT_PER_TOK = 10.00 / 1_000_000

# JSON schema for structured outputs: forces objects/relations in our format.
_KG_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "attributes"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["objects", "relations"],
    "additionalProperties": False,
}

# Relation-only schema for the retry pass (parity with Qwen's relation retry).
_REL_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relations"],
    "additionalProperties": False,
}


def _img_to_data_url(image: Image.Image, max_side: int = 768) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    # downscale large originals to control vision-token cost (reconstructions are 512)
    w, h = image.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        image = image.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class GPT4oExtractor:
    def __init__(self, model: str = MODEL):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from env
        self.model = model
        self.max_objects = 50
        self.max_relations = 80
        self.total_in = 0
        self.total_out = 0

    def _postprocess(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Identical contract to VLMSceneGraphExtractor._postprocess_scene_graph."""
        objects, attributes, relations, seen = [], [], [], set()
        for idx, obj in enumerate(parsed.get("objects", []) or []):
            if not isinstance(obj, dict):
                continue
            name = normalize_label(obj.get("name", obj.get("label", "")))
            if not name or name in seen:
                continue
            seen.add(name)
            clean_attrs = []
            for attr in obj.get("attributes", []) or []:
                an = normalize_label(attr)
                if an and an not in clean_attrs:
                    clean_attrs.append(an)
                    attributes.append({"object": name, "attribute": an})
                if len(clean_attrs) >= 8:
                    break
            objects.append({"id": f"o{idx+1}", "label": name,
                            "attributes": clean_attrs, "confidence": 1.0})
            if len(objects) >= self.max_objects:
                break
        names = {o["label"] for o in objects}
        seen_rel = set()
        for rel in parsed.get("relations", []) or []:
            if not isinstance(rel, dict):
                continue
            s = normalize_label(rel.get("subject", ""))
            p = normalize_relation(rel.get("predicate", rel.get("relation", "")))
            o = normalize_label(rel.get("object", ""))
            if not s or not p or not o or s not in names or o not in names:
                continue
            if (s, p, o) in seen_rel:
                continue
            seen_rel.add((s, p, o))
            relations.append({"subject": s, "predicate": p, "object": o, "confidence": 1.0})
            if len(relations) >= self.max_relations:
                break
        return {"objects": objects, "attributes": attributes, "relations": relations,
                "tuples": build_tuple_set(objects, attributes, relations),
                "graph": build_networkx_graph(objects, attributes, relations)}

    def _relation_retry(self, kg: Dict[str, Any], data_url: str,
                        image_id: Optional[str], min_relations: int = 5) -> Dict[str, Any]:
        """
        Parity with Qwen's extract_with_relation_retry: if fewer than min_relations
        survived, ask once more (structured) for relations among the objects ALREADY
        found. Both endpoints must be existing object names; duplicates dropped. This
        gives GPT-4o the SAME two-pass extraction the Qwen evaluation used, so the
        evaluator comparison is procedurally fair.
        """
        if len(kg.get("relations", [])) >= min_relations or len(kg.get("objects", [])) < 2:
            return kg
        obj_names = [o["label"] for o in kg["objects"]]
        obj_list = "\n".join(f"  - {n}" for n in obj_names)
        prompt = (
            "Look at this image. Using ONLY the objects listed below, list the real "
            "spatial and functional relations between them. Do NOT invent objects "
            "not in the list. Do NOT repeat the same relation.\n\n"
            f"OBJECTS (use ONLY these exact names):\n{obj_list}\n\n"
            "Both subject and object must be EXACT matches from the list. No "
            "self-relations. Use specific predicates (stopped at, mounted on, next "
            "to, behind, in front of, parked at, facing, attached to, etc.)."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "rels", "schema": _REL_SCHEMA, "strict": True}},
                max_tokens=900, temperature=0.0,
            )
            self.total_in += resp.usage.prompt_tokens
            self.total_out += resp.usage.completion_tokens
            new_rels = json.loads(resp.choices[0].message.content).get("relations", []) or []
        except Exception as exc:
            print(f"[GPT-4o] relation retry failed ({image_id}): {exc}")
            return kg
        names = {o["label"] for o in kg["objects"]}
        have = {(normalize_label(r["subject"]),
                 normalize_relation(r.get("predicate", r.get("relation", ""))),
                 normalize_label(r["object"])) for r in kg["relations"]}
        for r in new_rels:
            if not isinstance(r, dict):
                continue
            s = normalize_label(r.get("subject", ""))
            p = normalize_relation(r.get("predicate", r.get("relation", "")))
            o = normalize_label(r.get("object", ""))
            if not s or not p or not o or s == o or s not in names or o not in names or (s, p, o) in have:
                continue
            kg["relations"].append({"subject": s, "predicate": p, "object": o, "confidence": 1.0})
            have.add((s, p, o))
        kg["tuples"] = build_tuple_set(kg["objects"], kg["attributes"], kg["relations"])
        kg["graph"] = build_networkx_graph(kg["objects"], kg["attributes"], kg["relations"])
        return kg

    def extract(self, image: Image.Image, image_id: Optional[str] = None,
                retries: int = 3) -> Dict[str, Any]:
        data_url = _img_to_data_url(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": _VLM_KG_PROMPT_TEMPLATE},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "scene_kg", "schema": _KG_SCHEMA, "strict": True},
                    },
                    max_tokens=2000,
                    temperature=0.0,
                )
                self.total_in += resp.usage.prompt_tokens
                self.total_out += resp.usage.completion_tokens
                parsed = json.loads(resp.choices[0].message.content)
                kg = self._postprocess(parsed)
                kg = self._relation_retry(kg, data_url, image_id)
                kg["image_id"] = image_id
                return kg
            except Exception as exc:
                last_exc = exc
                print(f"[GPT-4o] attempt {attempt+1} failed ({image_id}): {exc}")
                time.sleep(2 * (attempt + 1))
        print(f"[GPT-4o] all retries failed ({image_id}); returning empty KG")
        kg = self._postprocess({"objects": [], "relations": []})
        kg["image_id"] = image_id
        return kg

    def cost(self) -> float:
        return self.total_in * _PRICE_IN_PER_TOK + self.total_out * _PRICE_OUT_PER_TOK


def load_pairs(results_glob: str, max_samples: Optional[int]) -> List[Dict[str, str]]:
    pairs = []
    for jf in sorted(glob.glob(results_glob)):
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        orig = d.get("original_image_path"); recon = d.get("final_reconstruction_path")
        if orig and recon and os.path.exists(orig) and os.path.exists(recon):
            pairs.append({"image_id": d.get("image_id", Path(jf).stem),
                          "status": d.get("status", ""), "orig": orig, "recon": recon})
        if max_samples and len(pairs) >= max_samples:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--compressed-only", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: set OPENAI_API_KEY env var first."); return

    out = Path(args.output_dir)
    pairs = load_pairs(str(out / "results" / "*.json"), args.max_samples)
    if args.compressed_only:
        pairs = [p for p in pairs if p["status"] == "compressed"]
    print(f"[eval] {len(pairs)} pairs to score with GPT-4o ({args.model})")
    if not pairs:
        print("[eval] no pairs found."); return

    ex = GPT4oExtractor(args.model)
    rows = []
    per_dir = out / "results_gpt4o"; per_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {p['image_id']} (status={p['status']})")
        I = Image.open(p["orig"]); Istar = Image.open(p["recon"])
        kg_I = ex.extract(I, image_id=f"{p['image_id']}_I")
        kg_star = ex.extract(Istar, image_id=f"{p['image_id']}_Istar")
        cmp = compare_scene_graphs(kg_I, kg_star)
        row = {"image_id": p["image_id"], "status": p["status"], "evaluator": "GPT-4o",
               "R_obj": cmp["object_recall"], "R_attr": cmp["attribute_recall"], "R_rel": cmp["relation_recall"],
               "obj_f1": cmp["object_f1"], "attr_f1": cmp["attribute_f1"], "rel_f1": cmp["relation_f1"],
               "kg_f1_hard": cmp["weighted_kg_f1_hard"], "kg_f1_soft": cmp["weighted_kg_f1_soft"],
               "chair_i": cmp["chair_i"],
               "n_obj_I": len(kg_I["objects"]), "n_obj_Istar": len(kg_star["objects"]),
               "n_rel_I": len(kg_I["relations"]), "n_rel_Istar": len(kg_star["relations"])}
        rows.append(row)
        print(f"    R_obj={row['R_obj']:.3f} R_attr={row['R_attr']:.3f} R_rel={row['R_rel']:.3f} "
              f"KG-F1soft={row['kg_f1_soft']:.3f} CHAIR={row['chair_i']:.3f} "
              f"(objI={row['n_obj_I']} relI={row['n_rel_I']})")
        print(f"    [cost so far] in={ex.total_in} out={ex.total_out} tok  ~${ex.cost():.4f}")
        with open(per_dir / f"{p['image_id']}.json", "w", encoding="utf-8") as f:
            json.dump({"comparison_text": {"missing": cmp["missing_text"], "extra": cmp["extra_text"]}, **row},
                      f, indent=2)

    csv_path = out / "results" / "metrics_gpt4o.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n[eval] wrote {csv_path}")
    print(f"[eval] TOTAL cost: in={ex.total_in} out={ex.total_out} tok  ~${ex.cost():.4f}  "
          f"(model={args.model})")

    def agg(subset, label):
        if not subset: return
        m = lambda k: float(np.mean([r[k] for r in subset]))
        s = lambda k: float(np.std([r[k] for r in subset]))
        print(f"\n=== GPT-4o evaluator — {label} (n={len(subset)}) ===")
        for k in ("R_obj","R_attr","R_rel","kg_f1_soft","chair_i"):
            print(f"  {k:12s}: {m(k):.3f} ± {s(k):.3f}")
    agg(rows, "all scored")
    agg([r for r in rows if r["status"]=="compressed"], "compressed-only")
    print(f"\n[paper] Record evaluator model = {args.model}, run date = {time.strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()