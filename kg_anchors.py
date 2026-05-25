from __future__ import annotations

import re
import string
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional

import networkx as nx


# =====================================================================
# Type alias for readability
# =====================================================================

AnchorDetail = Dict[str, Any]
# {
#   "dino_max_score":  float,   # max DINO bbox confidence for this label
#   "dino_ok":         bool,    # Condition 1 pass/fail
#   "caption_ok":      bool,    # Condition 2 pass/fail
#   "degree":          int,     # node degree in KG relation graph
#   "centrality_ok":   bool,    # Condition 3 pass/fail
#   "is_anchor":       bool,    # True iff all three conditions pass
# }


# =====================================================================
# Internal text helpers (mirrors kg_batch_excel.py — kept local so
# this module has zero imports from kg_batch_excel.py)
# =====================================================================

def _clean(text: str) -> str:
    """Lowercase, strip articles, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _default_sim(a: str, b: str) -> float:
    """
    Lightweight lexical similarity for anchor coverage matching.
    Exact → 1.0 | substring → 0.92 | else SequenceMatcher ratio.

    The main pipeline uses the mpnet-backed _sim() from kg_batch_excel.py.
    Here we intentionally use lexical-only so this module stays self-contained
    and import-free from the parent script.  You can override via sim_fn= in
    anchor_coverage() if you want to pass the heavier hybrid function.
    """
    a, b = _clean(a), _clean(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return SequenceMatcher(None, a, b).ratio()


# =====================================================================
# Condition 1 — DINO Confidence
# =====================================================================

def _condition_dino(label: str,
                    dino_results: Dict[str, List[Dict[str, Any]]],
                    thresh: float) -> tuple[bool, float]:
    """
    Returns (passes, max_score).

    dino_results is the dict returned by GroundingDINOGrounder.ground():
      { label: [{"box": [...], "score": float}, ...], ... }

    We look up by exact label first; if missing we accept any key whose
    cleaned form matches the cleaned label (handles minor normalisation
    differences between VLM output and DINO query strings).
    """
    dets = dino_results.get(label, None)

    # Fallback: try cleaned key match
    if dets is None:
        cl = _clean(label)
        for k, v in dino_results.items():
            if _clean(k) == cl:
                dets = v
                break

    if not dets:
        return False, 0.0

    max_score = max((d.get("score", 0.0) for d in dets), default=0.0)
    return max_score >= thresh, round(max_score, 4)


# =====================================================================
# Condition 2 — Caption Salience
# =====================================================================

def _condition_caption(label: str, caption: str) -> bool:
    """
    Returns True if the cleaned label appears as a substring of the
    cleaned caption.

    Examples that pass:
      label="person",  caption="A person standing on a bridge."  → True
      label="bridge",  caption="A person standing on a bridge."  → True
      label="bicycle", caption="There is a red bicycle near the fence." → True

    Examples that fail:
      label="cloud",   caption="A sunny day at the beach."  → False
    """
    if not caption:
        return False
    cl = _clean(label)
    cc = _clean(caption)
    if not cl:
        return False
    return cl in cc


# =====================================================================
# Condition 3 — Graph Centrality (degree-based)
# =====================================================================

def _build_relation_graph(kg: Dict[str, Any]) -> nx.Graph:
    """
    Undirected graph where:
      nodes = object labels
      edges = relations (subject, object) pairs
    Attributes are NOT edges in this graph.
    """
    G = nx.Graph()
    for obj in kg.get("objects", []):
        G.add_node(obj["label"])
    for rel in kg.get("relations", []):
        s, o = rel.get("subject", ""), rel.get("object", "")
        if s in G and o in G and s != o:
            # MultiGraph would allow parallel edges; we use Graph so duplicate
            # (s,o) predicates count as one edge — degree reflects connectivity,
            # not predicate count.
            G.add_edge(s, o)
    return G


def _condition_centrality(label: str, G: nx.Graph, min_degree: int) -> tuple[bool, int]:
    """Returns (passes, degree)."""
    if label not in G:
        return False, 0
    degree = G.degree(label)
    return degree >= min_degree, degree


# =====================================================================
# Main public function: compute_anchors
# =====================================================================

def compute_anchors(
    kg_orig: Dict[str, Any],
    dino_results: Dict[str, List[Dict[str, Any]]],
    caption: str,
    dino_conf_thresh: float = 0.30,
    min_degree: int = 2,
) -> Dict[str, Any]:
    """
    Evaluate every object in kg_orig against the three anchor conditions.

    Parameters
    ----------
    kg_orig : dict
        Fused KG dict (output of fuse_vlm_and_dino) for the original image.
        Expected keys: "objects" (list of {label, ...}), "relations" (list of triples).
    dino_results : dict
        Output of GroundingDINOGrounder.ground() called on the original image.
        { label_str: [{"box": [...], "score": float}, ...] }
    caption : str
        Original human-written caption for this sample.
    dino_conf_thresh : float
        Minimum DINO bounding-box confidence to satisfy Condition 1. Default 0.30.
    min_degree : int
        Minimum relation-graph node degree to satisfy Condition 3. Default 2.

    Returns
    -------
    dict with keys:
      "anchors"  : List[str]  — labels of nodes that pass ALL three conditions
      "detail"   : Dict[str, AnchorDetail]  — per-node condition breakdown
      "n_anchors": int
      "n_objects": int
      "thresholds": dict  — the thresholds used (for logging / reproducibility)
    """
    G = _build_relation_graph(kg_orig)
    objects = kg_orig.get("objects", [])

    anchors: List[str] = []
    detail: Dict[str, AnchorDetail] = {}

    for obj in objects:
        label = obj.get("label", "")
        if not label:
            continue

        dino_ok, dino_score = _condition_dino(label, dino_results, dino_conf_thresh)
        cap_ok              = _condition_caption(label, caption)
        cent_ok, degree     = _condition_centrality(label, G, min_degree)

        is_anchor = dino_ok and cap_ok and cent_ok

        detail[label] = {
            "dino_max_score": dino_score,
            "dino_ok":        dino_ok,
            "caption_ok":     cap_ok,
            "degree":         degree,
            "centrality_ok":  cent_ok,
            "is_anchor":      is_anchor,
        }

        if is_anchor:
            anchors.append(label)

    return {
        "anchors":    anchors,
        "detail":     detail,
        "n_anchors":  len(anchors),
        "n_objects":  len(objects),
        "thresholds": {
            "dino_conf_thresh": dino_conf_thresh,
            "min_degree":       min_degree,
        },
    }


# =====================================================================
# Main public function: anchor_coverage
# =====================================================================

def anchor_coverage(
    anchors: List[str],
    kg_recon: Dict[str, Any],
    sim_fn: Optional[Callable[[str, str], float]] = None,
    match_thresh: float = 0.45,
) -> Dict[str, Any]:
    """
    Check how many critical anchors from the original KG survived in the
    reconstructed KG.

    Matching uses sim_fn (default: _default_sim, lexical-only).
    To use the full mpnet-backed hybrid from kg_batch_excel.py, pass:
        from kg_batch_excel import _sim
        anchor_coverage(anchors, kg_recon, sim_fn=_sim)

    A match is declared when the best similarity between the anchor label
    and ANY object label in kg_recon >= match_thresh.

    Parameters
    ----------
    anchors     : list of anchor label strings (from compute_anchors)
    kg_recon    : fused KG dict for the reconstructed image
    sim_fn      : similarity function (str, str) -> float in [0,1]
    match_thresh: threshold above which two labels are considered matched

    Returns
    -------
    dict with keys:
      "covered"       : List[str]  — anchors found in reconstructed KG
      "dropped"       : List[str]  — anchors absent from reconstructed KG
      "coverage_rate" : float      — len(covered) / max(len(anchors), 1)
      "n_anchors"     : int
      "n_covered"     : int
      "match_scores"  : Dict[str, float]  — best match score per anchor label
    """
    if sim_fn is None:
        sim_fn = _default_sim

    recon_labels = [obj["label"] for obj in kg_recon.get("objects", [])]

    covered: List[str] = []
    dropped: List[str] = []
    match_scores: Dict[str, float] = {}

    for anchor in anchors:
        if not recon_labels:
            best_score = 0.0
        else:
            best_score = max(sim_fn(anchor, rl) for rl in recon_labels)

        match_scores[anchor] = round(best_score, 4)

        if best_score >= match_thresh:
            covered.append(anchor)
        else:
            dropped.append(anchor)

    n_anchors  = len(anchors)
    n_covered  = len(covered)
    rate       = round(n_covered / max(n_anchors, 1), 4)

    return {
        "covered":       covered,
        "dropped":       dropped,
        "coverage_rate": rate,
        "n_anchors":     n_anchors,
        "n_covered":     n_covered,
        "match_scores":  match_scores,
    }


# =====================================================================
# Formatting helper (used by both Excel builder and terminal printout)
# =====================================================================

def format_anchor_summary(
    anchor_result: Dict[str, Any],
    coverage_result: Dict[str, Any],
) -> tuple[str, str]:
    """
    Returns two strings ready to drop into Excel cells or print to terminal:
      (anchor_cell_text, coverage_cell_text)

    anchor_cell_text  — goes in column R "Critical Anchors"
    coverage_cell_text — goes in column S "Anchor Coverage"
    """
    n_obj = anchor_result["n_objects"]
    n_anc = anchor_result["n_anchors"]
    detail = anchor_result["detail"]
    thresh = anchor_result["thresholds"]

    anchors_str = ", ".join(anchor_result["anchors"]) if anchor_result["anchors"] else "(none)"

    # Per-node condition breakdown for non-anchors (shows why they were excluded)
    near_misses = []
    for label, d in detail.items():
        if not d["is_anchor"]:
            fails = []
            if not d["dino_ok"]:
                fails.append(f"DINO={d['dino_max_score']:.2f}")
            if not d["caption_ok"]:
                fails.append("caption=✗")
            if not d["centrality_ok"]:
                fails.append(f"deg={d['degree']}")
            near_misses.append(f"  {label}: {', '.join(fails)}")

    near_miss_block = "\n".join(near_misses[:12])
    if len(near_misses) > 12:
        near_miss_block += f"\n  ...+{len(near_misses) - 12} more"

    anchor_text = (
        f"ANCHORS ({n_anc}/{n_obj} objects)\n"
        f"  thresholds: DINO>={thresh['dino_conf_thresh']}, degree>={thresh['min_degree']}\n\n"
        f"ANCHOR LABELS:\n  {anchors_str}\n\n"
        f"NON-ANCHOR REASONS:\n{near_miss_block}"
    )

    # Coverage cell
    cov = coverage_result
    covered_str = ", ".join(cov["covered"]) if cov["covered"] else "(none)"
    dropped_str = ", ".join(cov["dropped"]) if cov["dropped"] else "(none)"

    coverage_text = (
        f"COVERAGE: {cov['n_covered']}/{cov['n_anchors']} "
        f"({cov['coverage_rate']:.1%})\n\n"
        f"COVERED:\n  {covered_str}\n\n"
        f"DROPPED:\n  {dropped_str}"
    )

    return anchor_text, coverage_text