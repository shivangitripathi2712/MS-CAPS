import json
import re
import string
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from PIL import Image


# =====================================================================
# Configuration constants
# =====================================================================

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_THRESHOLD_OBJECT = 0.45
_THRESHOLD_ATTRIBUTE = 0.50
_THRESHOLD_RELATION_PREDICATE = 0.55

_WEIGHT_OBJECT = 0.5
_WEIGHT_ATTRIBUTE = 0.3
_WEIGHT_RELATION = 0.2


# =====================================================================
# VLM extraction prompt  (faithfulness-first; bounded to prevent degeneration)
# =====================================================================

_VLM_KG_PROMPT_TEMPLATE = """You are a meticulous visual scene analyzer. Extract a structured knowledge graph (KG) of what is ACTUALLY VISIBLE in this image.

Two goals matter equally:
  - SALIENCE: capture the most important visible entities (up to about 15), their key visible properties, and how they relate. Prioritize the prominent and distinctive over the trivial. You do NOT need to list every leaf or brick.
  - FAITHFULNESS: assert ONLY what you can actually see. NEVER invent objects, attributes, or relations to pad the output. If a property is not clearly visible, leave it out. Omitting a fact is correct; fabricating one is a failure.

HARD LIMITS (critical — exceeding these is a failure):
  - At most 15 objects. Choose the 15 most salient if more exist.
  - At most 6 attributes per object. Never repeat an attribute or a phrase.
  - Never output the same word or phrase more than twice in a row. If you find yourself listing many similar items, STOP and move on.
  - Keep the entire response under the token budget; a COMPLETE small JSON is far better than a TRUNCATED large one.

==========================================================
OUTPUT FORMAT
==========================================================
Return ONLY one valid JSON object with EXACTLY this schema:

{{
  "objects": [
    {{"id": "o1", "name": "<short specific object name>", "attributes": ["<visible attr>", "..."]}}
  ],
  "relations": [
    {{"subject": "<object name>", "predicate": "<relation>", "object": "<object name>"}}
  ]
}}

NO markdown fences. NO commentary. Output the JSON object only.

==========================================================
1. OBJECTS — every distinct visible entity
==========================================================
Include foreground AND background entities: people, animals, vehicles, vehicle
parts, buildings and building parts, signs, banners, poles, road, lane markings,
traffic lights, the sky, trees, and so on.

ALSO include, when actually present:
  - Text-bearing items as objects when the text identifies them (e.g. a shop
    sign, a banner, a license plate, a brand name on a vehicle).
  - FRAMING DEVICES the photo is shot through: rearview mirror, side mirror,
    windshield, window frame, fence — if visible, list it as an object.
  - REFLECTED / MIRRORED CONTENT: anything visible *inside* a mirror or
    reflection is still a real visible object. List it (e.g. an "ambulance"
    seen in a "rearview mirror" is BOTH an object) and connect them with a
    relation in section 3.

Naming rules:
  - Specific, not generic: "white pickup truck" not "vehicle"; "rearview mirror"
    not "mirror"; "boston market sign" not "sign".
  - 1–4 words, lowercase, singular when possible.
  - Each distinct name appears once; if several identical items exist, list once
    and record the count as an attribute (e.g. "three").

NOT objects: abstractions ("scene", "background", "atmosphere", "view"),
actions/events ("driving", "sunset" — the sun is an object, "sunset" is not),
the camera/photographer/viewer.

==========================================================
2. ATTRIBUTES — intrinsic, single-object, VISIBLE properties
==========================================================
List the properties you can actually see for each object. Use any that apply:
  - appearance: color, brightness, pattern, texture, finish
  - form: shape, size, proportions
  - material: what it is made of
  - state/condition: open/closed, lit/unlit, wet/dry, stopped/moving, on/off,
    red/green (traffic light), parked, damaged, dusty
  - POSE / ORIENTATION (important for vehicles and people): "facing left",
    "facing the camera", "angled", "tilted", "parked diagonally", "rear view",
    "front view", "side-on"
  - count: when identical items were collapsed to one entry

Use specific VALUES, not category names:
  correct: {{"name": "car", "attributes": ["red", "compact", "stopped", "side-on"]}}
  wrong:   {{"name": "car", "attributes": ["color", "size", "state"]}}

FORBIDDEN in attributes (these are RELATIONS, put them in section 3):
  - any spatial preposition (above, below, on, under, in, behind, in front of,
    next to, near, between, inside, against, across, through)
  - any phrase that names or refers to ANOTHER object
  - any phrase describing what the object does TO another object
  - subjective judgments ("beautiful", "dramatic")

Only list attributes you can see. Do not pad. An object with one clearly visible
attribute should have one attribute, not an invented second.

==========================================================
3. RELATIONS — spatial and functional connections (the most important ones)
==========================================================
A relation is a triple (subject, predicate, object) where BOTH subject and
object are names that appear in your objects list.

PREDICATE VOCABULARY — use the most specific that fits:
  Spatial:    above, below, on, under, in, behind, in front of, next to,
              left of, right of, alongside, across from, near, between,
              on top of, overhead of, in background of, surrounding
  Functional: stopped at, moving through, traveling along, waiting at, crossing,
              parked at, facing, overlooking, illuminating, mounted on,
              attached to, written on, painted on, adjacent to, carrying
  Framing /   viewed through, reflected in, visible in, framed by
  reflection:

SYSTEMATIC METHOD — work through these to FIND the most important relations
(aim for the salient ones, not an exhaustive list; do it carefully, not by guessing):
  STEP 1 — Each VEHICLE: its action (stopped at / moving through / traveling
           along / parked at) AND its position relative to road/another object.
  STEP 2 — Each SIGN / BUILDING / STOREFRONT: how it sits in the scene
           (mounted on a pole, behind the road, adjacent to the intersection).
  STEP 3 — Each TRAFFIC LIGHT / LAMP / OVERHEAD FIXTURE: what it is above or
           mounted on.
  STEP 4 — SKY / CLOUDS: what they are above.
  STEP 5 — FRAMING / REFLECTION: if the photo is shot through a mirror, window,
           or windshield, add (main subject, viewed through, <device>). If an
           object appears inside a mirror/reflection, add
           (object, reflected in, <mirror>) or (object, visible in, <mirror>).
  STEP 6 — For every remaining pair that is clearly close or connected, ask
           "where is A relative to B?" and "is A doing something to B?" and add
           the relation.

Rules:
  - Both subject and object must be EXACT character-for-character matches of
    names in your objects list. Do not invent endpoints.
  - No self-relations. No vague predicates ("relates to", "complements").
  - Add a relation only if it is actually true in the image. Thoroughness means
    finding real relations, not manufacturing plausible ones.

==========================================================
4. CONSISTENCY
==========================================================
  - Names lowercase and identical between objects and relations.
  - Extract only what is verifiably visible. No interpretation, no guessing about
    occluded or off-frame content.
  - Prefer specific over generic when identifiable.

Output a single JSON object. No fences. No commentary."""


# Scene-context probe — returns METADATA only (never merged into the KG).
_VLM_SCENE_CONTEXT_PROMPT = """Describe this image's OVERALL SCENE STRUCTURE (not individual foreground objects).
Answer all six with a SHORT noun phrase (1-4 words), not a sentence.

1. POV — from where is the photo taken? (e.g. "viewed through rearview mirror",
   "from inside a car", "eye-level street view", "drone shot")
2. SETTING — type of place (e.g. "urban intersection", "rural roadside",
   "parking lot", "residential street", "highway")
3. GROUND — visible surface (e.g. "asphalt road", "dirt road", "grass field")
4. LIGHTING — lighting / time of day (e.g. "bright sunny day", "overcast",
   "evening light", "after rain")
5. FRAMING — framed THROUGH something visible? name it, else "none"
   (e.g. "rearview mirror frame", "car windshield", "none")
6. BACKGROUND — what dominates the background (e.g. "dense trees",
   "distant buildings", "open field")

Return ONLY this JSON object — no commentary, no markdown:
{
  "pov": "...", "setting": "...", "ground": "...",
  "lighting": "...", "framing": "...", "background": "..."
}"""


# Visible-text probe — for strict anchor harvesting from the image itself.
_VLM_VISIBLE_TEXT_PROMPT = """List ALL readable text visible in this image.
Include: brand names on vehicles (FORCE, TRAVELLER, TATA), license/number plates
(KL11BK4477), shop/building/banner signs, and any numbers or words painted on
roads, walls, or vehicles. Preserve capitalization.

For each piece of text return "text" (the exact string) and "location" (a short
phrase saying where it appears). If no text is readable, return an empty list.

Return ONLY this JSON object — no commentary, no markdown:
{
  "visible_text": [ {"text": "...", "location": "..."} ]
}"""


# =====================================================================
# Device / dtype helpers
# =====================================================================

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[SceneGraph] CUDA requested but not available. Using CPU.")
            return "cpu"
        if ":" in device:
            gpu_id = int(device.split(":")[-1])
            if gpu_id >= torch.cuda.device_count():
                print(
                    f"[SceneGraph] Requested {device}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) found. Using cuda:0."
                )
                return "cuda:0"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    dtype = dtype.lower()
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def _gpu_index(device: str) -> int:
    if device.startswith("cuda") and ":" in device:
        return int(device.split(":")[-1])
    return 0


# =====================================================================
# Text normalization
# =====================================================================

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> str:
    return clean_text(label)


def normalize_relation(rel: str) -> str:
    rel = clean_text(rel)
    surface_aliases = {
        "holds": "holding", "wears": "wearing",
        "carrying": "holding", "has on": "wearing",
    }
    return surface_aliases.get(rel, rel)


def canonicalize_for_matching(text: str) -> str:
    return clean_text(text)


def _repair_json(text: str) -> str:
    """
    Best-effort repair of a possibly-truncated JSON object: balance braces and
    brackets so a generation cut off by max_new_tokens can still parse. Drops a
    trailing incomplete element where possible.
    """
    start = text.find("{")
    if start == -1:
        return text
    text = text[start:]
    # Trim to the last plausible value boundary, then close open structures.
    last = max(text.rfind("}"), text.rfind("]"), text.rfind('"'))
    if last != -1:
        text = text[: last + 1]
    opens = text.count("{") - text.count("}")
    brs = text.count("[") - text.count("]")
    text = text + ("]" * max(brs, 0)) + ("}" * max(opens, 0))
    return text


def _matched_brace_spans(text: str) -> List[Tuple[int, int]]:
    """
    (start, end) index pairs for every balanced ``{...}`` block, respecting JSON
    string literals (braces/quotes inside strings are ignored). A block truncated
    mid-way is never closed, so it is simply absent — exactly what we want to drop.
    """
    spans: List[Tuple[int, int]] = []
    stack: List[int] = []
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                spans.append((stack.pop(), i))
    return spans


def _salvage_records(text: str) -> Dict[str, Any]:
    """
    Record-level salvage for truncated/malformed VLM JSON. Extracts every balanced
    LEAF ``{...}`` block (no nested object), parses each independently, and
    classifies it as a relation / object / visible-text record. Returns whichever
    of objects / relations / visible_text were recovered — so a response cut off
    mid-generation still yields the records that completed, instead of an empty KG.
    """
    spans = _matched_brace_spans(text)
    leaves = [(s, e) for (s, e) in spans
              if not any(s < s2 and e2 < e for (s2, e2) in spans)]

    objects: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []

    for (s, e) in leaves:
        block = text[s:e + 1]
        rec = None
        for cand in (block,
                     re.sub(r",\s*}", "}", block),
                     re.sub(r",\s*$", "", block.strip())):
            try:
                rec = json.loads(cand)
                break
            except Exception:
                continue
        if not isinstance(rec, dict):
            continue
        if "subject" in rec and "object" in rec and ("predicate" in rec or "relation" in rec):
            relations.append(rec)
        elif "name" in rec or "label" in rec:
            objects.append(rec)
        elif "text" in rec:
            texts.append(rec)

    out: Dict[str, Any] = {}
    if objects:
        out["objects"] = objects
    if relations:
        out["relations"] = relations
    if texts:
        out["visible_text"] = texts
    if out:
        return out

    # Last-ditch fallback: no balanced leaf survived (e.g. a single runaway object
    # whose attribute list looped forever and never closed). Recover object names
    # and relation triples by field regex so we still get a non-empty KG skeleton.
    obj_names = re.findall(r'"(?:name|label)"\s*:\s*"((?:[^"\\]|\\.){1,60})"', text)
    for nm in obj_names:
        nm = nm.strip()
        if nm:
            objects.append({"name": nm})
    rel_triples = re.findall(
        r'"subject"\s*:\s*"([^"]{1,60})"\s*,\s*"(?:predicate|relation)"\s*:\s*"([^"]{1,40})"\s*,\s*"object"\s*:\s*"([^"]{1,60})"',
        text,
    )
    for s, p, o in rel_triples:
        relations.append({"subject": s, "predicate": p, "object": o})
    txt_vals = re.findall(r'"text"\s*:\s*"((?:[^"\\]|\\.){1,80})"', text)
    for t in txt_vals:
        texts.append({"text": t})

    out = {}
    if objects:
        out["objects"] = objects
    if relations:
        out["relations"] = relations
    if texts:
        out["visible_text"] = texts
    return out


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse VLM output into a dict. Tiered recovery:
      1. direct json.loads
      2. first {...} span via regex
      3. brace-balanced repair of a truncated object
      4. record-level salvage (partial objects/relations/text survive truncation)
    Raises ValueError only if nothing parseable remains.
    """
    if not isinstance(text, str):
        raise ValueError("VLM output must be a string.")
    text = text.strip()
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    try:
        return json.loads(_repair_json(text))
    except Exception:
        pass
    salvaged = _salvage_records(text)
    if salvaged.get("objects") or salvaged.get("relations") or salvaged.get("visible_text"):
        print(f"[SceneGraph] JSON truncated; salvaged "
              f"{len(salvaged.get('objects', []))} object(s), "
              f"{len(salvaged.get('relations', []))} relation(s), "
              f"{len(salvaged.get('visible_text', []))} text record(s).")
        return salvaged
    raise ValueError("Could not parse or salvage any JSON from VLM output.")


# =====================================================================
# Semantic similarity (mpnet + char-ratio hybrid)
# =====================================================================

_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(_EMBEDDING_MODEL_NAME, device="cpu")
        print(f"[KG Compare] Loaded {_EMBEDDING_MODEL_NAME} for semantic KG matching.")
        return _EMBEDDER
    except Exception as exc:
        print(f"[KG Compare] sentence-transformers unavailable; char fallback. Error: {exc}")
        _EMBEDDER = False
        return None


@lru_cache(maxsize=20000)
def _embed_text(text: str):
    model = _get_embedder()
    if model is None or model is False:
        return None
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return tuple(float(x) for x in vec)


def semantic_text_similarity(a: str, b: str) -> float:
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    char_sim = SequenceMatcher(None, a, b).ratio()
    emb_a, emb_b = _embed_text(a), _embed_text(b)
    if emb_a is None or emb_b is None:
        return char_sim
    va, vb = np.array(emb_a, dtype=np.float32), np.array(emb_b, dtype=np.float32)
    cos = max(0.0, min(1.0, float(np.dot(va, vb))))
    return 0.75 * cos + 0.25 * char_sim


_sim = semantic_text_similarity


# =====================================================================
# VLM Scene Graph Extractor
# =====================================================================

class VLMSceneGraphExtractor:
    """Extracts a visual KG from an image using a vision-language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 1280,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_objects: int = 50,
        max_relations: int = 80,
        min_pixels: int = 256 * 28 * 28,    # ~200K px floor
        max_pixels: int = 1280 * 28 * 28,   # ~1M px cap (~1280 vision tokens)
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype, self.device)
        self.max_new_tokens = int(max_new_tokens)
        # Cap vision tokens so high-res images don't blow up VLM attention.
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        # Greedy/deterministic extraction so KG_I is stable across runs.
        self.temperature = max(float(temperature), 1e-4)
        self.do_sample = bool(do_sample)
        self.max_objects = int(max_objects)
        self.max_relations = int(max_relations)
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"[SceneGraph] Loading VLM='{self.model_name}' on '{self.device}', dtype='{self.torch_dtype}'")
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except Exception:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels,
        )
        if self.device.startswith("cuda"):
            gpu = _gpu_index(self.device)
            self.model = model_cls.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype, device_map={"": gpu},
            )
        else:
            self.model = model_cls.from_pretrained(
                self.model_name, torch_dtype=self.torch_dtype,
            ).to(self.device)
        self.model.eval()
        print("[SceneGraph] VLM loaded successfully.")

    def _build_prompt(self) -> str:
        return _VLM_KG_PROMPT_TEMPLATE

    # ----------------------------------------------------------------- #
    # Internal VLM runner (shared by all probes)
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def _run_vlm_with_prompt(self, image: Image.Image, prompt: str,
                             max_new_tokens: Optional[int] = None) -> str:
        from qwen_vl_utils import process_vision_info
        if image.mode != "RGB":
            image = image.convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.device if self.device.startswith("cuda") else "cpu")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": self.do_sample,
            # Anti-degeneration: a SOFT repetition penalty discourages the runaway
            # phrase loop ("mirror, windshield, fence, ...") without forbidding the
            # structural 3-grams that valid JSON must repeat (`", "name": "`, etc.).
            # NOTE: no_repeat_ngram_size is deliberately NOT used here — it makes
            # well-formed JSON impossible to emit. The salvage parser is the net for
            # any loop that still slips through.
            "repetition_penalty": 1.05,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()

    def _run_vlm(self, image: Image.Image) -> str:
        return self._run_vlm_with_prompt(image, self._build_prompt())

    @torch.no_grad()
    def extract(self, image: Image.Image, image_id: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image.")
        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_output = self._run_vlm(image)
        try:
            parsed = safe_json_loads(raw_output)
        except Exception as exc:
            print(f"[SceneGraph] JSON parse failed: {exc}")
            parsed = {"objects": [], "relations": []}
        kg = self._postprocess_scene_graph(parsed)

        # Retry-on-empty guard: a degenerate (looping) main pass can yield an
        # empty/near-empty KG_I, which poisons every downstream metric. If we got
        # <2 objects, retry ONCE with a stricter, shorter prompt at a lower token
        # budget — a tighter budget usually completes before any loop overflows.
        if len(kg.get("objects", [])) < 2:
            print(f"[SceneGraph] KG has {len(kg.get('objects', []))} object(s); "
                  f"retrying once with stricter concise prompt.")
            concise_prompt = self._build_prompt() + (
                "\n\nIMPORTANT: Be concise. List only the 10 most salient objects, "
                "at most 4 attributes each. Output a COMPLETE, valid, SMALL JSON "
                "object. Do not repeat phrases. Close all braces."
            )
            raw_retry = self._run_vlm_with_prompt(image, concise_prompt, max_new_tokens=896)
            try:
                parsed_retry = safe_json_loads(raw_retry)
                kg_retry = self._postprocess_scene_graph(parsed_retry)
                if len(kg_retry.get("objects", [])) > len(kg.get("objects", [])):
                    kg = kg_retry
                    raw_output = raw_retry
                    print(f"[SceneGraph] Retry succeeded: {len(kg['objects'])} objects.")
            except Exception as exc:
                print(f"[SceneGraph] Retry parse failed: {exc}; keeping original.")

        kg["image_id"] = image_id
        kg["raw_vlm_output"] = raw_output
        kg["scene_context"] = {}
        return kg

    @torch.no_grad()
    def extract_with_relation_retry(
        self, image: Image.Image, image_id: Optional[str] = None,
        min_relations: int = 5, max_retries: int = 1,
    ) -> Dict[str, Any]:
        kg = self.extract(image, image_id=image_id)
        current_rels = kg.get("relations", [])
        if len(current_rels) >= min_relations or max_retries <= 0:
            return kg

        print(f"[SceneGraph] Only {len(current_rels)} relations; running relation retry...")
        obj_names = [o.get("label", o.get("name", "")) for o in kg.get("objects", [])]
        obj_list = "\n".join(f"  - {n}" for n in obj_names if n)
        existing_rels = "\n".join(
            f"  ({r.get('subject','')}, {r.get('predicate', r.get('relation',''))}, {r.get('object','')})"
            for r in current_rels
        ) or "  (none yet)"

        need = max(min_relations - len(current_rels), 1)
        retry_prompt = f"""You previously extracted a KG for this image, but more real relations exist
between the objects already found. Look again and add ONLY relations that are
actually true in the image — do not invent.

OBJECTS (use ONLY these exact names):
{obj_list}

RELATIONS already found:
{existing_rels}

Find at least {need} ADDITIONAL true relations. For each pair ask: where is A
relative to B? is A connected to B (mounted on / attached to / part of)? is A
doing something to B (stopped at / moving through / facing / illuminating)? is A
framing or reflecting B (A viewed through B, A reflected in B)?

PREDICATES — spatial: above, below, on, under, in, behind, in front of, next to,
near, left of, right of, alongside, at, overhead of, in background of;
functional: stopped at, moving through, traveling along, facing, mounted on,
attached to, illuminating, crossing, adjacent to, parked at;
framing/reflection: viewed through, reflected in, visible in, framed by.

Return ONLY this JSON (NO objects field):
{{ "relations": [ {{"subject": "<name>", "predicate": "<rel>", "object": "<name>"}} ] }}

Both endpoints must be EXACT matches from the objects list. No self-relations.
Output only NEW relations. JSON only, no fences."""

        raw = self._run_vlm_with_prompt(image, retry_prompt, max_new_tokens=1024)
        try:
            new_rels_raw = safe_json_loads(raw).get("relations", []) or []
        except Exception as exc:
            print(f"[SceneGraph] Relation retry parse failed: {exc}")
            return kg

        names_set = {normalize_label(n) for n in obj_names if n}
        existing_keys = {
            (normalize_label(r.get("subject", "")),
             normalize_relation(r.get("predicate", r.get("relation", ""))),
             normalize_label(r.get("object", "")))
            for r in current_rels
        }
        added = 0
        for r in new_rels_raw:
            if not isinstance(r, dict):
                continue
            s = normalize_label(r.get("subject", ""))
            p = normalize_relation(r.get("predicate", r.get("relation", "")))
            o = normalize_label(r.get("object", ""))
            if not s or not p or not o or s == o:
                continue
            if s not in names_set or o not in names_set or (s, p, o) in existing_keys:
                continue
            kg["relations"].append({"subject": s, "predicate": p, "object": o, "confidence": 1.0})
            existing_keys.add((s, p, o))
            added += 1
            if len(kg["relations"]) >= self.max_relations:
                break

        kg["tuples"] = build_tuple_set(kg.get("objects", []), kg.get("attributes", []), kg.get("relations", []))
        kg["graph"] = build_networkx_graph(kg.get("objects", []), kg.get("attributes", []), kg.get("relations", []))
        print(f"[SceneGraph] Relation retry added {added}. Total now: {len(kg['relations'])}")
        return kg

    # ----------------------------------------------------------------- #
    # Scene context — METADATA ONLY (never merged into the KG tuples)
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def extract_scene_context(self, image: Image.Image) -> Dict[str, str]:
        """Return {pov, setting, ground, lighting, framing, background} or {}."""
        try:
            parsed = safe_json_loads(
                self._run_vlm_with_prompt(image, _VLM_SCENE_CONTEXT_PROMPT, max_new_tokens=300)
            )
            return {k: str(parsed.get(k, "")).strip()
                    for k in ("pov", "setting", "ground", "lighting", "framing", "background")}
        except Exception as exc:
            print(f"[SceneGraph] Scene context probe failed: {exc}")
            return {}

    @torch.no_grad()
    def extract_with_scene_context(
        self, image: Image.Image, image_id: Optional[str] = None,
        min_relations: int = 8, max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Full extraction for the pipeline:
          1) schema-strict comprehensive pass
          2) relation retry if sparse
          3) scene-context probe stored as kg['scene_context'] METADATA only

        Scene context is NOT merged into objects/attributes/relations, so it does
        not affect Q_KG / CHAIR. Visible framing devices and reflected content are
        already captured as real objects/relations by the main prompt.
        """
        kg = self.extract_with_relation_retry(
            image, image_id=image_id, min_relations=min_relations, max_retries=max_retries,
        )
        kg["scene_context"] = self.extract_scene_context(image)
        return kg

    @torch.no_grad()
    def extract_visible_text(self, image: Image.Image) -> List[Dict[str, str]]:
        """Return [{text, location}, ...] for readable in-image text (strict-anchor source)."""
        try:
            parsed = safe_json_loads(
                self._run_vlm_with_prompt(image, _VLM_VISIBLE_TEXT_PROMPT, max_new_tokens=400)
            )
            out = []
            for e in parsed.get("visible_text", []) or []:
                if not isinstance(e, dict):
                    continue
                t = str(e.get("text", "")).strip()
                loc = str(e.get("location", "")).strip()
                if t and 1 <= len(t) <= 80:
                    out.append({"text": t, "location": loc})
            return out
        except Exception as exc:
            print(f"[SceneGraph] Visible text probe failed: {exc}")
            return []

    def _postprocess_scene_graph(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        raw_objects = parsed.get("objects", []) or []
        raw_relations = parsed.get("relations", []) or []
        objects: List[Dict[str, Any]] = []
        attributes: List[Dict[str, str]] = []
        relations: List[Dict[str, str]] = []
        seen_objects: Set[str] = set()

        for idx, obj in enumerate(raw_objects):
            if not isinstance(obj, dict):
                continue
            name = normalize_label(obj.get("name", obj.get("label", "")))
            if not name or name in seen_objects:
                continue
            seen_objects.add(name)
            clean_attrs: List[str] = []
            # Cap attributes per object. A degenerate response can dump dozens of
            # junk/visible-text strings onto one object; a real object rarely has
            # more than a handful of genuinely visible properties.
            _MAX_ATTRS_PER_OBJECT = 8
            for attr in obj.get("attributes", []) or []:
                attr_norm = normalize_label(attr)
                if attr_norm and attr_norm not in clean_attrs:
                    clean_attrs.append(attr_norm)
                    attributes.append({"object": name, "attribute": attr_norm})
                if len(clean_attrs) >= _MAX_ATTRS_PER_OBJECT:
                    break
            objects.append({
                "id": str(obj.get("id", f"o{idx + 1}")),
                "label": name, "attributes": clean_attrs, "confidence": 1.0,
            })
            if len(objects) >= self.max_objects:
                break

        object_names = {o["label"] for o in objects}
        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue
            s = normalize_label(rel.get("subject", ""))
            p = normalize_relation(rel.get("predicate", rel.get("relation", "")))
            o = normalize_label(rel.get("object", ""))
            if not s or not p or not o:
                continue
            if s not in object_names or o not in object_names:
                continue
            relations.append({"subject": s, "predicate": p, "object": o, "confidence": 1.0})
            if len(relations) >= self.max_relations:
                break

        return {
            "objects": objects,
            "attributes": attributes,
            "relations": relations,
            "tuples": build_tuple_set(objects, attributes, relations),
            "graph": build_networkx_graph(objects, attributes, relations),
        }


# =====================================================================
# KG construction helpers
# =====================================================================

def build_tuple_set(objects, attributes, relations) -> Set[Tuple[str, ...]]:
    tuples: Set[Tuple[str, ...]] = set()
    for obj in objects:
        label = normalize_label(obj.get("label", obj.get("name", "")))
        if label:
            tuples.add(("object", label))
    for attr in attributes:
        o = normalize_label(attr.get("object", ""))
        a = normalize_label(attr.get("attribute", ""))
        if o and a:
            tuples.add(("attribute", o, a))
    for rel in relations:
        s = normalize_label(rel.get("subject", ""))
        p = normalize_relation(rel.get("predicate", rel.get("relation", "")))
        o = normalize_label(rel.get("object", ""))
        if s and p and o:
            tuples.add(("relation", s, p, o))
    return tuples


def build_networkx_graph(objects, attributes, relations) -> nx.DiGraph:
    graph = nx.DiGraph()
    for obj in objects:
        label = normalize_label(obj.get("label", obj.get("name", "")))
        if label:
            graph.add_node(label, node_type="object")
    for attr in attributes:
        o = normalize_label(attr.get("object", ""))
        a = normalize_label(attr.get("attribute", ""))
        if o and a:
            node = f"{o}::{a}"
            graph.add_node(node, node_type="attribute", label=a)
            graph.add_edge(o, node, relation="has_attribute")
    for rel in relations:
        s = normalize_label(rel.get("subject", ""))
        p = normalize_relation(rel.get("predicate", rel.get("relation", "")))
        o = normalize_label(rel.get("object", ""))
        if s and p and o:
            graph.add_node(s, node_type="object")
            graph.add_node(o, node_type="object")
            graph.add_edge(s, o, relation=p)
    return graph


# =====================================================================
# KG comparison  (math + output schema UNCHANGED — downstream depends on it)
# =====================================================================

def _structural_metrics(kg1, kg2) -> Dict[str, Any]:
    def _to_graph(kg):
        G = nx.Graph()
        for o in kg.get("objects", []):
            G.add_node(o.get("label", o.get("name", "")))
        for r in kg.get("relations", []):
            s, o = r.get("subject", ""), r.get("object", "")
            if s in G and o in G:
                G.add_edge(s, o)
        return G

    G1, G2 = _to_graph(kg1), _to_graph(kg2)

    def _avg_degree(G):
        return round(sum(dict(G.degree()).values()) / len(G), 3) if len(G) else 0.0

    return {
        "orig_n_nodes": G1.number_of_nodes(), "recon_n_nodes": G2.number_of_nodes(),
        "orig_n_edges": G1.number_of_edges(), "recon_n_edges": G2.number_of_edges(),
        "orig_avg_degree": _avg_degree(G1), "recon_avg_degree": _avg_degree(G2),
        "orig_n_components": nx.number_connected_components(G1) if len(G1) else 0,
        "recon_n_components": nx.number_connected_components(G2) if len(G2) else 0,
    }


def _prf_hard(supp_count: int, miss_count: int, contra_count: int) -> Dict[str, float]:
    p = supp_count / max(supp_count + contra_count, 1)
    r = supp_count / max(supp_count + miss_count, 1)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def compare_scene_graphs(kg_original: Dict[str, Any], kg_reconstructed: Dict[str, Any]) -> Dict[str, Any]:
    objs1 = [o.get("label", o.get("name", "")) for o in kg_original.get("objects", [])]
    objs2 = [o.get("label", o.get("name", "")) for o in kg_reconstructed.get("objects", [])]

    obj_pairs: List[Tuple[str, str, float]] = []
    used2: Set[int] = set()
    for o1 in objs1:
        best, best_s, best_i = None, 0.0, -1
        for i2, o2 in enumerate(objs2):
            if i2 in used2:
                continue
            s = _sim(o1, o2)
            if s > best_s:
                best, best_s, best_i = o2, s, i2
        if best is not None and best_s >= _THRESHOLD_OBJECT:
            obj_pairs.append((o1, best, best_s))
            used2.add(best_i)

    obj_match: Dict[str, str] = {p[0]: p[1] for p in obj_pairs}
    obj_supported = sorted(obj_match.keys())
    obj_unsupported = sorted(set(objs1) - set(obj_supported))
    obj_contradicted = sorted([o for i, o in enumerate(objs2) if i not in used2])

    attrs1: Dict[str, Set[str]] = defaultdict(set)
    for a in kg_original.get("attributes", []):
        attrs1[a.get("object", "")].add(a.get("attribute", ""))
    attrs2: Dict[str, Set[str]] = defaultdict(set)
    for a in kg_reconstructed.get("attributes", []):
        attrs2[a.get("object", "")].add(a.get("attribute", ""))

    attr_supported: List[Tuple[str, str]] = []
    attr_unsupported: List[Tuple[str, str]] = []
    attr_contradicted: List[Tuple[str, str]] = []

    for o_orig, o_recon in obj_match.items():
        for a in attrs1[o_orig]:
            best_s = max((_sim(a, b) for b in attrs2[o_recon]), default=0.0)
            (attr_supported if best_s >= _THRESHOLD_ATTRIBUTE else attr_unsupported).append((o_orig, a))
        for b in attrs2[o_recon]:
            best_s = max((_sim(a, b) for a in attrs1[o_orig]), default=0.0)
            if best_s < _THRESHOLD_ATTRIBUTE:
                attr_contradicted.append((o_recon, b))

    for o in obj_unsupported:
        for a in attrs1[o]:
            attr_unsupported.append((o, a))
    for o in obj_contradicted:
        for b in attrs2[o]:
            attr_contradicted.append((o, b))

    rels1 = kg_original.get("relations", [])
    rels2 = kg_reconstructed.get("relations", [])
    rel_supported: List[Tuple[str, str, str]] = []
    rel_unsupported: List[Tuple[str, str, str]] = []
    rel_contradicted: List[Tuple[str, str, str]] = []
    used_rel2: Set[int] = set()

    for r1 in rels1:
        s1 = r1.get("subject", "")
        p1 = r1.get("predicate", r1.get("relation", ""))
        o1 = r1.get("object", "")
        if s1 not in obj_match or o1 not in obj_match:
            rel_unsupported.append((s1, p1, o1))
            continue
        s1m, o1m = obj_match[s1], obj_match[o1]
        best_idx, best_psim = -1, 0.0
        for i2, r2 in enumerate(rels2):
            if i2 in used_rel2:
                continue
            r2s = r2.get("subject", "")
            r2p = r2.get("predicate", r2.get("relation", ""))
            r2o = r2.get("object", "")
            if (s1m == r2s and o1m == r2o) or (s1m == r2o and o1m == r2s):
                p_sim = _sim(p1, r2p)
                if p_sim >= _THRESHOLD_RELATION_PREDICATE and p_sim > best_psim:
                    best_idx, best_psim = i2, p_sim
        if best_idx >= 0:
            rel_supported.append((s1, p1, o1))
            used_rel2.add(best_idx)
        else:
            rel_unsupported.append((s1, p1, o1))

    for i2, r2 in enumerate(rels2):
        if i2 not in used_rel2:
            rel_contradicted.append((r2.get("subject", ""), r2.get("predicate", r2.get("relation", "")), r2.get("object", "")))

    obj_hard = _prf_hard(len(obj_supported), len(obj_unsupported), len(obj_contradicted))
    attr_hard = _prf_hard(len(attr_supported), len(attr_unsupported), len(attr_contradicted))
    rel_hard = _prf_hard(len(rel_supported), len(rel_unsupported), len(rel_contradicted))

    def _soft_objects() -> Dict[str, float]:
        sr = float(np.mean([max((_sim(o1, o2) for o2 in objs2), default=0.0) for o1 in objs1])) if objs1 else 0.0
        sp = float(np.mean([max((_sim(o2, o1) for o1 in objs1), default=0.0) for o2 in objs2])) if objs2 else 0.0
        sf1 = (2 * sp * sr / (sp + sr)) if (sp + sr) > 0 else 0.0
        return {"precision": sp, "recall": sr, "f1": sf1}

    def _soft_attributes() -> Dict[str, float]:
        all1 = [(o, a) for o, attrs in attrs1.items() for a in attrs]
        all2 = [(o, a) for o, attrs in attrs2.items() for a in attrs]

        def pair_sim(p1, p2):
            return 0.5 * _sim(p1[0], p2[0]) + 0.5 * _sim(p1[1], p2[1])

        sr = float(np.mean([max((pair_sim(p1, p2) for p2 in all2), default=0.0) for p1 in all1])) if all1 else 0.0
        sp = float(np.mean([max((pair_sim(p2, p1) for p1 in all1), default=0.0) for p2 in all2])) if all2 else 0.0
        sf1 = (2 * sp * sr / (sp + sr)) if (sp + sr) > 0 else 0.0
        return {"precision": sp, "recall": sr, "f1": sf1}

    def _soft_relations() -> Dict[str, float]:
        def rel_sim(r1, r2):
            r1p = r1.get("predicate", r1.get("relation", ""))
            r2p = r2.get("predicate", r2.get("relation", ""))
            return (0.35 * _sim(r1.get("subject", ""), r2.get("subject", ""))
                    + 0.30 * _sim(r1p, r2p)
                    + 0.35 * _sim(r1.get("object", ""), r2.get("object", "")))

        sr = float(np.mean([max((rel_sim(r1, r2) for r2 in rels2), default=0.0) for r1 in rels1])) if rels1 else 0.0
        sp = float(np.mean([max((rel_sim(r2, r1) for r1 in rels1), default=0.0) for r2 in rels2])) if rels2 else 0.0
        sf1 = (2 * sp * sr / (sp + sr)) if (sp + sr) > 0 else 0.0
        return {"precision": sp, "recall": sr, "f1": sf1}

    obj_soft, attr_soft, rel_soft = _soft_objects(), _soft_attributes(), _soft_relations()

    overall_supp = len(obj_supported) + len(attr_supported) + len(rel_supported)
    overall_miss = len(obj_unsupported) + len(attr_unsupported) + len(rel_unsupported)
    overall_contra = len(obj_contradicted) + len(attr_contradicted) + len(rel_contradicted)
    overall_hard = _prf_hard(overall_supp, overall_miss, overall_contra)

    chair_i = (len(obj_contradicted) / max(len(objs2), 1)) if objs2 else 0.0
    weighted_hard = (_WEIGHT_OBJECT * obj_hard["f1"] + _WEIGHT_ATTRIBUTE * attr_hard["f1"] + _WEIGHT_RELATION * rel_hard["f1"])
    weighted_soft = (_WEIGHT_OBJECT * obj_soft["f1"] + _WEIGHT_ATTRIBUTE * attr_soft["f1"] + _WEIGHT_RELATION * rel_soft["f1"])

    preserved: Set[Tuple[str, ...]] = set()
    missing: Set[Tuple[str, ...]] = set()
    extra: Set[Tuple[str, ...]] = set()
    for o in obj_supported:
        preserved.add(("object", o))
    for o in obj_unsupported:
        missing.add(("object", o))
    for o in obj_contradicted:
        extra.add(("object", o))
    for o, a in attr_supported:
        preserved.add(("attribute", o, a))
    for o, a in attr_unsupported:
        missing.add(("attribute", o, a))
    for o, a in attr_contradicted:
        extra.add(("attribute", o, a))
    for s, p, o in rel_supported:
        preserved.add(("relation", s, p, o))
    for s, p, o in rel_unsupported:
        missing.add(("relation", s, p, o))
    for s, p, o in rel_contradicted:
        extra.add(("relation", s, p, o))

    return {
        "preserved": preserved, "missing": missing, "extra": extra,
        "precision": overall_hard["precision"], "recall": overall_hard["recall"], "f1": overall_hard["f1"],
        "object_precision": obj_hard["precision"], "object_recall": obj_hard["recall"], "object_f1": obj_hard["f1"],
        "attribute_precision": attr_hard["precision"], "attribute_recall": attr_hard["recall"], "attribute_f1": attr_hard["f1"],
        "relation_precision": rel_hard["precision"], "relation_recall": rel_hard["recall"], "relation_f1": rel_hard["f1"],
        "relation_recall_valid": (len(rels1) > 0),
        "object_precision_soft": obj_soft["precision"], "object_recall_soft": obj_soft["recall"], "object_f1_soft": obj_soft["f1"],
        "attribute_precision_soft": attr_soft["precision"], "attribute_recall_soft": attr_soft["recall"], "attribute_f1_soft": attr_soft["f1"],
        "relation_precision_soft": rel_soft["precision"], "relation_recall_soft": rel_soft["recall"], "relation_f1_soft": rel_soft["f1"],
        "weighted_kg_f1_hard": float(weighted_hard), "weighted_kg_f1_soft": float(weighted_soft),
        "chair_i": float(chair_i),
        "num_original_tuples": len(objs1) + sum(len(v) for v in attrs1.values()) + len(rels1),
        "num_reconstructed_tuples": len(objs2) + sum(len(v) for v in attrs2.values()) + len(rels2),
        "num_preserved": len(preserved), "num_missing": len(missing), "num_extra": len(extra),
        "object_matches": obj_match,
        "structural": _structural_metrics(kg_original, kg_reconstructed),
        "preserved_text": [tuple_to_text(t) for t in sorted(preserved, key=str)],
        "missing_text": [tuple_to_text(t) for t in sorted(missing, key=str)],
        "extra_text": [tuple_to_text(t) for t in sorted(extra, key=str)],
    }


# =====================================================================
# Generic-noun filters shared by anchor helpers
# =====================================================================

_GENERIC_OBJECTS = {
    "road", "street", "ground", "sky", "cloud", "clouds", "horizon", "asphalt",
    "pavement", "dirt", "grass", "sand", "concrete", "mud", "cobblestone", "tarmac",
    "footpath", "sidewalk", "tree", "trees", "leaves", "vegetation", "bush", "bushes",
    "plant", "plants", "foliage", "grasses", "lawn", "field", "building", "buildings",
    "wall", "fence", "pole", "rooftop", "house", "houses", "shop", "shops", "structure",
    "vehicle", "vehicles", "car", "cars", "truck", "trucks", "bus", "buses", "van",
    "vans", "motorcycle", "motorcycles", "auto", "autos", "two wheeler", "pedestrian",
    "pedestrians", "person", "people", "street lamp", "street light", "lamp", "lamps",
    "light", "lights", "lamppost", "streetlight", "background", "foreground", "midground",
    "scene", "view", "area", "setting", "atmosphere", "shadow", "shadows",
}

_GENERIC_HEAD_NOUNS = {
    "trees", "tree", "buildings", "building", "road", "street", "sky", "ground",
    "grass", "vehicles", "vehicle", "people", "pedestrians", "lights", "lamps",
    "field", "view", "scene", "area", "setting",
}

_FRAMING_KEYWORDS = {
    "rearview mirror", "rear view mirror", "side mirror", "windshield",
    "window frame", "doorframe", "sign", "signboard", "hoarding", "billboard",
    "banner", "plate", "license plate", "number plate",
}


def _is_distinctive(obj: str) -> bool:
    """True if a (normalized) object label is a genuinely distinctive entity."""
    if any(kw in obj for kw in _FRAMING_KEYWORDS):
        return True
    if any(c.isdigit() for c in obj) and any(c.isalpha() for c in obj):  # plate-like code
        return True
    words = obj.split()
    if len(words) == 1 and 4 <= len(obj) <= 30:
        return True
    if len(words) >= 2:
        return words[-1] not in _GENERIC_HEAD_NOUNS
    return False


# =====================================================================
# Anchor derivation
# =====================================================================

def derive_anchors_from_missing(comparison: Dict[str, Any], max_anchors: int = 6,
                                exclude_labels: Optional[Set[str]] = None,
                                include_relations: bool = False) -> List[str]:
    """
    Distinctive facts the reconstruction MISSED (reconstruction-guided re-injection).

    Returns distinctive missing OBJECT labels (substring-eligible strict anchors).
    Generic objects are filtered. If ``include_relations`` is True, also returns
    salient missing RELATIONS rendered as natural phrases (soft anchors — these
    are validated by KG recall, not substring matching, so a relation like
    "ambulance viewed through rearview mirror" can be a must-keep).
    """
    missing = comparison.get("missing", set())
    if not missing:
        return []
    exclude = {str(x).strip().lower() for x in (exclude_labels or set())}

    anchors: List[str] = []
    seen: Set[str] = set()
    for t in missing:
        if not isinstance(t, tuple) or len(t) < 2:
            continue
        if t[0] != "object":
            continue
        label = str(t[1]).strip().lower()
        if not label or label in _GENERIC_OBJECTS or label in exclude or label in seen:
            continue
        seen.add(label)
        if _is_distinctive(label):
            anchors.append(label)
            if len(anchors) >= max_anchors:
                break

    if include_relations and len(anchors) < max_anchors:
        for t in missing:
            if isinstance(t, tuple) and len(t) == 4 and t[0] == "relation":
                s, p, o = t[1], t[2], t[3]
                # Keep relations whose endpoints carry distinctive or framing info.
                if any(kw in f"{s} {o}" for kw in _FRAMING_KEYWORDS) or _is_distinctive(s) or _is_distinctive(o):
                    phrase = f"{s} {p} {o}".strip()
                    if phrase and phrase not in seen:
                        seen.add(phrase)
                        anchors.append(phrase)
                        if len(anchors) >= max_anchors:
                            break
    return anchors[:max_anchors]


def derive_soft_anchors_from_kg(kg: Dict[str, Any], max_anchors: int = 6) -> List[str]:
    """
    Salient SOFT anchors from KG_I — important visual facts the compressed caption
    should preserve (spec: G_c = Anchor(KG_I, ...)). Returns natural-language
    phrases, prioritizing:
      1. framing / reflection relations (e.g. "ambulance viewed through rearview mirror")
      2. distinctive objects that participate in relations or carry attributes
      3. other salient spatial/functional relations

    These are SOFT (semantic) anchors: the compressor passes them to the rewriter
    as must-keep scenario and validates preservation via KG recall, not substring.
    """
    objects = kg.get("objects", [])
    relations = kg.get("relations", [])
    labels = [o.get("label", o.get("name", "")) for o in objects]
    attr_count: Dict[str, int] = defaultdict(int)
    for o in objects:
        attr_count[o.get("label", "")] = len(o.get("attributes", []) or [])
    rel_degree: Dict[str, int] = defaultdict(int)
    for r in relations:
        rel_degree[r.get("subject", "")] += 1
        rel_degree[r.get("object", "")] += 1

    anchors: List[str] = []
    seen: Set[str] = set()

    def _add(phrase: str):
        phrase = phrase.strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            anchors.append(phrase)

    # 1. Framing / reflection relations first (scenario-critical).
    framing_preds = {"viewed through", "reflected in", "visible in", "framed by"}
    for r in relations:
        p = r.get("predicate", r.get("relation", ""))
        if p in framing_preds:
            _add(f"{r.get('subject','')} {p} {r.get('object','')}")
            if len(anchors) >= max_anchors:
                return anchors[:max_anchors]

    # 2. Distinctive objects ranked by (relation degree + attribute richness).
    ranked = sorted(
        (lbl for lbl in labels if lbl and _is_distinctive(lbl)),
        key=lambda l: (rel_degree[l] + attr_count[l]), reverse=True,
    )
    for lbl in ranked:
        _add(lbl)
        if len(anchors) >= max_anchors:
            return anchors[:max_anchors]

    # 3. Remaining salient relations (skip purely generic-generic pairs).
    for r in relations:
        s = r.get("subject", "")
        o = r.get("object", "")
        p = r.get("predicate", r.get("relation", ""))
        if _is_distinctive(s) or _is_distinctive(o):
            _add(f"{s} {p} {o}")
            if len(anchors) >= max_anchors:
                break
    return anchors[:max_anchors]


# =====================================================================
# Compatibility wrappers
# =====================================================================

def compute_kg_f1(kg_original: Dict[str, Any], kg_reconstructed: Dict[str, Any]) -> Dict[str, float]:
    s = compare_scene_graphs(kg_original, kg_reconstructed)
    return {k: s[k] for k in (
        "f1", "precision", "recall",
        "object_precision", "object_recall", "object_f1",
        "attribute_precision", "attribute_recall", "attribute_f1",
        "relation_precision", "relation_recall", "relation_f1",
        "weighted_kg_f1_hard", "weighted_kg_f1_soft", "chair_i",
    )}


def compute_spice_f1(kg_original: Dict[str, Any], kg_reconstructed: Dict[str, Any]) -> Dict[str, float]:
    return compute_kg_f1(kg_original, kg_reconstructed)


# =====================================================================
# Pretty printing
# =====================================================================

def tuple_to_text(t: Tuple[str, ...]) -> str:
    if not t:
        return ""
    if t[0] == "object" and len(t) == 2:
        return f"object: {t[1]}"
    if t[0] == "attribute" and len(t) == 3:
        return f"attribute: {t[1]} has {t[2]}"
    if t[0] == "relation" and len(t) == 4:
        return f"relation: {t[1]} {t[2]} {t[3]}"
    return str(t)


def print_kg_comparison(comparison: Dict[str, Any], max_items: int = 20) -> None:
    print("=" * 80)
    print("KG COMPARISON")
    print("=" * 80)
    print(f"KG F1={comparison['f1']:.4f}, Precision={comparison['precision']:.4f}, Recall={comparison['recall']:.4f}")
    if "weighted_kg_f1_hard" in comparison:
        print(f"Weighted KG-F1 (hard)={comparison['weighted_kg_f1_hard']:.4f}, "
              f"(soft)={comparison.get('weighted_kg_f1_soft', 0.0):.4f}")
    if "chair_i" in comparison:
        print(f"CHAIR_i (object hallucination rate)={comparison['chair_i']:.4f}")
    if "object_matches" in comparison:
        print("\nObject matches:")
        for k, v in comparison["object_matches"].items():
            print(f"  {k}  ->  {v}")
    print("\nPreserved elements:")
    for item in list(comparison["preserved"])[:max_items]:
        print("  +", tuple_to_text(item))
    print("\nMissing elements:")
    for item in list(comparison["missing"])[:max_items]:
        print("  -", tuple_to_text(item))
    print("\nExtra / hallucinated elements:")
    for item in list(comparison["extra"])[:max_items]:
        print("  *", tuple_to_text(item))
    print("=" * 80)