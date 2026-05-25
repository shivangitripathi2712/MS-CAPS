import re
import string
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

# Reuse the KG semantic matcher / generic filters (same package, same mpnet).
try:
    from .scene_graph import (
        semantic_text_similarity, normalize_label, _is_distinctive, _GENERIC_OBJECTS,
    )
except Exception:  # allow running as a standalone script
    from scene_graph import (  # type: ignore
        semantic_text_similarity, normalize_label, _is_distinctive, _GENERIC_OBJECTS,
    )


_FRAMING_ANCHOR_KW = {
    "rearview mirror", "rear view mirror", "side mirror", "mirror",
    "windshield", "window", "doorframe", "sign", "signboard", "hoarding",
    "billboard", "banner", "plate", "license plate", "number plate",
}


# ---------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------

@dataclass
class DeletableSubtree:
    """A phrase/span that can be tested for deletion (legacy name preserved)."""
    label: str
    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int
    importance: float = 0.0
    depth: int = 0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_caption_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.replace("( ", "(").replace(" )", ")")
    return text.strip()


def count_tokens(text: str) -> int:
    return len(text.split())


_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "at", "by", "with",
    "for", "from", "as", "is", "are", "was", "were", "this", "that", "these",
    "those", "image", "photo", "picture", "scene", "showing", "shows", "it",
    "its", "view", "background", "foreground",
}


def content_words(text: str) -> List[str]:
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return [w for w in text.split() if len(w) > 2 and w not in _STOPWORDS]


def _strip_leading_articles(text: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", text.strip(), flags=re.IGNORECASE).strip()


# ---------------------------------------------------------------------
# Main pruner
# ---------------------------------------------------------------------

class ParseTreePruner:
    """
    Lightweight phrase extractor. Uses spaCy noun chunks + dependency subtrees
    with a regex fallback. Also hosts strict-anchor harvesting.
    """

    DEFAULT_DELETABLE_LABELS = {
        "NOUN_CHUNK", "PP", "ADJP", "ADVP", "RELCL", "ADVCL", "ACL", "APPOS",
        "ATTR_REGEX", "PP_REGEX",
    }

    def __init__(
        self,
        deletable_labels: Optional[List[str]] = None,
        use_benepar: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        self.deletable_labels = (
            set(deletable_labels) if deletable_labels is not None
            else set(self.DEFAULT_DELETABLE_LABELS)
        )
        self.use_benepar = use_benepar
        self.spacy_model = spacy_model
        self.nlp = None
        self._load_spacy()

    def _load_spacy(self) -> None:
        try:
            import spacy
            self.nlp = spacy.load(self.spacy_model)
            if self.use_benepar:
                try:
                    import benepar  # noqa: F401
                    if "benepar" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})
                    print("[ParseTree] Loaded spaCy + Benepar.")
                except Exception as exc:
                    print(f"[ParseTree] Benepar unavailable; spaCy only. Error: {exc}")
                    self.use_benepar = False
            else:
                print("[ParseTree] Loaded spaCy phrase extractor.")
        except Exception as exc:
            print(f"[ParseTree] spaCy unavailable; regex fallback only. Error: {exc}")
            self.nlp = None

    # ================================================================= #
    # STRICT ANCHOR HARVESTING  (primary use in current pipeline)
    # ================================================================= #
    def extract_strict_anchors(
        self,
        caption: str,
        kg: Optional[dict] = None,
        visible_text: Optional[Sequence[Union[str, dict]]] = None,
        kg_match_threshold: float = 0.55,
        max_anchors: Optional[int] = None,
    ) -> List[str]:
        """
        Build strict anchors S_c (exact caption phrases the rewriter must keep).

        A noun-phrase becomes a strict anchor if:
          (a) it is DISTINCTIVE  — contains a proper noun, an alphanumeric code
              (plate-like), or a framing keyword; OR
          (b) it CROSS-MATCHES a KG_I object at >= kg_match_threshold and the
              matched phrase is not purely generic.
        Visible in-image text is always added (image-confirmed).

        Returns surface phrases (original casing/spacing), de-duplicated by
        substring containment, ranked: visible-text > distinctive > KG-matched,
        longer phrases first. Capped at max_anchors if given.
        """
        caption = clean_caption_spacing(caption)
        if not caption:
            return []

        kg_labels = [
            o.get("label", o.get("name", "")) for o in (kg or {}).get("objects", [])
        ]
        kg_labels = [l for l in kg_labels if l]

        # priority: 3 = visible text, 2 = distinctive, 1 = KG-matched
        scored: List[Tuple[int, int, str]] = []  # (priority, length, phrase)
        seen_norm = set()

        def _consider(phrase: str, priority: int):
            phrase = _strip_leading_articles(phrase).strip(string.whitespace + ",.;:")
            if not phrase or not content_words(phrase):
                return
            norm = phrase.lower()
            if norm in seen_norm:
                return
            seen_norm.add(norm)
            scored.append((priority, len(phrase), phrase))

        # --- 1. Visible in-image text (always strict) ---
        for vt in (visible_text or []):
            txt = vt.get("text", "") if isinstance(vt, dict) else str(vt)
            txt = txt.strip()
            if txt:
                _consider(txt, priority=3)

        # --- 2/3. Caption noun phrases ---
        for phrase, is_distinctive in self._caption_noun_phrases(caption):
            if is_distinctive:
                _consider(phrase, priority=2)
            elif kg_labels:
                norm_phrase = normalize_label(phrase)
                best = max((semantic_text_similarity(norm_phrase, normalize_label(l))
                            for l in kg_labels), default=0.0)
                if best >= kg_match_threshold and norm_phrase not in _GENERIC_OBJECTS:
                    _consider(phrase, priority=1)

        # Rank, then drop anchors that are substrings of a longer kept anchor.
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        anchors: List[str] = []
        for _, _, phrase in scored:
            low = phrase.lower()
            if any(low in a.lower() and low != a.lower() for a in anchors):
                continue
            anchors.append(phrase)
            if max_anchors is not None and len(anchors) >= max_anchors:
                break
        return anchors

    def _caption_noun_phrases(self, caption: str) -> List[Tuple[str, bool]]:
        """
        Return [(surface_phrase, is_distinctive), ...] from the caption.
        Uses spaCy noun chunks when available, else a regex fallback.
        """
        out: List[Tuple[str, bool]] = []
        if self.nlp is not None:
            doc = self.nlp(caption)
            for chunk in doc.noun_chunks:
                toks = [t for t in chunk if not (t.pos_ == "DET" or t.lower_ in ("a", "an", "the"))]
                if not toks:
                    continue
                start = toks[0].idx
                end = toks[-1].idx + len(toks[-1].text)
                phrase = caption[start:end].strip()
                if not phrase:
                    continue
                distinctive = (
                    any(t.pos_ == "PROPN" for t in toks)
                    or (any(c.isdigit() for c in phrase) and any(c.isalpha() for c in phrase))
                    or any(kw in phrase.lower() for kw in _FRAMING_ANCHOR_KW)
                )
                out.append((phrase, distinctive))
        else:
            # Regex fallback: capitalized runs (proper nouns/brands) + codes.
            for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b", caption):
                out.append((m.group(1), True))
            for m in re.finditer(r"\b([A-Z0-9]{4,})\b", caption):  # plate-like codes
                out.append((m.group(1), True))
        return out

    # ================================================================= #
    # LEGACY phrase extraction / deletion API (backward compatible)
    # ================================================================= #
    def parse_and_find_deletable(
        self, caption: str, word_importance_scores: Optional[List[float]] = None,
    ) -> List[DeletableSubtree]:
        caption = clean_caption_spacing(caption)
        if not caption:
            return []
        spans = self._regex_spans(caption) if self.nlp is None else self._spacy_spans(caption)
        spans = self._deduplicate_and_filter(caption, spans)
        if word_importance_scores is not None:
            spans = self._assign_importance(caption, spans, word_importance_scores)
        spans.sort(key=lambda s: (s.importance, -count_tokens(s.text), s.start_char))
        return spans

    def _spacy_spans(self, caption: str) -> List[DeletableSubtree]:
        doc = self.nlp(caption)
        spans: List[DeletableSubtree] = []
        if "NOUN_CHUNK" in self.deletable_labels:
            for chunk in doc.noun_chunks:
                spans.append(DeletableSubtree(
                    "NOUN_CHUNK", chunk.text, chunk.start_char, chunk.end_char,
                    chunk.start, chunk.end - 1, 0.0, 0,
                ))
        for token in doc:
            if token.dep_ == "prep" and "PP" in self.deletable_labels:
                spans.append(self._subtree_to_span("PP", list(token.subtree)))
            if token.dep_ == "relcl" and "RELCL" in self.deletable_labels:
                spans.append(self._subtree_to_span("RELCL", list(token.subtree)))
            if token.dep_ == "advcl" and "ADVCL" in self.deletable_labels:
                spans.append(self._subtree_to_span("ADVCL", list(token.subtree)))
            if token.dep_ == "acl" and "ACL" in self.deletable_labels:
                spans.append(self._subtree_to_span("ACL", list(token.subtree)))
            if token.dep_ == "appos" and "APPOS" in self.deletable_labels:
                spans.append(self._subtree_to_span("APPOS", list(token.subtree)))
            if token.pos_ == "ADJ" and "ADJP" in self.deletable_labels:
                spans.append(DeletableSubtree(
                    "ADJP", token.text, token.idx, token.idx + len(token.text),
                    token.i, token.i, 0.0, 0,
                ))
            if token.pos_ == "ADV" and "ADVP" in self.deletable_labels:
                spans.append(DeletableSubtree(
                    "ADVP", token.text, token.idx, token.idx + len(token.text),
                    token.i, token.i, 0.0, 0,
                ))
        return spans

    def _subtree_to_span(self, label: str, subtree_tokens) -> DeletableSubtree:
        subtree_tokens = sorted(subtree_tokens, key=lambda t: t.i)
        s, e = subtree_tokens[0], subtree_tokens[-1]
        start_char, end_char = s.idx, e.idx + len(e.text)
        return DeletableSubtree(
            label, s.doc.text[start_char:end_char], start_char, end_char, s.i, e.i, 0.0, 0,
        )

    def _regex_spans(self, caption: str) -> List[DeletableSubtree]:
        spans: List[DeletableSubtree] = []
        prep_pattern = (
            r"\b(in front of|next to|on|in|under|over|near|behind|beside|with|without|"
            r"around|through|across|inside|outside|above|below)\b[^,.!?;:]*"
        )
        for match in re.finditer(prep_pattern, caption, flags=re.IGNORECASE):
            sw, ew = self._char_to_word_indices(caption, match.start(), match.end())
            spans.append(DeletableSubtree(
                "PP_REGEX", match.group(0), match.start(), match.end(), sw, ew, 0.0, 0,
            ))
        attr_words = [
            "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey",
            "small", "large", "big", "little", "tall", "short", "old", "young", "bright",
            "dark", "cloudy", "sunny", "wooden", "metal", "glass", "round", "square",
        ]
        for word in attr_words:
            for match in re.finditer(rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE):
                sw, ew = self._char_to_word_indices(caption, match.start(), match.end())
                spans.append(DeletableSubtree(
                    "ATTR_REGEX", match.group(0), match.start(), match.end(), sw, ew, 0.0, 0,
                ))
        return spans

    def _deduplicate_and_filter(self, caption: str, spans: List[DeletableSubtree]) -> List[DeletableSubtree]:
        final_spans: List[DeletableSubtree] = []
        seen = set()
        caption_tokens = count_tokens(caption)
        for span in spans:
            start = max(0, int(span.start_char))
            end = min(len(caption), int(span.end_char))
            if start >= end:
                continue
            text = caption[start:end].strip().strip(string.whitespace + string.punctuation)
            if not text:
                continue
            tc = count_tokens(text)
            if tc >= caption_tokens or tc > 12:
                continue
            if tc <= 1 and len(content_words(text)) == 0:
                continue
            key = (start, end, span.label)
            if key in seen:
                continue
            seen.add(key)
            final_spans.append(DeletableSubtree(
                span.label, caption[start:end].strip(), start, end,
                span.start_word, span.end_word, span.importance, span.depth,
            ))
        return final_spans

    def _assign_importance(self, caption, spans, word_scores):
        if not word_scores:
            return spans
        updated = []
        for span in spans:
            sw, ew = self._char_to_word_indices(caption, span.start_char, span.end_char)
            if sw is None or ew is None:
                avg = 0.0
            else:
                vals = word_scores[sw:ew + 1]
                avg = sum(vals) / max(len(vals), 1)
            updated.append(DeletableSubtree(
                span.label, span.text, span.start_char, span.end_char,
                sw if sw is not None else span.start_word,
                ew if ew is not None else span.end_word, float(avg), span.depth,
            ))
        return updated

    def _char_to_word_indices(self, caption, start_char, end_char):
        start_word = end_word = None
        for idx, match in enumerate(re.finditer(r"\S+", caption)):
            if match.end() > start_char and match.start() < end_char:
                if start_word is None:
                    start_word = idx
                end_word = idx
        return start_word, end_word

    def delete_subtree(self, caption: str, subtree: DeletableSubtree) -> str:
        return clean_caption_spacing(caption[:subtree.start_char] + " " + caption[subtree.end_char:])

    def get_compression_candidates(self, caption, word_scores=None, min_remaining_words=5):
        subtrees = self.parse_and_find_deletable(caption, word_importance_scores=word_scores)
        out = []
        for st in subtrees:
            nc = self.delete_subtree(caption, st)
            if count_tokens(nc) >= min_remaining_words:
                out.append((st, nc))
        return out


# ---------------------------------------------------------------------
# Module-level convenience (cached singleton pruner)
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _shared_pruner() -> ParseTreePruner:
    return ParseTreePruner()


def extract_strict_anchors(
    caption: str,
    kg: Optional[dict] = None,
    visible_text: Optional[Sequence[Union[str, dict]]] = None,
    kg_match_threshold: float = 0.55,
    max_anchors: Optional[int] = None,
) -> List[str]:
    """Convenience wrapper using a process-wide cached ParseTreePruner."""
    return _shared_pruner().extract_strict_anchors(
        caption, kg=kg, visible_text=visible_text,
        kg_match_threshold=kg_match_threshold, max_anchors=max_anchors,
    )