"""JBDetection — tag matcher.

A *much* simplified re-implementation of the original ``VectorMatcher``
class. The original had triple-nested loops and a confusing mix of
similarity metrics; this version exposes one clean entry point
(:meth:`TagMatcher.match_tag`) that returns one of three outcomes:

- ``("exact", 1.0, io_tag)``  — exact match (after OCR-confusion fixup)
- ``("similar", score, io_tag)``  — fuzzy match (Levenshtein ≥ threshold)
- ``("unmatched", 0.0, "")``  — no match

The matcher is built from an Excel IO List via :meth:`build_from_excel`,
which extracts tags from the ``Tag No`` column (case-insensitive) and
constructs a feature vector for each. Vectors are compared using cosine
similarity — same idea as the original, but flattened into a single
class without nested helper classes.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import (
    Config, INSTRUMENT_PREFIXES, OCR_CONFUSION_PAIRS,
)

logger = logging.getLogger("jb_detection.tag_matcher")


# ── Optional dependency: python-Levenshtein ─────────────────────────────
try:
    import Levenshtein as _lev  # type: ignore
    _HAS_LEVENSHTEIN = True
except Exception:
    _HAS_LEVENSHTEIN = False

    def _lev_ratio(a: str, b: str) -> float:
        """Pure-Python fallback using difflib."""
        import difflib
        return float(difflib.SequenceMatcher(None, a, b).ratio())

    class _LevFallback:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return _lev_ratio(a, b)

        @staticmethod
        def distance(a: str, b: str) -> int:
            # Naive edit distance — fine for short tag strings.
            if a == b:
                return 0
            if not a:
                return len(b)
            if not b:
                return len(a)
            prev = list(range(len(b) + 1))
            for i, ca in enumerate(a, 1):
                cur = [i]
                for j, cb in enumerate(b, 1):
                    cur.append(min(
                        prev[j] + 1,
                        cur[j - 1] + 1,
                        prev[j - 1] + (ca != cb),
                    ))
                prev = cur
            return prev[-1]

    _lev = _LevFallback()  # type: ignore


# ── TagMatcher ──────────────────────────────────────────────────────────
class TagMatcher:
    """Match OCR-detected tags against the IO List.

    The matcher is **stateful** — it holds the reference tag set and
    feature vectors in memory. Build once, query many times.
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 levenshtein_threshold: float = 0.92,
                 config: Optional[Config] = None) -> None:
        if config is None:
            from .config import DEFAULT_CONFIG
            config = DEFAULT_CONFIG
        self._config = config
        self.similarity_threshold = float(
            similarity_threshold
            if similarity_threshold is not None
            else config.match_similar_threshold
        )
        self.levenshtein_threshold = float(
            levenshtein_threshold
            if levenshtein_threshold is not None
            else config.match_levenshtein_threshold
        )
        # Reference data
        self.reference_tags: List[str] = []
        self.tag_vectors: Dict[str, Dict[str, float]] = {}
        self.tag_set_upper: Set[str] = set()
        # Track IO List tag in original case (for output)
        self.upper_to_original: Dict[str, str] = {}

        # Stats
        self.match_attempts = 0
        self.exact_matches = 0
        self.similar_matches = 0
        self.unmatched = 0

    # ── Building from Excel ────────────────────────────────────────
    def build_from_excel(self, excel_path: str,
                          tag_column: Optional[str] = None) -> int:
        """Load reference tags from an Excel IO List.

        Parameters
        ----------
        excel_path:
            Path to the .xlsx IO List.
        tag_column:
            Column name to read tags from. Defaults to
            ``config.excel_io_list_tag_column`` (``"Tag No"``).

        Returns
        -------
        int
            Number of unique tags loaded.
        """
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"pandas not available: {exc}") from exc

        col = tag_column or self._config.excel_io_list_tag_column
        df = pd.read_excel(excel_path)

        if col not in df.columns:
            # Try common alternatives
            for alt in ("Tag", "Tag No", "Tag No.", "tag", "tag no",
                         "TAG NO", "TAG NO.", "TAG"):
                if alt in df.columns:
                    col = alt
                    break
            else:
                raise ValueError(
                    f"Excel file must contain a '{col}' column. "
                    f"Available: {df.columns.tolist()}"
                )

        tags = df[col].dropna().astype(str).str.strip()
        tags = tags[tags.str.len() > 0]
        tags_upper = tags.str.upper().unique()

        for tag_upper in tags_upper:
            self.add_reference_tag(tag_upper)

        logger.info("Loaded %d unique tags from IO List: %s",
                    len(self.reference_tags), excel_path)
        return len(self.reference_tags)

    def add_reference_tag(self, tag: str) -> None:
        """Add a single reference tag (uppercase, trimmed)."""
        if not tag or not isinstance(tag, str):
            return
        tag = tag.upper().strip()
        if not tag:
            return
        if tag not in self.tag_set_upper:
            self.reference_tags.append(tag)
            self.tag_set_upper.add(tag)
            self.upper_to_original[tag] = tag
        self.tag_vectors[tag] = self.create_tag_vector(tag)

    # ── Feature extraction ─────────────────────────────────────────
    def create_tag_vector(self, tag: str) -> Dict[str, float]:
        """Create a feature vector for a tag (used for cosine similarity).

        The vector includes:

        - Length, digit/letter/hyphen counts, part count
        - One-hot prefix features for each known instrument prefix
        - Hashed middle/last parts (so different middle letters give
          different vectors — fixes a known false-positive in the
          original code)
        - Digit sequence features
        """
        tag = str(tag).upper().strip()
        v: Dict[str, float] = {}

        v["length"] = float(len(tag))
        v["num_digits"] = float(sum(c.isdigit() for c in tag))
        v["num_letters"] = float(sum(c.isalpha() for c in tag))
        v["num_hyphens"] = float(sum(c == "-" for c in tag))
        v["num_parts"] = float(len(tag.split("-")))

        # Prefix one-hot
        for prefix in INSTRUMENT_PREFIXES:
            v[f"prefix_{prefix}"] = 1.0 if tag.startswith(prefix) else 0.0

        parts = tag.split("-")
        if len(parts) >= 2:
            v["first_part"] = float(sum(ord(c) for c in parts[0]) % 1000)
            v["last_part"] = float(sum(ord(c) for c in parts[-1]) % 1000)
            v["last_part_digits"] = float(sum(c.isdigit() for c in parts[-1]))
            v["first_part_length"] = float(len(parts[0]))
            v["last_part_length"] = float(len(parts[-1]))
            v["first_part_nonzero_digits"] = float(
                sum(1 for c in parts[0] if c.isdigit() and c != "0")
            )
            v["last_part_nonzero_digits"] = float(
                sum(1 for c in parts[-1] if c.isdigit() and c != "0")
            )
            v["standard_number_pattern"] = (
                1.0 if re.search(r"\d{3}-\d{2,3}", tag) else 0.0
            )

        # Middle part hash (critical for multi-part tags)
        # Without this, "9000-HDL-039" and "9000-FGD-039" would have
        # cosine similarity ~1.0 (same length, same digits, same parts).
        if len(parts) >= 3:
            middle = "".join(parts[1:-1])
            v["middle_part"] = float(sum(ord(c) for c in middle) % 1000)
            v["middle_part_length"] = float(len(middle))
            v["middle_part_letters"] = float(sum(c.isalpha() for c in middle))
            v["middle_part_digits"] = float(sum(c.isdigit() for c in middle))
            v["middle_part_char_hash"] = float(
                sum(ord(c) * (i + 1) for i, c in enumerate(middle)) % 10000
            )

        # 2-part letter-hash for distinguishing "HDL-039" from "FGD-039"
        if len(parts) == 2:
            for idx, part in enumerate(parts):
                letters = "".join(c for c in part if c.isalpha())
                if letters:
                    v[f"part{idx}_letter_hash"] = float(
                        sum(ord(c) * (i + 1) for i, c in enumerate(letters)) % 10000
                    )
                    v[f"part{idx}_letter_length"] = float(len(letters))

        v["has_standard_format"] = (
            1.0 if re.match(r"[A-Z]{2,4}-\d{3}-\d{2,3}", tag) else 0.0
        )

        digit_seqs = re.findall(r"\d+", tag)
        if digit_seqs:
            v["first_digit_seq"] = float(int(digit_seqs[0]))
            v["last_digit_seq"] = float(int(digit_seqs[-1]))
            v["num_digit_sequences"] = float(len(digit_seqs))

        return v

    # ── OCR correction ─────────────────────────────────────────────
    def apply_ocr_corrections(self, tag: str) -> str:
        """Try common OCR-confusion substitutions to recover the real tag.

        For each character in ``tag`` that appears in
        :data:`OCR_CONFUSION_PAIRS` (in either direction), try
        substituting it with the alternate form. If any resulting
        string matches a reference tag exactly, return that. Otherwise
        return the original tag.

        The pairs in :data:`OCR_CONFUSION_PAIRS` map OCR→truth (e.g.
        ``'O'→'0'`` means OCR sees ``'O'`` where the truth is ``'0'``).
        But OCR errors go both ways — sometimes OCR sees ``'0'`` where
        the truth is ``'O'``. We therefore try BOTH directions for
        every pair, matching the behaviour of the original code.
        """
        if not tag:
            return tag
        tag_upper = tag.upper().strip()
        if tag_upper in self.tag_set_upper:
            return tag_upper

        # Build a bidirectional substitution list: for each pair (k, v),
        # we try k→v AND v→k.
        substitutions: List[Tuple[str, str]] = []
        for k, v in OCR_CONFUSION_PAIRS.items():
            substitutions.append((k, v))
            substitutions.append((v, k))

        # Try single-char substitutions at each position
        for i, c in enumerate(tag_upper):
            for old_ch, new_ch in substitutions:
                if c != old_ch:
                    continue
                candidate = tag_upper[:i] + new_ch + tag_upper[i + 1:]
                if candidate in self.tag_set_upper:
                    logger.debug("OCR correction: '%s' → '%s'", tag, candidate)
                    return candidate

        # Try double substitutions (rare but happens for UZSO/UZSC-style tags)
        for i, c1 in enumerate(tag_upper):
            for old1, new1 in substitutions:
                if c1 != old1:
                    continue
                cand1 = tag_upper[:i] + new1 + tag_upper[i + 1:]
                for j, c2 in enumerate(cand1):
                    if j == i:
                        continue
                    for old2, new2 in substitutions:
                        if c2 != old2:
                            continue
                        cand2 = cand1[:j] + new2 + cand1[j + 1:]
                        if cand2 in self.tag_set_upper:
                            logger.debug("OCR correction (double): '%s' → '%s'",
                                         tag, cand2)
                            return cand2

        return tag_upper

    # ── Similarity ─────────────────────────────────────────────────
    def calculate_similarity(self,
                              v1: Dict[str, float],
                              v2: Dict[str, float]) -> float:
        """Cosine similarity between two tag feature vectors."""
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        # Iterate over the smaller dict for speed
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        for k, a in v1.items():
            b = v2.get(k, 0.0)
            dot += a * b
            norm1 += a * a
        for b in v2.values():
            norm2 += b * b
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        sim = dot / ((norm1 ** 0.5) * (norm2 ** 0.5))
        return max(0.0, min(1.0, sim))

    def _levenshtein_ratio(self, a: str, b: str) -> float:
        try:
            return float(_lev.ratio(a, b))
        except Exception:
            return 0.0

    # ── Main matching ──────────────────────────────────────────────
    def match_tag(self, tag: str) -> Tuple[str, float, str]:
        """Match a single OCR tag against the IO List.

        Returns
        -------
        (match_type, score, matched_tag)
            ``match_type`` is one of ``"exact"``, ``"similar"``,
            ``"unmatched"``. ``matched_tag`` is the IO List tag (or "").
        """
        if not tag:
            return ("unmatched", 0.0, "")
        self.match_attempts += 1

        tag_upper = str(tag).upper().strip()
        if not tag_upper:
            return ("unmatched", 0.0, "")

        # ── 1. Exact match ────────────────────────────────────────
        if tag_upper in self.tag_set_upper:
            self.exact_matches += 1
            return ("exact", 1.0, tag_upper)

        # ── 2. OCR-confusion correction ───────────────────────────
        if self._config.match_use_confusion_pairs:
            corrected = self.apply_ocr_corrections(tag_upper)
            if corrected != tag_upper and corrected in self.tag_set_upper:
                self.exact_matches += 1
                return ("exact", 1.0, corrected)

        # ── 3. Fuzzy match (Levenshtein first — fast and reliable) ─
        if self.reference_tags:
            best_io = ""
            best_score = 0.0
            for io_tag in self.reference_tags:
                score = self._levenshtein_ratio(tag_upper, io_tag)
                if score > best_score:
                    best_score = score
                    best_io = io_tag
                if score >= 1.0:
                    break

            if best_io and best_score >= self.levenshtein_threshold:
                self.similar_matches += 1
                return ("similar", best_score, best_io)

            # ── 4. Fallback to vector similarity ──────────────────
            # Used when Levenshtein is just below threshold but the
            # structural features (digits, prefixes) line up well.
            tag_vec = self.create_tag_vector(tag_upper)
            best_vec_io = ""
            best_vec_score = 0.0
            for io_tag in self.reference_tags:
                io_vec = self.tag_vectors.get(io_tag)
                if io_vec is None:
                    continue
                # Quick filter: skip if length difference is too big
                if abs(len(io_tag) - len(tag_upper)) > max(5, len(io_tag) * 0.5):
                    continue
                sim = self.calculate_similarity(tag_vec, io_vec)
                if sim > best_vec_score:
                    best_vec_score = sim
                    best_vec_io = io_tag

            if best_vec_io and best_vec_score >= self.similarity_threshold:
                # Combine: take the max of Levenshtein and vector scores
                final_score = max(best_score, best_vec_score)
                self.similar_matches += 1
                return ("similar", final_score, best_vec_io)

        self.unmatched += 1
        return ("unmatched", 0.0, "")

    def match_many(self, tags: List[str]) -> Dict[str, Tuple[str, float, str]]:
        """Match a batch of tags. Returns ``{tag: (type, score, io_tag)}``."""
        return {tag: self.match_tag(tag) for tag in tags}

    # ── Stats ──────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        total = max(1, self.match_attempts)
        return {
            "match_attempts": self.match_attempts,
            "exact_matches": self.exact_matches,
            "similar_matches": self.similar_matches,
            "unmatched": self.unmatched,
            "exact_rate": self.exact_matches / total,
            "similar_rate": self.similar_matches / total,
            "reference_tag_count": len(self.reference_tags),
            "levenshtein_available": _HAS_LEVENSHTEIN,
        }

    def reset_stats(self) -> None:
        self.match_attempts = 0
        self.exact_matches = 0
        self.similar_matches = 0
        self.unmatched = 0


__all__ = ["TagMatcher"]
