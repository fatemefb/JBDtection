"""Unmatched candidate rules from DataAnalysisModule phases 0 and 2.5.

Profile scoring and learned-prefix acceptance follow the legacy extractor.
The caller applies its configured JB/MC/cable/spare exclusions.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Set


class LegacyCandidateRules:
    def __init__(self, references, distance):
        self.distance = distance
        self.profile = self._build_io_pattern_profile(set(references))
        prefixes = set()
        normalized = [self._normalize_ocr_tag_candidate(tag) for tag in references]
        for tag in normalized:
            for pattern in (r"^([A-Z]{2,6})[-]?", r"^(\d{1,4}[A-Z]{1,6})[-]?", r"^\d{1,4}-([A-Z]{1,6})"):
                match = re.match(pattern, tag)
                if match:
                    prefixes.add(match.group(1))
        self.prefix_patterns = []
        for prefix in prefixes:
            area_prefix = prefix.isalpha() and not any(tag.startswith(prefix) for tag in normalized)
            pattern = (r"^\d{1,4}-" if area_prefix and len(prefix) <= 4 else "^") + re.escape(prefix) + r"[-\d]"
            self.prefix_patterns.append(re.compile(pattern, re.IGNORECASE))
        self.general_pattern = re.compile(r"^[A-Z0-9]{2,6}[-]?[A-Z0-9]{1,10}(?:[-][A-Z0-9]{1,10}){0,3}$", re.IGNORECASE)
        self.token_pattern = re.compile(
            r"(?<![A-Z0-9])[A-Z0-9]+(?:[-_][A-Z0-9]+){0,3}(?![A-Z0-9])",
            re.IGNORECASE,
        )

    def extract_candidates(self, text):
        """Return independently reviewable tag-shaped tokens from an OCR line.

        This is a discovery fallback: it preserves likely tags that do not fit
        a literal IO family. It does not claim that a candidate matches IO.
        """
        candidates = []
        seen = set()
        for match in self.token_pattern.finditer(str(text or "")):
            raw = match.group().strip("-_")
            tag = self._normalize_ocr_tag_candidate(raw)
            if not tag or tag in seen or not self.accepts(tag):
                continue
            score = self._score_pattern_candidate(tag, self.profile)
            candidates.append((tag, score))
            seen.add(tag)
        return candidates

    def accepts(self, raw):
        tag = self._normalize_ocr_tag_candidate(raw)
        if len(tag) < 4 or not re.search(r"[A-Z]", tag) or not re.search(r"\d", tag):
            return False
        score = self._score_pattern_candidate(tag, self.profile)
        collected = bool(self.general_pattern.fullmatch(tag)) or score >= 0.62
        return collected and (score >= 0.62 or any(pattern.match(tag) for pattern in self.prefix_patterns))

    def _extract_tag_prefix(self, tag: str) -> str:
        tag_upper = tag.upper()
        digit_dash_letter_match = re.match('^\\d{1,4}-([A-Z]{1,6})[-\\d]', tag_upper)
        if digit_dash_letter_match:
            return digit_dash_letter_match.group(1)
        match = re.match('^([A-Z0-9]+)', tag_upper)
        return match.group(1) if match else ''

    def _normalize_ocr_tag_candidate(self, text: str) -> str:
        if not text:
            return ''
        normalized = str(text).strip().upper()
        normalized = re.sub('\\s+', '', normalized)
        normalized = normalized.replace('_', '-')
        normalized = normalized.strip('-.')
        return normalized

    def _build_io_pattern_profile(self, io_tags: 'Set[str]') -> Dict[str, Any]:
        prefixes: Set[str] = set()
        lengths: List[int] = []
        hyphen_count = 0
        numeric_lengths: List[int] = []
        for raw_tag in io_tags or set():
            tag = self._normalize_ocr_tag_candidate(raw_tag)
            if not tag:
                continue
            lengths.append(len(tag))
            if '-' in tag:
                hyphen_count += 1
            prefix_match = re.match('^([A-Z]{2,6})', tag)
            if prefix_match:
                prefixes.add(prefix_match.group(1))
            else:
                digit_letter_match = re.match('^\\d{1,4}([A-Z]{1,6})', tag)
                if digit_letter_match:
                    prefixes.add(digit_letter_match.group(1))
                digit_dash_letter_match = re.match('^\\d{1,4}-([A-Z]{1,6})', tag)
                if digit_dash_letter_match:
                    prefixes.add(digit_dash_letter_match.group(1))
            for num_part in re.findall('\\d+', tag):
                numeric_lengths.append(len(num_part))
        if not prefixes:
            prefixes = {'UZSO', 'UZSC', 'FIT', 'PIT', 'TIT', 'LIT', 'FCV', 'PCV', 'TCV', 'LCV', 'TY', 'LA', 'UY', 'UHSL', 'UHSH'}
        min_len = min(lengths) if lengths else 5
        max_len = max(lengths) if lengths else 16
        avg_num_len = sum(numeric_lengths) / len(numeric_lengths) if numeric_lengths else 3.0
        hyphen_ratio = hyphen_count / len(lengths) if lengths else 0.5
        return {'prefixes': prefixes, 'min_len': min_len, 'max_len': max_len, 'avg_num_len': avg_num_len, 'hyphen_ratio': hyphen_ratio}

    def _score_pattern_candidate(self, candidate: str, io_profile: Dict[str, Any]) -> float:
        tag = self._normalize_ocr_tag_candidate(candidate)
        if not tag:
            return 0.0
        if len(tag) < 4:
            return 0.0
        if not re.search('[A-Z]', tag) or not re.search('\\d', tag):
            return 0.0
        generic_patterns = [
            '^[A-Z]{2,6}-\\d{1,5}(?:-[A-Z0-9]{1,5})?$',
            '^[A-Z]{2,6}\\d{2,5}(?:[A-Z]{0,2})?$',
            '^[A-Z]{2,6}-[A-Z0-9]{2,8}(?:-[A-Z0-9]{1,5})?$',
            '^\\d{1,4}[A-Z]{1,6}[-]?\\d{1,5}(?:[A-Z]{0,3})?(?:-[A-Z0-9]{1,5})?$',
            '^\\d{1,4}[A-Z]{2,6}\\d{2,5}(?:[A-Z]{1,4})?$',
            '^\\d{1,4}[A-Z]{2,6}\\d{1,4}[A-Z]{2,6}\\d{1,4}[A-Z]{2,6}\\d{1,4}$',
            '^\\d{1,4}[A-Z]{2,6}\\d{1,4}[A-Z]{2,6}\\d{1,4}[A-Z]{0,2}[A-Z]{2,6}\\d{1,4}$',
            '^\\d{1,4}[A-Z]{2,8}(?:\\d{1,4}[A-Z]{0,8}){1,4}(?:\\d{1,4})?(?:[-][A-Z0-9]{1,8}){0,2}$',
            '^\\d{1,4}[A-Z]{2,10}[-][A-Z]{0,4}\\d{1,4}[-]?\\d{0,4}$',
            '^\\d{1,4}[A-Z]{2,10}[-]\\d{1,4}[-][A-Z]{0,4}\\d{0,4}$',
            '^\\d{1,4}-[A-Z]{2,6}-\\d{1,5}[A-Z]?$',
            '^\\d{1,4}-[A-Z]{2,6}-\\d{1,5}-[A-Z0-9]{1,5}$',
        ]
        if not any((re.match(pattern, tag) for pattern in generic_patterns)):
            return 0.0
        score = 0.35
        profile_prefixes = io_profile.get('prefixes', set()) if io_profile else set()
        prefix = self._extract_tag_prefix(tag)
        if prefix and profile_prefixes:
            if prefix in profile_prefixes:
                score += 0.35
            else:
                close_prefix = any((abs(len(prefix) - len(ref_prefix)) <= 1 and self.distance(prefix, ref_prefix) <= 1 for ref_prefix in profile_prefixes))
                if close_prefix:
                    score += 0.22
        elif prefix:
            score += 0.2
        min_len = io_profile.get('min_len', 5) if io_profile else 5
        max_len = io_profile.get('max_len', 16) if io_profile else 16
        if min_len - 2 <= len(tag) <= max_len + 2:
            score += 0.15
        avg_num_len = io_profile.get('avg_num_len', 3.0) if io_profile else 3.0
        digit_parts = re.findall('\\d+', tag)
        if digit_parts:
            digit_score = max(0.0, 1.0 - abs(len(digit_parts[0]) - avg_num_len) / max(avg_num_len, 1.0))
            score += 0.1 * digit_score
        hyphen_ratio = io_profile.get('hyphen_ratio', 0.5) if io_profile else 0.5
        has_hyphen = '-' in tag
        if hyphen_ratio >= 0.5 and has_hyphen or (hyphen_ratio < 0.5 and (not has_hyphen)):
            score += 0.1
        return max(0.0, min(1.0, score))
