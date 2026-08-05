import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
import os
import gc
import fitz 
import tempfile

# ─────────────────────────────────────────────────────────────────
# COMPATIBILITY: Resolve gray colorspace across PyMuPDF versions
# fitz.cs_GRAY (new) | fitz.csGRAY (old) | fitz.COLORSPACE_GRAY (very old)
# ─────────────────────────────────────────────────────────────────
def _get_gray_colorspace():
    """Return gray colorspace constant, compatible across PyMuPDF versions."""
    for attr in ('cs_GRAY', 'csGRAY', 'COLORSPACE_GRAY'):
        if hasattr(fitz, attr):
            return getattr(fitz, attr)
    return None  # None = default (RGB)

_CS_GRAY = _get_gray_colorspace()
import logging
from multiprocessing import Pool, cpu_count
import Levenshtein
import time
import os
import tempfile
import math
import json
import traceback
from typing import List, Dict, Tuple, Set, Union, Any, Optional
import math
import string
import shutil
import random
import sys
try:
    from pdf_classifier import PDFClassifier as _PDFClassifierClass
except ImportError:
    try:
        from PDFClassifier import PDFClassifier as _PDFClassifierClass
    except ImportError:
        _PDFClassifierClass = None
# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from apps.backend.utils.file_utils import standardize_path, copy_to_output_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorMatcher:
    def __init__(self, similarity_threshold: float = 0.90):
        self.tag_vectors = {}  # Dictionary to store tag vectors
        self.similarity_threshold = similarity_threshold
        self.tag_patterns = {}  # Store tag patterns for quick lookup
        self.reference_tags = []  # ✅ Added: store reference tags separately
        # Initialize tracking variables
        self.match_attempts = 0
        self.successful_matches = 0
        self.match_scores = []
        self.required_columns = {
            'generated_excel': ['JB', 'MC', 'Tag/SPARE'],
            'io_list': ['Tag No', 'Tag', 'Tag No.','Tag no','tag','tag no', 'tag no.','TAG NO','TAG NO.','TAG','TAG.']
        }

    def add_reference_tag(self, tag: str) -> None:
        """Add a reference tag and create a vector for it."""
        if not tag or not isinstance(tag, str):
            return
        
        tag = str(tag).upper().strip()
        if not tag:
            return

        # ✅ store both in reference_tags and tag_vectors
        if tag not in self.reference_tags:
            self.reference_tags.append(tag)
        self.tag_vectors[tag] = self.create_tag_vector(tag)
        logging.info(f"Reference tag added: {tag}")

    def create_tag_vector(self, tag: str) -> 'Dict[str, float]':
        """Create an enhanced feature vector for a tag."""
        tag = str(tag).upper().strip()
        vector = {}

        # Basic features
        vector['length'] = len(tag)
        vector['num_digits'] = sum(c.isdigit() for c in tag)
        vector['num_letters'] = sum(c.isalpha() for c in tag)
        vector['num_hyphens'] = sum(c == '-' for c in tag)
        vector['num_parts'] = len(tag.split('-'))

        # Prefix features
        prefixes = ['TIT', 'FIT', 'PIT', 'LIT', 'TCV', 'FCV', 'PCV', 'LCV',
                   'UZSO', 'UZSC', 'UY', 'UHSL', 'UHSH', 'TY', 'LA']
        for prefix in prefixes:
            vector[f'prefix_{prefix}'] = 1.0 if tag.startswith(prefix) else 0.0

        # Part-based features
        parts = tag.split('-')
        if len(parts) >= 2:
            vector['first_part'] = sum(ord(c) for c in parts[0]) % 1000
            vector['last_part'] = sum(ord(c) for c in parts[-1]) % 1000
            vector['last_part_digits'] = sum(c.isdigit() for c in parts[-1])
            vector['first_part_length'] = len(parts[0])
            vector['last_part_length'] = len(parts[-1])
            vector['first_part_nonzero_digits'] = sum(1 for c in parts[0] if c.isdigit() and c != '0')
            vector['last_part_nonzero_digits'] = sum(1 for c in parts[-1] if c.isdigit() and c != '0')
            vector['standard_number_pattern'] = 1.0 if re.search(r'\d{3}-\d{2,3}', tag) else 0.0

        # ── MIDDLE PART FEATURES (CRITICAL for multi-part tags) ──────────
        # For tags like "9000-HDL-039" vs "9000-FGD-039":
        #   - first_part (9000) is identical
        #   - last_part (039) is identical
        #   - middle part (HDL vs FGD) is DIFFERENT but was being ignored!
        # This caused false matches with similarity ~1.000.
        # Now we hash the middle part(s) so different middle letters produce
        # different vectors, lowering the similarity score appropriately.
        if len(parts) >= 3:
            middle_parts = parts[1:-1]  # all parts except first and last
            middle_str = ''.join(middle_parts)
            vector['middle_part'] = sum(ord(c) for c in middle_str) % 1000
            vector['middle_part_length'] = len(middle_str)
            vector['middle_part_letters'] = sum(c.isalpha() for c in middle_str)
            vector['middle_part_digits'] = sum(c.isdigit() for c in middle_str)
            # Character-level hash: each character contributes a unique value
            # so that "HDL" and "FGD" produce different hashes even though
            # they have the same length and letter count.
            vector['middle_part_char_hash'] = sum(ord(c) * (i + 1) for i, c in enumerate(middle_str)) % 10000
        # ────────────────────────────────────────────────────────────────────

        # For 2-part tags, also hash the letter segment to distinguish
        # e.g. "HDL-039" from "FGD-039"
        if len(parts) == 2:
            # Find the letter segment in each part
            for idx, part in enumerate(parts):
                letters = ''.join(c for c in part if c.isalpha())
                if letters:
                    vector[f'part{idx}_letter_hash'] = sum(ord(c) * (i + 1) for i, c in enumerate(letters)) % 10000
                    vector[f'part{idx}_letter_length'] = len(letters)
        # ────────────────────────────────────────────────────────────────────

        vector['has_standard_format'] = 1.0 if re.match(r'[A-Z]{2,4}-\d{3}-\d{2,3}', tag) else 0.0

        # Digit sequence features
        digit_sequences = re.findall(r'\d+', tag)
        if digit_sequences:
            vector['first_digit_seq'] = int(digit_sequences[0]) if digit_sequences else 0
            vector['last_digit_seq'] = int(digit_sequences[-1]) if digit_sequences else 0
            vector['num_digit_sequences'] = len(digit_sequences)

        return vector

    def _calculate_digit_similarity(self, seq1: str, seq2: str) -> float:
            """Calculate similarity between digit sequences using character-wise comparison, tolerating OCR errors."""
            if not seq1 or not seq2: return 0.0
            matches = 0
            min_len = min(len(seq1), len(seq2))
            max_len = max(len(seq1), len(seq2))

            for i in range(min_len):
                if seq1[i] == seq2[i]:
                    matches += 1
                # OCR common errors: O/D -> 0, I/L/l -> 1, S -> 5, B -> 8
                elif seq2[i] in {'O', 'D'} and seq1[i] == '0': matches += 0.8
                elif seq2[i] in {'I', 'L', 'l'} and seq1[i] == '1': matches += 0.8
                elif seq2[i] == 'S' and seq1[i] == '5': matches += 0.8
                elif seq2[i] == 'B' and seq1[i] == '8': matches += 0.8
                elif seq1[i] in {'O', 'D'} and seq2[i] == '0': matches += 0.8 # Added reverse check
                elif seq1[i] in {'I', 'L', 'l'} and seq2[i] == '1': matches += 0.8 # Added reverse check
                elif seq1[i] == 'S' and seq2[i] == '5': matches += 0.8 # Added reverse check
                elif seq1[i] == 'B' and seq2[i] == '8': matches += 0.8 # Added reverse check

            return matches / max_len if max_len > 0 else 0.0

    def _are_digits_similar(self, digits1: str, digits2: str) -> bool:
            """Check if two digit sequences are similar accounting for OCR errors."""
            # This function is used for internal checks and is less aggressive than a final scoring.
            if abs(len(digits1) - len(digits2)) > min(len(digits1), len(digits2)) * 0.3: return False
            
            # Using the dedicated OCR tolerance calculation
            similarity_ratio = self._calculate_digit_similarity(digits1, digits2)
            
            return similarity_ratio >= 0.85

    def calculate_similarity(self, v1: 'Dict[str, float]', v2: 'Dict[str, float]') -> float:
        """Calculate cosine similarity between two tag vectors."""
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for key in set(v1) | set(v2):
            a = v1.get(key, 0.0)
            b = v2.get(key, 0.0)
            dot_product += a * b
            norm1 += a ** 2
            norm2 += b ** 2

        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1**0.5 * norm2**0.5)
        return max(0.0, min(1.0, similarity))


    def find_similar_tags(self, input_tag: str) -> 'List[Tuple[str, float]]':
        """Find similar tags with improved matching logic and more flexible thresholds."""
        if not input_tag:
            return []
                 
        self.match_attempts += 1
        input_tag = str(input_tag).upper().strip()
        input_vector = self.create_tag_vector(input_tag)
        similarities = []
        
        input_len = len(input_tag)
        input_parts_list = input_tag.split('-')
        input_parts = len(input_parts_list)
        input_digit_seqs = re.findall(r'\d+', input_tag)

        for ref_tag, ref_vector in self.tag_vectors.items():
            ref_len = len(ref_tag)
            if abs(ref_len - input_len) > max(5, ref_len * 0.5):
                continue

            ref_parts_list = ref_tag.split('-')
            ref_parts = len(ref_parts_list)
            if abs(ref_parts - input_parts) > 2:
                continue
            
            should_continue = False
            ref_digit_seqs = re.findall(r'\d+', ref_tag)
            if input_digit_seqs and ref_digit_seqs and len(input_digit_seqs[0]) > 1 and len(ref_digit_seqs[0]) > 1:
                digit_match = False
                for input_seq in input_digit_seqs:
                    for ref_seq in ref_digit_seqs:
                        if (input_seq == ref_seq or 
                            self._are_digits_similar(input_seq, ref_seq) or
                            (len(input_seq) > 2 and len(ref_seq) > 2 and 
                             (input_seq in ref_seq or ref_seq in input_seq or
                              self._calculate_digit_similarity(input_seq, ref_seq) > 0.85))):
                            digit_match = True
                            break
                    if digit_match:
                        break
                if not digit_match:
                    should_continue = True
            if should_continue:
                continue

            similarity = self.calculate_similarity(input_vector, ref_vector)

            # ── LAST PART EXACT CHECK (anti-false-positive gate #2) ──────────
            # Even if digits are similar, if the last part of the tag (after
            # the last dash) differs significantly, it's a different tag.
            # e.g. "LI-9001B1" vs "LI-7001" — last parts are "9001B1" vs "7001"
            # which are clearly different despite 75% digit similarity.
            input_last_part = input_parts_list[-1] if len(input_parts_list) >= 2 else ''
            ref_last_part = ref_parts_list[-1] if len(ref_parts_list) >= 2 else ''
            if input_last_part and ref_last_part and input_last_part != ref_last_part:
                try:
                    last_dist = Levenshtein.distance(input_last_part, ref_last_part)
                except Exception:
                    last_dist = sum(1 for a, b in zip(input_last_part, ref_last_part) if a != b)
                    last_dist += abs(len(input_last_part) - len(ref_last_part))
                if last_dist > 1:
                    # Different last parts — apply penalty proportional to distance
                    penalty = min(0.7, last_dist * 0.20)
                    similarity = similarity * (1.0 - penalty)
                    logger.debug(
                        "find_similar_tags: last part mismatch — input='%s' ref='%s' dist=%d penalty=%.2f → sim=%.4f",
                        input_last_part, ref_last_part, last_dist, penalty, similarity
                    )

            # ── MIDDLE PART EXACT CHECK (anti-false-positive gate) ──────────
            # Cosine similarity can be ~1.000 even when middle letters differ
            # (e.g. "9000-HDL-039" vs "9000-FGD-039" — same digits, same length,
            # but HDL ≠ FGD). This gate penalizes tags whose middle part(s) differ
            # significantly, preventing false matches.
            #
            # We use Levenshtein distance on the middle segments. If the distance
            # is more than 1 (allowing for 1-char OCR drift), we heavily penalize
            # the similarity score.
            if len(input_parts_list) >= 3 and len(ref_parts_list) >= 3:
                input_middle = ''.join(input_parts_list[1:-1])
                ref_middle = ''.join(ref_parts_list[1:-1])
                if input_middle and ref_middle:
                    try:
                        middle_dist = Levenshtein.distance(input_middle, ref_middle)
                    except Exception:
                        # Fallback: simple character comparison
                        middle_dist = sum(1 for a, b in zip(input_middle, ref_middle) if a != b)
                        middle_dist += abs(len(input_middle) - len(ref_middle))
                    if middle_dist > 1:
                        # Different middle parts — apply heavy penalty
                        # 2-char diff → ~50% penalty, 3+ char diff → ~80% penalty
                        penalty = min(0.8, middle_dist * 0.25)
                        similarity = similarity * (1.0 - penalty)
                        logger.debug(
                            "find_similar_tags: middle part mismatch — input='%s' ref='%s' dist=%d penalty=%.2f → sim=%.4f",
                            input_middle, ref_middle, middle_dist, penalty, similarity
                        )
            # ────────────────────────────────────────────────────────────────────

            lower_threshold = max(0.6, self.similarity_threshold * 0.85)
            if similarity >= lower_threshold:
                similarities.append((ref_tag, similarity))
                self.match_scores.append(similarity)

        if similarities:
            self.successful_matches += 1
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def _are_digits_similar(self, digits1: str, digits2: str) -> bool:
        """Check if two digit sequences are similar accounting for OCR errors."""
        # If lengths are too different, they're not similar
        if abs(len(digits1) - len(digits2)) > min(len(digits1), len(digits2)) * 0.3:
            return False

        # Check for common OCR confusions
        substitutions = {
            '0': ['O', 'D'],
            'O': ['0', 'D'],
            'D': ['0', 'O'],
            '1': ['I', 'L', 'l'],
            'I': ['1', 'L', 'l'],
            'L': ['1', 'I', 'l'],
            'l': ['1', 'I', 'L'],
            '5': ['S'],
            'S': ['5'],
            '8': ['B'],
            'B': ['8']
        }

        matches = 0
        for c1, c2 in zip(digits1, digits2):
            if c1 == c2:
                matches += 1
            elif c2 in substitutions.get(c1, []):
                matches += 1

        similarity_ratio = matches / max(len(digits1), len(digits2))
        return similarity_ratio >= 0.9 # Acceptable OCR similarity threshold
        
class TagJBExtractor:
    """
    کلاسی برای استخراج تگ‌ها و شناسه‌های JB از نمودارهای PDF و تطبیق آن‌ها با داده‌های اکسل.
    """
    
    def __init__(self, tesseract_path: 'Optional[str]' = None, excel_path: 'Optional[str]' = None):
        """Initialize the extractor with optional tesseract and excel paths."""
        if tesseract_path:
            if not os.path.exists(tesseract_path):
                raise ValueError(f"Provided Tesseract path does not exist: {tesseract_path}")
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            common_locations = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract'
            ]
            
            tesseract_found = False
            for location in common_locations:
                if os.path.exists(location):
                    pytesseract.pytesseract.tesseract_cmd = location
                    tesseract_found = True
                    break
                    
            if not tesseract_found:
                raise RuntimeError("Tesseract not found in common locations. Please provide tesseract_path.")
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tesseract: {e}")
        
        self.jb_pattern = re.compile(r'JB-\d+', re.IGNORECASE)
        self.vector_matcher = VectorMatcher()
        self.all_tags = set()
        self.matched_tags = set()
        self.all_jbs = set()
        self.all_mcs = set()
        self.all_spares = []
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0
        self.similarity_reports = [] 
        self.latest_pattern_unmatched_candidates = []
        self.latest_pattern_unmatched_details = []
        # 🆕 Warnings collected per-run:
        #   - JB not found on a page that has tags/spares
        #   - Same JB appearing on multiple pages
        #   - Same tag appearing on multiple pages
        #   - Pages skipped due to multiple JBs
        # Each entry: {type, item, pages, tag_count, severity, description, action}
        self.latest_warnings = []
        # Per-page warnings accumulated during _process_single_page_data
        # (merged into latest_warnings at the end of run_with_annotated_pdf)
        self._page_warnings = []
        
        # تنظیم مقادیر پیش‌فرض الگوها (به عنوان رشته)
        self.jb_examples = None
        self.mc_examples = None
        self.spare_examples = None
        # Multi-pattern support (lists of prefixes)
        self.jb_examples_list = []
        self.mc_examples_list = []
        self.spare_examples_list = []
        self.cable_examples = None
        self.wire_color_rule = None
        self.scr_number_rule = None
        
        # الگوهای regex کامپایل شده
        self.jb_regex = None
        self.mc_regex = None
        self.spare_regex = None
        self.terminal_pattern = None
        self.terminal_pattern_dict = {}
        
        # کامپایل اولیه الگوها
        self._compile_regex_patterns()
        self._classifier = None
        self._current_pdf_type = "diagrams"
        self.document_type_by_path = {}
        self.document_nature_by_path = {}

    def set_classifier(self, classifier) -> None:
        """
        Inject a PDFClassifier instance for automatic diagram/table routing.
        Call once after construction, before processing any PDFs.
 
        Args:
            classifier: A PDFClassifier instance (from PDFClassifier.py).
                        Pass None to disable classification (diagram mode only).
        """
        self._classifier = classifier
        logger.info(
            "PDFClassifier injected: %s",
            type(classifier).__name__ if classifier is not None else "None (diagram-only mode)"
        )
        
    def build_tag_vectors_from_excel(self, excel_path: str) -> None:
        """
        Build tag vectors from Excel file's Tag NO column.

        Args:
            excel_path: Path to Excel file containing tags
        """
        try:
            logger.info("="*70)
            logger.info(f"🔄 Building tag vectors from Excel: {excel_path}")
            logger.info("="*70)

            # بررسی وجود فایل
            if not os.path.exists(excel_path):
                logger.error(f"❌ Excel file not found: {excel_path}")
                return
            
            # Read Excel file
            df = pd.read_excel(excel_path)
            logger.info(f"✅ Excel file loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"   Columns: {df.columns.tolist()}")

            # بررسی وجود ستون Tag No
            if 'Tag No' not in df.columns:
                logger.error("❌ Excel file must contain a 'Tag No' column")
                logger.error(f"   Available columns: {df.columns.tolist()}")
                raise ValueError("Excel file must contain a 'Tag No' column")

            # Extract and clean tags
            tags = df['Tag No'].dropna().astype(str).str.strip().str.upper().unique()
            logger.info(f"✅ Extracted {len(tags)} unique tags from 'Tag No' column")
            
            # نمایش نمونه
            logger.info(f"   Sample tags: {tags[:5].tolist()}")

            # Add tags to vector matcher
            added_count = 0
            for tag in tags:
                self.vector_matcher.add_reference_tag(tag)
                added_count += 1

            logger.info(f"✅ Added {added_count} tags to vector matcher")

            # بررسی نهایی
            if hasattr(self.vector_matcher, 'reference_tags'):
                final_count = len(self.vector_matcher.reference_tags)
                logger.info(f"✅ Vector matcher now contains {final_count} reference tags")
            else:
                logger.error("❌ Vector matcher has no reference_tags attribute!")

            # Create tag patterns for later use
            instrument_prefixes = ['TIT', 'FIT', 'PIT', 'LIT', 'TCV', 'FCV', 'PCV', 'LCV']
            tag_patterns = []

            for tag in tags:
                pattern_parts = [
                    len(tag),
                    sum(c.isdigit() for c in tag),
                    sum(c.isalpha() for c in tag),
                    sum(c == '-' for c in tag),
                    1 if any(tag.startswith(prefix) for prefix in instrument_prefixes) else 0,
                    len(tag.split('-'))
                ]
                tag_patterns.append(pattern_parts)

            self.tag_patterns = {tag: pattern for tag, pattern in zip(tags, tag_patterns)}
            logger.info(f"✅ Created tag patterns for {len(self.tag_patterns)} tags")

            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"❌ Error building tag vectors: {e}")
            logger.error(traceback.format_exc())
            raise

    def calculate_tag_similarity(self, tag1: str, tag2: str) -> float:
        """
        محاسبه شباهت دقیق بین دو تگ (با توجه به ساختار و بخش‌های عددی و متنی)
        """
        try:
            tag1_str = str(tag1).strip().upper()
            tag2_str = str(tag2).strip().upper()

            # تطابق کامل
            if tag1_str == tag2_str:
                return 1.0

            tag1_parts = re.split(r'[-_]', tag1_str)
            tag2_parts = re.split(r'[-_]', tag2_str)

            # اگر اختلاف ساختار زیاد باشد
            if abs(len(tag1_parts) - len(tag2_parts)) > 1:
                return 0.0

            total_score = 0.0
            weights = [0.4, 0.3, 0.3]
            for i in range(min(len(tag1_parts), len(tag2_parts))):
                p1, p2 = tag1_parts[i], tag2_parts[i]
                part_score = self.calculate_string_similarity(p1, p2)

                # اگر بخش‌ها عددی بودن → دقیق‌تر بسنج
                if p1.isdigit() and p2.isdigit():
                    diff = abs(int(p1) - int(p2))
                    if diff == 0:
                        part_score = 1.0
                    elif diff <= 5:
                        part_score = 0.85
                    elif diff <= 20:
                        part_score = 0.7
                    else:
                        part_score = 0.4

                # پیشوند متفاوت (مثلاً PCV vs FCV)
                if i == 0 and part_score < 0.7:
                    total_score += weights[i] * 0.5  # جریمه بزرگ
                else:
                    total_score += weights[i] * part_score

            return round(max(0.0, min(1.0, total_score)), 3)

        except Exception as e:
            logger.error(f"Error in tag similarity calculation: {e}")
            return 0.0


    def _calculate_vector_similarity_for_tags(self, tag1: str, tag2: str) -> float:
        """
        محاسبه شباهت وکتوری بین دو تگ
        
        Args:
            tag1: تگ اول
            tag2: تگ دوم
            
        Returns:
            امتیاز شباهت وکتوری
        """
        # تبدیل تگ‌ها به بردارهای ویژگی
        vec1 = self.tag_to_vector(tag1)
        vec2 = self.tag_to_vector(tag2)
        
        # محاسبه شباهت وکتوری
        return self.calculate_vector_similarity(vec1, vec2)

    def calculate_string_similarity(self, s1: str, s2: str) -> float:
        """
        محاسبه شباهت دقیق دو رشته (سخت‌گیرانه‌تر از difflib)
        """
        s1 = s1.strip().upper()
        s2 = s2.strip().upper()

        if s1 == s2:
            return 1.0

        # طول‌ها خیلی متفاوت → شباهت پایین
        if abs(len(s1) - len(s2)) > 3:
            return 0.5

        import difflib
        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()

        # جریمه‌ی شروع متفاوت
        if s1 and s2 and s1[0] != s2[0]:
            ratio *= 0.8

        # اگر فقط چند کاراکتر متفاوت باشند، نمره بالا
        if ratio > 0.97 and s1 != s2:
            ratio = 0.95  # هیچ رشته‌ی متفاوتی نباید 1.0 یا نزدیکش بشه

        return round(ratio, 3)

    def calculate_numeric_similarity(self, tag1: str, tag2: str) -> float:
        """
        محاسبه شباهت بخش‌های عددی دو تگ
        
        Args:
            tag1: تگ اول
            tag2: تگ دوم
            
        Returns:
            امتیاز شباهت بین 0 تا 1 (1 برای تگ‌های با بخش‌های عددی یکسان)
        """
        import re
        
        # استخراج بخش‌های عددی
        numbers1 = [int(x) for x in re.findall(r'\d+', tag1)]
        numbers2 = [int(x) for x in re.findall(r'\d+', tag2)]
        
        if not numbers1 or not numbers2:
            return 0.0
            
        # محاسبه شباهت بر اساس اشتراک بخش‌های عددی
        common_numbers = set(numbers1).intersection(set(numbers2))
        all_numbers = set(numbers1).union(set(numbers2))
        
        if not all_numbers:
            return 0.0
            
        return len(common_numbers) / len(all_numbers)

    def calculate_parts_similarity(self, tag1: str, tag2: str) -> float:
        """
        محاسبه شباهت دقیق بین بخش‌های جدا شده با '-'
        """
        tag1 = tag1.strip().upper()
        tag2 = tag2.strip().upper()

        if tag1 == tag2:
            return 1.0

        parts1 = tag1.split('-')
        parts2 = tag2.split('-')

        if abs(len(parts1) - len(parts2)) > 1:
            return 0.4

        total_score = 0.0
        matched_parts = 0

        for p1, p2 in zip(parts1, parts2):
            total_score += self.calculate_string_similarity(p1, p2)
            matched_parts += 1

        avg_sim = total_score / max(matched_parts, 1)

        # اگر نوع تجهیز (بخش اول) خیلی فرق داشته باشه → جریمه سنگین
        if parts1 and parts2 and self.calculate_string_similarity(parts1[0], parts2[0]) < 0.7:
            avg_sim *= 0.7

        return round(max(0.0, min(1.0, avg_sim)), 3)

    # تابع جایگزین برای اجرای find_candidate_tags روی کل لیست
    def find_candidate_tags_wrapper(self, ocr_text_items: list, io_list_tags: list, threshold: float) -> dict:
        results = {}
        for item in ocr_text_items:
            results[item] = self.find_candidate_tags(item, io_list_tags, threshold)
        return results

    def extract_and_match_tags(self, ocr_text_items: list, io_list_tags: list, threshold: float = 0.75) -> Tuple[dict, dict]:
        """
        [اصلاح نهایی] تطبیق تگ‌ها و تفکیک آن‌ها به Matched و Unmatched.
        
        خروجی: (Matched_tags, Unmatched_ocr_tags)
        """
        matched_tags = {}
        unmatched_ocr_tags = {}
        
        # ۱. پیدا کردن کاندیداها برای تمام آیتم‌های OCR
        tag_candidates = self.find_candidate_tags_wrapper(ocr_text_items, io_list_tags, threshold)
        
        # ۲. پردازش نهایی نتایج برای Matched و Unmatched
        for ocr_text in ocr_text_items:
            candidates = tag_candidates.get(ocr_text, [])
            
            # در ابتدا، فرض می‌کنیم تگ Unmatched است.
            is_matched = False
            
            if candidates:
                best_match, best_similarity = candidates[0]
                
                # اگر بهترین امتیاز تطبیق از آستانه بالاتر باشد: Matched
                if best_similarity >= threshold: 
                    matched_tags[ocr_text] = best_match
                    is_matched = True
            
            # اگر Matched نشد، آن را در لیست Unmatched قرار بده.
            if not is_matched:
                unmatched_ocr_tags[ocr_text] = "No match found in IO list"

        return matched_tags, unmatched_ocr_tags

    def process_ocr_results_with_io_list(self, ocr_results: list, io_list_tags: list) -> Tuple[dict, dict]:
        """
        [اصلاح نهایی] نقطه ورود اصلی برای پردازش OCR و تفکیک خروجی.
        """
        ocr_text_items = []
        for result in ocr_results:
            if isinstance(result, dict) and 'text' in result:
                ocr_text_items.append(result['text'])
            elif isinstance(result, str):
                ocr_text_items.append(result)
        
        ocr_text_items = [text.strip() for text in ocr_text_items if text.strip()]
        ocr_text_items = list(set(ocr_text_items))

        logger.info(f"Processing {len(ocr_text_items)} unique OCR text items")
        
        # تنظیم آستانه تطبیق نهایی به 0.75
        matched_tags, unmatched_ocr_tags = self.extract_and_match_tags(ocr_text_items, io_list_tags, threshold=0.75)
        
        return matched_tags, unmatched_ocr_tags

    def find_candidate_tags(self, query_tag: str, io_tags: List[str], 
                        final_match_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """
        ✅ FIX: شرط خاص برای UZSO vs UZSC
        """
        logger.info(f"Finding candidates for: {query_tag}")
        
        if not query_tag:
            return []
        
        query_tag_upper = str(query_tag).strip().upper()
        
        # ✅ FIX: اگر query شبیه UZSO یا UZSC است
        if re.match(r'^[UuVv][ZzSs2][Ss5][O0oC][-_]?\d+', query_tag_upper):
            logger.info(f"🔍 UZSO/UZSC pattern detected in query: {query_tag_upper}")
            
            # استخراج شماره
            number_match = re.search(r'(\d+)', query_tag_upper)
            if number_match:
                number = number_match.group(1)
                
                # ساخت هر دو حالت
                candidate_uzso = f"UZSO-{number}"
                candidate_uzsc = f"UZSC-{number}"
                
                results = []
                
                # چک کردن هر دو در IO List
                if candidate_uzso in io_tags:
                    # محاسبه شباهت دقیق
                    sim = self._calculate_final_similarity_score(query_tag_upper, candidate_uzso)
                    results.append((candidate_uzso, sim))
                    logger.info(f"   Found UZSO candidate: {candidate_uzso} (score: {sim:.3f})")
                
                if candidate_uzsc in io_tags:
                    sim = self._calculate_final_similarity_score(query_tag_upper, candidate_uzsc)
                    results.append((candidate_uzsc, sim))
                    logger.info(f"   Found UZSC candidate: {candidate_uzsc} (score: {sim:.3f})")
                
                # اگر هر دو پیدا شدند، بر اساس کاراکتر چهارم تصمیم بگیریم
                if len(results) == 2:
                    fourth_char = query_tag_upper[3] if len(query_tag_upper) > 3 else ''
                    
                    if fourth_char in ['O', '0']:
                        # ترجیح به UZSO
                        results = [r for r in results if r[0].startswith('UZSO')]
                        logger.info(f"   Selected UZSO based on 4th character: '{fourth_char}'")
                    elif fourth_char == 'C':
                        # ترجیح به UZSC
                        results = [r for r in results if r[0].startswith('UZSC')]
                        logger.info(f"   Selected UZSC based on 4th character: '{fourth_char}'")
                    else:
                        # نمی‌دانیم - هر دو را نگه دار و بر اساس score انتخاب کن
                        logger.warning(f"   Ambiguous 4th char: '{fourth_char}', keeping both candidates")
                
                if results:
                    results.sort(key=lambda x: x[1], reverse=True)
                    return results
        
        # روش عادی برای بقیه تگ‌ها
        similarity_scores: List[Tuple[str, float]] = []
        
        query_prefix, query_first_num, _, _ = self._split_tag_to_parts(query_tag)
        
        for io_tag in io_tags:
            if not io_tag or pd.isna(io_tag):
                continue
            io_tag_str = str(io_tag).strip().upper()
            io_prefix, io_first_num, _, _ = self._split_tag_to_parts(io_tag_str)

            prefix_sim = self._calculate_prefix_similarity(query_prefix, io_prefix)
            if prefix_sim < 0.7:
                continue
            
            similarity = self._calculate_final_similarity_score(query_tag, io_tag)
            
            if similarity >= final_match_threshold:
                similarity_scores.append((io_tag, similarity))

        sorted_candidates = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        filtered_candidates = [c for c in sorted_candidates if c[1] >= final_match_threshold]
        
        return filtered_candidates

    def _calculate_numeric_part_similarity(self, num1, num2):
            """
            [اصلاح: سخت‌گیری بیشتر روی اختلاف عددی واقعی]
            """
            if not num1 and not num2: return 1.0
            if not num1 or not num2: return 0.2

            ocr_sim = self.vector_matcher._calculate_digit_similarity(num1, num2)
            
            # 1. تطابق کامل (در صورت یکسان بودن رشته‌ها)
            if num1 == num2: 
                return 1.0
            # 2. تطابق تقریباً کامل OCR (مثل O10 vs 010)
            if ocr_sim >= 0.95: 
                return 0.99 
                
            # 3. محاسبه اختلاف عددی
            n1_clean = re.sub(r'[^0-9]', '', num1).lstrip('0') or '0'
            n2_clean = re.sub(r'[^0-9]', '', num2).lstrip('0') or '0'
            
            try:
                n1 = int(n1_clean)
                n2 = int(n2_clean)
                diff = abs(n1 - n2)
                
                # 🛑 سختگیری: اگر اختلاف زیاد است، امتیاز را به شدت کاهش بده.
                if diff > 5:
                    # 74 vs 81 (diff=7) اینجا می‌افتد. امتیاز باید کم باشد.
                    # از 0.4 شروع می‌شود و با اختلاف بیشتر، کمتر می‌شود.
                    if diff > 15:
                        return 0.1 
                    return 0.4 * (1.0 - (diff / max(n1, n2, 1) * 0.5)) # امتیاز را به زیر 0.4 می‌آورد

                # تلرانس کوچک عددی (1 تا 5):
                return max(ocr_sim, 0.9 - (diff * 0.1)) # امتیاز در محدوده 0.9 تا 0.5 می‌ماند
                
            except ValueError:
                pass

            return ocr_sim # در بدترین حالت، فقط شباهت کاراکتری
    def _extract_tag_prefix(self, tag: str) -> str:
        """استخراج پیشوند تگ (قبل از اولین '-' یا اعداد)"""
        # مثال: PDIT-100-11 → PDIT
        #       FIT100-A → FIT
        match = re.match(r'^([A-Z]+)', tag.upper())
        return match.group(1) if match else ''

    def _calculate_numeric_similarity(self, tag1: str, tag2: str) -> float:
        """محاسبه شباهت بخش‌های عددی"""
        nums1 = re.findall(r'\d+', tag1)
        nums2 = re.findall(r'\d+', tag2)
        
        if len(nums1) != len(nums2):
            return 0.0
        
        total_similarity = 0.0
        for n1, n2 in zip(nums1, nums2):
            if n1 == n2:
                total_similarity += 1.0
            else:
                # تلرانس ±2 برای اعداد
                try:
                    diff = abs(int(n1) - int(n2))
                    if diff <= 2:
                        total_similarity += 0.9
                    elif diff <= 5:
                        total_similarity += 0.7
                except:
                    pass
        
        return total_similarity / len(nums1) if nums1 else 0.0

    def _calculate_prefix_similarity(self, prefix1, prefix2):
        """محاسبه میزان شباهت بین دو پیشوند با Levenshtein (تابع اصلی حفظ می‌شود)."""
        # ... (Original implementation)
        try:
            if prefix1 == prefix2: return 1.0
            if not prefix1 or not prefix2: return 0.0
            distance = Levenshtein.distance(prefix1, prefix2)
            max_len = max(len(prefix1), len(prefix2))
            if max_len == 0: return 1.0
            return 1.0 - (distance / max_len)
        except Exception as e:
            logger.error(f"Error calculating prefix similarity: {e}")
            return 0.0

    def _calculate_final_similarity_score(self, tag1, tag2):
        """
        [اصلاح نهایی] حذف جریمه‌های سختگیرانه برای پیشوند و افزایش وزن اعداد.
        """
        try:
            prefix1, first_num1, second_num1, suffix1 = self._split_tag_to_parts(tag1)
            prefix2, first_num2, second_num2, suffix2 = self._split_tag_to_parts(tag2)
            
            prefix_sim = self._calculate_prefix_similarity(prefix1, prefix2)
            first_num_sim = self._calculate_numeric_part_similarity(first_num1, first_num2)
            second_num_sim = self._calculate_numeric_part_similarity(second_num1, second_num2)
            suffix_sim = self._calculate_prefix_similarity(suffix1, suffix2)

            # تنظیم وزن‌ها: وزن پیشوند کمی کمتر شد و وزن بخش عددی اصلی افزایش یافت.
            if second_num1 and second_num2 and len(second_num1) > 0 and len(second_num2) > 0:
                final_score = (
                    0.35 * prefix_sim +
                    0.35 * first_num_sim +
                    0.25 * second_num_sim +
                    0.05 * suffix_sim
                )
            else:
                final_score = (
                    0.35 * prefix_sim +      
                    0.60 * first_num_sim +   
                    0.05 * suffix_sim
                )

            # 🛑 جریمه ضعیف بودن پیشوند (prefix_sim < 0.7) کاملاً حذف شد.
                
            # جریمه‌ی اختلاف طول (ضریب جریمه از 0.5 به 0.3 کاهش یافت - ملایم‌تر شد)
            len_diff_ratio = abs(len(tag1) - len(tag2)) / max(len(tag1), len(tag2), 1)
            if len_diff_ratio > 0.3:
                final_score *= (1.0 - len_diff_ratio * 0.3) 
                
            return max(0.0, min(1.0, round(final_score, 4)))
            
        except Exception as e:
            logger.error(f"Error in final similarity calculation for {tag1} vs {tag2}: {e}")
            return 0.0

    def _split_tag_to_parts(self, tag):
        try:
            tag = str(tag).strip().upper()
            
            # 1. Type-Number-NumberSuffix (e.g., FIT-101-01A)
            pattern1 = r'^([A-Z]+)-(\d+)-(\d+)([A-Z]*)$'
            
            # 2. Type-Number-AlphaNumeric Suffix (e.g., FCV-101-A or FCV-101-02)
            # این الگو به خصوص برای پسوندهای تک حرفی (مثل A, B) مهم است.
            pattern4 = r'^([A-Z]+)-(\d+)-([A-Z\d]+)$' 
            
            # 3. Type-Number-Suffix (e.g., PIT-101A)
            pattern2 = r'^([A-Z]+)-(\d+)([A-Z]*)$'
            
            # 4. TypeNumberSuffix (e.g., PIT101A)
            pattern3 = r'^([A-Z]+)(\d+)([A-Z]*)$'
            
            # ترتیب بررسی مهم است (پیچیده‌ترین‌ها اول)
            
            match = re.match(pattern1, tag)
            if match: 
                # Prefix, FirstNum, SecondNum, Suffix
                return match.groups()
                
            match = re.match(pattern4, tag)
            if match: 
                prefix, first_num, last_part = match.groups()
                # فرض می‌کنیم اگر بخش آخر عدد باشد، SecondNum است. در غیر این صورت، Suffix است.
                if last_part.isdigit():
                     return prefix, first_num, last_part, "" # Prefix, FirstNum, SecondNum, Suffix
                else:
                     return prefix, first_num, "", last_part # Prefix, FirstNum, SecondNum(Empty), Suffix
            
            match = re.match(pattern2, tag)
            if match: 
                return match.groups()[0], match.groups()[1], "", match.groups()[2] # Prefix, FirstNum, SecondNum(Empty), Suffix
            
            match = re.match(pattern3, tag)
            if match: 
                return match.groups()[0], match.groups()[1], "", match.groups()[2] # Prefix, FirstNum, SecondNum(Empty), Suffix
            
            # اگر هیچ الگوی استانداردی منطبق نبود، جداسازی ساده
            prefix = ''.join(c for c in tag if c.isalpha())
            numbers = ''.join(c for c in tag if c.isdigit())
            # این منطق ساده نمی‌تواند پسوند را به درستی جدا کند
            return prefix, numbers, "", "" 
            
        except Exception as e:
            logger.error(f"Error splitting tag {tag}: {e}")
            return "", "", "", ""

    def _normalize_ocr_tag_candidate(self, text: str) -> str:
        """Normalize raw OCR token so it can be compared against IO-list tag patterns."""
        if not text:
            return ""
        normalized = str(text).strip().upper()
        normalized = re.sub(r'\s+', '', normalized)
        normalized = normalized.replace('_', '-')
        normalized = normalized.strip("-.")
        return normalized

    def _build_io_pattern_profile(self, io_tags: 'Set[str]') -> Dict[str, Any]:
        """
        Build a lightweight pattern profile from IO list to detect tag-like OCR text,
        even when the exact tag does not exist in IO list.
        """
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

            # Extract prefix — try letter-first, then digit+letter combination
            prefix_match = re.match(r'^([A-Z]{2,6})', tag)
            if prefix_match:
                prefixes.add(prefix_match.group(1))
            else:
                # For digit-starting tags (e.g. 21HS-001, 11SAM10...), extract the
                # letter segment after the leading digits as the "prefix family".
                # e.g. 21HS-001 → HS, 11SAM10AN020 → SAM, 1M1 → M
                digit_letter_match = re.match(r'^\d{1,4}([A-Z]{1,6})', tag)
                if digit_letter_match:
                    prefixes.add(digit_letter_match.group(1))

            for num_part in re.findall(r'\d+', tag):
                numeric_lengths.append(len(num_part))

        if not prefixes:
            # fallback to common instrumentation prefixes
            prefixes = {
                'UZSO', 'UZSC', 'FIT', 'PIT', 'TIT', 'LIT', 'FCV', 'PCV',
                'TCV', 'LCV', 'TY', 'LA', 'UY', 'UHSL', 'UHSH'
            }

        min_len = min(lengths) if lengths else 5
        max_len = max(lengths) if lengths else 16
        avg_num_len = (sum(numeric_lengths) / len(numeric_lengths)) if numeric_lengths else 3.0
        hyphen_ratio = (hyphen_count / len(lengths)) if lengths else 0.5

        return {
            'prefixes': prefixes,
            'min_len': min_len,
            'max_len': max_len,
            'avg_num_len': avg_num_len,
            'hyphen_ratio': hyphen_ratio
        }

    def _score_pattern_candidate(self, candidate: str, io_profile: Dict[str, Any]) -> float:
        """
        Score how likely a token is a valid tag based on IO-list-derived structure.
        """
        tag = self._normalize_ocr_tag_candidate(candidate)
        if not tag:
            return 0.0
        
        if self._is_non_tag_pattern(tag):
            return 0.0

        if len(tag) < 4:
            return 0.0
        if not re.search(r'[A-Z]', tag) or not re.search(r'\d', tag):
            return 0.0

        generic_patterns = [
            # Letter-starting tags (original)
            r'^[A-Z]{2,6}-\d{1,5}(?:-[A-Z0-9]{1,5})?$',
            r'^[A-Z]{2,6}\d{2,5}(?:[A-Z]{0,2})?$',
            r'^[A-Z]{2,6}-[A-Z0-9]{2,8}(?:-[A-Z0-9]{1,5})?$',
            # Digit-starting tags — simple (e.g. 21HS-001, 1M1)
            r'^\d{1,4}[A-Z]{1,6}[-]?\d{1,5}(?:[A-Z]{0,3})?(?:-[A-Z0-9]{1,5})?$',
            r'^\d{1,4}[A-Z]{2,6}\d{2,5}(?:[A-Z]{1,4})?$',
            # ── Siemens-style long tags (e.g. 11SAM10AN020XB91) ──
            # Pattern: digits + letters + digits + letters + digits + letters + digits
            # This is the KKS naming convention used by Siemens.
            # 11SAM10AN020XB91 = 11+SAM+10+AN+020+XB+91
            r'^\d{1,4}[A-Z]{2,6}\d{1,4}[A-Z]{2,6}\d{1,4}[A-Z]{2,6}\d{1,4}$',
            # Siemens with terminal suffix (e.g. 11MBL30AT050AXB91 = 11+MBL+30+AT+050A+XB+91)
            r'^\d{1,4}[A-Z]{2,6}\d{1,4}[A-Z]{2,6}\d{1,4}[A-Z]{0,2}[A-Z]{2,6}\d{1,4}$',
            # Flexible catch-all for mixed alphanumeric (segments of digits+letters, possibly with dashes)
            r'^\d{1,4}[A-Z]{2,8}(?:\d{1,4}[A-Z]{0,8}){1,4}(?:\d{1,4})?(?:[-][A-Z0-9]{1,8}){0,2}$',
            # With dashes between segments (e.g. 11HOTSPARE-DO22-03, 11SPARE-DO21-15)
            r'^\d{1,4}[A-Z]{2,10}[-][A-Z]{0,4}\d{1,4}[-]?\d{0,4}$',
            r'^\d{1,4}[A-Z]{2,10}[-]\d{1,4}[-][A-Z]{0,4}\d{0,4}$',
        ]

        if not any(re.match(pattern, tag) for pattern in generic_patterns):
            return 0.0

        score = 0.35
        profile_prefixes = io_profile.get('prefixes', set()) if io_profile else set()
        prefix = self._extract_tag_prefix(tag)

        if prefix and profile_prefixes:
            if prefix in profile_prefixes:
                score += 0.35
            else:
                # tolerate one-char OCR drift in prefix
                close_prefix = any(
                    abs(len(prefix) - len(ref_prefix)) <= 1 and
                    Levenshtein.distance(prefix, ref_prefix) <= 1
                    for ref_prefix in profile_prefixes
                )
                if close_prefix:
                    score += 0.22
        elif prefix:
            score += 0.2

        min_len = io_profile.get('min_len', 5) if io_profile else 5
        max_len = io_profile.get('max_len', 16) if io_profile else 16
        if min_len - 2 <= len(tag) <= max_len + 2:
            score += 0.15

        avg_num_len = io_profile.get('avg_num_len', 3.0) if io_profile else 3.0
        digit_parts = re.findall(r'\d+', tag)
        if digit_parts:
            digit_score = max(0.0, 1.0 - (abs(len(digit_parts[0]) - avg_num_len) / max(avg_num_len, 1.0)))
            score += 0.1 * digit_score

        hyphen_ratio = io_profile.get('hyphen_ratio', 0.5) if io_profile else 0.5
        has_hyphen = '-' in tag
        if (hyphen_ratio >= 0.5 and has_hyphen) or (hyphen_ratio < 0.5 and not has_hyphen):
            score += 0.1

        return max(0.0, min(1.0, score))

    def validate_tag_candidates(self, query_tag, candidates):
        """
        اعتبارسنجی و فیلتر کردن کاندیداهای تگ با قوانین سخت‌گیرانه‌تر
        
        Args:
            query_tag: تگ مورد جستجو
            candidates: لیست کاندیداها با امتیازات آنها
        
        Returns:
            لیست فیلتر شده کاندیداها
        """
        logger.info(f"CANDIDATE_FUNCTION_CALLED: Validating {len(candidates)} candidates for tag '{query_tag}'")

        try:
            if not candidates:
                return []
                
            logger.debug(f"Validating {len(candidates)} candidates for tag {query_tag}")
            
            # فیلتر کردن کاندیداها با قوانین سخت‌گیرانه‌تر
            valid_candidates = []
            query_parts = self._split_tag_to_parts(query_tag)
            query_prefix, query_first_num, query_second_num, query_suffix = query_parts
            
            for candidate, score in candidates:
                candidate_parts = self._split_tag_to_parts(candidate)
                candidate_prefix, candidate_first_num, candidate_second_num, candidate_suffix = candidate_parts
                
                # قانون 1: پیشوندها باید یکسان باشند (نه فقط شبیه)
                if query_prefix.upper() != candidate_prefix.upper():
                    logger.debug(f"  Rejecting {candidate} due to different prefix: {query_prefix} != {candidate_prefix}")
                    continue
                
                # قانون 2: اگر query_tag عدد اول دارد، candidate هم باید داشته باشد و باید یکسان باشند
                if query_first_num:
                    if not candidate_first_num:
                        logger.debug(f"  Rejecting {candidate} due to missing first number")
                        continue
                    if not self._are_numeric_parts_very_similar(query_first_num, candidate_first_num):
                        logger.debug(f"  Rejecting {candidate} due to different first number: {query_first_num} != {candidate_first_num}")
                        continue
                
                # قانون 3: اگر query_tag عدد دوم دارد، candidate هم باید داشته باشد و باید یکسان باشند
                if query_second_num:
                    if not candidate_second_num:
                        logger.debug(f"  Rejecting {candidate} due to missing second number")
                        continue
                    if not self._are_numeric_parts_very_similar(query_second_num, candidate_second_num):
                        logger.debug(f"  Rejecting {candidate} due to different second number: {query_second_num} != {candidate_second_num}")
                        continue
                
                # قانون 4: اگر پسوند وجود دارد، باید یکسان باشد
                if query_suffix and candidate_suffix:
                    if query_suffix.upper() != candidate_suffix.upper():
                        logger.debug(f"  Rejecting {candidate} due to different suffix: {query_suffix} != {candidate_suffix}")
                        continue
                
                # قانون 5: امتیاز شباهت باید بالاتر از 0.8 باشد
                if score < 0.8:
                    logger.debug(f"  Rejecting {candidate} due to low similarity score: {score:.2f}")
                    continue
                
                # کاندیدای معتبر
                valid_candidates.append((candidate, score))
                logger.debug(f"  Accepted {candidate} with score {score:.2f}")
            
            logger.debug(f"Validated candidates: {len(valid_candidates)} out of {len(candidates)}")
            
            # اگر چندین کاندیدا باقی مانده، فقط بهترین را نگه دار
            if len(valid_candidates) > 1:
                best_candidate = max(valid_candidates, key=lambda x: x[1])
                # فقط کاندیداهایی که امتیازشان خیلی نزدیک به بهترین است را نگه دار
                valid_candidates = [c for c in valid_candidates if c[1] >= best_candidate[1] - 0.05]
            
            return valid_candidates
            
        except Exception as e:
            logger.error(f"Error validating candidates for {query_tag}: {e}")
            return []


    def _are_numeric_parts_very_similar(self, num1, num2):
        """
        بررسی شباهت بسیار نزدیک بخش‌های عددی
        
        Args:
            num1: عدد اول (به صورت رشته)
            num2: عدد دوم (به صورت رشته)
        
        Returns:
            True اگر اعداد بسیار شبیه هم باشند، False در غیر این صورت
        """
        try:
            # اگر هر دو خالی باشند
            if not num1 and not num2:
                return True
                
            # اگر فقط یکی خالی باشد
            if not num1 or not num2:
                return False
                
            # تبدیل به عدد
            n1 = int(num1)
            n2 = int(num2)
            
            # بررسی برابری دقیق
            if n1 == n2:
                return True
                
            # بررسی شباهت با تلرانس کمتر - اختلاف کمتر از 2 یا 5% اختلاف
            diff = abs(n1 - n2)
            if diff <= 2 or diff / max(n1, n2) <= 0.05:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error comparing numeric parts {num1} and {num2}: {e}")
            return False

    def _parse_multi_patterns(self, value):
        """Parse a comma/space/newline separated list of patterns into a list of uppercased strings.
        
        Accepts:
          - "JSF"                    → ["JSF"]
          - "JSF,JSX,JSY"            → ["JSF", "JSX", "JSY"]
          - "JSF, JSX, JSY"          → ["JSF", "JSX", "JSY"]
          - "JSF JSX JSY"            → ["JSF", "JSX", "JSY"]
          - ["JSF", "JSX"]           → ["JSF", "JSX"]
          - ["JSF,JSX"]              → ["JSF", "JSX"]
        Returns: list of uppercased non-empty strings (may be empty list)
        """
        if value is None:
            return []
        if isinstance(value, list):
            # flatten list items that may themselves contain commas
            items = []
            for v in value:
                items.extend(str(v).replace('\n', ',').replace(' ', ',').split(','))
        else:
            items = str(value).replace('\n', ',').replace(' ', ',').split(',')
        return [i.strip().upper() for i in items if i.strip()]

    def set_patterns(self, jb_examples=None, mc_examples=None, spare_examples=None, 
                    cable_examples=None, wire_color_rule=None, scr_number_rule=None):
        """
        تنظیم الگوهای سفارشی برای بهبود تشخیص — پشتیبانی از چند الگو
        
        Args:
            jb_examples: مثال JB (رشته با کاما، یا لیست). مثال: "JSF,JSX,JSY"
            mc_examples: مثال MC (رشته با کاما، یا لیست). مثال: "NC,IC"
            spare_examples: مثال SPARE (رشته با کاما، یا لیست)
            cable_examples: مثال توصیف کابل (رشته یا لیست)
            wire_color_rule: قاعده تولید رنگ سیم
            scr_number_rule: قاعده تولید شماره SCR
        
        Note:
            - jb_examples می‌تونه چند الگو با کاما داشته باشه: "JSF,JSX,JSY"
            - همه الگوها برای تشخیص استفاده می‌شن
            - self.jb_examples به‌صورت رشته با کاما ذخیره می‌شه (backward compatible)
            - self.jb_examples_list لیست واقعی الگوهاست
        """
        
        # JB examples — support multiple patterns
        if jb_examples is not None:
            jb_list = self._parse_multi_patterns(jb_examples)
            self.jb_examples_list = jb_list
            # Store as comma-separated string for backward compatibility (logging, etc.)
            self.jb_examples = ','.join(jb_list) if jb_list else None
            logger.info(f"JB examples Set: {self.jb_examples} ({len(jb_list)} patterns)")
        
        # MC examples — support multiple patterns
        if mc_examples is not None:
            mc_list = self._parse_multi_patterns(mc_examples)
            self.mc_examples_list = mc_list
            self.mc_examples = ','.join(mc_list) if mc_list else None
            logger.info(f"MC examples Set: {self.mc_examples} ({len(mc_list)} patterns)")
        
        # SPARE examples — support multiple patterns
        if spare_examples is not None:
            spare_list = self._parse_multi_patterns(spare_examples)
            self.spare_examples_list = spare_list
            self.spare_examples = ','.join(spare_list) if spare_list else None
            logger.info(f"SPARE examples Set: {self.spare_examples} ({len(spare_list)} patterns)")
        
        if cable_examples is not None:
            if isinstance(cable_examples, list):
                self.cable_examples = ', '.join(cable_examples)
            elif isinstance(cable_examples, str):
                self.cable_examples = cable_examples.strip()
            logger.info(f"Cable examples Set: {self.cable_examples}")
        
        if wire_color_rule is not None:
            self.wire_color_rule = wire_color_rule
            logger.info(f"Wire color rule Set: {wire_color_rule}")
        
        if scr_number_rule is not None:
            self.scr_number_rule = scr_number_rule
            logger.info(f"SCR number rule Set: {scr_number_rule}")
        
        # به‌روزرسانی الگوهای regex بر اساس مثال‌های جدید
        self._compile_regex_patterns()
        
    def _compile_regex_patterns(self):
        """
        کامپایل الگوهای regex بر اساس مثال‌های تنظیم شده — پشتیبانی از چند الگو
        """
        try:
            # JB regex — combine all patterns into one alternation
            jb_list = getattr(self, 'jb_examples_list', None) or []
            if jb_list:
                # Build alternation: (JSF|JSX|JSY)-?\d+
                jb_alt = '|'.join(re.escape(p) for p in jb_list)
                self.jb_regex = re.compile(rf'\b({jb_alt})-?\d+\b', re.IGNORECASE)
                logger.debug(f"JB regex compiled: {self.jb_regex.pattern}")
            else:
                self.jb_regex = None
            
            # MC regex — combine all patterns
            mc_list = getattr(self, 'mc_examples_list', None) or []
            if mc_list:
                mc_alt = '|'.join(re.escape(p) for p in mc_list)
                self.mc_regex = re.compile(rf'\b({mc_alt})-?\d+\b', re.IGNORECASE)
                logger.debug(f"MC regex compiled: {self.mc_regex.pattern}")
            else:
                self.mc_regex = None
            
            # SPARE regex — simple pattern
            spare_list = getattr(self, 'spare_examples_list', None) or []
            if spare_list:
                self.spare_regex = re.compile(r'\b(spare)\b', re.IGNORECASE)
                logger.debug(f"SPARE regex compiled (simple pattern): {self.spare_regex.pattern}")
            else:
                self.spare_regex = None
            
            logger.info(f"✅ Regex patterns compiled: JB={len(jb_list)}, MC={len(mc_list)}, SPARE={len(spare_list)}")
                
        except Exception as e:
            logger.error(f"Error compiling regex patterns: {e}")

    def _has_reasonable_prefix_suffix(self, a, b):
        # دو حرف اول معمولاً نوع تجهیز را مشخص می‌کنند (مثلاً PCV, FCV)
        prefix_a, prefix_b = a[:3], b[:3]
        if prefix_a != prefix_b:
            return False  # تجهیزهای متفاوت، پس احتمالاً بی‌ربط

        # بررسی اینکه حداقل نیمی از عدد وسط یکی باشد
        digits_a = ''.join(ch for ch in a if ch.isdigit())
        digits_b = ''.join(ch for ch in b if ch.isdigit())
        match_digits = sum(1 for x, y in zip(digits_a, digits_b) if x == y)
        return match_digits >= len(digits_a) // 2

    def _count_different_chars(self, str1: str, str2: str) -> int:
        """شمارش کاراکترهای متفاوت"""
        if len(str1) != len(str2):
            return abs(len(str1) - len(str2)) + sum(c1 != c2 for c1, c2 in zip(str1, str2))
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))

    def _get_different_chars(self, str1: str, str2: str) -> 'Tuple[str, str]':
        """پیدا کردن اولین کاراکتر متفاوت"""
        for c1, c2 in zip(str1, str2):
            if c1 != c2:
                return c1, c2
        return '', ''

    def _get_similarity_reason(self, ocr_tag: str, io_tag: str) -> str:
        """توضیح دلیل similar match"""
        if len(ocr_tag) != len(io_tag):
            return f"Length diff: {len(ocr_tag)} vs {len(io_tag)}"
        
        diff_count = self._count_different_chars(ocr_tag, io_tag)
        if diff_count == 1:
            char_ocr, char_io = self._get_different_chars(ocr_tag, io_tag)
            return f"OCR confusion: '{char_ocr}' → '{char_io}'"
        elif diff_count > 1:
            return f"{diff_count} char differences"
        
        return "Minor OCR noise"

    def _are_numbers_identical(self, num1, num2, tolerance=0):
        """
        مقایسه دو شماره با احتساب تلرانس
        
        Args:
            num1: شماره اول
            num2: شماره دوم
            tolerance: میزان تلرانس مجاز
            
        Returns:
            True اگر دو شماره معادل باشند، در غیر این صورت False
        """
        try:
            # تبدیل به عدد صحیح
            n1 = int(num1) if num1 and str(num1).strip().isdigit() else 0
            n2 = int(num2) if num2 and str(num2).strip().isdigit() else 0
            
            # مقایسه با احتساب تلرانس
            return abs(n1 - n2) <= tolerance
        except (ValueError, TypeError):
            # اگر تبدیل به عدد امکان‌پذیر نبود، مقایسه رشته‌ای انجام دهیم
            return str(num1).strip() == str(num2).strip()

    def assign_tag_numbers_by_position(self, tags_with_positions: List[Dict], 
                                    spare_identifiers_with_positions: List[Dict] = None) -> Dict[str, int]:
        """
        شماره‌گذاری تگ‌ها و SPARE ها بر اساس موقعیت عمودی (از بالا به پایین)
        
        Args:
            tags_with_positions: لیست دیکشنری‌های حاوی {'tag': str, 'y': int, 'x': int}
            spare_identifiers_with_positions: لیست دیکشنری‌های حاوی SPARE ها
            
        Returns:
            دیکشنری {tag/spare_id: number}
        """
        try:
            logger.info("="*70)
            logger.info("📍 Assigning tag numbers based on VERTICAL POSITION (top to bottom)")
            logger.info("="*70)
            
            # ترکیب تگ‌ها و SPARE ها
            all_items = []
            
            # اضافه کردن تگ‌ها
            for item in tags_with_positions:
                all_items.append({
                    'name': item['tag'],
                    'y_position': item['y'],
                    'x_position': item.get('x', 0),
                    'type': 'tag'
                })
            
            # اضافه کردن SPARE ها
            if spare_identifiers_with_positions:
                for idx, item in enumerate(spare_identifiers_with_positions):
                    spare_id = f"{getattr(self, 'spare_examples', None)}_{idx + 1}"
                    all_items.append({
                        'name': spare_id,
                        'y_position': item['y'],
                        'x_position': item.get('x', 0),
                        'type': 'spare',
                        'original_text': item.get('spare', 'SPARE')
                    })
            
            if not all_items:
                logger.warning("No items to number")
                return {}
            
            # مرتب‌سازی بر اساس موقعیت عمودی (y) و در صورت برابری بر اساس افقی (x)
            all_items.sort(key=lambda x: (x['y_position'], x['x_position']))
            
            logger.info(f"Sorted {len(all_items)} items by position:")
            
            # شماره‌گذاری
            tag_to_number = {}
            for number, item in enumerate(all_items, start=1):
                tag_to_number[item['name']] = number
                logger.info(f"  #{number:3d} → {item['type']:6s} {item['name']:20s} (y={item['y_position']:4d}, x={item['x_position']:4d})")
            
            logger.info(f"✅ Successfully assigned {len(tag_to_number)} numbers based on position")
            logger.info("="*70)
            
            return tag_to_number
            
        except Exception as e:
            logger.error(f"Error in assign_tag_numbers_by_position: {e}")
            logger.error(traceback.format_exc())
            return {}
            
    def clean_cable_description(self, text, mc_identifiers=None):
        import re

        if not text:
            return text

        # حذف دقیق نام‌های موجود در mc_identifiers
        if mc_identifiers:
            for mc in mc_identifiers:
                pattern = re.escape(mc)
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # حذف حالت عمومی MC + شماره (مثلاً MC12, MC-45, MC_33)
        text = re.sub(r'\bMC[-_\s]?\d+\b', '', text, flags=re.IGNORECASE)

        # پاک کردن فاصله‌های اضافی
        text = re.sub(r'\s{2,}', ' ', text).strip()

        return text

    def _normalize_code_token(self, token: 'Any') -> str:
        """
        Normalize a single OCR token into a code-like identifier.
        Keeps only A-Z, 0-9, '.', '_' and '-' and strips leading/trailing separators.
        """
        if token is None:
            return ""
        token_str = str(token).strip().upper()
        if not token_str:
            return ""
        token_str = re.sub(r"[^A-Z0-9._-]", "", token_str)
        token_str = token_str.strip("._-")
        return token_str

    def _is_prefixed_identifier(self, token: str, prefix: str, *, require_digit: bool = True) -> bool:
        """
        Heuristic check that token is an identifier starting with prefix (not merely containing it).
        Designed to reduce false-positives like 'ATACHONICAL' when prefix='IC'.
        """
        if not token or not prefix:
            return False
        prefix = str(prefix).strip().upper()
        if not prefix:
            return False
        token = str(token).strip().upper()

        if not token.startswith(prefix):
            return False
        if len(token) <= len(prefix):
            return False

        if require_digit and not any(ch.isdigit() for ch in token):
            return False

        # If prefix ends with alnum, require a separator or digit next (avoid prefix as substring of longer word)
        if prefix[-1].isalnum():
            next_ch = token[len(prefix)]
            if next_ch.isalpha():
                return False

        return True

    def _select_best_mc_identifier(self, mc_identifiers: 'Union[Set[str], List[str]]', jb_identifiers: 'Union[Set[str], List[str]]') -> str:
        """
        Select a single best MC identifier for a page in a deterministic way.
        Prefers codes that start with mc_examples and look structurally valid; when a JB exists,
        prefers MC that matches the JB suffix (e.g., JB-EEV-101 -> IC-EEV-101).
        """
        mc_prefix = (getattr(self, "mc_examples", "") or "").strip().upper()
        if not mc_prefix:
            return ""

        raw_candidates = list(mc_identifiers) if mc_identifiers else []
        norm_candidates_all = []
        for c in raw_candidates:
            norm = self._normalize_code_token(c)
            if norm:
                norm_candidates_all.append(norm)

        # Pass 1: strict filter (prefix + digit)
        norm_candidates = [c for c in norm_candidates_all if self._is_prefixed_identifier(c, mc_prefix, require_digit=True)]
        # Pass 2: relax digit requirement (still must start with prefix)
        if not norm_candidates:
            norm_candidates = [c for c in norm_candidates_all if self._is_prefixed_identifier(c, mc_prefix, require_digit=False)]
        if not norm_candidates:
            return ""

        jb_prefix = (getattr(self, "jb_examples", "") or "").strip().upper()
        expected_mc = None
        if jb_identifiers:
            jb_raw = list(jb_identifiers)[0]
            jb_norm = self._normalize_code_token(jb_raw)
            if jb_norm and jb_prefix and any(jb_norm.startswith(p) for p in jb_prefix.split(',') if p):
                expected_mc = mc_prefix + jb_norm[len(jb_prefix):]

        def candidate_score(cand: str) -> 'Tuple[float, int, int, int]':
            digits = sum(ch.isdigit() for ch in cand)
            separators = cand.count("-") + cand.count("_") + cand.count(".")
            # Higher is better; shorter is slightly preferred when all else equal
            length_penalty = -len(cand)

            similarity = 0.0
            if expected_mc:
                try:
                    similarity = float(Levenshtein.ratio(cand, expected_mc))
                except Exception:
                    similarity = 0.0
            return (similarity, digits, separators, length_penalty)

        # Deterministic: tie-break by lexicographic order
        norm_candidates_sorted = sorted(set(norm_candidates))
        best = max(norm_candidates_sorted, key=lambda c: (candidate_score(c), c))
        return best

    def _is_non_tag_pattern(self, token: str) -> bool:
        """
        True = این توکن یک الگوی غیر-تگ است و باید رد شود.
 
        الگوهای چک‌شده:
          1. JB identifier
          2. MC identifier
          3. SPARE keyword
          4. Cable descriptions (built-in + cable_examples کاربر)
             ─ مهم: «PREFIX-NxUNIT» مثل FRT-2P هم رد می‌شود
          5. Wire color codes
          6. SCR terminal
        """
        if not token:
            return False
 
        t = str(token).strip().upper()
 
        # ── 1. JB (split comma-separated prefixes) ──────────────────────
        _jb_list = getattr(self, 'jb_examples_list', None) or []
        if not _jb_list:
            _jb_raw = getattr(self, 'jb_examples', '') or ''
            if _jb_raw:
                _jb_list = self._parse_multi_patterns(_jb_raw)
        for p in _jb_list:
            if self._is_prefixed_identifier(t, p, require_digit=False):
                return True

        # ── 2. MC (split comma-separated prefixes) ────────────────────────
        _mc_list = getattr(self, 'mc_examples_list', None) or []
        if not _mc_list:
            _mc_raw = getattr(self, 'mc_examples', '') or ''
            if _mc_raw:
                _mc_list = self._parse_multi_patterns(_mc_raw)
        for p in _mc_list:
            if self._is_prefixed_identifier(t, p, require_digit=False):
                return True

        # ── 3. SPARE (split comma-separated prefixes) ─────────────────────
        _spare_list = getattr(self, 'spare_examples_list', None) or []
        if not _spare_list:
            _spare_raw = getattr(self, 'spare_examples', '') or ''
            if _spare_raw:
                _spare_list = self._parse_multi_patterns(_spare_raw)
        for p in _spare_list:
            if re.search(rf'\b{re.escape(p)}\b', t, re.IGNORECASE):
                return True
 
        # ── 4. Cable patterns ──────────────────────────────────────────────
        # 4a. built-in: خالص «عدد + واحد» مثل 9P، 12PAIR، 4CORE
        cable_builtin = re.compile(
            r'^\d{1,4}\s*(?:PAIR|PR|TRIPLE|TR|CORE|CR)(?:\b|X|×|\*|/|-|\.|\d|$)'
            r'|^\d{1,4}[PTC](?:\b|\d|X|×|$)',
            re.IGNORECASE,
        )
        if cable_builtin.match(t):
            return True
 
        # 4b. cable_examples کاربر — چند استراتژی همزمان
        cable_examples_str = getattr(self, 'cable_examples', '') or ''
        if cable_examples_str:
            for sample in re.split(r'[,;\s]+', cable_examples_str):
                sample = sample.strip().upper()
                if not sample:
                    continue
 
                # استراتژی A: sample دقیقاً «عدد + واحد» است  (مثل «12P» یا «12PAIR»)
                # → الگوی عمومی: هر prefix-NxUNIT یا NxUNIT همان واحد را رد کن
                cable_sample_m = re.match(
                    r'^(\d+)\s*(PAIR|PR|TRIPLE|TR|CORE|CR|[PTC])$',
                    sample, re.IGNORECASE
                )
                if cable_sample_m:
                    unit = cable_sample_m.group(2).upper()
                    # مستقیم: «NxUNIT»
                    if re.match(rf'^\d+\s*{re.escape(unit)}(?:\b|\d|X|×|$)', t, re.IGNORECASE):
                        return True
                    # غیر مستقیم: «PREFIX-NxUNIT» مثل FRT-2P  یا  FRT-7PX1MM
                    if re.match(
                        rf'^[A-Z]{{2,6}}[-_]\d+\s*{re.escape(unit)}(?:\b|X|×|\*|/|-|\.|\d|$)',
                        t, re.IGNORECASE
                    ):
                        return True
                    continue
 
                # استراتژی B: sample پیشوند حرفی ثابت دارد  (مثل «FRT-12PX1MM2»)
                # → هر توکنی که با همان پیشوند شروع شود را رد کن
                alpha_prefix_m = re.match(r'^([A-Z]{2,})', sample)
                if alpha_prefix_m:
                    prefix_str = alpha_prefix_m.group(1)
                    if t.startswith(prefix_str):
                        return True
 
        # ── 5. Wire color codes ────────────────────────────────────────────
        if re.match(r'^(?:BK|WT|RD|BL|GN|YL|WH|OR|GY|VI|BN|PK)\d{1,3}$', t):
            return True
        wire_rule = getattr(self, 'wire_color_rule', '') or ''
        if wire_rule:
            for part in re.split(r'[,;\s]+', wire_rule):
                part = part.strip()
                alpha_m = re.match(r'^([A-Za-z]{2,})', part)
                if alpha_m:
                    prefix_str = alpha_m.group(1).upper()
                    if len(prefix_str) >= 2 and t.startswith(prefix_str) and any(c.isdigit() for c in t):
                        return True
 
        # ── 6. SCR terminal ────────────────────────────────────────────────
        if re.match(r'^SCR[-_]?\d*$', t):
            return True
        scr_rule = getattr(self, 'scr_number_rule', '') or ''
        if scr_rule:
            scr_alpha_m = re.match(r'^([A-Za-z]{2,})', scr_rule)
            if scr_alpha_m:
                scr_prefix = scr_alpha_m.group(1).upper()
                if t.startswith(scr_prefix) and any(c.isdigit() for c in t):
                    return True
 
        return False


    def _get_mc_prefix(self):
        """Get the MC prefix from user settings. Returns None if not set.
        Returns comma-separated string for backward compatibility.
        Use _get_mc_prefixes() for list."""
        val = getattr(self, 'mc_examples', None)
        if not val:
            return None  # No prefix set — don't match anything
        return str(val).strip().upper()
    
    def _get_mc_prefixes(self):
        """Get list of MC prefixes. Returns empty list if not set."""
        return list(getattr(self, 'mc_examples_list', None) or [])

    def _get_jb_prefix(self):
        """Get the JB prefix from user settings. Returns None if not set.
        Returns comma-separated string for backward compatibility.
        Use _get_jb_prefixes() for list."""
        val = getattr(self, 'jb_examples', None)
        if not val:
            return None  # No prefix set — don't match anything
        return str(val).strip().upper()
    
    def _get_jb_prefixes(self):
        """Get list of JB prefixes. Returns empty list if not set."""
        return list(getattr(self, 'jb_examples_list', None) or [])

    def _is_mc_token(self, text):
        """Check if a token is an MC identifier. Returns False if no prefix set.
        Checks against ALL configured MC prefixes."""
        t = str(text).upper().strip()
        prefixes = self._get_mc_prefixes()
        if not prefixes:
            return False  # No prefix configured — cannot be an MC token
        return any(t.startswith(p) for p in prefixes)

    def _is_jb_token(self, text):
        """Check if a token is a JB identifier. Returns False if no prefix set.
        Checks against ALL configured JB prefixes."""
        t = str(text).upper().strip()
        prefixes = self._get_jb_prefixes()
        if not prefixes:
            return False  # No prefix configured — cannot be a JB token
        return any(t.startswith(p) for p in prefixes)

    def _inline_table_preprocess(self, image: 'np.ndarray') -> 'np.ndarray':
        """
        Inline table preprocessing — applied as a fallback when
        self.preprocess_image() does not accept the pdf_type parameter
        (e.g. legacy subclasses like LinuxTagJBExtractor / LoggedLinuxTagJBExtractor
        before they were updated).

        This duplicates the table branch of preprocess_image() so that
        table-mode extraction still works even if the active subclass
        overrides preprocess_image with an incompatible signature.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        # Light Gaussian blur + Otsu threshold — identical to the table
        # branch in preprocess_image().
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.info("_inline_table_preprocess: applied Gaussian + Otsu fallback preprocessing")
        return gray


    def extract_from_image(self, image: np.ndarray) -> 'Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict]]':
        """
        ✅ بازنویسی کامل: استخراج تگ‌ها با شماره‌گذاری بر اساس موقعیت عمودی
        
        Returns:
            Tuple of (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
                    spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info,all_ocr_tags)
        """
        # ============================================================
        # مقداردهی اولیه
        # ============================================================
        # Do NOT default to 'JB'/'MC'/'SPARE' — leave as None if not set by user
        # Defaulting causes false positives (e.g., 'JUNCTION' detected as JB)
        if not hasattr(self, 'jb_examples'):
            self.jb_examples = None
        if not hasattr(self, 'mc_examples'):
            self.mc_examples = None
        if not hasattr(self, 'spare_examples'):
            self.spare_examples = None
        
        logger.info(f"Using patterns - JB: '{self.jb_examples}', MC: '{self.mc_examples}', SPARE: '{self.spare_examples}'")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        pdf_type = getattr(self, '_current_pdf_type', 'diagrams')
        # Normalize to canonical values to avoid mismatches
        _l = str(pdf_type or '').lower()
        if 'table' in _l:
            pdf_type = 'table'
        elif 'diagram' in _l or 'drawing' in _l:
            pdf_type = 'diagrams'
        else:
            pdf_type = 'diagrams'
        # Normalize classifier/state label to canonical values
        _l = str(pdf_type or '').lower()
        if 'table' in _l:
            pdf_type = 'table'
        elif 'diagram' in _l or 'drawing' in _l:
            pdf_type = 'diagrams'
        else:
            pdf_type = 'diagrams'

        if pdf_type == 'table':
            # Table-optimised OCR config (REVISED v2):
            #   psm 6  → treat image as a single uniform block of text;
            #            more reliable for horizontally aligned table rows
            #   NO WHITELIST → previous whitelist was dropping tokens like
            #     "JSF-576S" (the trailing 'S' was being stripped, producing
            #     "JSF-576" or "JSF-5765"). Empirically, Tesseract without a
            #     whitelist produces far better results on CID-broken table
            #     PDFs because it's free to recognise the actual characters
            #     rather than being forced into the whitelist alphabet.
            #     The post-processor (_post_process_table_extractions)
            #     handles cleanup of any false positives.
            custom_config = r'--oem 1 --psm 6 -l eng'
            logger.info("extract_from_image: using TABLE OCR config (psm 6, no whitelist)")
            # REVISED: actually apply preprocess_image for table mode.
            # The previous code path declared a table-specific preprocessing
            # branch in preprocess_image() but never invoked it here, so the
            # raw BGR image was fed directly to Tesseract. For CID-broken
            # PDFium PDFs this returned garbage tokens for small cell text.
            # We now apply the (simple) table preprocessing pipeline before
            # handing the image to Tesseract.
            #
            # ⚠ SUBCLASS COMPATIBILITY: We use a try/except + **kwargs-style
            # call because subclasses (LinuxTagJBExtractor, LoggedLinuxTagJBExtractor)
            # historically overrode preprocess_image(self, image) WITHOUT the
            # pdf_type parameter. If the subclass version doesn't accept
            # pdf_type, we fall back to calling it without the keyword argument,
            # and apply the table preprocessing inline as a last resort.
            try:
                try:
                    image = self.preprocess_image(image, pdf_type='table')
                except TypeError as te:
                    if "pdf_type" in str(te) or "unexpected keyword argument" in str(te):
                        logger.warning(
                            "extract_from_image: subclass %s.preprocess_image does not accept "
                            "pdf_type parameter — calling without it and applying table "
                            "preprocessing inline",
                            type(self).__name__
                        )
                        # Try without pdf_type (legacy subclass signature)
                        try:
                            image = self.preprocess_image(image)
                        except Exception as inner:
                            logger.warning(
                                "extract_from_image: preprocess_image(image) also failed: %s — "
                                "applying inline table preprocessing",
                                inner
                            )
                            image = self._inline_table_preprocess(image)
                    else:
                        raise
            except Exception as preprocess_err:
                logger.warning(
                    "extract_from_image: preprocess_image failed (%s) — applying inline "
                    "table preprocessing",
                    preprocess_err
                )
                image = self._inline_table_preprocess(image)
        else:
            # Diagram path — byte-for-byte identical to original
            custom_config = r'--oem 1 --psm 11 -c tessedit_char_whiteList=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'

        logger.info("Starting OCR extraction...")
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        return self._extract_from_ocr_data(ocr_data, pdf_type)

    def _extract_from_image_table_multipass(self, page, temp_dir, page_num):
        """
        Table-mode CELL-BASED extractor (REVISED v4 — cell-by-cell OCR).
        
        Instead of full-page multi-pass OCR, this method:
        1. Uses pdfplumber to detect the table grid (rows × columns)
        2. OCRs each DATA CELL individually at 300 DPI with PSM 7
           (single line mode — much more accurate than block mode)
        3. Classifies cells by column position:
           - Col 1 (x≈124-179) → Tags / SPARE
           - Col 4 (x≈225-281) → Cable codes
        4. Header (rows 0-3) is handled by _extract_header_references_only
        
        WHY this is better than full-page OCR:
        - Each cell is OCR'd in isolation → no interference from neighbors
        - V/Y confusion disappears (cell boundary gives Tesseract context)
        - 5/S confusion disappears (single-line PSM 7 is more precise)
        - Column position tells us the element type → no post-processing needed
        - Garbage tokens from SPARE rows are filtered by tag pattern
        
        Args:
            page: fitz.Page object
            temp_dir: not used (we render directly)
            page_num: 0-indexed page number
            
        Returns:
            9-tuple from _extract_from_ocr_data
        """
        pdf_type = 'table'
        BASE_DPI = 300
        
        try:
            import pdfplumber
        except ImportError:
            logger.warning("_extract_from_image_table_multipass: pdfplumber not available, falling back to simple OCR")
            # Fallback: simple single-pass OCR
            scale = BASE_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=_CS_GRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # OPTIMIZATION: direct RGBA→GRAY
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_data = pytesseract.image_to_data(binary, config=r'--oem 1 --psm 6 -l eng', output_type=pytesseract.Output.DICT)
            return self._extract_from_ocr_data(ocr_data, pdf_type)
        
        # ── Step 1: Detect table grid ────────────────────────────────
        try:
            doc = page.parent
            if doc is None:
                raise RuntimeError("page has no parent doc")
            page_index = page.number if hasattr(page, 'number') else 0
            pdf_path = getattr(doc, 'name', None)
            if not pdf_path or not os.path.exists(pdf_path):
                raise RuntimeError("cannot resolve PDF path")
            
            with pdfplumber.open(pdf_path) as pp_doc:
                if page_index >= len(pp_doc.pages):
                    raise RuntimeError("page_index out of range")
                pp_page = pp_doc.pages[page_index]
                rects = pp_page.rects or []
        except Exception as grid_err:
            logger.warning("_extract_from_image_table_multipass: grid detection failed (%s), falling back", grid_err)
            scale = BASE_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=_CS_GRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # OPTIMIZATION: direct RGBA→GRAY
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_data = pytesseract.image_to_data(binary, config=r'--oem 1 --psm 6 -l eng', output_type=pytesseract.Output.DICT)
            return self._extract_from_ocr_data(ocr_data, pdf_type)
        
        h_lines = sorted([r for r in rects if r['width'] > 100 and r['height'] < 3],
                         key=lambda r: r['top'])
        v_lines = sorted([r for r in rects if r['height'] > 100 and r['width'] < 3],
                         key=lambda r: r['x0'])
        
        if not h_lines or not v_lines:
            logger.warning("_extract_from_image_table_multipass: no table lines detected, falling back")
            scale = BASE_DPI / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=_CS_GRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # OPTIMIZATION: direct RGBA→GRAY
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_data = pytesseract.image_to_data(binary, config=r'--oem 1 --psm 6 -l eng', output_type=pytesseract.Output.DICT)
            return self._extract_from_ocr_data(ocr_data, pdf_type)
        
        # Group H lines by top (within 3pt tolerance)
        row_tops = []
        for r in h_lines:
            top = round(r['top'], 0)
            if not row_tops or abs(row_tops[-1] - top) > 3:
                row_tops.append(top)
        
        col_lefts = sorted(set(round(r['x0'], 0) for r in v_lines))
        
        logger.info(
            "_extract_from_image_table_multipass: page %d — %d rows, %d cols detected",
            page_num + 1, len(row_tops), len(col_lefts)
        )
        
        # ── Step 2: Render full page at 300 DPI ──────────────────────
        scale = BASE_DPI / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=_CS_GRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # ── Step 3: Cell-by-cell OCR ─────────────────────────────────
        # Build ocr_data dict from cell results
        ocr_data = {
            'text': [], 'left': [], 'top': [], 'width': [], 'height': [],
            'conf': [], 'block_num': [], 'par_num': [], 'line_num': [], 'word_num': []
        }
        
        # Tag pattern for filtering garbage from SPARE cells
        # These patterns are GENERIC — they work on any table PDF:
        #   tag_like_re: matches any LETTERS-DIGITS-LETTER? pattern
        #   cable_re: matches PREFIX-DIGIT-DIGIT-DIGIT-LETTER-DIGIT-LETTERS
        #     where PREFIX comes from self.mc_examples (not hardcoded "NC")
        tag_like_re = re.compile(r'^[A-Z]{1,5}-\d{3,4}[A-Z]?$', re.IGNORECASE)
        _mc_prefix = self._get_mc_prefix()
        cable_re = re.compile(
            re.escape(_mc_prefix) + r'-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}',
            re.IGNORECASE
        )
        spare_re = re.compile(r'SPARE', re.IGNORECASE)
        
        # ── DYNAMIC column detection via header text ──────────────────
        # Instead of hardcoding x-position ranges (which are page-specific),
        # we OCR the HEADER ROW (row 0-3) and look for column labels:
        #   "TAG No."  → tag column
        #   "Multi Cable No." / "MC No." → cable column
        # This makes the code work on ANY table PDF regardless of page size
        # or column layout.
        tag_col_indices = []
        cable_col_indices = []
        DATA_ROW_START = 4  # default; will be adjusted below
        
        # OCR each header cell to find which column is which
        page_w = page.rect.width
        for hdr_row in range(min(4, len(row_tops) - 1)):
            top_h = row_tops[hdr_row]
            bottom_h = row_tops[hdr_row + 1] if hdr_row + 1 < len(row_tops) else top_h + 20
            if bottom_h - top_h < 5:
                continue
            
            for col_idx in range(len(col_lefts) - 1):
                left_h = col_lefts[col_idx]
                right_h = col_lefts[col_idx + 1]
                
                px0_h = max(0, int(left_h * scale) - 2)
                px1_h = min(img.shape[1], int(right_h * scale) + 2)
                py0_h = max(0, int(top_h * scale) - 2)
                py1_h = min(img.shape[0], int(bottom_h * scale) + 2)
                
                if px1_h - px0_h < 8 or py1_h - py0_h < 8:
                    continue
                
                hdr_cell = img[py0_h:py1_h, px0_h:px1_h]
                hdr_gray = cv2.cvtColor(hdr_cell, cv2.COLOR_BGR2GRAY)
                _, hdr_bin = cv2.threshold(hdr_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                hdr_text = pytesseract.image_to_string(hdr_bin, config=r'--oem 1 --psm 7 -l eng').strip().upper()
                
                # Check for tag column label
                if 'TAG' in hdr_text and col_idx not in tag_col_indices:
                    tag_col_indices.append(col_idx)
                    # Data starts AFTER the header row that contains "TAG No."
                    if hdr_row + 1 > DATA_ROW_START - 1:
                        DATA_ROW_START = hdr_row + 1
                
                # Check for cable column label
                if ('CABLE' in hdr_text or 'MC' in hdr_text) and col_idx not in cable_col_indices:
                    # Make sure it's not just "MC:" label — check for "Multi" or "Cable"
                    if 'CABLE' in hdr_text or 'MULTI' in hdr_text or 'MC' in hdr_text:
                        cable_col_indices.append(col_idx)
        
        # If header-based detection failed, fall back to width-based heuristic:
        # The tag column is typically the narrowest column that can fit a tag
        # (e.g. "UY-5021"). The cable column is typically wider (e.g. "NC-0-1-2-C-3-BL").
        if not tag_col_indices and len(col_lefts) > 2:
            # Find columns that are wide enough for a tag but narrower than the cable column
            col_widths = [(i, col_lefts[i+1] - col_lefts[i]) for i in range(len(col_lefts)-1)]
            col_widths.sort(key=lambda x: x[1])
            # Pick the 2nd narrowest column (1st is usually the row-number column)
            if len(col_widths) >= 2:
                tag_col_indices = [col_widths[1][0]]
        
        if not cable_col_indices and len(col_lefts) > 5:
            # Find the widest non-edge column
            col_widths = [(i, col_lefts[i+1] - col_lefts[i]) for i in range(1, len(col_lefts)-2)]
            if col_widths:
                col_widths.sort(key=lambda x: x[1], reverse=True)
                cable_col_indices = [col_widths[0][0]]
        
        logger.info(
            "_extract_from_image_table_multipass: tag cols=%s, cable cols=%s, data_row_start=%d",
            tag_col_indices, cable_col_indices, DATA_ROW_START
        )
        cfg_psm7 = r'--oem 1 --psm 7 -l eng'  # Single line — best for individual cells
        cfg_psm6 = r'--oem 1 --psm 6 -l eng'  # Block — fallback
        
        cells_processed = 0
        tags_found = 0
        spares_found = 0
        cables_found = 0
        
        for row_idx in range(DATA_ROW_START, len(row_tops) - 1):
            top = row_tops[row_idx]
            bottom = row_tops[row_idx + 1]
            if bottom - top < 5:
                continue
            
            # Process tag column
            for col_idx in tag_col_indices:
                if col_idx >= len(col_lefts) - 1:
                    continue
                left = col_lefts[col_idx]
                right = col_lefts[col_idx + 1]
                
                # Convert to pixel coordinates
                px0 = max(0, int(left * scale) - 3)
                px1 = min(img.shape[1], int(right * scale) + 3)
                py0 = max(0, int(top * scale) - 3)
                py1 = min(img.shape[0], int(bottom * scale) + 3)
                
                if px1 - px0 < 10 or py1 - py0 < 10:
                    continue
                
                cell_img = img[py0:py1, px0:px1]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # Use image_to_data to get EXACT text bounding box within the cell
                # (not the full cell dimensions). This makes bounding boxes tight
                # around the actual text, matching the pre-cell-based behavior.
                cell_tsv = pytesseract.image_to_data(binary, config=cfg_psm7, output_type=pytesseract.Output.DICT)
                cell_text_parts = []
                # Collect the tightest bounding box from all non-empty words
                min_x, min_y = 999999, 999999
                max_x, max_y = 0, 0
                has_words = False
                for ci in range(len(cell_tsv['text'])):
                    w = str(cell_tsv['text'][ci]).strip()
                    if not w:
                        continue
                    has_words = True
                    cell_text_parts.append(w)
                    wx = int(cell_tsv['left'][ci])
                    wy = int(cell_tsv['top'][ci])
                    ww = int(cell_tsv['width'][ci])
                    wh = int(cell_tsv['height'][ci])
                    if wx < min_x: min_x = wx
                    if wy < min_y: min_y = wy
                    if wx + ww > max_x: max_x = wx + ww
                    if wy + wh > max_y: max_y = wy + wh
                
                if not has_words:
                    # Fallback to image_to_string
                    text7 = pytesseract.image_to_string(binary, config=cfg_psm7).strip()
                    text6 = pytesseract.image_to_string(binary, config=cfg_psm6).strip()
                    text = text7 if len(text7) >= len(text6) else text6
                    # Use full cell coords as fallback
                    tight_left = int(left * scale)
                    tight_top = int(top * scale)
                    tight_width = int((right - left) * scale)
                    tight_height = int((bottom - top) * scale)
                else:
                    text = ' '.join(cell_text_parts)
                    # Convert cell-local coords (in 2x upscaled space) back to
                    # full-page 300 DPI coords.
                    # The cell image was: cell_img[py0:py1, px0:px1] → upscaled 2x
                    # So: page_coord = px0 + (cell_coord / 2)
                    # But we also added padding (-3) when extracting, so:
                    # tight_left = px0 + min_x / 2  (in page pixels at 300 DPI)
                    tight_left = px0 + int(min_x / 2)
                    tight_top = py0 + int(min_y / 2)
                    tight_width = int((max_x - min_x) / 2)
                    tight_height = int((max_y - min_y) / 2)
                    # Clamp to page bounds
                    tight_left = max(0, tight_left)
                    tight_top = max(0, tight_top)
                    tight_width = max(10, tight_width)
                    tight_height = max(10, tight_height)
                
                if not text or len(text) < 2:
                    continue
                
                # Clean the text
                text = text.strip().strip('[](){}|<>\'"')
                
                # Check if it's SPARE
                if spare_re.search(text.upper()):
                    ocr_data['text'].append('SPARE')
                    ocr_data['left'].append(tight_left)
                    ocr_data['top'].append(tight_top)
                    ocr_data['width'].append(tight_width)
                    ocr_data['height'].append(tight_height)
                    ocr_data['conf'].append(95)
                    ocr_data['block_num'].append(1)
                    ocr_data['par_num'].append(1)
                    ocr_data['line_num'].append(row_idx)
                    ocr_data['word_num'].append(1)
                    spares_found += 1
                    cells_processed += 1
                    continue
                
                # Check if it looks like a tag (letters-digits pattern)
                text_upper = text.upper().strip()
                if tag_like_re.match(text_upper):
                    ocr_data['text'].append(text_upper)
                    ocr_data['left'].append(tight_left)
                    ocr_data['top'].append(tight_top)
                    ocr_data['width'].append(tight_width)
                    ocr_data['height'].append(tight_height)
                    ocr_data['conf'].append(90)
                    ocr_data['block_num'].append(1)
                    ocr_data['par_num'].append(1)
                    ocr_data['line_num'].append(row_idx)
                    ocr_data['word_num'].append(1)
                    tags_found += 1
                    cells_processed += 1
                # Skip garbage tokens from SPARE cells
            
            # Process cable column
            for col_idx in cable_col_indices:
                if col_idx >= len(col_lefts) - 1:
                    continue
                left = col_lefts[col_idx]
                right = col_lefts[col_idx + 1]
                
                px0 = max(0, int(left * scale) - 3)
                px1 = min(img.shape[1], int(right * scale) + 3)
                py0 = max(0, int(top * scale) - 3)
                py1 = min(img.shape[0], int(bottom * scale) + 3)
                
                if px1 - px0 < 10 or py1 - py0 < 10:
                    continue
                
                cell_img = img[py0:py1, px0:px1]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                binary = cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                # Use image_to_data for tight bounding box (same as tag column)
                cell_tsv = pytesseract.image_to_data(binary, config=cfg_psm7, output_type=pytesseract.Output.DICT)
                cell_text_parts = []
                min_x, min_y = 999999, 999999
                max_x, max_y = 0, 0
                has_words = False
                for ci in range(len(cell_tsv['text'])):
                    w = str(cell_tsv['text'][ci]).strip()
                    if not w:
                        continue
                    has_words = True
                    cell_text_parts.append(w)
                    wx = int(cell_tsv['left'][ci])
                    wy = int(cell_tsv['top'][ci])
                    ww = int(cell_tsv['width'][ci])
                    wh = int(cell_tsv['height'][ci])
                    if wx < min_x: min_x = wx
                    if wy < min_y: min_y = wy
                    if wx + ww > max_x: max_x = wx + ww
                    if wy + wh > max_y: max_y = wy + wh
                
                if has_words:
                    text = ' '.join(cell_text_parts)
                    tight_left = px0 + int(min_x / 2)
                    tight_top = py0 + int(min_y / 2)
                    tight_width = int((max_x - min_x) / 2)
                    tight_height = int((max_y - min_y) / 2)
                else:
                    text = pytesseract.image_to_string(binary, config=cfg_psm6).strip()
                    tight_left = int(left * scale)
                    tight_top = int(top * scale)
                    tight_width = int((right - left) * scale)
                    tight_height = int((bottom - top) * scale)
                
                tight_left = max(0, tight_left)
                tight_top = max(0, tight_top)
                tight_width = max(10, tight_width)
                tight_height = max(10, tight_height)
                
                if not text or len(text) < 5:
                    continue
                
                # Check for cable pattern
                text_upper = text.upper().strip()
                cable_match = re.search(r'NC-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}', text_upper)
                if cable_match:
                    cable_text = cable_match.group(0)
                    ocr_data['text'].append(cable_text)
                    ocr_data['left'].append(tight_left)
                    ocr_data['top'].append(tight_top)
                    ocr_data['width'].append(tight_width)
                    ocr_data['height'].append(tight_height)
                    ocr_data['conf'].append(90)
                    ocr_data['block_num'].append(2)
                    ocr_data['par_num'].append(1)
                    ocr_data['line_num'].append(row_idx)
                    ocr_data['word_num'].append(1)
                    cables_found += 1
                    cells_processed += 1
        
        # ── Step 4: Also do a full-page pass for JB/MC/Cable fallback ─
        # The cell-based approach may miss JB/MC (which are in the header,
        # not in data cells). We do a full-page pass at 300 AND 400 DPI
        # to catch JB/MC that the 300 DPI pass misses.
        jb_re = re.compile(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', re.IGNORECASE)
        mc_re = re.compile(r'^' + re.escape(_mc_prefix) + r'-?[A-Z]{2,5}-?\d{2,5}[A-Z]?$', re.IGNORECASE)
        full_cable_re = re.compile(
            r'^' + re.escape(_mc_prefix) + r'-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}$',
            re.IGNORECASE
        )
        
        seen_jb_tokens = set()
        seen_mc_tokens = set()
        seen_cable_tokens = set()
        
        for extra_dpi in [300]:  # OPTIMIZATION: dropped 400 DPI (marginal accuracy gain, 1.78x slower)
            extra_scale = extra_dpi / 72
            extra_pix = page.get_pixmap(matrix=fitz.Matrix(extra_scale, extra_scale), colorspace=_CS_GRAY)
            extra_img = np.frombuffer(extra_pix.samples, dtype=np.uint8).reshape(extra_pix.height, extra_pix.width, extra_pix.n)
            if extra_pix.n == 3:
                extra_img = cv2.cvtColor(extra_img, cv2.COLOR_RGB2BGR)
            else:
                extra_img = cv2.cvtColor(extra_img, cv2.COLOR_RGBA2BGR)
            
            gray_extra = cv2.cvtColor(extra_img, cv2.COLOR_BGR2GRAY)
            gray_extra = cv2.GaussianBlur(gray_extra, (3, 3), 0)
            _, binary_extra = cv2.threshold(gray_extra, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            extra_data = pytesseract.image_to_data(binary_extra, config=r'--oem 1 --psm 6 -l eng', output_type=pytesseract.Output.DICT)
            
            # Normalize coordinates to 300 DPI base
            coord_scale = BASE_DPI / extra_dpi
            
            for i in range(len(extra_data['text'])):
                text = str(extra_data['text'][i]).strip().upper()
                if not text or len(text) < 4:
                    continue
                
                # Add MC tokens (NC-*)
                if mc_re.match(text) and not full_cable_re.match(text):
                    if text not in seen_mc_tokens:
                        seen_mc_tokens.add(text)
                        ocr_data['text'].append(text)
                        ocr_data['left'].append(int(int(extra_data['left'][i]) * coord_scale))
                        ocr_data['top'].append(int(int(extra_data['top'][i]) * coord_scale))
                        ocr_data['width'].append(int(int(extra_data['width'][i]) * coord_scale))
                        ocr_data['height'].append(int(int(extra_data['height'][i]) * coord_scale))
                        ocr_data['conf'].append(int(extra_data['conf'][i]) if extra_data['conf'][i] != '-1' else 80)
                        ocr_data['block_num'].append(3)
                        ocr_data['par_num'].append(1)
                        ocr_data['line_num'].append(0)
                        ocr_data['word_num'].append(1)
                
                # Add JB tokens (matching jb_examples prefix, not hardcoded "JSF")
                _jb_prefix = str(getattr(self, 'jb_examples', None) or '').strip().upper()
                if jb_re.match(text) and _jb_prefix and any(text.startswith(p) for p in _jb_prefix.split(',') if p):
                    if text not in seen_jb_tokens:
                        seen_jb_tokens.add(text)
                        ocr_data['text'].append(text)
                        ocr_data['left'].append(int(int(extra_data['left'][i]) * coord_scale))
                        ocr_data['top'].append(int(int(extra_data['top'][i]) * coord_scale))
                        ocr_data['width'].append(int(int(extra_data['width'][i]) * coord_scale))
                        ocr_data['height'].append(int(int(extra_data['height'][i]) * coord_scale))
                        ocr_data['conf'].append(int(extra_data['conf'][i]) if extra_data['conf'][i] != '-1' else 80)
                        ocr_data['block_num'].append(3)
                        ocr_data['par_num'].append(1)
                        ocr_data['line_num'].append(0)
                        ocr_data['word_num'].append(1)
                
                # Add Cable tokens that weren't caught by cell-based
                if full_cable_re.match(text):
                    if text not in seen_cable_tokens:
                        seen_cable_tokens.add(text)
                        ocr_data['text'].append(text)
                        ocr_data['left'].append(int(int(extra_data['left'][i]) * coord_scale))
                        ocr_data['top'].append(int(int(extra_data['top'][i]) * coord_scale))
                        ocr_data['width'].append(int(int(extra_data['width'][i]) * coord_scale))
                        ocr_data['height'].append(int(int(extra_data['height'][i]) * coord_scale))
                        ocr_data['conf'].append(int(extra_data['conf'][i]) if extra_data['conf'][i] != '-1' else 80)
                        ocr_data['block_num'].append(3)
                        ocr_data['par_num'].append(1)
                        ocr_data['line_num'].append(0)
                        ocr_data['word_num'].append(1)
        
        logger.info(
            "_extract_from_image_table_multipass: page %d — %d cells processed, %d tags, %d spares, %d cables",
            page_num + 1, cells_processed, tags_found, spares_found, cables_found
        )
        logger.info(
            "_extract_from_image_table_multipass: page %d — %d total OCR tokens (cell + full-page)",
            page_num + 1, len(ocr_data['text'])
        )
        
        # ── Step 5: Apply OCR corrections and extract ─────────────────
        corrected_ocr_data = self._apply_table_ocr_corrections(ocr_data, page_num)
        
        return self._extract_from_ocr_data(corrected_ocr_data, pdf_type)

    def _apply_table_ocr_corrections(self, ocr_data, page_num):r_data(corrected_ocr_data, pdf_type)

    def _apply_table_ocr_corrections(self, ocr_data, page_num):
        """
        Apply OCR error corrections specifically for table-mode PDFs.

        Three transformations:
        1. MC TOKEN MERGING:
           OCR often splits "NC-JSF-5125" into two tokens: "NC" and "JSF-5125"
           (or "NCJSF" without dash). We scan for these patterns and merge them
           back into a single "NC-JSF-XXXX" token.

        2. CHARACTER CORRECTION using IO List:
           Common OCR errors on CID-broken PDFs:
             - U ↔ V (UV-5021 → UY-5021)
             - 5 ↔ S (FUY-S407 → FUY-5407)
             - 0 ↔ 5 (JSF-012S → JSF-512S)
             - 7 ↔ 1 (JSF-5725 → JSF-5125)
           We use the IO List as ground truth: if an OCR token is fuzzy-similar
           to an IO List tag (sim ≥ 0.85), we replace it with the IO List form.

        3. CABLE/MC SEPARATION:
           - Tokens matching NC-X-X-X-X-X-XXX (5+ dashes) → cable
           - Tokens matching NC-LETTERS-DIGITS (1-2 dashes) → MC
           Any token that's a cable is removed from MC candidates.

        Args:
            ocr_data: OCR data dict (text, left, top, width, height, conf, ...)
            page_num: 0-indexed page number (for logging).

        Returns:
            Corrected ocr_data dict with the same structure.
        """
        if not ocr_data or not ocr_data.get('text'):
            return ocr_data

        n_tokens = len(ocr_data['text'])
        if n_tokens == 0:
            return ocr_data

        # Make a mutable copy
        new_text = list(ocr_data['text'])
        new_left = list(ocr_data.get('left', [0] * n_tokens))
        new_top = list(ocr_data.get('top', [0] * n_tokens))
        new_width = list(ocr_data.get('width', [0] * n_tokens))
        new_height = list(ocr_data.get('height', [0] * n_tokens))
        new_conf = list(ocr_data.get('conf', [0] * n_tokens))

        # ── Get IO List tags for character correction ──────────────────
        io_list_tags = set()
        if hasattr(self, 'io_list_tags') and self.io_list_tags:
            io_list_tags = set(str(t).upper().strip() for t in self.io_list_tags)
        elif hasattr(self, 'excel_df') and hasattr(self, 'excel_tag_column'):
            if self.excel_df is not None and not self.excel_df.empty:
                tag_col = self.excel_tag_column
                io_list_tags = set(str(t).strip().upper() for t in self.excel_df[tag_col] if pd.notna(t))

        try:
            import Levenshtein as _lev
            _lev_available = True
        except ImportError:
            _lev_available = False

        # ── STEP 1: MC Token Merging ───────────────────────────────────
        # Look for patterns like:
        #   token[i] = "NC" or "NCJ" or "NCJSF"  (no dash)
        #   token[i+1] = "JSF-XXXX" or "XXX-XXXX"
        # and merge them into "NC-JSF-XXXX"
        mc_examples = str(getattr(self, 'mc_examples', None) or '').strip().upper()
        jb_examples = str(getattr(self, 'jb_examples', None) or '').strip().upper()

        # Build a list of (index, action) for merges
        merges = []  # list of (i, i+1, merged_text)
        used_indices = set()

        for i in range(n_tokens - 1):
            if i in used_indices:
                continue
            t1 = str(new_text[i]).strip().upper()
            t2 = str(new_text[i + 1]).strip().upper()

            if not t1 or not t2:
                continue

            # Pattern 1: "NC" + "JSF-XXXX" → "NC-JSF-XXXX"
            if t1 == 'NC' and len(t2) >= 5:
                # Check if t2 looks like a JB identifier (e.g. JSF-5125)
                if re.match(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', t2):
                    merged = f"NC-{t2}"
                    merges.append((i, i + 1, merged))
                    used_indices.add(i)
                    used_indices.add(i + 1)
                    continue

            # Pattern 2: "NCJSF" (no dash) + nothing — already merged by OCR
            # We'll handle this in Step 2 (character correction)

            # Pattern 3: "NC" + "JSF" + separate digits — too complex, skip

        # Apply merges: replace token[i] with merged, mark token[i+1] for removal
        indices_to_remove = set()
        for i, j, merged_text in merges:
            new_text[i] = merged_text
            # Extend the bounding box to cover both tokens
            new_left[i] = min(int(new_left[i]), int(new_left[j]))
            new_top[i] = min(int(new_top[i]), int(new_top[j]))
            new_width[i] = max(int(new_left[i]) + int(new_width[i]),
                              int(new_left[j]) + int(new_width[j])) - int(new_left[i])
            new_height[i] = max(int(new_height[i]), int(new_height[j]))
            new_conf[i] = max(int(new_conf[i]), int(new_conf[j]))
            indices_to_remove.add(j)

        if merges:
            logger.info(
                "_apply_table_ocr_corrections: page %d — merged %d split MC tokens",
                page_num + 1, len(merges)
            )

        # ── STEP 2: Character Correction using IO List ─────────────────
        # For each token that looks like a tag/JB/MC, try to find a close
        # match in the IO List. If found (sim ≥ 0.85), replace with IO List form.
        corrections_applied = 0

        # Common character confusions to try as substitutions
        CHAR_CONFUSIONS = [
            ('V', 'Y'),  # UV-5021 → UY-5021
            ('S', '5'),  # FUY-S407 → FUY-5407
            ('5', 'S'),  # JSF-576S → JSF-576S (already correct, but for safety)
            ('0', '5'),  # JSF-012S → JSF-512S
            ('5', '0'),  # reverse
            ('1', '7'),  # JSF-5725 → JSF-5125
            ('7', '1'),  # reverse
            ('O', '0'),  # O → 0
            ('0', 'O'),  # reverse (rare)
        ]

        for i in range(n_tokens):
            if i in indices_to_remove:
                continue
            t = str(new_text[i]).strip().upper()
            if not t or len(t) < 4:
                continue

            # Only correct tokens that look like tags/JBs/MCs
            looks_like_tag = bool(re.match(r'^[A-Z]{1,5}-\d{3,4}[A-Z]?$', t))
            looks_like_jb = bool(re.match(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', t))
            looks_like_mc = bool(re.match(r'^NC-?[A-Z]{2,5}-?\d{2,5}[A-Z]?$', t))

            if not (looks_like_tag or looks_like_jb or looks_like_mc):
                continue

            # Skip if already exact match in IO List
            if t in io_list_tags:
                continue

            # Try character substitutions
            best_match = None
            best_score = 0.0
            best_match_source = None  # 'io_list', 'jb_ref', or 'mc_ref'

            # ── Build reference set based on token type ──────────────────
            # For JB-like tokens: use known JBs (from header) + IO List tags
            # For MC-like tokens: use known MCs (from header) + IO List tags
            # For tag-like tokens: use IO List tags
            #
            # IMPORTANT: We also add IO List tags to JB/MC reference sets
            # because the header extraction may produce incorrect JB/MC
            # (e.g. JSF-5776S instead of JSF-576S). By including IO List
            # tags, we give the correction algorithm more options to find
            # the correct form.
            reference_set = set()
            if looks_like_mc:
                # MC token — use known MCs from header
                known_mcs = getattr(self, '_table_known_mcs', set())
                reference_set = set(str(m).upper() for m in known_mcs)
                # Also add IO List tags (in case MC has similar pattern)
                reference_set |= io_list_tags
            elif looks_like_jb:
                # JB token — use known JBs from header
                known_jbs = getattr(self, '_table_known_jbs', set())
                reference_set = set(str(j).upper() for j in known_jbs)
                # Also add IO List tags (in case a tag looks like a JB)
                reference_set |= io_list_tags
            else:
                # Regular tag — use IO List tags
                reference_set = io_list_tags

            if _lev_available and reference_set:
                # ── Pattern-based suffix correction (no reference needed) ─
                # If a JB/MC token ends with '5' and matches the JB pattern
                # (letters-digits-optional letter), the trailing '5' is very
                # likely an OCR error for 'S' (these characters look similar).
                # We apply this correction WITHOUT requiring a reference match,
                # because the reference set may also contain errors.
                if (looks_like_jb or looks_like_mc) and len(t) >= 5:
                    if t[-1] == '5':
                        parts = t.split('-')
                        # For JB: parts = ['JSF', '5765'] (2 parts)
                        # For MC: parts = ['NC', 'JSF', '5765'] (3 parts)
                        # In both cases, the last part is the digits
                        if len(parts) >= 2:
                            digits = parts[-1]
                            # Generic check: if the digit part has 3+ digits
                            # and ends with '5', it's likely an OCR error for 'S'.
                            # We don't hardcode "4 digits" because different
                            # projects may have 3-digit or 5-digit JB numbers.
                            # The pattern [A-Z]{2,5}-\d{2,5}[A-Z]? means the
                            # JB can have 2-5 digits. If we have 3+ digits
                            # ending with 5, converting to S is reasonable.
                            if len(digits) >= 3 and digits[-1] == '5':
                                # Convert last 5 to S
                                prefix = '-'.join(parts[:-1])
                                corrected = f"{prefix}-{digits[:-1]}S"
                                new_text[i] = corrected
                                corrections_applied += 1
                                logger.info(
                                    "_apply_table_ocr_corrections: suffix-corrected '%s' → '%s' (5→S, pattern-based)",
                                    t, corrected
                                )
                                continue

                # First try direct fuzzy match (no substitution)
                for ref_tag in reference_set:
                    score = _lev.ratio(t, str(ref_tag).upper())
                    if score > best_score:
                        best_score = score
                        best_match = ref_tag
                        best_match_source = 'direct'

                # If direct match is good enough, use it
                if best_score >= 0.92:
                    new_text[i] = best_match
                    corrections_applied += 1
                    logger.debug(
                        "_apply_table_ocr_corrections: corrected '%s' → '%s' (sim=%.2f, direct)",
                        t, best_match, best_score
                    )
                    continue

                # Try SINGLE character substitutions
                for old_ch, new_ch in CHAR_CONFUSIONS:
                    positions = [j for j, c in enumerate(t) if c == old_ch]
                    if not positions:
                        continue
                    for pos in positions:
                        candidate = t[:pos] + new_ch + t[pos + 1:]
                        for ref_tag in reference_set:
                            score = _lev.ratio(candidate, str(ref_tag).upper())
                            if score > best_score:
                                best_score = score
                                best_match = ref_tag
                                best_match_source = 'substitution-1'

                # If single substitution is good enough, use it
                if best_match and best_score >= 0.92:
                    new_text[i] = best_match
                    corrections_applied += 1
                    logger.debug(
                        "_apply_table_ocr_corrections: corrected '%s' → '%s' (sim=%.2f, %s)",
                        t, best_match, best_score, best_match_source
                    )
                    continue

                # Try DOUBLE character substitutions (for cases like JSF-5725 → JSF-512S
                # which needs both 7→1 and 5→S)
                if best_score < 0.92:
                    for old_ch1, new_ch1 in CHAR_CONFUSIONS:
                        positions1 = [j for j, c in enumerate(t) if c == old_ch1]
                        if not positions1:
                            continue
                        for pos1 in positions1:
                            candidate1 = t[:pos1] + new_ch1 + t[pos1 + 1:]
                            for old_ch2, new_ch2 in CHAR_CONFUSIONS:
                                positions2 = [j for j, c in enumerate(candidate1) if c == old_ch2]
                                if not positions2:
                                    continue
                                for pos2 in positions2:
                                    if pos2 == pos1:
                                        continue
                                    candidate2 = candidate1[:pos2] + new_ch2 + candidate1[pos2 + 1:]
                                    for ref_tag in reference_set:
                                        score = _lev.ratio(candidate2, str(ref_tag).upper())
                                        if score > best_score:
                                            best_score = score
                                            best_match = ref_tag
                                            best_match_source = 'substitution-2'

                # If double substitution brought us close enough, use it
                if best_match and best_score >= 0.92:
                    new_text[i] = best_match
                    corrections_applied += 1
                    logger.debug(
                        "_apply_table_ocr_corrections: corrected '%s' → '%s' (sim=%.2f, %s)",
                        t, best_match, best_score, best_match_source
                    )

        if corrections_applied:
            logger.info(
                "_apply_table_ocr_corrections: page %d — applied %d character corrections",
                page_num + 1, corrections_applied
            )

        # ── STEP 3: Build final ocr_data ───────────────────────────────
        # Remove merged-away tokens
        final_indices = [i for i in range(n_tokens) if i not in indices_to_remove]

        corrected = {
            'text': [new_text[i] for i in final_indices],
            'left': [new_left[i] for i in final_indices],
            'top': [new_top[i] for i in final_indices],
            'width': [new_width[i] for i in final_indices],
            'height': [new_height[i] for i in final_indices],
            'conf': [new_conf[i] for i in final_indices],
            'block_num': [ocr_data.get('block_num', [0] * n_tokens)[i] for i in final_indices],
            'par_num': [ocr_data.get('par_num', [0] * n_tokens)[i] for i in final_indices],
            'line_num': [ocr_data.get('line_num', [0] * n_tokens)[i] for i in final_indices],
            'word_num': [ocr_data.get('word_num', [0] * n_tokens)[i] for i in final_indices],
        }

        logger.info(
            "_apply_table_ocr_corrections: page %d — %d tokens after corrections (was %d)",
            page_num + 1, len(final_indices), n_tokens
        )

        return corrected

    def _extract_from_ocr_data(self, ocr_data, pdf_type: str,
                                coord_source: str = 'ocr',
                                dpi_factor: float = 300 / 72):
        # Ensure pdf_type is canonical to keep thresholds/heuristics consistent
        _l = str(pdf_type or '').lower()
        if 'table' in _l:
            pdf_type = 'table'
        elif 'diagram' in _l or 'drawing' in _l:
            pdf_type = 'diagrams'
        else:
            pdf_type = 'diagrams'

        dominant_prefix = self._detect_dominant_prefix_in_page(ocr_data, ['UZSO', 'UZSC'])
        # ✅ [COORD FIX] هلپر داخلی برای ساخت bbox که منبع مختصات را هم ثبت می‌کند
        def _bbox(i):
            return {
                'x': int(ocr_data['left'][i]),
                'y': int(ocr_data['top'][i]),
                'width': int(ocr_data['width'][i]),
                'height': int(ocr_data['height'][i]),
                'coord_source': coord_source,   # 'ocr' یا 'digital'
                'dpi_factor': dpi_factor,       # ضریب تبدیل پوینت→پیکسل (فقط برای digital کاربرد دارد)
            }

        if dominant_prefix:
            logger.info(f"🎯 Page context: This page primarily contains {dominant_prefix} tags")
            self._current_page_dominant_prefix = dominant_prefix
        else:
            self._current_page_dominant_prefix = None

        # متغیرها
        tags = set()
        jb_identifiers = set()
        mc_identifiers = set()
        cable_descriptions = []
        spare_identifiers = []
        raw_cable_descriptions = []
        tag_match_info = {}
        
        all_ocr_tags = set()
        exact_matched_tags = set()
        similar_matched_tags = set()
        jb_positions = []
        mc_positions = []
        cable_positions = []
        
        io_list_tags = set()
        if hasattr(self, 'io_list_tags'):
            io_list_tags = self.io_list_tags
        elif hasattr(self, 'excel_df') and hasattr(self, 'excel_tag_column'):
            if self.excel_df is not None and not self.excel_df.empty:
                tag_col = self.excel_tag_column
                io_list_tags = set(str(tag).strip().upper() for tag in self.excel_df[tag_col] if pd.notna(tag))
        
        cable_patterns = [
            ('pair', re.compile(r'\b(\d{1,4})\s*(?:PAIR|PR|P)(?=\b|X|×|\*|/|-|\.|\d)', re.IGNORECASE)),
            ('triple', re.compile(r'\b(\d{1,4})\s*(?:TRIPLE|TR|T)(?=\b|X|×|\*|/|-|\.|\d)', re.IGNORECASE)),
            ('core', re.compile(r'\b(\d{1,4})\s*(?:CORE|CR|C)(?=\b|X|×|\*|/|-|\.|\d)', re.IGNORECASE)),
        ]
        
        mc_positions = []
        mc_indices = []
        spare_found_count = 0
        
        processed_tag_texts = set()
        processed_spare_indices = set()
        ocr_candidate_scores: Dict[str, float] = {}
        ocr_tag_positions: Dict[str, Dict[str, int]] = {}
        
        # GENERAL_TAG_PATTERN matches tag-like tokens.
        # Supports BOTH letter-starting (LUSY-2474A) AND digit-starting tags
        # (21HS-001, 11SAM10AN020XB91, 1M1).
        # Pattern breakdown:
        #   - Must contain at least one letter AND one digit
        #   - May start with 2-6 letters OR 1-4 digits
        #   - After the initial segment: dash/digits/letters combination
        #   - Total length 4-25 chars
        GENERAL_TAG_PATTERN = re.compile(
            r'^[A-Z0-9]{2,6}[-]?[A-Z0-9]{1,10}(?:[-][A-Z0-9]{1,10}){0,3}$',
            re.IGNORECASE
        )
        spare_pattern = re.compile(r'\b(spare)\b', re.IGNORECASE)
        io_pattern_profile = self._build_io_pattern_profile(io_list_tags)

        # ── STOP WORDS / COMMON ENGLISH WORDS FILTER ────────────────────
        # These are common English words that appear in PDF documents (headers,
        # footers, notes, labels) and should NEVER be treated as tags.
        # They get through the GENERAL_TAG_PATTERN because they consist of
        # uppercase letters with length >= 4, but they are clearly NOT tags.
        # We also filter pure-letter tokens that don't contain any digits
        # AND are not in the IO List — a real tag almost always has digits.
        _STOP_WORDS = frozenset({
            # Common document words
            'PORTION', 'ENERGY', 'PAIDAR', 'REMARKS', 'PLANT', 'EPCC',
            'PROPERTY', 'PAGE', 'WITH', 'MUST', 'COMPANY', 'HELD', 'OTHER',
            'TERMINATION', 'REPRODUCTION', 'CABINET', 'MULTIPLE', 'THIS',
            'TYPE', 'DOCUMENT', 'WITHOUT', 'UNIT', 'CONNECTION', 'WIRING',
            'SCREEN', 'IDENT', 'CABLE', 'CONSENT', 'JUNCTION', 'ROOM',
            'RIGHTS', 'METHANOL', 'MADE', 'MULTI', 'SECONDARY', 'WHOLE',
            'OVERALL', 'INSTRUMENT', 'DIAGRAM', 'PRIOR', 'LIST', 'GROUP',
            'FIELD', 'STRIP', 'CODE', 'SERIAL', 'CONFIDENCE', 'HOLD',
            'SOROUSH', 'SEPID', 'SHEET', 'NOTE', 'NOTES', 'DATE', 'REV',
            'REVISION', 'DRAWING', 'APPROVED', 'CHECKED', 'DESIGNED',
            'PROJECT', 'CLIENT', 'CONTRACTOR', 'CONSULTANT', 'VENDOR',
            'LOCATION', 'DESCRIPTION', 'QUANTITY', 'REMARK', 'STATUS',
            'SERVICE', 'BUILDING', 'AREA', 'LEVEL', 'FLOOR', 'ZONE',
            'SYSTEM', 'SUBSYSTEM', 'EQUIPMENT', 'MATERIAL', 'MANUFACTURER',
            'MODEL', 'SUPPLIER', 'ORDER', 'ITEM', 'REF', 'REFERENCE',
            'CONTINUE', 'CONTINUED', 'SHEET', 'SCALE', 'SIZE', 'DWG',
            'ISSUE', 'RECORD', 'MASTER', 'INDEX', 'GENERAL', 'DETAIL',
            'SECTION', 'ELEVATION', 'PLAN', 'LEGEND', 'ABBREVIATION',
            'NOTE', 'TYPICAL', 'TYP', 'MINIMUM', 'MAXIMUM', 'NOMINAL',
            'SCHEDULE', 'SUMMARY', 'TOTAL', 'SUBTOTAL', 'GRAND',
            'INPUT', 'OUTPUT', 'SIGNAL', 'POWER', 'GROUND', 'EARTH',
            'PHASE', 'NEUTRAL', 'LINE', 'LOAD', 'SOURCE', 'RETURN',
            'FEEDER', 'BRANCH', 'CIRCUIT', 'PANEL', 'BOARD', 'RACK',
            'CHASSIS', 'FRAME', 'ENCLOSURE', 'BOX', 'OUTLET', 'SWITCH',
            'BREAKER', 'FUSE', 'RELAY', 'CONTACTOR', 'TRANSFORMER',
            'MOTOR', 'GENERATOR', 'BATTERY', 'UPS', 'RECTIFIER',
            'INVERTER', 'CONVERTER', 'DRIVE', 'STARTER', 'CONTROLLER',
            'SENSOR', 'TRANSMITTER', 'INDICATOR', 'RECORDER', 'GAUGE',
            'METER', 'VALVE', 'ACTUATOR', 'SOLENOID', 'COIL', 'HEATER',
            'WARNING', 'CAUTION', 'DANGER', 'NOTICE', 'IMPORTANT',
            'REQUIRED', 'OPTIONAL', 'PENDING', 'COMPLETE', 'CANCELLED',
            'VOID', 'VALID', 'EXPIRED', 'APPROVED', 'REJECTED',
            'CHECK', 'VERIFY', 'INSPECT', 'TEST', 'COMMISSION',
            'INSTALL', 'COMPILE', 'PREPARE', 'SUBMIT', 'APPROVE',
            'REVIEW', 'COMMENT', 'RESPONSE', 'ACTION', 'REPLY',
            # Numbers that are page numbers or IDs, not tags
            'LINE', 'LOOP', 'TAG', 'NO', 'NUMBER',
        })
        # ────────────────────────────────────────────────────────────────────
        
        # ============================================================
        # 🆕 ذخیره موقعیت‌های تگ‌ها و SPARE ها
        # ============================================================
        tags_with_positions = []
        spares_with_positions = []
        phase0_pattern_threshold = 0.50 if pdf_type == 'table' else 0.62
        logger.info(
            "extract_from_image Phase 0 pattern threshold: %.2f (%s mode)",
            phase0_pattern_threshold, pdf_type
        )
        # ============================================================
        # Phase 0: Extract ALL OCR tags
        # ============================================================
        logger.info("Phase 0: Extracting ALL OCR tags...")
        
        # ── JB/MC FRAGMENT PATTERN (applies to ALL pdf types, not just table) ──
        # Pre-compile a JB fragment pattern. This catches tokens like "JSF-5765"
        # that are JB identifiers split from their "NC-" prefix by OCR.
        # Previously this was table-only, but JB/MC fragments leak into tags
        # in diagram mode too (especially in digital extraction path).
        _table_jb_re = re.compile(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', re.IGNORECASE)
        _table_mc_re = re.compile(r'^NC-?[A-Z]{0,5}-?\d{2,5}[A-Z]?$', re.IGNORECASE)
        # ────────────────────────────────────────────────────────────────────

        # ── Parse JB/MC prefixes ONCE (clean, uppercased, no whitespace) ──
        # Use jb_examples_list / mc_examples_list if available (set by set_patterns),
        # otherwise fall back to parsing the raw string.
        _jb_prefixes = list(getattr(self, 'jb_examples_list', None) or [])
        if not _jb_prefixes and getattr(self, 'jb_examples', None):
            _jb_prefixes = self._parse_multi_patterns(self.jb_examples)
        _mc_prefixes = list(getattr(self, 'mc_examples_list', None) or [])
        if not _mc_prefixes and getattr(self, 'mc_examples', None):
            _mc_prefixes = self._parse_multi_patterns(self.mc_examples)
        logger.debug(
            "Phase 0 filter prefixes — JB: %s, MC: %s (pdf_type=%s)",
            _jb_prefixes, _mc_prefixes, pdf_type
        )

        # ── STRUCTURE-BASED JB DETECTION (project-agnostic) ────────────────
        # Even if a JB prefix is not in jb_examples, detect it by structure.
        # We learn the structure from the user's provided JB prefixes:
        #   - Extract the common first letter(s) as "roots"
        #   - Extract the typical prefix length
        #   - Build a regex that matches siblings of the same family
        #
        # Examples:
        #   User provides: JSF,JDF,JAF,JAE,JAG,JSG,JDG
        #     → roots = ['J'], prefix_len = 3
        #     → pattern: ^(J)[A-Z]{2,3}-\d{2,5}[A-Z]?$
        #     → catches: JSE-312S, JSG-5765, JSD-123 (siblings)
        #
        #   User provides: JB,JBOX,CB
        #     → roots = ['J', 'C'], prefix_len varies
        #     → pattern: ^(J|C)[A-Z]{1,3}-\d{2,5}[A-Z]?$
        #     → catches: JB-123, JBOX-456, CB-001, CD-789
        _jb_root_pattern = None
        if _jb_prefixes:
            # Extract the first character of each JB prefix as a "root".
            # This groups prefixes into families (J-family, C-family, etc.)
            _jb_roots = sorted(set(p[0] for p in _jb_prefixes if p))
            if _jb_roots:
                # Determine the range of JB prefix lengths (min, max)
                _jb_len_min = min(len(p) for p in _jb_prefixes if p)
                _jb_len_max = max(len(p) for p in _jb_prefixes if p)
                # Build a regex that matches:
                #   <root> + 0 to (max_len - 1) extra letters + dash + 2-5 digits + optional letter
                _root_alt = '|'.join(re.escape(r) for r in _jb_roots)
                # Allow the prefix to be 2 to max_len+1 chars total (root + extra letters)
                _extra_min = max(0, _jb_len_min - 1)
                _extra_max = max(1, _jb_len_max)
                _jb_root_pattern = re.compile(
                    r'^(' + _root_alt + r')[A-Z]{' + str(_extra_min) + r',' + str(_extra_max) + r'}-\d{2,5}[A-Z]?$',
                    re.IGNORECASE
                )
                logger.debug(
                    "Phase 0 structure-based JB detection: roots=%s, prefix_len_range=(%d,%d), pattern=%s",
                    _jb_roots, _jb_len_min, _jb_len_max, _jb_root_pattern.pattern
                )
        # ────────────────────────────────────────────────────────────────────
        
        for i, word in enumerate(ocr_data['text']):
            word_clean = self._normalize_ocr_tag_candidate(word)
            # ── TABLE-MODE cleanup ─────────────────────────────────────────────
            # Table cell borders are often misread by OCR as leading/trailing
            # punctuation like "[UY-5021" or "UY-5021]". Strip these so that
            # GENERAL_TAG_PATTERN can match.
            if pdf_type == 'table':
                word_clean = word_clean.strip('[](){}|<>\'"')
            # ────────────────────────────────────────────────────────────────────
            if not word_clean or len(word_clean) < 4:
                continue
            
            # ── JB/MC FRAGMENT FILTERING (applies to ALL pdf types) ────────
            # Skip tokens that look like JB or MC identifiers even when OCR
            # splits them from their prefix (e.g. "JSF-5765" without "NC-").
            # Previously this was table-only, but the same fragments leak in
            # diagram mode too — causing JB IDs to be misclassified as tags.
            _w_upper = word_clean.upper()
            _skip_token = False

            # 1) Catch any token starting with MC prefix (e.g. "NC", "FG")
            if _mc_prefixes and any(_w_upper.startswith(p) for p in _mc_prefixes):
                _skip_token = True

            # 2) Catch JB fragments that match the JB pattern AND start with a JB prefix
            #    e.g. "JSF-427S" matches _table_jb_re and starts with "JSF"
            if not _skip_token and _table_jb_re and _table_jb_re.match(word_clean):
                if _jb_prefixes and any(_w_upper.startswith(p) for p in _jb_prefixes):
                    _skip_token = True

            # 3) Catch tokens that are exactly a JB prefix + dash + digits
            #    e.g. "JSF-427", "JDF-123" (even if not matching _table_jb_re exactly)
            if not _skip_token and _jb_prefixes:
                for p in _jb_prefixes:
                    if _w_upper.startswith(p) and len(_w_upper) > len(p):
                        next_ch = _w_upper[len(p)]
                        if next_ch == '-' or next_ch.isdigit():
                            # Looks like a JB identifier (prefix-dash-digits or prefix-digits)
                            _skip_token = True
                            break

            # 4) STRUCTURE-BASED JB DETECTION (catches siblings not in jb_examples)
            #    e.g. user provided "JSF,JDF,JAF,JAE,JAG,JSG,JDG" but PDF has "JSE-312S"
            #    → JSE is not in the list, but matches the J+letters-dash-digits pattern.
            #    This step catches it by structure, preventing JB IDs from leaking into tags.
            #
            #    IMPORTANT: Do NOT filter if the token is already in the IO list —
            #    it's a real tag, not a JB sibling, even if it looks like one.
            if not _skip_token and _jb_root_pattern and _jb_root_pattern.match(word_clean):
                # Check if this exact token is in the IO list — if so, keep it
                if word_clean not in io_list_tags:
                    _skip_token = True
                else:
                    logger.debug("Phase 0: keeping '%s' despite JB-like structure (exact IO list match)", word_clean)

            if _skip_token:
                continue
            # ────────────────────────────────────────────────────────────────────

            # ── MAIN JB/MC/SPARE FILTER (uses clean prefix lists) ──────────
            # Use _jb_prefixes / _mc_prefixes (already parsed, uppercased, no
            # whitespace) instead of raw self.jb_examples string — this avoids
            # issues with comma/space-separated patterns not being split properly.
            #
            # IMPORTANT: Do NOT filter if the token is already in the IO list.
            # A token that matches a JB prefix but is also in the IO list is a
            # real tag, not a JB identifier.
            _in_io_list = word_clean in io_list_tags
            _is_jb = bool(_jb_prefixes) and any(word_clean.startswith(p) for p in _jb_prefixes)
            _is_mc = bool(_mc_prefixes) and any(
                self._is_prefixed_identifier(word_clean, p, require_digit=False)
                for p in _mc_prefixes
            )
            # Also use structure-based JB detection in the main filter as a safety net
            if not _is_jb and _jb_root_pattern and _jb_root_pattern.match(word_clean):
                _is_jb = True
            if (
                (_is_jb or _is_mc) and not _in_io_list
            ) or (
                spare_pattern.search(word_clean) or
                self._is_non_tag_pattern(word_clean)
            ):
                continue

            # ── STOP WORDS + PURE-LETTER FILTER ──────────────────────────
            # Filter out common English words (PORTION, ENERGY, REMARKS, etc.)
            # and pure-letter tokens that have no digits and are not in the
            # IO List. A real industrial tag almost always contains digits
            # (e.g. LUSY-2474A, 21HS-001, 11SAM10AN020XB91).
            # Exception: if the token IS in the IO List, keep it regardless.
            if not _in_io_list:
                # Check stop words list
                if word_clean in _STOP_WORDS:
                    continue
                # Filter pure-letter tokens (no digits at all) that are >= 4 chars
                # These are almost certainly document text, not tags.
                # Exception: very short tokens (3 chars) might be tag prefixes
                # like "FIT", "PIT" — but those should have digits in the full tag.
                if len(word_clean) >= 4 and not any(c.isdigit() for c in word_clean):
                    continue
                # Filter pure-number tokens (no letters) — these are page numbers,
                # dates, or other numeric values, not tags.
                if len(word_clean) >= 4 and not any(c.isalpha() for c in word_clean):
                    continue
                # Filter wire color codes (e.g. 12BK, 18BK, 20BK)
                if re.match(r'^\d{1,3}(BK|WH|RD|BL|GN|YL|OR|GY|VI|BN|PK)$', word_clean):
                    continue
            # ───────────────────────────────────────────────────────────────
            
            pattern_score = self._score_pattern_candidate(word_clean, io_pattern_profile)

            if GENERAL_TAG_PATTERN.match(word_clean) or pattern_score >= 0.62:
                all_ocr_tags.add(word_clean)
                ocr_candidate_scores[word_clean] = max(ocr_candidate_scores.get(word_clean, 0.0), pattern_score)
                if word_clean not in ocr_tag_positions:
                    ocr_tag_positions[word_clean] = _bbox(i)
                logger.debug(f"Found OCR tag: {word_clean}")
        
        logger.info(f"Phase 0 complete: {len(all_ocr_tags)} OCR tags")
        if all_ocr_tags:
            logger.info(f"Phase 0 sample tags before matching: {sorted(list(all_ocr_tags))[:20]}")
        else:
            logger.info("Phase 0 candidate tag list is empty")
        
        # ============================================================
        # Phase 1: EXACT matches
        # ============================================================
        logger.info("Phase 1: Searching for EXACT matches...")

        if not hasattr(self, 'vector_matcher'):
            logger.error("❌ vector_matcher NOT FOUND!")
        else:
            for ocr_tag in all_ocr_tags:
                similar_tags = self.vector_matcher.find_similar_tags(ocr_tag)
                
                if similar_tags:
                    best_match, best_score = similar_tags[0]
                    
                    if best_score >= 1.0 and ocr_tag == best_match.upper():
                        exact_matched_tags.add(best_match)
                        tags.add(best_match)
                        
                        # 🆕 ذخیره موقعیت تگ
                        bbox = None
                        for i, word in enumerate(ocr_data['text']):
                            if word.strip().upper() == ocr_tag:
                                bbox = _bbox(i)
                                tags_with_positions.append({
                                    'tag': best_match,
                                    'y': ocr_data['top'][i],
                                    'x': ocr_data['left'][i],
                                    'width': ocr_data['width'][i],
                                    'height': ocr_data['height'][i],
                                    'ocr_text': ocr_tag,
                                    'coord_source': coord_source,
                                    'dpi_factor': dpi_factor
                                })
                                break           
                        
                        tag_match_info[best_match] = {
                            'match_type': 'exact',
                            'score': best_score,
                            'ocr_text': ocr_tag,
                            'bbox': bbox or {},
                            'coord_source': coord_source,
                            'dpi_factor': dpi_factor
                        }
                        
                        processed_tag_texts.add(ocr_tag)
                        logger.info(f"✅ EXACT: {ocr_tag} → {best_match}")

        logger.info(f"Phase 1 complete: {len(exact_matched_tags)} exact")
        
        # ============================================================
        # Phase 2: STRICT Similar matches
        # ============================================================
        logger.info("Phase 2: Searching for SIMILAR matches (STRICT mode)...")
        phase2_lower_gate = 0.94 if pdf_type == 'table' else 0.96
        logger.info(
            "extract_from_image Phase 2 similarity gate: %.2f (%s mode)",
            phase2_lower_gate, pdf_type
        )
        
        similar_rejected_count = 0
        
        for ocr_tag in all_ocr_tags:
            if ocr_tag in processed_tag_texts:
                continue
            
            similar_tags = self.vector_matcher.find_similar_tags(ocr_tag)
            
            if similar_tags:
                best_match, best_score = similar_tags[0]
                
                # STRICT VALIDATION RULES
                if not (0.96 <= best_score < 1.0):
                    continue
                
                len_diff = abs(len(ocr_tag) - len(best_match))
                if len_diff > 1:
                    logger.debug(f"❌ REJECTED (length): {ocr_tag} → {best_match}")
                    similar_rejected_count += 1
                    continue
                
                ocr_prefix = self._extract_tag_prefix(ocr_tag)
                io_prefix = self._extract_tag_prefix(best_match)
                
                if ocr_prefix != io_prefix:
                    logger.debug(f"❌ REJECTED (prefix): {ocr_tag} → {best_match}")
                    similar_rejected_count += 1
                    continue
                
                ocr_parts = ocr_tag.split('-')
                io_parts = best_match.split('-')
                
                if len(ocr_parts) != len(io_parts):
                    logger.debug(f"❌ REJECTED (structure): {ocr_tag} → {best_match}")
                    similar_rejected_count += 1
                    continue
                
                if not self._are_numbers_identical(ocr_tag, best_match):
                    logger.debug(f"❌ REJECTED (numbers differ): {ocr_tag} → {best_match}")
                    similar_rejected_count += 1
                    continue
                
                if len_diff == 0 and self._count_different_chars(ocr_tag, best_match) == 1:
                    diff_char_ocr, diff_char_io = self._get_different_chars(ocr_tag, best_match)
                    
                    ocr_confusion_pairs = [
                        ('O', '0'), ('0', 'O'),
                        ('I', '1'), ('1', 'I'), ('l', '1'), ('1', 'l'),
                        ('S', '5'), ('5', 'S'),
                        ('B', '8'), ('8', 'B'),
                        ('Z', '2'), ('2', 'Z'),
                    ]
                    
                    is_ocr_error = (diff_char_ocr, diff_char_io) in ocr_confusion_pairs or \
                                (diff_char_io, diff_char_ocr) in ocr_confusion_pairs
                    
                    if not is_ocr_error:
                        logger.debug(f"❌ REJECTED (not OCR error): {ocr_tag} → {best_match}")
                        similar_rejected_count += 1
                        continue
                
                # PASSED ALL RULES - Accept
                if best_match not in exact_matched_tags and best_match not in similar_matched_tags:
                    similar_matched_tags.add(best_match)
                    tags.add(best_match)
                    
                    # 🆕 ذخیره موقعیت تگ
                    bbox = None
                    for i, word in enumerate(ocr_data['text']):
                        if word.strip().upper() == ocr_tag:
                            bbox = _bbox(i)
                            tags_with_positions.append({
                                'tag': best_match,
                                'y': ocr_data['top'][i],
                                'x': ocr_data['left'][i],
                                'width': ocr_data['width'][i],
                                'height': ocr_data['height'][i],
                                'ocr_text': ocr_tag,
                                'coord_source': coord_source,
                                'dpi_factor': dpi_factor
                            })
                            break
                
                    tag_match_info[best_match] = {
                        'match_type': 'similar',
                        'score': best_score,
                        'ocr_text': ocr_tag,
                        'reason': self._get_similarity_reason(ocr_tag, best_match),
                        'bbox': bbox or {},
                        'coord_source': coord_source,
                        'dpi_factor': dpi_factor
                    }
                    
                    processed_tag_texts.add(ocr_tag)
                    logger.info(f"⚠️ SIMILAR: {ocr_tag} → {best_match} ({best_score:.3f})")

        logger.info(f"Phase 2 complete: {len(similar_matched_tags)} similar, {similar_rejected_count} rejected")

        # ============================================================
        # Phase 2.5: Pattern-based unmatched candidates
        # ============================================================
        # Learn the tag naming pattern from the IO List (e.g. "LUSY-XXXX",
        # "USY-XXXXX", "21HS-XXX") and find ALL OCR tokens in the PDF that
        # match this pattern but are NOT in the IO List. These are drawn
        # as error/unmatched bounding boxes (red) on the annotated PDF.
        logger.info("Phase 2.5: Detecting IO-pattern candidates not found in IO List...")
        unmatched_pattern_count = 0

        # ── LEARN TAG PATTERNS FROM IO LIST ────────────────────────────
        # Extract prefix patterns: e.g. "LUSY-2474A" → pattern "LUSY-"
        # "USY-2482A" → pattern "USY-", "21HS-001" → pattern "\d{2}HS-"
        _io_prefixes_set = set()
        _io_regex_patterns = []
        for io_tag in (io_list_tags or set()):
            tag_upper = str(io_tag).upper().strip()
            if not tag_upper or len(tag_upper) < 4:
                continue
            # Extract the alpha prefix (letters before first digit or dash)
            prefix_match = re.match(r'^([A-Z]{2,6})[-]?', tag_upper)
            if prefix_match:
                prefix = prefix_match.group(1)
                _io_prefixes_set.add(prefix)
            # Also extract digit-starting patterns
            digit_prefix_match = re.match(r'^(\d{1,4}[A-Z]{1,6})[-]?', tag_upper)
            if digit_prefix_match:
                _io_prefixes_set.add(digit_prefix_match.group(1))

        # Build regex patterns from learned prefixes
        # e.g. prefix "LUSY" → r'^LUSY[-\d]'
        for prefix in _io_prefixes_set:
            _io_regex_patterns.append(re.compile(r'^' + re.escape(prefix) + r'[-\d]', re.IGNORECASE))

        logger.info(
            f"Phase 2.5: learned {len(_io_prefixes_set)} tag prefixes from IO List: {sorted(_io_prefixes_set)[:15]}"
        )

        # ── FIND ALL OCR TAGS THAT MATCH IO PATTERN BUT ARE NOT IN IO LIST ──
        for idx, ocr_tag in enumerate(sorted(all_ocr_tags)):
            if ocr_tag in processed_tag_texts:
                continue

            # Check if this OCR tag matches any learned IO pattern
            matches_io_pattern = False
            for pat in _io_regex_patterns:
                if pat.match(ocr_tag):
                    matches_io_pattern = True
                    break

            # Also accept if it has a high pattern_score (original logic)
            candidate_score = ocr_candidate_scores.get(ocr_tag, 0.0)
            if not matches_io_pattern and candidate_score < 0.62:
                continue

            # Double-check: is it actually in the IO List? (case-insensitive)
            is_in_io_list = ocr_tag.upper() in {str(t).upper().strip() for t in (io_list_tags or set())}
            if is_in_io_list:
                continue  # Already matched, skip

            # This is an unmatched candidate — it matches the IO pattern
            # but is NOT in the IO List. Draw an error bounding box for it.
            candidate_key = f"UNMATCHED_CANDIDATE::{ocr_tag}::{idx}"
            candidate_pos = ocr_tag_positions.get(ocr_tag)
            if candidate_pos:
                tags_with_positions.append({
                    'tag': ocr_tag,
                    'y': candidate_pos.get('y', 0),
                    'x': candidate_pos.get('x', 0),
                    'width': candidate_pos.get('width', 0),
                    'height': candidate_pos.get('height', 0),
                    'ocr_text': ocr_tag,
                    'coord_source': coord_source,
                    'dpi_factor': dpi_factor
                })
            tag_match_info[candidate_key] = {
                'match_type': 'unmatched_candidate',
                'score': round(max(candidate_score, 0.8 if matches_io_pattern else candidate_score), 3),
                'ocr_text': ocr_tag,
                'display_text': ocr_tag,
                'reason': 'Tag matches IO pattern but not found in IO List',
                'bbox': candidate_pos if candidate_pos else {},
                'coord_source': coord_source,
                'dpi_factor': dpi_factor
            }
            unmatched_pattern_count += 1
            logger.info(f"  ⚠️ UNMATCHED CANDIDATE: '{ocr_tag}' (matches IO pattern, score={candidate_score:.2f})")

        logger.info(f"Phase 2.5 complete: {unmatched_pattern_count} pattern-based unmatched candidates")
        
        # ============================================================
        # Phase 3: Process JB, MC, SPARE
        # ============================================================
        logger.info("Phase 3: Processing SPARE, MC, JB identifiers...")
        
        for i, word in enumerate(ocr_data['text']):
            word_clean = word.strip().upper()
            if not word_clean:
                continue
            
            if spare_pattern.search(word_clean):
                curr_x = ocr_data['left'][i]
                curr_y = ocr_data['top'][i]
                
                # position-based duplicate check به جای index-based
                is_duplicate = any(
                    abs(s['x'] - curr_x) < 30 and abs(s['y'] - curr_y) < 15
                    for s in spares_with_positions
                )
                
                logger.info(
                    f"SPARE candidate: '{word_clean}' "
                    f"index={i} x={curr_x} y={curr_y} "
                    f"duplicate={is_duplicate}"
                )
                
                if not is_duplicate:
                    spare_identifiers.append(word_clean)
                    spare_found_count += 1
                    
                    spares_with_positions.append({
                        'spare': word_clean,
                        'y': curr_y,
                        'x': curr_x,
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'coord_source': coord_source,
                        'dpi_factor': dpi_factor
                    })
                    
                    spare_id = f"{self.spare_examples}_{spare_found_count}"
                    
                    tag_match_info[spare_id] = {
                        'match_type': 'spare',
                        'score': 1.0,
                        'ocr_text': word_clean,
                        'bbox': _bbox(i),
                        'coord_source': coord_source,
                        'dpi_factor': dpi_factor
                    }
                    
                    logger.info(f"✅ SPARE FOUND: {word_clean} → ID: {spare_id} x={curr_x} y={curr_y}")
                continue
            
            # MC — use properly parsed prefix list (not raw string)
            mc_token = self._normalize_code_token(word_clean)
            _mc_prefixes_phase3 = getattr(self, 'mc_examples_list', None) or []
            if not _mc_prefixes_phase3 and getattr(self, 'mc_examples', None):
                _mc_prefixes_phase3 = self._parse_multi_patterns(self.mc_examples)
            if any(self._is_prefixed_identifier(mc_token, p, require_digit=False) for p in _mc_prefixes_phase3 if p):
                x, y = int(ocr_data['left'][i]), int(ocr_data['top'][i])
                mc_positions.append({
                    'mc': mc_token,
                    'x': x,
                    'y': y,
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i])
                })
                mc_indices.append(i)
                mc_identifiers.add(mc_token)
                tag_match_info[mc_token] = {
                    'match_type': 'mc',
                    'score': 1.0,
                    'ocr_text': word_clean,
                    'bbox': _bbox(i),
                    'coord_source': coord_source,
                    'dpi_factor': dpi_factor
                }
                logger.info(f"MC: {mc_token}")
                continue
            
            # JB — use properly parsed prefix list (not raw split)
            _jb_prefixes_phase3 = getattr(self, 'jb_examples_list', None) or []
            if not _jb_prefixes_phase3 and getattr(self, 'jb_examples', None):
                _jb_prefixes_phase3 = self._parse_multi_patterns(self.jb_examples)
            if any(word_clean.startswith(p) for p in _jb_prefixes_phase3 if p):
                x, y = int(ocr_data['left'][i]), int(ocr_data['top'][i])
                jb_positions.append({
                    'jb': word_clean,
                    'x': x,
                    'y': y,
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                    'coord_source': coord_source,
                    'dpi_factor': dpi_factor
                })
                jb_identifiers.add(word_clean)
                tag_match_info[word_clean] = {
                    'match_type': 'jb',
                    'score': 1.0,
                    'ocr_text': word_clean,
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': int(ocr_data['width'][i]),
                        'height': int(ocr_data['height'][i])
                    }
                }
                logger.info(f"JB: {word_clean}")
                continue
        
        # ============================================================
        # Phase 4: Cable descriptions
        # ============================================================
        logger.info("Phase 4: Extracting cable descriptions...")
        
        for mc_i in mc_indices:
            mc_x, mc_y = ocr_data['left'][mc_i], ocr_data['top'][mc_i]
            mc_text = str(ocr_data['text'][mc_i]).strip().upper()

            # جستجوی چندمرحله‌ای:
            # 1) پنجره محدود نزدیک MC
            # 2) در صورت عدم یافتن، پنجره کمی بازتر برای جابه‌جایی OCR
            window_candidates = [
                (40, 180, 100, 18),
                (120, 300, 130, 24),
                (220, 380, 170, 30),
            ]

            best_hit = None
            for win_idx, (max_left_offset, max_right_offset, search_radius_y, same_row_tolerance) in enumerate(window_candidates):
                nearby_entries = []
                for j, word_j in enumerate(ocr_data['text']):
                    token = str(word_j).strip().upper()
                    if not token:
                        continue

                    word_x, word_y = ocr_data['left'][j], ocr_data['top'][j]
                    word_w, word_h = ocr_data['width'][j], ocr_data['height'][j]
                    distance_y = abs(word_y - mc_y)
                    x_offset = int(word_x) - int(mc_x)

                    if (distance_y <= search_radius_y and
                        -max_left_offset <= x_offset <= max_right_offset):
                        nearby_entries.append({
                            'idx': j,
                            'text': token,
                            'x': int(word_x),
                            'y': int(word_y),
                            'w': int(word_w),
                            'h': int(word_h)
                        })

                if not nearby_entries:
                    continue

                candidate_hits = []
                seen_hits = set()

                def _collect_cable_matches(source_text, source_x, source_y, source_w=0, source_h=0):
                    normalized_text = re.sub(r'\s+', ' ', str(source_text).upper()).strip()
                    if not normalized_text:
                        return

                    for cable_type_full, pattern in cable_patterns:
                        for match in pattern.finditer(normalized_text):
                            number_raw = match.group(1)
                            try:
                                number = int(number_raw)
                            except Exception:
                                continue

                            if number <= 0:
                                continue

                            cable_desc = f"{number} {cable_type_full}"
                            is_above_mc = int(source_y) < int(mc_y)
                            distance_score = abs(int(source_x) - int(mc_x)) + (3.0 * abs(int(source_y) - int(mc_y)))
                            if is_above_mc:
                                distance_score += 40.0
                            hit_key = (cable_desc, int(source_x), int(source_y), normalized_text)
                            if hit_key in seen_hits:
                                continue

                            seen_hits.add(hit_key)
                            candidate_hits.append({
                                'cable_desc': cable_desc,
                                'source_text': normalized_text,
                                'source_x': int(source_x),
                                'source_y': int(source_y),
                                'source_w': int(source_w),
                                'source_h': int(source_h),
                                'distance': distance_score
                            })

                # کاندید مستقیم از خود توکن (مثل FRT-12PX0.75MM2)
                for entry in nearby_entries:
                    _collect_cable_matches(entry['text'], entry['x'], entry['y'], entry.get('w', 0), entry.get('h', 0))

                # کاندید ترکیبی از دو توکن هم‌ردیف (مثل "12" + "PAIR")
                row_tolerance = same_row_tolerance
                max_pair_gap = 150 if win_idx == 0 else 220
                for i in range(len(nearby_entries)):
                    for j in range(i + 1, len(nearby_entries)):
                        e1 = nearby_entries[i]
                        e2 = nearby_entries[j]

                        if abs(e1['y'] - e2['y']) > row_tolerance:
                            continue

                        left, right = (e1, e2) if e1['x'] <= e2['x'] else (e2, e1)
                        if (right['x'] - left['x']) > max_pair_gap:
                            continue

                        center_x = int((left['x'] + right['x']) / 2)
                        center_y = int((left['y'] + right['y']) / 2)
                        # Merged width = from left edge of left token to right edge of right token
                        merged_w = int(right['x'] + right.get('w', 0) - left['x'])
                        merged_h = max(left.get('h', 0), right.get('h', 0))
                        _collect_cable_matches(f"{left['text']} {right['text']}", center_x, center_y, merged_w, merged_h)
                        _collect_cable_matches(f"{left['text']}{right['text']}", center_x, center_y, merged_w, merged_h)

                if candidate_hits:
                    best_hit = min(candidate_hits, key=lambda item: (item['distance'], len(item['source_text'])))
                    break

            if not best_hit:
                debug_nearby = []
                for j, word_j in enumerate(ocr_data['text']):
                    token = str(word_j).strip().upper()
                    if not token:
                        continue
                    word_x, word_y = int(ocr_data['left'][j]), int(ocr_data['top'][j])
                    dx = abs(word_x - int(mc_x))
                    dy = abs(word_y - int(mc_y))
                    if dx <= 420 and dy <= 200:
                        debug_nearby.append((dx + dy, token))
                debug_nearby.sort(key=lambda x: x[0])
                sample_tokens = [t for _, t in debug_nearby[:10]]
                logger.warning(f"No cable description matched near MC '{mc_text}' at ({mc_x},{mc_y}). Nearby OCR sample: {sample_tokens}")
                continue

            best_desc = best_hit['cable_desc']
            best_text = self.clean_cable_description(best_hit['source_text'], mc_identifiers)
            # Use the ACTUAL source token position and size, not the MC position
            x = int(best_hit.get('source_x', mc_x))
            y = int(best_hit.get('source_y', mc_y))
            # Use actual source width/height if available, otherwise estimate
            src_w = int(best_hit.get('source_w', 0))
            src_h = int(best_hit.get('source_h', 0))
            # If source_w is 0 (not available), use a reasonable default based on text length
            if src_w <= 0:
                src_w = max(60, len(best_text) * 12)  # ~12px per char at 300 DPI
            if src_h <= 0:
                src_h = 24  # default text height
            logger.info(f"Cable matched near MC '{mc_text}': code='{best_desc}', raw='{best_text}', bbox=({x},{y}) {src_w}x{src_h}")

            if best_desc not in cable_descriptions:
                cable_descriptions.append(best_desc)

            if best_text and best_text not in raw_cable_descriptions:
                raw_cable_descriptions.append(best_text)

            cable_positions.append({
                'text': best_desc,
                'display_text': best_desc,
                'x': x,
                'y': y,
                'width': src_w,
                'height': src_h,
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': src_w,
                    'height': src_h,
                    'coord_source': coord_source,
                    'dpi_factor': dpi_factor
                }
            })
            tag_match_info[f"CABLE::{best_desc}::{len(cable_positions)}"] = {
                'match_type': 'cable',
                'score': 1.0,
                'ocr_text': best_text,
                'display_text': best_desc,
                'bbox': {
                    'x': x,
                    'y': y,
                    'width': src_w,
                    'height': src_h,
                    'coord_source': coord_source,
                    'dpi_factor': dpi_factor
                },
                'coord_source': coord_source,
                'dpi_factor': dpi_factor
            }
        
        # ============================================================
        # 🆕 Phase 5: شماره‌گذاری بر اساس موقعیت عمودی
        # ============================================================
        logger.info("Phase 5: Assigning numbers based on VERTICAL POSITION...")
        
        tag_to_number = self.assign_tag_numbers_by_position(
            tags_with_positions,
            spares_with_positions
        )
        
        # اضافه کردن SPARE IDs به tag_match_info
        for idx in range(len(spares_with_positions)):
            spare_id = f"{self.spare_examples}_{idx + 1}"
            if spare_id not in tag_match_info:
                tag_match_info[spare_id] = {
                    'match_type': 'spare',
                    'score': 1.0,
                    'ocr_text': spares_with_positions[idx].get('spare', 'SPARE')
                }

        # Enrich unmatched candidates with position-based numbering and derived columns
        default_cable_desc = raw_cable_descriptions[0] if raw_cable_descriptions else ''
        default_cable_code = cable_descriptions[0] if cable_descriptions else ''
        for candidate_key, info in tag_match_info.items():
            if not isinstance(info, dict) or info.get('match_type') != 'unmatched_candidate':
                continue
            candidate_text = self._normalize_ocr_tag_candidate(info.get('ocr_text', info.get('display_text', '')))
            if not candidate_text:
                continue
            candidate_number = tag_to_number.get(candidate_text)
            if not candidate_number:
                continue

            terminal_info = self.generate_terminal_numbers(candidate_number)
            wire_colors_str = self.generate_mc_wire_colors_enhanced(candidate_number)
            wire_colors = [c.strip() for c in str(wire_colors_str).split(',') if str(c).strip()]
            wire_code_1 = wire_colors[0] if len(wire_colors) > 0 else ''
            wire_code_2 = wire_colors[1] if len(wire_colors) > 1 else ''

            info['tag_number'] = int(candidate_number)
            info['wire_colors_text'] = wire_colors_str
            info['wire_colors'] = wire_colors
            info['wire_code_1'] = wire_code_1
            info['wire_code_2'] = wire_code_2
            info['terminal_first_number'] = terminal_info.get('terminal_first', '')
            info['terminal_second_number'] = terminal_info.get('terminal_second', '')
            info['scr_terminal_number'] = terminal_info.get('scr_terminal', '')
            info['cable_code'] = default_cable_code
            info['cable_description'] = default_cable_desc
            info['type'] = 'Tag'
            info['tag_number_status'] = 'Assigned (Position-based candidate)'

        # ============================================================
        # Final logging
        # ============================================================
        logger.info(f'='*60)
        logger.info(f'EXTRACTION COMPLETE - Final Results:')
        logger.info(f'  📊 Tags:')
        logger.info(f'     - Exact matches: {len(exact_matched_tags)}')
        logger.info(f'     - Similar matches: {len(similar_matched_tags)}')
        logger.info(f'     - Total unique tags: {len(tags)}')
        logger.info(f'  🔧 Components:')
        logger.info(f'     - JB identifiers: {len(jb_identifiers)}')
        logger.info(f'     - MC identifiers: {len(mc_identifiers)}')
        logger.info(f'     - SPARE identifiers: {len(spare_identifiers)}')
        logger.info(f'  📋 Numbering:')
        logger.info(f'     - Tags numbered: {len([k for k in tag_to_number.keys() if not k.startswith("SPARE")])}')
        logger.info(f'     - SPAREs numbered: {len([k for k in tag_to_number.keys() if k.startswith("SPARE")])}')
        logger.info(f'='*60)
        
        # Update global sets
        if hasattr(self, 'all_tags'):
            self.all_tags.update(tags)
        if hasattr(self, 'all_jbs'):
            self.all_jbs.update(jb_identifiers)
        if hasattr(self, 'all_mcs'):
            self.all_mcs.update(mc_identifiers)
        if hasattr(self, 'all_spares'):
            self.all_spares = spare_identifiers

        # ── POSITION-BASED JB/MC RECOVERY (digital extraction) ──────────
        # If no JB was found on this page, look at the positions where JB
        # was found on the PREVIOUS page. In table-mode PDFs, each page
        # has the same layout — the JB/MC identifier appears in the same
        # approximate (x, y) location on every page. If OCR missed it this
        # time, we can recover it by checking what text IS at those positions.
        #
        # We require that the text at that position is similar (Levenshtein
        # distance ≤ 2) to the previous JB text — this prevents picking up
        # a completely different tag that happens to be at the same position.
        #
        # Same logic applies for MC identifiers.
        try:
            import Levenshtein as _lev_recovery
        except ImportError:
            _lev_recovery = None

        if not jb_identifiers and hasattr(self, '_last_jb_positions') and self._last_jb_positions:
            prev_jb_text = getattr(self, '_table_last_valid_jb', None)
            if prev_jb_text:
                prev_pos = self._last_jb_positions[0]
                prev_x = prev_pos.get('x', 0)
                prev_y = prev_pos.get('y', 0)
                for i, word in enumerate(ocr_data['text']):
                    word_x = ocr_data['left'][i]
                    word_y = ocr_data['top'][i]
                    # Within 50px horizontally and 30px vertically
                    if abs(word_x - prev_x) < 50 and abs(word_y - prev_y) < 30:
                        word_text = str(word).strip().upper()
                        if word_text and len(word_text) >= 4 and any(c.isdigit() for c in word_text):
                            # Check similarity to previous JB text
                            # Allow up to 2 character differences (OCR drift)
                            if _lev_recovery:
                                sim_dist = _lev_recovery.distance(word_text, prev_jb_text.upper())
                            else:
                                # Fallback: simple char-diff count
                                sim_dist = sum(1 for a, b in zip(word_text, prev_jb_text.upper()) if a != b)
                                sim_dist += abs(len(word_text) - len(prev_jb_text))
                            if sim_dist <= 2:
                                jb_identifiers.add(word_text)
                                jb_positions.append({
                                    'jb': word_text,
                                    'x': word_x,
                                    'y': word_y,
                                    'width': ocr_data['width'][i],
                                    'height': ocr_data['height'][i],
                                    'coord_source': coord_source,
                                    'dpi_factor': dpi_factor
                                })
                                tag_match_info[word_text] = {
                                    'match_type': 'jb',
                                    'score': 1.0,
                                    'ocr_text': word_text,
                                    'bbox': {
                                        'x': word_x, 'y': word_y,
                                        'width': ocr_data['width'][i],
                                        'height': ocr_data['height'][i]
                                    },
                                    'coord_source': coord_source,
                                    'dpi_factor': dpi_factor
                                }
                                logger.info(
                                    f"POSITION-BASED JB RECOVERY: found '{word_text}' at "
                                    f"({word_x},{word_y}) near previous JB position ({prev_x},{prev_y}), "
                                    f"levenshtein_dist={sim_dist} to '{prev_jb_text}'"
                                )
                                break
                            else:
                                logger.debug(
                                    f"POSITION-BASED JB RECOVERY: '{word_text}' at ({word_x},{word_y}) "
                                    f"is too different from '{prev_jb_text}' (dist={sim_dist}), skipping"
                                )

        # Same position-based recovery for MC
        if not mc_identifiers and hasattr(self, '_last_mc_positions') and self._last_mc_positions:
            prev_mc_text = getattr(self, '_table_last_valid_mc', None)
            if prev_mc_text:
                prev_pos = self._last_mc_positions[0]
                prev_x = prev_pos.get('x', 0)
                prev_y = prev_pos.get('y', 0)
                for i, word in enumerate(ocr_data['text']):
                    word_x = ocr_data['left'][i]
                    word_y = ocr_data['top'][i]
                    if abs(word_x - prev_x) < 50 and abs(word_y - prev_y) < 30:
                        word_text = str(word).strip().upper()
                        if word_text and len(word_text) >= 4 and any(c.isdigit() for c in word_text):
                            if _lev_recovery:
                                sim_dist = _lev_recovery.distance(word_text, prev_mc_text.upper())
                            else:
                                sim_dist = sum(1 for a, b in zip(word_text, prev_mc_text.upper()) if a != b)
                                sim_dist += abs(len(word_text) - len(prev_mc_text))
                            if sim_dist <= 2:
                                mc_token = self._normalize_code_token(word_text)
                                mc_identifiers.add(mc_token)
                                mc_positions.append({
                                    'mc': mc_token,
                                    'x': word_x, 'y': word_y,
                                    'width': ocr_data['width'][i],
                                    'height': ocr_data['height'][i]
                                })
                                mc_indices.append(i)
                                tag_match_info[mc_token] = {
                                    'match_type': 'mc',
                                    'score': 1.0,
                                    'ocr_text': word_text,
                                    'bbox': _bbox(i),
                                    'coord_source': coord_source,
                                    'dpi_factor': dpi_factor
                                }
                                logger.info(
                                    f"POSITION-BASED MC RECOVERY: found '{mc_token}' at "
                                    f"({word_x},{word_y}) near previous MC position ({prev_x},{prev_y}), "
                                    f"levenshtein_dist={sim_dist} to '{prev_mc_text}'"
                                )
                                break
                            else:
                                logger.debug(
                                    f"POSITION-BASED MC RECOVERY: '{word_text}' at ({word_x},{word_y}) "
                                    f"is too different from '{prev_mc_text}' (dist={sim_dist}), skipping"
                                )
        # ────────────────────────────────────────────────────────────────────

        # Store JB/MC positions for next page's position-based recovery
        if jb_positions:
            self._last_jb_positions = jb_positions
        if mc_positions:
            self._last_mc_positions = mc_positions

        # ── TABLE-MODE cable fallback ─────────────────────────────────────
        # Phase 4 above relies on MC token positions to locate cables. In
        # table-mode PDFs with CID-broken fonts, MC tokens are often missed
        # by the main OCR pass (they're recovered later by the header
        # enhancer). This fallback scans ALL OCR tokens for cable patterns
        # directly, ensuring cables like "NC-0-1-2-C-3-BL" are captured.
        # IMPORTANT: We also store the bbox in tag_match_info so that
        # draw_bounding_boxes can draw a tight box around the actual text.
        if pdf_type == 'table':
            _mc_prefix_fb = self._get_mc_prefix()
            cable_re_fb = re.compile(
                r'\b' + re.escape(_mc_prefix_fb) + r'-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}\b',
                re.IGNORECASE
            )
            for idx, word in enumerate(ocr_data['text']):
                text = str(word).strip()
                if not text:
                    continue
                m = cable_re_fb.search(text)
                if m:
                    cable_match = m.group(0).upper()
                    if cable_match not in cable_descriptions:
                        cable_descriptions.append(cable_match)
                        raw_cable_descriptions.append(cable_match)
                        logger.info(f"TABLE cable fallback: found cable '{cable_match}'")
                    
                    # Store bbox in tag_match_info (even if already in cable_descriptions,
                    # so draw_bounding_boxes has the exact position)
                    # Use the OCR token's bounding box — this is the EXACT pixel
                    # position of the text, so the box will be tight around it.
                    cable_x = int(ocr_data['left'][idx]) if idx < len(ocr_data['left']) else 0
                    cable_y = int(ocr_data['top'][idx]) if idx < len(ocr_data['top']) else 0
                    cable_w = int(ocr_data['width'][idx]) if idx < len(ocr_data['width']) else 0
                    cable_h = int(ocr_data['height'][idx]) if idx < len(ocr_data['height']) else 0
                    
                    # Only store if we have valid coordinates
                    if cable_w > 0 and cable_h > 0:
                        cable_key = f"CABLE::{cable_match}::{idx}"
                        if cable_key not in tag_match_info:
                            tag_match_info[cable_key] = {
                                'match_type': 'cable',
                                'score': 1.0,
                                'ocr_text': text,
                                'display_text': cable_match,
                                'bbox': {
                                    'x': cable_x,
                                    'y': cable_y,
                                    'width': cable_w,
                                    'height': cable_h,
                                    'coord_source': coord_source,
                                    'dpi_factor': dpi_factor
                                },
                                'coord_source': coord_source,
                                'dpi_factor': dpi_factor
                            }
                            logger.debug(
                                f"TABLE cable fallback: stored bbox for '{cable_match}' at ({cable_x},{cable_y}) {cable_w}x{cable_h}"
                            )
        # ────────────────────────────────────────────────────────────────────

        return (
            tags,
            jb_identifiers,
            mc_identifiers,
            cable_descriptions,
            spare_identifiers,
            tag_to_number,
            raw_cable_descriptions,
            tag_match_info,
            all_ocr_tags,
        )

    def extract_from_text_page(self, page, dpi_factor: float = 300 / 72) -> 'Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict], Set[str]]':
        words = page.get_text("words")
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
        num_blocks = len(blocks)
        num_lines = sum(len(block.get("lines", [])) for block in blocks if isinstance(block, dict))
        num_words = len(words) if words else 0
        tables_detected = sum(
            1 for block in blocks
            if isinstance(block, dict) and 'table' in str(block.get('text', '')).lower()
        )
        sample_texts = [str(w[4]).strip() for w in words if str(w[4]).strip()][:20] if words else []
        logger.info(
            "extract_from_text_page: digital stats blocks=%d, words=%d, lines=%d, tables=%d, dpi_factor=%.4f, sample_texts=%s",
            num_blocks, num_words, num_lines, tables_detected, dpi_factor, sample_texts
        )
        if not words:
            logger.info("extract_from_text_page: no words extracted from digital PDF page")
            return set(), set(), set(), [], [], {}, [], {}, set()
        pdf_type = getattr(self, '_current_pdf_type', 'diagrams')
        ocr_data = {
            "text": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
            "conf": []
        }
        for w in words:
            x0, y0, x1, y1, text, *_ = w
            text = str(text).strip()
            if not text:
                continue
            ocr_data["text"].append(text)
            ocr_data["left"].append(int(x0))
            ocr_data["top"].append(int(y0))
            ocr_data["width"].append(int(x1 - x0))
            ocr_data["height"].append(int(y1 - y0))
            ocr_data["conf"].append(95)

        # ✅ ذخیره dpi_factor برای استفاده در draw_bounding_boxes
        self._digital_dpi_factor = dpi_factor

        return self._extract_from_ocr_data(
            ocr_data,
            pdf_type,
            coord_source='digital',   # ✅ مهم: می‌گه این مختصات واحدشون "پوینت PDF" است نه پیکسل
            dpi_factor=dpi_factor      # ✅ مهم: ضریب تبدیل پوینت به پیکسل
    )
    def get_similarity_reports(self) -> 'List[Dict[str, Any]]':
        """
        دریافت گزارشات کامل شباهت بین تگ‌های شناسایی شده و تگ‌های مرجع
        
        Returns:
            لیستی از دیکشنری‌های حاوی گزارشات شباهت
        """
        return self.similarity_reports
    
    def get_top_similar_tags(self, n: int = 10) -> 'List[Dict[str, Any]]':
        """
        دریافت n تگ با بالاترین شباهت
        
        Args:
            n: تعداد تگ‌های برتر
            
        Returns:
            لیستی از دیکشنری‌های حاوی اطلاعات تگ‌های برتر
        """
        sorted_reports = sorted(self.similarity_reports, key=lambda x: x['similarity_score'], reverse=True)
        return sorted_reports[:n]
    
    def preprocess_image(self, image: np.ndarray, pdf_type: str = "diagrams") -> np.ndarray:
        """Preprocess the image to improve OCR accuracy for tags."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
 
        if pdf_type == 'table':
            # ── TABLE preprocessing branch (REVISED for CID-broken PDFs) ─────
            # Rationale: many table PDFs (e.g. PDFium-generated ones) ship with
            # Identity-H encoded fonts that have NO ToUnicode CMap. Direct text
            # extraction returns (cid:XX) garbage, so we MUST rely on OCR.
            # The previous CLAHE+median+Otsu combo was destroying small cell text.
            # Empirically, the simplest pipeline works best:
            #   1. No upscaling (image is already at 300 DPI from process_pdf)
            #   2. Light Gaussian blur (removes JPEG artefacts without smearing)
            #   3. Otsu global threshold (clean black-on-white table cells)
            logger.info("preprocess_image: using TABLE preprocessing path (revised)")

            # Light Gaussian blur — removes PDFium JPEG artefacts without
            # smearing the small text inside table cells
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Otsu global threshold — works well for the globally-consistent
            # black-on-white contrast of printed tables
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # No morphological operations — avoids fusing adjacent cell lines
            return gray
            # ─────────────────────────────────────────────────────────────────
 
        # ── DIAGRAM preprocessing branch (original, byte-for-byte) ───────────
        # Diagram path: UNCHANGED ✓
        scale_factor = 2
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 31, 2)
        
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        kernel_dilate = np.ones((1, 1), np.uint8)
        gray = cv2.dilate(gray, kernel_dilate, iterations=1)
        
        return gray
            
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه امتیاز شباهت بین دو رشته متنی
        
        Args:
            text1: متن اول
            text2: متن دوم
            
        Returns:
            امتیاز شباهت بین 0 تا 1
        """
        try:
            # تبدیل به رشته و حذف فضاهای خالی
            text1 = str(text1).strip().upper()
            text2 = str(text2).strip().upper()
            
            # اگر هر دو متن خالی هستند، شباهت 1 است
            if not text1 and not text2:
                return 1.0
                
            # اگر یکی از متن‌ها خالی است، شباهت 0 است
            if not text1 or not text2:
                return 0.0
                
            # اگر متن‌ها یکسان هستند، شباهت 1 است
            if text1 == text2:
                return 1.0
            
            # محاسبه فاصله لونشتاین
            distance = Levenshtein.distance(text1, text2)
            max_len = max(len(text1), len(text2))
            
            # تبدیل فاصله به امتیاز شباهت
            similarity = 1.0 - (distance / max_len)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def match_tags(self, extracted_tags, io_list_tags, similarity_threshold=0.8):
        """
        تطبیق تگ‌های استخراج شده با لیست IO
        
        این متد برای حفظ سازگاری با کد قبلی حفظ شده است، اما از متد جدید match_tags_with_io_list استفاده می‌کند.
        
        Args:
            extracted_tags: لیست تگ‌های استخراج شده
            io_list_tags: لیست تگ‌های IO
            similarity_threshold: آستانه شباهت
            
        Returns:
            دیکشنری تگ‌های تطبیق داده شده
        """
        # استفاده از متد جدید
        matched_tags_dict, unmatched_io_tags, unknown_signals = self.match_tags_with_io_list(
            extracted_tags, io_list_tags, similarity_threshold
        )
        
        # برای حفظ سازگاری با کد قبلی، فقط دیکشنری تطبیق را برگردان
        return matched_tags_dict

    def match_tags_with_io_list(self, pdf_tags, io_tags, min_similarity=0.75):
        """
        تطبیق تگ‌های استخراج شده از PDF با تگ‌های IO List با استفاده از روش بهبود یافته
        
        Args:
            pdf_tags: لیست تگ‌های استخراج شده از PDF
            io_tags: لیست تگ‌های IO
            min_similarity: حداقل میزان شباهت مورد نیاز (افزایش یافته به 0.75)
        
        Returns:
            دیکشنری نگاشت تگ‌های PDF به تگ‌های IO
        """
        logger.info(f"CANDIDATE_FUNCTION_CALLED: Matching {len(pdf_tags)} PDF tags with {len(io_tags)} IO tags using improved approach")

        try:
            matched_tags = {}
            
            logger.info(f"Matching {len(pdf_tags)} PDF tags with {len(io_tags)} IO tags")
            
            # تبدیل io_tags به set برای جستجوی سریع‌تر
            io_tags_set = set(str(tag).strip().upper() for tag in io_tags if tag and not pd.isna(tag))
            
            # جستجوی تگ‌های دقیق - مرحله اول (اولویت بالا)
            exact_matches = []
            pdf_tags_exact_not_found = []
            
            logger.info("Step 1: Searching for exact tag matches...")
            for pdf_tag in pdf_tags:
                pdf_tag_upper = str(pdf_tag).strip().upper()
                if pdf_tag_upper in io_tags_set:
                    matched_tags[pdf_tag] = pdf_tag_upper
                    exact_matches.append(pdf_tag)
                    logger.info(f"Found exact match: {pdf_tag}")
                else:
                    pdf_tags_exact_not_found.append(pdf_tag)
            
            logger.info(f"Found {len(exact_matches)} exact matches")
            
            # استفاده از روش کاندید تگ برای تگ‌های باقی‌مانده - مرحله دوم
            logger.info(f"Step 2: Finding similar tags for {len(pdf_tags_exact_not_found)} unmatched tags...")
            
            for pdf_tag in pdf_tags_exact_not_found:
                # پیدا کردن کاندیداهای مشابه
                candidates = self.find_candidate_tags(pdf_tag, io_tags, min_similarity)
                
                # اگر هیچ کاندیدایی پیدا نشد، ادامه بده
                if not candidates:
                    logger.debug(f"No candidates found for tag: {pdf_tag}")
                    continue
                
                # اعتبارسنجی کاندیداها
                valid_candidates = self.validate_tag_candidates(pdf_tag, candidates)
                
                # انتخاب بهترین کاندیدا
                if valid_candidates:
                    # اگر چند کاندیدای معتبر وجود دارد، بهترین را انتخاب کن
                    best_match, similarity = valid_candidates[0]
                    
                    # بررسی نهایی: آیا این تگ قبلاً به تگ دیگری نسبت داده شده؟
                    if best_match.upper() in [v.upper() for v in matched_tags.values()]:
                        logger.warning(f"Tag {best_match} already matched to another PDF tag, skipping {pdf_tag}")
                        continue
                    
                    matched_tags[pdf_tag] = best_match
                    logger.info(f"Found similar tag: {pdf_tag} -> {best_match} (similarity: {similarity:.2f})")
                else:
                    logger.debug(f"No valid candidates after validation for tag: {pdf_tag}")
            
            # آمار تطبیق
            matched_count = len(matched_tags)
            exact_count = len(exact_matches)
            similar_count = matched_count - exact_count
            unmatched_count = len(pdf_tags) - matched_count
            
            logger.info(f"Tag matching stats - Total: {len(pdf_tags)}, Matched: {matched_count} " +
                    f"(Exact: {exact_count}, Similar: {similar_count}), Unmatched: {unmatched_count}")
            
            if unmatched_count > 0:
                unmatched_tags = [tag for tag in pdf_tags if tag not in matched_tags]
                logger.warning(f"Unmatched tags ({unmatched_count}): {unmatched_tags[:10]}...")  # نمایش 10 مورد اول
            
            # بررسی تکراری
            matched_io_tags = list(matched_tags.values())
            unique_io_tags = set(matched_io_tags)
            if len(matched_io_tags) != len(unique_io_tags):
                duplicates = [tag for tag in unique_io_tags if matched_io_tags.count(tag) > 1]
                logger.error(f"WARNING: Found duplicate IO tag matches: {duplicates}")
                # حذف تکراری‌ها - فقط اولین تطبیق را نگه دار
                seen = set()
                cleaned_matched_tags = {}
                for pdf_tag, io_tag in matched_tags.items():
                    if io_tag.upper() not in seen:
                        cleaned_matched_tags[pdf_tag] = io_tag
                        seen.add(io_tag.upper())
                    else:
                        logger.warning(f"Removing duplicate match: {pdf_tag} -> {io_tag}")
                matched_tags = cleaned_matched_tags
            
            return matched_tags
            
        except Exception as e:
            logger.error(f"Error matching tags with IO list: {e}")
            return {}

    
                
    def create_tag_jb_mapping(self, pdf_results: 'Dict[int, Tuple[Any, ...]]') -> 'Dict[str, str]':
        """
        Create a mapping from tags to JB identifiers.
        
        Args:
            pdf_results: Dictionary mapping page numbers to Tuples of (tags, jb_identifiers)
            
        Returns:
            Dictionary mapping tags to their associated JB identifiers
        """
        try:
            tag_to_jb = {}
            
            # بررسی ساختار pdf_results
            logger.info(f"In create_tag_jb_mapping: pdf_results type: {type(pdf_results)}")
            if pdf_results:
                first_key = next(iter(pdf_results.keys()))
                first_value = pdf_results[first_key]
                logger.info(f"First value type: {type(first_value)}")
                logger.info(f"First value length: {len(first_value)}")
            
            # برای هر صفحه
            for page_num, page_results in pdf_results.items():
                # بررسی نوع page_results
                if isinstance(page_results, tuple):
                    # اگر تاپل باشد، تگ‌ها و JB ها را استخراج کن
                    if len(page_results) >= 2:
                        tags = page_results[0]
                        jbs = page_results[1]
                        
                        # اگر فقط یک JB در صفحه وجود دارد، همه تگ‌های صفحه را به آن نسبت بده
                        if len(jbs) == 1:
                            jb = next(iter(jbs))
                            for tag in tags:
                                if tag not in tag_to_jb:  # فقط اگر تگ قبلاً نگاشت نشده باشد
                                    tag_to_jb[tag] = jb
                        # اگر چندین JB وجود دارد، از الگوریتم مشابهت استفاده کن
                        elif len(jbs) > 1:
                            for tag in tags:
                                # یافتن نزدیک‌ترین JB به تگ بر اساس مشابهت
                                best_jb = None
                                best_similarity = -1
                                
                                for jb in jbs:
                                    similarity = self._calculate_similarity(tag, jb)
                                    if similarity > best_similarity:
                                        best_similarity = similarity
                                        best_jb = jb
                                
                                if best_jb and tag not in tag_to_jb:
                                    tag_to_jb[tag] = best_jb
                elif isinstance(page_results, dict):
                    # اگر دیکشنری باشد، تگ‌ها و JB ها را از کلیدهای مناسب استخراج کن
                    tags = page_results.get('tags', set())
                    jbs = page_results.get('jbs', set())
                     
                    # اگر فقط یک JB در صفحه وجود دارد، همه تگ‌های صفحه را به آن نسبت بده
                    if len(jbs) == 1:
                        jb = next(iter(jbs))
                        for tag in tags:
                            if tag not in tag_to_jb:  # فقط اگر تگ قبلاً نگاشت نشده باشد
                                tag_to_jb[tag] = jb
                    # اگر چندین JB وجود دارد، از الگوریتم مشابهت استفاده کن
                    elif len(jbs) > 1:
                        for tag in tags:
                            # یافتن نزدیک‌ترین JB به تگ بر اساس مشابهت
                            best_jb = None
                            best_similarity = -1
                            
                            for jb in jbs:
                                similarity = self._calculate_similarity(tag, jb)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_jb = jb
                            
                            if best_jb and tag not in tag_to_jb:
                                tag_to_jb[tag] = best_jb
                else:
                    logger.warning(f"Unexpected type for page_results: {type(page_results)}")
            
            # آمار نگاشت
            logger.info(f"Total tags mapped: {len(tag_to_jb)}")
            logger.info(f"Tags mapped in first stage: {sum(1 for jb in tag_to_jb.values() if jb)}")
            logger.info(f"Tags mapped in second stage: {sum(1 for jb in tag_to_jb.values() if not jb)}")
            
            return tag_to_jb
            
        except Exception as e:
            logger.error(f"Error in create_tag_jb_mapping: {e}")
            return {} 
    
# Add this new method to the TagJBExtractor class (place it before process_excel method)

    def detect_columns_and_find_new_tags(self, image, tag_coordinates, column_threshold=50):
        """
        Enhanced column detection and tag finding with improved processing techniques.
        
        Args:
            image: numpy.ndarray - The preprocessed image
            tag_coordinates: List - List of Dictionaries containing tag coordinates
            column_threshold: int - Threshold for column detection
            
        Returns:
            Set - New tags found during column analysis
        """
        height, width = image.shape
        columns = {}
        new_tags = set()

        # 1. Improved column detection
        x_coordinates = [coord['left'] for coord in tag_coordinates]
        if x_coordinates:
            x_coordinates.sort()
            # Dynamic column width based on actual tag positions
            avg_tag_width = sum(coord['width'] for coord in tag_coordinates) / len(tag_coordinates)
            column_threshold = max(int(avg_tag_width * 1.5), column_threshold)

        # 2. Group tags into columns with dynamic thresholds
        for tag_info in tag_coordinates:
            tag = tag_info['tag']
            left = tag_info['left']
            column_found = False
            for col_left in columns:
                if abs(left - col_left) <= column_threshold:
                    columns[col_left].append(tag_info)
                    column_found = True
                    break
            if not column_found:
                columns[left] = [tag_info]

        # 3. Enhanced column processing
        for col_left, col_tags in columns.items():
            # Calculate column boundaries based on actual tag positions
            min_x = min(tag['left'] for tag in col_tags)
            max_x = max(tag['left'] + tag['width'] for tag in col_tags)
            
            # Add padding to ensure we capture full tags
            padding = int(column_threshold * 0.5)
            col_start = max(0, min_x - padding)
            col_end = min(width, max_x + padding)
            
            # Extract and process column region
            column_image = image[:, col_start:col_end]
            
            # Apply additional preprocessing to column image
            column_image = cv2.GaussianBlur(column_image, (3, 3), 0)
            column_image = cv2.adaptiveThreshold(
                column_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
            )

            # Multiple OCR passes with different configurations
            ocr_configs = [
                r'--oem 1 --psm 6 -l eng', 
                r'--oem 1 --psm 11 -l eng',  
                r'--oem 1 --psm 12 -l eng' 
            ]
            
            for config in ocr_configs:
                column_text = pytesseract.image_to_string(
                    column_image, 
                    config=config + r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZsprea0123456-'
                )
                
                # Process extracted text
                for line in column_text.split('\n'):
                    words = line.strip().split()
                    for word in words:
                        # Clean and normalize the word
                        word = re.sub(r'^[-CP\s]+', '', word.upper())
                        
                        # Extended tag pattern matching
                        tag_patterns = [
                            r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b',
                            r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b',
                            r'\b[A-Z]{2,4}[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b'
                        ]
                        
                        for pattern in tag_patterns:
                            if re.search(pattern, word):
                                new_tags.add(word)
                                break

        return new_tags


    def _extract_jb_mc_from_digital_header(self, page, page_num):
        """
        Extract JB and MC identifiers from the DIGITAL text layer of a PDF page.

        Uses a dictionary-based label search: scans the header for any label
        synonym (e.g. "JB", "JB No.", "Junction Box", "Multi Cable", "MC"),
        then reads the text immediately after the label and checks if it
        matches the user's JB/MC input patterns (jb_examples / mc_examples).

        Label dictionaries:
            JB labels: "JB", "JB No.", "JB No:", "Junction Box", "Junction Box No."
            MC labels: "MC", "MC No.", "MC No:", "Multi Cable", "Multi Cable No."

        Returns:
            (set of JB identifiers, set of MC identifiers)
        """
        jb_ids = set()
        mc_ids = set()

        try:
            words = page.get_text("words")
            if not words:
                return jb_ids, mc_ids

            # ── Get user's JB/MC prefix patterns ────────────────────────
            _jb_prefixes = list(getattr(self, 'jb_examples_list', None) or [])
            if not _jb_prefixes and getattr(self, 'jb_examples', None):
                _jb_prefixes = self._parse_multi_patterns(self.jb_examples)
            _mc_prefixes = list(getattr(self, 'mc_examples_list', None) or [])
            if not _mc_prefixes and getattr(self, 'mc_examples', None):
                _mc_prefixes = self._parse_multi_patterns(self.mc_examples)

            # ── Label dictionaries ──────────────────────────────────────
            # Each entry: (label_words_tuple, type)
            # label_words_tuple is a tuple of uppercase words to match
            # sequentially in the header line.
            _LABEL_DICT = [
                # JB labels — ordered by specificity (longest first)
                (('JUNCTION', 'BOX', 'NO'), 'jb'),
                (('JUNCTION', 'BOX'), 'jb'),
                (('JB', 'NO'), 'jb'),
                (('JB',), 'jb'),
                # MC labels — ordered by specificity (longest first)
                (('MULTI', 'CABLE', 'NO'), 'mc'),
                (('MULTI', 'CABLE'), 'mc'),
                (('MC', 'NO'), 'mc'),
                (('MC',), 'mc'),
            ]

            # ── Group words by line ─────────────────────────────────────
            lines = {}
            for w in words:
                key = (w[5], w[6])  # (block_no, line_no)
                if key not in lines:
                    lines[key] = []
                lines[key].append(w)

            for key in lines:
                lines[key].sort(key=lambda w: w[0])

            sorted_lines = sorted(lines.values(), key=lambda line: line[0][1] if line else 0)

            # ── Scan header lines (first 10) ────────────────────────────
            header_line_count = min(10, len(sorted_lines))

            for line_idx in range(header_line_count):
                line_words = sorted_lines[line_idx]
                word_texts = [str(w[4]).strip().upper() for w in line_words]
                line_text = ' '.join(word_texts)

                logger.debug(
                    f"_extract_jb_mc_from_digital_header: line {line_idx}: '{line_text[:120]}'"
                )

                # ── Scan for label matches ──────────────────────────────
                for i, wt in enumerate(word_texts):
                    for label_words, label_type in _LABEL_DICT:
                        # Check if words starting at position i match the label
                        if i + len(label_words) > len(word_texts):
                            continue

                        match = True
                        for j, lw in enumerate(label_words):
                            wt_j = word_texts[i + j]
                            # Allow "NO", "NO.", "NO:" to match each other
                            if lw == 'NO':
                                if wt_j not in ('NO', 'NO.', 'NO:', 'NUMBER', 'NO,', 'NO;'):
                                    match = False
                                    break
                            else:
                                if wt_j != lw:
                                    match = False
                                    break

                        if not match:
                            continue

                        # ── Label found! Now read the next words after it ──
                        # Skip optional separators: ":", ".", "No.", "No:", etc.
                        search_start = i + len(label_words)
                        while search_start < len(word_texts):
                            wt_skip = word_texts[search_start]
                            # Skip pure punctuation
                            if wt_skip in (':', '.', ':.', '.:', ',', ';'):
                                search_start += 1
                                continue
                            # Skip "NO" + any punctuation combination (NO., NO:, NO.:, etc.)
                            if re.match(r'^NO[:.;,]+$', wt_skip):
                                search_start += 1
                                continue
                            # Skip "NUMBER"
                            if wt_skip == 'NUMBER':
                                search_start += 1
                                continue
                            break

                        # Read up to 3 words after the label as candidate identifier
                        # (identifier may be split across words, e.g. "JSF" + "-" + "309S")
                        candidate_parts = []
                        for k in range(search_start, min(search_start + 4, len(word_texts))):
                            wt_k = word_texts[k]
                            # Stop if we hit another label
                            if wt_k in ('JB', 'MC', 'MULTI', 'JUNCTION', 'TAG', 'PAGE', 'SHEET'):
                                break
                            candidate_parts.append(wt_k)

                        if not candidate_parts:
                            continue

                        # Join parts and clean up
                        candidate = ''.join(candidate_parts)
                        candidate = re.sub(r'[:.;,\s]+', '', candidate)
                        candidate = candidate.upper()

                        if len(candidate) < 4 or not any(c.isdigit() for c in candidate):
                            continue

                        # ── Validate against user's JB/MC patterns ─────────
                        if label_type == 'jb':
                            # Check if candidate starts with any JB prefix
                            # OR matches general JB pattern (letters-digits)
                            starts_with_jb = bool(_jb_prefixes) and any(
                                candidate.startswith(p) for p in _jb_prefixes
                            )
                            looks_like_jb = bool(re.match(r'^[A-Z]{2,5}-?\d{2,5}[A-Z]?$', candidate))
                            if starts_with_jb or looks_like_jb:
                                jb_ids.add(candidate)
                                logger.info(
                                    f"_extract_jb_mc_from_digital_header: page {page_num + 1} — "
                                    f"found JB '{candidate}' after label {' '.join(label_words)} "
                                    f"in line: '{line_text[:80]}'"
                                )

                        elif label_type == 'mc':
                            # Check if candidate starts with any MC prefix
                            starts_with_mc = bool(_mc_prefixes) and any(
                                candidate.startswith(p) for p in _mc_prefixes
                            )
                            # Also accept NC- prefixed identifiers
                            starts_with_nc = candidate.startswith('NC')
                            if starts_with_mc or starts_with_nc:
                                mc_ids.add(candidate)
                                logger.info(
                                    f"_extract_jb_mc_from_digital_header: page {page_num + 1} — "
                                    f"found MC '{candidate}' after label {' '.join(label_words)} "
                                    f"in line: '{line_text[:80]}'"
                                )

            # ── Derive MC from JB (or vice versa) ────────────────────────
            if jb_ids and not mc_ids:
                for jb in jb_ids:
                    derived_mc = f"NC-{jb}"
                    mc_ids.add(derived_mc)
                    logger.info(
                        f"_extract_jb_mc_from_digital_header: page {page_num + 1} — "
                        f"derived MC '{derived_mc}' from JB '{jb}'"
                    )

            if mc_ids and not jb_ids:
                for mc in mc_ids:
                    if mc.upper().startswith('NC-'):
                        derived_jb = mc[3:]
                        if len(derived_jb) >= 4:
                            jb_ids.add(derived_jb)
                            logger.info(
                                f"_extract_jb_mc_from_digital_header: page {page_num + 1} — "
                                f"derived JB '{derived_jb}' from MC '{mc}'"
                            )

        except Exception as exc:
            logger.warning(f"_extract_jb_mc_from_digital_header: failed: {exc}")
        return jb_ids, mc_ids


    def _extract_header_references_only(self, page, page_num):
        """
        Extract JB/MC identifiers from the table header ONLY (no body).

        This is called BEFORE _extract_from_image_table_multipass so that
        self._table_known_jbs and self._table_known_mcs are populated
        before _apply_table_ocr_corrections runs. This way, character
        corrections can use the header-extracted JB/MC as ground truth.

        Strategy:
        1. Use pdfplumber to find the header row (between 1st and 4th H lines)
        2. Render the header at 900 DPI
        3. OCR with PSM 6 and PSM 11 (no whitelist)
        4. Extract JB/MC identifiers using semantic label-based parsing
        5. Store in self._table_known_jbs / self._table_known_mcs

        Args:
            page: fitz.Page object
            page_num: 0-indexed page number (for logging)
        """
        try:
            import pdfplumber
        except ImportError:
            logger.warning("_extract_header_references_only: pdfplumber not available, skipping")
            return

        try:
            doc = page.parent
            if doc is None:
                return

            page_index = page.number if hasattr(page, 'number') else 0
            pdf_path = None
            try:
                pdf_path = doc.name
            except Exception:
                pdf_path = None

            if not pdf_path or not os.path.exists(pdf_path):
                return

            with pdfplumber.open(pdf_path) as pp_doc:
                if page_index >= len(pp_doc.pages):
                    return
                pp_page = pp_doc.pages[page_index]
                rects = pp_page.rects or []

            h_lines = [r for r in rects if r['width'] > 100 and r['height'] < 3]
            v_lines = [r for r in rects if r['height'] > 100 and r['width'] < 3]
            if not h_lines or not v_lines:
                return

            h_lines.sort(key=lambda r: r['top'])
            v_lines.sort(key=lambda r: r['x0'])

            table_left = min(r['x0'] for r in v_lines)
            table_right = max(r['x0'] for r in v_lines)
            header_top = h_lines[0]['top']
            header_bottom = h_lines[min(3, len(h_lines) - 1)]['top']

            pad = 6.0
            header_top -= pad
            header_bottom += pad
            table_left -= pad
            table_right += pad

            page_w = page.rect.width
            page_h = page.rect.height
            header_top = max(0.0, header_top)
            header_bottom = min(page_h, header_bottom)
            table_left = max(0.0, table_left)
            table_right = min(page_w, table_right)

            clip_rect = fitz.Rect(table_left, header_top, table_right, header_bottom)
            header_dpi_matrix = fitz.Matrix(900 / 72, 900 / 72)

            try:
                pix = page.get_pixmap(matrix=header_dpi_matrix, clip=clip_rect)
            except Exception:
                return

            header_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                header_img = cv2.cvtColor(header_img, cv2.COLOR_RGB2BGR)
            else:
                header_img = cv2.cvtColor(header_img, cv2.COLOR_RGBA2BGR)

            gray_h = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
            gray_h = cv2.GaussianBlur(gray_h, (3, 3), 0)
            _, binary_h = cv2.threshold(gray_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cfg_psm6 = r'--oem 1 --psm 6 -l eng'
            cfg_psm11 = r'--oem 1 --psm 11 -l eng'

            text_psm6 = pytesseract.image_to_string(binary_h, config=cfg_psm6)
            text_psm11 = pytesseract.image_to_string(binary_h, config=cfg_psm11)
            combined_header_text = (text_psm6 or '') + '\n' + (text_psm11 or '')

            # ── Normalize OCR errors in header text ─────────────────────
            def _normalize_ocr_id(s):
                s = str(s).upper().strip()
                s = s.strip(' \t,;:()[].')
                # Replace "/" between digits with "7"
                s = re.sub(r'(\d)/(\d)', r'\g<1>7\g<2>', s)
                s = re.sub(r'\s+', '', s)
                return s

            # ── SEMANTIC JB extraction ──────────────────────────────────
            new_jbs = set()
            jb_label_pat = re.compile(
                r'JB\s*No\.?\s*[:.]?\s*(.+?)(?:MC\s*No\.?\s*[:.]?|Multi\s*Cable\s*No\.?\s*[:.]?|\]|$)',
                re.IGNORECASE | re.DOTALL
            )
            for m in jb_label_pat.finditer(combined_header_text):
                raw = m.group(1).strip()
                token_pat = re.compile(r'[A-Z]{2,5}\s*-?\s*[\d/]{2,5}[A-Z0-9]{0,3}', re.IGNORECASE)
                for tm in token_pat.finditer(raw):
                    candidate = _normalize_ocr_id(tm.group(0))
                    if len(candidate) >= 4:
                        new_jbs.add(candidate)

            # ── SEMANTIC MC extraction ──────────────────────────────────
            new_mcs = set()
            mc_label_pat = re.compile(
                r'(?:MC\s*No\.?\s*[:.]?|Multi\s*Cable\s*No\.?\s*[:.]?)\s*(.+?)(?:\(|$)',
                re.IGNORECASE | re.DOTALL
            )
            for m in mc_label_pat.finditer(combined_header_text):
                raw = m.group(1).strip()
                token_pat = re.compile(
                    r'NC-?[A-Z]{0,5}-?[\d/]{2,5}(?:[A-Z0-9-]{0,15})?',
                    re.IGNORECASE
                )
                for tm in token_pat.finditer(raw):
                    candidate = _normalize_ocr_id(tm.group(0))
                    if len(candidate) >= 4:
                        new_mcs.add(candidate)

            # ── FALLBACK: permissive regex on whole header text ─────────
            if not new_jbs:
                jb_pat = re.compile(r'J?[A-Z]{2,5}\s*-?[\d/]{2,5}[A-Z0-9]{0,3}', re.IGNORECASE)
                for m in jb_pat.finditer(combined_header_text):
                    candidate = _normalize_ocr_id(m.group(0))
                    if len(candidate) >= 4:
                        new_jbs.add(candidate)

            if not new_mcs:
                mc_pat = re.compile(r'NC-?[A-Z]{0,5}\s*-?[\d/]{2,5}[A-Z0-9-]{0,15}', re.IGNORECASE)
                for m in mc_pat.finditer(combined_header_text):
                    candidate = _normalize_ocr_id(m.group(0))
                    if len(candidate) >= 4:
                        new_mcs.add(candidate)

            # Filter noise
            def _is_plausible_id(s):
                s = str(s).upper().strip()
                if len(s) < 4:
                    return False
                if not any(c.isalpha() for c in s):
                    return False
                return True

            new_jbs = {j.upper() for j in new_jbs if _is_plausible_id(j)}
            new_mcs = {m.upper() for m in new_mcs if _is_plausible_id(m)}

            label_blacklist = {'JB', 'JBNO', 'MC', 'MCNO', 'NO'}
            new_jbs = {j for j in new_jbs if j not in label_blacklist}
            new_mcs = {m for m in new_mcs if m not in label_blacklist}

            # ── Store as references ─────────────────────────────────────
            # Filter to only valid JB/MC identifiers (must match JB/MC pattern)
            jb_re = re.compile(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', re.IGNORECASE)
            mc_re = re.compile(r'^NC-?[A-Z]{2,5}-?\d{2,5}[A-Z]?$', re.IGNORECASE)

            valid_jbs = {j for j in new_jbs if jb_re.match(j)}
            valid_mcs = {m for m in new_mcs if mc_re.match(m)}

            # If no valid JBs found on this page, use JB preservation
            # (reuse JB from previous page)
            if not valid_jbs:
                last_jb = getattr(self, '_table_last_valid_jb', None)
                if last_jb:
                    valid_jbs = {last_jb}
                    logger.info(
                        "_extract_header_references_only: page %d — no valid JB in header, reusing '%s' from previous page",
                        page_num + 1, last_jb
                    )
            else:
                # Store the best JB for subsequent pages
                best_jb = max(valid_jbs, key=len)
                self._table_last_valid_jb = best_jb
                logger.info(
                    "_extract_header_references_only: page %d — stored JB '%s' for subsequent pages",
                    page_num + 1, best_jb
                )

            # If no valid MCs found on this page, use MC preservation
            if not valid_mcs:
                last_mc = getattr(self, '_table_last_valid_mc', None)
                if last_mc:
                    valid_mcs = {last_mc}
                    logger.info(
                        "_extract_header_references_only: page %d — no valid MC in header, reusing '%s' from previous page",
                        page_num + 1, last_mc
                    )
            else:
                # Store the best MC for subsequent pages
                best_mc = max(valid_mcs, key=len)
                self._table_last_valid_mc = best_mc
                logger.info(
                    "_extract_header_references_only: page %d — stored MC '%s' for subsequent pages",
                    page_num + 1, best_mc
                )

            # Also try to derive MC from JB and vice versa
            # (if we have JB="JSF-576S", derive MC="NC-JSF-576S")
            if valid_jbs and not valid_mcs:
                for jb in valid_jbs:
                    derived_mc = f"NC-{jb}"
                    if mc_re.match(derived_mc):
                        valid_mcs.add(derived_mc)
                        logger.info(
                            "_extract_header_references_only: page %d — derived MC '%s' from JB '%s'",
                            page_num + 1, derived_mc, jb
                        )
                        break
            elif valid_mcs and not valid_jbs:
                for mc in valid_mcs:
                    # Strip "NC-" prefix to get JB
                    if mc.upper().startswith('NC-'):
                        derived_jb = mc[3:]
                        if jb_re.match(derived_jb):
                            valid_jbs.add(derived_jb)
                            logger.info(
                                "_extract_header_references_only: page %d — derived JB '%s' from MC '%s'",
                                page_num + 1, derived_jb, mc
                            )
                            break

            # ── Cross-validate JB and MC references ─────────────────────
            # If JB="JSF-5776S" (extra digit) and MC="NC-JSF-576" (missing S),
            # we can infer the correct form by comparing them.
            # The JB suffix after "JSF-" should match the MC suffix after "NC-JSF-".
            try:
                import Levenshtein as _lev_cross
                if valid_jbs and valid_mcs:
                    corrected_jbs = set()
                    corrected_mcs = set()
                    for jb in valid_jbs:
                        # Find the best matching MC
                        best_mc = None
                        best_mc_score = 0.0
                        for mc in valid_mcs:
                            # Compare JB suffix with MC suffix (strip NC- prefix)
                            mc_suffix = mc.upper()
                            if mc_suffix.startswith('NC-'):
                                mc_suffix = mc_suffix[3:]
                            score = _lev_cross.ratio(jb.upper(), mc_suffix)
                            if score > best_mc_score:
                                best_mc_score = score
                                best_mc = mc_suffix

                        # If JB and MC suffix are similar but not identical,
                        # try to find the correct form by character correction
                        if best_mc and best_mc_score >= 0.80 and best_mc != jb.upper():
                            # Try to correct JB using MC suffix as reference
                            corrected_jb, jb_score = self._correct_token_against_reference(
                                jb, {best_mc}, threshold=0.92
                            )
                            if jb_score >= 0.92 and corrected_jb != jb:
                                logger.info(
                                    "_extract_header_references_only: cross-validated JB '%s' → '%s' (using MC suffix '%s', sim=%.2f)",
                                    jb, corrected_jb, best_mc, jb_score
                                )
                                corrected_jbs.add(corrected_jb)
                            else:
                                corrected_jbs.add(jb)
                        else:
                            corrected_jbs.add(jb)

                    for mc in valid_mcs:
                        mc_suffix = mc.upper()
                        if mc_suffix.startswith('NC-'):
                            mc_suffix = mc_suffix[3:]
                        # Find the best matching JB
                        best_jb = None
                        best_jb_score = 0.0
                        for jb in valid_jbs:
                            score = _lev_cross.ratio(mc_suffix, jb.upper())
                            if score > best_jb_score:
                                best_jb_score = score
                                best_jb = jb

                        if best_jb and best_jb_score >= 0.80 and best_jb.upper() != mc_suffix:
                            corrected_mc_suffix, mc_score = self._correct_token_against_reference(
                                mc_suffix, {best_jb.upper()}, threshold=0.92
                            )
                            if mc_score >= 0.92 and corrected_mc_suffix != mc_suffix:
                                corrected_mc = f"NC-{corrected_mc_suffix}"
                                logger.info(
                                    "_extract_header_references_only: cross-validated MC '%s' → '%s' (using JB '%s', sim=%.2f)",
                                    mc, corrected_mc, best_jb, mc_score
                                )
                                corrected_mcs.add(corrected_mc)
                            else:
                                corrected_mcs.add(mc)
                        else:
                            corrected_mcs.add(mc)

                    valid_jbs = corrected_jbs if corrected_jbs else valid_jbs
                    valid_mcs = corrected_mcs if corrected_mcs else valid_mcs
            except ImportError:
                pass

            if valid_jbs:
                if not hasattr(self, '_table_known_jbs'):
                    self._table_known_jbs = set()
                self._table_known_jbs.update(valid_jbs)
                logger.info(
                    "_extract_header_references_only: page %d — extracted %d JB references: %s",
                    page_num + 1, len(valid_jbs), sorted(valid_jbs)
                )
            if valid_mcs:
                if not hasattr(self, '_table_known_mcs'):
                    self._table_known_mcs = set()
                self._table_known_mcs.update(valid_mcs)
                logger.info(
                    "_extract_header_references_only: page %d — extracted %d MC references: %s",
                    page_num + 1, len(valid_mcs), sorted(valid_mcs)
                )

        except Exception as exc:
            logger.warning("_extract_header_references_only: failed (%s)", exc)

    def _correct_token_against_reference(self, token, reference_set, threshold=0.92):
        """
        Try to correct a token using character substitutions against a reference set.
        Returns (corrected_token, best_score).
        """
        try:
            import Levenshtein as _lev
        except ImportError:
            return token, 0.0

        t = str(token).upper().strip()
        if t in reference_set:
            return t, 1.0

        best_match = None
        best_score = 0.0

        # Direct match
        for ref in reference_set:
            score = _lev.ratio(t, str(ref).upper())
            if score > best_score:
                best_score = score
                best_match = ref

        if best_score >= threshold:
            return best_match, best_score

        # Character confusions
        CHAR_CONFUSIONS = [
            ('V', 'Y'), ('S', '5'), ('5', 'S'),
            ('0', '5'), ('5', '0'),
            ('1', '7'), ('7', '1'),
            ('O', '0'), ('0', 'O'),
        ]

        # Single substitution
        for old_ch, new_ch in CHAR_CONFUSIONS:
            positions = [i for i, c in enumerate(t) if c == old_ch]
            for pos in positions:
                candidate = t[:pos] + new_ch + t[pos+1:]
                for ref in reference_set:
                    score = _lev.ratio(candidate, str(ref).upper())
                    if score > best_score:
                        best_score = score
                        best_match = ref

        if best_match and best_score >= threshold:
            return best_match, best_score

        # Double substitution
        for old_ch1, new_ch1 in CHAR_CONFUSIONS:
            positions1 = [i for i, c in enumerate(t) if c == old_ch1]
            for pos1 in positions1:
                candidate1 = t[:pos1] + new_ch1 + t[pos1+1:]
                for old_ch2, new_ch2 in CHAR_CONFUSIONS:
                    positions2 = [i for i, c in enumerate(candidate1) if c == old_ch2]
                    for pos2 in positions2:
                        if pos2 == pos1:
                            continue
                        candidate2 = candidate1[:pos2] + new_ch2 + candidate1[pos2+1:]
                        for ref in reference_set:
                            score = _lev.ratio(candidate2, str(ref).upper())
                            if score > best_score:
                                best_score = score
                                best_match = ref

        if best_match and best_score >= threshold:
            return best_match, best_score

        return token, best_score

    def _enhance_table_with_header_ocr(self, page, image, extract_result):
        """
        Table-mode enhancer: re-OCRs the table header row at 600 DPI to recover
        JB / MC identifiers that the full-page 300 DPI pass garbles.

        Why this is needed:
        - Many table PDFs (PDFium-generated) ship with Identity-H encoded fonts
          and NO ToUnicode CMap. Direct text extraction returns (cid:XX) tokens.
        - The full-page 300 DPI OCR pass recovers tags, cables and SPARE rows
          well, but the JB / MC header row contains smaller text that gets
          mangled (e.g. "JSF-576S" might come out as "JSF-5/6S" or "JSF-5765").
        - Re-rendering ONLY the header strip at 600 DPI and OCR'ing it with
          PSM 11 (sparse text) gives a much cleaner read of the JB/MC numbers.

        Args:
            page: fitz.Page object (used to render the header at 600 DPI).
            image: full-page image array (reserved for future cell-based use).
            extract_result: 9-tuple returned by extract_from_image.

        Returns:
            9-tuple with potentially augmented jb_identifiers, mc_identifiers
            and all_ocr_tags. Other fields are left untouched.
        """
        # Defensive unpacking — caller may pass anything
        if not isinstance(extract_result, tuple) or len(extract_result) != 9:
            logger.warning("_enhance_table_with_header_ocr: malformed extract_result, returning as-is")
            return extract_result

        (tags, jb_identifiers, mc_identifiers, cable_descriptions,
         spare_identifiers, tag_to_number, raw_cable_descriptions,
         tag_match_info, all_ocr_tags) = extract_result

        # Convert sets to mutable copies
        jb_identifiers = set(jb_identifiers) if jb_identifiers else set()
        mc_identifiers = set(mc_identifiers) if mc_identifiers else set()
        all_ocr_tags = set(all_ocr_tags) if all_ocr_tags else set()
        cable_descriptions = list(cable_descriptions) if cable_descriptions else []
        raw_cable_descriptions = list(raw_cable_descriptions) if raw_cable_descriptions else []

        try:
            import pdfplumber  # local import — keeps the module optional
        except ImportError:
            logger.warning("_enhance_table_with_header_ocr: pdfplumber not available, skipping enhancement")
            return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                    spare_identifiers, tag_to_number, raw_cable_descriptions,
                    tag_match_info, all_ocr_tags)

        try:
            doc = page.parent
            if doc is None:
                logger.warning("_enhance_table_with_header_ocr: page has no parent doc, skipping")
                return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                        spare_identifiers, tag_to_number, raw_cable_descriptions,
                        tag_match_info, all_ocr_tags)

            page_index = page.number if hasattr(page, 'number') else 0

            pdf_path = None
            try:
                pdf_path = doc.name
            except Exception:
                pdf_path = None

            if not pdf_path or not os.path.exists(pdf_path):
                logger.warning("_enhance_table_with_header_ocr: cannot resolve PDF path from fitz.Doc, skipping")
                return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                        spare_identifiers, tag_to_number, raw_cable_descriptions,
                        tag_match_info, all_ocr_tags)

            with pdfplumber.open(pdf_path) as pp_doc:
                if page_index >= len(pp_doc.pages):
                    logger.warning("_enhance_table_with_header_ocr: page_index out of range for pdfplumber, skipping")
                    return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                            spare_identifiers, tag_to_number, raw_cable_descriptions,
                            tag_match_info, all_ocr_tags)
                pp_page = pp_doc.pages[page_index]
                rects = pp_page.rects or []

            # Identify candidate header rows by analyzing the rectangle geometry.
            h_lines = [r for r in rects if r['width'] > 100 and r['height'] < 3]
            v_lines = [r for r in rects if r['height'] > 100 and r['width'] < 3]
            if not h_lines or not v_lines:
                logger.info("_enhance_table_with_header_ocr: no table lines detected, skipping header OCR")
                return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                        spare_identifiers, tag_to_number, raw_cable_descriptions,
                        tag_match_info, all_ocr_tags)

            h_lines.sort(key=lambda r: r['top'])
            v_lines.sort(key=lambda r: r['x0'])

            table_left = min(r['x0'] for r in v_lines)
            table_right = max(r['x0'] for r in v_lines)

            # Use the strip between the 1st and 4th horizontal lines as the
            # header region. This captures "JB No.:" + "MC No.:" + column headers.
            header_top = h_lines[0]['top']
            header_bottom = h_lines[min(3, len(h_lines) - 1)]['top']

            # Padding to avoid clipping text that hugs the line
            pad = 6.0
            header_top -= pad
            header_bottom += pad
            table_left -= pad
            table_right += pad

            # Clamp to page bounds
            page_w = page.rect.width
            page_h = page.rect.height
            header_top = max(0.0, header_top)
            header_bottom = min(page_h, header_bottom)
            table_left = max(0.0, table_left)
            table_right = min(page_w, table_right)

            logger.info(
                "_enhance_table_with_header_ocr: header region (pts) top=%.1f bottom=%.1f left=%.1f right=%.1f",
                header_top, header_bottom, table_left, table_right
            )

            # Render the header region at 900 DPI for maximum OCR accuracy
            # on the small JB/MC header text. 600 DPI was OK but still
            # produced character errors (J→C, 7→/, S→5). 900 DPI gives
            # Tesseract enough pixel density to read these reliably.
            clip_rect = fitz.Rect(table_left, header_top, table_right, header_bottom)
            header_dpi_matrix = fitz.Matrix(900 / 72, 900 / 72)
            try:
                pix = page.get_pixmap(matrix=header_dpi_matrix, clip=clip_rect)
            except Exception as render_err:
                logger.warning(
                    "_enhance_table_with_header_ocr: header render failed (%s), skipping",
                    render_err
                )
                return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                        spare_identifiers, tag_to_number, raw_cable_descriptions,
                        tag_match_info, all_ocr_tags)

            header_img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                header_img = cv2.cvtColor(header_img, cv2.COLOR_RGB2BGR)
            else:
                header_img = cv2.cvtColor(header_img, cv2.COLOR_RGBA2BGR)

            # Preprocess the high-DPI header (same simple pipeline as table branch)
            gray_h = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)
            gray_h = cv2.GaussianBlur(gray_h, (3, 3), 0)
            _, binary_h = cv2.threshold(gray_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR with multiple PSMs and combine (NO whitelist — see extract_from_image)
            cfg_psm6 = r'--oem 1 --psm 6 -l eng'
            cfg_psm11 = r'--oem 1 --psm 11 -l eng'

            text_psm6 = pytesseract.image_to_string(binary_h, config=cfg_psm6)
            text_psm11 = pytesseract.image_to_string(binary_h, config=cfg_psm11)
            combined_header_text = (text_psm6 or '') + '\n' + (text_psm11 or '')

            logger.info("_enhance_table_with_header_ocr: header PSM6=%r", text_psm6[:300])
            logger.info("_enhance_table_with_header_ocr: header PSM11=%r", text_psm11[:300])

            # Extract JB / MC / cable candidates from the header text.
            # Strategy: use SEMANTIC label-based extraction first (find text
            # between "JB No.:" and "MC No.:" labels), then fall back to
            # permissive regex.
            #
            # Why permissive? OCR on small CID-broken table text returns
            # errors like:
            #   - "JSF-576S" → "CSF -5/65"   (J→C, 7→/, S→5)
            #   - "NC-JSF-576S" → "NC-JSF-5/6"  (7→/, trailing S dropped)
            # We accept these and let the downstream similarity matcher
            # reconcile them with the IO list.

            def _normalize_ocr_id(s):
                """Fix common OCR errors in extracted identifiers."""
                s = str(s).strip().upper()
                # Remove leading/trailing punctuation
                s = s.strip(' \t,;:()[].')
                # Replace "/" between digits with "7" (OCR error: 7 → /)
                s = re.sub(r'(\d)/(\d)', r'\g<1>7\g<2>', s)
                # Remove internal spaces
                s = re.sub(r'\s+', '', s)
                return s

            # ── SEMANTIC JB extraction: text between "JB No.:" and "MC No.:" ──
            new_jbs = set()
            jb_label_pat = re.compile(
                r'JB\s*No\.?\s*[:.]?\s*(.+?)(?:MC\s*No\.?\s*[:.]?|Multi\s*Cable\s*No\.?\s*[:.]?|\]|$)',
                re.IGNORECASE | re.DOTALL
            )
            for m in jb_label_pat.finditer(combined_header_text):
                raw = m.group(1).strip()
                # Extract the first JB-like token from this slice
                token_pat = re.compile(r'[A-Z]{2,5}\s*-?\s*[\d/]{2,5}[A-Z0-9]{0,3}', re.IGNORECASE)
                for tm in token_pat.finditer(raw):
                    candidate = _normalize_ocr_id(tm.group(0))
                    if len(candidate) >= 4:
                        new_jbs.add(candidate)

            # ── SEMANTIC MC extraction: text between "MC No.:" and "(" or end ──
            new_mcs = set()
            mc_label_pat = re.compile(
                r'(?:MC\s*No\.?\s*[:.]?|Multi\s*Cable\s*No\.?\s*[:.]?)\s*(.+?)(?:\(|$)',
                re.IGNORECASE | re.DOTALL
            )
            for m in mc_label_pat.finditer(combined_header_text):
                raw = m.group(1).strip()
                # Extract MC-like tokens (NC-XXX-XXX or NC-0-1-2-C-3-BL)
                token_pat = re.compile(
                    r'NC-?[A-Z]{0,5}-?[\d/]{2,5}(?:[A-Z0-9-]{0,15})?',
                    re.IGNORECASE
                )
                for tm in token_pat.finditer(raw):
                    candidate = _normalize_ocr_id(tm.group(0))
                    if len(candidate) >= 4:
                        new_mcs.add(candidate)

            # ── FALLBACK: permissive regex on the whole header text ──
            # Only add candidates not already found by semantic extraction.
            # NOTE: no \b word boundary — underscores before "JSF" (e.g. "____JSF")
            # count as word chars and would block \b from matching.
            if not new_jbs:
                jb_pat = re.compile(r'J?[A-Z]{2,5}\s*-?[\d/]{2,5}[A-Z0-9]{0,3}', re.IGNORECASE)
                for m in jb_pat.finditer(combined_header_text):
                    candidate = _normalize_ocr_id(m.group(0))
                    if len(candidate) >= 4:
                        new_jbs.add(candidate)

            if not new_mcs:
                mc_pat = re.compile(r'NC-?[A-Z]{0,5}\s*-?[\d/]{2,5}[A-Z0-9-]{0,15}', re.IGNORECASE)
                for m in mc_pat.finditer(combined_header_text):
                    candidate = _normalize_ocr_id(m.group(0))
                    if len(candidate) >= 4:
                        new_mcs.add(candidate)

            # ── CABLE extraction: NC-X-X-X-X-X-XXX pattern ──
            new_cables = set()
            cable_pat = re.compile(
                r'\bNC-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}\b',
                re.IGNORECASE
            )
            for m in cable_pat.finditer(combined_header_text):
                candidate = _normalize_ocr_id(m.group(0))
                if len(candidate) >= 8:
                    new_cables.add(candidate)

            # Filter out obvious false positives
            def _is_plausible_id(s):
                s = str(s).upper().strip()
                if len(s) < 4:
                    return False
                if not any(c.isalpha() for c in s):
                    return False
                return True

            new_jbs = {j.upper() for j in new_jbs if _is_plausible_id(j)}
            new_mcs = {m.upper() for m in new_mcs if _is_plausible_id(m)}
            new_cables = {c.upper() for c in new_cables if _is_plausible_id(c)}

            # Strip tokens that are clearly the JB / MC label itself
            label_blacklist = {'JB', 'JBNO', 'MC', 'MCNO', 'NO'}
            new_jbs = {j for j in new_jbs if j not in label_blacklist}
            new_mcs = {m for m in new_mcs if m not in label_blacklist}

            logger.info(
                "_enhance_table_with_header_ocr: header OCR added %d JB candidates, %d MC candidates, %d cable candidates",
                len(new_jbs), len(new_mcs), len(new_cables)
            )
            if new_jbs:
                logger.info("   header JBs: %s", sorted(new_jbs))
            if new_mcs:
                logger.info("   header MCs: %s", sorted(new_mcs))
            if new_cables:
                logger.info("   header cables: %s", sorted(new_cables))

            # Merge into the result sets
            jb_identifiers.update(new_jbs)
            mc_identifiers.update(new_mcs)
            all_ocr_tags.update(new_jbs)
            all_ocr_tags.update(new_mcs)
            all_ocr_tags.update(new_cables)

            # ── Store JB/MC as reference for character correction ───────
            # These are extracted from the header (which is the most reliable
            # source for JB/MC identifiers). We'll use them in
            # _apply_table_ocr_corrections to fix OCR errors in the body.
            if new_jbs:
                if not hasattr(self, '_table_known_jbs'):
                    self._table_known_jbs = set()
                self._table_known_jbs.update(new_jbs)
                logger.info("_enhance_table: stored %d JB references (total: %d)",
                           len(new_jbs), len(self._table_known_jbs))
            if new_mcs:
                if not hasattr(self, '_table_known_mcs'):
                    self._table_known_mcs = set()
                self._table_known_mcs.update(new_mcs)
                logger.info("_enhance_table: stored %d MC references (total: %d)",
                           len(new_mcs), len(self._table_known_mcs))

            if new_cables:
                for c in sorted(new_cables):
                    if c not in cable_descriptions:
                        cable_descriptions.append(c)
                        raw_cable_descriptions.append(c)

        except Exception as exc:
            logger.warning("_enhance_table_with_header_ocr: enhancement failed (%s); returning original result", exc)

        # ── TABLE-MODE POST-PROCESSING ───────────────────────────────────
        # This is the critical cleanup step that fixes the issues reported
        # by the user:
        #   1. MC identifiers (NC-XXX-XXX) were leaking into tags
        #   2. Cable codes (NC-X-X-X-X-X-XXX) were treated as MC
        #   3. JB identifiers were sometimes noise (TZ17, LZ009, etc.)
        #   4. JB was only detected on one page (not preserved)
        #   5. Tags found in OCR were not matching against IO List
        try:
            (tags, jb_identifiers, mc_identifiers, cable_descriptions,
             spare_identifiers, tag_to_number, raw_cable_descriptions,
             tag_match_info, all_ocr_tags) = self._post_process_table_extractions(
                tags, jb_identifiers, mc_identifiers, cable_descriptions,
                spare_identifiers, tag_to_number, raw_cable_descriptions,
                tag_match_info, all_ocr_tags
            )
        except Exception as pp_err:
            logger.warning("_enhance_table_with_header_ocr: post-processing failed (%s)", pp_err)
        # ────────────────────────────────────────────────────────────────────

        return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                spare_identifiers, tag_to_number, raw_cable_descriptions,
                tag_match_info, all_ocr_tags)

    def _post_process_table_extractions(
        self, tags, jb_identifiers, mc_identifiers, cable_descriptions,
        spare_identifiers, tag_to_number, raw_cable_descriptions,
        tag_match_info, all_ocr_tags
    ):
        """
        Table-mode post-processor: cleans up the extraction results to fix
        the issues reported in production:

        1. MC vs Tag separation:
           Any token starting with 'NC' (or 'NCJ'/'NCS' after hyphen stripping)
           is treated as MC or Cable, NOT a tag.

        2. Cable vs MC separation:
           - Cable code pattern: NC-X-X-X-X-X-XXX (X = digit, last = letters)  (e.g. NC-0-1-2-C-3-BL)
           - MC identifier pattern: NC-LETTERS-DIGITS (e.g. NC-JSF-576S)  (e.g. NC-JSF-576S)
           Tokens matching the cable pattern go to cable_descriptions.
           Tokens matching the MC pattern go to mc_identifiers.

        3. JB cleanup:
           - Remove obvious noise (tokens < 5 chars, tokens without digits,
             tokens that are clearly OCR artefacts like 'JBNO:', 'TZ17',
             'LZ009', 'OULV1274').
           - Keep only tokens matching the JB identifier pattern
             (e.g. JSF-576S, JSF-513S).
           - PRESERVE the last valid JB across pages: if the current page
             has no valid JB after cleanup, reuse the JB from the previous
             page (stored in self._table_last_valid_jb).

        4. Tag cleanup:
           - Remove any token from tags/all_ocr_tags that looks like a JB
             (starts with JSF or matches the jb_examples prefix).
           - Remove any token starting with NC (these are MC/cable).

        5. SPARE count normalization:
           - In table-mode, count SPARE occurrences in the OCR data and
             ensure the spare_identifiers list contains one entry per
             actual SPARE row.

        6. Tag matching fallback (table-mode only):
           - If IO List tags are available, do a fuzzy match (threshold 0.85)
             between all_ocr_tags and IO List tags.
           - Any OCR tag that fuzzy-matches an IO List tag is added to the
             final tags set with its IO List form (corrected OCR errors).
        """
        import re as _re

        # Make mutable copies
        tags = set(tags) if tags else set()
        jb_identifiers = set(jb_identifiers) if jb_identifiers else set()
        mc_identifiers = set(mc_identifiers) if mc_identifiers else set()
        all_ocr_tags = set(all_ocr_tags) if all_ocr_tags else set()
        cable_descriptions = list(cable_descriptions) if cable_descriptions else []
        raw_cable_descriptions = list(raw_cable_descriptions) if raw_cable_descriptions else []
        spare_identifiers = list(spare_identifiers) if spare_identifiers else []

        # ── Patterns ────────────────────────────────────────────────────
        # Cable: NC-X-X-X-X-X-XXX (X = digit, last = letters)
        cable_re = _re.compile(r'^NC-\d{1,2}-\d{1,2}-\d{1,2}-[A-Z]-\d{1,2}-[A-Z]{1,3}$', _re.IGNORECASE)
        # MC: NC-LETTERS-DIGITS (e.g. NC-JSF-576S, NC-JSF-5135)
        mc_re = _re.compile(r'^NC-?[A-Z]{2,5}-?\d{2,5}[A-Z]?$', _re.IGNORECASE)
        # JB identifier: 2-5 letters, dash, 2-5 digits, optional letter (e.g. JSF-576S)
        jb_re = _re.compile(r'^[A-Z]{2,5}-\d{2,5}[A-Z]?$', _re.IGNORECASE)
        # Tag: similar to JB but with at least one letter prefix that is NOT JSF/NC
        # (e.g. UY-5021, FUY-5041)
        # IO List tag: starts with letters, dash, 3-4 digits, optional letter
        iolist_tag_re = _re.compile(r'^[A-Z]{1,5}-\d{3,4}[A-Z]?$', _re.IGNORECASE)
        # Noise patterns
        noise_re = _re.compile(r'^(JBNO|JBNO:|TZ\d+|LZ\d+|OULV\d+|MCNO|MCNO:)$', _re.IGNORECASE)

        # ── Step 1: Separate MC and Cable from all_ocr_tags ─────────────
        # Any token starting with NC is either an MC or a Cable, NOT a tag.
        nc_tokens_from_tags = set()
        clean_ocr_tags = set()
        for tok in all_ocr_tags:
            t = str(tok).upper().strip()
            if t.startswith('NC'):
                nc_tokens_from_tags.add(tok)
            elif t.startswith('NCJSF') or t.startswith('NC-JSF'):
                nc_tokens_from_tags.add(tok)
            else:
                clean_ocr_tags.add(tok)
        all_ocr_tags = clean_ocr_tags

        # Now distribute nc_tokens to cable_descriptions or mc_identifiers
        for tok in nc_tokens_from_tags:
            t = str(tok).upper().strip()
            # Try cable pattern first (more specific)
            # Normalize: ensure NC-X-X-X-X-X-XXX form
            t_normalized = t
            if t.startswith('NC') and not t.startswith('NC-'):
                # Insert dash after NC
                t_normalized = 'NC-' + t[2:]
            if cable_re.match(t_normalized):
                if t_normalized not in cable_descriptions:
                    cable_descriptions.append(t_normalized)
                    raw_cable_descriptions.append(t_normalized)
                # Remove from mc_identifiers if present
                mc_identifiers.discard(tok)
                mc_identifiers.discard(t)
                mc_identifiers.discard(t_normalized)
            elif mc_re.match(t_normalized):
                mc_identifiers.add(t_normalized)
                # Remove from cable_descriptions if present
                if t_normalized in cable_descriptions:
                    cable_descriptions.remove(t_normalized)
                    if t_normalized in raw_cable_descriptions:
                        raw_cable_descriptions.remove(t_normalized)

        # ── Step 2: Clean up cable_descriptions (remove anything that's actually MC) ──
        clean_cables = []
        for c in cable_descriptions:
            t = str(c).upper().strip()
            if cable_re.match(t):
                clean_cables.append(c)
            elif mc_re.match(t):
                # This is actually an MC, not a cable
                mc_identifiers.add(t)
            # else: keep it as cable anyway (might be a different format)
        cable_descriptions = clean_cables if clean_cables else cable_descriptions

        # ── Step 2b: Clean up mc_identifiers (move cable-pattern tokens to cables) ──
        # Some MC identifiers may actually be cable codes (NC-X-X-X-X-X-XXX).
        # Move them to cable_descriptions.
        clean_mcs = set()
        for m in mc_identifiers:
            t = str(m).upper().strip()
            if cable_re.match(t):
                # This is a cable, not an MC — move it
                if t not in cable_descriptions:
                    cable_descriptions.append(t)
                    raw_cable_descriptions.append(t)
                logger.info(f"_post_process: moved cable-pattern token from MC to cables: {t}")
            else:
                clean_mcs.add(m)
        mc_identifiers = clean_mcs

        # ── Step 3: Clean up JB identifiers ────────────────────────────
        clean_jbs = set()
        for j in jb_identifiers:
            t = str(j).upper().strip().strip(':').strip('.')
            # Skip noise
            if noise_re.match(t):
                continue
            # Skip if too short or no digits
            if len(t) < 5 or not any(c.isdigit() for c in t):
                continue
            # Keep if matches JB pattern
            if jb_re.match(t):
                clean_jbs.add(t)
            # Also keep tokens that match the user-configured jb_examples prefix
            elif self.jb_examples and any(t.startswith(p) for p in (self._parse_multi_patterns(self.jb_examples) or []) if p):
                if any(c.isdigit() for c in t):
                    clean_jbs.add(t)

        # ── Step 3b: DERIVE JB from MC identifier ─────────────────────
        # In table-mode PDFs, the header contains "JB No.: JSF-576S" and
        # "MC No.: NC-JSF-576S". If we have an MC like NC-JSF-576S but no
        # JB, we can derive JB = JSF-576S by stripping the "NC-" prefix.
        if not clean_jbs and mc_identifiers:
            for mc in mc_identifiers:
                m = _re.match(r'^NC-([A-Z]{2,5}-?\d{2,5}[A-Z]?)$', str(mc).upper().strip())
                if m:
                    derived_jb = m.group(1)
                    if jb_re.match(derived_jb):
                        clean_jbs.add(derived_jb)
                        logger.info(f"_post_process: derived JB '{derived_jb}' from MC '{mc}'")
                        break  # Use the first valid derivation

        jb_identifiers = clean_jbs

        # ── Step 4: JB Preservation across pages ───────────────────────
        # If current page has no valid JB after cleanup, we do NOT reuse the
        # JB from the previous page. Instead, we mark the JB as "unknown" so
        # that tags/spares on this page are NOT incorrectly assigned to a
        # different JB.
        #
        # The previous behavior (reusing JB from previous page) caused tags
        # and spares from a page with no JB to be incorrectly assigned to
        # the previous page's JB — which is wrong because each page in a
        # table-mode PDF typically belongs to a DIFFERENT JB.
        if not jb_identifiers:
            # Do NOT reuse — leave jb_identifiers empty so tags on this page
            # get marked as "JB_NOT_FOUND" instead of being assigned to the
            # wrong JB.
            last_jb = getattr(self, '_table_last_valid_jb', None)
            if last_jb:
                logger.info(
                    f"_post_process: NO JB found on this page — "
                    f"NOT reusing '{last_jb}' from previous page "
                    f"(tags/spares will be marked as JB_NOT_FOUND to avoid wrong assignment)"
                )
            else:
                logger.info(f"_post_process: NO JB found on this page and no previous JB available")
        else:
            # Deduplicate: if multiple JB candidates, keep only the longest one
            # (it's the most complete identifier — e.g. JSF-5776S over JSF-57).
            # Also remove tokens that are strict prefixes of other tokens.
            jb_list = sorted(jb_identifiers, key=len, reverse=True)
            kept_jbs = []
            for jb in jb_list:
                is_prefix_of_another = any(
                    jb != other and other.startswith(jb)
                    for other in jb_list
                )
                if not is_prefix_of_another:
                    kept_jbs.append(jb)
            # If everything was filtered as a prefix, keep the longest
            if not kept_jbs:
                kept_jbs = [jb_list[0]]
            jb_identifiers = set(kept_jbs[:1])  # Keep only the single best JB per page

            # Store the best valid JB (longest one) for use by subsequent pages
            self._table_last_valid_jb = max(jb_identifiers, key=len)
            logger.info(f"_post_process: stored JB for subsequent pages: {self._table_last_valid_jb}")

        # ── Step 4b: Remove JB OCR-variants from all_ocr_tags ──────────
        # OCR may read "JSF-5135" as "USF-5135" (J→U error). If we already
        # have the correct JB identifier, remove the OCR variant from
        # all_ocr_tags so it doesn't get classified as a tag.
        if jb_identifiers:
            try:
                import Levenshtein as _lev
                jbs_to_check = list(jb_identifiers)
                tags_to_remove = set()
                for ocr_tag in list(all_ocr_tags):
                    ocr_upper = str(ocr_tag).upper().strip()
                    # Skip if it starts with NC (already handled)
                    if ocr_upper.startswith('NC'):
                        continue
                    # Check similarity to each JB identifier
                    for jb_id in jbs_to_check:
                        jb_upper = str(jb_id).upper().strip()
                        sim = _lev.ratio(ocr_upper, jb_upper)
                        if sim >= 0.85:
                            tags_to_remove.add(ocr_tag)
                            logger.info(
                                "_post_process: removing OCR variant '%s' from tags (similar to JB '%s', sim=%.2f)",
                                ocr_tag, jb_id, sim
                            )
                            break
                if tags_to_remove:
                    all_ocr_tags -= tags_to_remove
                    tags -= tags_to_remove
            except ImportError:
                pass

        # ── Step 5: Clean up tags (remove JB-like and NC-like tokens) ──
        # Tokens that look like JB (match jb_examples prefix + JB pattern) are
        # moved to a temporary set; we'll merge and deduplicate them with the
        # existing jb_identifiers AFTER this step.
        jb_candidates_from_tags = set()
        clean_tags = set()
        for t in tags:
            s = str(t).upper().strip()
            # Skip if starts with NC (MC/cable, not tag)
            if s.startswith('NC'):
                continue
            # Skip if matches JB pattern AND starts with jb_examples prefix
            _jb_p = self._parse_multi_patterns(self.jb_examples) if self.jb_examples else []
            if _jb_p and any(s.startswith(p) for p in _jb_p if p):
                if jb_re.match(s):
                    # This is a JB, not a tag — collect for merging below
                    jb_candidates_from_tags.add(s)
                    continue
            # Skip noise
            if noise_re.match(s):
                continue
            clean_tags.add(t)
        tags = clean_tags

        # Also clean all_ocr_tags (remove JB-like tokens)
        clean_ocr_tags = set()
        for t in all_ocr_tags:
            s = str(t).upper().strip()
            if s.startswith('NC'):
                continue
            if _jb_p and any(s.startswith(p) for p in _jb_p if p):
                if jb_re.match(s):
                    jb_candidates_from_tags.add(s)
                    continue
            if noise_re.match(s):
                continue
            clean_ocr_tags.add(t)
        all_ocr_tags = clean_ocr_tags

        # Merge any JB candidates found in tags/ocr_tags back into jb_identifiers,
        # then re-run the deduplication + preservation logic from Step 4.
        if jb_candidates_from_tags:
            jb_identifiers.update(jb_candidates_from_tags)
            # Re-derive best JB with deduplication
            jb_list = sorted(jb_identifiers, key=len, reverse=True)
            kept_jbs = []
            for jb in jb_list:
                is_prefix_of_another = any(
                    jb != other and other.startswith(jb)
                    for other in jb_list
                )
                if not is_prefix_of_another:
                    kept_jbs.append(jb)
            if not kept_jbs:
                kept_jbs = [jb_list[0]]
            jb_identifiers = set(kept_jbs[:1])
            if jb_identifiers:
                self._table_last_valid_jb = max(jb_identifiers, key=len)
                logger.info(f"_post_process: re-stored JB after tag cleanup: {self._table_last_valid_jb}")

        # ── Step 6: Tag matching fallback ──────────────────────────────
        # If IO List tags are available, do a fuzzy match between all_ocr_tags
        # and IO List tags. This catches OCR variants like UY-5021 / UV-5021.
        io_list_tags = set()
        if hasattr(self, 'io_list_tags') and self.io_list_tags:
            io_list_tags = set(str(t).upper().strip() for t in self.io_list_tags)
        elif hasattr(self, 'excel_df') and hasattr(self, 'excel_tag_column'):
            if self.excel_df is not None and not self.excel_df.empty:
                tag_col = self.excel_tag_column
                io_list_tags = set(str(t).strip().upper() for t in self.excel_df[tag_col] if pd.notna(t))

        if io_list_tags:
            try:
                import Levenshtein as _lev
            except ImportError:
                _lev = None

            matched_pairs = []
            for ocr_tag in list(all_ocr_tags):
                ocr_upper = str(ocr_tag).upper().strip()
                # Try exact match first
                if ocr_upper in io_list_tags:
                    if ocr_upper not in tags:
                        tags.add(ocr_upper)
                        matched_pairs.append((ocr_tag, ocr_upper, 1.0))
                    continue
                # Fuzzy match
                if _lev:
                    best_io = None
                    best_score = 0.0
                    for io_tag in io_list_tags:
                        score = _lev.ratio(ocr_upper, str(io_tag).upper())
                        if score > best_score:
                            best_score = score
                            best_io = io_tag
                    # Threshold 0.85 — catches OCR errors like UY→UV (sim=0.88)
                    if best_io and best_score >= 0.85:
                        if best_io not in tags:
                            tags.add(best_io)
                            matched_pairs.append((ocr_tag, best_io, best_score))

            if matched_pairs:
                logger.info(
                    "_post_process: tag matching fallback recovered %d tags: %s",
                    len(matched_pairs),
                    [(o, i, round(s, 2)) for o, i, s in matched_pairs]
                )

        # ── Step 6b: Remove unmatched JB-like tokens from all_ocr_tags ──
        # After tag matching, any token in all_ocr_tags that:
        #   1. Matches the JB pattern AND starts with jb_examples prefix
        #      (e.g. JSF-XXXX — these are JB identifiers, not tags)
        #   2. Was NOT matched to any IO List tag (not in `tags`)
        #   3. Is NOT similar to any IO List tag
        # ...is likely a JB OCR variant and should be removed.
        # IMPORTANT: We ONLY remove tokens that start with the jb_examples
        # prefix (e.g. "JS"). Tokens like "FUY-5239" that don't start with
        # the JB prefix are NOT removed, even if they match the JB pattern,
        # because they could be real tags that aren't in the IO List.
        _jb_prefix = str(getattr(self, 'jb_examples', None) or '').strip().upper()
        if io_list_tags and _jb_prefix:
            final_tags_upper = set(str(t).upper() for t in tags)
            tags_to_remove_final = set()
            for ocr_tag in list(all_ocr_tags):
                ocr_upper = str(ocr_tag).upper().strip()
                # Skip if it's a matched tag
                if ocr_upper in final_tags_upper:
                    continue
                # Skip if it's an exact IO List tag
                if ocr_upper in io_list_tags:
                    continue
                # Only check tokens that start with JB prefix (e.g. JS, JB)
                # This prevents removing real tags like FUY-5239
                if not any(ocr_upper.startswith(p) for p in _jb_prefix.split(',') if p):
                    continue
                # Check if it matches JB pattern (letters-digits)
                if jb_re.match(ocr_upper):
                    # Check if it's similar to any IO List tag (might be a real tag)
                    is_real_tag = False
                    try:
                        import Levenshtein as _lev
                        for io_tag in io_list_tags:
                            if _lev.ratio(ocr_upper, str(io_tag).upper()) >= 0.80:
                                is_real_tag = True
                                break
                    except ImportError:
                        pass
                    if not is_real_tag:
                        tags_to_remove_final.add(ocr_tag)
                        logger.info(
                            "_post_process: removing unmatched JB-like token '%s' from all_ocr_tags",
                            ocr_tag
                        )
            if tags_to_remove_final:
                all_ocr_tags -= tags_to_remove_final

        # ── Step 7: SPARE count normalization ──────────────────────────
        # Count distinct SPARE entries (handle both 'SPARE' and '|SPARE')
        clean_spares = []
        seen_spare_indices = set()
        for i, s in enumerate(spare_identifiers):
            s_str = str(s).upper().strip()
            if 'SPARE' in s_str:
                # Use a canonical form
                canonical = f"SPARE_{len(clean_spares) + 1}"
                clean_spares.append(canonical)
        # If we found any spares, replace the list
        if clean_spares:
            spare_identifiers = clean_spares

        logger.info(
            "_post_process: final counts — tags=%d, jbs=%d, mcs=%d, cables=%d, spares=%d, ocr_tags=%d",
            len(tags), len(jb_identifiers), len(mc_identifiers),
            len(cable_descriptions), len(spare_identifiers), len(all_ocr_tags)
        )

        return (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                spare_identifiers, tag_to_number, raw_cable_descriptions,
                tag_match_info, all_ocr_tags)

    def process_pdf_page(self, page_info: 'Tuple[fitz.Page, str, int]') -> 'Tuple[int, Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict], Set[str]]':        
        """
        ✅ بازنویسی: پردازش یک صفحه PDF با شماره‌گذاری بر اساس موقعیت
        """
        page, temp_dir, page_num = page_info
        
        try:
            pdf_type = getattr(self, '_current_pdf_type', 'diagrams')
            # Normalize pdf_type at this routing point to guarantee consistency
            _l = str(pdf_type or '').lower()
            if 'table' in _l:
                pdf_type = 'table'
            elif 'diagram' in _l or 'drawing' in _l:
                pdf_type = 'diagrams'
            else:
                pdf_type = 'diagrams'

            if pdf_type == 'table':
                # REVISED: was 150/72, now 300/72.
                # 150 DPI was too low for the small text inside table cells of
                # CID-broken PDFium PDFs — OCR was returning garbage for JB
                # numbers, MC numbers and cable codes. 300 DPI gives ~2x the
                # pixel density and recovers these tokens reliably.
                dpi_factor = 300 / 72
                logger.info(f"process_pdf_page {page_num + 1}: TABLE mode — dpi_factor={dpi_factor:.3f}")
            else:
                dpi_factor = 300 / 72  # original diagram value, unchanged

            # default extraction mode
            extraction_mode = 'ocr'
            pdf_nature = getattr(self, '_current_pdf_nature', 'scanned')
            if pdf_nature == 'digital':
                logger.info(
                    f"process_pdf_page {page_num + 1}: DIGITAL page text extraction "
                    f"(dpi_factor={dpi_factor:.4f})"
                )
                extraction_mode = 'digital'
                result = self.extract_from_text_page(page, dpi_factor=dpi_factor)
                if not result or not isinstance(result, tuple) or len(result) != 9:
                    logger.warning(
                        "extract_from_text_page returned unexpected result for page %s, falling back to OCR",
                        page_num + 1
                    )
                    image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor), colorspace=_CS_GRAY)
                    pix.save(image_path)
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"Failed to load image for page {page_num + 1}")
                        return page_num + 1, set(), set(), set(), [], [], {}, [], {}, set()
                    result = self.extract_from_image(image)
                    extraction_mode = 'ocr'
            else:
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor), colorspace=_CS_GRAY)
                pix.save(image_path)
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"Failed to load image for page {page_num + 1}")
                    return page_num + 1, set(), set(), set(), [], [], {}, [], {}, set()
                # TABLE-MODE: use multi-pass extraction (S13 strategy) for
                # maximum recall. Diagram-mode keeps the original single-pass.
                if pdf_type == 'table':
                    # ── STEP 1: Try digital text extraction for JB/MC labels ──
                    # This uses the PDF's digital text layer (not OCR) to find
                    # "JB No.: JSF-309S" and "Multi Cable No.: NC-JSF-309S"
                    # in the header. Much more accurate than OCR.
                    #
                    # This is NOT mandatory — it's an ADDITIONAL method that runs
                    # alongside the OCR-based header extraction. If digital finds
                    # something, great. If not, OCR still runs. If both find
                    # something, we merge the results (digital takes priority).
                    digital_jbs, digital_mcs = set(), set()
                    try:
                        digital_jbs, digital_mcs = self._extract_jb_mc_from_digital_header(page, page_num)
                        if digital_jbs:
                            if not hasattr(self, '_table_known_jbs'):
                                self._table_known_jbs = set()
                            self._table_known_jbs.update(digital_jbs)
                            self._table_last_valid_jb = max(digital_jbs, key=len)
                            logger.info(
                                "process_pdf_page %d: digital header found JBs: %s",
                                page_num + 1, digital_jbs
                            )
                        if digital_mcs:
                            if not hasattr(self, '_table_known_mcs'):
                                self._table_known_mcs = set()
                            self._table_known_mcs.update(digital_mcs)
                            self._table_last_valid_mc = max(digital_mcs, key=len)
                            logger.info(
                                "process_pdf_page %d: digital header found MCs: %s",
                                page_num + 1, digital_mcs
                            )
                    except Exception as digital_hdr_err:
                        logger.warning(
                            "process_pdf_page %d: digital header extraction failed: %s",
                            page_num + 1, digital_hdr_err
                        )

                    # ── STEP 2: Run OCR-based header extraction ONLY if digital found nothing ──
                    # If digital already found JB/MC, skip OCR (saves time).
                    _prev_jb = getattr(self, '_table_last_valid_jb', None)
                    if not digital_jbs and not digital_mcs:
                        if _prev_jb:
                            # Digital found nothing, but we have JB from previous page
                            logger.info(
                                "process_pdf_page %d: skipping header OCR — "
                                "reusing JB '%s' from previous page (digital found nothing)",
                                page_num + 1, _prev_jb
                            )
                            if not hasattr(self, '_table_known_jbs') or not self._table_known_jbs:
                                self._table_known_jbs = {_prev_jb}
                            if not hasattr(self, '_table_known_mcs') or not self._table_known_mcs:
                                mc_prefix = self._get_mc_prefix()
                                if mc_prefix:
                                    mc_val = mc_prefix + '-' + _prev_jb if not _prev_jb.startswith(mc_prefix) else _prev_jb
                                    self._table_known_mcs = {mc_val}
                        else:
                            # Digital found nothing, no previous JB — run OCR
                            try:
                                self._extract_header_references_only(page, page_num)
                            except Exception as hdr_err:
                                logger.warning(
                                    "process_pdf_page %d: header reference extraction failed: %s",
                                    page_num + 1, hdr_err
                                )
                    logger.info(f"process_pdf_page {page_num + 1}: TABLE mode — using multi-pass OCR (S13)")
                    result = self._extract_from_image_table_multipass(page, temp_dir, page_num)
                else:
                    result = self.extract_from_image(image)
                extraction_mode = 'ocr'

            logger.debug(f"Page {page_num + 1} - raw result: {result} (len={len(result)})")
            if len(result) != 9:
                logger.error(f"❌ Expected 9 values, got {len(result)}")
                return page_num + 1, set(), set(), set(), [], [], {}, [], {}, set()
            
            logger.info(f"Page {page_num + 1} - pdf_type_used={pdf_type} extraction_mode={extraction_mode}")

            # ✅ Unpack
            (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
            spare_identifiers, tag_to_number, raw_cable_descriptions, 
            tag_match_info, all_ocr_tags) = result

            # If digital extraction produced too little usable data, fall back to OCR
            insufficient_content = (
                (not tags or len(tags) < 3) and
                (not jb_identifiers) and
                (not mc_identifiers)
            )
            if insufficient_content:
                logger.info(
                    "Digital extraction produced insufficient tags (tags=%d, jbs=%d, mcs=%d). Falling back to OCR.",
                    len(tags), len(jb_identifiers), len(mc_identifiers)
                )

                # Ensure we have a rasterized image to feed OCR
                try:
                    if 'image' not in locals():
                        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor), colorspace=_CS_GRAY)
                        pix.save(image_path)
                        image = cv2.imread(image_path)
                        if image is None:
                            raise RuntimeError("Failed to load rasterized page image for OCR fallback")

                    ocr_result = self.extract_from_image(image)
                    if isinstance(ocr_result, tuple) and len(ocr_result) == 9:
                        (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
                         spare_identifiers, tag_to_number, raw_cable_descriptions, 
                         tag_match_info, all_ocr_tags) = ocr_result
                        logger.info("OCR fallback succeeded: %d tags", len(tags))
                    else:
                        logger.warning("OCR fallback did not return expected structure; keeping digital result")
                except Exception as fb_err:
                    logger.warning(f"OCR fallback failed: {fb_err}; keeping digital result")

            logger.info(f"\u2705 Page {page_num + 1}: {len(tags)} tags numbered by position")

            # ── TABLE-MODE ENHANCEMENT ──────────────────────────────────────────
            # For table PDFs with CID-broken fonts, the full-page 300 DPI OCR
            # pass MAY miss the JB / MC header row. In that case, re-OCR ONLY
            # the header strip at 900 DPI and merge the recovered identifiers.
            #
            # IMPORTANT: We only call the header enhancer if the full-page OCR
            # pass did NOT find any JB identifiers. This avoids polluting the
            # results with high-DPI OCR variants (e.g. JSF-5776S instead of
            # JSF-5765) that conflict with the more accurate full-page results.
            if pdf_type == 'table' and not jb_identifiers and not mc_identifiers:
                try:
                    enhanced = self._enhance_table_with_header_ocr(page, image, result)
                    if isinstance(enhanced, tuple) and len(enhanced) == 9:
                        result = enhanced
                        (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                         spare_identifiers, tag_to_number, raw_cable_descriptions,
                         tag_match_info, all_ocr_tags) = result
                        logger.info(
                            "\u2705 Page %d table-mode header enhancement: JBs=%d, MCs=%d, cables=%d",
                            page_num + 1, len(jb_identifiers), len(mc_identifiers), len(cable_descriptions)
                        )
                except Exception as enh_err:
                    logger.warning("Table-mode header enhancement failed for page %d: %s", page_num + 1, enh_err)
            # ─────────────────────────────────────────────────────────────────────

            # ── TABLE-MODE POST-PROCESSING (always run for table PDFs) ────────
            # This is the critical cleanup step that fixes the issues reported
            # in production:
            #   1. MC identifiers (NC-XXX-XXX) were leaking into tags
            #   2. Cable codes (NC-X-X-X-X-X-XXX) were treated as MC
            #   3. JB identifiers were sometimes noise (TZ17, LZ009, etc.)
            #   4. JB was only detected on one page (not preserved)
            #   5. Tags found in OCR were not matching against IO List
            # We run this ALWAYS for table PDFs, regardless of whether the
            # header enhancer ran.
            if pdf_type == 'table':
                try:
                    pp_result = self._post_process_table_extractions(
                        tags, jb_identifiers, mc_identifiers, cable_descriptions,
                        spare_identifiers, tag_to_number, raw_cable_descriptions,
                        tag_match_info, all_ocr_tags
                    )
                    if isinstance(pp_result, tuple) and len(pp_result) == 9:
                        (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                         spare_identifiers, tag_to_number, raw_cable_descriptions,
                         tag_match_info, all_ocr_tags) = pp_result
                        # Update result tuple
                        result = (tags, jb_identifiers, mc_identifiers, cable_descriptions,
                                  spare_identifiers, tag_to_number, raw_cable_descriptions,
                                  tag_match_info, all_ocr_tags)
                        logger.info(
                            "\u2705 Page %d table-mode post-process: tags=%d, JBs=%d, MCs=%d, cables=%d, spares=%d",
                            page_num + 1, len(tags), len(jb_identifiers), len(mc_identifiers),
                            len(cable_descriptions), len(spare_identifiers)
                        )
                except Exception as pp_err:
                    logger.warning("Table-mode post-process failed for page %d: %s", page_num + 1, pp_err)
            # ─────────────────────────────────────────────────────────────────────

        # ✅ Return 10 values
            return (page_num + 1, tags, jb_identifiers, mc_identifiers,
                    cable_descriptions, spare_identifiers, tag_to_number,
                    raw_cable_descriptions, tag_match_info, all_ocr_tags)
               
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            logger.error(traceback.format_exc())
            return page_num + 1, set(), set(), set(), [], [], {}, [], {}, set()


    def detect_pdf_nature(self, pdf_path: str, sample_pages: int = 2) -> str:
        """Detect whether a PDF contains extractable text (digital) or is image/scanned."""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            if total_pages == 0:
                doc.close()
                return 'scanned'

            pages_to_sample = min(sample_pages, total_pages)
            extracted_text = []
            for page_index in range(pages_to_sample):
                page = doc[page_index]
                page_text = page.get_text('text') or ''
                if page_text.strip():
                    extracted_text.append(page_text.strip())
            doc.close()

            combined_text = '\n'.join(extracted_text)
            word_count = len(re.findall(r'\w+', combined_text))
            if len(combined_text) >= 120 or word_count >= 20:
                return 'digital'
            return 'scanned'
        except Exception as exc:
            logger.warning(
                "PDF nature detection failed for %s: %s — defaulting to 'scanned'",
                os.path.basename(pdf_path),
                exc
            )
            return 'scanned'



    # ─────────────────────────────────────────────────────────────
    # BATCH PROCESSING WITH CHECKPOINT
    # For large PDFs (100+ pages), process in batches and save
    # checkpoint after each batch. If worker dies (OOM/timeout),
    # resume from last checkpoint on next run.
    # ─────────────────────────────────────────────────────────────

    BATCH_SIZE = 20  # Process 20 pages per batch

    def _get_checkpoint_path(self, pdf_path: str) -> str:
        """Get checkpoint file path for a PDF."""
        import hashlib
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        checkpoint_dir = os.path.join(os.path.dirname(pdf_path), '.checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f'checkpoint_{pdf_hash}.json')

    def _save_checkpoint(self, pdf_path: str, results: dict, last_page: int):
        """Save processing checkpoint after each batch."""
        try:
            checkpoint_path = self._get_checkpoint_path(pdf_path)
            checkpoint_data = {
                'pdf_path': pdf_path,
                'last_page': last_page,
                'results': {str(k): list(v) if isinstance(v, (set, tuple)) else v 
                           for k, v in results.items()},
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, default=str, indent=2)
            logger.info(f"Checkpoint saved: page {last_page + 1}, {len(results)} pages processed")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, pdf_path: str) -> dict:
        """Load checkpoint if exists. Returns empty dict if no checkpoint."""
        try:
            checkpoint_path = self._get_checkpoint_path(pdf_path)
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Checkpoint loaded: resuming from page {data['last_page'] + 1}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
        return {}

    def _clear_checkpoint(self, pdf_path: str):
        """Clear checkpoint after successful completion."""
        try:
            checkpoint_path = self._get_checkpoint_path(pdf_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.info("Checkpoint cleared (processing completed)")
        except Exception:
            pass

    def process_pdf(self, pdf_path: str) -> 'Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict] ,Set[str]]]':
        """
        Process all pages in a PDF file.
        """
        results = {}
        
        try:
            common_locations = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract'
            ]
            tesseract_found = False
            for location in common_locations:
                if os.path.exists(location):
                    pytesseract.pytesseract.tesseract_cmd = location
                    tesseract_found = True
                    break
            if not tesseract_found:
                raise RuntimeError("Tesseract not found in common locations")
        except Exception as e:
            logger.error(f"Error initializing Tesseract in process: {e}")
            return {}

        pdf_nature = 'scanned'
        try:
            if pdf_path in getattr(self, 'document_nature_by_path', {}):
                pdf_nature = self.document_nature_by_path[pdf_path]
            else:
                pdf_nature = self.detect_pdf_nature(pdf_path)
                self.document_nature_by_path[pdf_path] = pdf_nature
            self._current_pdf_nature = pdf_nature
            logger.info(
                "PDF nature detected as: '%s' for %s",
                pdf_nature,
                os.path.basename(pdf_path)
            )
        except Exception as nature_err:
            logger.warning(
                "PDF nature detection failed for '%s': %s — defaulting to 'scanned'",
                os.path.basename(pdf_path),
                nature_err
            )
            self._current_pdf_nature = 'scanned'

        if self._classifier is not None:
            try:
                if pdf_path in getattr(self, 'document_type_by_path', {}):
                    pdf_type = self.document_type_by_path[pdf_path]
                else:
                    raw_label = self._classifier.classify_pdf(pdf_path)
                    label_l = (raw_label or '').strip().lower()
                    if 'table' in label_l:
                        pdf_type = 'table'
                    elif 'diagram' in label_l or 'diagra' in label_l or 'drawing' in label_l:
                        pdf_type = 'diagrams'
                    else:
                        pdf_type = 'diagrams'
                    self.document_type_by_path[pdf_path] = pdf_type

                self._current_pdf_type = pdf_type
                logger.info(
                    "PDF classified as: '%s' (raw='%s') → routing to '%s'  [%s]",
                    pdf_type,
                    raw_label if 'raw_label' in locals() else '',
                    pdf_type,
                    os.path.basename(pdf_path)
                )
            except Exception as clf_err:
                logger.warning(
                    "PDFClassifier raised an exception for '%s': %s — falling back to 'diagrams'",
                    os.path.basename(pdf_path),
                    clf_err
                )
                self._current_pdf_type = "diagrams"
        else:
            # No classifier injected — default to diagram mode (original behaviour)
            self._current_pdf_type = "diagrams"
            logger.info(
                "No PDFClassifier injected — defaulting to 'diagrams' mode for '%s'",
                os.path.basename(pdf_path)
            )
        # ─────────────────────────────────────────────────────────────────────
 
        logger.info(f"Opening PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        pdf_filename = os.path.basename(pdf_path)
        print(f"\nProcessing PDF: {pdf_filename}")
        print("-" * 50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # ── BATCH PROCESSING WITH CHECKPOINT ──
            # Load checkpoint if exists (resume from last saved page)
            checkpoint = self._load_checkpoint(pdf_path)
            start_page = checkpoint.get('last_page', -1) + 1 if checkpoint else 0
            if start_page > 0:
                logger.info(f"Resuming from page {start_page + 1} (checkpoint found)")
            
            # Restore previously processed results from checkpoint
            if checkpoint and 'results' in checkpoint:
                for page_str, page_data in checkpoint['results'].items():
                    # Convert lists back to sets where needed
                    try:
                        results[int(page_str)] = page_data
                    except Exception:
                        pass
            
            total_pages = len(pdf_document)
            for page_num in range(start_page, total_pages):
                try:
                    logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
                    
                    page = pdf_document[page_num]
 
                    # ── CHANGE 5 (process_pdf loop): same DPI branch as process_pdf_page
                    # Diagram path: UNCHANGED ✓  (300/72)
                    # Table path: REVISED from 150/72 to 300/72 (see process_pdf_page)
                    if self._current_pdf_type == 'table':
                        dpi_factor = 300 / 72
                    else:
                        dpi_factor = 300 / 72
 
                    extract_result = None
                    if getattr(self, '_current_pdf_nature', 'scanned') == 'digital':
                        page_dict = page.get_text("dict")
                        blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
                        num_blocks = len(blocks)
                        num_lines = sum(len(block.get("lines", [])) for block in blocks if isinstance(block, dict))
                        words = page.get_text("words")
                        num_words = len(words) if words else 0
                        tables_detected = sum(
                            1 for block in blocks
                            if isinstance(block, dict) and 'table' in str(block.get('text', '')).lower()
                        )
                        sample_texts = [str(w[4]).strip() for w in words if str(w[4]).strip()][:20] if words else []
                        logger.info(
                            "process_pdf page %d DIGITAL diagnostics: blocks=%d, words=%d, lines=%d, tables=%d, sample_texts=%s",
                            page_num + 1,
                            num_blocks,
                            num_words,
                            num_lines,
                            tables_detected,
                            sample_texts
                        )
                        logger.info(
                            f"process_pdf page {page_num + 1}: DIGITAL mode — attempting "
                            f"text-based extraction first (dpi_factor={dpi_factor:.4f})"
                        )
                        try:
                            extract_result = self.extract_from_text_page(page, dpi_factor=dpi_factor)
                        except Exception as text_err:
                            logger.warning(
                                "Digital text extraction failed for page %s: %s — falling back to OCR",
                                page_num + 1,
                                text_err
                            )
                            extract_result = None
 
                        if isinstance(extract_result, tuple) and len(extract_result) == 9:
                            tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info, all_ocr_tags = extract_result
                            if (not tags or len(tags) < 3) and (not jb_identifiers) and (not mc_identifiers):
                                logger.info(
                                    "Digital extraction produced insufficient content for page %s — falling back to OCR",
                                    page_num + 1
                                )
                                logger.info(
                                    "Digital extraction failure reason: tags=%d, jb_identifiers=%d, mc_identifiers=%d",
                                    len(tags),
                                    len(jb_identifiers),
                                    len(mc_identifiers)
                                )
                                logger.info(
                                    "Digital insufficient condition triggered: (not tags or len(tags) < 3) and not jb_identifiers and not mc_identifiers"
                                )
                                extract_result = None
                            else:
                                logger.info(
                                    "✅ Digital extraction ACCEPTED for page %s: tags=%d, jb_identifiers=%d, mc_identifiers=%d, spare_identifiers=%d, all_ocr_tags=%d",
                                    page_num + 1,
                                    len(tags),
                                    len(jb_identifiers),
                                    len(mc_identifiers),
                                    len(spare_identifiers),
                                    len(all_ocr_tags)
                                )
                                logger.info(
                                    "   Digital tags found: %s",
                                    sorted(list(tags)) if tags else "[]"
                                )
                                logger.info(
                                    "   Digital JB identifiers: %s",
                                    list(jb_identifiers) if jb_identifiers else "[]"
                                )
                                logger.info(
                                    "   Digital MC identifiers: %s",
                                    list(mc_identifiers) if mc_identifiers else "[]"
                                )
                        else:
                            logger.info(
                                "Digital extraction returned invalid result structure for page %s: %s — falling back to OCR",
                                page_num + 1,
                                type(extract_result).__name__ if extract_result is not None else 'None'
                            )
                            extract_result = None
 
                    if extract_result is None:
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor), colorspace=_CS_GRAY)
                        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                        pix.save(image_path)
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.error(f"Failed to load image for page {page_num + 1}")
                            continue

                        # TABLE-MODE: use multi-pass extraction (S13 strategy)
                        if self._current_pdf_type == 'table':
                            # ── STEP 1: Try digital text extraction for JB/MC labels ──
                            # Additional method — not mandatory. Runs alongside OCR.
                            _prev_jb = getattr(self, '_table_last_valid_jb', None)
                            _digital_jbs2, _digital_mcs2 = set(), set()
                            try:
                                _digital_jbs2, _digital_mcs2 = self._extract_jb_mc_from_digital_header(page, page_num)
                                if _digital_jbs2:
                                    if not hasattr(self, '_table_known_jbs'):
                                        self._table_known_jbs = set()
                                    self._table_known_jbs.update(_digital_jbs2)
                                    self._table_last_valid_jb = max(_digital_jbs2, key=len)
                                    logger.info(f"process_pdf page {page_num + 1}: digital header found JBs: {_digital_jbs2}")
                                if _digital_mcs2:
                                    if not hasattr(self, '_table_known_mcs'):
                                        self._table_known_mcs = set()
                                    self._table_known_mcs.update(_digital_mcs2)
                                    self._table_last_valid_mc = max(_digital_mcs2, key=len)
                                    logger.info(f"process_pdf page {page_num + 1}: digital header found MCs: {_digital_mcs2}")
                            except Exception as dh_err2:
                                logger.warning(f"process_pdf page {page_num + 1}: digital header extraction failed: {dh_err2}")

                            # ── STEP 2: Run OCR ONLY if digital found nothing ──
                            # If digital already found JB/MC, skip OCR (saves time).
                            if not _digital_jbs2 and not _digital_mcs2:
                                if _prev_jb:
                                    logger.info(f"process_pdf page {page_num + 1}: skipping header OCR — reusing JB '{_prev_jb}' from previous page (digital found nothing)")
                                    if not hasattr(self, '_table_known_jbs') or not self._table_known_jbs:
                                        self._table_known_jbs = {_prev_jb}
                                    if not hasattr(self, '_table_known_mcs') or not self._table_known_mcs:
                                        mc_prefix = self._get_mc_prefix()
                                        if mc_prefix:
                                            mc_val = mc_prefix + '-' + _prev_jb if not _prev_jb.startswith(mc_prefix) else _prev_jb
                                            self._table_known_mcs = {mc_val}
                                else:
                                    try:
                                        self._extract_header_references_only(page, page_num)
                                    except Exception as hdr_err:
                                        logger.warning(f"process_pdf page {page_num + 1}: header reference extraction failed: {hdr_err}")
                            logger.info(f"process_pdf page {page_num + 1}: TABLE mode — using multi-pass OCR (S13)")
                            extract_result = self._extract_from_image_table_multipass(page, temp_dir, page_num)
                        else:
                            extract_result = self.extract_from_image(image)

                    # ── TABLE-MODE ENHANCEMENT (process_pdf path) ──────────────────
                    # Only run header enhancement if the full-page OCR pass
                    # returned no JB or MC identifiers (see process_pdf_page
                    # for the rationale).
                    if self._current_pdf_type == 'table':
                        try:
                            # Check if full-page OCR found any JB/MC
                            _has_jb = (
                                isinstance(extract_result, tuple) and len(extract_result) >= 9
                                and extract_result[1]  # jb_identifiers
                            )
                            _has_mc = (
                                isinstance(extract_result, tuple) and len(extract_result) >= 9
                                and extract_result[2]  # mc_identifiers
                            )
                            # PERFORMANCE FIX: Also skip if we have JB from previous page
                            _has_prev_jb = bool(getattr(self, '_table_last_valid_jb', None))
                            if not _has_jb and not _has_mc and not _has_prev_jb:
                                enhanced = self._enhance_table_with_header_ocr(page, image, extract_result)
                                if isinstance(enhanced, tuple) and len(enhanced) == 9:
                                    extract_result = enhanced
                                    logger.info("   \u2705 Table-mode header enhancement applied to page %d", page_num + 1)
                        except Exception as enh_err:
                            logger.warning("Table-mode header enhancement failed for page %d: %s", page_num + 1, enh_err)
                        # ── TABLE-MODE POST-PROCESSING (always run for table PDFs) ──
                        try:
                            if isinstance(extract_result, tuple) and len(extract_result) == 9:
                                (_t, _j, _m, _c, _s, _tn, _rc, _tm, _o) = extract_result
                                pp_result = self._post_process_table_extractions(
                                    _t, _j, _m, _c, _s, _tn, _rc, _tm, _o
                                )
                                if isinstance(pp_result, tuple) and len(pp_result) == 9:
                                    extract_result = pp_result
                                    logger.info(
                                        "   \u2705 Table-mode post-process page %d: tags=%d, JBs=%d, MCs=%d, cables=%d",
                                        page_num + 1, len(pp_result[0]), len(pp_result[1]),
                                        len(pp_result[2]), len(pp_result[3])
                                    )
                        except Exception as pp_err:
                            logger.warning("Table-mode post-process failed for page %d: %s", page_num + 1, pp_err)
                    # ────────────────────────────────────────────────────────────────

                    logger.info(f"   \U0001f4ca extract_result length: {len(extract_result)}")
                    if len(extract_result) >= 9:
                        logger.info(f"   📊 all_ocr_tags at index 8: {extract_result[8]}")
                    
                    if len(extract_result) != 9:
                        logger.error(f"❌ Expected 9 values, got {len(extract_result)}")
                        continue
                    
                    (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
                    spare_identifiers, tag_to_number, raw_cable_descriptions, 
                    tag_match_info, all_ocr_tags) = extract_result
                    
                    logger.info(f"✅ Page {page_num + 1}: {len(tags)} matched, {len(all_ocr_tags)} OCR tags")
                    
                    results[page_num + 1] = extract_result
                    
                    # ── SAVE CHECKPOINT EVERY BATCH_SIZE PAGES ──
                    # This allows resuming if worker dies (OOM/timeout)
                    if (page_num + 1) % self.BATCH_SIZE == 0 or page_num == total_pages - 1:
                        self._save_checkpoint(pdf_path, results, page_num)
                    
                    # ── MEMORY CLEANUP AFTER EACH PAGE ──
                    
                    # ── MEMORY CLEANUP AFTER EACH PAGE ──
                    # For 400+ page PDFs, memory accumulates from:
                    # - cv2 image buffers (not garbage collected immediately)
                    # - pdfplumber page objects
                    # - Tesseract subprocess overhead
                    # Force cleanup to prevent OOM/SIGKILL
                    del extract_result
                    if 'image' in dir():
                        del image
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    # Still cleanup on error
                    gc.collect()
                    continue
            
            # Clear checkpoint after successful completion
            self._clear_checkpoint(pdf_path)
            return results

    def process_multiple_pdfs(self, pdf_paths: 'List[str]') -> 'Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str]]]':
        """
        Process multiple PDF files with improved memory management and page numbering
        """
        combined_results = {}
        global_page_number = 1
        
        logger.info(f"Processing {len(pdf_paths)} PDF files")
        
        for pdf_idx, pdf_path in enumerate(pdf_paths):
            try:
                logger.info(f"Processing PDF {pdf_idx + 1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
                
                # Process individual PDF
                pdf_result = self.process_pdf(pdf_path)
                
                if pdf_result:
                    # Add results with global page numbering
                    for local_page_num, page_data in pdf_result.items():
                        combined_results[global_page_number] = page_data
                        global_page_number += 1
                        
                    logger.info(f"Added {len(pdf_result)} pages from {os.path.basename(pdf_path)}")
                    
                    # Force memory cleanup between PDFs
                    gc.collect()
                    
                else:
                    logger.warning(f"No results from PDF: {pdf_path}")
                    
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")
                logger.error(traceback.format_exc())
                # Continue with next PDF
                continue
        
        logger.info(f"Combined results: {len(combined_results)} total pages from {len(pdf_paths)} PDFs")
        return combined_results
        
    def calculate_vector_similarity(self, vec1: 'List[float]', vec2: 'List[float]') -> float:
            """Calculate improved similarity between two vectors"""
            if len(vec1) != len(vec2):
                return 0.0
            
            try:
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = math.sqrt(sum(a * a for a in vec1))
                norm2 = math.sqrt(sum(b * b for b in vec2))
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                    
                cosine_sim = dot_product / (norm1 * norm2)
                
               
                length_sim = min(vec1[0], vec2[0]) / max(vec1[0], vec2[0])
                
                prefix_indices = range(5, 20)  
                prefix_match = sum(1 for i in prefix_indices if vec1[i] > 0 and vec2[i] > 0)
                prefix_sim = prefix_match / sum(1 for i in prefix_indices if vec1[i] > 0 or vec2[i] > 0) if prefix_match else 0
                
                
                combined_sim = (
                    0.4 * cosine_sim +   
                    0.3 * length_sim +    
                    0.3 * prefix_sim    
                )
                
                return combined_sim
                
            except Exception as e:
                logger.error(f"Error in similarity calculation: {e}")
                return 0.0

    def run(self, pdf_paths: 'List[str]', excel_path: str, output_excel_path: str, intermediate_excel_path: str, all_ocr_tags:'Set[str]') -> 'Tuple[List[str], List[str], List[str]]':
        """
        Run the complete process with parallel processing support.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path
            output_excel_path: Output Excel file path
            intermediate_excel_path: Path for intermediate Excel file
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # Build tag pattern from Excel first
        self.build_tag_vectors_from_excel(excel_path)
        logger.info(f"Using tag pattern: {self.tag_patterns}")
        
        # Process all PDF files in parallel
        logger.info(f"Processing {len(pdf_paths)} PDF files")
        pdf_results = self.process_multiple_pdfs(pdf_paths)
        
        # Create tag to JB mapping
        tag_to_jb = self.create_tag_jb_mapping(pdf_results)
        
        # Print total statistics
        total_tags = sum(len(tags) for tags, _ in pdf_results.values())
        total_jbs = sum(len(jbs) for _, jbs in pdf_results.values())
        print("\nTotal Statistics:")
        print("-" * 50)
        print(f"Total Tags found: {total_tags}")
        print(f"Total JB identifiers found: {total_jbs}")
        print(f"Total Pages processed: {len(pdf_results)}")
        
        # Process Excel in parallel - FIX: Pass excel_path instead of pdf_paths
        final_df, unmatched_io_tags, unmatched_tags = self.process_excel_with_io_list(
            intermediate_excel_path, 
            excel_path,  # FIXED: Use excel_path instead of pdf_paths
            output_excel_path,
            all_ocr_tags)
        
        # Save updated Excel
        final_df.to_excel(output_excel_path, index=False)
        logger.info(f"Updated Excel saved to: {output_excel_path}")
        
        return unmatched_io_tags, unmatched_tags

    def draw_bounding_boxes(self, image, tags=None, jb_identifiers=None, mc_identifiers=None,
                        cable_descriptions=None, spare_identifiers=None, tag_to_number=None,
                        tag_match_info=None, all_ocr_tags=None ):
        """
        ✅ بازنویسی کامل: رسم باندینگ باکس‌ها با شماره‌های صحیح (بر اساس موقعیت عمودی)
        """
        # مقداردهی اولیه
        if tags is None:
            tags = set()
        if jb_identifiers is None:
            jb_identifiers = set()
        if mc_identifiers is None:
            mc_identifiers = set()
        if cable_descriptions is None:
            cable_descriptions = []
        if spare_identifiers is None:
            spare_identifiers = []
        if tag_to_number is None:
            tag_to_number = {}
        if tag_match_info is None:
            tag_match_info = {}
        
        # اطمینان از تنظیم الگوها
        # Do NOT default to 'JB'/'MC'/'SPARE' — leave as None if not set by user
        if not hasattr(self, 'jb_examples'):
            self.jb_examples = None
        if not hasattr(self, 'mc_examples'):
            self.mc_examples = None
        if not hasattr(self, 'spare_examples'):
            self.spare_examples = None

        raw_mc_count = len(mc_identifiers) if mc_identifiers else 0
        selected_mc = self._select_best_mc_identifier(mc_identifiers, jb_identifiers)
        mc_identifiers = {selected_mc} if selected_mc else set()

        logger.info(f"="*70)
        logger.info(f"🎨 Drawing bounding boxes with POSITION-BASED numbering")
        logger.info(
            f"  Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {raw_mc_count} (selected: {selected_mc or '-'})"
            f", SPAREs: {len(spare_identifiers)}"
        )
        logger.info(f"  tag_to_number entries: {len(tag_to_number)}")
        logger.info(f"="*70)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # OCR config — NO whitelist (same fix as table mode)
        # The old whitelist was causing tags to be missed because Tesseract
        # couldn't recognize certain characters.
        custom_config = r'--oem 1 --psm 11 -l eng'
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        # جمع‌آوری تمام موارد با موقعیت‌ها
        all_found_items = []
        processed_regions = set()

        def bbox_to_region(bbox):
            """
            ✅ [COORD FIX] تبدیل bbox به مختصات پیکسلی روی تصویر رستر شده.
            اگر bbox از استخراج دیجیتال (PDF point space) آمده باشد،
            مختصات را با dpi_factor ضرب می‌کنیم تا به پیکسل تبدیل شوند.
            اگر از OCR آمده باشد (که از قبل پیکسل است)، بدون تغییر برمی‌گردد.
            """
            if not bbox:
                return None
            try:
                source = bbox.get('coord_source', 'ocr')
                factor = bbox.get('dpi_factor', 300 / 72)

                x = int(bbox.get('x', 0))
                y = int(bbox.get('y', 0))
                w = int(bbox.get('width', 0))
                h = int(bbox.get('height', 0))

                if source == 'digital':
                    raw = (x, y, w, h)
                    # Scale from PDF points to pixels using the ACTUAL rendering DPI
                    # (not the extraction DPI, which may differ for large PDFs)
                    render_dpi = getattr(self, '_current_render_dpi', 300)
                    scale = render_dpi / 72.0  # PDF points → render pixels
                    x = int(x * scale)
                    y = int(y * scale)
                    w = max(1, int(w * scale))
                    h = max(1, int(h * scale))
                    logger.debug(
                        "bbox_to_region: DIGITAL→PIXEL  render_dpi=%d  scale=%.4f  raw=%s  pixel=(%d,%d,%d,%d)",
                        render_dpi, scale, raw, x, y, w, h
                    )
                else:
                    logger.debug(
                        "bbox_to_region: OCR (already pixel)  pixel=(%d,%d,%d,%d)",
                        x, y, w, h
                    )

                return (x, y, w, h)
            except Exception as e:
                logger.error(f"bbox_to_region: error resolving bbox {bbox}: {e}")
                return None
        
        # ============================================================
        # Phase 1: جمع‌آوری Exact Matches
        # ============================================================
        logger.info("Phase 1: Collecting EXACT matches...")
        exact_found_count = 0
        
        for tag in tags:
            # ── FALLBACK: If tag is not in tag_match_info but IS in all_ocr_tags,
            # try to find it via OCR data directly. This happens when tags were
            # matched via fuzzy matching in the post-processor (which doesn't
            # populate tag_match_info with bbox for every tag).
            if tag not in tag_match_info:
                tag_upper = tag.upper()
                found_via_ocr = False
                for i, text in enumerate(ocr_data['text']):
                    text_clean = text.strip().upper()
                    if text_clean == tag_upper:
                        region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                    ocr_data['width'][i], ocr_data['height'][i])
                        if region_key not in processed_regions:
                            all_found_items.append({
                                'type': 'tag',
                                'text': tag,
                                'position': region_key,
                                'match_type': 'exact',
                                'score': 1.0,
                                'y_position': ocr_data['top'][i]
                            })
                            processed_regions.add(region_key)
                            exact_found_count += 1
                            found_via_ocr = True
                            break
                if found_via_ocr:
                    continue
                logger.warning(f"Tag '{tag}' not in tag_match_info and not found in OCR data")
                continue
            
            info = tag_match_info[tag]
            match_type = info.get('match_type', 'unknown')
            
            if match_type != 'exact':
                continue
            
            tag_upper = tag.upper()
            ocr_text_used = info.get('ocr_text', tag).upper()
            bbox = bbox_to_region(info.get('bbox'))
            if bbox is not None:
                if bbox not in processed_regions:
                    all_found_items.append({
                        'type': 'tag',
                        'text': tag,
                        'position': bbox,
                        'match_type': 'exact',
                        'score': info.get('score', 1.0),
                        'y_position': bbox[1]
                    })
                    processed_regions.add(bbox)
                    exact_found_count += 1
                continue
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                if text_clean == tag_upper or text_clean == ocr_text_used:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'tag',
                            'text': tag,
                            'position': region_key,
                            'match_type': 'exact',
                            'score': info.get('score', 1.0),
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        exact_found_count += 1
                        
        
        logger.info(f"Phase 1: Found {exact_found_count} exact matches")
        
        # ============================================================
        # Phase 2: جمع‌آوری Similar Matches
        # ============================================================
        logger.info("Phase 2: Collecting SIMILAR matches...")
        similar_found_count = 0
        
        for tag in tags:
            if tag not in tag_match_info:
                continue
            
            info = tag_match_info[tag]
            match_type = info.get('match_type', 'unknown')
            
            if match_type != 'similar':
                continue
            
            ocr_text_used = info.get('ocr_text', tag).upper()
            bbox = bbox_to_region(info.get('bbox'))
            if bbox is not None:
                if bbox not in processed_regions:
                    all_found_items.append({
                        'type': 'tag',
                        'text': tag,
                        'position': bbox,
                        'match_type': 'similar',
                        'score': info.get('score', 0.0),
                        'original_text': ocr_text_used,
                        'y_position': bbox[1]
                    })
                    processed_regions.add(bbox)
                    similar_found_count += 1
                continue
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                region_key = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                
                if region_key in processed_regions:
                    continue
                
                if text_clean == ocr_text_used:
                    all_found_items.append({
                        'type': 'tag',
                        'text': tag,
                        'position': region_key,
                        'match_type': 'similar',
                        'score': info.get('score', 0.0),
                        'original_text': text_clean,
                        'y_position': ocr_data['top'][i]
                    })
                    processed_regions.add(region_key)
                    similar_found_count += 1
                    
        
        logger.info(f"Phase 2: Found {similar_found_count} similar matches")
        
        # ============================================================
        # Phase 2.5: Pattern-based unmatched candidates
        # ============================================================
        logger.info("Phase 2.5: Collecting unmatched pattern candidates...")
        unmatched_candidate_found_count = 0
        for info in tag_match_info.values():
            if not isinstance(info, dict):
                continue
            if info.get('match_type') != 'unmatched_candidate':
                continue

            candidate_text = self._normalize_ocr_tag_candidate(
                info.get('ocr_text', info.get('display_text', ''))
            )
            if not candidate_text:
                continue

            bbox = bbox_to_region(info.get('bbox'))
            if bbox is not None:
                if bbox not in processed_regions:
                    all_found_items.append({
                        'type': 'unmatched_candidate',
                        'text': info.get('display_text', candidate_text),
                        'position': bbox,
                        'score': info.get('score', 0.0),
                        'y_position': bbox[1]
                    })
                    processed_regions.add(bbox)
                    unmatched_candidate_found_count += 1
                continue

            for i, text in enumerate(ocr_data['text']):
                text_clean = self._normalize_ocr_tag_candidate(text)
                if not text_clean:
                    continue

                region_key = (
                    ocr_data['left'][i], ocr_data['top'][i],
                    ocr_data['width'][i], ocr_data['height'][i]
                )
                if region_key in processed_regions:
                    continue

                if text_clean == candidate_text:
                    all_found_items.append({
                        'type': 'unmatched_candidate',
                        'text': info.get('display_text', candidate_text),
                        'position': region_key,
                        'score': info.get('score', 0.0),
                        'y_position': ocr_data['top'][i]
                    })
                    processed_regions.add(region_key)
                    unmatched_candidate_found_count += 1
                    break

        logger.info(f"Phase 2.5: Found {unmatched_candidate_found_count} unmatched candidates")

        # ============================================================
        # Phase 3: JB identifiers
        # ============================================================
        logger.info("Phase 3: Collecting JB identifiers...")
        jb_found_count = 0
        
        for jb in jb_identifiers:
            jb_bbox = bbox_to_region(tag_match_info.get(jb, {}).get('bbox'))
            if jb_bbox is not None and jb_bbox not in processed_regions:
                all_found_items.append({
                    'type': 'jb',
                    'text': jb,
                    'position': jb_bbox,
                    'y_position': jb_bbox[1]
                })
                processed_regions.add(jb_bbox)
                jb_found_count += 1
                continue

            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == jb.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'jb',
                            'text': jb,
                            'position': region_key,
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        jb_found_count += 1
                        break
        
        logger.info(f"Phase 3 complete: Found {jb_found_count} JBs")
        
        # ============================================================
        # Phase 4: MC identifiers
        # ============================================================
        logger.info("Phase 4: Collecting MC identifiers...")
        mc_found_count = 0
        
        for mc in mc_identifiers:
            mc_bbox = bbox_to_region(tag_match_info.get(mc, {}).get('bbox'))
            if mc_bbox is not None and mc_bbox not in processed_regions:
                all_found_items.append({
                    'type': 'mc',
                    'text': mc,
                    'position': mc_bbox,
                    'y_position': mc_bbox[1]
                })
                processed_regions.add(mc_bbox)
                mc_found_count += 1
                continue

            for i, text in enumerate(ocr_data['text']):
                text_norm = self._normalize_code_token(text)
                if text_norm and text_norm == self._normalize_code_token(mc):
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'mc',
                            'text': mc,
                            'position': region_key,
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        mc_found_count += 1
                        break
        
        logger.info(f"Phase 4: Found {mc_found_count} MCs")
        
        # ============================================================
        # Phase 5: SPARE identifiers
        # ============================================================
        logger.info(f"Phase 5: Collecting SPARE identifiers...")
        spare_found_count = 0
        spare_pattern = re.compile(rf'\b{re.escape(self.spare_examples)}\b', re.IGNORECASE)
        
        for spare_idx, spare in enumerate(spare_identifiers):
            spare_id = f"{self.spare_examples}_{spare_idx + 1}"
            spare_bbox = bbox_to_region(tag_match_info.get(spare_id, {}).get('bbox'))
            if spare_bbox is not None and spare_bbox not in processed_regions:
                all_found_items.append({
                    'type': 'spare',
                    'text': spare,
                    'position': spare_bbox,
                    'id': spare_id,
                    'y_position': spare_bbox[1]
                })
                processed_regions.add(spare_bbox)
                spare_found_count += 1
                continue
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                if spare_pattern.search(text_clean):
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'spare',
                            'text': spare,
                            'position': region_key,
                            'id': spare_id,
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        spare_found_count += 1
                        break
        
        logger.info(f"Phase 5: Found {spare_found_count} SPAREs")
        
        # ============================================================
        # Phase 6: Cable descriptions
        # ============================================================
        logger.info("Phase 6: Collecting cable descriptions...")
        cable_found_count = 0
        _mc_prefix = self._get_mc_prefix()
        
        for cable_desc in cable_descriptions:
            cable_bbox = None
            for info in tag_match_info.values():
                if isinstance(info, dict) and info.get('match_type') == 'cable' and info.get('display_text') == cable_desc:
                    cable_bbox = bbox_to_region(info.get('bbox'))
                    break
            if cable_bbox is not None and cable_bbox not in processed_regions:
                all_found_items.append({
                    'type': 'cable',
                    'text': cable_desc,
                    'position': cable_bbox,
                    'y_position': cable_bbox[1]
                })
                processed_regions.add(cable_bbox)
                cable_found_count += 1
                continue

            # ── UNIVERSAL FIX: Search for the cable code in OCR tokens ──
            # This works for BOTH diagram and table mode. The cable code
            # (e.g. NC-0-1-2-C-3-BL) may be stored as a single OCR token
            # or split across multiple tokens. We try:
            #   1. Exact match (full cable code = single token)
            #   2. Substring match (cable code is part of a token)
            #   3. Multi-token match (cable code spans multiple adjacent tokens)
            # The bounding box is drawn using the EXACT OCR token coordinates,
            # so it's tight around the text — not on the MC row.
            cable_desc_upper = cable_desc.upper().strip()
            found = False
            
            # 1. Exact match of the full cable code
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == cable_desc_upper:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'cable',
                            'text': cable_desc,
                            'position': region_key,
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        cable_found_count += 1
                        found = True
                        break
            
            if found:
                continue
            
            # 2. Substring match (cable code is part of a larger token)
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if cable_desc_upper in text_clean and len(text_clean) < len(cable_desc_upper) + 10:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'cable',
                            'text': cable_desc,
                            'position': region_key,
                            'y_position': ocr_data['top'][i]
                        })
                        processed_regions.add(region_key)
                        cable_found_count += 1
                        found = True
                        break
            
            if found:
                continue
            
            # 3. Multi-token match: cable code is split across adjacent tokens
            # e.g. "NC-0-1-2" + "-" + "C-3-BL" → combine and check
            # We look for consecutive tokens on the same line that together
            # form the cable code.
            cable_parts = cable_desc_upper.split('-')
            if len(cable_parts) >= 4:
                # Try to find the first part and then check if subsequent
                # tokens on the same line complete the cable code
                for i, text in enumerate(ocr_data['text']):
                    text_clean = text.strip().upper()
                    # Skip MC tokens
                    if any(text_clean.startswith(p + '-') for p in _mc_prefix.split(',') if p):
                        continue
                    
                    # Check if this token starts the cable code
                    if not cable_desc_upper.startswith(text_clean):
                        # Maybe this token contains the start of the cable code
                        if cable_desc_upper.split('-')[0] in text_clean:
                            pass  # potential start
                        else:
                            continue
                    
                    # Look at the next few tokens on the same line
                    combined = text_clean
                    min_x = int(ocr_data['left'][i])
                    min_y = int(ocr_data['top'][i])
                    max_x = int(ocr_data['left'][i]) + int(ocr_data['width'][i])
                    max_y = int(ocr_data['top'][i]) + int(ocr_data['height'][i])
                    y_ref = int(ocr_data['top'][i])
                    
                    for j in range(i + 1, min(i + 8, len(ocr_data['text']))):
                        next_text = str(ocr_data['text'][j]).strip().upper()
                        if not next_text:
                            continue
                        next_y = int(ocr_data['top'][j])
                        # Must be on same line (within 10px)
                        if abs(next_y - y_ref) > 10:
                            break
                        
                        combined += next_text
                        nx = int(ocr_data['left'][j])
                        nw = int(ocr_data['width'][j])
                        max_x = max(max_x, nx + nw)
                        max_y = max(max_y, int(ocr_data['top'][j]) + int(ocr_data['height'][j]))
                        
                        # Check if combined text now contains the cable code
                        if cable_desc_upper in combined:
                            # Found it! Create a merged bounding box
                            region_key = (min_x, min_y, max_x - min_x, max_y - min_y)
                            if region_key not in processed_regions:
                                all_found_items.append({
                                    'type': 'cable',
                                    'text': cable_desc,
                                    'position': region_key,
                                    'y_position': min_y
                                })
                                processed_regions.add(region_key)
                                cable_found_count += 1
                                found = True
                            break
                    
                    if found:
                        break
            
            if found:
                continue
            
            # 4. Last resort: search for distinctive middle part
            # (skip MC tokens to avoid drawing on MC row)
            if len(cable_parts) >= 5:
                search_pattern = '-'.join(cable_parts[2:5])
                for i, text in enumerate(ocr_data['text']):
                    text_clean = text.strip().upper()
                    if any(text_clean.startswith(p + '-') for p in _mc_prefix.split(',') if p):
                        continue
                    if search_pattern in text_clean:
                        region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                    ocr_data['width'][i], ocr_data['height'][i])
                        if region_key not in processed_regions:
                            all_found_items.append({
                                'type': 'cable',
                                'text': cable_desc,
                                'position': region_key,
                                'y_position': ocr_data['top'][i]
                            })
                            processed_regions.add(region_key)
                            cable_found_count += 1
                            found = True
                            break
        
        logger.info(f"Phase 6: Found {cable_found_count} cables")
        
        # ============================================================
        # رسم bounding boxes
        # ============================================================
        logger.info(f"Drawing all found items...")
        
        for item in all_found_items:
            x, y, w, h = item['position']
            item_type = item['type']
            text = item['text']
            
            if item_type == 'tag':
                cleaned_text = self.clean_text_for_display(text)
                match_type = item.get('match_type', 'unknown')
                score = item.get('score', 0.0)
                
                tag_number = tag_to_number.get(text, 0)

                if not tag_number:
                    logger.warning(f"⚠️ Tag '{text}' has no number in tag_to_number — drawing unnumbered tag")

                    # Draw a distinct unnumbered tag box so the reviewer can see
                    # what was detected in the digital path even if numbering wasn't assigned.
                    color = (50, 50, 200)  # muted blue for unnumbered
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                    # Label includes match type and optional score to aid review
                    if match_type == 'exact':
                        label_prefix = '✓'
                    elif match_type == 'similar':
                        label_prefix = '≈'
                    else:
                        label_prefix = '?'

                    if score and score > 0:
                        label = f"{label_prefix} {cleaned_text} ({score:.2f})"
                    else:
                        label = f"{label_prefix} {cleaned_text}"

                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    continue
                
                # رنگ‌بندی بر اساس match type
                if match_type == 'exact':
                    color = (255, 0, 0)      # سبز
                    label_prefix = "✓"
                elif match_type == 'similar':
                    color = (0, 165, 255)    # نارنجی
                    label_prefix = "≈"
                else:
                    color = (128, 128, 128)  # خاکستری
                    label_prefix = "?"
                
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                if match_type == 'similar' and score > 0:
                    label = f"#{tag_number} {cleaned_text} ({score:.2f})"
                else:
                    label = f"#{tag_number} {cleaned_text}"
                
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            elif item_type == 'jb':
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # آبی
                cv2.putText(image, f"JB: {text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            elif item_type == 'mc':
                cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)  # آبی روشن
                cv2.putText(image, f"MC: {text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 255), 2)
                
            elif item_type == 'spare':
                spare_id = item['id']
                spare_number = tag_to_number.get(spare_id, 0)
                
                if not spare_number:
                    logger.warning(f"⚠️ SPARE '{spare_id}' has no number in tag_to_number — drawing unnumbered SPARE")
                    cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)
                    cv2.putText(image, f"SPARE", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
                    continue
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)  # بنفش
                cv2.putText(image, f"SPARE #{spare_number}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
                
            elif item_type == 'cable':
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 200), 2)  # زرد
                cv2.putText(image, f"Cable: {text}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

            elif item_type == 'unmatched_candidate':
                # RED bounding box for tags that match IO pattern but are
                # NOT in the IO List — these are potential errors or
                # missing tags that need engineering review.
                candidate_text = self.clean_text_for_display(text)
                score = item.get('score', 0.0)
                color = (0, 0, 255)  # RED (BGR) — clearly visible error indicator
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)  # thicker border (3px)
                label = f"⚠ {candidate_text}"
                if score:
                    label = f"⚠ {candidate_text} ({score:.2f})"
                cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # ============================================================
        # آمار و Legend
        # ============================================================
        exact_count = len([item for item in all_found_items if item.get('type') == 'tag' and item.get('match_type') == 'exact'])
        similar_count = len([item for item in all_found_items if item.get('type') == 'tag' and item.get('match_type') == 'similar'])
        spare_count = len([item for item in all_found_items if item.get('type') == 'spare'])
        candidate_count = len([item for item in all_found_items if item.get('type') == 'unmatched_candidate'])
        
        legend_y_pos = image.shape[0] - 100
        legend_x_pos = 10
        
        # Legend header
        cv2.putText(image, "Legend (Ordered by Y-position):", (legend_x_pos, legend_y_pos - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Match types
        cv2.putText(image, f"Exact: {exact_count}", (legend_x_pos, legend_y_pos - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Similar: {similar_count}", (legend_x_pos + 150, legend_y_pos - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(image, f"Not in IO (Pattern): {candidate_count}", (legend_x_pos + 300, legend_y_pos - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Components
        cv2.putText(image, f"JB: {jb_found_count}", (legend_x_pos, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"MC: {mc_found_count}", (legend_x_pos + 100, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"SPARE: {spare_count}", (legend_x_pos + 200, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
        cv2.putText(image, f"Cable: {cable_found_count}", (legend_x_pos + 330, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        # Stats summary
        stats_text = (
            f"Total: {exact_count + similar_count} tags, "
            f"{candidate_count} pattern-candidates, {spare_count} spares"
        )
        cv2.putText(image, stats_text, (legend_x_pos, legend_y_pos + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        logger.info(f"✅ Bounding boxes drawn with position-based numbering:")
        logger.info(f"   Tags: {exact_count} exact, {similar_count} similar")
        logger.info(f"   Pattern-based unmatched candidates: {candidate_count}")
        logger.info(f"   Components: {jb_found_count} JBs, {mc_found_count} MCs, {spare_count} SPAREs")
        logger.info(f"="*70)
        
        return image, tag_to_number

    
    def add_tag_numbers_to_dataframe(self, df: pd.DataFrame, tag_to_number: 'Dict[str, int]') -> pd.DataFrame:
        """
        Add tag numbers to the dataframe based on the mapping from draw_bounding_boxes.
        
        Args:
            df: Input DataFrame containing tag information
            tag_to_number: Dictionary mapping tags to their assigned numbers
            
        Returns:
            Updated DataFrame with tag numbers
        """
        # Add new column for tag numbers
        df['Tag_Number'] = None
        
        # Assign tag numbers based on the mapping
        for idx, row in df.iterrows():
            tag = str(row['Tag No']).strip().upper()
            if tag in tag_to_number:
                df.at[idx, 'Tag_Number'] = tag_to_number[tag]
        
        return df

    def create_annotated_pdf(self, pdf_path: str, output_pdf_path: str) -> 'Dict[str, int]':
        """
        ✅ بازنویسی: ایجاد PDF حاشیه‌گذاری شده با شماره‌گذاری بر اساس موقعیت
        """
        all_tag_numbers = {}
        pdf_document = None
        new_pdf = None
        
        try:
            logger.info(f"Creating annotated PDF from: {pdf_path}")
            page_results = self.process_pdf(pdf_path)
            pdf_document = fitz.open(pdf_path)
            new_pdf = fitz.open()
            total_pages = len(pdf_document)
            
            # استفاده از DPI پایین‌تر برای حافظه بهتر در PDF های چند صفحه‌ای
            dpi_factor = 200/72 if total_pages > 10 else 300/72
            # Store the rendering DPI for coordinate scaling in draw_bounding_boxes
            render_dpi = 200 if total_pages > 10 else 300
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num in range(total_pages):
                    try:
                        logger.info(f"Annotating page {page_num + 1}/{total_pages}")
                        
                        page = pdf_document[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor), colorspace=_CS_GRAY)
                        
                        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                        pix.save(image_path)
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning(f"Failed to load image for page {page_num + 1}")
                            new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                            pix = None
                            continue
                        
                        # استخراج با شماره‌گذاری بر اساس موقعیت
                        result = page_results.get(page_num + 1)
                        if result is not None and len(result) == 9:
                            logger.info(f"Using precomputed extraction result for page {page_num + 1}")
                        else:
                            logger.info(f"No valid precomputed result for page {page_num + 1}; falling back to image-based extraction")
                            result = self.extract_from_image(image)
                        
                        # ✅ FIX: چک بدون warning
                        if not isinstance(result, tuple) or len(result) != 9:
                            logger.error(f"❌ Expected 9 values, got {len(result) if result is not None else 'None'}")
                            # استفاده از مقادیر پیش‌فرض
                            tags, jb_identifiers, mc_identifiers = set(), set(), set()
                            cable_descriptions, spare_identifiers = [], []
                            tag_to_number, raw_cable_descriptions, tag_match_info = {}, [], {}
                            all_ocr_tags = set()
                        else:
                            # ✅ Unpack عادی
                            (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
                            spare_identifiers, tag_to_number, raw_cable_descriptions, 
                            tag_match_info, all_ocr_tags) = result

                        selected_mc = self._select_best_mc_identifier(mc_identifiers, jb_identifiers)
                        
                        # Set the rendering DPI for coordinate scaling in draw_bounding_boxes
                        self._current_render_dpi = render_dpi
                        
                        # رسم bounding boxes
                        try:
                            annotated_image, page_tag_numbers = self.draw_bounding_boxes(
                                image, tags, jb_identifiers, mc_identifiers,
                                cable_descriptions, spare_identifiers, tag_to_number,
                                tag_match_info
                            )
                            all_tag_numbers.update(page_tag_numbers)
                            
                            logger.info(f"✅ Page {page_num + 1}: Added {len(page_tag_numbers)} numbers (position-based)")
                        except Exception as e:
                            logger.error(f"Error drawing bounding boxes on page {page_num + 1}: {e}")
                            annotated_image = image.copy()
                            page_tag_numbers = tag_to_number
                            continue
                        # Add info overlay
                        try:
                            info_text = [
                                f"Page {page_num + 1}/{total_pages}",
                                f"Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MC: {selected_mc or '-'}",
                                f"Numbered by vertical position (top to bottom)"
                            ]
                            
                            overlay = annotated_image.copy()
                            overlay_h = len(info_text) * 25 + 15
                            x, y, w, h = 5, 5, 450, overlay_h
                            
                            if y + h <= annotated_image.shape[0] and x + w <= annotated_image.shape[1]:
                                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
                                blended = cv2.addWeighted(overlay[y:y+h, x:x+w], 0.6,
                                                        annotated_image[y:y+h, x:x+w], 0.4, 0)
                                annotated_image[y:y+h, x:x+w] = blended
                                
                                for i, text in enumerate(info_text):
                                    cv2.putText(annotated_image, text, (x + 5, y + 20 + i * 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                            
                        except Exception as e:
                            logger.error(f"Error adding overlay to page {page_num + 1}: {e}")
                            
                        # ── 🆕 WARNING BANNERS ──────────────────────────────────────
                        # Draw attention-grabbing banners at the BOTTOM of the page when:
                        #   - JB_NOT_FOUND: page has tags/spares but no JB detected  (RED)
                        #   - MULTIPLE_JB:  page has more than one JB                (ORANGE)
                        #   - DUPLICATE_JB: this page's JB also appears on another page (ORANGE)
                        #   - DUPLICATE_TAG: this page has a tag that also appears on another page (YELLOW)
                        # The operator sees these directly in the annotated PDF — no need
                        # to dig through Excel.
                        # ──────────────────────────────────────────────────────────
                        try:
                            _banners = []  # list of (text, color)

                            # JB not found on this page but tags exist
                            if not jb_identifiers and (tags or spare_identifiers):
                                _banners.append((
                                    f"WARNING: JB NOT FOUND on page {page_num + 1} "
                                    f"— {len(tags)} tag(s) + {len(spare_identifiers)} spare(s) "
                                    f"exported WITHOUT a JB. Check page header!",
                                    (0, 0, 220)  # RED (BGR)
                                ))

                            # Multiple JBs on this page
                            if len(jb_identifiers) > 1:
                                _banners.append((
                                    f"WARNING: MULTIPLE JB DETECTED on page {page_num + 1} "
                                    f"— {sorted(str(j) for j in jb_identifiers)}. "
                                    f"Page was SKIPPED in Excel export.",
                                    (0, 140, 220)  # ORANGE (BGR)
                                ))

                            # Duplicate JB across pages — check page_results
                            if jb_identifiers and page_results:
                                _jb_pages_map = {}
                                for pn, pr in page_results.items():
                                    if not isinstance(pr, (tuple, list)) or len(pr) < 9:
                                        continue
                                    page_jbs = pr[1] or set()
                                    for j in page_jbs:
                                        ju = str(j).strip().upper()
                                        if ju:
                                            _jb_pages_map.setdefault(ju, []).append(int(pn))
                                for jb in jb_identifiers:
                                    ju = str(jb).strip().upper()
                                    other_pages = [p for p in _jb_pages_map.get(ju, []) if p != page_num + 1]
                                    if other_pages:
                                        _banners.append((
                                            f"WARNING: DUPLICATE JB '{jb}' — also appears on page(s) "
                                            f"{sorted(set(other_pages))}.",
                                            (0, 140, 220)  # ORANGE
                                        ))

                            # Duplicate tags across pages — check page_results
                            if tags and page_results:
                                _tag_pages_map = {}
                                for pn, pr in page_results.items():
                                    if not isinstance(pr, (tuple, list)) or len(pr) < 9:
                                        continue
                                    page_tags_set = pr[0] or set()
                                    for tg in page_tags_set:
                                        tu = str(tg).strip().upper()
                                        if tu and 'SPARE' not in tu:
                                            _tag_pages_map.setdefault(tu, []).append(int(pn))
                                for tag in tags:
                                    tu = str(tag).strip().upper()
                                    if not tu or 'SPARE' in tu:
                                        continue
                                    other_pages = [p for p in _tag_pages_map.get(tu, []) if p != page_num + 1]
                                    if other_pages:
                                        _banners.append((
                                            f"INFO: Tag '{tag}' also appears on page(s) "
                                            f"{sorted(set(other_pages))}.",
                                            (0, 200, 220)  # YELLOW-ish (BGR)
                                        ))

                            # Draw the banners (stacked from bottom-up)
                            if _banners:
                                _banner_h = 32
                                _banner_pad = 4
                                _banner_w = min(annotated_image.shape[1] - 10, 1100)
                                _start_y = annotated_image.shape[0] - len(_banners) * (_banner_h + _banner_pad) - 5
                                for bi, (text, color) in enumerate(_banners):
                                    by = _start_y + bi * (_banner_h + _banner_pad)
                                    bx = 5
                                    if by + _banner_h > annotated_image.shape[0]:
                                        break
                                    # Draw filled rectangle
                                    cv2.rectangle(annotated_image, (bx, by), (bx + _banner_w, by + _banner_h),
                                                color, -1)
                                    # Draw text
                                    # Truncate text if too long
                                    _max_chars = _banner_w // 9  # rough estimate
                                    _disp_text = text if len(text) <= _max_chars else text[:_max_chars - 3] + '...'
                                    cv2.putText(annotated_image, _disp_text, (bx + 8, by + 22),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        except Exception as banner_err:
                            logger.error(f"Error drawing warning banners on page {page_num + 1}: {banner_err}")

                        # Save annotated image and add to PDF
                        try:
                            annotated_path = os.path.join(temp_dir, f"annotated_{page_num + 1}.png")
                            cv2.imwrite(annotated_path, annotated_image)
                            
                            new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                            new_page.insert_image(new_page.rect, filename=annotated_path)
                            
                            os.remove(annotated_path)
                        except Exception as e:
                            logger.error(f"Error saving page {page_num + 1}: {e}")
                        
                        # Clean up
                        del image, annotated_image
                        pix = None
                        
                        try:
                            os.remove(image_path)
                        except:
                            pass
                        
                        # Garbage collect every 3 pages
                        if (page_num + 1) % 3 == 0:
                            gc.collect()
                            logger.debug(f"Memory cleanup after page {page_num + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {e}")
                        logger.error(traceback.format_exc())
                        continue
            
            # Save the annotated PDF
            try:
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                new_pdf.save(output_pdf_path)
                logger.info(f"✅ Annotated PDF saved: {output_pdf_path}")
                logger.info(f"   Total pages: {total_pages}")
                logger.info(f"   Total tags numbered (by position): {len(all_tag_numbers)}")
            except Exception as e:
                logger.error(f"Error saving annotated PDF: {e}")
            
            return all_tag_numbers
            
        except Exception as e:
            logger.error(f"Error creating annotated PDF: {e}")
            logger.error(traceback.format_exc())
            return {}
            
        finally:
            if new_pdf:
                try:
                    new_pdf.close()
                except:
                    pass
            if pdf_document:
                try:
                    pdf_document.close()
                except:
                    pass
    def process_unmatched_tags(self, unmatched_pdf_tags: List[str], io_tags: Set[str]) -> Dict[str, str]:
        """
        پردازش و اصلاح تگ‌های موجود در PDF که در IO List وجود ندارند
        
        Args:
            unmatched_pdf_tags: لیست تگ‌های PDF که در IO List نیستند
            io_tags: مجموعه تگ‌های IO List
            
        Returns:
            دیکشنری {تگ اصلی: تگ اصلاح شده} یا {تگ اصلی: تگ اصلی} اگر اصلاح نشود
        """
        corrections = {}
        io_tags_upper = {tag.strip().upper() for tag in io_tags if tag and not pd.isna(tag)}
        
        # تنظیمات fuzzy matching
        fuzzy_threshold = 0.85  # آستانه شباهت برای fuzzy matching
        
        logger.info(f"🔍 Processing {len(unmatched_pdf_tags)} unmatched PDF tags...")
        
        for tag in unmatched_pdf_tags:
            original_tag = tag
            corrected_tag = None
            
            # مرحله 1: اعمال تصحیحات OCR استاندارد
            fixed_tag = self.fix_common_ocr_errors(tag)
            if fixed_tag.upper() in io_tags_upper:
                # تگ اصلاح شده در IO List وجود دارد
                corrected_tag = fixed_tag
                logger.info(f"✅ OCR fix matched: '{original_tag}' -> '{corrected_tag}'")
                corrections[original_tag] = corrected_tag
                continue
            
            # مرحله 2: بررسی الگوهای خاص UZSO/UZSC با شماره‌های متفاوت
            if re.match(r'^[UuVv][ZzSs2][Ss5][O0oCcGg][-_]?\d+$', tag.upper()):
                # استخراج پیشوند و شماره
                prefix_match = re.match(r'^([UuVv][ZzSs2][Ss5][O0oCcGg])[-_]?(\d+)$', tag.upper())
                if prefix_match:
                    prefix = prefix_match.group(1)
                    number = prefix_match.group(2)
                    
                    # تصحیح پیشوند
                    if re.match(r'[UuVv][ZzSs2][Ss5][O0oD]', prefix):
                        corrected_prefix = "UZSO"
                    else:
                        corrected_prefix = "UZSC"
                    
                    # بررسی شماره‌های مشابه در IO List
                    number_pattern = r'^' + corrected_prefix + r'[-_]?(\d+)$'
                    potential_matches = []
                    
                    for io_tag in io_tags_upper:
                        num_match = re.match(number_pattern, io_tag)
                        if num_match:
                            io_number = num_match.group(1)
                            # محاسبه فاصله Levenshtein بین شماره‌ها
                            distance = Levenshtein.distance(number, io_number)
                            if distance <= 2:  # حداکثر 2 کاراکتر اختلاف
                                similarity = 1.0 - (distance / max(len(number), len(io_number)))
                                potential_matches.append((io_tag, similarity))
                    
                    if potential_matches:
                        # انتخاب بهترین تطبیق
                        best_match = max(potential_matches, key=lambda x: x[1])
                        if best_match[1] >= fuzzy_threshold:
                            corrected_tag = best_match[0]
                            logger.info(f"✅ Number correction: '{original_tag}' -> '{corrected_tag}' (score: {best_match[1]:.3f})")
                            corrections[original_tag] = corrected_tag
                            continue
            
            # مرحله 3: جستجوی fuzzy match در کل IO List
            best_match = None
            best_score = 0
            
            for io_tag in io_tags_upper:
                # محاسبه شباهت با استفاده از Levenshtein distance
                distance = Levenshtein.distance(fixed_tag.upper(), io_tag)
                max_len = max(len(fixed_tag), len(io_tag))
                similarity = 1.0 - (distance / max_len) if max_len > 0 else 0
                
                if similarity > best_score and similarity >= fuzzy_threshold:
                    best_score = similarity
                    best_match = io_tag
            
            if best_match:
                corrected_tag = best_match
                logger.info(f"✅ Fuzzy match: '{original_tag}' -> '{corrected_tag}' (score: {best_score:.3f})")
                corrections[original_tag] = corrected_tag
                continue
            
            # اگر هیچ اصلاحی انجام نشد، تگ اصلی را برگردان
            corrections[original_tag] = original_tag
            logger.debug(f"⚠️ No correction found for: '{original_tag}'")
        
        # آمار اصلاحات
        corrected_count = sum(1 for k, v in corrections.items() if k != v)
        logger.info(f"✅ Corrected {corrected_count}/{len(unmatched_pdf_tags)} unmatched tags")
        
        return corrections

    def _create_unmatched_tags_excel(self, unmatched_pdf_tags: 'List[str]', 
                                    unmatched_io_tags: 'List[str]', 
                                    output_path: str):
        """
        ایجاد فایل اکسل برای تگ‌های تطبیق نیافته با فرمت بهتر
        """
        try:
            logger.info("="*70)
            logger.info("📝 Creating Unmatched Tags Excel Report")
            logger.info("="*70)
            
            # بررسی مسیر
            if not output_path or not output_path.strip():
                output_path = "unmatched_tags.xlsx"
            
            if not output_path.endswith('.xlsx'):
                output_path = output_path + '.xlsx'
            
            # اطمینان از وجود دایرکتوری
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # ایجاد داده‌ها
            data = []
            
            # تگ‌های PDF که در IO نیستند
            for tag in sorted(unmatched_pdf_tags):
                data.append({
                    'Tag': tag,
                    'Source': 'PDF',
                    'Status': 'Not found in IO List',
                    'Severity': 'WARNING',
                    'Action': 'Add to IO List or verify tag correctness'
                })
            
            # تگ‌های IO که در PDF نیستند
            for tag in sorted(unmatched_io_tags):
                data.append({
                    'Tag': tag,
                    'Source': 'IO List',
                    'Status': 'Not found in PDF',
                    'Severity': 'INFO',
                    'Action': 'Check if tag should be in PDF'
                })
            
            # ایجاد دیتافریم
            if data:
                unmatched_df = pd.DataFrame(data)
                logger.info(f"✅ Created unmatched report with {len(data)} rows")
                logger.info(f"   - PDF tags not in IO: {len(unmatched_pdf_tags)}")
                logger.info(f"   - IO tags not in PDF: {len(unmatched_io_tags)}")
            else:
                # اگر همه تطبیق داشتند
                unmatched_df = pd.DataFrame([{
                    'Tag': 'N/A',
                    'Source': 'N/A',
                    'Status': '✅ All tags matched successfully!',
                    'Severity': 'SUCCESS',
                    'Action': 'No action needed'
                }])
                logger.info("✅ All tags matched! Creating success report.")
            
            # ذخیره
            unmatched_df.to_excel(output_path, index=False)
            logger.info(f"✅ Unmatched tags Excel saved to: {output_path}")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"❌ Error creating unmatched tags Excel: {e}")
            logger.error(traceback.format_exc())
            
    def run_with_annotated_pdf(self, pdf_paths: 'List[str]', excel_path: str, output_excel_path: str, output_pdf_dir: str, 
                            create_zip: bool = True, zip_path: str = None) -> 'Tuple[List[str], List[str]]':
        """
        Run complete process with vector-based matching and generate annotated PDFs.
        Also adds tag numbers to the output Excel file and creates a ZIP archive of all output files.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path (IO List)
            output_excel_path: Output Excel file path
            output_pdf_dir: Directory path for storing processed PDFs
            create_zip: Whether to create a ZIP archive of all output files
            zip_path: Path for the ZIP archive (if None, will use output_pdf_dir + '.zip')
            
        Returns:
            Tuple of (unmatched_io_tags, unmatched_pdf_tags)
        """
        import zipfile
        import os
        
        try:
            logger.info("="*80)
            logger.info("🚀 STARTING COMPLETE PROCESSING WITH ANNOTATED PDFs")
            logger.info("="*80)
            
            # 🆕 Reset per-run warning state
            self._page_warnings = []
            self.latest_warnings = []
            
            # ============================================================
            # مرحله 1: مقداردهی اولیه و بارگذاری داده‌ها
            # ============================================================
            start_time = time.time()
            
            # Build tag vectors from Excel first
            logger.info("📚 Step 1: Building tag vectors from IO List...")
            self.build_tag_vectors_from_excel(excel_path)
            logger.info(f"✅ Loaded {len(self.tag_patterns)} tag patterns from IO List")
            
            # خواندن تگ‌های IO List
            io_tags = set()
            if excel_path and os.path.exists(excel_path):
                try:
                    io_df = pd.read_excel(excel_path)
                    if 'Tag No' in io_df.columns:
                        io_tags = set(str(tag).strip().upper() for tag in io_df['Tag No'] if pd.notna(tag) and str(tag).strip())
                        logger.info(f"✅ Loaded {len(io_tags)} tags from IO List")
                        logger.info(f"   Sample IO tags: {list(io_tags)[:5]}")
                except Exception as e:
                    logger.error(f"❌ Error reading IO List: {e}")
            
            # Create output PDF directory
            os.makedirs(output_pdf_dir, exist_ok=True)
            logger.info(f"📁 Output directory: {output_pdf_dir}")
            
            # ============================================================
            # مرحله 2: پردازش PDF ها
            # ============================================================
            all_similarity_reports = []
            master_tag_numbers = {}
            all_pdf_results = {}
            output_files = []
            
            all_pdf_tags = set()  # 🆕 جمع‌آوری تمام تگ‌های PDF
            all_pdf_ocr_tags = set()  # 🆕 تگ‌های خام OCR (همه تگ‌های شناسایی شده)
            all_pattern_unmatched_candidates = set()  # 🆕 کاندیداهای مشابه الگوی IO ولی خارج از IO List
            all_pattern_unmatched_details = []

            logger.info("\n" + "="*80)
            logger.info(f"📄 Step 2: Processing {len(pdf_paths)} PDF file(s)...")
            logger.info("="*80)

            for pdf_idx, pdf_path in enumerate(pdf_paths):
                pdf_filename = os.path.basename(pdf_path)
                logger.info(f"\n{'─'*80}")
                logger.info(f"📄 Processing PDF {pdf_idx + 1}/{len(pdf_paths)}: {pdf_filename}")
                logger.info(f"{'─'*80}")
                
                try:
                    # Process PDF
                    logger.info(f"   🔍 Extracting tags from PDF pages...")
                    pdf_result = self.process_pdf(pdf_path)
                    
                    if not pdf_result:
                        logger.warning(f"   ⚠️ No results from PDF: {pdf_filename}")
                        continue
                    
                    # Store PDF results
                    all_pdf_results[pdf_filename] = pdf_result

                    # ✅ FIX: جمع‌آوری all_ocr_tags از pdf_result
                    logger.info(f"   📊 Collecting OCR tags from {len(pdf_result)} pages...")
                    
                    # ✅ جمع‌آوری تگ‌ها
                    for page_num, page_data in pdf_result.items():
                        if isinstance(page_data, tuple):
                            # تگ‌های matched
                            if len(page_data) > 0:
                                tags = page_data[0]
                                if isinstance(tags, set):
                                    all_pdf_tags.update(tags)
                                elif isinstance(tags, list):
                                    all_pdf_tags.update(tags)
                            
                            # ✅ FIX: تگ‌های OCR (index 8)
                            if len(page_data) >= 9:
                                ocr_tags = page_data[8]
                                if ocr_tags:  # چک کنیم خالی نباشد
                                    if isinstance(ocr_tags, set):
                                        all_pdf_ocr_tags.update(ocr_tags)  # ✅ استفاده از متغیر صحیح
                                        logger.info(f"      Page {page_num}: collected {len(ocr_tags)} OCR tags")
                                    elif isinstance(ocr_tags, list):
                                        all_pdf_ocr_tags.update(ocr_tags)  # ✅ استفاده از متغیر صحیح
                                        logger.info(f"      Page {page_num}: collected {len(ocr_tags)} OCR tags")
                                else:
                                    logger.warning(f"      Page {page_num}: ocr_tags is empty or None!")
                            else:
                                logger.warning(f"      Page {page_num}: page_data has only {len(page_data)} elements (expected 9)")

                            # 🆕 جمع‌آوری کاندیداهای unmatched pattern (index 7 = tag_match_info)
                            if len(page_data) >= 8 and isinstance(page_data[7], dict):
                                page_match_info = page_data[7]
                                page_tag_to_number = page_data[5] if len(page_data) > 5 and isinstance(page_data[5], dict) else {}
                                page_jbs = []
                                page_mcs = []
                                page_cables = []
                                page_raw_cables = []
                                if len(page_data) > 1 and page_data[1]:
                                    page_jbs = sorted([str(jb).strip() for jb in page_data[1] if str(jb).strip()])
                                if len(page_data) > 2 and page_data[2]:
                                    page_mcs = sorted([str(mc).strip() for mc in page_data[2] if str(mc).strip()])
                                if len(page_data) > 3 and page_data[3]:
                                    page_cables = [str(c).strip() for c in page_data[3] if str(c).strip()]
                                if len(page_data) > 6 and page_data[6]:
                                    page_raw_cables = [str(c).strip() for c in page_data[6] if str(c).strip()]

                                for info in page_match_info.values():
                                    if not isinstance(info, dict):
                                        continue
                                    if info.get('match_type') != 'unmatched_candidate':
                                        continue
                                    candidate_text = self._normalize_ocr_tag_candidate(
                                        info.get('ocr_text', info.get('display_text', ''))
                                    )
                                    if candidate_text:
                                        candidate_number = info.get('tag_number') or page_tag_to_number.get(candidate_text)
                                        if candidate_number:
                                            try:
                                                candidate_number = int(candidate_number)
                                            except Exception:
                                                candidate_number = None

                                        wire_code_1 = str(info.get('wire_code_1') or '').strip()
                                        wire_code_2 = str(info.get('wire_code_2') or '').strip()
                                        terminal_first = str(info.get('terminal_first_number') or '').strip()
                                        terminal_second = str(info.get('terminal_second_number') or '').strip()
                                        scr_terminal = str(info.get('scr_terminal_number') or '').strip()
                                        cable_code = str(info.get('cable_code') or (page_cables[0] if page_cables else '')).strip()
                                        cable_description = str(info.get('cable_description') or (page_raw_cables[0] if page_raw_cables else '')).strip()
                                        wire_colors_text = str(info.get('wire_colors_text') or '').strip()
                                        wire_colors = info.get('wire_colors') if isinstance(info.get('wire_colors'), list) else []

                                        if candidate_number and (not terminal_first or not terminal_second):
                                            try:
                                                terminal_info = self.generate_terminal_numbers(candidate_number)
                                                terminal_first = terminal_first or str(terminal_info.get('terminal_first', '')).strip()
                                                terminal_second = terminal_second or str(terminal_info.get('terminal_second', '')).strip()
                                                scr_terminal = scr_terminal or str(terminal_info.get('scr_terminal', '')).strip()
                                            except Exception:
                                                pass

                                        if candidate_number and (not wire_code_1 and not wire_code_2):
                                            try:
                                                generated_wire = self.generate_mc_wire_colors_enhanced(candidate_number)
                                                parts = [p.strip() for p in str(generated_wire).split(',') if str(p).strip()]
                                                wire_code_1 = parts[0] if len(parts) > 0 else wire_code_1
                                                wire_code_2 = parts[1] if len(parts) > 1 else wire_code_2
                                                if not wire_colors:
                                                    wire_colors = parts
                                                if not wire_colors_text:
                                                    wire_colors_text = generated_wire
                                            except Exception:
                                                pass

                                        all_pattern_unmatched_candidates.add(candidate_text)
                                        all_pattern_unmatched_details.append({
                                            'source_type': 'pattern_unmatched_candidate',
                                            'ocr_text': candidate_text,
                                            'display_text': str(info.get('display_text', candidate_text)).strip(),
                                            'score': float(info.get('score', 0.0) or 0.0),
                                            'reason': str(info.get('reason', '')).strip(),
                                            'bbox': info.get('bbox') if isinstance(info.get('bbox'), dict) else {},
                                            'pdf_name': pdf_filename,
                                            'page': int(page_num),
                                            'jb': page_jbs[0] if page_jbs else '',
                                            'jb_all': page_jbs,
                                            'mc': page_mcs[0] if page_mcs else '',
                                            'mc_all': page_mcs,
                                            'tag_number': candidate_number if candidate_number else None,
                                            'wire_code_1': wire_code_1,
                                            'wire_code_2': wire_code_2,
                                            'terminal_first_number': terminal_first,
                                            'terminal_second_number': terminal_second,
                                            'scr_terminal_number': scr_terminal,
                                            'wire_colors_text': wire_colors_text,
                                            'wire_colors': wire_colors,
                                            'cable_code': cable_code,
                                            'cable_description': cable_description,
                                            'type': str(info.get('type') or 'Tag').strip(),
                                            'tag_number_status': str(info.get('tag_number_status') or 'Assigned (Position-based candidate)').strip(),
                                            'cable_descriptions': page_cables,
                                            'raw_cable_descriptions': page_raw_cables
                                        })
                    
                    logger.info(f"   ✅ PDF {pdf_filename}: {len(all_pdf_tags)} matched, {len(all_pdf_ocr_tags)} OCR total")
                    
                    # Create annotated PDF
                    output_pdf_path = os.path.join(output_pdf_dir, f"annotated_{pdf_filename}")
                    logger.info(f"   🎨 Creating annotated PDF...")
                    
                    pdf_tag_numbers = self.create_annotated_pdf(pdf_path, output_pdf_path)
                    master_tag_numbers.update(pdf_tag_numbers)
                    
                    output_files.append(output_pdf_path)
                    logger.info(f"   ✅ Annotated PDF saved: {output_pdf_path}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error processing PDF {pdf_filename}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            # ✅ لاگ نهایی
            logger.info(f"\n✅ All PDFs processed:")
            logger.info(f"   - Total matched tags: {len(all_pdf_tags)}")
            logger.info(f"   - Total OCR tags: {len(all_pdf_ocr_tags)}")  # ✅ نام صحیح
            logger.info(f"   - Sample OCR tags: {list(all_pdf_ocr_tags)[:10]}")
            logger.info(f"   - Pattern-based unmatched candidates: {len(all_pattern_unmatched_candidates)}")

            # ✅ چک کردن خالی بودن
            if not all_pdf_ocr_tags:
                logger.error("❌ CRITICAL: all_pdf_ocr_tags is empty after processing all PDFs!")
                logger.error("   This means page_data[8] was empty or missing in all pages")
            
            # ============================================================
            # مرحله 3: ایجاد فایل اکسل میانی
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("📊 Step 3: Creating intermediate Excel file...")
            logger.info("="*80)
            
            intermediate_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Intermediate.xlsx")
            
            try:
                self.add_wire_colors_and_scr_to_dataframe(
                    pd.DataFrame(), 
                    master_tag_numbers, 
                    intermediate_excel_path, 
                    all_pdf_results,
                    io_tags
                )
                output_files.append(intermediate_excel_path)
                logger.info(f"✅ Intermediate Excel saved: {intermediate_excel_path}")
            except Exception as e:
                logger.error(f"❌ Error creating intermediate Excel: {e}")
                logger.error(traceback.format_exc())
            
            # ============================================================
            # مرحله 4: ترکیب با IO List (در صورت وجود)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("🔗 Step 4: Combining with IO List...")
            logger.info("="*80)
            
            unmatched_pdf_tags = []
            unmatched_io_tags = []
            
            if excel_path and os.path.exists(excel_path):
                try:
                    # نام‌گذاری فایل اکسل نهایی
                    if not output_excel_path.endswith(".xlsx"):
                        output_excel_path = output_excel_path.replace(".xls", ".xlsx") if output_excel_path.endswith(".xls") else f"{output_excel_path}.xlsx"
                    
                    if not os.path.basename(output_excel_path):
                        output_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx")
                    
                    logger.info(f"   📊 Sending {len(all_pdf_ocr_tags)} OCR tags to process_excel_with_io_list")
                    # 🆕 ارسال all_ocr_tags به متد
                    final_df, unmatched_io_tags, unmatched_pdf_tags = self.process_excel_with_io_list(
                        intermediate_excel_path, 
                        excel_path, 
                        output_excel_path,
                        all_pdf_ocr_tags 
                    )
                                
                    output_files.append(output_excel_path)
                    logger.info(f"✅ Final Excel saved: {output_excel_path}")
                    logger.info(f"   📊 Total rows: {len(final_df)}")
                    logger.info(f"   ⚠️ Unmatched PDF tags: {len(unmatched_pdf_tags)}")
                    logger.info(f"   ⚠️ Unmatched IO tags: {len(unmatched_io_tags)}")
                    
                except Exception as e:
                    logger.error(f"❌ Error combining with IO List: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Fallback: copy intermediate to output
                    shutil.copy2(intermediate_excel_path, output_excel_path)
                    output_files.append(output_excel_path)
                    logger.warning(f"⚠️ Copied intermediate to output (fallback)")
            else:
                # اگر IO List نباشد
                logger.warning("⚠️ No IO List provided, using intermediate as final")
                
                if not output_excel_path.endswith(".xlsx"):
                    output_excel_path = output_excel_path.replace(".xls", ".xlsx") if output_excel_path.endswith(".xls") else f"{output_excel_path}.xlsx"
                
                if not os.path.basename(output_excel_path):
                    output_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx")
                
                shutil.copy2(intermediate_excel_path, output_excel_path)
                output_files.append(output_excel_path)
                logger.info(f"✅ Final Excel saved: {output_excel_path}")
                
                # 🆕 محاسبه unmatched از روی all_pdf_tags
                unmatched_pdf_tags = sorted(list(all_pdf_tags - io_tags)) if io_tags else []
                unmatched_io_tags = sorted(list(io_tags - all_pdf_tags)) if io_tags else []
            
            # ============================================================
            # 🆕 مرحله 5: ایجاد فایل Unmatched Tags (همیشه)
            # ============================================================
            logger.info("\n" + "="*80)
            logger.info("📝 Step 5: Creating Unmatched Tags Report...")
            logger.info("="*80)

            unmatched_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Unmatched_Tags.xlsx")

            try:
                # فقط OCR tags که در IO List نیستند
                ocr_only_unmatched = set()
                if all_pdf_ocr_tags:
                    ocr_only_unmatched = all_pdf_ocr_tags - io_tags  # ✅ فقط از OCR کم شود

                # تگ‌های IO که اصلاً در PDF پیدا نشده‌اند
                io_only_tags = io_tags - all_pdf_ocr_tags

                logger.info(f"   📊 OCR-only unmatched: {len(ocr_only_unmatched)}")
                logger.info(f"   📊 IO-only tags: {len(io_only_tags)}")

                # نمونه
                if ocr_only_unmatched:
                    logger.info(f"      Sample OCR-only: {list(ocr_only_unmatched)[:5]}")
                if io_only_tags:
                    logger.info(f"      Sample IO-only: {list(io_only_tags)[:5]}")

                # ایجاد Excel
                unmatched_data = []

                # 1. تگ‌های OCR که در IO List نیستند
                for tag in sorted(ocr_only_unmatched):
                    unmatched_data.append({
                        'Tag': tag,
                        'Source': 'PDF (OCR)',
                        'Status': 'Found in PDF but not in IO List',
                        'Severity': 'WARNING',
                        'Action': 'Verify tag correctness or add to IO List',
                        'Match_Type': 'Not Matched'
                    })

                # 2. تگ‌های IO که در PDF نیستند
                for tag in sorted(io_only_tags):
                    unmatched_data.append({
                        'Tag': tag,
                        'Source': 'IO List',
                        'Status': 'In IO List but not found in PDF',
                        'Severity': 'INFO',
                        'Action': 'Check if tag should appear in PDF',
                        'Match_Type': 'N/A'
                    })

                # ذخیره Excel
                if unmatched_data:
                    unmatched_df = pd.DataFrame(unmatched_data)
                    logger.info(f"   ✅ Created report with {len(unmatched_data)} unmatched tags")
                else:
                    unmatched_df = pd.DataFrame([{
                        'Tag': 'N/A',
                        'Source': 'N/A',
                        'Status': '✅ All tags matched successfully!',
                        'Severity': 'SUCCESS',
                        'Action': 'No action needed',
                        'Match_Type': 'N/A'
                    }])
                    logger.info("   ✅ All tags matched!")

                unmatched_df.to_excel(unmatched_excel_path, index=False)
                output_files.append(unmatched_excel_path)

                logger.info(f"✅ Unmatched tags Excel saved: {unmatched_excel_path}")
                logger.info(f"   📄 File contains {len(unmatched_df)} rows")

            except Exception as e:
                logger.error(f"❌ Error creating unmatched tags Excel: {e}")
                logger.error(traceback.format_exc())

                # ایجاد فایل خالی در صورت خطا
                try:
                    empty_df = pd.DataFrame(columns=['Tag', 'Source', 'Status', 'Severity', 'Action', 'Match_Type'])
                    empty_df.loc[0] = ['ERROR', 'SYSTEM', f'Error creating report: {str(e)}', 'ERROR', 'Check logs', 'N/A']
                    empty_df.to_excel(unmatched_excel_path, index=False)
                    output_files.append(unmatched_excel_path)
                    logger.warning(f"⚠️ Created error report at: {unmatched_excel_path}")
                except:
                    logger.error(f"❌ Could not create error report file")
            # ============================================================
            # مرحله 6: ایجاد ZIP (اختیاری)
            # ============================================================
            if create_zip:
                logger.info("\n" + "="*80)
                logger.info("📦 Step 6: Creating ZIP archive...")
                logger.info("="*80)
                
                try:
                    if zip_path is None:
                        zip_path = output_pdf_dir.rstrip('/\\') + '.zip'
                    
                    # حذف فایل ZIP قبلی در صورت وجود
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                        logger.info(f"   🗑️ Removed existing ZIP: {zip_path}")
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_path in output_files:
                            if os.path.exists(file_path):
                                arcname = os.path.basename(file_path)
                                zipf.write(file_path, arcname=arcname)
                                logger.info(f"   ✅ Added to ZIP: {arcname}")
                    
                    logger.info(f"✅ ZIP archive created: {zip_path}")
                    logger.info(f"   📦 Contains {len(output_files)} files")
                    
                except Exception as e:
                    logger.error(f"❌ Error creating ZIP archive: {e}")
                    logger.error(traceback.format_exc())
            
            # ============================================================
            # مرحله 7: گزارش نهایی
            # ============================================================
            self.processing_time = time.time() - start_time
            
            logger.info("\n" + "="*80)
            logger.info("📊 FINAL SUMMARY")
            logger.info("="*80)
            
            stats = self.get_processing_stats()
            logger.info(f"⏱️  Total processing time: {self.processing_time:.2f} seconds")
            logger.info(f"📄 PDFs processed: {len(pdf_paths)}")
            logger.info(f"🏷️  Total unique PDF tags: {len(all_pdf_tags)}")
            logger.info(f"🏷️  Total IO List tags: {len(io_tags)}")
            logger.info(f"✅ Tags numbered: {len(master_tag_numbers)}")
            logger.info(f"⚠️  Unmatched PDF tags: {len(unmatched_pdf_tags)}")
            logger.info(f"⚠️  Unmatched IO tags: {len(unmatched_io_tags)}")
            logger.info(f"🚩 Pattern-based unmatched candidates: {len(all_pattern_unmatched_candidates)}")
            logger.info(f"📁 Output files created: {len(output_files)}")
            
            logger.info("\n📂 Output Files:")
            for i, file_path in enumerate(output_files, 1):
                file_size = os.path.getsize(file_path) / 1024  # KB
                logger.info(f"   {i}. {os.path.basename(file_path)} ({file_size:.1f} KB)")
            
            logger.info("\n" + "="*80)
            logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("="*80 + "\n")

            # برای نمایش در UI
            self.latest_pattern_unmatched_candidates = sorted(all_pattern_unmatched_candidates)
            # جزئیات برای ذخیره در DB و نمایش در داشبورد
            unique_details = []
            seen_keys = set()
            for item in all_pattern_unmatched_details:
                key = (
                    str(item.get('pdf_name', '')).upper(),
                    int(item.get('page', 0) or 0),
                    str(item.get('ocr_text', '')).upper(),
                    str(item.get('jb', '')).upper()
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                unique_details.append(item)
            self.latest_pattern_unmatched_details = unique_details
            
            # ============================================================
            # 🆕 مرحله 5b: جمع‌آوری هشدارهای cross-page
            # ============================================================
            #   - JB های که در چندین صفحه تکرار شده‌اند
            #   - Tag هایی که در چندین صفحه تکرار شده‌اند
            #   - Pages که به دلیل multiple JB skip شده‌اند
            #   - Pages که tags دارند ولی JB پیدا نشده
            # این هشدارها در UI / Excel / Annotated PDF نمایش داده می‌شوند.
            # ============================================================
            try:
                cross_page_warnings = self._collect_cross_page_warnings(all_pdf_results)
                # Merge per-page warnings (JB_NOT_FOUND, PAGE_SKIPPED_MULTIPLE_JB)
                # with cross-page warnings (DUPLICATE_JB, DUPLICATE_TAG)
                merged_warnings = list(self._page_warnings or []) + list(cross_page_warnings or [])
                # Deduplicate by (type, item, pdf_name)
                seen_warn_keys = set()
                deduped = []
                for w in merged_warnings:
                    k = (
                        str(w.get('type', '')).upper(),
                        str(w.get('item', '')).upper(),
                        str(w.get('pdf_name', '')).upper(),
                    )
                    if k in seen_warn_keys:
                        continue
                    seen_warn_keys.add(k)
                    deduped.append(w)
                self.latest_warnings = deduped
                logger.info(
                    f"📋 Warnings collected: {len(self.latest_warnings)} "
                    f"(JB_NOT_FOUND={sum(1 for w in self.latest_warnings if w.get('type')=='JB_NOT_FOUND')}, "
                    f"DUPLICATE_JB={sum(1 for w in self.latest_warnings if w.get('type')=='DUPLICATE_JB')}, "
                    f"DUPLICATE_TAG={sum(1 for w in self.latest_warnings if w.get('type')=='DUPLICATE_TAG')}, "
                    f"PAGE_SKIPPED_MULTIPLE_JB={sum(1 for w in self.latest_warnings if w.get('type')=='PAGE_SKIPPED_MULTIPLE_JB')})"
                )
                
                # 🆕 ایجاد فایل Excel جداگانه برای warnings
                warnings_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Warnings.xlsx")
                try:
                    self._create_warnings_excel(self.latest_warnings, warnings_excel_path)
                    output_files.append(warnings_excel_path)
                    logger.info(f"✅ Warnings Excel saved: {warnings_excel_path}")
                except Exception as w_err:
                    logger.error(f"❌ Error creating Warnings Excel: {w_err}")
                    logger.error(traceback.format_exc())
            except Exception as warn_err:
                logger.error(f"❌ Error collecting warnings: {warn_err}")
                logger.error(traceback.format_exc())
                self.latest_warnings = list(self._page_warnings or [])
            
            return list(io_only_tags), list(ocr_only_unmatched)
            
        except Exception as e:
            logger.error("="*80)
            logger.error("❌ CRITICAL ERROR IN run_with_annotated_pdf")
            logger.error("="*80)
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            logger.error("="*80)
            self.latest_pattern_unmatched_candidates = []
            self.latest_pattern_unmatched_details = []
            self.latest_warnings = list(getattr(self, '_page_warnings', []) or [])
            return [], []
        
    def _collect_cross_page_warnings(self, all_pdf_results: 'Dict[str, Dict[int, Tuple[Any, ...]]]') -> 'List[Dict[str, Any]]':
        """
        🆕 Detect cross-page anomalies and return them as a list of warning dicts.

        Warnings produced:
          - DUPLICATE_JB:      same JB identifier appears on 2+ pages of the same PDF
          - DUPLICATE_TAG:     same tag appears on 2+ pages of the same PDF (or across PDFs)

        Args:
            all_pdf_results: dict keyed by PDF filename → dict keyed by page number → 9-tuple

        Returns:
            list of warning dicts (each compatible with self._page_warnings entries)
        """
        warnings_list = []

        if not all_pdf_results:
            return warnings_list

        try:
            # Per-PDF analysis (JBs and tags duplicated within the same PDF)
            for pdf_name, page_results_dict in all_pdf_results.items():
                if not page_results_dict:
                    continue

                # Build JB → [pages] and Tag → [pages] maps
                jb_pages: 'Dict[str, List[int]]' = {}
                tag_pages: 'Dict[str, List[int]]' = {}

                for page_num, page_results in page_results_dict.items():
                    if not isinstance(page_results, (tuple, list)) or len(page_results) < 9:
                        continue
                    jbs = page_results[1] or set()
                    tags = page_results[0] or set()

                    for jb in jbs:
                        jb_u = str(jb).strip().upper()
                        if not jb_u:
                            continue
                        jb_pages.setdefault(jb_u, []).append(int(page_num))

                    for tag in tags:
                        tag_u = str(tag).strip().upper()
                        if not tag_u:
                            continue
                        # Skip pure SPARE / placeholder tokens
                        if 'SPARE' in tag_u:
                            continue
                        tag_pages.setdefault(tag_u, []).append(int(page_num))

                # Duplicate JBs (same JB on 2+ pages within one PDF)
                for jb_u, pages in jb_pages.items():
                    if len(pages) > 1:
                        warnings_list.append({
                            'type': 'DUPLICATE_JB',
                            'item': jb_u,
                            'pages': sorted(set(pages)),
                            'tag_count': 0,
                            'severity': 'WARNING',
                            'description': (
                                f"JB '{jb_u}' appears on {len(pages)} pages of '{pdf_name}': "
                                f"{sorted(set(pages))}. "
                                f"In table-mode PDFs each page typically belongs to a DIFFERENT JB — "
                                f"a duplicate JB often means the wrong JB was assigned on one of the pages "
                                f"(e.g. previous page's JB was reused by mistake)."
                            ),
                            'action': (
                                'Open the annotated PDF and verify that the JB label on each listed '
                                'page actually matches. If a page has no JB, it should be flagged as '
                                'JB_NOT_FOUND rather than inheriting the previous JB.'
                            ),
                            'pdf_name': pdf_name,
                        })

                # Duplicate tags (same tag on 2+ pages within one PDF)
                for tag_u, pages in tag_pages.items():
                    if len(pages) > 1:
                        warnings_list.append({
                            'type': 'DUPLICATE_TAG',
                            'item': tag_u,
                            'pages': sorted(set(pages)),
                            'tag_count': 0,
                            'severity': 'WARNING',
                            'description': (
                                f"Tag '{tag_u}' appears on {len(pages)} pages of '{pdf_name}': "
                                f"{sorted(set(pages))}. "
                                f"A single instrument tag should normally appear on only ONE page. "
                                f"Duplicates may indicate OCR double-read, page-header leakage, "
                                f"or a genuinely duplicated tag in the source PDF."
                            ),
                            'action': (
                                'Check the annotated PDF — confirm the tag appears at the correct '
                                'position on each listed page. If it is an OCR artefact, ignore. '
                                'If it is a real duplicate, fix the source PDF or IO List.'
                            ),
                            'pdf_name': pdf_name,
                        })

            # Cross-PDF duplicate tags (same tag across different PDFs)
            cross_tag_pdfs: 'Dict[str, List[str]]' = {}
            for pdf_name, page_results_dict in all_pdf_results.items():
                if not page_results_dict:
                    continue
                pdf_tag_set = set()
                for page_num, page_results in page_results_dict.items():
                    if not isinstance(page_results, (tuple, list)) or len(page_results) < 9:
                        continue
                    tags = page_results[0] or set()
                    for tag in tags:
                        tag_u = str(tag).strip().upper()
                        if not tag_u or 'SPARE' in tag_u:
                            continue
                        pdf_tag_set.add(tag_u)
                for tag_u in pdf_tag_set:
                    cross_tag_pdfs.setdefault(tag_u, []).append(pdf_name)

            for tag_u, pdfs in cross_tag_pdfs.items():
                if len(pdfs) > 1:
                    warnings_list.append({
                        'type': 'DUPLICATE_TAG',
                        'item': tag_u,
                        'pages': sorted(set(pdfs)),
                        'tag_count': 0,
                        'severity': 'INFO',
                        'description': (
                            f"Tag '{tag_u}' appears in {len(pdfs)} different PDFs: {sorted(set(pdfs))}. "
                            f"This is normal if the same instrument is wired through multiple JBs, "
                            f"but please verify it is intentional."
                        ),
                        'action': 'Verify the tag is expected to appear in multiple PDFs.',
                        'pdf_name': ','.join(sorted(set(pdfs))),
                    })

        except Exception as e:
            logger.error(f"Error in _collect_cross_page_warnings: {e}")
            logger.error(traceback.format_exc())

        return warnings_list

    def _create_warnings_excel(self, warnings: 'List[Dict[str, Any]]', output_path: str):
        """
        🆕 Create a dedicated Excel file listing all warnings collected during the run.

        Columns:
            Warning_Type | Item | PDF_Name | Pages | Tag_Count | Severity | Description | Action
        """
        try:
            if warnings:
                rows = []
                for w in warnings:
                    pages = w.get('pages', [])
                    if pages and all(isinstance(p, int) for p in pages):
                        pages_str = ', '.join(str(p) for p in sorted(pages))
                    else:
                        pages_str = ', '.join(str(p) for p in pages)
                    rows.append({
                        'Warning_Type': w.get('type', ''),
                        'Item': w.get('item', ''),
                        'PDF_Name': w.get('pdf_name', ''),
                        'Pages': pages_str,
                        'Tag_Count': w.get('tag_count', 0),
                        'Severity': w.get('severity', ''),
                        'Description': w.get('description', ''),
                        'Action': w.get('action', ''),
                    })
                df = pd.DataFrame(rows, columns=[
                    'Warning_Type', 'Item', 'PDF_Name', 'Pages',
                    'Tag_Count', 'Severity', 'Description', 'Action'
                ])
                # Sort: ERROR first, then WARNING, then INFO
                severity_order = {'ERROR': 0, 'WARNING': 1, 'INFO': 2}
                df['_sev_order'] = df['Severity'].map(severity_order).fillna(3)
                df = df.sort_values(by=['_sev_order', 'Warning_Type', 'Item']).drop(columns=['_sev_order'])
                df.to_excel(output_path, index=False)
                logger.info(f"   Warnings Excel: {len(df)} rows")
            else:
                # No warnings — create a "success" placeholder file so the
                # operator can see the system actually ran the check.
                df = pd.DataFrame([{
                    'Warning_Type': 'NONE',
                    'Item': 'N/A',
                    'PDF_Name': 'N/A',
                    'Pages': 'N/A',
                    'Tag_Count': 0,
                    'Severity': 'SUCCESS',
                    'Description': 'No JB-not-found, duplicate-JB, or duplicate-tag issues detected.',
                    'Action': 'No action needed.',
                }])
                df.to_excel(output_path, index=False)
                logger.info("   Warnings Excel: no warnings — wrote success placeholder")
        except Exception as e:
            logger.error(f"Error in _create_warnings_excel: {e}")
            logger.error(traceback.format_exc())
            # Last-resort: write an empty file so the rest of the pipeline doesn't break
            try:
                pd.DataFrame([{
                    'Warning_Type': 'ERROR',
                    'Item': 'SYSTEM',
                    'PDF_Name': 'N/A',
                    'Pages': 'N/A',
                    'Tag_Count': 0,
                    'Severity': 'ERROR',
                    'Description': f'Error generating warnings report: {e}',
                    'Action': 'Check logs for details.',
                }]).to_excel(output_path, index=False)
            except Exception:
                pass

    def set_wire_color_rule(self, rule):
        """
        تنظیم قانون تولید رنگ سیم
        
        Args:
            rule: قانون تولید رنگ سیم
        """
        try:
            logger.info(f"Setting wire color rule: {rule}")
            self.wire_color_rule = rule
            
            # تست قانون با یک نمونه
            test_colors = self.generate_mc_wire_colors(1)
            logger.info(f"Test wire colors for tag #1: {test_colors}")
        except Exception as e:
            logger.error(f"Error Setting wire color rule: {e}")

    def set_scr_number_rule(self, rule):
        """
        تنظیم قانون تولید شماره SCR
        
        Args:
            rule: قانون تولید شماره SCR
        """
        try:
            logger.info(f"Setting SCR number rule: {rule}")
            self.scr_number_rule = rule
            
            # تست قانون با یک نمونه
            test_scr = self.generate_scr_number(1)
            logger.info(f"Test SCR number for tag #1: {test_scr}")
        except Exception as e:
            logger.error(f"Error Setting SCR number rule: {e}")

    def generate_scr_number(self, tag_number):
        """
        تولید شماره SCR بر اساس شماره تگ و قانون تعریف شده
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            شماره SCR
        """
        try:
            if not hasattr(self, 'scr_number_rule') or not self.scr_number_rule:
                return ''
                    
            # جایگزینی {number} با شماره تگ
            if '{number' in self.scr_number_rule:
                # بررسی فرمت اختیاری
                format_match = re.search(r'\{number:([^}]+)\}', self.scr_number_rule)
                if format_match:
                    format_spec = format_match.group(1)
                    formatted_number = format(tag_number, format_spec)
                    scr_number = self.scr_number_rule.replace(format_match.group(0), formatted_number)
                else:
                    scr_number = self.scr_number_rule.replace('{number}', str(tag_number))
            else:
                # جایگزینی ساده عبارات ریاضی
                # مثال: {number*2-1} {number*2} SCR -> "1 2 SCR" برای tag_number=1
                def replace_expr(match):
                    expr = match.group(1).replace('number', str(tag_number))
                    try:
                        result = eval(expr)
                        return str(result)
                    except Exception as e:
                        logger.error(f"Error evaluating expression {expr}: {e}")
                        return match.group(0)
                    
                scr_number = re.sub(r'\{([^}]+)\}', replace_expr, self.scr_number_rule)
                
            return scr_number
        except Exception as e:
            logger.error(f"Error generating SCR number: {e}")
            return ''

    def set_terminal_wire_patterns(self, config: Dict[str, Any]):
        """
        تنظیم الگوهای جدید ترمینال و سیم
        
        Args:
            config: دیکشنری حاوی:
                - terminal_pattern: الگوی ترمینال
                - wire_color_pattern: الگوی رنگ سیم
                - include_scr: آیا SCR شامل شود
                - selected_colors: لیست رنگ‌های انتخاب شده
        """
        try:
            self.terminal_pattern = config.get('terminal_pattern', '')
            self.wire_color_rule = config.get('wire_color_pattern', '')
            self.terminal_pattern_dict = config
            
            logger.info("✨ New terminal/wire patterns set:")
            logger.info(f"   Terminal: {self.terminal_pattern}")
            logger.info(f"   Wire Color: {self.wire_color_rule}")
            logger.info(f"   Include SCR: {config.get('include_scr', True)}")
            logger.info(f"   Colors: {config.get('selected_colors', [])}")
            
            # تست الگوها با یک نمونه
            test_terminals = self.generate_terminal_numbers(1)
            test_wire_colors = self.generate_mc_wire_colors_enhanced(1)
            logger.info(f"   Test output for tag #1:")
            logger.info(f"      Terminals: {test_terminals}")
            logger.info(f"      Wire Colors: {test_wire_colors}")
            
        except Exception as e:
            logger.error(f"Error setting terminal/wire patterns: {e}")
    
    def generate_terminal_numbers(self, tag_number: int) -> Dict[str, str]:
        """
        تولید شماره‌های ترمینال بر اساس الگو
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            دیکشنری حاوی شماره‌های ترمینال
        """
        try:
            if not self.terminal_pattern:
                # الگوی پیش‌فرض
                return {
                    'terminal_first': str(tag_number),
                    'terminal_second': str(tag_number + 1),
                    'scr_terminal': self.generate_scr_number(tag_number),
                    'full_string': f"{tag_number}, {tag_number + 1}"
                }
            
            pattern = self.terminal_pattern
            include_scr = self.terminal_pattern_dict.get('include_scr', True)
            
            # جایگزینی x با شماره تگ و محاسبه عبارات
            def replace_expr(match):
                expr = match.group(1)
                expr = expr.replace('x', str(tag_number))
                try:
                    result = eval(expr)
                    return str(int(result))
                except Exception as e:
                    logger.error(f"Error evaluating expression {expr}: {e}")
                    return match.group(0)
            
            # پردازش الگو
            result = re.sub(r'\{([^}]+)\}', replace_expr, pattern)
            
            # حذف SCR اگر غیرفعال باشد
            if not include_scr:
                result = re.sub(r',?\s*SCR\s*,?', '', result)
                result = re.sub(r',\s*,', ',', result)
                result = result.strip(', ')
            
            # تجزیه نتیجه به اجزا
            parts = [p.strip() for p in result.split(',')]
            
            terminal_first = ''
            terminal_second = ''
            scr_terminal = ''
            
            # پیدا کردن SCR
            scr_parts = [p for p in parts if 'SCR' in p.upper()]
            if scr_parts:
                scr_terminal = scr_parts[0]
                parts = [p for p in parts if 'SCR' not in p.upper()]
            
            # اولین و دومین ترمینال
            if len(parts) >= 1:
                terminal_first = parts[0]
            if len(parts) >= 2:
                terminal_second = parts[1]
            
            return {
                'terminal_first': terminal_first,
                'terminal_second': terminal_second,
                'scr_terminal': scr_terminal,
                'full_string': result
            }
            
        except Exception as e:
            logger.error(f"Error generating terminal numbers: {e}")
            return {
                'terminal_first': str(tag_number),
                'terminal_second': str(tag_number + 1),
                'scr_terminal': '',
                'full_string': f"{tag_number}, {tag_number + 1}"
            }
    
    def generate_mc_wire_colors_enhanced(self, tag_number: int) -> str:
        """
        نسخه بهبود یافته تولید رنگ‌های سیم با پشتیبانی از الگوهای جدید
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            رشته رنگ‌های سیم
        """
        try:
            # اگر الگوی جدید تنظیم شده، از آن استفاده کن
            if hasattr(self, 'terminal_pattern_dict') and self.terminal_pattern_dict:
                wire_pattern = self.terminal_pattern_dict.get('wire_color_pattern', '')
                if wire_pattern:
                    # جایگزینی {x} یا {x:02d} با شماره تگ
                    def replace_number(match):
                        format_spec = match.group(1)
                        if format_spec and ':' in format_spec:
                            fmt = format_spec.split(':')[1].rstrip('d}')
                            width = int(fmt) if fmt else 2
                            return str(tag_number).zfill(width)
                        return str(tag_number)
                    
                    result = re.sub(r'\{x(?::(\d+)d)?\}', replace_number, wire_pattern)
                    return result
            
            # در غیر این صورت از روش قدیمی استفاده کن
            return self.generate_mc_wire_colors(tag_number)
            
        except Exception as e:
            logger.error(f"Error generating wire colors: {e}")
            return self.generate_mc_wire_colors(tag_number)
    
    # 🔧 اصلاح متد generate_mc_wire_colors موجود
    def generate_mc_wire_colors(self, tag_number):
        """
        تولید رنگ‌های سیم بر اساس شماره تگ و قانون تعریف شده (متد قدیمی)
        """
        try:
            if not hasattr(self, 'wire_color_rule') or not self.wire_color_rule:
                return f"BK{tag_number:02d}, WT{tag_number:02d}"  # پیش‌فرض
                
            # ... بقیه کد موجود بدون تغییر ...
            color_rules = [rule.strip() for rule in self.wire_color_rule.split(',')]
            colors = []
            for rule in color_rules:
                if '{number' in rule:
                    format_match = re.search(r'\{number:([^}]+)\}', rule)
                    if format_match:
                        format_spec = format_match.group(1)
                        formatted_number = format(tag_number, format_spec)
                        color = rule.replace(format_match.group(0), formatted_number)
                    else:
                        color = rule.replace('{number}', str(tag_number))
                else:
                    expr_match = re.search(r'\{([^}]+)\}', rule)
                    if expr_match:
                        expr = expr_match.group(1).replace('number', str(tag_number))
                        try:
                            result = eval(expr)
                            color = rule.replace(expr_match.group(0), str(result))
                        except Exception as e:
                            logger.error(f"Error evaluating expression {expr}: {e}")
                            color = rule
                    else:
                        color = rule
                colors.append(color)
            return ', '.join(colors)
        except Exception as e:
            logger.error(f"Error generating wire colors: {e}")
            return f"BK{tag_number:02d}, WT{tag_number:02d}"

    def add_wire_colors_and_scr_to_dataframe(self, df: pd.DataFrame, tag_to_number: 'Dict[str, int]', 
                                            output_path: str, pdf_results: 'Dict[str, Dict[int, Tuple[Any, ...]]]',
                                            io_tags: 'Set[str]' = None,
                                            pdf_name: str = None) -> pd.DataFrame:
            """
            رنگ‌های سیم MC و شماره‌های SCR را به دیتافریم اضافه می‌کند و یک فایل اکسل جدید ایجاد می‌کند.
            استفاده مستقیم از شماره‌های تگ استخراج شده توسط bounding box.
            
            Args:
                df: دیتافریم ورودی حاوی اطلاعات تگ
                tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها از bounding box
                output_path: مسیر فایل اکسل خروجی
                pdf_results: نتایج پردازش PDF ها (دیکشنری با کلید نام PDF و مقدار نتایج صفحات)
                pdf_name: نام فایل PDF (اختیاری)
                
            Returns:
                دیتافریم به‌روزرسانی شده با ستون‌های جدید
            """
            try:
                # ایجاد دیتافریم جدید برای فایل اکسل خروجی
                new_df_data = []
                
                # بررسی ساختار pdf_results
                logger.info(f"pdf_results structure: {type(pdf_results)}")
                
                # پردازش هر PDF به صورت جداگانه
                for pdf_name, page_results_dict in pdf_results.items():
                    logger.info(f"Processing PDF: {pdf_name}")
                    
                    if page_results_dict is None:
                        logger.warning(f"page_results_dict for PDF {pdf_name} is None, skipping.")
                        continue

                    # پردازش هر صفحه از این PDF
                    for page_num, page_results in page_results_dict.items():
                        # فراخوانی متد کمکی برای پردازش هر صفحه
                        self._process_single_page_data(new_df_data, page_num, page_results, pdf_name, tag_to_number , io_tags)
                
                # ایجاد دیتافریم جدید
                new_df = pd.DataFrame(new_df_data)
                
                # اگر دیتافریم خالی نیست، مرتب‌سازی انجام بده
                if not new_df.empty:
                    # مرتب‌سازی بر اساس نام PDF، صفحه، JB و شماره تگ
                    new_df = new_df.sort_values(by=['PDF_Name', 'Page', 'JB', 'Tag_Number'])
                    
                    # تنظیم ترتیب ستون‌های نهایی
                    column_order = [
                        'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                        'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number','Cable_Code', 'SCR_Terminal_Number',
                        'Cable_Description', 'Type', 'Tag_Number_Status',  # اضافه کردن ستون وضعیت
                        'Warning'  # 🆕 ستون هشدار (خالی اگر مشکلی نیست)
                    ]
                    
                    # فقط ستون‌هایی که وجود دارند را انتخاب کن
                    available_columns = [col for col in column_order if col in new_df.columns]
                    new_df = new_df[available_columns]
                else:
                    # اگر دیتافریم خالی است، ایجاد دیتافریم با ستون‌های مناسب
                    new_df = pd.DataFrame(columns=[
                        'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                        'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number','Cable_Code', 'SCR_Terminal_Number',
                        'Cable_Description', 'Type', 'Tag_Number_Status', 'Warning'
                    ])
                    logger.warning("Created empty DataFrame with proper columns")
                
                # ذخیره دیتافریم به عنوان فایل اکسل
                new_df.to_excel(output_path, index=False)
                
                # آمار کلی
                total_tags = len(new_df[new_df['Type'] == 'Tag'].drop_duplicates(subset=['Tag/SPARE'])) if not new_df.empty else 0
                total_spares = len(new_df[new_df['Type'] == 'SPARE'].drop_duplicates(subset=['Tag/SPARE'])) if not new_df.empty else 0
                
                # تعداد هشدارها
                warnings_count = len(new_df[new_df['Tag_Number_Status'].str.contains('WARNING', na=False)]) if 'Tag_Number_Status' in new_df.columns else 0
                
                logger.info(f"Created Excel file with {len(new_df)} rows:")
                logger.info(f"Total rows: {len(new_df)} ({total_tags} unique tags, {total_spares} unique spares)")
                logger.info(f"Number of warnings: {warnings_count}")
                logger.info(f"Output file: {output_path}")
                
                return new_df
                
            except Exception as e:
                logger.error(f"Error in add_wire_colors_and_scr_to_dataframe: {e}")
                logger.error(traceback.format_exc())
                
                # ایجاد دیتافریم خالی در صورت خطا
                empty_df = pd.DataFrame(columns=[
                    'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                    'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number','Cable_Code', 'SCR_Terminal_Number',
                    'Cable_Description', 'Type', 'Tag_Number_Status'
                ])
                empty_df.to_excel(output_path, index=False)
                return empty_df


    def extract_pair_number(self, cable_description):
        """
        استخراج عدد پشت 'Pair' از توضیحات کابل
        
        Args:
            cable_description: توضیحات کابل (مثلاً "12 pair", "12P", "12 PAIR", "12 CORE", "12C")
            
        Returns:
            عدد استخراج شده یا None اگر هیچ عددی پیدا نشد
        """
        
        
        if not cable_description:
            return None
            
        # الگوهای مختلف برای استخراج شماره زوج
        pair_patterns = [
            r'(\d+)\s*(?:pair|P|PR)',  # مثل "12 pair", "12P", "12 P"
            r'(\d+)P',                 # مثل "12P"
            r'(\d+)\s*PAIR',           # مثل "12 PAIR"
            r'(\d+)\s*CORE',           # مثل "12 CORE"
            r'(\d+)\s*C',              # مثل "12 C"
            r'(\d+)C'                  # مثل "12C"
        ]
        
        desc_str = str(cable_description).upper()  # تبدیل به رشته و حروف بزرگ برای جستجوی بهتر
        
        for pattern in pair_patterns:
            match = re.search(pattern, desc_str, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
                    
        return None

    def _process_page_results(self, pdf_results: 'Dict[str, Dict[int, Tuple[Any, ...]]]', 
                            tag_to_number: 'Dict[str, int]', output_path: str) -> pd.DataFrame:
        """
        پردازش نتایج صفحات PDF و ایجاد دیتافریم intermediate
        
        Args:
            pdf_results: نتایج پردازش PDF ها
            tag_to_number: نگاشت تگ‌ها به شماره‌ها
            output_path: مسیر فایل خروجی
            
        Returns:
            دیتافریم ایجاد شده
        """
        try:
            intermediate_data = []
            row_counter = 1
            skipped_pages_multiple_jb = 0
            skipped_pages_no_match = 0
            
            logger.info(f"Processing PDF results with {len(pdf_results)} PDFs")
            
            for pdf_name, page_results_dict in pdf_results.items():
                logger.info(f"Processing PDF: {pdf_name}")
                
                if not page_results_dict:
                    logger.warning(f"Empty page results for PDF: {pdf_name}")
                    continue
                    
                for page_num, page_data in page_results_dict.items():
                    logger.info(f"Processing page {page_num} of PDF {pdf_name}")
                    
                    try:
                        # بررسی نوع و ساختار page_data
                        if not isinstance(page_data, (tuple, list)):
                            logger.error(f"Invalid page_data type for page {page_num}: {type(page_data)}")
                            continue
                        
                        if len(page_data) < 5:
                            logger.error(f"Insufficient data in page_data for page {page_num}: {len(page_data)} items")
                            continue
                        
                        # استخراج داده‌ها با امان
                        tags = page_data[0] if len(page_data) > 0 else set()
                        jb_identifiers = page_data[1] if len(page_data) > 1 else set()
                        mc_identifiers = page_data[2] if len(page_data) > 2 else set()
                        cable_descriptions = page_data[3] if len(page_data) > 3 else []
                        spare_identifiers = page_data[4] if len(page_data) > 4 else []
                        page_tag_to_number = page_data[5] if len(page_data) > 5 else {}
                        raw_cable_descriptions = page_data[6] if len(page_data) > 6 else []
                        
                        # تبدیل به لیست در صورت نیاز
                        if isinstance(tags, set):
                            tags = list(tags)
                        if isinstance(jb_identifiers, set):
                            jb_identifiers = list(jb_identifiers)
                        if isinstance(mc_identifiers, set):
                            mc_identifiers = list(mc_identifiers)
                        if isinstance(spare_identifiers, set):
                            spare_identifiers = list(spare_identifiers)
                        
                        logger.debug(f"Extracted data - Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}, Spares: {len(spare_identifiers)}")
                        
                        # شرط 1: اگر بیش از یک JB در صفحه شناسایی شود، تگ‌های آن صفحه وارد اکسل میانی نشوند
                        if len(jb_identifiers) > 1:
                            logger.warning(f"Skipping page {page_num} of PDF {pdf_name} - Multiple JBs detected: {jb_identifiers}")
                            skipped_pages_multiple_jb += 1
                            continue

                        selected_mc = self._select_best_mc_identifier(mc_identifiers, jb_identifiers)
                        
                        # شرط 2: بررسی مطابقت کامل تگ‌ها
                        # ابتدا بررسی می‌کنیم آیا حداقل یکی از تگ‌ها مطابقت کامل دارد
                        has_exact_match = False
                        for tag in tags:
                            if tag in tag_to_number or tag in page_tag_to_number:
                                has_exact_match = True
                                break
                        
                        if not has_exact_match and tags:  # فقط اگر تگی وجود داشته باشد این شرط را بررسی کن
                            logger.warning(f"Skipping page {page_num} of PDF {pdf_name} - No exact tag matches found")
                            skipped_pages_no_match += 1
                            continue
                        
                        # پردازش تگ‌ها
                        for tag in tags:
                            tag_number = tag_to_number.get(tag, page_tag_to_number.get(tag, ''))
                            
                            # ایجاد ردیف داده
                            row_data = {
                                'PDF_Name': pdf_name,
                                'Page': page_num,
                                'Tag/SPARE': tag,
                                'JB': jb_identifiers[0] if jb_identifiers else '',
                                'MC': selected_mc,
                                'Tag_Number': tag_number if tag_number else row_counter,
                                'Wire_Code_1': self.generate_mc_wire_colors(tag_number) if tag_number else '',
                                'Wire_Code_2': '',
                                'Terminal_First_Number': str(tag_number) if tag_number else str(row_counter),
                                'Terminal_Second_Number': str(tag_number + 1) if tag_number else str(row_counter + 1),
                                'SCR_Terminal_Number': self.generate_scr_number(tag_number) if tag_number else '',
                                'Cable_code': cable_descriptions[0] if cable_descriptions else '',
                                'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else (cable_descriptions[0] if cable_descriptions else ''),
                                'Type': 'Tag',
                                'Tag_Number_Status': 'Assigned' if tag_number else 'Auto-Assigned'
                            }
                            
                            intermediate_data.append(row_data)
                            row_counter += 1
                            logger.debug(f"Added tag row: {tag}")
                        
                        # پردازش SPARE ها
                        for spare_idx, spare in enumerate(spare_identifiers):
                            spare_id = f"{getattr(self, 'spare_examples', None)}_{spare_idx + 1}"
                            spare_number = tag_to_number.get(spare_id, page_tag_to_number.get(spare_id, ''))
                            
                            if not spare_number:
                                spare_number = row_counter
                            
                            row_data = {
                                'PDF_Name': pdf_name,
                                'Page': page_num,
                                'Tag/SPARE': spare,
                                'JB': jb_identifiers[0] if jb_identifiers else '',
                                'MC': selected_mc,
                                'Tag_Number': spare_number,
                                'Wire_Code_1': self.generate_mc_wire_colors(spare_number),
                                'Wire_Code_2': '',
                                'Terminal_First_Number': str(spare_number),
                                'Terminal_Second_Number': str(spare_number + 1),
                                'SCR_Terminal_Number': self.generate_scr_number(spare_number),
                                'Cable_code': cable_descriptions[0] if cable_descriptions else '',
                                'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else (cable_descriptions[0] if cable_descriptions else ''),
                                'Type': 'SPARE',
                                'Tag_Number_Status': 'Assigned' if tag_to_number.get(spare_id) else 'Auto-Assigned'
                            }
                            
                            intermediate_data.append(row_data)
                            row_counter += 1
                            logger.debug(f"Added spare row: {spare}")
                    
                    except Exception as e:
                        logger.error(f"Error processing page {page_num} of PDF {pdf_name}: {e}")
                        logger.error(traceback.format_exc())
                        continue
            
            # گزارش تعداد صفحات حذف شده
            if skipped_pages_multiple_jb > 0 or skipped_pages_no_match > 0:
                logger.warning(f"Skipped {skipped_pages_multiple_jb} pages due to multiple JBs")
                logger.warning(f"Skipped {skipped_pages_no_match} pages due to no exact tag matches")
            
            # ایجاد دیتافریم
            if intermediate_data:
                df = pd.DataFrame(intermediate_data)
                logger.info(f"Created intermediate dataframe with {len(df)} rows")
                
                # مرتب‌سازی
                df = df.sort_values(['PDF_Name', 'Page', 'Tag_Number'], na_position='last')
                
                # ذخیره به فایل
                df.to_excel(output_path, index=False)
                logger.info(f"Intermediate Excel file saved to: {output_path}")
                
                # نمایش نمونه داده‌ها
                if len(df) > 0:
                    logger.info(f"Sample data (first 3 rows):")
                    for i, row in df.head(3).iterrows():
                        logger.info(f"  Row {i}: Tag={row['Tag/SPARE']}, JB={row['JB']}, MC={row['MC']}, Number={row['Tag_Number']}")
                
                return df
            else:
                logger.warning("No data to create intermediate DataFrame, creating empty file")
                # ایجاد دیتافریم خالی با ستون‌های مناسب
                empty_df = pd.DataFrame(columns=[
                    'PDF_Name', 'Page', 'Tag/SPARE', 'JB', 'MC', 'Tag_Number',
                    'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number',
                    'SCR_Terminal_Number', 'Cable_code', 'Cable_Description', 'Type', 'Tag_Number_Status'
                ])
                empty_df.to_excel(output_path, index=False)
                logger.info(f"Empty intermediate Excel file saved to: {output_path}")
                return empty_df
                
        except Exception as e:
            logger.error(f"Error in _process_page_results: {e}")
            logger.error(traceback.format_exc())
            
            # ایجاد فایل خالی در صورت خطا
            empty_df = pd.DataFrame(columns=[
                'PDF_Name', 'Page', 'Tag/SPARE', 'JB', 'MC', 'Tag_Number',
                'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number',
                'SCR_Terminal_Number', 'Cable_code', 'Cable_Description', 'Type', 'Tag_Number_Status'
            ])
            empty_df.to_excel(output_path, index=False)
            return empty_df

    def _get_unique_wire_colors(self, tag: str, wire_colors: 'Dict[str, List[str]]', 
                                used_wire_colors: 'Dict[str, Dict[str, bool]]', 
                                tag_to_number: 'Dict[str, int]', as_list: bool = False) -> 'Union[str, List[str]]':
            """
            برای هر تگ، رنگ‌های سیم منحصر به فرد را برمی‌گرداند و از تکرار جلوگیری می‌کند.
            
            Args:
                tag: نام تگ
                wire_colors: دیکشنری رنگ‌های سیم
                used_wire_colors: دیکشنری رنگ‌های استفاده شده
                tag_to_number: دیکشنری شماره تگ‌ها
                as_List: اگر True باشد، لیست رنگ‌ها را برمی‌گرداند، در غیر این صورت رشته
            
            Returns:
                رشته رنگ‌های سیم با کاما جدا شده یا لیست رنگ‌ها
            """
            if tag in wire_colors:
                colors = wire_colors[tag]
                # حذف تکراری‌ها
                unique_colors = list(dict.fromkeys(colors))
                return unique_colors if as_list else ', '.join(unique_colors)
            
            # اگر رنگ برای این تگ تعریف نشده، تولید کن
            if tag not in used_wire_colors:
                used_wire_colors[tag] = {}
            
            # تولید رنگ پیش‌فرض
            tag_num = tag_to_number.get(tag, 1)
            tag_num_str = f"{tag_num:02d}"
            default_colors = [f"BK{tag_num_str}", f"WT{tag_num_str}"]
            
            return default_colors if as_list else ', '.join(default_colors)

    def _process_single_page_data(self, new_df_data: list, page_num: int, page_results: tuple, 
                                pdf_name: str, tag_to_number: dict, io_tags: 'Set[str]' = None):
        """
        ✅ بازنویسی: پردازش داده‌های یک صفحه با استفاده از شماره‌های بر اساس موقعیت
        """
        try:
            # بررسی ساختار
            if not isinstance(page_results, (tuple, list)):
                logger.error(f"Invalid page_results type: {type(page_results)}")
                return
            
            # ✅ FIX: انتظار 9 مقدار
            if len(page_results) < 9:
                logger.error(f"Insufficient data: expected 9, got {len(page_results)}")
                return
            
            # ✅ FIX: استخراج 9 مقدار
            tags = page_results[0] if len(page_results) > 0 else set()
            jb_identifiers = page_results[1] if len(page_results) > 1 else set()
            mc_identifiers = page_results[2] if len(page_results) > 2 else set()
            cable_descriptions = page_results[3] if len(page_results) > 3 else []
            spare_identifiers = page_results[4] if len(page_results) > 4 else []
            page_tag_to_number = page_results[5] if len(page_results) > 5 else {}  # ✅ FIX: این متغیر!
            raw_cable_descriptions = page_results[6] if len(page_results) > 6 else []
            tag_match_info = page_results[7] if len(page_results) > 7 else {}
            all_ocr_tags = page_results[8] if len(page_results) > 8 else set()
            
            logger.debug(f"Page {page_num} - OCR tags: {len(all_ocr_tags)}")
            
            # تبدیل به لیست
            if isinstance(tags, set):
                tags = list(tags)
            if isinstance(jb_identifiers, set):
                jb_identifiers = list(jb_identifiers)
            if isinstance(mc_identifiers, set):
                mc_identifiers = list(mc_identifiers)
            if isinstance(spare_identifiers, set):
                spare_identifiers = list(spare_identifiers)
            
            logger.debug(f"Page {page_num} - Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}, OCR tags: {len(all_ocr_tags)}")
            
            # بررسی شرط multiple JB
            # 🆕 Instead of silently skipping, record a warning so the operator
            # knows data was lost. We still skip the page (ambiguous JB) but the
            # warning surfaces in UI / Excel / annotated PDF.
            if len(jb_identifiers) > 1:
                logger.warning(f"⚠️ Skipping page {page_num}: Multiple JBs {jb_identifiers}")
                if not hasattr(self, '_page_warnings'):
                    self._page_warnings = []
                self._page_warnings.append({
                    'type': 'PAGE_SKIPPED_MULTIPLE_JB',
                    'item': ','.join(sorted(str(j) for j in jb_identifiers)),
                    'pages': [page_num],
                    'tag_count': len(tags),
                    'severity': 'ERROR',
                    'description': (
                        f"Page {page_num} of '{pdf_name}' was SKIPPED because "
                        f"multiple JB identifiers were detected: {sorted(str(j) for j in jb_identifiers)}. "
                        f"{len(tags)} tags on this page were NOT exported."
                    ),
                    'action': (
                        'Manually review the page and choose the correct JB, '
                        'or split the page so it contains a single JB.'
                    ),
                    'pdf_name': pdf_name,
                })
                return

            # 🆕 JB_NOT_FOUND detection: tags/spares exist on this page but no
            # JB was identified. We DO NOT reuse the previous page's JB (that
            # caused wrong assignments). Instead, the JB column is filled with
            # a sentinel value so the operator sees the problem in the Excel.
            _jb_value = jb_identifiers[0] if jb_identifiers else ''
            _jb_not_found_on_page = False
            if not jb_identifiers and (tags or spare_identifiers):
                _jb_not_found_on_page = True
                _jb_value = f'JB_NOT_FOUND (page {page_num})'
                logger.warning(
                    f"⚠️ JB NOT FOUND on page {page_num} of '{pdf_name}' — "
                    f"{len(tags)} tags + {len(spare_identifiers)} spares will be "
                    f"exported with JB='JB_NOT_FOUND (page {page_num})'"
                )
                if not hasattr(self, '_page_warnings'):
                    self._page_warnings = []
                self._page_warnings.append({
                    'type': 'JB_NOT_FOUND',
                    'item': _jb_value,
                    'pages': [page_num],
                    'tag_count': len(tags) + len(spare_identifiers),
                    'severity': 'WARNING',
                    'description': (
                        f"Page {page_num} of '{pdf_name}' contains "
                        f"{len(tags)} tag(s) and {len(spare_identifiers)} spare(s), "
                        f"but NO JB identifier could be detected on this page. "
                        f"Tags were NOT assigned to any JB — please verify the "
                        f"page header manually."
                    ),
                    'action': (
                        'Check the page header (top of page) — the JB label may '
                        'be missing, smudged, or use an unfamiliar prefix. If the '
                        'JB is genuinely absent, split the page or fix the source PDF.'
                    ),
                    'pdf_name': pdf_name,
                })

            selected_mc = self._select_best_mc_identifier(mc_identifiers, jb_identifiers)
            
            # ============================================================
            # پردازش تگ‌ها با استفاده از شماره‌های از قبل تعیین شده
            # ============================================================
            for tag in tags:
                # ✅ FIX: استفاده از page_tag_to_number که حالا تعریف شده
                tag_number = page_tag_to_number.get(tag) or tag_to_number.get(tag)
                
                if not tag_number:
                    logger.warning(f"⚠️ Tag '{tag}' has no number assigned, skipping")
                    continue
                
                # تولید اطلاعات ترمینال
                terminal_info = self.generate_terminal_numbers(tag_number)
                
                # تولید رنگ‌های سیم
                wire_colors_str = self.generate_mc_wire_colors_enhanced(tag_number)
                wire_colors = [c.strip() for c in wire_colors_str.split(',')]
                
                wire_code_1 = wire_colors[0] if len(wire_colors) > 0 else ''
                wire_code_2 = wire_colors[1] if len(wire_colors) > 1 else ''
                
                # تعیین وضعیت match
                match_status = 'Assigned'
                if tag in tag_match_info:
                    info = tag_match_info[tag]
                    match_type = info.get('match_type', 'unknown')
                    match_score = info.get('score', 0.0)
                    
                    if match_type == 'exact':
                        match_status = f'Exact Match (score: {match_score:.3f})'
                    elif match_type == 'similar':
                        match_status = f'Similar Match (score: {match_score:.3f})'
                    else:
                        match_status = f'Unknown: {match_type}'
                
                # 🆕 Per-row warning text (empty when everything is fine)
                _row_warning = ''
                if _jb_not_found_on_page:
                    _row_warning = (
                        f'JB_NOT_FOUND: page {page_num} has tag {tag} but no JB '
                        f'was detected on this page. Tag was NOT assigned to any JB.'
                    )

                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': tag,
                    'JB': _jb_value,
                    'MC': selected_mc,
                    'Tag_Number': tag_number,
                    'Wire_Code_1': wire_code_1,
                    'Wire_Code_2': wire_code_2,
                    'Terminal_First_Number': terminal_info['terminal_first'],
                    'Terminal_Second_Number': terminal_info['terminal_second'],
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': terminal_info['scr_terminal'],
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else '',
                    'Type': 'Tag',
                    'Tag_Number_Status': match_status,
                    'Warning': _row_warning,
                }
                
                new_df_data.append(row_data)
                logger.debug(f"✅ Added tag: {tag} with number #{tag_number}")
            
            # ============================================================
            # پردازش SPAREs
            # ============================================================
            for spare_idx, spare in enumerate(spare_identifiers):
                spare_id = f"{getattr(self, 'spare_examples', None)}_{spare_idx + 1}"
 
                # ── روش ۱: مستقیم از page_tag_to_number ─────────────────
                spare_number = page_tag_to_number.get(spare_id)
 
                # ── روش ۲: از master tag_to_number ───────────────────────
                if not spare_number:
                    spare_number = tag_to_number.get(spare_id)
 
                # ── روش ۳: اگر spare_id با SPARE_ شروع می‌شود،
                #           بررسی کن آیا کلید مشابهی وجود دارد ──────────
                if not spare_number:
                    spare_prefix_key = getattr(self, 'spare_examples', None).strip().upper()
                    for k, v in page_tag_to_number.items():
                        if str(k).upper().startswith(spare_prefix_key + '_'):
                            # بررسی index
                            try:
                                idx_in_key = int(str(k).rsplit('_', 1)[-1])
                                if idx_in_key == spare_idx + 1:
                                    spare_number = v
                                    break
                            except ValueError:
                                pass
 
                # ── اگر هیچ شماره‌ای پیدا نشد → WARNING (نه skip) ────────
                if not spare_number:
                    logger.warning(
                        f"⚠️ WARNING: SPARE '{spare_id}' has no number in tag_to_number "
                        f"(page={page_num}, pdf={pdf_name}). "
                        f"Available keys: {list(page_tag_to_number.keys())[:10]}"
                    )
                    # تخصیص شماره بر اساس موقعیت نسبی (بعد از آخرین تگ)
                    max_existing = max(page_tag_to_number.values()) if page_tag_to_number else 0
                    spare_number = max_existing + spare_idx + 1
                    logger.warning(
                        f"   Auto-assigned number {spare_number} to {spare_id}"
                    )
 
                # ── تولید اطلاعات ترمینال و سیم برای SPARE ────────────────
                terminal_info = self.generate_terminal_numbers(spare_number)
                wire_colors_str = self.generate_mc_wire_colors_enhanced(spare_number)
                wire_colors = [c.strip() for c in wire_colors_str.split(',')]
                wire_code_1 = wire_colors[0] if len(wire_colors) > 0 else ''
                wire_code_2 = wire_colors[1] if len(wire_colors) > 1 else ''
 
                # 🆕 Per-row warning text for SPARE (mirror the JB-not-found logic)
                _row_warning = ''
                if _jb_not_found_on_page:
                    _row_warning = (
                        f'JB_NOT_FOUND: page {page_num} has SPARE {spare} but no JB '
                        f'was detected on this page. SPARE was NOT assigned to any JB.'
                    )

                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': spare,
                    'JB': _jb_value,
                    'MC': selected_mc,
                    'Tag_Number': spare_number,
                    'Wire_Code_1': wire_code_1,
                    'Wire_Code_2': wire_code_2,
                    'Terminal_First_Number': terminal_info['terminal_first'],
                    'Terminal_Second_Number': terminal_info['terminal_second'],
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': terminal_info['scr_terminal'],
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else '',
                    'Type': 'SPARE',
                    'Tag_Number_Status': 'Assigned (Position-based)'
                    if page_tag_to_number.get(spare_id) or tag_to_number.get(spare_id)
                    else 'Auto-assigned (WARNING: not in tag_to_number)',
                    'Warning': _row_warning,
                }
 
                new_df_data.append(row_data)
                logger.info(f"✅ Added SPARE: {spare} → ID: {spare_id} → #{spare_number}")
                
            logger.info(f"✅ Page {page_num} processed: {len([d for d in new_df_data if d.get('Page') == page_num])} rows added")
        
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            logger.error(traceback.format_exc())

    def process_excel_with_io_list(self, intermediate_excel_path: str, excel_path: str, 
                                output_path: str, 
                                all_ocr_tags: 'Set[str]' = None) -> 'Tuple[pd.DataFrame, List[str], List[str]]':
        """
        ترکیب intermediate Excel با IO List و محاسبه unmatched فقط از OCR tags.
        """
        try:
            # ====== خواندن فایل‌ها ======
            intermediate_df = pd.read_excel(intermediate_excel_path)
            io_list_df = pd.read_excel(excel_path)

            intermediate_tag_col = 'Tag/SPARE'
            io_list_tag_col = 'Tag No'

            # نگاشت UPPER -> Original برای intermediate
            intermediate_map_upper_to_original = {str(v).strip().upper(): str(v).strip() 
                                                for v in intermediate_df[intermediate_tag_col] if pd.notna(v)}
            # نگاشت UPPER -> Original برای IO List
            io_map_upper_to_original = {str(v).strip().upper(): str(v).strip() 
                                    for v in io_list_df[io_list_tag_col] if pd.notna(v)}

            intermediate_tags_upper = set(intermediate_map_upper_to_original.keys())
            io_tags_upper = set(io_map_upper_to_original.keys())
            
            # ====== اطلاعات دیباگ ======
            logger.info(f"IO List tags count: {len(io_tags_upper)}")
            logger.info(f"Intermediate tags count: {len(intermediate_tags_upper)}")
            
            # اگر all_ocr_tags خالی است، از page_results استفاده کنیم
            if not all_ocr_tags and hasattr(self, 'page_results'):
                logger.warning("all_ocr_tags is empty, trying to reconstruct from page_results")
                reconstructed_ocr_tags = set()
                for page_num, page_data in self.page_results.items():
                    if isinstance(page_data, tuple) and len(page_data) > 0:
                        page_tags = page_data[0]
                        reconstructed_ocr_tags.update(page_tags)
                
                if reconstructed_ocr_tags:
                    all_ocr_tags = reconstructed_ocr_tags
                    logger.info(f"Reconstructed {len(all_ocr_tags)} OCR tags from page_results")
            
            if all_ocr_tags:
                logger.info(f"all_ocr_tags received: {len(all_ocr_tags)} tags")
                # نمایش نمونه‌ای از تگ‌های OCR
                sample_tags = list(all_ocr_tags)[:10] if len(all_ocr_tags) > 10 else list(all_ocr_tags)
                logger.info(f"Sample OCR tags: {sample_tags}")
            else:
                logger.warning("all_ocr_tags is None or empty! Using intermediate_tags_upper as fallback")
                all_ocr_tags = intermediate_tags_upper
                logger.info(f"Using {len(all_ocr_tags)} intermediate tags as fallback")

            # ====== محاسبه unmatched فقط از OCR ======
            ocr_upper = set(str(tag).strip().upper() for tag in all_ocr_tags if tag and str(tag).strip())
            
            # ── FUZZY MATCHING: OCR tags → IO List tags ─────────────────
            # The previous code only did EXACT matching (ocr_upper - io_tags_upper).
            # This failed for OCR errors like UV-5021 (should be UY-5021).
            # We now do fuzzy matching with character substitutions.
            try:
                import Levenshtein as _lev_fuzzy
                _fuzzy_available = True
            except ImportError:
                _fuzzy_available = False
            
            # Character confusions for OCR error correction
            _CHAR_CONFUSIONS = [
                ('V', 'Y'), ('Y', 'V'),
                ('S', '5'), ('5', 'S'),
                ('0', '5'), ('5', '0'),
                ('1', '7'), ('7', '1'),
                ('O', '0'), ('0', 'O'),
                ('B', '8'), ('8', 'B'),
                ('G', '6'), ('6', 'G'),
                ('Z', '2'), ('2', 'Z'),
                ('I', '1'), ('1', 'I'),
                ('D', '0'), ('0', 'D'),
            ]
            
            # Build OCR→IO List mapping using fuzzy matching
            ocr_to_io_map = {}  # ocr_tag_upper → io_tag_upper
            matched_io_tags = set()
            
            if _fuzzy_available:
                for ocr_tag in ocr_upper:
                    # Skip NC* tokens (MC/cable, not tags)
                    if ocr_tag.startswith('NC'):
                        continue
                    
                    # 1. Try exact match
                    if ocr_tag in io_tags_upper:
                        ocr_to_io_map[ocr_tag] = ocr_tag
                        matched_io_tags.add(ocr_tag)
                        continue
                    
                    # 2. Try fuzzy match (sim >= 0.85)
                    best_io = None
                    best_score = 0.0
                    for io_tag in io_tags_upper:
                        score = _lev_fuzzy.ratio(ocr_tag, io_tag)
                        if score > best_score:
                            best_score = score
                            best_io = io_tag
                    
                    if best_io and best_score >= 0.92:
                        ocr_to_io_map[ocr_tag] = best_io
                        matched_io_tags.add(best_io)
                        logger.info(f"   FUZZY MATCH: '{ocr_tag}' → '{best_io}' (sim={best_score:.2f})")
                        continue
                    
                    # 3. Try character substitutions
                    if best_score < 0.92:
                        for old_ch, new_ch in _CHAR_CONFUSIONS:
                            positions = [j for j, c in enumerate(ocr_tag) if c == old_ch]
                            for pos in positions:
                                candidate = ocr_tag[:pos] + new_ch + ocr_tag[pos+1:]
                                for io_tag in io_tags_upper:
                                    score = _lev_fuzzy.ratio(candidate, io_tag)
                                    if score > best_score:
                                        best_score = score
                                        best_io = io_tag
                        
                        if best_io and best_score >= 0.92:
                            ocr_to_io_map[ocr_tag] = best_io
                            matched_io_tags.add(best_io)
                            logger.info(f"   CHAR-CORRECTED MATCH: '{ocr_tag}' → '{best_io}' (sim={best_score:.2f})")
                            continue
                    
                    # 4. Try double character substitutions
                    if best_score < 0.92:
                        for old_ch1, new_ch1 in _CHAR_CONFUSIONS:
                            pos1_list = [j for j, c in enumerate(ocr_tag) if c == old_ch1]
                            for pos1 in pos1_list:
                                cand1 = ocr_tag[:pos1] + new_ch1 + ocr_tag[pos1+1:]
                                for old_ch2, new_ch2 in _CHAR_CONFUSIONS:
                                    pos2_list = [j for j, c in enumerate(cand1) if c == old_ch2]
                                    for pos2 in pos2_list:
                                        if pos2 == pos1:
                                            continue
                                        cand2 = cand1[:pos2] + new_ch2 + cand1[pos2+1:]
                                        for io_tag in io_tags_upper:
                                            score = _lev_fuzzy.ratio(cand2, io_tag)
                                            if score > best_score:
                                                best_score = score
                                                best_io = io_tag
                        
                        if best_io and best_score >= 0.92:
                            ocr_to_io_map[ocr_tag] = best_io
                            matched_io_tags.add(best_io)
                            logger.info(f"   DOUBLE-CORRECTED MATCH: '{ocr_tag}' → '{best_io}' (sim={best_score:.2f})")
                            continue
                
                logger.info(f"Fuzzy matching: {len(ocr_to_io_map)} OCR tags matched to IO List")
            else:
                # Fallback to exact matching
                for ocr_tag in ocr_upper:
                    if ocr_tag in io_tags_upper:
                        ocr_to_io_map[ocr_tag] = ocr_tag
                        matched_io_tags.add(ocr_tag)
            
            # Unmatched OCR tags = OCR tags that didn't match any IO List tag
            unmatched_pdf_tags_upper = set(ocr_upper) - set(ocr_to_io_map.keys())
            # Also remove NC* tokens from unmatched (they're MC/cable, not tags)
            unmatched_pdf_tags_upper = {t for t in unmatched_pdf_tags_upper if not t.startswith('NC')}
            
            # Unmatched IO tags = IO List tags that no OCR tag matched
            unmatched_io_tags_upper = io_tags_upper - matched_io_tags
            
            logger.info(f"Unmatched OCR tags count: {len(unmatched_pdf_tags_upper)}")
            sample_unmatched = list(unmatched_pdf_tags_upper)[:10]
            logger.info(f"Sample unmatched OCR tags: {sample_unmatched}")
            logger.info(f"Unmatched IO List tags count: {len(unmatched_io_tags_upper)}")

            # ====== ایجاد final DataFrame ======
            final_df = io_list_df.copy()
            intermediate_columns_to_add = [col for col in intermediate_df.columns if col != intermediate_tag_col]
            for col in intermediate_columns_to_add:
                if col not in final_df.columns:
                    final_df[col] = None

            # پر کردن matched rows از intermediate + fuzzy-matched OCR tags
            intermediate_df_indexed = intermediate_df.copy()
            intermediate_df_indexed['_TAG_UPPER_HELPER_'] = intermediate_df_indexed[intermediate_tag_col].apply(lambda x: str(x).strip().upper() if pd.notna(x) else "")
            
            # Build a comprehensive mapping: IO List tag → intermediate row
            # Use BOTH exact matching AND fuzzy OCR→IO mapping
            pdf_to_io_map = intermediate_tags_upper.intersection(io_tags_upper)
            
            # Also add fuzzy-matched OCR tags to the mapping
            # For each OCR→IO fuzzy match, try to find the corresponding intermediate row
            for ocr_tag_upper, io_tag_upper in ocr_to_io_map.items():
                if ocr_tag_upper != io_tag_upper:
                    # OCR tag was fuzzy-corrected to IO tag
                    # Try to find the OCR tag in intermediate
                    if ocr_tag_upper in intermediate_tags_upper:
                        pdf_to_io_map.add(ocr_tag_upper)
                    # Also try the corrected IO tag
                    if io_tag_upper not in pdf_to_io_map:
                        pdf_to_io_map.add(io_tag_upper)
            
            for idx, row in final_df.iterrows():
                io_tag = str(row[io_list_tag_col]).strip().upper() if pd.notna(row[io_list_tag_col]) else ""
                if not io_tag:
                    continue
                
                # 1. Try exact match in intermediate
                if io_tag in pdf_to_io_map:
                    match_row = intermediate_df_indexed[intermediate_df_indexed['_TAG_UPPER_HELPER_'] == io_tag]
                    if not match_row.empty:
                        src_row = match_row.iloc[0]
                        for col in intermediate_columns_to_add:
                            final_df.at[idx, col] = src_row.get(col, None)
                        continue
                
                # 2. Try fuzzy-matched OCR tags
                # Check if any OCR tag was fuzzy-matched to this IO tag
                matched_ocr_tags = [ocr for ocr, io in ocr_to_io_map.items() if io == io_tag]
                for ocr_tag in matched_ocr_tags:
                    if ocr_tag in intermediate_tags_upper:
                        match_row = intermediate_df_indexed[intermediate_df_indexed['_TAG_UPPER_HELPER_'] == ocr_tag]
                        if not match_row.empty:
                            src_row = match_row.iloc[0]
                            for col in intermediate_columns_to_add:
                                final_df.at[idx, col] = src_row.get(col, None)
                            # Also set the Tag No to the IO List form (corrected)
                            break
                    else:
                        # OCR tag was found but not in intermediate (Phase 1/2 failed)
                        # Fill with basic info from OCR
                        for col in intermediate_columns_to_add:
                            if col == 'Type':
                                final_df.at[idx, col] = 'Tag'
                            elif col == 'Tag_Number_Status':
                                final_df.at[idx, col] = 'Matched (Fuzzy OCR)'
                        break
                
                # 3. If still no match but IO tag is in matched_io_tags (fuzzy matched)
                if io_tag in matched_io_tags:
                    # The tag WAS matched via fuzzy OCR, but we don't have intermediate data
                    # Set basic info
                    for col in intermediate_columns_to_add:
                        if col == 'Type' and pd.isna(final_df.at[idx, col]):
                            final_df.at[idx, col] = 'Tag'
                        elif col == 'Tag_Number_Status' and pd.isna(final_df.at[idx, col]):
                            final_df.at[idx, col] = 'Matched (Fuzzy OCR)'

            # ستون تعداد SPARE موجود در هر JB (برای نمایش UI و استفاده در IO Assignment)
            spare_count_col = "JB_SPARE_COUNT"
            jb_col = "JB" if "JB" in final_df.columns else ("JB No" if "JB No" in final_df.columns else None)
            if jb_col:
                intermediate_jb_col = "JB" if "JB" in intermediate_df.columns else None
                if intermediate_jb_col:
                    type_upper = intermediate_df.get("Type", pd.Series([""] * len(intermediate_df))).astype(str).str.strip().str.upper()
                    tag_upper = intermediate_df.get(intermediate_tag_col, pd.Series([""] * len(intermediate_df))).astype(str).str.strip().str.upper()
                    spare_mask = type_upper.eq("SPARE") | tag_upper.str.contains("SPARE", na=False)
                    jb_norm = intermediate_df[intermediate_jb_col].astype(str).str.strip().str.upper()
                    spare_counts_by_jb = (
                        intermediate_df.loc[spare_mask]
                        .assign(_JB_NORM=jb_norm[spare_mask])
                        .groupby("_JB_NORM")
                        .size()
                        .to_dict()
                    )
                    final_df[spare_count_col] = final_df[jb_col].apply(
                        lambda jb: int(spare_counts_by_jb.get(str(jb).strip().upper(), 0))
                    )
                else:
                    final_df[spare_count_col] = 0

            # ذخیره Excel نهایی
            final_df.to_excel(output_path, index=False)

            # تبدیل unmatched به original case
            unmatched_pdf_tags_original = []
            for tag_upper in unmatched_pdf_tags_upper:
                found = False
                if all_ocr_tags:
                    for ocr_tag in all_ocr_tags:
                        if str(ocr_tag).strip().upper() == tag_upper:
                            unmatched_pdf_tags_original.append(str(ocr_tag).strip())
                            found = True
                            break
                if not found:
                    unmatched_pdf_tags_original.append(tag_upper)

            unmatched_io_tags_original = [io_map_upper_to_original.get(tag, tag) for tag in unmatched_io_tags_upper]
            
            logger.info(f"Final unmatched OCR tags count: {len(unmatched_pdf_tags_original)}")
            logger.info(f"Final unmatched IO List tags count: {len(unmatched_io_tags_original)}")

            return final_df, sorted(unmatched_io_tags_original), sorted(unmatched_pdf_tags_original)

        except Exception as e:
            logger.error(f"❌ Error in process_excel_with_io_list: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame(), [], []

    def check_tag_number_consistency(self, tag_to_number: 'Dict[str, int]') -> 'Tuple[bool, int, int]':
        """
        بررسی می‌کند که آیا بزرگترین شماره تگ با شماره زوج در توضیحات کابل مطابقت دارد یا خیر.
        
        Args:
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها از bounding box
                
        Returns:
            Tuple of (is_consistent, max_tag_number, extracted_pair_number)
        """
        try:
            # اگر tag_to_number خالی است، مقادیر پیش‌فرض را برگردان
            if not tag_to_number:
                logger.warning("tag_to_number is empty, returning default values")
                return True, 0, 0
            
            # اطلاعات دیباگ برای بررسی مقادیر tag_to_number
            logger.debug(f"Tag to number Dictionary: {tag_to_number}")
            
            # پیدا کردن بزرگترین شماره تگ - با بررسی دقیق‌تر
            # ابتدا همه مقادیر را به عنوان لیست استخراج می‌کنیم
            all_tag_numbers = list(tag_to_number.values())
            logger.debug(f"All tag numbers: {all_tag_numbers}")
            
            # بزرگترین شماره تگ را پیدا می‌کنیم
            if all_tag_numbers:
                max_tag_number = max(all_tag_numbers)
                logger.debug(f"Maximum tag number from all tags: {max_tag_number}")
            else:
                max_tag_number = 0
                logger.warning("No tag numbers found in tag_to_number Dictionary")
            
            # استخراج شماره زوج از توضیحات کابل
            cable_descriptions = []
            
            # بررسی کنیم آیا در صفحه فعلی توضیحات کابل داریم
            if hasattr(self, 'cable_descriptions') and self.cable_descriptions:
                cable_descriptions.extend(self.cable_descriptions)
                logger.debug(f"Cable descriptions from current page: {self.cable_descriptions}")
            
            # مقدار پیش‌فرض برای شماره زوج
            extracted_pair_number = 0
            matched_description = ""
            
            # استخراج شماره زوج از توضیحات کابل
            for desc in cable_descriptions:
                if not desc:
                    continue
                
                # الگوهای مختلف برای استخراج شماره زوج
                pair_patterns = [
                    r'(\d+)\s*(?:pair|P|PR)',  # مثل "12 pair", "12P", "12 P"
                    r'(\d+)P',                 # مثل "12P"
                    r'(\d+)\s*PAIR',           # مثل "12 PAIR"
                    r'(\d+)\s*CORE',           # مثل "12 CORE"
                    r'(\d+)\s*C',              # مثل "12 C"
                    r'(\d+)C'                  # مثل "12C"
                ]
                
                desc_str = str(desc).upper()  # تبدیل به رشته و حروف بزرگ برای جستجوی بهتر
                logger.debug(f"Processing cable description: {desc_str}")
                
                for pattern in pair_patterns:
                    match = re.search(pattern, desc_str, re.IGNORECASE)
                    if match:
                        try:
                            pair_number = int(match.group(1))
                            logger.debug(f"Found potential pair number {pair_number} with pattern {pattern} in: {desc_str}")
                            if pair_number > extracted_pair_number:
                                extracted_pair_number = pair_number
                                matched_description = desc_str
                                logger.info(f"Found pair number {pair_number} in cable description: {desc_str}")
                        except ValueError:
                            logger.debug(f"Could not convert {match.group(1)} to integer")
                            continue
            
            # اگر هیچ شماره زوجی پیدا نشد، سعی کنیم از سایر منابع استخراج کنیم
            if extracted_pair_number == 0:
                # بررسی اطلاعات در page_results
                if hasattr(self, 'page_results') and self.page_results:
                    logger.debug(f"Checking page_results for cable descriptions")
                    for page_num, page_data in self.page_results.items():
                        if isinstance(page_data, tuple) and len(page_data) >= 4:
                            page_cable_descriptions = page_data[3]
                            logger.debug(f"Cable descriptions from page {page_num}: {page_cable_descriptions}")
                            for desc in page_cable_descriptions:
                                desc_str = str(desc).upper()
                                for pattern in pair_patterns:
                                    match = re.search(pattern, desc_str, re.IGNORECASE)
                                    if match:
                                        try:
                                            pair_number = int(match.group(1))
                                            logger.debug(f"Found potential pair number {pair_number} with pattern {pattern} in page {page_num}: {desc_str}")
                                            if pair_number > extracted_pair_number:
                                                extracted_pair_number = pair_number
                                                matched_description = desc_str
                                                logger.info(f"Found pair number {pair_number} in page {page_num} cable description: {desc_str}")
                                        except ValueError:
                                            continue
            
            # اگر هنوز هیچ شماره زوجی پیدا نشد، از تعداد تگ‌ها استفاده کنیم
            if extracted_pair_number == 0 and tag_to_number:
                # تعداد تگ‌ها (بدون SPARE) را به عنوان تخمینی از شماره زوج استفاده کنیم
                non_spare_tags = [tag for tag in tag_to_number.keys() if not str(tag).upper().startswith('SPARE')]
                tag_count = len(non_spare_tags)
                if tag_count > 0:
                    extracted_pair_number = tag_count
                    logger.info(f"No pair number found in cable descriptions, using tag count: {tag_count}")
            
            # بررسی تطابق - با دقت بیشتر
            if extracted_pair_number == 0:
                logger.warning("Could not extract pair number, skipping consistency check")
                return True, max_tag_number, 0
            
            # بررسی دقیق تطابق - بدون تلرانس
            is_consistent = max_tag_number == extracted_pair_number
            
            # اگر مطابقت نداشت، پیام هشدار مناسب را ثبت کن
            if not is_consistent:
                if max_tag_number > extracted_pair_number:
                    logger.warning(f"WARNING: Maximum tag number ({max_tag_number}) is GREATER than cable pair number ({extracted_pair_number}) from '{matched_description}'")
                else:
                    logger.warning(f"WARNING: Maximum tag number ({max_tag_number}) is LESS than cable pair number ({extracted_pair_number}) from '{matched_description}'")
            else:
                logger.info(f"Tag number consistency check PASSED: max_tag_number={max_tag_number}, pair_number={extracted_pair_number} from '{matched_description}'")
            
            # برای اطمینان از صحت مقادیر، اطلاعات دیباگ بیشتری اضافه می‌کنیم
            logger.debug(f"Final values: is_consistent={is_consistent}, max_tag_number={max_tag_number}, extracted_pair_number={extracted_pair_number}")
            
            return is_consistent, max_tag_number, extracted_pair_number
                
        except Exception as e:
            logger.error(f"Error checking tag number consistency: {e}")
            
            logger.error(traceback.format_exc())
            return False, 0, 0
        
    # اصلاح تابع get_processing_stats برای رفع خطای تقسیم set بر int
    def get_processing_stats(self) -> 'Dict[str, Any]':
        """
        بازگرداندن آمار پردازش با پشتیبانی از PDF های چندصفحه‌ای
        """
        try:
            # اطمینان از اینکه متغیرها به درستی مقداردهی شده‌اند
            if not hasattr(self, 'all_tags'):
                self.all_tags = set()
            if not hasattr(self, 'matched_tags_set'):
                self.matched_tags_set = set()
            
            total_tags = getattr(self, 'total_tags', len(self.all_tags))
            matched_tags = getattr(self, 'matched_tags', len(self.matched_tags_set))
            exact_matches = getattr(self, 'exact_matches', 0)
            similar_matches = getattr(self, 'similar_matches', 0)
            total_jbs = len(getattr(self, 'all_jbs', set()))
            processing_time = getattr(self, 'processing_time', 0)
            
            # محاسبه نرخ تطبیق - با بررسی تقسیم بر صفر
            match_rate = f"{(matched_tags / total_tags * 100):.1f}%" if total_tags > 0 else "0.0%"
            exact_match_rate = f"{(exact_matches / total_tags * 100):.0f}%" if total_tags > 0 else "0%"
            
            return {
                'total_tags': total_tags,
                'matched_tags': matched_tags,
                'exact_matches': exact_matches,
                'similar_matches': similar_matches,
                'total_jbs': total_jbs,
                'processing_time': f"{processing_time:.2f} seconds" if processing_time else "0.00 seconds",
                'match_rate': match_rate,
                'exact_match_rate': exact_match_rate,
                'unmatched_tags': total_tags - matched_tags
            }
            
        except Exception as e:
            logger.error(f"Error calculating processing stats: {e}")
            return {
                'total_tags': 0,
                'matched_tags': 0,
                'exact_matches': 0,
                'similar_matches': 0,
                'total_jbs': 0,
                'processing_time': '0.00 seconds',
                'match_rate': '0.0%',
                'exact_match_rate': '0%',
                'unmatched_tags': 0
            }

    def check_page_io_coverage(self, page_tags: 'Set[str]', io_tags: 'Set[str]', threshold: float = 0.8) -> 'Tuple[bool, float, List[str]]':
        """
        بررسی می‌کند که آیا تگ‌های یک صفحه در IO List هستند یا خیر.
        
        Args:
            page_tags: تگ‌های استخراج شده از صفحه
            io_tags: تگ‌های IO List
            threshold: حداقل درصد تطابق برای تایید صفحه (پیش‌فرض: 80%)
        
        Returns:
            Tuple of (is_valid_page, coverage_percentage, missing_tags)
        """
        if not page_tags:
            return True, 100.0, []
        
        # تبدیل به مجموعه uppercase
        page_tags_upper = set(str(self.fix_common_ocr_errors(tag)).strip().upper() for tag in page_tags)
        io_tags_upper = set(str(tag).strip().upper() for tag in io_tags if tag and not pd.isna(tag))
        
        # تعداد تگ‌هایی که در IO List هستند
        matched_count = len(page_tags_upper.intersection(io_tags_upper))
        
        # درصد تطابق
        coverage = (matched_count / len(page_tags_upper)) * 100 if page_tags_upper else 100.0
        
        # لیست تگ‌های موجود در صفحه ولی نه در IO List
        missing_tags = list(page_tags_upper - io_tags_upper)
        
        # آیا صفحه معتبر است؟
        is_valid = coverage >= (threshold * 100)
        
        logger.info(f"Page coverage: {coverage:.1f}% ({matched_count}/{len(page_tags_upper)} tags matched)")
        if missing_tags:
            logger.warning(f"Missing tags in IO List: {missing_tags}")
        
        return is_valid, coverage, missing_tags
    
    def generate_processing_report(self, pdf_results: dict, io_tags: set, output_path: str = None):
        """
        تولید گزارش کامل با جزئیات exact/similar matches
        """
        report = {
            'total_pdfs': len(pdf_results),
            'total_pages': 0,
            'valid_pages': 0,
            'invalid_pages': 0,
            'pages_with_exact_matches': 0,
            'pages_without_exact_matches': 0,
            'skipped_pages': [],
            'duplicate_warnings': [],
            'missing_tags': [],
            'page_details': []
        }
        
        for pdf_name, page_results_dict in pdf_results.items():
            for page_num, page_results in page_results_dict.items():
                report['total_pages'] += 1
                
                tags = page_results[0] if len(page_results) > 0 else set()
                jb_identifiers = page_results[1] if len(page_results) > 1 else set()
                
                # 🆕 بررسی exact matches
                tags_upper = set(str(tag).strip().upper() for tag in tags)
                io_tags_upper = set(str(tag).strip().upper() for tag in io_tags if tag and not pd.isna(tag))
                exact_matches = tags_upper.intersection(io_tags_upper)
                has_exact_match = len(exact_matches) > 0
                
                # بررسی اعتبار
                is_valid, coverage, missing = self.check_page_io_coverage(tags, io_tags)
                
                page_info = {
                    'pdf': pdf_name,
                    'page': page_num,
                    'tags_count': len(tags),
                    'jb_count': len(jb_identifiers),
                    'exact_matches': len(exact_matches),
                    'coverage': f"{coverage:.1f}%",
                    'status': 'Valid' if is_valid else 'Invalid',
                    'has_exact_match': has_exact_match
                }
                
                # 🆕 شمارش صفحات با/بدون exact match
                if has_exact_match:
                    report['pages_with_exact_matches'] += 1
                else:
                    report['pages_without_exact_matches'] += 1
                
                # بررسی شروط رد شدن
                skip_reason = None
                
                # 🆕 شرط 1: عدم وجود exact match
                if not has_exact_match and tags:
                    skip_reason = f"No exact matches found (similar only)"
                    report['invalid_pages'] += 1
                    report['skipped_pages'].append({
                        'location': f"{pdf_name} - Page {page_num}",
                        'reason': skip_reason,
                        'tags': list(tags_upper)[:5]
                    })
                    page_info['status'] = 'Skipped'
                    page_info['reason'] = skip_reason
                
                # شرط 2: چندین JB
                elif len(jb_identifiers) > 1:
                    skip_reason = f"Multiple JBs: {list(jb_identifiers)}"
                    report['invalid_pages'] += 1
                    report['skipped_pages'].append({
                        'location': f"{pdf_name} - Page {page_num}",
                        'reason': skip_reason,
                        'jbs': list(jb_identifiers)
                    })
                    page_info['status'] = 'Skipped'
                    page_info['reason'] = skip_reason
                
                # صفحه معتبر
                else:
                    report['valid_pages'] += 1
                    if missing:
                        report['missing_tags'].extend([
                            f"{pdf_name} - Page {page_num}: {tag}" for tag in missing
                        ])
                
                report['page_details'].append(page_info)
        
        # ============================================================
        # چاپ گزارش تفصیلی
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("📊 PROCESSING REPORT - DETAILED STATISTICS")
        logger.info("="*70)
        logger.info(f"Total PDFs processed: {report['total_pdfs']}")
        logger.info(f"Total pages scanned: {report['total_pages']}")
        logger.info("")
        logger.info("─" * 70)
        logger.info("PAGE VALIDATION:")
        logger.info(f"  ✅ Valid pages (added to intermediate): {report['valid_pages']} ({report['valid_pages']/report['total_pages']*100:.1f}%)")
        logger.info(f"  ❌ Invalid/Skipped pages: {report['invalid_pages']} ({report['invalid_pages']/report['total_pages']*100:.1f}%)")
        logger.info("")
        logger.info("─" * 70)
        logger.info("EXACT MATCH ANALYSIS:")
        logger.info(f"  ✅ Pages with exact matches: {report['pages_with_exact_matches']}")
        logger.info(f"  ❌ Pages without exact matches (skipped): {report['pages_without_exact_matches']}")
        
        # 🆕 جزئیات صفحات رد شده
        if report['skipped_pages']:
            logger.info("")
            logger.info("─" * 70)
            logger.warning(f"⚠️ SKIPPED PAGES DETAILS ({len(report['skipped_pages'])} total):")
            
            # گروه‌بندی بر اساس دلیل
            no_exact_match_pages = [p for p in report['skipped_pages'] if 'No exact matches' in p['reason']]
            multiple_jb_pages = [p for p in report['skipped_pages'] if 'Multiple JBs' in p['reason']]
            
            if no_exact_match_pages:
                logger.warning(f"\n  📋 No Exact Matches ({len(no_exact_match_pages)} pages):")
                for page in no_exact_match_pages[:5]:  # نمایش 5 مورد اول
                    logger.warning(f"     • {page['location']}")
                    if 'tags' in page:
                        logger.warning(f"       Tags found: {page['tags']}")
                if len(no_exact_match_pages) > 5:
                    logger.warning(f"     ... and {len(no_exact_match_pages) - 5} more")
            
            if multiple_jb_pages:
                logger.warning(f"\n  📋 Multiple JBs ({len(multiple_jb_pages)} pages):")
                for page in multiple_jb_pages[:5]:
                    logger.warning(f"     • {page['location']}")
                    if 'jbs' in page:
                        logger.warning(f"       JBs found: {page['jbs']}")
                if len(multiple_jb_pages) > 5:
                    logger.warning(f"     ... and {len(multiple_jb_pages) - 5} more")
        
        # تگ‌های موجود در صفحات ولی نه در IO List
        if report['missing_tags']:
            logger.info("")
            logger.info("─" * 70)
            logger.warning(f"⚠️ TAGS FOUND IN PAGES BUT NOT IN IO LIST ({len(report['missing_tags'])} total):")
            unique_missing = list(set(report['missing_tags']))[:10]
            for tag_info in unique_missing:
                logger.warning(f"  • {tag_info}")
            if len(report['missing_tags']) > 10:
                logger.warning(f"  ... and {len(report['missing_tags']) - 10} more")
        
        logger.info("")
        logger.info("="*70 + "\n")
        
        # ============================================================
        # 🆕 ذخیره گزارش JSON با جزئیات بیشتر
        # ============================================================
        if output_path:
            import json
            
            # افزودن خلاصه‌ای از تگ‌های exact/similar
            report['summary'] = {
                'validation_rules': [
                    'Rule 1: At least one EXACT match required',
                    'Rule 2: Only one JB per page allowed',
                    'Rule 3: Similar matches ignored if no exact match exists'
                ],
                'exact_match_requirement': 'ENFORCED',
                'pages_accepted': report['valid_pages'],
                'pages_rejected': report['invalid_pages']
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Detailed JSON report saved to: {output_path}")
        
        return report
    
    def check_and_resolve_duplicates(self, tags_dict: dict) -> dict:
        """
        بررسی و حل مشکل duplicate ها با اولویت exact match
        
        Args:
            tags_dict: دیکشنری {tag_name: {'match_type': ..., 'score': ..., 'ocr_text': ...}}
        
        Returns:
            دیکشنری تمیز شده بدون duplicate
        """
        cleaned_tags = {}
        tag_occurrences = {}  # {tag_name: [match_info1, match_info2, ...]}
        
        # گروه‌بندی تگ‌های یکسان
        for tag, info in tags_dict.items():
            fixed_tag = self.fix_common_ocr_errors(tag)
            tag_upper = tag.upper()
            if tag_upper not in tag_occurrences:
                tag_occurrences[tag_upper] = []
            tag_occurrences[tag_upper].append({'tag': tag, 'info': info})
        
        # حل duplicate ها
        for tag_upper, occurrences in tag_occurrences.items():
            if len(occurrences) == 1:
                # تگ منحصر به فرد
                tag = occurrences[0]['tag']
                cleaned_tags[tag] = occurrences[0]['info']
            else:
                # چندین occurrence از یک تگ
                logger.warning(f"⚠️ Found {len(occurrences)} occurrences of tag: {tag_upper}")
                
                # جستجو برای exact match
                exact_matches = [occ for occ in occurrences if occ['info']['match_type'] == 'exact']
                
                if exact_matches:
                    # اولویت با exact match
                    selected = exact_matches[0]
                    logger.info(f"✅ Selected EXACT match for {tag_upper}")
                    cleaned_tags[selected['tag']] = selected['info']
                    
                    # لاگ موارد رد شده
                    for occ in occurrences:
                        if occ != selected:
                            logger.debug(f"   Rejected: {occ['info']['ocr_text']} → {occ['tag']} "
                                    f"(type: {occ['info']['match_type']}, score: {occ['info']['score']:.3f})")
                else:
                    # همه similar هستند، بالاترین score را انتخاب کن
                    selected = max(occurrences, key=lambda x: x['info']['score'])
                    logger.info(f"⚠️ No exact match, selected SIMILAR with highest score for {tag_upper}")
                    cleaned_tags[selected['tag']] = selected['info']
                    
                    # لاگ موارد رد شده
                    for occ in occurrences:
                        if occ != selected:
                            logger.debug(f"   Rejected: {occ['info']['ocr_text']} → {occ['tag']} "
                                    f"(score: {occ['info']['score']:.3f})")
        
        logger.info(f"Duplicate resolution: {len(tags_dict)} → {len(cleaned_tags)} unique tags")
        return cleaned_tags

    def test_match_info_flow(self):
        """
        تست جریان tag_match_info از extract تا draw
        """
        logger.info("="*70)
        logger.info("🧪 TESTING MATCH INFO FLOW")
        logger.info("="*70)
        
        # مرحله 1: ساخت تصویر ساده تست
        import numpy as np
        test_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        try:
            # مرحله 2: استخراج
            logger.info("Step 1: Extracting from test image...")
            result = self.extract_from_image(test_image)
            
            if len(result) < 8:
                logger.error(f"❌ FAIL: extract_from_image returned {len(result)} values, expected 8")
                return False
            
            tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info , all_ocr_tags = result
            
            logger.info(f"✅ extract_from_image returned 8 values")
            logger.info(f"   - tags: {len(tags)}")
            logger.info(f"   - tag_match_info: {len(tag_match_info)}")
            
            # مرحله 3: بررسی tag_match_info
            logger.info("Step 2: Validating tag_match_info structure...")
            
            for tag in tags:
                if tag not in tag_match_info:
                    logger.error(f"❌ FAIL: Tag '{tag}' not in tag_match_info")
                    return False
                
                info = tag_match_info[tag]
                if 'match_type' not in info:
                    logger.error(f"❌ FAIL: Tag '{tag}' has no 'match_type'")
                    return False
                
                match_type = info['match_type']
                if match_type not in ['exact', 'similar']:
                    logger.error(f"❌ FAIL: Tag '{tag}' has invalid match_type: {match_type}")
                    return False
            
            logger.info(f"✅ All {len(tags)} tags have valid match_type")
            
            # مرحله 4: تست draw_bounding_boxes
            logger.info("Step 3: Testing draw_bounding_boxes...")
            
            try:
                annotated_image, updated_tag_numbers = self.draw_bounding_boxes(
                    test_image, tags, jb_identifiers, mc_identifiers,
                    cable_descriptions, spare_identifiers, tag_to_number,
                    tag_match_info ,all_ocr_tags
                )
                logger.info(f"✅ draw_bounding_boxes executed successfully")
            except Exception as e:
                logger.error(f"❌ FAIL: draw_bounding_boxes raised exception: {e}")
                return False
            
            logger.info("="*70)
            logger.info("✅ ALL TESTS PASSED")
            logger.info("="*70)
            return True
            
        except Exception as e:
            logger.error(f"❌ TEST FAILED: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def clean_text_for_display(self, text):
        """
        پاکسازی متن برای نمایش توسط OpenCV
        """
        if not text:
            return ""
        
        # حذف کاراکترهای کنترلی و نامرئی
        text = ''.join(c if c.isprintable() else '' for c in text)
        
        # تبدیل به ASCII برای اطمینان از سازگاری با فونت OpenCV
        text = text.encode('ascii', 'replace').decode('ascii')
        
        # جایگزینی علامت‌های سؤال با خط تیره
        text = text.replace('???', '-').replace('??', '-').replace('?', '-')
        
        return text

    def _resolve_uzso_uzsc(self, prefix_type: str, number: str, original_tag: str) -> str:
        """
        تعیین قطعی نوع UZSO یا UZSC با استفاده از IO List و context
        
        Args:
            prefix_type: 'UZSO' یا 'UZSC' (حدس اولیه)
            number: شماره تگ
            original_tag: تگ اصلی OCR شده
            
        Returns:
            تگ تصحیح شده
        """
        potential_uzso = f"UZSO-{number}"
        potential_uzsc = f"UZSC-{number}"
        
        # استراتژی 1: بررسی IO List
        if hasattr(self, 'vector_matcher') and hasattr(self.vector_matcher, 'tag_vectors'):
            io_tags = set(self.vector_matcher.tag_vectors.keys())
            
            uzso_exists = potential_uzso in io_tags
            uzsc_exists = potential_uzsc in io_tags
            
            if uzso_exists and not uzsc_exists:
                logger.info(f"✅ OCR fix (IO List): {original_tag} -> {potential_uzso}")
                return potential_uzso
            elif uzsc_exists and not uzso_exists:
                logger.info(f"✅ OCR fix (IO List): {original_tag} -> {potential_uzsc}")
                return potential_uzsc
            elif uzso_exists and uzsc_exists:
                # هر دو وجود دارند - استفاده از context یا حدس اولیه
                logger.warning(f"⚠️ Both UZSO and UZSC exist for {number}, using prefix_type: {prefix_type}")
        
        # استراتژی 2: استفاده از context صفحه
        if hasattr(self, '_current_page_dominant_prefix') and self._current_page_dominant_prefix:
            dominant = self._current_page_dominant_prefix
            corrected = f"{dominant}-{number}"
            logger.info(f"✅ OCR fix (page context): {original_tag} -> {corrected}")
            return corrected
        
        # استراتژی 3: استفاده از حدس اولیه
        corrected = f"{prefix_type}-{number}"
        logger.info(f"✅ OCR fix (prefix guess): {original_tag} -> {corrected}")
        return corrected
    
    def fix_common_ocr_errors(self, tag: str) -> str:
        """
        ✅ ENHANCED: تشخیص قوی UZSO/UZSC با الگوهای گسترده‌تر OCR
        """
        if not tag:
            return tag
        
        original_tag = tag
        tag_upper = tag.upper().strip()
        
        # ============================================================
        # 🔧 FIX 1: الگوی بسیار گسترده برای UZSO/UZSC
        # ============================================================
        # الگوهای ممکن OCR:
        # - UZSO, UZS0, UZ50, UZSO, UZSo, VZS0, VZSO, U2SO, UZS (بدون O)
        # - UZSC, UZSC, U2SC, VZSC, UZS (بدون C)
        # - U250, U25C (حروف به جای اعداد)
        
        # Pattern 1: حالت کامل با خطاهای احتمالی
        uzso_pattern_full = r'^[UuVv][ZzSs2]?[Ss5]?[O0oDdQq][-_]?(\d+)$'
        uzsc_pattern_full = r'^[UuVv][ZzSs2]?[Ss5]?[CcGgQq][-_]?(\d+)$'
        
        # Pattern 2: حالت ناقص (فقط UZS یا UZ + شماره)
        uzs_pattern_incomplete = r'^[UuVv][ZzSs2][Ss5]?[-_]?(\d+)$'
        
        # Pattern 3: حالت با اعداد به جای حروف
        number_as_letter_pattern = r'^[UuVv]2[Ss5]?[O0oDdCcGgQq]?[-_]?(\d+)$'
        
        # بررسی Pattern 3: اعداد به جای حروف
        match_number_as_letter = re.match(number_as_letter_pattern, tag_upper)
        if match_number_as_letter:
            number = match_number_as_letter.group(1)
            # تشخیص نوع (UZSO یا UZSC) بر اساس آخرین حرف
            last_char = re.search(r'([O0oDdCcGgQq])', tag_upper)
            if last_char:
                char = last_char.group(1)
                if char in 'O0oDdQq':
                    return self._resolve_uzso_uzsc('UZSO', number, original_tag)
                else:
                    return self._resolve_uzso_uzsc('UZSC', number, original_tag)
            else:
                # اگر حرف آخر تشخیص داده نشد، از context استفاده کن
                return self._resolve_uzso_uzsc('UZSO', number, original_tag)  # پیش‌فرض UZSO
        
        # بررسی Pattern 1: UZSO
        match_uzso = re.match(uzso_pattern_full, tag_upper)
        if match_uzso:
            number = match_uzso.group(1)
            return self._resolve_uzso_uzsc('UZSO', number, original_tag)
        
        # بررسی Pattern 1: UZSC
        match_uzsc = re.match(uzsc_pattern_full, tag_upper)
        if match_uzsc:
            number = match_uzsc.group(1)
            return self._resolve_uzso_uzsc('UZSC', number, original_tag)
        
        # بررسی Pattern 2: ناقص (UZS + number)
        match_incomplete = re.match(uzs_pattern_incomplete, tag_upper)
        if match_incomplete:
            number = match_incomplete.group(1)
            logger.info(f"🔍 Incomplete UZSO/UZSC pattern detected: {original_tag}")
            
            # بررسی IO List برای تعیین نوع
            if hasattr(self, 'vector_matcher') and hasattr(self.vector_matcher, 'tag_vectors'):
                io_tags = set(self.vector_matcher.tag_vectors.keys())
                
                potential_uzso = f"UZSO-{number}"
                potential_uzsc = f"UZSC-{number}"
                
                if potential_uzso in io_tags:
                    logger.info(f"✅ OCR fix (incomplete -> UZSO): {original_tag} -> {potential_uzso}")
                    return potential_uzso
                elif potential_uzsc in io_tags:
                    logger.info(f"✅ OCR fix (incomplete -> UZSC): {original_tag} -> {potential_uzsc}")
                    return potential_uzsc
            
            # استفاده از context صفحه
            if hasattr(self, '_current_page_dominant_prefix') and self._current_page_dominant_prefix:
                dominant = self._current_page_dominant_prefix
                corrected = f"{dominant}-{number}"
                logger.info(f"✅ OCR fix (incomplete -> page context): {original_tag} -> {corrected}")
                return corrected
            
            # Default: UZSO
            logger.info(f"✅ OCR fix (incomplete -> default UZSO): {original_tag} -> UZSO-{number}")
            return f"UZSO-{number}"
        
        # ============================================================
        # 🔧 FIX 2: بررسی حالت‌های خاص OCR
        # ============================================================
        # حالت خاص: کاراکترهای جدا شده با فاصله (مثلاً "U Z S O 100 01")
        if ' ' in tag:
            tag_no_space = tag.replace(' ', '').replace('-', '')
            return self.fix_common_ocr_errors(tag_no_space)  # بازگشتی
        
        # حالت خاص: اعداد درون پیشوند (مثلاً "UZ5O" به جای "UZSO")
        if re.match(r'^[UuVv][ZzSs2]5[O0oDdQq][-_]?(\d+)$', tag_upper):
            number = re.search(r'(\d+)$', tag_upper).group(1)
            return self._resolve_uzso_uzsc('UZSO', number, original_tag)
        
        if re.match(r'^[UuVv][ZzSs2]5[CcGgQq][-_]?(\d+)$', tag_upper):
            number = re.search(r'(\d+)$', tag_upper).group(1)
            return self._resolve_uzso_uzsc('UZSC', number, original_tag)
        
        # ============================================================
        # 🔧 FIX 3: اصلاح اعداد و حروف مشابه
        # ============================================================
        parts = re.split(r'(-)', tag)
        fixed_parts = []
        
        for p in parts:
            if p == '-':
                fixed_parts.append(p)
                continue
                
            sub = p
            
            # اصلاح اعداد و حروف مشابه
            if re.search(r'\d', sub):
                sub = sub.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('Q', '0')
            
            # اصلاح حروف
            sub = sub.replace('0', 'O', 1) if sub.startswith('U') and len(sub) >= 3 and sub[2] == '0' else sub
            sub = sub.replace('5', 'S', 1) if sub.startswith('U') and len(sub) >= 2 and sub[1] == '5' else sub
            
            # حذف فضاهای خالی
            sub = re.sub(r'\s+', '', sub)
            fixed_parts.append(sub)
        
        result = ''.join(fixed_parts)
        
        if result != original_tag:
            logger.debug(f"OCR fix (general): '{original_tag}' -> '{result}'")
        
        return result

    def _is_fuzzy_match(self, ocr_text: str, reference: str, threshold: float = 0.75) -> bool:
        """
        بررسی شباهت fuzzy بین متن OCR و مرجع
        """
        if not ocr_text or not reference:
            return False
        
        # Levenshtein distance
        distance = Levenshtein.distance(ocr_text.upper(), reference.upper())
        max_len = max(len(ocr_text), len(reference))
        
        if max_len == 0:
            return True
        
        similarity = 1.0 - (distance / max_len)
        return similarity >= threshold
    
    def _detect_dominant_prefix_in_page(self, ocr_data, candidate_prefixes: List[str] = ['UZSO', 'UZSC']) -> str:
        """
        ✅ IMPROVED: تشخیص پیشوند غالب با الگوهای گسترده OCR
        
        Args:
            ocr_data: می‌تواند دیکشنری OCR یا یک رشته متن باشد
            candidate_prefixes: لیست پیشوندهای کاندیدا
        
        Returns:
            پیشوند غالب یا None
        """
        prefix_counts = {prefix: 0 for prefix in candidate_prefixes}
        
        logger.info("="*70)
        logger.info("🔍 Detecting dominant prefix with improved pattern matching...")
        
        # ✅ الگوهای گسترده برای هر پیشوند
        uzso_patterns = [
            r'[UuVv][ZzSs2][Ss5][O0oDd][-_]?\d+',      # Standard
            r'[UuVv][ZzSs2]5[O0oDd][-_]?\d+',          # UZ5O
        ]
        
        uzsc_patterns = [
            r'[UuVv][ZzSs2][Ss5][CcGg][-_]?\d+',      # Standard
            r'[UuVv][ZzSs2]5[CcGg][-_]?\d+',          # UZ5C
        ]
        
        # تعریف الگوهای جستجو برای هر پیشوند
        prefix_patterns = {
            'UZSO': uzso_patterns,
            'UZSC': uzsc_patterns,
        }
        
        # بررسی نوع داده ورودی و استخراج متن
        texts_to_process = []
        
        if isinstance(ocr_data, dict) and 'text' in ocr_data:
            # حالت دیکشنری OCR
            for text_item in ocr_data.get('text', []):
                if isinstance(text_item, dict) and 'text' in text_item:
                    texts_to_process.append(text_item.get('text', '').upper())
                elif isinstance(text_item, str):
                    texts_to_process.append(text_item.upper())
        elif isinstance(ocr_data, str):
            # حالت رشته ساده
            texts_to_process.append(ocr_data.upper())
        elif isinstance(ocr_data, list):
            # حالت لیست رشته‌ها
            for item in ocr_data:
                if isinstance(item, str):
                    texts_to_process.append(item.upper())
                elif isinstance(item, dict) and 'text' in item:
                    texts_to_process.append(item.get('text', '').upper())
        
        # جستجوی الگوها در متن OCR
        for text in texts_to_process:
            for prefix, patterns in prefix_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        prefix_counts[prefix] += len(matches)
                        logger.debug(f"Found {len(matches)} matches for {prefix} with pattern {pattern} in: {text}")
        
        # یافتن پیشوند غالب
        dominant_prefix = max(prefix_counts.items(), key=lambda x: x[1])[0] if any(prefix_counts.values()) else None
        
        if dominant_prefix:
            logger.info(f"✅ Detected dominant prefix: {dominant_prefix} (counts: {prefix_counts})")
        else:
            logger.warning(f"⚠️ Could not detect dominant prefix. Counts: {prefix_counts}")
        
        return dominant_prefix

    def reset_stats(self):
        """
        Reset all statistics counters.
        """
        self.all_tags.clear()
        self.matched_tags.clear()
        self.all_jbs.clear()
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0


class DataAnalysis:
    """
    Facade layer that centralizes initial PDF type detection and delegates the
    actual tag extraction pipeline to the underlying TagJBExtractor instance.
    """

    DEFAULT_MODEL_PATH = os.path.join(current_dir, 'modules', 'keras_model.h5')
    DEFAULT_LABELS_PATH = os.path.join(current_dir, 'modules', 'labels.txt')
    DEFAULT_PDF_TYPE = 'diagrams'

    def __init__(self, extractor, classifier_model_path: str = None, classifier_labels_path: str = None):
        self.extractor = extractor
        self.classifier = None
        self.document_types = {}
        self.classifier_model_path = classifier_model_path or self.DEFAULT_MODEL_PATH
        self.classifier_labels_path = classifier_labels_path or self.DEFAULT_LABELS_PATH
        self._load_classifier()

    def _load_classifier(self):
        if _PDFClassifierClass is None:
            logger.warning('PDFClassifier backend unavailable; document type detection disabled.')
            return

        if not os.path.exists(self.classifier_model_path) or not os.path.exists(self.classifier_labels_path):
            logger.warning(
                'PDFClassifier assets missing; document type detection disabled: %s, %s',
                self.classifier_model_path,
                self.classifier_labels_path,
            )
            return

        try:
            self.classifier = _PDFClassifierClass(
                model_path=self.classifier_model_path,
                labels_path=self.classifier_labels_path,
            )
            if hasattr(self.extractor, 'set_classifier'):
                self.extractor.set_classifier(self.classifier)
            logger.info('DataAnalysis initialized with PDFClassifier: %s', self.classifier_model_path)
        except Exception as exc:
            logger.error('Failed to initialize PDFClassifier: %s', exc)
            self.classifier = None

    def detect_pdf_type(self, pdf_path: str) -> str:
        if self.classifier is None:
            return self.DEFAULT_PDF_TYPE

        try:
            raw_label = self.classifier.classify_pdf(pdf_path)
            # Normalize classifier label to expected internal types
            label_l = (raw_label or '').strip().lower()
            if 'table' in label_l:
                pdf_type = 'table'
            elif 'diagram' in label_l or 'diagra' in label_l or 'drawing' in label_l:
                pdf_type = 'diagrams'
            else:
                # fallback to default
                pdf_type = self.DEFAULT_PDF_TYPE

            logger.info("PDFClassifier returned label '%s' -> normalized to '%s'", raw_label, pdf_type)
            self.document_types[pdf_path] = pdf_type
            return pdf_type
        except Exception as exc:
            logger.warning(
                'PDFClassifier failed for %s: %s — defaulting to %s',
                os.path.basename(pdf_path),
                exc,
                self.DEFAULT_PDF_TYPE,
            )
            self.document_types[pdf_path] = self.DEFAULT_PDF_TYPE
            return self.DEFAULT_PDF_TYPE

    def run_with_annotated_pdf(self, pdf_paths, *args, **kwargs):
        if self.classifier is not None and isinstance(pdf_paths, list):
            for pdf_path in pdf_paths:
                self.document_types[pdf_path] = self.detect_pdf_type(pdf_path)
                if hasattr(self.extractor, 'document_type_by_path'):
                    self.extractor.document_type_by_path[pdf_path] = self.document_types[pdf_path]

        return self.extractor.run_with_annotated_pdf(pdf_paths, *args, **kwargs)

    def run(self, *args, **kwargs):
        return self.extractor.run(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.extractor, attr)

