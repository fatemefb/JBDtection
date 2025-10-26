import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
import os
import gc
import fitz 
import tempfile
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
        # Initialize tracking variables
        self.match_attempts = 0
        self.successful_matches = 0
        self.match_scores = []
        self.required_columns = {
        'generated_excel': ['JB', 'MC', 'Tag/SPARE'],
        'io_list': ['Tag No', 'Tag']  # حداقل یکی از این ستون‌ها باید وجود داشته باشد
    }
    def add_reference_tag(self, tag: str) -> None:
        """Add a reference tag and create a vector for it."""
        if not tag or not isinstance(tag, str):
            return
        
        tag = str(tag).upper().strip()
        if not tag:
            return
            
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
        input_parts = len(input_tag.split('-'))
        input_digit_seqs = re.findall(r'\d+', input_tag)
        
        for ref_tag, ref_vector in self.tag_vectors.items():
            ref_len = len(ref_tag)
            if abs(ref_len - input_len) > max(5, ref_len * 0.5):
                continue
                
            ref_parts = len(ref_tag.split('-'))
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
                              self._calculate_digit_similarity(input_seq, ref_seq) > 0.5))):
                            digit_match = True
                            break
                    if digit_match:
                        break
                if not digit_match:
                    should_continue = True
            if should_continue:
                continue

            similarity = self.calculate_similarity(input_vector, ref_vector)
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
        
        # تنظیم مقادیر پیش‌فرض الگوها (به عنوان رشته)
        self.jb_examples = None
        self.mc_examples = None
        self.spare_examples = None
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

    def set_patterns(self, jb_examples=None, mc_examples=None, spare_examples=None, 
                    cable_examples=None, wire_color_rule=None, scr_number_rule=None):
        """
        تنظیم الگوهای سفارشی برای بهبود تشخیص
        
        Args:
            jb_examples: مثال JB (رشته یا لیست)
            mc_examples: مثال MC (رشته یا لیست)
            spare_examples: مثال SPARE (رشته یا لیست)
            cable_examples: مثال توصیف کابل (رشته یا لیست)
            wire_color_rule: قاعده تولید رنگ سیم
            scr_number_rule: قاعده تولید شماره SCR
        """
        
        # تبدیل لیست‌ها به رشته (اولین عنصر) اگر لازم باشد
        if jb_examples is not None:
            if isinstance(jb_examples, list) and jb_examples:
                self.jb_examples = jb_examples[0].upper()  # اولین عنصر را انتخاب کن
            elif isinstance(jb_examples, str) and jb_examples.strip():
                self.jb_examples = jb_examples.strip().upper()
            logger.info(f"JB examples Set: {self.jb_examples}")
        
        if mc_examples is not None:
            if isinstance(mc_examples, list) and mc_examples:
                self.mc_examples = mc_examples[0].upper()  # اولین عنصر را انتخاب کن
            elif isinstance(mc_examples, str) and mc_examples.strip():
                self.mc_examples = mc_examples.strip().upper()
            logger.info(f"MC examples Set: {self.mc_examples}")
        
        if spare_examples is not None:
            if isinstance(spare_examples, list) and spare_examples:
                self.spare_examples = spare_examples[0].upper()  # اولین عنصر را انتخاب کن
            elif isinstance(spare_examples, str) and spare_examples.strip():
                self.spare_examples = spare_examples.strip().upper()
            logger.info(f"SPARE examples Set: {self.spare_examples}")
        
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
        کامپایل الگوهای regex بر اساس مثال‌های تنظیم شده
        """
        try:
            # الگوی JB
            if self.jb_examples:
                self.jb_regex = re.compile(rf'\b{re.escape(self.jb_examples)}-?\d+\b', re.IGNORECASE)
                logger.debug(f"JB regex compiled: {self.jb_regex.pattern}")
            
            # الگوی MC
            if self.mc_examples:
                self.mc_regex = re.compile(rf'\b{re.escape(self.mc_examples)}-?\d+\b', re.IGNORECASE)
                logger.debug(f"MC regex compiled: {self.mc_regex.pattern}")
            
            # 🔧 FIX: الگوی SPARE ساده‌تر - فقط کلمه "spare" یا "sp"
            if self.spare_examples:
                self.spare_regex = re.compile(r'\b(spare)\b', re.IGNORECASE)
                logger.debug(f"SPARE regex compiled (simple pattern): {self.spare_regex.pattern}")
            
            logger.info("✅ All regex patterns compiled successfully")
                
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

    def extract_from_image(self, image: np.ndarray) -> 'Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict]]':
        """
        Extract tags with STRICT similar matching to prevent false positives.
        
        ✅ FIX: Similar match فقط برای تفاوت‌های جزئی (typo, OCR noise)
        ❌ REJECT: تفاوت‌های ساختاری (مثل LDIT vs LIT)
        """
        # مقداردهی اولیه (بدون تغییر)
        if not hasattr(self, 'jb_examples') or not self.jb_examples:
            self.jb_examples = "JB"
        if not hasattr(self, 'mc_examples') or not self.mc_examples:
            self.mc_examples = "MC"
        if not hasattr(self, 'spare_examples') or not self.spare_examples:
            self.spare_examples = "SPARE"
        
        logger.info(f"Using patterns - JB: '{self.jb_examples}', MC: '{self.mc_examples}', SPARE: '{self.spare_examples}'")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whiteList=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'
        
        logger.info("Starting OCR extraction...")
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        dominant_prefix = self._detect_dominant_prefix_in_page(ocr_data, ['UZSO', 'UZSC'])

        if dominant_prefix:
            logger.info(f"🎯 Page context: This page primarily contains {dominant_prefix} tags")
            # ذخیره برای استفاده در OCR correction
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
        tag_to_number = {}
        tag_match_info = {}
        
        all_ocr_tags = set()
        exact_matched_tags = set()
        similar_matched_tags = set()
        unmatched_ocr_tags = set()
        missing_io_tags = set()

        # مقداردهی متغیرهای مورد نیاز برای جلوگیری از خطاها
        io_list_tags = set()  # تگ‌های IO List
        all_ocr_tag_candidates = set()  # تمام تگ‌های بالقوه OCR
        matched_ocr_texts = set()  # تگ‌های OCR که با IO List تطبیق یافته‌اند
        matched_io_tags = set() 
        
        processed_tag_texts = set()
        processed_spare_indices = set()
        
        GENERAL_TAG_PATTERN = re.compile(r'^[A-Z]{2,5}-[\w\d]+(?:-\w+)?$', re.IGNORECASE)
        spare_pattern = re.compile(r'\b(spare)\b', re.IGNORECASE)
          
            
                # بررسی کنیم آیا تگ‌های IO List قبلا تنظیم شده‌اند
        if hasattr(self, 'io_list_tags'):
            io_list_tags = self.io_list_tags
        elif hasattr(self, 'excel_df') and hasattr(self, 'excel_tag_column'):
            # استخراج از دیتافریم اکسل
            if self.excel_df is not None and not self.excel_df.empty:
                tag_col = self.excel_tag_column
                io_list_tags = set(str(tag).strip().upper() for tag in self.excel_df[tag_col] if pd.notna(tag))
        
        cable_patterns = [
            re.compile(r'(\d+)\s*(P|PR|PAIR)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(T|TR|TRIPLE)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(C|CR|CORE)', re.IGNORECASE),
        ]
        
        mc_positions = []
        mc_indices = []
        sequence_number = 1
        spare_found_count = 0

        # ============================================================
        # Phase 0: Extract ALL OCR tags
        # ============================================================
        logger.info("Phase 0: Extracting ALL OCR tags...")
        
        for i, word in enumerate(ocr_data['text']):
            word_clean = word.strip().upper()
            if not word_clean or len(word_clean) < 4:
                continue
            
            if (self.jb_examples in word_clean or 
                self.mc_examples in word_clean or 
                spare_pattern.search(word_clean)):
                continue
            
            if GENERAL_TAG_PATTERN.match(word_clean):
                all_ocr_tags.add(word_clean)
                logger.debug(f"Found OCR tag: {word_clean}")
        
        logger.info(f"Phase 0 complete: {len(all_ocr_tags)} OCR tags")
        
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
                        
                        if best_match not in tag_to_number:
                            tag_to_number[best_match] = sequence_number
                            sequence_number += 1
                        
                        tag_match_info[best_match] = {
                            'match_type': 'exact',
                            'score': best_score,
                            'ocr_text': ocr_tag
                        }
                        
                        processed_tag_texts.add(ocr_tag)
                        logger.info(f"✅ EXACT: {ocr_tag} → {best_match}")

        logger.info(f"Phase 1 complete: {len(exact_matched_tags)} exact")
        
        # ============================================================
        # ✅ Phase 2: STRICT Similar matches (با فیلترهای قوی)
        # ============================================================
        logger.info("Phase 2: Searching for SIMILAR matches (STRICT mode)...")
        
        similar_rejected_count = 0
        
        for ocr_tag in all_ocr_tags:
            if ocr_tag in processed_tag_texts:
                continue
            
            similar_tags = self.vector_matcher.find_similar_tags(ocr_tag)
            
            if similar_tags:
                best_match, best_score = similar_tags[0]
                
                # ============================================================
                # 🔒 STRICT VALIDATION RULES
                # ============================================================
                
                # Rule 1: Score باید بین 0.96 تا 0.999 باشد
                if not (0.96 <= best_score < 1.0):
                    continue
                
                # Rule 2: طول رشته‌ها باید یکسان باشد (± 1 کاراکتر)
                len_diff = abs(len(ocr_tag) - len(best_match))
                if len_diff > 1:
                    logger.debug(f"❌ REJECTED (length): {ocr_tag} → {best_match} (diff: {len_diff} chars)")
                    similar_rejected_count += 1
                    continue
                
                # Rule 3: پیشوند (prefix) باید یکسان باشد
                ocr_prefix = self._extract_tag_prefix(ocr_tag)
                io_prefix = self._extract_tag_prefix(best_match)
                
                if ocr_prefix != io_prefix:
                    logger.debug(f"❌ REJECTED (prefix): {ocr_tag} [{ocr_prefix}] → {best_match} [{io_prefix}]")
                    similar_rejected_count += 1
                    continue
                
                # Rule 4: تعداد بخش‌های جدا شده با '-' باید یکسان باشد
                ocr_parts = ocr_tag.split('-')
                io_parts = best_match.split('-')
                
                if len(ocr_parts) != len(io_parts):
                    logger.debug(f"❌ REJECTED (structure): {ocr_tag} ({len(ocr_parts)} parts) → {best_match} ({len(io_parts)} parts)")
                    similar_rejected_count += 1
                    continue
                
                # Rule 5: بخش‌های عددی باید دقیقاً یکسان باشند (هیچ تلرانس عددی مجاز نیست!)
                if not self._are_numbers_identical(ocr_tag, best_match):
                    logger.debug(f"❌ REJECTED (numbers differ): {ocr_tag} → {best_match}")
                    similar_rejected_count += 1
                    continue
                
                # Rule 6: اگر دقیقاً یک کاراکتر تفاوت دارند، باید OCR error باشد (نه حرف متفاوت)
                if len_diff == 0 and self._count_different_chars(ocr_tag, best_match) == 1:
                    diff_char_ocr, diff_char_io = self._get_different_chars(ocr_tag, best_match)
                    
                    # لیست اشتباهات معمول OCR
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
                        logger.debug(f"❌ REJECTED (not OCR error): {ocr_tag} → {best_match} ('{diff_char_ocr}' vs '{diff_char_io}')")
                        similar_rejected_count += 1
                        continue
                
                # ============================================================
                # ✅ PASSED ALL RULES - Accept as Similar Match
                # ============================================================
                if best_match not in exact_matched_tags and best_match not in similar_matched_tags:
                    similar_matched_tags.add(best_match)
                    tags.add(best_match)
                    
                    if best_match not in tag_to_number:
                        tag_to_number[best_match] = sequence_number
                        sequence_number += 1
                    
                    tag_match_info[best_match] = {
                        'match_type': 'similar',
                        'score': best_score,
                        'ocr_text': ocr_tag,
                        'reason': self._get_similarity_reason(ocr_tag, best_match)  # توضیح دلیل شباهت
                    }
                    
                    processed_tag_texts.add(ocr_tag)
                    logger.info(f"⚠️ SIMILAR (VALIDATED): {ocr_tag} → {best_match} ({best_score:.3f}) - {tag_match_info[best_match]['reason']}")

        logger.info(f"Phase 2 complete: {len(similar_matched_tags)} similar matches (STRICT), {similar_rejected_count} rejected")
        
        # ============================================================
        # Phase 2.5: UNMATCHED OCR tags
        # ============================================================
        logger.info("Phase 2.5: Identifying UNMATCHED OCR tags...")
        
        for ocr_tag in all_ocr_tags:
            if ocr_tag not in processed_tag_texts:
                unmatched_ocr_tags.add(ocr_tag)
                
                unmatched_id = f"UNMATCHED_OCR_{ocr_tag}"
                tag_match_info[unmatched_id] = {
                    'match_type': 'unmatched_ocr',
                    'score': 0.0,
                    'ocr_text': ocr_tag
                }
                
                logger.warning(f"❌ UNMATCHED OCR: {ocr_tag} (not in IO List)")
        
        logger.info(f"Phase 2.5 complete: {len(unmatched_ocr_tags)} unmatched OCR")
        
        # ============================================================
        # Phase 2.75: MISSING from OCR
        # ============================================================
        logger.info("Phase 2.75: Identifying MISSING IO tags...")
        
        matched_io_tags = exact_matched_tags | similar_matched_tags
        missing_io_tags = io_list_tags - matched_io_tags
        
        for io_tag in missing_io_tags:
            missing_id = f"MISSING_IO_{io_tag}"
            tag_match_info[missing_id] = {
                'match_type': 'missing_from_ocr',
                'score': 0.0,
                'ocr_text': '',
                'io_tag': io_tag
            }
            
            logger.warning(f"⚠️ MISSING from OCR: {io_tag}")
        
        logger.info(f"Phase 2.75 complete: {len(missing_io_tags)} missing IO tags")
    
        
        # ============================================================
        # 🔧 FIX 4: پردازش SPARE، JB، MC (مستقل از تگ‌ها)
        # ============================================================
        logger.info("Phase 3: Processing SPARE, MC, JB identifiers...")
        
        for i, word in enumerate(ocr_data['text']):
            word_clean = word.strip().upper()
            if not word_clean:
                continue
            
            # 🔧 FIX 5: شناسایی SPARE (بدون تداخل با تگ‌ها)
            # استفاده از الگوی ساده‌تر
            if spare_pattern.search(word_clean):
                # بررسی که این index قبلاً پردازش نشده
                if i not in processed_spare_indices:
                    spare_identifiers.append(word_clean)
                    processed_spare_indices.add(i)
                    spare_found_count += 1
                    
                    # ایجاد شناسه یکتا برای SPARE
                    spare_id = f"{self.spare_examples}_{spare_found_count}"
                    tag_to_number[spare_id] = sequence_number
                    sequence_number += 1
                    
                    # 🔧 FIX 6: اضافه کردن SPARE به tag_match_info
                    tag_match_info[spare_id] = {
                        'match_type': 'spare',
                        'score': 1.0,
                        'ocr_text': word_clean
                    }
                    
                    logger.info(f"✅ {self.spare_examples} FOUND: {word_clean} (#{tag_to_number[spare_id]})")
                continue
            
            # شناسایی MC
            if len(word_clean) >= len(self.mc_examples) + 1 and self.mc_examples in word_clean and 'AS' not in word_clean:
                x, y = ocr_data['left'][i], ocr_data['top'][i]
                mc_positions.append((x, y))
                mc_indices.append(i)
                mc_identifiers.add(word_clean)
                logger.info(f"{self.mc_examples} identifier found: {word_clean}")
                continue
            
            # شناسایی JB
            if word_clean.startswith(self.jb_examples):
                jb_identifiers.add(word_clean)
                logger.info(f"{self.jb_examples} identifier found: {word_clean}")
                continue
        
        # ============================================================
        # مرحله 4: استخراج cable descriptions
        # ============================================================
        logger.info("Phase 4: Extracting cable descriptions...")
        
        for mc_i in mc_indices:
            mc_x, mc_y = ocr_data['left'][mc_i], ocr_data['top'][mc_i]
            
            search_radius_x = 300
            search_radius_y = 100
            
            nearby_words = []
            for j, word_j in enumerate(ocr_data['text']):
                if not word_j.strip() or len(word_j.strip()) < 1:
                    continue
                
                word_x, word_y = ocr_data['left'][j], ocr_data['top'][j]
                distance_x = abs(word_x - mc_x)
                distance_y = abs(word_y - mc_y)
                
                if distance_x <= search_radius_x and distance_y <= search_radius_y:
                    nearby_words.append(word_j.strip())
            
            combined_text = ' '.join(nearby_words).upper()
            
            if combined_text:
                raw_cable_descriptions.append(combined_text)
                logger.info(f"Added raw cable description: '{combined_text}'")
            
            # جستجو با patterns
            for pattern in cable_patterns:
                matches = pattern.findall(combined_text)
                for match in matches:
                    if isinstance(match, tuple):
                        number = match[0]
                        cable_type = match[1] if len(match) > 1 else ''
                    else:
                        number = match
                        cable_type = ''
                    
                    cable_type_full = ''
                    if cable_type:
                        cable_type_upper = cable_type.upper()
                        if cable_type_upper in ['P', 'PR', 'PAIR']:
                            cable_type_full = 'pair'
                        elif cable_type_upper in ['T', 'TR', 'TRIPLE']:
                            cable_type_full = 'triple'
                        elif cable_type_upper in ['C', 'CR', 'CORE']:
                            cable_type_full = 'core'
                    else:
                        if 'PAIR' in combined_text:
                            cable_type_full = 'pair'
                        elif 'TRIPLE' in combined_text:
                            cable_type_full = 'triple'
                        elif 'CORE' in combined_text:
                            cable_type_full = 'core'
                        else:
                            cable_type_full = 'pair'
                    
                    cable_desc = f"{number} {cable_type_full}"
                    if cable_desc not in cable_descriptions:
                        cable_descriptions.append(cable_desc)
                        logger.info(f"Found cable description: {cable_desc}")
        
        # ============================================================
        # 🆕 مرحله 5: شناسایی تگ‌های OCR شده که در IO لیست پیدا نشدند (Unmatched)
        # ============================================================
        logger.info("Phase 5: Identifying unmatched OCR tags...")
        
        # 🆕 مجموعه ای از تمام متن های OCR که منجر به تطبیق معتبر شدند
        # توجه: tag_match_info شامل تگ‌های IO Matched است. ما به متن OCR آن‌ها نیاز داریم.
        matched_ocr_texts = {info['ocr_text'] for tag, info in tag_match_info.items() if info['match_type'] in ['exact', 'similar']}

        # کاندیداهایی که Match نشده‌اند: آنهایی که ساختار تگ را دارند اما Match پیدا نکردند.
        unmatched_ocr_tags = all_ocr_tag_candidates - matched_ocr_texts
        
        # 🆕 اضافه کردن هر تگ OCR که Match نشده است به tag_match_info
        for ocr_tag in unmatched_ocr_tags:
            # مطمئن شویم که OCR text خودش یک JB/MC/SPARE نیست
            if (self.jb_examples in ocr_tag or 
                self.mc_examples in ocr_tag or 
                spare_pattern.search(ocr_tag)):
                continue

            # فقط تگ‌هایی که قبلا در info ثبت نشده‌اند (یعنی نه match و نه spare)
            is_already_matched = False
            for info in tag_match_info.values():
                if info.get('ocr_text') == ocr_tag:
                    is_already_matched = True
                    break
            
            if not is_already_matched:
                tag_match_info[f"UNMATCHED_{ocr_tag}"] = { # استفاده از یک شناسه یکتا برای ردیابی
                    'match_type': 'unmatched',
                    'score': 0.0,
                    'ocr_text': ocr_tag
                }
                logger.debug(f"⚠️ UNMATCHED tag found: {ocr_tag}")
        
        # ============================================================
        # 🔧 FIX 7: اطمینان از وجود tag_match_info برای همه تگ‌ها
        # ============================================================
        for tag in tags:
            if tag not in tag_match_info:
                logger.warning(f"⚠️ Tag '{tag}' missing from tag_match_info, adding default 'unknown'")
                tag_match_info[tag] = {
                    'match_type': 'unknown',
                    'score': 0.0,
                    'ocr_text': tag
                }
        
        # ============================================================
        # مرحله 5: ذخیره اطلاعات در instance
        # ============================================================
        if not hasattr(self, 'page_match_info'):
            self.page_match_info = {}
        
        self.page_match_info['tags'] = tags
        self.page_match_info['tag_match_info'] = tag_match_info
        self.page_match_info['exact_matched_tags'] = exact_matched_tags
        self.page_match_info['similar_matched_tags'] = similar_matched_tags
        
        # ============================================================
        # لاگ نهایی با جزئیات کامل
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
        logger.info(f'     - Cable descriptions: {len(cable_descriptions)}')
        logger.info(f'  📋 Metadata:')
        logger.info(f'     - tag_match_info entries: {len(tag_match_info)}')
        logger.info(f'     - tag_to_number entries: {len(tag_to_number)}')
        logger.info(f'='*60)
        
        # 🔧 Debug: نمایش نمونه‌هایی از هر نوع
        if exact_matched_tags:
            logger.debug(f"Sample exact matches: {list(exact_matched_tags)[:3]}")
        if similar_matched_tags:
            logger.debug(f"Sample similar matches: {list(similar_matched_tags)[:3]}")
        if spare_identifiers:
            logger.debug(f"SPARE identifiers: {spare_identifiers}")
        
        # به‌روزرسانی مجموعه‌های کلی
        if hasattr(self, 'all_tags'):
            self.all_tags.update(tags)
        if hasattr(self, 'all_jbs'):
            self.all_jbs.update(jb_identifiers)
        if hasattr(self, 'all_mcs'):
            self.all_mcs.update(mc_identifiers)
        if hasattr(self, 'all_spares'):
            self.all_spares = spare_identifiers
        if hasattr(self, 'matched_tags_set'):
            self.matched_tags_set.update(tags)
        
        # ============================================================
        # بازگشت 8 مقدار
        # ============================================================
        logger.info(f"✅ Returning 8 values from extract_from_image")
        return (tags, jb_identifiers, mc_identifiers, cable_descriptions, 
                spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info)

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
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image to improve OCR accuracy for tags."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Enhance resolution for better character detection
        scale_factor = 2
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Reduce noise
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 31, 2)
        
        # Morphological operations to close gaps in characters
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Dilate slightly to connect broken character components
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
                r'--oem 3 --psm 6', 
                r'--oem 3 --psm 11',  
                r'--oem 3 --psm 12' 
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


    def process_pdf_page(self, page_info: 'Tuple[fitz.Page, str, int]') -> 'Tuple[int, Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str]], Dict[str, Dict]]':
        """
        Process a single PDF page - اصلاح شده برای بازگرداندن 7 مقدار
        
        Args:
            page_info: Tuple containing (page object, temp_dir path, page number)
            
        Returns:
            Tuple of (page_number, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions , tag_match_info)
        """
        page, temp_dir, page_num = page_info
        
        try:
            # Create image path
            image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            pix.save(image_path)
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image for page {page_num + 1}")
                return page_num + 1, set(), set(), set(), [], [], {}, [] , {}
                
            result = self.extract_from_image(image)
            
            # Handle different return formats
            if len(result) >= 8:
                tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions , process_pdf_page ,tag_match_info= result[:8]
            elif len(result) >= 7:
                tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number ,process_pdf_page , tag_match_info = result[:7]
                raw_cable_descriptions = []
                tag_match_info = {}
            else:
                logger.error(f"Unexpected result format: {len(result)} values")
                tags, jb_identifiers, mc_identifiers = set(), set(), set()
                cable_descriptions, spare_identifiers = [], []
                tag_to_number, raw_cable_descriptions , tag_match_info  = {}, [] ,{}
            
            # Clean up temporary image file
            try:
                os.remove(image_path)
            except:
                pass
                
            return page_num + 1, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions , tag_match_info
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            logger.error(traceback.format_exc())
            return page_num + 1, set(), set(), set(), [], [], {}, [] , {}

    def process_pdf(self, pdf_path: str) -> 'Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict]]]':
        """
        Process all pages in a PDF file.
        
        Returns:
            Dictionary mapping page numbers to Tuples of (tags, jb_identifiers, mc_identifiers, 
                                                        cable_descriptions, spare_identifiers, 
                                                        tag_to_number, raw_cable_descriptions, tag_match_info)
        """
        results = {}
        
        # Reinitialize Tesseract
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
        
        logger.info(f"Opening PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        pdf_filename = os.path.basename(pdf_path)
        print(f"\nProcessing PDF: {pdf_filename}")
        print("-" * 50)
        
        # Create temporary directory for image processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process pages sequentially
            for page_num in range(len(pdf_document)):
                try:
                    logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
                    
                    # Get page
                    page = pdf_document[page_num]
                    
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix.save(image_path)
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"Failed to load image for page {page_num + 1}")
                        continue
                    
                    # 🆕 Extract tags with match info (8 مقدار)
                    extract_result = self.extract_from_image(image)
                    
                    if len(extract_result) >= 8:
                        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info = extract_result[:8]
                    elif len(extract_result) >= 7:
                        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions , tag_match_info = extract_result[:7]
                        tag_match_info = {}
                    else:
                        logger.error(f"Unexpected extract result length: {len(extract_result)}")
                        continue
                    
                    # 🆕 Store results with match info (8 مقدار)
                    results[page_num + 1] = (tags, jb_identifiers, mc_identifiers, 
                                            cable_descriptions, spare_identifiers, 
                                            tag_to_number, raw_cable_descriptions, 
                                            tag_match_info)
                    
                    # Print results
                    print(f"Page {page_num + 1}:")
                    print(f"  Tags found ({len(tags)}): {', '.join(sorted(tags))}")
                    if tag_match_info:
                        exact_count = sum(1 for info in tag_match_info.values() if info.get('match_type') == 'exact')
                        similar_count = sum(1 for info in tag_match_info.values() if info.get('match_type') == 'similar')
                        print(f"  Match types: {exact_count} exact, {similar_count} similar")
                    print(f"  JB identifiers found ({len(jb_identifiers)}): {', '.join(sorted(jb_identifiers))}")
                    print(f"  MC identifiers found ({len(mc_identifiers)}): {', '.join(sorted(mc_identifiers))}")
                    
                    # Clean up
                    try:
                        os.remove(image_path)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
        
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


    def process_excel_with_io_list(self, intermediate_excel_path: str, excel_path: str, output_path: str) -> 'Tuple[pd.DataFrame, List[str], List[str]]':
        """
        ترکیب داده‌های فایل intermediate با فایل IO List و ایجاد فایل اکسل نهایی.
        این تابع تمام ستون‌های IO List را حفظ می‌کند و ستون‌های جدید از فایل intermediate را به آن اضافه می‌کند.
        
        Args:
            intermediate_excel_path: مسیر فایل اکسل intermediate
            excel_path: مسیر فایل اکسل IO List
            output_path: مسیر فایل اکسل خروجی
            
        Returns:
            Tuple of (final_df, unmatched_io_tags, unmatched_tags)
        """
        try:
            # خواندن فایل‌های اکسل
            intermediate_df = pd.read_excel(intermediate_excel_path)
            io_list_df = pd.read_excel(excel_path)
            
            logger.info(f"Loaded intermediate Excel with {len(intermediate_df)} rows and {len(intermediate_df.columns)} columns")
            logger.info(f"Loaded IO List Excel with {len(io_list_df)} rows and {len(io_list_df.columns)} columns")
            
            # نام ستون تگ در هر دو فایل
            intermediate_tag_col = 'Tag/SPARE'
            io_list_tag_col = 'Tag No'  # نام ستون تگ در IO List
            
            # استخراج لیست تگ‌ها از هر دو فایل
            intermediate_tags = list(str(tag).strip().upper() for tag in intermediate_df[intermediate_tag_col] if pd.notna(tag))
            io_list_tags = list(str(tag).strip().upper() for tag in io_list_df[io_list_tag_col] if pd.notna(tag))
            
            # تطبیق تگ‌ها با استفاده از مکانیزم کاندید تگ
            logger.info(f"Finding tag candidates for {len(intermediate_tags)} PDF tags with {len(io_list_tags)} IO List tags")
            
            # ایجاد دیکشنری برای نگاشت تگ‌های PDF به تگ‌های IO List
            pdf_to_io_tag_map = {}
            
            # برای هر تگ PDF، کاندیداهای مشابه در IO List را پیدا کن
            for pdf_tag in intermediate_tags:
                # پیدا کردن کاندیداها
                candidates = []
                for io_tag in io_list_tags:
                    similarity = self.calculate_tag_similarity(pdf_tag, io_tag)
                    if similarity >= 0.8:  # حد آستانه شباهت
                        candidates.append((io_tag, similarity))
                
                # مرتب‌سازی کاندیداها بر اساس امتیاز شباهت
                candidates.sort(key=lambda x: x[1], reverse=True)
                
                # انتخاب بهترین کاندیدا
                if candidates:
                    best_match, best_similarity = candidates[0]
                    pdf_to_io_tag_map[pdf_tag] = best_match
                    logger.info(f"Matched PDF tag '{pdf_tag}' to IO tag '{best_match}' with similarity {best_similarity:.2f}")
            
            logger.info(f"Found {len(pdf_to_io_tag_map)} tag matches")
            
            # یافتن تگ‌های تطبیق نیافته
            matched_pdf_tags = set(pdf_to_io_tag_map.keys())
            matched_io_tags = set(pdf_to_io_tag_map.values())
            
            unmatched_tags = list(set(intermediate_tags) - matched_pdf_tags)
            unmatched_io_tags = list(set(io_list_tags) - matched_io_tags)
            
            logger.info(f"Unmatched PDF tags: {len(unmatched_tags)}")
            logger.info(f"Unmatched IO List tags: {len(unmatched_io_tags)}")
            
            # ایجاد کپی از IO List برای حفظ تمام ستون‌های آن
            final_df = io_list_df.copy()
            
            # ستون‌های intermediate که می‌خواهیم اضافه کنیم
            intermediate_columns_to_add = [
                'PDF_Name', 'Page', 'JB', 'MC', 'Tag_Number', 
                'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number', 'SCR_Terminal_Number', 'Cable_code',
                'Cable_Description', 'Type', 'Tag_Number_Status'
            ]
            
            # فقط ستون‌هایی که در intermediate وجود دارند را اضافه کنیم
            intermediate_columns_to_add = [col for col in intermediate_columns_to_add if col in intermediate_df.columns]
            
            # اضافه کردن ستون‌های جدید به final_df
            for col in intermediate_columns_to_add:
                if col not in final_df.columns:
                    final_df[col] = None
            
            # تطبیق داده‌ها با استفاده از دیکشنری pdf_to_io_tag_map
            for idx, row in final_df.iterrows():
                io_tag = str(row[io_list_tag_col]).strip().upper() if pd.notna(row[io_list_tag_col]) else ""
                
                # جستجو در تگ‌های PDF که تطبیق داده شده‌اند
                pdf_tag = None
                for pdf_t, io_t in pdf_to_io_tag_map.items():
                    if str(io_t).strip().upper() == io_tag:
                        pdf_tag = pdf_t
                        break
                
                if pdf_tag:
                    # پیدا کردن اطلاعات تگ در intermediate_df
                    matching_rows = intermediate_df[intermediate_df[intermediate_tag_col].apply(
                        lambda x: str(x).strip().upper() == pdf_tag if pd.notna(x) else False
                    )]
                    
                    if not matching_rows.empty:
                        # اگر تگ در intermediate پیدا شد، اطلاعات را به final_df اضافه کن
                        for col in intermediate_columns_to_add:
                            final_df.at[idx, col] = matching_rows.iloc[0][col]
            
            # اضافه کردن تگ‌های intermediate که در IO List نیستند به final_df
            if unmatched_tags:
                # فیلتر کردن ردیف‌های intermediate_df که تگ‌های آن‌ها در IO List نیستند
                unmatched_rows = intermediate_df[intermediate_df[intermediate_tag_col].apply(
                    lambda x: str(x).strip().upper() in unmatched_tags if pd.notna(x) else False
                )]
                
                # ایجاد دیتافریم جدید با ستون‌های final_df
                new_rows = pd.DataFrame(columns=final_df.columns)
                
                # اضافه کردن ردیف‌های جدید
                for _, row in unmatched_rows.iterrows():
                    new_row = pd.Series(index=final_df.columns)
                    
                    # کپی مقادیر از ستون‌های intermediate
                    for col in intermediate_columns_to_add:
                        new_row[col] = row[col]
                    
                    # تنظیم مقدار ستون تگ در IO List
                    new_row[io_list_tag_col] = row[intermediate_tag_col]
                    
                    # اضافه کردن ردیف جدید به new_rows
                    new_rows = pd.concat([new_rows, pd.DataFrame([new_row])], ignore_index=True)
                
                # اضافه کردن ردیف‌های جدید به final_df
                final_df = pd.concat([final_df, new_rows], ignore_index=True)
            
            # ذخیره دیتافریم نهایی به عنوان فایل اکسل
            final_df.to_excel(output_path, index=False)
            
            logger.info(f"Combined Excel file saved to: {output_path}")
            logger.info(f"Final Excel has {len(final_df)} rows and {len(final_df.columns)} columns")
            
            return final_df, unmatched_io_tags, unmatched_tags
            
        except Exception as e:
            logger.error(f"Error processing Excel files: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame(), [], []

        
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

    def run(self, pdf_paths: 'List[str]', excel_path: str, output_excel_path: str, intermediate_excel_path: str) -> 'Tuple[List[str], List[str], List[str]]':
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
            output_excel_path)
        
        # Save updated Excel
        final_df.to_excel(output_excel_path, index=False)
        logger.info(f"Updated Excel saved to: {output_excel_path}")
        
        return unmatched_io_tags, unmatched_tags

    def draw_bounding_boxes(self, image, tags=None, jb_identifiers=None, mc_identifiers=None,
                        cable_descriptions=None, spare_identifiers=None, tag_to_number=None,
                        tag_match_info=None):
        """
        رسم باندینگ باکس‌های یک تگ در تصویر
        """
        
        # ============================================================
        # مرحله 0: بررسی و مقداردهی اولیه
        # ============================================================
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
        
        # 🔧 FIX 1: مقداردهی صحیح tag_match_info
        if tag_match_info is None or not tag_match_info:
            tag_match_info = {}
            logger.warning("⚠️ tag_match_info is None/empty, creating default entries for all tags")
            
            # ایجاد entry پیش‌فرض برای همه تگ‌ها
            for tag in tags:
                tag_match_info[tag] = {
                    'match_type': 'unknown',
                    'score': 0.0,
                    'ocr_text': tag
                }
        
        # اطمینان از تنظیم الگوها
        if not hasattr(self, 'jb_examples') or self.jb_examples is None:
            self.jb_examples = "JB"
        if not hasattr(self, 'mc_examples') or self.mc_examples is None:
            self.mc_examples = "MC"
        if not hasattr(self, 'spare_examples') or self.spare_examples is None:
            self.spare_examples = "SPARE"
        
        # Debug logging
        logger.info(f"Drawing bounding boxes:")
        logger.info(f"  Tags: {len(tags)}")
        logger.info(f"  JBs: {len(jb_identifiers)}")
        logger.info(f"  MCs: {len(mc_identifiers)}")
        logger.info(f"  SPAREs: {len(spare_identifiers)}")
        logger.info(f"  tag_match_info entries: {len(tag_match_info)}")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # OCR config
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whiteList=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        # ============================================================
        # 🔧 FIX: جمع‌آوری تمام موارد یافت شده با موقعیت‌های آنها
        # ============================================================
        all_found_items = []
        processed_regions = set()
        
        # ============================================================
        # Phase 1: جمع‌آوری Exact Matches
        # ============================================================
        logger.info("Phase 1: Collecting EXACT matches...")
        exact_found_count = 0
        
        for tag in tags:
            match_type = 'unknown'
            match_score = 0.0
            ocr_text_used = tag
            
            if tag in tag_match_info:
                info = tag_match_info[tag]
                match_type = info.get('match_type', 'unknown')
                match_score = info.get('score', 0.0)
                ocr_text_used = info.get('ocr_text', tag)
            else:
                logger.warning(f"⚠️ Tag '{tag}' not found in tag_match_info, treating as 'unknown'")
            
            # فقط exact matches
            if match_type != 'exact':
                continue
            
            tag_upper = tag.upper()
            
            # یافتن تمام نمونه‌های این تگ در متن OCR شده
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                if text_clean == tag_upper or text_clean == ocr_text_used.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'tag',
                            'text': tag,
                            'position': region_key,
                            'match_type': 'exact',
                            'score': match_score,
                            'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                        })
                        processed_regions.add(region_key)
                        exact_found_count += 1
                        logger.debug(f"✅ Found exact match: {tag}")
        
        logger.info(f"Phase 1 complete: Found {exact_found_count} exact matches")
        
        # ============================================================
        # Phase 2: جمع‌آوری Similar Matches
        # ============================================================
        logger.info("Phase 2: Collecting SIMILAR matches...")
        similar_found_count = 0
        
        for tag in tags:
            match_type = 'unknown'
            match_score = 0.0
            ocr_text_used = tag
            
            if tag in tag_match_info:
                info = tag_match_info[tag]
                match_type = info.get('match_type', 'unknown')
                match_score = info.get('score', 0.0)
                ocr_text_used = info.get('ocr_text', tag)
            
            # فقط similar matches
            if match_type != 'similar':
                continue
            
            # یافتن تمام نمونه‌های این تگ در متن OCR شده
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                region_key = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                
                if region_key in processed_regions:
                    continue
                
                if text_clean == ocr_text_used.upper():
                    all_found_items.append({
                        'type': 'tag',
                        'text': tag,
                        'position': region_key,
                        'match_type': 'similar',
                        'score': match_score,
                        'original_text': text_clean,
                        'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                    })
                    processed_regions.add(region_key)
                    similar_found_count += 1
                    logger.debug(f"⚠️ Found similar match: {text_clean} → {tag}")
        
        logger.info(f"Phase 2 complete: Found {similar_found_count} similar matches")
        
        # ============================================================
        # Phase 3: جمع‌آوری JB identifiers
        # ============================================================
        logger.info("Phase 3: Collecting JB identifiers...")
        jb_found_count = 0
        
        for jb in jb_identifiers:
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
                            'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                        })
                        processed_regions.add(region_key)
                        jb_found_count += 1
                        logger.debug(f"Found JB: {jb}")
        
        logger.info(f"Phase 3 complete: Found {jb_found_count} JBs")
        
        # ============================================================
        # Phase 4: جمع‌آوری MC identifiers
        # ============================================================
        logger.info("Phase 4: Collecting MC identifiers...")
        mc_found_count = 0
        
        for mc in mc_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == mc.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        all_found_items.append({
                            'type': 'mc',
                            'text': mc,
                            'position': region_key,
                            'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                        })
                        processed_regions.add(region_key)
                        mc_found_count += 1
                        logger.debug(f"Found MC: {mc}")
        
        logger.info(f"Phase 4 complete: Found {mc_found_count} MCs")
            
        # ============================================================
        # Phase 5: جمع‌آوری SPARE identifiers
        # ============================================================
        logger.info(f"Phase 5: Collecting {len(spare_identifiers)} SPARE identifiers...")
        spare_found_count = 0
        
        # ایجاد regex pattern برای SPARE
        spare_pattern = re.compile(rf'\b{re.escape(self.spare_examples)}\b', re.IGNORECASE)
        
        for spare in spare_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                # بررسی دقیق‌تر SPARE
                # فقط اگر کلمه SPARE به تنهایی باشد (نه بخشی از کلمه دیگر)
                if spare_pattern.search(text_clean):
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        spare_id = f"{self.spare_examples}_{spare_found_count + 1}"
                        
                        all_found_items.append({
                            'type': 'spare',
                            'text': spare,
                            'position': region_key,
                            'id': spare_id,
                            'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                        })
                        processed_regions.add(region_key)
                        spare_found_count += 1
                        logger.debug(f"✅ Found SPARE: {spare}")
        
        logger.info(f"Phase 5 complete: Found {spare_found_count} SPAREs")
        
        # ============================================================
        # Phase 6: جمع‌آوری Cable descriptions
        # ============================================================
        logger.info("Phase 6: Collecting cable descriptions...")
        cable_found_count = 0
        
        for cable_desc in cable_descriptions:
            cable_parts = cable_desc.split()
            if len(cable_parts) >= 1:
                number_part = cable_parts[0]
                
                for i, text in enumerate(ocr_data['text']):
                    text_clean = text.strip().upper()
                    if number_part in text_clean:
                        region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                    ocr_data['width'][i], ocr_data['height'][i])
                        if region_key not in processed_regions:
                            all_found_items.append({
                                'type': 'cable',
                                'text': cable_desc,
                                'position': region_key,
                                'y_position': ocr_data['top'][i]  # موقعیت عمودی برای مرتب‌سازی
                            })
                            processed_regions.add(region_key)
                            cable_found_count += 1
                            logger.debug(f"Found cable: {cable_desc}")
        
        logger.info(f"Phase 6 complete: Found {cable_found_count} cables")
        
        # ============================================================
        # 🔧 FIX: مرتب‌سازی تمام آیتم‌ها بر اساس موقعیت عمودی (y_position)
        # ============================================================
        logger.info("Sorting all items by vertical position (top to bottom)...")
        
        # جدا کردن تگ‌ها و SPARE ها برای شماره‌گذاری
        tags_and_spares = [item for item in all_found_items if item['type'] in ('tag', 'spare')]
        
        # مرتب‌سازی بر اساس موقعیت عمودی (از بالا به پایین)
        tags_and_spares.sort(key=lambda x: x['y_position'])
        
        logger.info(f"Sorted {len(tags_and_spares)} tags and spares by vertical position")
        
        # ============================================================
        # 🔧 FIX: شماره‌گذاری بر اساس ترتیب عمودی
        # ============================================================
        all_tag_numbers = dict(tag_to_number)  # کپی از tag_to_number
        sequence_number = max(all_tag_numbers.values()) + 1 if all_tag_numbers else 1
        
        # شماره‌گذاری تگ‌ها و SPARE ها بر اساس ترتیب عمودی
        for item in tags_and_spares:
            if item['type'] == 'tag':
                tag = item['text']
                if tag not in all_tag_numbers:
                    all_tag_numbers[tag] = sequence_number
                    sequence_number += 1
                    logger.info(f"Assigned number {all_tag_numbers[tag]} to tag {tag}")
            elif item['type'] == 'spare':
                spare_id = item['id']
                if spare_id not in all_tag_numbers:
                    all_tag_numbers[spare_id] = sequence_number
                    sequence_number += 1
                    logger.info(f"Assigned number {all_tag_numbers[spare_id]} to SPARE {spare_id}")
        
        # ============================================================
        # رسم bounding boxes با رنگ‌بندی صحیح
        # ============================================================
        logger.info(f"Drawing all found items...")
        
        # رسم تمام آیتم‌ها
        for item in all_found_items:
            x, y, w, h = item['position']
            item_type = item['type']
            text = item['text']
            
            if item_type == 'tag':
                cleaned_text = self.clean_text_for_display(text)
                match_type = item.get('match_type', 'unknown')
                score = item.get('score', 0.0)
                
                tag_number = all_tag_numbers[text]
                
                # رنگ‌بندی بر اساس match type
                if match_type == 'exact':
                    color = (255, 0, 0)      # سبز
                    label_prefix = "✓"
                elif match_type == 'similar':
                    color = (0, 165, 255)    # نارنجی
                    label_prefix = "≈"
                else:  # unknown یا هر چیز دیگر
                    color = (128, 128, 128)  # خاکستری
                    label_prefix = ""
                    logger.warning(f"⚠️ Unknown match type for tag '{text}': {match_type}")
                
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                if match_type == 'similar' and score > 0:
                    label = f" #{tag_number} {cleaned_text} ({score:.2f})"
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
                spare_number = all_tag_numbers.get(spare_id, 0)
                
                cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)  # بنفش
                cv2.putText(image, f"SPARE #{spare_number}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
                
            elif item_type == 'cable':
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 200), 2)  # زرد
                cv2.putText(image, f"Cable: {text}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # ============================================================
        # آمار و Legend
        # ============================================================
        exact_count = len([item for item in all_found_items if item.get('type') == 'tag' and item.get('match_type') == 'exact'])
        similar_count = len([item for item in all_found_items if item.get('type') == 'tag' and item.get('match_type') == 'similar'])
        unknown_count = len([item for item in all_found_items if item.get('type') == 'tag' and item.get('match_type') == 'unknown'])
        spare_count = len([item for item in all_found_items if item.get('type') == 'spare'])
        
        legend_y_pos = image.shape[0] - 100
        legend_x_pos = 10
        
        # Legend header
        cv2.putText(image, "Legend:", (legend_x_pos, legend_y_pos - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Match types
        cv2.putText(image, f"Exact: {exact_count}", (legend_x_pos, legend_y_pos - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, f"Similar: {similar_count}", (legend_x_pos + 150, legend_y_pos - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(image, f"Unknown: {unknown_count}", (legend_x_pos + 300, legend_y_pos - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
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
        stats_text = f"Total: {exact_count + similar_count + unknown_count} tags, {spare_count} spares"
        cv2.putText(image, stats_text, (legend_x_pos, legend_y_pos + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        logger.info(f"✅ Bounding boxes drawn:")
        logger.info(f"   Tags: {exact_count} exact, {similar_count} similar, {unknown_count} unknown")
        logger.info(f"   Components: {jb_found_count} JBs, {mc_found_count} MCs, {spare_count} SPAREs")
        
        return image, all_tag_numbers
    
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
        Create annotated PDF with memory-efficient processing for multi-page documents
        """
        all_tag_numbers = {}
        pdf_document = None
        new_pdf = None
        
        try:
            logger.info(f"Creating annotated PDF from: {pdf_path}")
            pdf_document = fitz.open(pdf_path)
            new_pdf = fitz.open()
            total_pages = len(pdf_document)
            
            # Use lower DPI for multi-page PDFs to conserve memory
            dpi_factor = 200/72 if total_pages > 10 else 300/72
            
            with tempfile.TemporaryDirectory() as temp_dir:
                for page_num in range(total_pages):
                    try:
                        logger.info(f"Annotating page {page_num + 1}/{total_pages}")
                        
                        page = pdf_document[page_num]
                        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_factor, dpi_factor))
                        
                        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                        pix.save(image_path)
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            logger.warning(f"Failed to load image for page {page_num + 1}")
                            new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                            pix = None
                            continue
                        
                        # 🆕 استخراج با 8 مقدار
                        result = self.extract_from_image(image)
                        
                        if len(result) >= 8:
                            tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info = result[:8]
                        elif len(result) >= 7:
                            tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions ,tag_match_info = result[:7]
                            tag_match_info = {}
                        else:
                            tags, jb_identifiers, mc_identifiers = set(), set(), set()
                            cable_descriptions, spare_identifiers = [], []
                            tag_to_number, raw_cable_descriptions, tag_match_info = {}, [], {}
                        
                        # Draw bounding boxes
                        try:
                            annotated_image, page_tag_numbers = self.draw_bounding_boxes(
                                image, tags, jb_identifiers, mc_identifiers,
                                cable_descriptions, spare_identifiers, tag_to_number,
                                tag_match_info  # 🆕 پارامتر اضافی
                            )
                            all_tag_numbers.update(page_tag_numbers)
                        except Exception as e:
                            logger.error(f"Error drawing bounding boxes on page {page_num + 1}: {e}")
                            annotated_image = image.copy()
                            page_tag_numbers = tag_to_number
                        
                        # Add info overlay
                        try:
                            stats = self.get_processing_stats() if hasattr(self, 'get_processing_stats') else {}
                            info_text = [
                                f"Page {page_num + 1}/{total_pages}",
                                f"Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}",
                                f"Memory: {page_num + 1} pages processed"
                            ]
                            
                            # Add semi-transparent overlay
                            overlay = annotated_image.copy()
                            overlay_h = len(info_text) * 25 + 15
                            x, y, w, h = 5, 5, 350, overlay_h
                            
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
                        
                        # Save annotated image and add to PDF
                        try:
                            annotated_path = os.path.join(temp_dir, f"annotated_{page_num + 1}.png")
                            cv2.imwrite(annotated_path, annotated_image)
                            
                            new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                            new_page.insert_image(new_page.rect, filename=annotated_path)
                            
                            os.remove(annotated_path)
                        except Exception as e:
                            logger.error(f"Error saving page {page_num + 1}: {e}")
                            # Add original page if annotation fails
                            try:
                                new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                                new_page.insert_image(new_page.rect, filename=image_path)
                            except:
                                pass
                        
                        # Clean up memory for this page
                        del image, annotated_image
                        pix = None
                        
                        # Clean up temp file
                        try:
                            os.remove(image_path)
                        except:
                            pass
                        
                        # Garbage collect every 3 pages for memory management
                        if (page_num + 1) % 3 == 0:
                            gc.collect()
                            logger.debug(f"Memory cleanup after page {page_num + 1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1} for annotation: {e}")
                        logger.error(traceback.format_exc())
                        continue
            
            # Save the annotated PDF
            try:
                os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
                new_pdf.save(output_pdf_path)
                logger.info(f"Annotated PDF saved: {output_pdf_path}")
                logger.info(f"Total pages processed: {total_pages}")
                logger.info(f"Total tags numbered: {len(all_tag_numbers)}")
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

    # اصلاح تابع _create_unmatched_tags_excel برای رفع خطای فایل Excel
    def _create_unmatched_tags_excel(self, unmatched_excel_tags: 'List[str]', unmatched_pdf_tags: 'List[str]', output_path: str):
        """
        ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
        
        Args:
            unmatched_excel_tags: لیست تگ‌های اکسل که در PDF پیدا نشده‌اند
            unmatched_pdf_tags: لیست تگ‌های PDF که در اکسل پیدا نشده‌اند
            output_path: مسیر فایل خروجی
        """
        try:
            # بررسی مسیر فایل خروجی
            if not output_path or not output_path.strip():
                output_path = "unmatched_tags.xlsx"  # مسیر پیش‌فرض
            
            # اطمینان از پسوند صحیح
            if not output_path.endswith('.xlsx'):
                output_path = output_path + '.xlsx'
            
            # ایجاد دیتافریم برای تگ‌های تطبیق نیافته
            excel_tags_df = pd.DataFrame({"Tag": unmatched_excel_tags, "Source": "Excel", "Status": "Not found in PDF"})
            pdf_tags_df = pd.DataFrame({"Tag": unmatched_pdf_tags, "Source": "PDF", "Status": "Not found in Excel"})
            
            # ترکیب دو دیتافریم
            unmatched_df = pd.concat([excel_tags_df, pdf_tags_df], ignore_index=True)
            
            # اگر دیتافریم خالی است، ستون‌های مناسب را اضافه کن
            if unmatched_df.empty:
                unmatched_df = pd.DataFrame(columns=["Tag", "Source", "Status"])
            
            # اطمینان از وجود دایرکتوری مسیر خروجی
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # ذخیره به فایل اکسل
            unmatched_df.to_excel(output_path, index=False)
            logger.info(f"Unmatched tags Excel file created with {len(unmatched_df)} rows")
            
        except Exception as e:
            logger.error(f"Error creating unmatched tags Excel file: {e}")
            logger.error(traceback.format_exc())
            
            try:
                # تلاش مجدد با مسیر ساده‌تر
                simple_path = "unmatched_tags.xlsx"
                pd.DataFrame(columns=["Tag", "Source", "Status"]).to_excel(simple_path, index=False)
                logger.info(f"Created simple unmatched tags file at: {simple_path}")
            except:
                logger.error("Failed to create even a simple Excel file")
        
    def run_with_annotated_pdf(self, pdf_paths: 'List[str]', excel_path: str, output_excel_path: str, output_pdf_dir: str, 
                            create_zip: bool = True, zip_path: str = None) -> 'Tuple[List[str], List[str]]':
        """
        Run complete process with vector-based matching and generate annotated PDFs.
        Also adds tag numbers to the output Excel file and creates a ZIP archive of all output files.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path
            output_excel_path: Output Excel file path
            output_pdf_dir: Directory path for storing processed PDFs
            create_zip: Whether to create a ZIP archive of all output files
            zip_path: Path for the ZIP archive (if None, will use output_pdf_dir + '.zip')
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        import zipfile
        import os
        
        # Build tag vectors from Excel first
        start_time = time.time()
        self.build_tag_vectors_from_excel(excel_path)
        logger.info(f"Using tag patterns with {len(self.tag_patterns)} patterns")

        # 🆕 خواندن تگ‌های IO List
        io_tags = set()
        if excel_path:
            try:
                df = pd.read_excel(excel_path)
                if 'Tag No' in df.columns:
                    io_tags = set(str(tag).strip().upper() for tag in df['Tag No'] 
                                if pd.notna(tag))
                    logger.info(f"Loaded {len(io_tags)} tags from IO List")
            except Exception as e:
                logger.error(f"Error loading IO List: {e}")
        
        # Create output PDF directory if it doesn't exist
        os.makedirs(output_pdf_dir, exist_ok=True)
        
        # Store all similarity reports for detailed analysis
        all_similarity_reports = []
        
        # Master Dictionary to store tag-to-number mappings from all PDFs
        master_tag_numbers = {}
        
        # Dictionary to store all PDF processing results
        all_pdf_results = {}
        
        # لیست فایل‌های خروجی برای اضافه کردن به ZIP
        output_files = []
        
        # Process each PDF file
        for pdf_idx, pdf_path in enumerate(pdf_paths):
            pdf_filename = os.path.basename(pdf_path)
            logger.info(f"Processing PDF {pdf_idx + 1}/{len(pdf_paths)}: {pdf_filename}")
            
            # Process PDF to extract tags and JBs
            pdf_result = self.process_pdf(pdf_path)
            
            # Store PDF results with the PDF filename as key
            all_pdf_results[pdf_filename] = pdf_result
            
            # Create annotated PDF with vector matching results and get tag numbers
            output_pdf_path = os.path.join(output_pdf_dir, f"annotated_{pdf_filename}")
            pdf_tag_numbers = self.create_annotated_pdf(pdf_path, output_pdf_path)
            master_tag_numbers.update(pdf_tag_numbers)
            
            # اضافه کردن PDF حاشیه‌گذاری شده به لیست فایل‌های خروجی
            output_files.append(output_pdf_path)
            
            # Update master tag numbers Dictionary
            master_tag_numbers.update(pdf_tag_numbers)
            
            # Collect similarity reports from this PDF
            all_similarity_reports.extend(self.similarity_reports)
            
            # Generate per-PDF statistics
            pdf_stats = self.get_processing_stats()
            logger.info(f"PDF {pdf_filename} statistics:")
            for key, value in pdf_stats.items():
                logger.info(f"  {key}: {value}")
        
        # نام‌گذاری مناسب فایل‌های اکسل
        # فایل اکسل میانی با نام مشخص JB Wiring Diagram
        intermediate_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Intermediate.xlsx")
        
        # Pass all_pdf_results to add_wire_colors_and_scr_to_dataframe
        self.add_wire_colors_and_scr_to_dataframe(
            pd.DataFrame(), 
            master_tag_numbers, 
            intermediate_excel_path, 
            all_pdf_results,
            io_tags  # 🆕 ارسال تگ‌های IO
        )  
        
        # اضافه کردن فایل اکسل میانی به لیست فایل‌های خروجی
        output_files.append(intermediate_excel_path)
        
        # If IO List is provided, process and combine both Excel files
        if excel_path:
            # نام‌گذاری فایل اکسل نهایی با پسوند مناسب
            if not output_excel_path.endswith(".xlsx"):
                output_excel_path = output_excel_path.replace(".xls", ".xlsx") if output_excel_path.endswith(".xls") else f"{output_excel_path}.xlsx"
            
            # اگر نام فایل خروجی مشخص نشده، یک نام پیش‌فرض تعیین کنیم
            if not os.path.basename(output_excel_path):
                output_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx")
            
            final_df, unmatched_io_tags, unmatched_tags = self.process_excel_with_io_list(
                intermediate_excel_path, 
                excel_path, 
                output_excel_path
            )
            logger.info(f"Combined Excel file with IO List saved to: {output_excel_path}")
            
            # اضافه کردن فایل اکسل نهایی به لیست فایل‌های خروجی
            output_files.append(output_excel_path)
            
            # For function output
            unmatched_excel_tags = unmatched_io_tags
            unmatched_pdf_tags = unmatched_tags
            
            # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
            unmatched_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Unmatched_Tags.xlsx")
            self._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
            logger.info(f"Unmatched tags Excel file saved to: {unmatched_excel_path}")
            
            # اضافه کردن فایل اکسل تگ‌های تطبیق نیافته به لیست فایل‌های خروجی
            output_files.append(unmatched_excel_path)
        else:
            # If no IO List, just copy the intermediate file to the output path
            # نام‌گذاری فایل اکسل نهایی با پسوند مناسب
            if not output_excel_path.endswith(".xlsx"):
                output_excel_path = output_excel_path.replace(".xls", ".xlsx") if output_excel_path.endswith(".xls") else f"{output_excel_path}.xlsx"
                
            # اگر نام فایل خروجی مشخص نشده، یک نام پیش‌فرض تعیین کنیم
            if not os.path.basename(output_excel_path):
                output_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Final.xlsx")
                
            shutil.copy2(intermediate_excel_path, output_excel_path)
            logger.info(f"Excel file saved to: {output_excel_path}")
            
            # اضافه کردن فایل اکسل نهایی به لیست فایل‌های خروجی
            output_files.append(output_excel_path)
            
            # For function output
            unmatched_excel_tags = []
            unmatched_pdf_tags = []
            
            # ایجاد فایل اکسل خالی برای تگ‌های تطبیق نیافته
            unmatched_excel_path = os.path.join(output_pdf_dir, "JB_Wiring_Diagram_Unmatched_Tags.xlsx")
            self._create_unmatched_tags_excel([], [], unmatched_excel_path)
            logger.info(f"Empty unmatched tags Excel file saved to: {unmatched_excel_path}")
            
            # اضافه کردن فایل اکسل تگ‌های تطبیق نیافته به لیست فایل‌های خروجی
            output_files.append(unmatched_excel_path)
        
        # Generate summary statistics
        self.processing_time = time.time() - start_time
        stats = self.get_processing_stats()
        
        logger.info(f"Processing completed in {self.processing_time:.2f} seconds")
        logger.info(f"Summary statistics: {stats}")
        logger.info(f"Reports and tag numbers saved to: {reports_path}")
        logger.info(f"Total tags numbered: {len(master_tag_numbers)}")
        
        return unmatched_excel_tags, unmatched_pdf_tags
    
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
                        'Cable_Description', 'Type', 'Tag_Number_Status'  # اضافه کردن ستون وضعیت
                    ]
                    
                    # فقط ستون‌هایی که وجود دارند را انتخاب کن
                    available_columns = [col for col in column_order if col in new_df.columns]
                    new_df = new_df[available_columns]
                else:
                    # اگر دیتافریم خالی است، ایجاد دیتافریم با ستون‌های مناسب
                    new_df = pd.DataFrame(columns=[
                        'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                        'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number','Cable_Code', 'SCR_Terminal_Number',
                        'Cable_Description', 'Type', 'Tag_Number_Status'
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
                                'MC': mc_identifiers[0] if mc_identifiers else '',
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
                            spare_id = f"{getattr(self, 'spare_examples', 'SPARE')}_{spare_idx + 1}"
                            spare_number = tag_to_number.get(spare_id, page_tag_to_number.get(spare_id, ''))
                            
                            if not spare_number:
                                spare_number = row_counter
                            
                            row_data = {
                                'PDF_Name': pdf_name,
                                'Page': page_num,
                                'Tag/SPARE': spare,
                                'JB': jb_identifiers[0] if jb_identifiers else '',
                                'MC': mc_identifiers[0] if mc_identifiers else '',
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
        پردازش داده‌های یک صفحه با بررسی اعتبار
        
        ✅ FIX: حذف شرط exact match - همه تگ‌های OCR شده را پردازش می‌کند
        ✅ FIX: تگ‌های unmatched به intermediate اضافه می‌شوند
        """
        try:
            # بررسی ساختار
            if not isinstance(page_results, (tuple, list)):
                logger.error(f"Invalid page_results type: {type(page_results)}")
                return
            
            if len(page_results) < 8:
                logger.error(f"Insufficient data: {len(page_results)} items")
                return
            
            # استخراج داده‌ها
            tags = page_results[0] if len(page_results) > 0 else set()
            jb_identifiers = page_results[1] if len(page_results) > 1 else set()
            mc_identifiers = page_results[2] if len(page_results) > 2 else set()
            cable_descriptions = page_results[3] if len(page_results) > 3 else []
            spare_identifiers = page_results[4] if len(page_results) > 4 else []
            page_tag_to_number = page_results[5] if len(page_results) > 5 else {}
            raw_cable_descriptions = page_results[6] if len(page_results) > 6 else []
            tag_match_info = page_results[7] if len(page_results) > 7 else {}
            
            # تبدیل به لیست
            if isinstance(tags, set):
                tags = list(tags)
            if isinstance(jb_identifiers, set):
                jb_identifiers = list(jb_identifiers)
            if isinstance(mc_identifiers, set):
                mc_identifiers = list(mc_identifiers)
            if isinstance(spare_identifiers, set):
                spare_identifiers = list(spare_identifiers)
            
            logger.debug(f"Page {page_num} - Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}")
            
            # ============================================================
            # ✅ FIX 1: حذف شرط exact match
            # ============================================================
            # همه تگ‌ها (exact, similar, unmatched) را پردازش می‌کنیم
            
            # فقط شرط multiple JB را نگه می‌داریم
            if len(jb_identifiers) > 1:
                logger.warning(f"⚠️ Skipping page {page_num}: Multiple JBs {jb_identifiers}")
                return
            
            # ============================================================
            # ✅ FIX 2: پردازش همه تگ‌ها (شامل unmatched)
            # ============================================================
            row_counter = len(new_df_data) + 1
            
            # استخراج unmatched tags از tag_match_info
            unmatched_ocr_tags = [
                info['ocr_text'] 
                for key, info in tag_match_info.items() 
                if info.get('match_type') == 'unmatched'
            ]
            
            # پردازش تگ‌های matched (exact + similar)
            for tag in tags:
                fixed_tag = self.fix_common_ocr_errors(tag)
                tag_number = tag_to_number.get(tag, page_tag_to_number.get(tag, ''))
                
                if not tag_number:
                    tag_number = row_counter
                    row_counter += 1
                
                # 🆕 تولید اطلاعات ترمینال با الگوهای جدید
                terminal_info = self.generate_terminal_numbers(tag_number)
                
                # 🆕 تولید رنگ‌های سیم با الگوهای جدید
                wire_colors_str = self.generate_mc_wire_colors_enhanced(tag_number)
                wire_colors = [c.strip() for c in wire_colors_str.split(',')]
                
                # انتخاب دو رنگ اول
                wire_code_1 = wire_colors[0] if len(wire_colors) > 0 else ''
                wire_code_2 = wire_colors[1] if len(wire_colors) > 1 else ''
                
                match_status = 'Auto-Assigned'
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
                
                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': tag,
                    'JB': jb_identifiers[0] if jb_identifiers else '',
                    'MC': mc_identifiers[0] if mc_identifiers else '',
                    'Tag_Number': tag_number,
                    'Wire_Code_1': wire_code_1,  # 🆕 از الگوی جدید
                    'Wire_Code_2': wire_code_2,  # 🆕 از الگوی جدید
                    'Terminal_First_Number': terminal_info['terminal_first'],  # 🆕 از الگوی جدید
                    'Terminal_Second_Number': terminal_info['terminal_second'],  # 🆕 از الگوی جدید
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': terminal_info['scr_terminal'],  # 🆕 از الگوی جدید
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else '',
                    'Type': 'Tag',
                    'Tag_Number_Status': match_status
                }
                
                new_df_data.append(row_data)
                logger.debug(f"Added matched tag: {tag} ({match_status})")
            
            # ============================================================
            # ✅ FIX 3: اضافه کردن UNMATCHED tags به intermediate
            # ============================================================
            for ocr_tag in unmatched_ocr_tags:
                tag_number = row_counter
                row_counter += 1
                
                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': ocr_tag,
                    'JB': jb_identifiers[0] if jb_identifiers else '',
                    'MC': mc_identifiers[0] if mc_identifiers else '',
                    'Tag_Number': tag_number,
                    'Wire_Code_1': self.generate_mc_wire_colors(tag_number) if hasattr(self, 'generate_mc_wire_colors') else '',
                    'Wire_Code_2': '',
                    'Terminal_First_Number': str(tag_number),
                    'Terminal_Second_Number': str(tag_number + 1),
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': self.generate_scr_number(tag_number) if hasattr(self, 'generate_scr_number') else '',
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else '',
                    'Type': 'Tag',
                    'Tag_Number_Status': '❌ NOT IN IO LIST (Unmatched)'
                }
                
                new_df_data.append(row_data)
                logger.warning(f"❌ Added UNMATCHED tag: {ocr_tag}")
            
            # پردازش SPAREs (بدون تغییر)
            for spare_idx, spare in enumerate(spare_identifiers):
                spare_id = f"{getattr(self, 'spare_examples', 'SPARE')}_{spare_idx + 1}"
                spare_number = tag_to_number.get(spare_id, page_tag_to_number.get(spare_id, ''))
                
                if not spare_number:
                    spare_number = row_counter
                    row_counter += 1
                
                # 🆕 تولید اطلاعات ترمینال برای SPARE
                terminal_info = self.generate_terminal_numbers(spare_number)
                
                # 🆕 تولید رنگ‌های سیم برای SPARE
                wire_colors_str = self.generate_mc_wire_colors_enhanced(spare_number)
                wire_colors = [c.strip() for c in wire_colors_str.split(',')]
                
                wire_code_1 = wire_colors[0] if len(wire_colors) > 0 else ''
                wire_code_2 = wire_colors[1] if len(wire_colors) > 1 else ''
                
                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': spare,
                    'JB': jb_identifiers[0] if jb_identifiers else '',
                    'MC': mc_identifiers[0] if mc_identifiers else '',
                    'Tag_Number': spare_number,
                    'Wire_Code_1': wire_code_1,  # 🆕
                    'Wire_Code_2': wire_code_2,  # 🆕
                    'Terminal_First_Number': terminal_info['terminal_first'],  # 🆕
                    'Terminal_Second_Number': terminal_info['terminal_second'],  # 🆕
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': terminal_info['scr_terminal'],  # 🆕
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else '',
                    'Type': 'SPARE',
                    'Tag_Number_Status': 'Assigned'
                }
                
                new_df_data.append(row_data)
                logger.debug(f"Added spare: {spare}")
        
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            logger.error(traceback.format_exc())

    def process_excel_with_io_list(self, intermediate_excel_path: str, excel_path: str, output_path: str) -> 'Tuple[pd.DataFrame, List[str], List[str]]':
        """
        ترکیب داده‌های فایل intermediate با فایل IO List و ایجاد فایل اکسل نهایی.
        این تابع تمام ستون‌های IO List را حفظ می‌کند و ستون‌های جدید از فایل intermediate را به آن اضافه می‌کند.
        
        Args:
            intermediate_excel_path: مسیر فایل اکسل intermediate
            excel_path: مسیر فایل اکسل IO List
            output_path: مسیر فایل اکسل خروجی
            
        Returns:
            Tuple of (final_df, unmatched_io_tags, unmatched_tags)
        """
        try:
            # خواندن فایل‌های اکسل
            intermediate_df = pd.read_excel(intermediate_excel_path)
            io_list_df = pd.read_excel(excel_path)
            
            logger.info(f"Loaded intermediate Excel with {len(intermediate_df)} rows and {len(intermediate_df.columns)} columns")
            logger.info(f"Loaded IO List Excel with {len(io_list_df)} rows and {len(io_list_df.columns)} columns")
            
            # نام ستون تگ در هر دو فایل
            intermediate_tag_col = 'Tag/SPARE'
            io_list_tag_col = 'Tag No'  # نام ستون تگ در IO List
            
            # استخراج لیست تگ‌ها از هر دو فایل
            intermediate_tags = set(str(tag).strip().upper() for tag in intermediate_df[intermediate_tag_col] if pd.notna(tag))
            io_list_tags = set(str(tag).strip().upper() for tag in io_list_df[io_list_tag_col] if pd.notna(tag))
            
            # یافتن تگ‌های تطبیق نیافته
            unmatched_io_tags = list(io_list_tags - intermediate_tags)  # تگ‌های IO List که در intermediate نیستند
            unmatched_tags = list(intermediate_tags - io_list_tags)  # تگ‌های intermediate که در IO List نیستند
            
            logger.info(f"Unmatched IO List tags: {len(unmatched_io_tags)}")
            logger.info(f"Unmatched intermediate tags: {len(unmatched_tags)}")
            
            # ایجاد کپی از IO List برای حفظ تمام ستون‌های آن
            final_df = io_list_df.copy()
            
            # ستون‌های intermediate که می‌خواهیم اضافه کنیم
            intermediate_columns_to_add = [
                'PDF_Name', 'Page', 'JB', 'MC', 'Tag_Number', 
                'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number', 'SCR_Terminal_Number', 'Cable_code',
                'Cable_Description', 'Type', 'Tag_Number_Status'
            ]
            
            # فقط ستون‌هایی که در intermediate وجود دارند را اضافه کنیم
            intermediate_columns_to_add = [col for col in intermediate_columns_to_add if col in intermediate_df.columns]
            
            # اضافه کردن ستون‌های جدید به final_df
            for col in intermediate_columns_to_add:
                if col not in final_df.columns:
                    final_df[col] = None
            
            # تطبیق داده‌ها بر اساس تگ
            for idx, row in final_df.iterrows():
                io_tag = str(row[io_list_tag_col]).strip().upper() if pd.notna(row[io_list_tag_col]) else ""
                
                # جستجوی تگ در intermediate_df
                matching_rows = intermediate_df[intermediate_df[intermediate_tag_col].apply(
                    lambda x: str(x).strip().upper() == io_tag if pd.notna(x) else False
                )]
                
                if not matching_rows.empty:
                    # اگر تگ در intermediate پیدا شد، اطلاعات را به final_df اضافه کن
                    for col in intermediate_columns_to_add:
                        final_df.at[idx, col] = matching_rows.iloc[0][col]
            
            # اضافه کردن تگ‌های intermediate که در IO List نیستند به final_df
            if unmatched_tags:
                # فیلتر کردن ردیف‌های intermediate_df که تگ‌های آن‌ها در IO List نیستند
                unmatched_rows = intermediate_df[intermediate_df[intermediate_tag_col].apply(
                    lambda x: str(x).strip().upper() in unmatched_tags if pd.notna(x) else False
                )]
                
                # ایجاد دیتافریم جدید با ستون‌های final_df
                new_rows = pd.DataFrame(columns=final_df.columns)
                
                # اضافه کردن ردیف‌های جدید
                for _, row in unmatched_rows.iterrows():
                    new_row = pd.Series(index=final_df.columns)
                    
                    # کپی مقادیر از ستون‌های intermediate
                    for col in intermediate_columns_to_add:
                        new_row[col] = row[col]
                    
                    # تنظیم مقدار ستون تگ در IO List
                    new_row[io_list_tag_col] = row[intermediate_tag_col]
                    
                    # اضافه کردن ردیف جدید به new_rows
                    new_rows = pd.concat([new_rows, pd.DataFrame([new_row])], ignore_index=True)
                
                # اضافه کردن ردیف‌های جدید به final_df
                final_df = pd.concat([final_df, new_rows], ignore_index=True)
            
            # ذخیره دیتافریم نهایی به عنوان فایل اکسل
            final_df.to_excel(output_path, index=False)
            
            logger.info(f"Combined Excel file saved to: {output_path}")
            logger.info(f"Final Excel has {len(final_df)} rows and {len(final_df.columns)} columns")
            
            return final_df, unmatched_io_tags, unmatched_tags
            
        except Exception as e:
            logger.error(f"Error processing Excel files: {e}")
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
            
            tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number, raw_cable_descriptions, tag_match_info = result
            
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
                    tag_match_info  # پارامتر هشتم
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
        ✅ COMPLETE FIX: تشخیص قوی UZSO/UZSC با الگوهای گسترده OCR
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
        
        # Pattern 1: حالت کامل با خطاهای احتمالی
        uzso_pattern_full = r'^[UuVv][ZzSs2]?[Ss5]?[O0oDd][-_]?(\d+)$'
        uzsc_pattern_full = r'^[UuVv][ZzSs2]?[Ss5]?[CcGg][-_]?(\d+)$'
        
        # Pattern 2: حالت ناقص (فقط UZS یا UZ + شماره)
        uzs_pattern_incomplete = r'^[UuVv][ZzSs2][Ss5]?[-_]?(\d+)$'
        
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
        if re.match(r'^[UuVv][ZzSs2]5[O0oDd][-_]?(\d+)$', tag_upper):
            number = re.search(r'(\d+)$', tag_upper).group(1)
            return self._resolve_uzso_uzsc('UZSO', number, original_tag)
        
        if re.match(r'^[UuVv][ZzSs2]5[CcGg][-_]?(\d+)$', tag_upper):
            number = re.search(r'(\d+)$', tag_upper).group(1)
            return self._resolve_uzso_uzsc('UZSC', number, original_tag)
        
        # ============================================================
        # بقیه تصحیحات OCR عمومی (بدون تغییر)
        # ============================================================
        parts = re.split(r'(-)', tag)
        fixed_parts = []
        
        for p in parts:
            if p == '-':
                fixed_parts.append(p)
                continue
                
            sub = p
            
            if re.search(r'\d', sub):
                sub = sub.replace('O', '0').replace('I', '1').replace('Q', '0')
            
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