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
        """Calculate similarity between digit sequences using character-wise comparison."""
        if not seq1 or not seq2:
            return 0.0

        matches = 0
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))

        for i in range(min_len):
            if seq1[i] == seq2[i]:
                matches += 1
            elif seq2[i] in {'O', 'D'} and seq1[i] == '0':
                matches += 0.8
            elif seq2[i] in {'I', 'L', 'l'} and seq1[i] == '1':
                matches += 0.8
            elif seq2[i] == 'S' and seq1[i] == '5':
                matches += 0.8
            elif seq2[i] == 'B' and seq1[i] == '8':
                matches += 0.8

        return matches / max_len if max_len > 0 else 0.0

    def _are_digits_similar(self, digits1: str, digits2: str) -> bool:
        """Check if two digit sequences are similar accounting for OCR errors."""
        if abs(len(digits1) - len(digits2)) > min(len(digits1), len(digits2)) * 0.3:
            return False

        # Simplified Levenshtein distance with OCR tolerance
        ocr_map = {'0': 'OD', '1': 'ILl', '5': 'S', '8': 'B'}
        score = 0
        for a, b in zip(digits1, digits2):
            if a == b:
                score += 1
            elif b in ocr_map.get(a, ''):
                score += 0.8
        return score / max(len(digits1), len(digits2)) > 0.6

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
        return dot_product / (norm1**0.5 * norm2**0.5)

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
        
        # کامپایل اولیه الگوها
        self._compile_regex_patterns()
        
    def build_tag_vectors_from_excel(self, excel_path: str) -> None:
        """
        Build tag vectors from Excel file's Tag NO column.

        Args:
            excel_path: Path to Excel file containing tags
        """
        try:
            logger.info(f"Building tag vectors from Excel: {excel_path}")

            # Read Excel file
            df = pd.read_excel(excel_path)

            if 'Tag No' not in df.columns:
                raise ValueError("Excel file must contain a 'Tag No' column")

            # Extract and clean tags
            tags = df['Tag No'].dropna().astype(str).str.strip().str.upper().unique()

            # Add tags to vector matcher
            for tag in tags:
                self.vector_matcher.add_reference_tag(tag)

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

            logger.info(f"Successfully built tag vectors and tag patterns for {len(tags)} tags")

        except Exception as e:
            logger.error(f"Error building tag vectors: {e}")
            raise
    def calculate_tag_similarity(self, text1: str, text2: str) -> float:
        """
        محاسبه شباهت بین دو رشته متن با ترکیبی از روش‌های مختلف
        
        Args:
            text1: رشته اول
            text2: رشته دوم
            
        Returns:
            امتیاز شباهت بین 0 تا 1
        """
        # روش 1: شباهت رشته‌ای ساده
        string_similarity = self.calculate_string_similarity(text1, text2)
        
        # روش 2: شباهت بخش‌های تگ (اگر با خط تیره جدا شده‌اند)
        parts_similarity = 0
        if '-' in text1 and '-' in text2:
            parts_similarity = self.calculate_parts_similarity(text1, text2)
        
        # روش 3: شباهت عددی
        numeric_similarity = self.calculate_numeric_similarity(text1, text2)
        
        # روش 4: بررسی پیشوند مشترک
        prefix_similarity = 0
        if '-' in text1 and '-' in text2:
            prefix1 = text1.split('-')[0]
            prefix2 = text2.split('-')[0]
            if prefix1 == prefix2:
                prefix_similarity = 0.8
        
        # ترکیب امتیازهای مختلف - وزن بیشتر به شباهت رشته‌ای و پیشوند
        final_similarity = max(
            string_similarity,
            0.7 * parts_similarity + 0.3 * string_similarity,
            0.6 * prefix_similarity + 0.4 * string_similarity,
            0.5 * numeric_similarity + 0.5 * string_similarity
        )
        
        return final_similarity
    
    def calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        محاسبه شباهت دو رشته با استفاده از فاصله لواشتاین
        
        Args:
            str1: رشته اول
            str2: رشته دوم
            
        Returns:
            امتیاز شباهت بین 0 تا 1 (1 برای رشته‌های کاملاً یکسان)
        """
        try:
            import difflib
            similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
            return similarity
        except:
            # اگر کتابخانه difflib در دسترس نبود، از روش ساده‌تر استفاده می‌کنیم
            max_len = max(len(str1), len(str2))
            if max_len == 0:
                return 1.0
                
            # محاسبه تعداد کاراکترهای یکسان
            same_chars = sum(c1 == c2 for c1, c2 in zip(str1, str2))
            return same_chars / max_len
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
        محاسبه شباهت بخش‌های جدا شده با خط تیره دو تگ
        
        Args:
            tag1: تگ اول
            tag2: تگ دوم
            
        Returns:
            امتیاز شباهت بین 0 تا 1
        """
        # جدا کردن تگ‌ها به بخش‌های مختلف با خط تیره
        parts1 = tag1.split('-')
        parts2 = tag2.split('-')
        
        # اگر تعداد بخش‌های تگ‌ها متفاوت است، امتیاز کمتری بده
        if len(parts1) != len(parts2):
            return 0.5
            
        # محاسبه امتیاز شباهت برای هر بخش
        similarity_sum = 0
        for p1, p2 in zip(parts1, parts2):
            similarity_sum += self.calculate_string_similarity(p1, p2)
        
        # میانگین امتیازهای شباهت
        return similarity_sum / max(len(parts1), 1)
    
    def extract_and_match_tags(self, ocr_text_items: list, io_list_tags: list, threshold: float = 0.8) -> tuple:
        """
        استخراج تگ‌های کاندید از متن OCR و تطبیق آنها با تگ‌های IO List
        
        Args:
            ocr_text_items: لیست متن‌های استخراج شده از OCR
            io_list_tags: لیست تگ‌های IO List
            threshold: حد آستانه شباهت
            
        Returns:
            tuple of (matched_tags, tag_candidates)
            matched_tags: دیکشنری {ocr_text: io_tag} از تگ‌های تطبیق داده شده
            tag_candidates: دیکشنری {ocr_text: [(io_tag, similarity), ...]} از کاندیداهای هر متن
        """
        # پیدا کردن کاندیداهای تگ برای هر متن OCR
        tag_candidates = self.find_candidate_tags(ocr_text_items, io_list_tags, threshold)
        
        # انتخاب بهترین کاندیدا برای هر متن OCR
        matched_tags = {}
        for text, candidates in tag_candidates.items():
            if candidates:
                # بهترین کاندیدا (با بیشترین شباهت)
                best_match, best_similarity = candidates[0]
                
                # اگر شباهت از حد آستانه بالاتر است
                if best_similarity >= threshold:
                    matched_tags[text] = best_match
                    logger.info(f"Matched OCR text '{text}' to IO tag '{best_match}' with similarity {best_similarity:.2f}")
                    
                    # اضافه کردن به آمار
                    if text.upper() == best_match.upper():
                        self.exact_matches += 1
                    else:
                        self.similar_matches += 1
                    
                    # اضافه کردن به مجموعه تگ‌های تطبیق داده شده
                    if hasattr(self, 'matched_tags_set'):
                        self.matched_tags_set.add(text)
        
        # گزارش آمار
        logger.info(f"Tag matching completed: {len(matched_tags)} matches found out of {len(ocr_text_items)} OCR text items")
        logger.info(f"Exact matches: {self.exact_matches}, Similar matches: {self.similar_matches}")
        
        return matched_tags, tag_candidates
    
    def process_ocr_results_with_io_list(self, ocr_results: list, io_list_tags: list) -> dict:
        """
        پردازش نتایج OCR و تطبیق با تگ‌های IO List
        
        Args:
            ocr_results: لیست نتایج OCR (متن‌های استخراج شده از تصویر)
            io_list_tags: لیست تگ‌های IO List
            
        Returns:
            دیکشنری تگ‌های تطبیق داده شده {ocr_text: io_tag}
        """
        # استخراج متن‌های OCR
        ocr_text_items = []
        for result in ocr_results:
            if isinstance(result, dict) and 'text' in result:
                ocr_text_items.append(result['text'])
            elif isinstance(result, str):
                ocr_text_items.append(result)
        
        # حذف متن‌های خالی و تکراری
        ocr_text_items = [text.strip() for text in ocr_text_items if text.strip()]
        ocr_text_items = list(set(ocr_text_items))
        
        logger.info(f"Processing {len(ocr_text_items)} unique OCR text items")
        
        # تطبیق تگ‌ها
        matched_tags, _ = self.extract_and_match_tags(ocr_text_items, io_list_tags)
        
        return matched_tags

    def find_candidate_tags(self, query_tag, io_tags, min_similarity=0.6):
        """
        پیدا کردن کاندیداهای مشابه برای یک تگ در لیست IO با فیلترهای بهبود یافته
        
        Args:
            query_tag: تگ مورد جستجو
            io_tags: لیست تگ‌های IO
            min_similarity: حداقل میزان شباهت مورد نیاز
        
        Returns:
            لیست مرتب شده از کاندیداها با امتیاز شباهت آنها
        """
        logger.info(f"CANDIDATE_FUNCTION_CALLED: Finding candidate tags for '{query_tag}' among {len(io_tags)} IO tags")
        try:
            logger.debug(f"Finding candidate tags for {query_tag} among {len(io_tags)} IO tags")
            
            # مرحله 0: بررسی تطبیق دقیق اولیه
            query_tag_upper = str(query_tag).strip().upper()
            for io_tag in io_tags:
                if not io_tag or pd.isna(io_tag):
                    continue
                io_tag_upper = str(io_tag).strip().upper()
                if query_tag_upper == io_tag_upper:
                    logger.info(f"Found exact match: {query_tag} -> {io_tag}")
                    return [(io_tag, 1.0)]
            
            # استخراج اجزای تگ مورد جستجو
            query_prefix, query_first_num, query_second_num, query_suffix = self._split_tag_to_parts(query_tag)
            
            # محاسبه امتیاز شباهت برای تمام تگ‌های IO
            similarity_scores = []
            for io_tag in io_tags:
                if not io_tag or pd.isna(io_tag):
                    continue
                
                io_tag_str = str(io_tag).strip().upper()
                
                # استخراج اجزای تگ IO
                io_prefix, io_first_num, io_second_num, io_suffix = self._split_tag_to_parts(io_tag)
                
                # فیلتر 1: پیشوند باید یکسان یا بسیار شبیه باشد
                prefix_similarity = self._calculate_prefix_similarity(query_prefix, io_prefix)
                if prefix_similarity < 0.85:  # آستانه بالاتر برای پیشوند
                    continue
                
                # فیلتر 2: حداقل یکی از اعداد باید تطبیق نزدیک داشته باشد
                first_num_match = self._are_numeric_parts_very_similar(query_first_num, io_first_num)
                second_num_match = self._are_numeric_parts_very_similar(query_second_num, io_second_num)
                
                if not (first_num_match or second_num_match):
                    # اگر هیچ یک از اعداد تطبیق نزدیک ندارند، رد می‌شود
                    continue
                
                # فیلتر 3: اگر هر دو عدد وجود دارند، هر دو باید شبیه باشند
                if query_first_num and query_second_num and io_first_num and io_second_num:
                    if not (first_num_match and second_num_match):
                        continue
                
                # محاسبه امتیاز نهایی شباهت
                similarity = self._calculate_final_similarity_score(query_tag, io_tag)
                
                # افزودن به لیست اگر امتیاز از حداقل بیشتر باشد
                if similarity >= min_similarity:
                    similarity_scores.append((io_tag, similarity))
            
            # مرتب‌سازی بر اساس امتیاز شباهت (نزولی)
            sorted_candidates = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # فیلتر نهایی: فقط کاندیداهایی با امتیاز بالا را نگه دار
            # اگر بهترین کاندیدا امتیاز بالای 0.95 دارد، فقط کاندیداهای بالای 0.9 را نگه دار
            if sorted_candidates and sorted_candidates[0][1] > 0.95:
                sorted_candidates = [c for c in sorted_candidates if c[1] > 0.9]
            
            logger.debug(f"Found {len(sorted_candidates)} candidate tags for {query_tag}")
            for candidate, score in sorted_candidates[:3]:  # لاگ سه مورد اول
                logger.debug(f"  Candidate: {candidate}, Similarity: {score:.2f}")
            
            return sorted_candidates
            
        except Exception as e:
            logger.error(f"Error finding candidate tags for {query_tag}: {e}")
            return []


    def _split_tag_to_parts(self, tag):
        """
        تجزیه تگ به بخش‌های مختلف شامل پیشوند، اعداد و پسوند
        
        مثال: 
        - PIT-100-01A -> ('PIT', '100', '01', 'A')
        - PDIT-100-13 -> ('PDIT', '100', '13', '')
        - FT-101A -> ('FT', '101', '', 'A')
        
        Args:
            tag: رشته تگ
        
        Returns:
            Tuple of (prefix, first_number, second_number, suffix)
        """
        try:
            tag = str(tag).strip().upper()
            
            # الگوی استاندارد: پیشوند، خط تیره، عدد، خط تیره، عدد، پسوند (اختیاری)
            pattern1 = r'^([A-Z]+)-(\d+)-(\d+)([A-Z]*)$'
            
            # الگوی جایگزین: پیشوند، خط تیره، عدد، پسوند (اختیاری)
            pattern2 = r'^([A-Z]+)-(\d+)([A-Z]*)$'
            
            # الگوی ساده: پیشوند و عدد و پسوند (اختیاری)
            pattern3 = r'^([A-Z]+)(\d+)([A-Z]*)$'
            
            # تلاش برای تطبیق با الگوی استاندارد
            match = re.match(pattern1, tag)
            if match:
                prefix, first_num, second_num, suffix = match.groups()
                return prefix, first_num, second_num, suffix
                
            # تلاش برای تطبیق با الگوی جایگزین
            match = re.match(pattern2, tag)
            if match:
                prefix, first_num, suffix = match.groups()
                return prefix, first_num, "", suffix
                
            # تلاش برای تطبیق با الگوی ساده
            match = re.match(pattern3, tag)
            if match:
                prefix, first_num, suffix = match.groups()
                return prefix, first_num, "", suffix
            
            # اگر هیچکدام از الگوها تطبیق نکرد
            logger.debug(f"Tag {tag} does not match any pattern, returning parts as is")
            
            # جداسازی اجزاء بدون استفاده از الگو
            prefix = ''.join(c for c in tag if c.isalpha())
            numbers = ''.join(c for c in tag if c.isdigit())
            
            return prefix, numbers, "", ""
            
        except Exception as e:
            logger.error(f"Error splitting tag {tag}: {e}")
            return "", "", "", ""


    def _calculate_prefix_similarity(self, prefix1, prefix2):
        """
        محاسبه میزان شباهت بین دو پیشوند
        
        Args:
            prefix1: پیشوند اول
            prefix2: پیشوند دوم
        
        Returns:
            امتیاز شباهت بین 0 تا 1
        """
        try:
            if prefix1 == prefix2:
                return 1.0
                
            # اگر یکی از پیشوندها خالی باشد
            if not prefix1 or not prefix2:
                return 0.0
                
            # محاسبه فاصله Levenshtein
            distance = Levenshtein.distance(prefix1, prefix2)
            max_len = max(len(prefix1), len(prefix2))
            
            if max_len == 0:
                return 1.0
                
            return 1.0 - (distance / max_len)
            
        except Exception as e:
            logger.error(f"Error calculating prefix similarity: {e}")
            return 0.0

    def _are_numeric_parts_very_similar(self, num1, num2):
        """
        بررسی شباهت بسیار نزدیک بخش‌های عددی (سخت‌گیرانه‌تر)
        
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
            
            # حذف صفرهای ابتدایی برای مقایسه
            num1_stripped = num1.lstrip('0') or '0'
            num2_stripped = num2.lstrip('0') or '0'
            
            # تبدیل به عدد
            try:
                n1 = int(num1_stripped)
                n2 = int(num2_stripped)
            except ValueError:
                # اگر تبدیل به عدد ممکن نبود، مقایسه رشته‌ای انجام بده
                return num1_stripped == num2_stripped
            
            # بررسی برابری دقیق
            if n1 == n2:
                return True
            
            # تلرانس بسیار کم - فقط اختلاف 1 مجاز است
            diff = abs(n1 - n2)
            if diff <= 1:
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error comparing numeric parts {num1} and {num2}: {e}")
            return False

    def _calculate_final_similarity_score(self, tag1, tag2):
        """
        محاسبه امتیاز نهایی شباهت بین دو تگ
        
        Args:
            tag1: تگ اول
            tag2: تگ دوم
        
        Returns:
            امتیاز شباهت بین 0 تا 1
        """
        try:
            # تجزیه تگ‌ها به اجزا
            prefix1, first_num1, second_num1, suffix1 = self._split_tag_to_parts(tag1)
            prefix2, first_num2, second_num2, suffix2 = self._split_tag_to_parts(tag2)
            
            # محاسبه شباهت پیشوند - وزن 40%
            prefix_similarity = self._calculate_prefix_similarity(prefix1, prefix2) * 0.4
            
            # بررسی شباهت اعداد اول - وزن 30%
            if self._are_numeric_parts_very_similar(first_num1, first_num2):
                first_num_similarity = 0.3  # امتیاز کامل
            elif self._are_numeric_parts_similar(first_num1, first_num2):
                first_num_similarity = 0.2  # امتیاز نسبی
            else:
                first_num_similarity = 0.0  # بدون امتیاز
                
            # بررسی شباهت اعداد دوم - وزن 20%
            if self._are_numeric_parts_very_similar(second_num1, second_num2):
                second_num_similarity = 0.2  # امتیاز کامل
            elif self._are_numeric_parts_similar(second_num1, second_num2):
                second_num_similarity = 0.1  # امتیاز نسبی
            else:
                second_num_similarity = 0.0  # بدون امتیاز
                
            # بررسی شباهت پسوندها - وزن 10%
            suffix_similarity = 0.1 if suffix1 == suffix2 else 0.0
            
            # محاسبه امتیاز نهایی
            total_similarity = prefix_similarity + first_num_similarity + second_num_similarity + suffix_similarity
            
            return total_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity score between {tag1} and {tag2}: {e}")
            return 0.0

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
                self.spare_regex = re.compile(r'\b(spare|sp)\b', re.IGNORECASE)
                logger.debug(f"SPARE regex compiled (simple pattern): {self.spare_regex.pattern}")
            
            logger.info("✅ All regex patterns compiled successfully")
                
        except Exception as e:
            logger.error(f"Error compiling regex patterns: {e}")


    def extract_from_image(self, image: np.ndarray) -> 'Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int], List[str], Dict[str, Dict]]':
        """
        Extract tags, JB identifiers, MC identifiers, cable descriptions, SPAREs, and raw cable descriptions from the image.
        Also assigns and returns sequential numbers to tags and spares.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, 
                    tag_to_number, raw_cable_descriptions, tag_match_info)
        """
        # ============================================================
        # مرحله 0: مقداردهی اولیه و بررسی الگوها
        # ============================================================
        if not hasattr(self, 'jb_examples') or not self.jb_examples:
            logger.warning("JB examples not set, using default 'JB'")
            self.jb_examples = "JB"
        if not hasattr(self, 'mc_examples') or not self.mc_examples:
            logger.warning("MC examples not set, using default 'MC'")
            self.mc_examples = "MC"
        if not hasattr(self, 'spare_examples') or not self.spare_examples:
            logger.warning("SPARE examples not set, using default 'SPARE'")
            self.spare_examples = "SPARE"
        
        logger.info(f"Using patterns - JB: '{self.jb_examples}', MC: '{self.mc_examples}', SPARE: '{self.spare_examples}'")
        
        # تبدیل تصویر به RGB در صورت نیاز
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # تنظیم OCR config
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whiteList=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'
        
        # اجرای OCR
        logger.info("Starting OCR extraction...")
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        # مقداردهی متغیرها
        tags = set()
        jb_identifiers = set()
        mc_identifiers = set()
        cable_descriptions = []
        spare_identifiers = []
        raw_cable_descriptions = []
        tag_to_number = {}
        tag_match_info = {}  # 🔧 دیکشنری اصلی برای ردیابی نوع تطبیق
        
        # 🔧 FIX 1: جداسازی processed_identifiers برای تگ‌ها و SPARE‌ها
        processed_tag_texts = set()  # فقط برای OCR texts تگ‌ها
        processed_spare_indices = set()  # برای index های SPARE در OCR
        
        # کامپایل regex patterns
        if not hasattr(self, 'spare_regex') or not self.spare_regex:
            self._compile_regex_patterns()
        
        # 🔧 FIX: ایجاد الگوی SPARE ساده
        spare_pattern = re.compile(r'\b(spare|sp)\b', re.IGNORECASE)
        logger.debug(f"SPARE pattern compiled: {spare_pattern.pattern}")
        
        # الگوهای cable description
        cable_patterns = [
            re.compile(r'(\d+)\s*(P|PR|PAIR)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(T|TR|TRIPLE)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(C|CR|CORE)', re.IGNORECASE),
            re.compile(r'(\d+)\s*PAIR', re.IGNORECASE),
            re.compile(r'(\d+)\s*TRIPLE', re.IGNORECASE),
            re.compile(r'(\d+)\s*CORE', re.IGNORECASE),
            re.compile(r'(\d+)P\b', re.IGNORECASE),
            re.compile(r'(\d+)T\b', re.IGNORECASE),
            re.compile(r'(\d+)C\b', re.IGNORECASE),
        ]

        mc_positions = []
        mc_indices = []
        
        # شمارنده‌ها
        sequence_number = 1
        spare_found_count = 0

        # ============================================================
        # 🆕 مرحله 1: جستجوی EXACT TAG MATCHES (اولویت اول)
        # ============================================================
        logger.info("Phase 1: Searching for EXACT tag matches...")
        exact_matched_tags = set()
        
        # 🔧 بررسی وجود vector_matcher
        if not hasattr(self, 'vector_matcher'):
            logger.error("❌ vector_matcher NOT FOUND - Phase 1 will be skipped!")
            logger.error("   All tags will be marked as 'unknown'")
        else:
            logger.info(f"✅ vector_matcher exists")
            if hasattr(self.vector_matcher, 'reference_tags'):
                logger.info(f"   Reference tags count: {len(self.vector_matcher.reference_tags)}")
            
            for i, word in enumerate(ocr_data['text']):
                word_clean = word.strip().upper()
                if not word_clean or len(word_clean) < 4:
                    continue
                
                # 🔧 رد کردن JB/MC/SPARE (بدون اضافه به processed)
                if (self.jb_examples in word_clean or 
                    self.mc_examples in word_clean or 
                    spare_pattern.search(word_clean)):
                    continue
                
                # بررسی exact match با vector matcher
                similar_tags = self.vector_matcher.find_similar_tags(word_clean)
                if similar_tags:
                    best_match, best_score = similar_tags[0]
                    
                    # فقط exact matches (score >= 0.999)
                    if best_score >= 0.999:
                        if best_match not in exact_matched_tags:
                            exact_matched_tags.add(best_match)
                            tags.add(best_match)
                            
                            # تخصیص شماره
                            if best_match not in tag_to_number:
                                tag_to_number[best_match] = sequence_number
                                sequence_number += 1
                            
                            # 🔧 FIX 2: ثبت صحیح اطلاعات match
                            tag_match_info[best_match] = {
                                'match_type': 'exact',
                                'score': best_score,
                                'ocr_text': word_clean
                            }
                            
                            processed_tag_texts.add(word_clean)
                            logger.info(f"✅ EXACT match: {word_clean} → {best_match} (score: {best_score:.3f})")
        
        logger.info(f"Phase 1 complete: Found {len(exact_matched_tags)} exact matches")
        
        # ============================================================
        # 🆕 مرحله 2: جستجوی SIMILAR TAG MATCHES
        # ============================================================
        logger.info("Phase 2: Searching for SIMILAR tag matches...")
        similar_matched_tags = set()
        
        if hasattr(self, 'vector_matcher'):
            for i, word in enumerate(ocr_data['text']):
                word_clean = word.strip().upper()
                if not word_clean or len(word_clean) < 4:
                    continue
                
                # اگر قبلاً پردازش شده، رد کن
                if word_clean in processed_tag_texts:
                    continue
                
                # رد کردن JB/MC/SPARE
                if (self.jb_examples in word_clean or 
                    self.mc_examples in word_clean or 
                    spare_pattern.search(word_clean)):
                    continue
                
                # بررسی similar match
                similar_tags = self.vector_matcher.find_similar_tags(word_clean)
                if similar_tags:
                    best_match, best_score = similar_tags[0]
                    
                    # فقط similar matches (threshold <= score < 0.999)
                    if best_score >= self.vector_matcher.similarity_threshold and best_score < 0.999:
                        
                        # بررسی: آیا این تگ قبلاً با exact match اضافه شده؟
                        if best_match in exact_matched_tags:
                            logger.debug(f"⚠️ Ignoring similar match for {word_clean} → {best_match} (exact exists)")
                            continue
                        
                        # بررسی: آیا این تگ قبلاً با similar match اضافه شده؟
                        if best_match in similar_matched_tags:
                            logger.debug(f"⚠️ Ignoring duplicate similar match for {word_clean} → {best_match}")
                            continue
                        
                        similar_matched_tags.add(best_match)
                        tags.add(best_match)
                        
                        # تخصیص شماره
                        if best_match not in tag_to_number:
                            tag_to_number[best_match] = sequence_number
                            sequence_number += 1
                        
                        # 🔧 FIX 3: ثبت صحیح اطلاعات similar match
                        tag_match_info[best_match] = {
                            'match_type': 'similar',
                            'score': best_score,
                            'ocr_text': word_clean
                        }
                        
                        processed_tag_texts.add(word_clean)
                        
                        # ثبت گزارش
                        if hasattr(self, 'similarity_reports'):
                            self.similarity_reports.append({
                                'input_tag': word_clean,
                                'matched_tag': best_match,
                                'similarity_score': best_score,
                            })
                        
                        logger.info(f"⚠️ SIMILAR match: {word_clean} → {best_match} (score: {best_score:.3f})")
        else:
            logger.warning("⚠️ vector_matcher not found - Phase 2 skipped")
        
        logger.info(f"Phase 2 complete: Found {len(similar_matched_tags)} similar matches")
        
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

    def draw_bounding_boxes(self, image: np.ndarray, tags: 'Set[str]', jb_identifiers: 'Set[str]', 
                            mc_identifiers: 'Set[str]', cable_descriptions: 'List[str]', spare_identifiers: 'List[str]',
                            tag_to_number: 'Dict[str, int]', tag_match_info: 'Dict[str, Dict]' = None) -> 'Tuple[np.ndarray, Dict[str, int]]':
        """
        Draw bounding boxes with correct match type information and SPARE handling
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

        found_items = {
            'tags': [],
            'jbs': [],
            'mcs': [],
            'spares': [],
            'cables': []
        }
        
        processed_regions = set()
        all_tag_numbers = dict(tag_to_number)
        sequence_number = max(all_tag_numbers.values()) + 1 if all_tag_numbers else 1
        drawn_io_tags = set()
        
        # ============================================================
        # Phase 1: رسم Exact Matches
        # ============================================================
        logger.info("Phase 1: Drawing EXACT matches...")
        exact_drawn_count = 0
        
        for tag in tags:
            if tag.upper() in drawn_io_tags:
                continue
            
            # دریافت match info
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
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                if text_clean == tag_upper or text_clean == ocr_text_used.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        found_items['tags'].append({
                            'text': tag,
                            'position': region_key,
                            'match_type': 'exact',
                            'score': match_score
                        })
                        processed_regions.add(region_key)
                        drawn_io_tags.add(tag_upper)
                        exact_drawn_count += 1
                        logger.debug(f"✅ Drew exact match: {tag}")
                        break
        
        logger.info(f"Phase 1 complete: Drew {exact_drawn_count} exact matches")
        
        # ============================================================
        # Phase 2: رسم Similar Matches
        # ============================================================
        logger.info("Phase 2: Drawing SIMILAR matches...")
        similar_drawn_count = 0
        
        for tag in tags:
            if tag.upper() in drawn_io_tags:
                continue
            
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
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                region_key = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                
                if region_key in processed_regions:
                    continue
                
                if text_clean == ocr_text_used.upper():
                    found_items['tags'].append({
                        'text': tag,
                        'position': region_key,
                        'match_type': 'similar',
                        'score': match_score,
                        'original_text': text_clean
                    })
                    processed_regions.add(region_key)
                    drawn_io_tags.add(tag.upper())
                    similar_drawn_count += 1
                    logger.debug(f"⚠️ Drew similar match: {text_clean} → {tag}")
                    break
        
        logger.info(f"Phase 2 complete: Drew {similar_drawn_count} similar matches")
        
        # ============================================================
        # Phase 3: رسم JB identifiers
        # ============================================================
        logger.info("Phase 3: Drawing JB identifiers...")
        jb_drawn_count = 0
        
        for jb in jb_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == jb.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        found_items['jbs'].append({
                            'text': jb,
                            'position': region_key
                        })
                        processed_regions.add(region_key)
                        jb_drawn_count += 1
                        logger.debug(f"Found JB: {jb}")
                        break
        
        logger.info(f"Phase 3 complete: Drew {jb_drawn_count} JBs")
        
        # ============================================================
        # Phase 4: رسم MC identifiers
        # ============================================================
        logger.info("Phase 4: Drawing MC identifiers...")
        mc_drawn_count = 0
        
        for mc in mc_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == mc.upper():
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key not in processed_regions:
                        found_items['mcs'].append({
                            'text': mc,
                            'position': region_key
                        })
                        processed_regions.add(region_key)
                        mc_drawn_count += 1
                        logger.debug(f"Found MC: {mc}")
                        break
        
        logger.info(f"Phase 4 complete: Drew {mc_drawn_count} MCs")
        
        # ============================================================
        # 🔧 FIX 2: Phase 5 - رسم SPARE identifiers (بهبود یافته)
        # ============================================================
        logger.info(f"Phase 5: Drawing {len(spare_identifiers)} SPARE identifiers...")
        spare_drawn_count = 0
        
        # ایجاد regex pattern برای SPARE
        spare_pattern = re.compile(rf'\b{re.escape(self.spare_examples)}\b', re.IGNORECASE)
        
        for spare in spare_identifiers:
            spare_found = False
            
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                
                # 🔧 بررسی دقیق‌تر SPARE
                # فقط اگر کلمه SPARE به تنهایی باشد (نه بخشی از کلمه دیگر)
                if spare_pattern.search(text_clean):
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    
                    if region_key not in processed_regions:
                        spare_drawn_count += 1
                        spare_id = f"{self.spare_examples}_{spare_drawn_count}"
                        
                        # دریافت شماره SPARE
                        if spare_id in all_tag_numbers:
                            spare_number = all_tag_numbers[spare_id]
                        elif spare_id in tag_to_number:
                            spare_number = tag_to_number[spare_id]
                        else:
                            spare_number = sequence_number
                            all_tag_numbers[spare_id] = spare_number
                            sequence_number += 1
                        
                        found_items['spares'].append({
                            'text': spare,
                            'position': region_key,
                            'id': spare_id,
                            'number': spare_number
                        })
                        processed_regions.add(region_key)
                        spare_found = True
                        logger.debug(f"✅ Found SPARE: {spare} (#{spare_number})")
                        break
            
            if not spare_found:
                logger.warning(f"⚠️ Could not find bounding box for SPARE: {spare}")
        
        logger.info(f"Phase 5 complete: Drew {spare_drawn_count} SPAREs")
        
        # ============================================================
        # Phase 6: رسم Cable descriptions
        # ============================================================
        logger.info("Phase 6: Drawing cable descriptions...")
        cable_drawn_count = 0
        
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
                            found_items['cables'].append({
                                'text': cable_desc,
                                'position': region_key
                            })
                            processed_regions.add(region_key)
                            cable_drawn_count += 1
                            logger.debug(f"Found cable: {cable_desc}")
                            break
        
        logger.info(f"Phase 6 complete: Drew {cable_drawn_count} cables")
        
        # ============================================================
        # 🔧 FIX 3: رسم bounding boxes با رنگ‌بندی صحیح
        # ============================================================
        logger.info(f"Drawing all found items...")
        
        # رسم تگ‌ها
        for item in found_items['tags']:
            x, y, w, h = item['position']
            text = item['text']
            match_type = item.get('match_type', 'unknown')
            score = item.get('score', 0.0)
            
            if text not in all_tag_numbers:
                all_tag_numbers[text] = sequence_number
                sequence_number += 1
            
            tag_number = all_tag_numbers[text]
            
            # رنگ‌بندی بر اساس match type
            if match_type == 'exact':
                color = (0, 255, 0)      # سبز
                label_prefix = "✓"
            elif match_type == 'similar':
                color = (0, 165, 255)    # نارنجی
                label_prefix = "≈"
            else:  # unknown یا هر چیز دیگر
                color = (128, 128, 128)  # خاکستری
                label_prefix = "?"
                logger.warning(f"⚠️ Unknown match type for tag '{text}': {match_type}")
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            if match_type == 'similar' and score > 0:
                label = f"{label_prefix} #{tag_number} {text} ({score:.2f})"
            else:
                label = f"{label_prefix} #{tag_number} {text}"
            
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # رسم JBها
        for item in found_items['jbs']:
            x, y, w, h = item['position']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # آبی
            cv2.putText(image, f"JB: {item['text']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # رسم MCها
        for item in found_items['mcs']:
            x, y, w, h = item['position']
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)  # آبی روشن
            cv2.putText(image, f"MC: {item['text']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 🔧 FIX 4: رسم SPARE‌ها با رنگ مشخص
        for item in found_items['spares']:
            x, y, w, h = item['position']
            spare_number = item['number']
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 128), 2)  # بنفش
            cv2.putText(image, f"SPARE #{spare_number}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 128), 2)
        
        # رسم کابل‌ها
        for item in found_items['cables']:
            x, y, w, h = item['position']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 200), 2)  # زرد
            cv2.putText(image, f"Cable: {item['text']}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
        
        # ============================================================
        # آمار و Legend
        # ============================================================
        exact_count = len([item for item in found_items['tags'] if item.get('match_type') == 'exact'])
        similar_count = len([item for item in found_items['tags'] if item.get('match_type') == 'similar'])
        unknown_count = len([item for item in found_items['tags'] if item.get('match_type') == 'unknown'])
        spare_count = len(found_items['spares'])
        
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
        cv2.putText(image, f"JB: {len(found_items['jbs'])}", (legend_x_pos, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, f"MC: {len(found_items['mcs'])}", (legend_x_pos + 100, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"SPARE: {spare_count}", (legend_x_pos + 200, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
        cv2.putText(image, f"Cable: {len(found_items['cables'])}", (legend_x_pos + 330, legend_y_pos + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        # Stats summary
        stats_text = f"Total: {len(found_items['tags'])} tags, {spare_count} spares"
        cv2.putText(image, stats_text, (legend_x_pos, legend_y_pos + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        logger.info(f"✅ Bounding boxes drawn:")
        logger.info(f"   Tags: {exact_count} exact, {similar_count} similar, {unknown_count} unknown")
        logger.info(f"   Components: {len(found_items['jbs'])} JBs, {len(found_items['mcs'])} MCs, {spare_count} SPAREs")
        
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

    def generate_mc_wire_colors(self, tag_number):
        """
        تولید رنگ‌های سیم بر اساس شماره تگ و قانون تعریف شده
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            لیست رنگ‌های سیم
        """
        try:
            if not hasattr(self, 'wire_color_rule') or not self.wire_color_rule:
                return []
                
            # جداسازی قانون‌ها با کاما
            color_rules = [rule.strip() for rule in self.wire_color_rule.split(',')]
            
            # تولید رنگ‌ها با استفاده از قانون
            colors = []
            for rule in color_rules:
                # جایگزینی {number} با شماره تگ
                if '{number' in rule:
                    # بررسی فرمت اختیاری
                    format_match = re.search(r'\{number:([^}]+)\}', rule)
                    if format_match:
                        format_spec = format_match.group(1)
                        formatted_number = format(tag_number, format_spec)
                        color = rule.replace(format_match.group(0), formatted_number)
                    else:
                        color = rule.replace('{number}', str(tag_number))
                else:
                    # جایگزینی ساده عبارات ریاضی
                    # مثال: BK{number*2-1} -> BK1 برای tag_number=1
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
            return ""

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
        پردازش داده‌های یک صفحه با بررسی اعتبار و شرط exact match
        
        Args:
            new_df_data: لیست داده‌های دیتافریم جدید
            page_num: شماره صفحه
            page_results: نتایج پردازش صفحه
            pdf_name: نام فایل PDF
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌ها
            io_tags: مجموعه تگ‌های IO List (برای بررسی اعتبار)
        """
        try:
            # بررسی نوع و ساختار page_results
            if not isinstance(page_results, (tuple, list)):
                logger.error(f"Invalid page_results type for page {page_num}: {type(page_results)}")
                return
            
            if len(page_results) < 5:
                logger.error(f"Insufficient data in page_results for page {page_num}: {len(page_results)} items")
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
            
            logger.debug(f"Page {page_num} - Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}, Spares: {len(spare_identifiers)}")
            
            should_process_tags = True
            # ============================================================
            # 🆕 شرط 1: بررسی وجود حداقل یک exact match
            # ============================================================
            if io_tags and tags:
                tags_upper = set(str(tag).strip().upper() for tag in tags)
                io_tags_upper = set(str(tag).strip().upper() for tag in io_tags if tag and not pd.isna(tag))
                
                # 🆕 شمارش exact matches با استفاده از tag_match_info
                exact_match_count = 0
                exact_matched_tags = []
                
                if tag_match_info:
                    # استفاده از اطلاعات match type
                    for tag in tags:
                        if tag in tag_match_info:
                            info = tag_match_info[tag]
                            if info.get('match_type') == 'exact':
                                exact_match_count += 1
                                exact_matched_tags.append(tag)
                else:
                    # Fallback: بررسی مستقیم با IO List
                    logger.warning(f"⚠️ tag_match_info not available for page {page_num}, using direct comparison")
                    exact_matches = tags_upper.intersection(io_tags_upper)
                    exact_match_count = len(exact_matches)
                    exact_matched_tags = list(exact_matches)
                
                # 🆕 اگر هیچ exact match وجود ندارد، کل صفحه را رد کن
                if exact_match_count == 0:
                    logger.warning(f"❌ REJECTING page {page_num} of PDF {pdf_name}: NO EXACT MATCHES found")
                    logger.warning(f"   Tags in page: {list(tags_upper)[:5]}...")
                    should_process_tags = False

                    if tag_match_info:
                        similar_count = sum(1 for info in tag_match_info.values() if info.get('match_type') == 'similar')
                        logger.warning(f"   Found {similar_count} similar matches, but NO exact matches")
                    logger.warning(f"   This page will be completely skipped from intermediate Excel")
                    return  # خروج - هیچ داده‌ای اضافه نمی‌شود
                
                logger.info(f"✅ Page {page_num} validation PASSED: {exact_match_count} exact match(es) found")
                logger.debug(f"   Exact matched tags: {exact_matched_tags[:5]}...")
            
            # ============================================================
            # شرط 2: بیش از یک JB
            # ============================================================
            if len(jb_identifiers) > 1:
                logger.warning(f"⚠️ Skipping page {page_num} of PDF {pdf_name} - Multiple JBs detected: {jb_identifiers}")
                return
            
            # ============================================================
            # اضافه کردن داده‌ها به intermediate
            # ============================================================
            row_counter = len(new_df_data) + 1
            
            # پردازش تگ‌ها
            if should_process_tags:
                for tag in tags:
                    tag_number = tag_to_number.get(tag, page_tag_to_number.get(tag, ''))
                    
                    if not tag_number:
                        tag_number = row_counter
                        row_counter += 1
                    
                    # 🆕 تشخیص نوع match از tag_match_info
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
                            match_status = f'Unknown Match Type: {match_type}'
                            logger.warning(f"⚠️ Unknown match type '{match_type}' for tag '{tag}'")
                    else:
                        # Fallback: بررسی مستقیم
                        if io_tags:
                            tag_upper = str(tag).strip().upper()
                            io_tags_upper = set(str(t).strip().upper() for t in io_tags if t and not pd.isna(t))
                            if tag_upper in io_tags_upper:
                                match_status = 'Exact Match (fallback check)'
                            else:
                                match_status = 'Similar Match (assumed)'
                    
                    row_data = {
                        'PDF_Name': pdf_name,
                        'Page': page_num,
                        'Tag/SPARE': tag,
                        'JB': jb_identifiers[0] if jb_identifiers else '',
                        'MC': mc_identifiers[0] if mc_identifiers else '',
                        'Tag_Number': tag_number,
                        'Wire_Code_1': self.generate_mc_wire_colors(tag_number) if hasattr(self, 'generate_mc_wire_colors') else '',
                        'Wire_Code_2': '',
                        'Terminal_First_Number': str(tag_number),
                        'Terminal_Second_Number': str(tag_number + 1) if isinstance(tag_number, int) else str(int(tag_number) + 1) if str(tag_number).isdigit() else '',
                        'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                        'SCR_Terminal_Number': self.generate_scr_number(tag_number) if hasattr(self, 'generate_scr_number') else '',
                        'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else (cable_descriptions[0] if cable_descriptions else ''),
                        'Type': 'Tag',
                        'Tag_Number_Status': match_status  # 🆕 وضعیت بهبود یافته
                    }
                    
                    new_df_data.append(row_data)
                    logger.debug(f"Added tag row: {tag} with number {tag_number} ({match_status})")
                
            # پردازش SPARE ها (بدون تغییر)
            for spare_idx, spare in enumerate(spare_identifiers):
                spare_id = f"{getattr(self, 'spare_examples', 'SPARE')}_{spare_idx + 1}"
                spare_number = tag_to_number.get(spare_id, page_tag_to_number.get(spare_id, ''))
                
                if not spare_number:
                    spare_number = row_counter
                    row_counter += 1
                
                row_data = {
                    'PDF_Name': pdf_name,
                    'Page': page_num,
                    'Tag/SPARE': spare,
                    'JB': jb_identifiers[0] if jb_identifiers else '',
                    'MC': mc_identifiers[0] if mc_identifiers else '',
                    'Tag_Number': spare_number,
                    'Wire_Code_1': self.generate_mc_wire_colors(spare_number) if hasattr(self, 'generate_mc_wire_colors') else '',
                    'Wire_Code_2': '',
                    'Terminal_First_Number': str(spare_number),
                    'Terminal_Second_Number': str(spare_number + 1) if isinstance(spare_number, int) else str(int(spare_number) + 1) if str(spare_number).isdigit() else '',
                    'Cable_Code': cable_descriptions[0] if cable_descriptions else '',
                    'SCR_Terminal_Number': self.generate_scr_number(spare_number) if hasattr(self, 'generate_scr_number') else '',
                    'Cable_Description': raw_cable_descriptions[0] if raw_cable_descriptions else (cable_descriptions[0] if cable_descriptions else ''),
                    'Type': 'SPARE',
                    'Tag_Number_Status': 'Assigned' if tag_to_number.get(spare_id) or page_tag_to_number.get(spare_id) else 'Auto-Assigned'
                }
                
                new_df_data.append(row_data)
                logger.debug(f"Added spare row: {spare} with number {spare_number}")
        
        except Exception as e:
            logger.error(f"Error processing page {page_num} of PDF {pdf_name}: {e}")
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
        page_tags_upper = set(str(tag).strip().upper() for tag in page_tags)
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