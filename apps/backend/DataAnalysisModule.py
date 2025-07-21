import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
import os
import gc
import fitz 
from typing import Dict, List, Set, Tuple, Optional ,Union
import tempfile
import logging
from multiprocessing import Pool, cpu_count
import Levenshtein
import time
from typing import Any
import os
import tempfile
import math
import json
import traceback
from typing import List, Dict, Tuple
import math
import string
import shutil
import random

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
    
    def create_tag_vector(self, tag: str) -> Dict[str, float]:
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

    def calculate_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
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

    def find_similar_tags(self, input_tag: str) -> List[Tuple[str, float]]:
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
    
    def __init__(self, tesseract_path: Optional[str] = None, excel_path: Optional[str] = None):
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
            logger.info(f"JB examples set: {self.jb_examples}")
        
        if mc_examples is not None:
            if isinstance(mc_examples, list) and mc_examples:
                self.mc_examples = mc_examples[0].upper()  # اولین عنصر را انتخاب کن
            elif isinstance(mc_examples, str) and mc_examples.strip():
                self.mc_examples = mc_examples.strip().upper()
            logger.info(f"MC examples set: {self.mc_examples}")
        
        if spare_examples is not None:
            if isinstance(spare_examples, list) and spare_examples:
                self.spare_examples = spare_examples[0].upper()  # اولین عنصر را انتخاب کن
            elif isinstance(spare_examples, str) and spare_examples.strip():
                self.spare_examples = spare_examples.strip().upper()
            logger.info(f"SPARE examples set: {self.spare_examples}")
        
        if cable_examples is not None:
            if isinstance(cable_examples, list):
                self.cable_examples = ', '.join(cable_examples)
            elif isinstance(cable_examples, str):
                self.cable_examples = cable_examples.strip()
            logger.info(f"Cable examples set: {self.cable_examples}")
        
        if wire_color_rule is not None:
            self.wire_color_rule = wire_color_rule
            logger.info(f"Wire color rule set: {wire_color_rule}")
        
        if scr_number_rule is not None:
            self.scr_number_rule = scr_number_rule
            logger.info(f"SCR number rule set: {scr_number_rule}")
        
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
            
            # الگوی SPARE
            if self.spare_examples:
                self.spare_regex = re.compile(rf'\b{re.escape(self.spare_examples)}\b', re.IGNORECASE)
                logger.debug(f"SPARE regex compiled: {self.spare_regex.pattern}")
                
        except Exception as e:
            logger.error(f"Error compiling regex patterns: {e}")
        
    def extract_from_image(self, image: np.ndarray) -> Tuple[Set[str], Set[str], Set[str], List[str], List[str], Dict[str, int]]:
        """
        Extract tags, JB identifiers, MC identifiers, cable descriptions, and SPAREs from the image.
        Also assigns and returns sequential numbers to tags and spares.
            
        Args:
            image: Input image as numpy array
                
        Returns:
            Tuple of (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number)
        """
        # اطمینان از وجود الگوها
        if not self.jb_examples:
            logger.warning("JB examples not set, using default 'JB'")
            self.jb_examples = "JB"
        if not self.mc_examples:
            logger.warning("MC examples not set, using default 'MC'")
            self.mc_examples = "MC"
        if not self.spare_examples:
            logger.warning("SPARE examples not set, using default 'SPARE'")
            self.spare_examples = "SPARE"
            
        logger.info(f"Using patterns - JB: '{self.jb_examples}', MC: '{self.mc_examples}', SPARE: '{self.spare_examples}'")
        
        # اگر تصویر grayscale است، به RGB تبدیل کن
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # تطبیق کامل config با draw_bounding_boxes
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'

        # OCR output with position data
        logger.info("Starting OCR extraction...")
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        tags = set()
        jb_identifiers = set()
        mc_identifiers = set()
        cable_descriptions = []
        spare_identifiers = []
        tag_to_number = {}  # Dictionary to store tag/spare to number mapping
        processed_identifiers = set()

        # استفاده از الگوی کامپایل شده برای SPARE
        if not self.spare_regex:
            self._compile_regex_patterns()
        
        # بهبود regex pattern برای cable descriptions
        cable_patterns = [
            re.compile(r'(\d+)\s*(P|PR|PAIR)', re.IGNORECASE),  # pair patterns
            re.compile(r'(\d+)\s*(T|TR|TRIPLE)', re.IGNORECASE),  # triple patterns  
            re.compile(r'(\d+)\s*(C|CR|CORE)', re.IGNORECASE),  # core patterns
            re.compile(r'(\d+)\s*PAIR', re.IGNORECASE),  # direct pair
            re.compile(r'(\d+)\s*TRIPLE', re.IGNORECASE),  # direct triple
            re.compile(r'(\d+)\s*CORE', re.IGNORECASE),  # direct core
            re.compile(r'(\d+)P\b', re.IGNORECASE),  # shorthand like "12P"
            re.compile(r'(\d+)T\b', re.IGNORECASE),  # shorthand like "12T"
            re.compile(r'(\d+)C\b', re.IGNORECASE),  # shorthand like "12C"
        ]

        mc_positions = []
        mc_indices = []
        
        # Initialize sequence number for tags and spares
        sequence_number = 1
            
        # Step 1: پردازش اولیه تمام کلمات
        logger.info("Processing all words from OCR...")
        spare_found_count = 0
        
        for i, word in enumerate(ocr_data['text']):
            word_clean = word.strip().upper()
            if not word_clean:
                continue

            # DEBUG: چاپ همه کلمات برای یافتن SPARE
            if self.spare_examples in word_clean:
                logger.debug(f"Potential {self.spare_examples} word found: '{word_clean}' at index {i}")

            # تشخیص SPARE identifiers - استفاده از regex کامپایل شده
            if self.spare_regex and self.spare_regex.search(word_clean):
                spare_identifiers.append(word_clean)
                processed_identifiers.add(word_clean)
                spare_found_count += 1
                logger.info(f"*** {self.spare_examples} FOUND ***: {word_clean} at index {i}")
                
                # Assign a sequence number to this SPARE
                spare_id = f"{self.spare_examples}_{spare_found_count}"
                tag_to_number[spare_id] = sequence_number
                sequence_number += 1
                continue
            
            # جستجوی manual برای کلمات مشابه SPARE
            spare_match = False
            spare_variations = [self.spare_examples, self.spare_examples.replace('A', 'E'), self.spare_examples.replace('A', 'I')]
            for variation in spare_variations:
                if variation in word_clean:
                    spare_identifiers.append(word_clean)
                    processed_identifiers.add(word_clean)
                    spare_found_count += 1
                    logger.info(f"*** {self.spare_examples} VARIATION FOUND ***: {word_clean} (matched: {variation}) at index {i}")
                    
                    # Assign a sequence number to this SPARE variation
                    spare_id = f"{self.spare_examples}_{spare_found_count}"
                    tag_to_number[spare_id] = sequence_number
                    sequence_number += 1
                    spare_match = True
                    break
            
            if spare_match:
                continue
                    
            # تشخیص MC identifiers
            if len(word_clean) >= len(self.mc_examples) + 1 and self.mc_examples in word_clean and 'AS' not in word_clean:
                x, y = ocr_data['left'][i], ocr_data['top'][i]
                mc_positions.append((x, y))
                mc_indices.append(i)
                mc_identifiers.add(word_clean)
                processed_identifiers.add(word_clean)
                logger.info(f"{self.mc_examples} identifier found: {word_clean}")
                continue

            # تشخیص JB identifiers
            if word_clean.startswith(self.jb_examples):
                jb_identifiers.add(word_clean)
                processed_identifiers.add(word_clean)
                logger.info(f"{self.jb_examples} identifier found: {word_clean}")
                continue
            
            # تشخیص Tags - این بخش در حلقه اصلی قرار می‌گیرد
            if (len(word_clean) >= 4 and 
                self.jb_examples not in word_clean and 
                self.mc_examples not in word_clean and 
                word_clean not in processed_identifiers):
                if hasattr(self, 'vector_matcher'):
                    similar_tags = self.vector_matcher.find_similar_tags(word_clean)
                    if similar_tags:
                        best_match, best_score = similar_tags[0]
                        if best_score >= self.vector_matcher.similarity_threshold:
                            tags.add(best_match)
                            
                            # Assign a sequence number to this tag if not already assigned
                            if best_match not in tag_to_number:
                                tag_to_number[best_match] = sequence_number
                                sequence_number += 1
                                
                            self.similarity_reports.append({
                                'input_tag': word_clean,
                                'matched_tag': best_match,
                                'similarity_score': best_score,
                            })
                            logger.info(f"Found tag: {best_match} (similarity: {best_score}, number: {tag_to_number[best_match]})")
                            processed_identifiers.add(word_clean)
                            continue

        logger.info(f"{self.spare_examples} search completed. Found {spare_found_count} {self.spare_examples} identifiers.")
        logger.info(f"Final spare_identifiers found: {spare_identifiers}")
        logger.info(f"Total tags with assigned numbers: {len(tag_to_number) - spare_found_count}")
        logger.info(f"Total spares with assigned numbers: {spare_found_count}")

        # Step 2: Find cable descriptions near each MC using spatial proximity
        for mc_i in mc_indices:
            mc_x, mc_y = ocr_data['left'][mc_i], ocr_data['top'][mc_i]
            
            # تعریف محدوده مکانی جستجو (پیکسل)
            search_radius_x = 300  # محدوده افقی
            search_radius_y = 100  # محدوده عمودی
            
            # جستجو بر اساس موقعیت مکانی
            nearby_words = []
            nearby_indices = []
            
            for j, word_j in enumerate(ocr_data['text']):
                if not word_j.strip() or len(word_j.strip()) < 1:
                    continue
                    
                word_x, word_y = ocr_data['left'][j], ocr_data['top'][j]
                
                # محاسبه فاصله مکانی
                distance_x = abs(word_x - mc_x)
                distance_y = abs(word_y - mc_y)
                
                # اگر در محدوده مکانی باشد
                if distance_x <= search_radius_x and distance_y <= search_radius_y:
                    nearby_words.append(word_j.strip())
                    nearby_indices.append(j)
                    logger.debug(f"Word '{word_j.strip()}' at distance ({distance_x}, {distance_y}) from {self.mc_examples}")
                
            # تمیز کردن و ترکیب متن
            combined_text = ' '.join(nearby_words).upper()
            logger.debug(f"Combined text near {self.mc_examples} {mc_i}: '{combined_text}'")
            
            # جستجو با patterns مختلف
            found_cable = False
            for pattern in cable_patterns:
                matches = pattern.findall(combined_text)
                for match in matches:
                    if isinstance(match, tuple):
                        number = match[0]
                        cable_type = match[1] if len(match) > 1 else ''
                    else:
                        # برای patterns ساده‌تر
                        number = match
                        cable_type = ''
                    
                    # تعیین نوع کابل
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
                        # حدس زدن نوع بر اساس context
                        if 'PAIR' in combined_text:
                            cable_type_full = 'pair'
                        elif 'TRIPLE' in combined_text:
                            cable_type_full = 'triple'
                        elif 'CORE' in combined_text:
                            cable_type_full = 'core'
                        else:
                            cable_type_full = 'pair'  # default
                    
                    cable_desc = f"{number} {cable_type_full}"
                    if cable_desc not in cable_descriptions:
                        cable_descriptions.append(cable_desc)
                        logger.info(f"Found cable description: {cable_desc}")
                        found_cable = True
            
            # اگر هیچ cable description پیدا نشد، تمام کلمات نزدیک را چاپ کن
            if not found_cable:
                logger.debug(f"No cable description found near {self.mc_examples}. Nearby words: {nearby_words}")
                
                # جستجوی دستی برای اعداد
                for word in nearby_words:
                    clean_word = word.strip().upper()
                    # جستجو برای اعداد منفرد که ممکن است cable باشند
                    if re.match(r'^\d+$', clean_word):
                        potential_cable = f"{clean_word} pair"  # default to pair
                        logger.debug(f"Found potential cable number: {clean_word}")
                        if potential_cable not in cable_descriptions:
                            cable_descriptions.append(potential_cable)

        logger.info('Final cable_descriptions:', cable_descriptions)
        logger.info('Final spare_identifiers:', spare_identifiers)
        logger.info('Final tag_to_number mapping:', tag_to_number)
        logger.info('Final tags found:', tags)

        # Update final sets
        self.all_tags.update(tags)
        self.all_jbs.update(jb_identifiers)
        self.all_mcs.update(mc_identifiers)
        self.all_spares = spare_identifiers

        return tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number
        
    def get_similarity_reports(self) -> List[Dict[str, Any]]:
        """
        دریافت گزارشات کامل شباهت بین تگ‌های شناسایی شده و تگ‌های مرجع
        
        Returns:
            لیستی از دیکشنری‌های حاوی گزارشات شباهت
        """
        return self.similarity_reports
    
    def get_top_similar_tags(self, n: int = 10) -> List[Dict[str, Any]]:
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
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings with enhanced focus on alphabetic characters.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Clean and normalize texts
        text1 = str(text1).upper().strip()
        text2 = str(text2).upper().strip()
        
        # If texts are identical, return 1
        if text1 == text2:
            return 1.0
            
        # If one text is empty, return 0
        if not text1 or not text2:
            return 0.0
        
        # Calculate length similarity
        len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        # Calculate character-based similarity with focus on alphabetic chars
        chars1 = set(text1)
        chars2 = set(text2)
        common_chars = len(chars1.intersection(chars2))
        char_ratio = common_chars / max(len(chars1), len(chars2))
        
        # Extract alphabetic parts
        alpha1 = ''.join(c for c in text1 if c.isalpha())
        alpha2 = ''.join(c for c in text2 if c.isalpha())
        
        # Calculate alphabetic similarity using Levenshtein distance if both have alphabetic chars
        if alpha1 and alpha2:
            # Normalize the distance to get a similarity score between 0 and 1
            alpha_distance = Levenshtein.distance(alpha1, alpha2)
            alpha_max_len = max(len(alpha1), len(alpha2))
            alpha_similarity = 1 - (alpha_distance / alpha_max_len) if alpha_max_len > 0 else 0
        else:
            # If one or both strings have no alphabetic chars, use a lower similarity
            alpha_similarity = 0.5 if (not alpha1 and not alpha2) else 0
        
        # Calculate pattern similarity
        pattern1 = [1 if c.isdigit() else 2 if c.isalpha() else 3 for c in text1]
        pattern2 = [1 if c.isdigit() else 2 if c.isalpha() else 3 for c in text2]
        pattern_ratio = sum(a == b for a, b in zip(pattern1[:min(len(pattern1), len(pattern2))]))
        pattern_ratio /= max(len(pattern1), len(pattern2))
        
        # Calculate position-weighted similarity with higher weight for alphabetic chars
        pos_weight = 0
        min_len = min(len(text1), len(text2))
        for i in range(min_len):
            if text1[i] == text2[i]:
                # Give higher weight to matching alphabetic characters
                if text1[i].isalpha():
                    pos_weight += 1.5 / (i + 1)  # 50% more weight for alphabetic matches
                else:
                    pos_weight += 1 / (i + 1)
        
        # Calculate the denominator for normalization
        pos_denominator = sum(1.5 / (i + 1) if i < min_len and text1[i].isalpha() else 1 / (i + 1) for i in range(max(len(text1), len(text2))))
        pos_ratio = pos_weight / pos_denominator if pos_denominator > 0 else 0
        
        # Calculate prefix similarity (first few chars are often more important)
        prefix_len = min(3, min(len(text1), len(text2)))
        if prefix_len > 0:
            prefix_matches = sum(text1[i] == text2[i] for i in range(prefix_len))
            prefix_ratio = prefix_matches / prefix_len
        else:
            prefix_ratio = 0
        
        # Combine all metrics with adjusted weights to emphasize alphabetic similarity
        similarity = (
            0.15 * len_ratio +       # Length similarity weight (reduced)
            0.30 * char_ratio +      # Character similarity weight (increased)
            0.25 * alpha_similarity + # Alphabetic similarity (new component)
            0.15 * pattern_ratio +   # Pattern similarity weight (reduced)
            0.10 * pos_ratio +       # Position similarity weight (reduced)
            0.05 * prefix_ratio      # Prefix similarity (new component)
        )
        
        return min(1.0, similarity)  # Ensure result is at most 1.0
    
                
    def create_tag_jb_mapping(self, pdf_results: Dict[int, Tuple]) -> Dict[str, str]:
        """
        Create a mapping from tags to JB identifiers.
        
        Args:
            pdf_results: Dictionary mapping page numbers to tuples of (tags, jb_identifiers)
            
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
            tag_coordinates: list - List of dictionaries containing tag coordinates
            column_threshold: int - Threshold for column detection
            
        Returns:
            set - New tags found during column analysis
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


    def process_pdf_page(self, page_info: Tuple[fitz.Page, str, int]) -> Tuple[int, Set[str], Set[str], Set[str], List[str], List[str]]:
        """
        Process a single PDF page in parallel.
        
        Args:
            page_info: Tuple containing (page object, temp_dir path, page number)
            
        Returns:
            Tuple of (page_number, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number)
        """
        page, temp_dir, page_num = page_info
        
        # Create image path
        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        pix.save(image_path)
        
        # Load and process image
        image = cv2.imread(image_path)
        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers , tag_to_number = self.extract_from_image(image)
        
        # Clean up temporary image file
        try:
            os.remove(image_path)
        except:
            pass
            
        return page_num + 1, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number


    def process_pdf(self, pdf_path: str) -> Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str]]]:
        """
        Process all pages in a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to tuples of (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number)
        """
        results = {}
        
        # Reinitialize Tesseract for this process
        try:
            # Try to find Tesseract in common locations
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
            # Process pages sequentially within each PDF
            for page_num in range(len(pdf_document)):
                try:
                    logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
                    
                    # Get page
                    page = pdf_document[page_num]
                    
                    # Convert page to image with higher resolution
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix.save(image_path)
                    
                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"Failed to load image for page {page_num + 1}")
                        continue
                    
                    # Extract tags and JB identifiers
                    tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers , tag_to_number= self.extract_from_image(image)
                    
                    # Store results
                    results[page_num + 1] = (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers,tag_to_number)
                    
                    # Print results immediately for this page
                    print(f"Page {page_num + 1}:")
                    print(f"  Tags found ({len(tags)}): {', '.join(sorted(tags))}")
                    print(f"  JB identifiers found ({len(jb_identifiers)}): {', '.join(sorted(jb_identifiers))}")
                    print(f"  MC identifiers found ({len(mc_identifiers)}): {', '.join(sorted(mc_identifiers))}")
                    print(f"  Cable descriptions found ({len(cable_descriptions)}): {', '.join(sorted(cable_descriptions))}")
                    print(f"  Spare identifiers found ({len(spare_identifiers)}): {', '.join(sorted(spare_identifiers))}")
                    print(f"  Spare identifiers found ({len(tag_to_number)}): {', '.join(sorted(tag_to_number))}")
                    # Clean up temporary image file
                    try:
                        os.remove(image_path)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
        
        return results

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str]]]:
        """
        Process multiple PDF files in parallel with improved resource management.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping page numbers to tuples of (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers,tag_to_number)
        """
        combined_results = {}
        page_offset = 0
        
        # Calculate optimal number of processes
        num_processes = min(cpu_count(), len(pdf_paths), 4)  
        logger.info(f"Processing {len(pdf_paths)} PDF files using {num_processes} processes")
        
        try:
            # Use context manager for better resource management
            with Pool(processes=num_processes) as pool:
                # Use imap instead of map for better memory management with large files
                for pdf_result in pool.imap_unordered(self.process_pdf, pdf_paths):
                    if pdf_result:
                        for page_num, (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number) in pdf_result.items():
                            combined_results[page_num + page_offset] = (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number)
                        page_offset += max(pdf_result.keys()) if pdf_result else 0
                        
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            logger.info("Falling back to sequential processing")
            for pdf_path in pdf_paths:
                try:
                    pdf_result = self.process_pdf(pdf_path)
                    for page_num, (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number) in pdf_result.items():
                        combined_results[page_num + page_offset] = (tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers ,tag_to_number)
                    page_offset += max(pdf_result.keys()) if pdf_result else 0
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {e}")
                    continue
        
        return combined_results

    def process_excel_with_io_list(self, intermediate_excel_path: str, excel_path: str ,output_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        ترکیب داده‌های فایل intermediate با فایل IO List و ایجاد فایل اکسل نهایی.
        این تابع تمام ستون‌های IO List را حفظ می‌کند و ستون‌های جدید از فایل intermediate را به آن اضافه می‌کند.
        
        Args:
            intermediate_excel_path: مسیر فایل اکسل intermediate
            io_list_path: مسیر فایل اکسل IO List
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
                'BK_Colors', 'WT_Colors', 'Terminal_First', 'Terminal_Second', 'Terminal_Label',
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
        
    def create_simple_vector(self, tag: str) -> List[float]:
            """Create an improved feature vector for a tag"""
            tag = str(tag).upper().strip()
            
            
            vector = [
                len(tag) * 1.5,  
                sum(c.isdigit() for c in tag) * 2.0,  
                sum(c.isalpha() for c in tag) * 1.5,
                sum(c == '-' for c in tag) * 3.0, 
                len(tag.split('-')) * 2.0,  
            ]
            
           
            prefixes = ['TIT', 'FIT', 'PIT', 'LIT', 'TCV', 'FCV', 'PCV', 'LCV', 
                        'UZSO', 'UZSC', 'UY', 'UHSL', 'UHSH', 'TY', 'LA']
            for prefix in prefixes:
                vector.append(5.0 if tag.startswith(prefix) else 0.0)
            
        
            parts = tag.split('-')
            if len(parts) >= 2:
                vector.extend([
                    float(hash(parts[0]) % 1000),  
                    float(hash(parts[-1]) % 1000),  
                    sum(c.isdigit() for c in parts[-1]) * 2.0  
                ])
            else:
                vector.extend([0.0, 0.0, 0.0])
            
            return vector

        
    def calculate_vector_similarity(self, vec1: List[float], vec2: List[float]) -> float:
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

    def run(self, pdf_paths: List[str], excel_path: str, output_excel_path: str, intermediate_excel_path: str) -> Tuple[List[str], List[str], List[str]]:
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

    def draw_bounding_boxes(self, image: np.ndarray, tags: Set[str], jb_identifiers: Set[str], 
                    mc_identifiers: Set[str], cable_descriptions: List[str], spare_identifiers: List[str],
                    tag_to_number: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, int]]:
    
        jb_examples = self.jb_examples
        mc_examples = self.mc_examples
        spare_examples = self.spare_examples
        cable_examples = self.cable_examples
    
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZsparetcoilpr0123456789-.'
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

        processed_regions = set()
        processed_identifiers = set()
        all_components = []
        tag_to_number = {}

        # پردازش Tags
        for tag in tags:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == tag:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key in processed_regions:
                        continue
                    processed_regions.add(region_key)
                    processed_identifiers.add(text_clean)
                    x, y, w, h = region_key
                    all_components.append({
                        'type': 'Tag',
                        'text': tag,
                        'position': (x, y, w, h),
                        'color': (255, 0, 0)
                    })
        
        # پردازش Spare identifiers
        spare_count = 0
        spare_pattern = re.compile(rf'\b{spare_examples}\b', re.IGNORECASE)
        for i, text in enumerate(ocr_data['text']):
            text_clean = text.strip().upper()
            if spare_pattern.search(text_clean):
                region_key = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                if region_key in processed_regions:
                    continue
                processed_regions.add(region_key)
                processed_identifiers.add(text_clean)
                x, y, w, h = region_key
                spare_count += 1
                spare_id = f"{spare_examples}_{spare_count}"
                all_components.append({
                    'type': 'Spare',
                    'text': text_clean,
                    'id': spare_id,
                    'position': (x, y, w, h),
                    'color': (128, 0, 128)
                })

        # پردازش Cable descriptions با استفاده از جستجوی مکانی
        if mc_identifiers and cable_descriptions:
            # ابتدا موقعیت MC ها را پیدا کن
            mc_positions = {}
            for mc in mc_identifiers:
                for i, text in enumerate(ocr_data['text']):
                    text_clean = text.strip().upper()
                    if text_clean == mc:
                        mc_x, mc_y = ocr_data['left'][i], ocr_data['top'][i]
                        mc_positions[mc] = (mc_x, mc_y)
                        break
            
            # برای هر cable description، نزدیک‌ترین MC را پیدا کن و متن مربوطه را علامت‌گذاری کن
            for cd in cable_descriptions:
                print(f"Processing cable description: {cd}")
                
                # اگر cable description یک pattern خاص مثل FRT-01P1.5RM2 باشد
                if re.match(r'[A-Z]+-\d+P\d*\.?\d*MM\d*', cd, re.IGNORECASE):
                    # جستجو برای این pattern خاص در OCR data
                    for i, text in enumerate(ocr_data['text']):
                        text_clean = text.strip().upper()
                        if text_clean == cd.upper() or cd.upper() in text_clean:
                            region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                        ocr_data['width'][i], ocr_data['height'][i])
                            if region_key in processed_regions:
                                continue
                            processed_regions.add(region_key)
                            processed_identifiers.add(text_clean)
                            x, y, w, h = region_key
                            all_components.append({
                                'type': 'Cable',
                                'text': cd,
                                'position': (x, y, w, h),
                                'color': (0, 200, 200)
                            })
                            print(f"Found cable description pattern: {cd} at ({x}, {y})")
                            break
                else:
                    # برای cable descriptions عادی (مثل "12 pair")
                    cd_parts = cd.split()
                    if len(cd_parts) != 2:
                        continue
                        
                    number, cable_type = cd_parts
                    
                    # جستجوی مکانی برای number نزدیک MC ها
                    for mc, (mc_x, mc_y) in mc_positions.items():
                        search_radius_x = 300
                        search_radius_y = 100
                        
                        for i, text in enumerate(ocr_data['text']):
                            text_clean = text.strip().upper()
                            if number in text_clean:
                                word_x, word_y = ocr_data['left'][i], ocr_data['top'][i]
                                distance_x = abs(word_x - mc_x)
                                distance_y = abs(word_y - mc_y)
                                
                                # اگر در محدوده مکانی MC باشد
                                if distance_x <= search_radius_x and distance_y <= search_radius_y:
                                    # بررسی وجود cable type در نزدیکی
                                    found_cable_type = False
                                    for j in range(max(0, i - 3), min(len(ocr_data['text']), i + 4)):
                                        neighbor_text = ocr_data['text'][j].strip().upper()
                                        cable_type_terms = ['PAIR', 'P', 'PR', 'TRIPLE', 'T', 'TR', 'CORE', 'C', 'CR']
                                        if any(term in neighbor_text for term in cable_type_terms):
                                            found_cable_type = True
                                            break
                                    
                                    if found_cable_type:
                                        region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                                    ocr_data['width'][i], ocr_data['height'][i])
                                        if region_key in processed_regions:
                                            continue
                                        processed_regions.add(region_key)
                                        processed_identifiers.add(text_clean)
                                        x, y, w, h = region_key
                                        all_components.append({
                                            'type': 'Cable',
                                            'text': cd,
                                            'position': (x, y, w, h),
                                            'color': (0, 200, 200)
                                        })
                                        print(f"Found cable description: {cd} at ({x}, {y}) near MC {mc}")
                                        break

        # پردازش MC identifiers
        for mc in mc_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == mc:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key in processed_regions:
                        continue
                    processed_regions.add(region_key)
                    processed_identifiers.add(text_clean)
                    x, y, w, h = region_key
                    all_components.append({
                        'type': 'MC',
                        'text': mc,
                        'position': (x, y, w, h),
                        'color': (0, 255, 0)
                    })

        # پردازش JB identifiers
        for jb in jb_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == jb:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    if region_key in processed_regions:
                        continue
                    processed_regions.add(region_key)
                    processed_identifiers.add(text_clean)
                    x, y, w, h = region_key
                    all_components.append({
                        'type': 'JB',
                        'text': jb,
                        'position': (x, y, w, h),
                        'color': (0, 0, 255)
                    })

        # پردازش Tags با vector matching
        if hasattr(self, 'vector_matcher'):
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if not text_clean or len(text_clean) < 4:
                    continue
                region_key = (ocr_data['left'][i], ocr_data['top'][i],
                            ocr_data['width'][i], ocr_data['height'][i])
                if region_key in processed_regions:
                    continue
                # بررسی کن که آیا این متن قبلاً به عنوان JB، MC یا SPARE شناسایی شده است
                if text_clean in processed_identifiers:
                    continue
                similar_tags = self.vector_matcher.find_similar_tags(text_clean)
                if similar_tags:
                    best_match, similarity = similar_tags[0]
                    # بررسی کن که آیا بهترین تطابق قبلاً به عنوان JB، MC یا SPARE شناسایی شده است
                    if best_match not in processed_identifiers and best_match not in tags and similarity >= self.vector_matcher.similarity_threshold:
                        processed_regions.add(region_key)
                        processed_identifiers.add(text_clean)  # اضافه کردن متن به لیست شناسایی‌شده‌ها
                        processed_identifiers.add(best_match)  # اضافه کردن بهترین تطابق به لیست شناسایی‌شده‌ها
                        x, y, w, h = region_key
                        color = (255, 0, 0) if similarity == 1.0 else (0, 165, 255)
                        all_components.append({
                            'type': 'Tag',
                            'text': best_match,
                            'position': (x, y, w, h),
                            'color': color,
                            'similarity': similarity
                        })

        # مرتب‌سازی و نمایش components
        all_components.sort(key=lambda comp: comp['position'][1])
        sequence_number = 1
        
        for component in all_components:
            if component['type'] == 'Tag':
                x, y, w, h = component['position']
                color = component['color']
                text = component['text']
                if text not in tag_to_number:
                    tag_to_number[text] = sequence_number
                    sequence_number += 1
                assigned_number = tag_to_number[text]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f"#{assigned_number} {text}" if 'similarity' not in component else f"#{assigned_number} {text} ({component['similarity']:.2f})"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            elif component['type'] == 'Spare':
                x, y, w, h = component['position']
                color = component['color']
                text = component['text']
                spare_id = component['id']
                assigned_number = sequence_number
                sequence_number += 1
                tag_to_number[spare_id] = assigned_number
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f"#{assigned_number} {text}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # نمایش سایر components (JB, MC, Cable)
        for component in all_components:
            if component['type'] in ['JB', 'MC', 'Cable']:
                x, y, w, h = component['position']
                color = component['color']
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                label = f"{component['type']}: {component['text']}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # افزودن legend
        legend_y_pos = image.shape[0] - 60
        legend_x_pos = 10
        cv2.putText(image, "Tag", (legend_x_pos, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(image, "JB", (legend_x_pos + 100, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "MC", (legend_x_pos + 200, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image, "Cable", (legend_x_pos + 300, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        cv2.putText(image, "Spare", (legend_x_pos + 400, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
        cv2.putText(image, "Numbering:Tags are unique by name, {spare_examples}s have individual numbers", 
                    (legend_x_pos, legend_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # بررسی تطابق شماره تگ با شماره زوج
        is_consistent, max_tag_number, extracted_pair_number = self.check_tag_number_consistency(tag_to_number)
        
        # افزودن وضعیت تطابق به تصویر
        status_y_pos = image.shape[0] - 90
        status_x_pos = 10
        
        if is_consistent:
            status_message = f"Tag numbering OK: Max #{max_tag_number} matches pair number {extracted_pair_number}"
            status_color = (0, 255, 0)  # سبز برای موفقیت
        else:
            status_message = f"WARNING: Max tag #{max_tag_number}doesn't match pair number {extracted_pair_number} - CHECK NUMBERING!"
            status_color = (0, 0, 255)  # قرمز برای هشدار
        
        # افزودن پیام وضعیت با پس‌زمینه برای وضوح بیشتر
        text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (status_x_pos - 5, status_y_pos - 20), 
                    (status_x_pos + text_size[0] + 5, status_y_pos + 5), 
                    (255, 255, 255), -1)  # پس‌زمینه سفید
        cv2.putText(image, status_message, (status_x_pos, status_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return image, tag_to_number
    
    def add_tag_numbers_to_dataframe(self, df: pd.DataFrame, tag_to_number: Dict[str, int]) -> pd.DataFrame:
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


    def create_annotated_pdf(self, pdf_path: str, output_pdf_path: str) -> Dict[str, int]:
        """
        Create a new PDF with bounding boxes for tags and JBs using vector matching.
        Also returns a mapping of tags to their assigned numbers.

        Args:
            pdf_path: Input PDF file path
            output_pdf_path: Output PDF file path
            
        Returns:
            Dictionary mapping tags to their assigned numbers
        """
        # Master dictionary to store all tag-to-number mappings across all pages
        all_tag_numbers = {}
        
        try:
            # Open PDF document
            pdf_document = fitz.open(pdf_path)

            # Create a new PDF to store processed pages
            new_pdf = fitz.open()

            # Create temporary directory for storing page images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process each page
                for page_num in range(len(pdf_document)):
                    logger.info(f"Annotating page {page_num + 1}/{len(pdf_document)}")

                    # Get page
                    page = pdf_document[page_num]

                    # Convert page to image with higher resolution
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                    image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    pix.save(image_path)

                    # Load image
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.warning(f"Failed to load image for page {page_num + 1}")
                        continue

                    # Process image to extract text
                    processed_image = self.preprocess_image(image)

                    # Extract tags and JB identifiers with vector matching
                    # بررسی تعداد مقادیر برگشتی از تابع extract_from_image
                    result = self.extract_from_image(image)
                    
                    # اگر تابع 5 مقدار برمی‌گرداند
                    if len(result) == 5:
                        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers = result
                        tag_to_number = {}  # مقدار پیش‌فرض خالی
                    # اگر تابع 6 مقدار برمی‌گرداند
                    elif len(result) == 6:
                        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number = result
                    else:
                        # اگر تعداد مقادیر متفاوت است، خطا را ثبت کن و مقادیر پیش‌فرض را استفاده کن
                        logger.error(f"Unexpected number of values returned from extract_from_image: {len(result)}")
                        tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers = set(), set(), set(), [], []
                        tag_to_number = {}

                    # Draw bounding boxes on the image and get tag-to-number mapping
                    annotated_image, page_tag_numbers = self.draw_bounding_boxes(
                        image, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number
                    )
                    
                    # Update the master dictionary with this page's tag numbers
                    all_tag_numbers.update(page_tag_numbers)

                    # Add information overlay with stats
                    stats = self.get_processing_stats()
                    info_text = [
                        f"Page {page_num + 1}/{len(pdf_document)}",
                        f"Tags: {len(tags)}, JBs: {len(jb_identifiers)}, MCs: {len(mc_identifiers)}",
                        f"Vector match rate: {stats['match_rate']}"
                    ]

                    # Add overlay box
                    overlay = annotated_image.copy()
                    overlay_h = len(info_text) * 30 + 20
                    x, y, w, h = 10, 10, 390, overlay_h
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)

                    # Blend the overlay region only
                    blended_region = cv2.addWeighted(overlay[y:y+h, x:x+w], 0.7,
                                                    annotated_image[y:y+h, x:x+w], 0.3, 0)
                    annotated_image[y:y+h, x:x+w] = blended_region

                    # Add text info on top
                    for i, text in enumerate(info_text):
                        y_pos = y + 30 + i * 30
                        cv2.putText(annotated_image, text, (x + 10, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Save annotated image
                    annotated_image_path = os.path.join(temp_dir, f"annotated_page_{page_num + 1}.png")
                    cv2.imwrite(annotated_image_path, annotated_image)

                    # Add processed image to new PDF
                    new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                    new_page.insert_image(new_page.rect, filename=annotated_image_path)

                    # Clean up memory after processing each page
                    del pix, image, processed_image, annotated_image
                    gc.collect()

            # Save new PDF
            new_pdf.save(output_pdf_path)
            new_pdf.close()
            logger.info(f"Annotated PDF saved to: {output_pdf_path}")
            logger.info(f"Total tags numbered: {len(all_tag_numbers)}")
            
            return all_tag_numbers

        except Exception as e:
            logger.error(f"Error creating annotated PDF: {e}")
            return {}

    def _create_unmatched_tags_excel(self, unmatched_excel_tags: List[str], unmatched_pdf_tags: List[str], output_path: str) -> None:
        """
        ایجاد فایل اکسل دو ستونه برای نمایش تگ‌های تطبیق نیافته
        
        Args:
            unmatched_excel_tags: لیست تگ‌های یافت نشده در فایل intermediate (تگ‌های IO List که در PDF پیدا نشدند)
            unmatched_pdf_tags: لیست تگ‌های یافت نشده در IO List (تگ‌های PDF که در IO List پیدا نشدند)
            output_path: مسیر فایل اکسل خروجی
        """
        try:
            # تعیین تعداد ردیف‌های مورد نیاز
            max_rows = max(len(unmatched_excel_tags), len(unmatched_pdf_tags))
            
            # اگر هر دو لیست خالی هستند، یک ردیف خالی اضافه کنیم
            if max_rows == 0:
                max_rows = 1
            
            # ایجاد دیتافریم
            data = {
                'Tags in IO List not found in PDFs': pd.Series(unmatched_excel_tags + [''] * (max_rows - len(unmatched_excel_tags))),
                'Tags in PDFs not found in IO List': pd.Series(unmatched_pdf_tags + [''] * (max_rows - len(unmatched_pdf_tags)))
            }
            
            df = pd.DataFrame(data)
            
            # ذخیره به فایل اکسل
            df.to_excel(output_path, index=False)
            
            logger.info(f"Created unmatched tags Excel file with {len(unmatched_excel_tags)} IO List tags and {len(unmatched_pdf_tags)} PDF tags")
            
        except Exception as e:
            logger.error(f"Error creating unmatched tags Excel file: {e}")
            logger.error(traceback.format_exc())

    def run_with_annotated_pdf(self, pdf_paths: List[str], excel_path: str, output_excel_path: str, output_pdf_dir: str) -> Tuple[List[str], List[str]]:
        """
        Run complete process with vector-based matching and generate annotated PDFs.
        Also adds tag numbers to the output Excel file.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path
            output_excel_path: Output Excel file path
            output_pdf_dir: Directory path for storing processed PDFs
            io_list_path: Optional path to IO List Excel file
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # Build tag vectors from Excel first
        start_time = time.time()
        self.build_tag_vectors_from_excel(excel_path)
        logger.info(f"Using tag patterns with {len(self.tag_patterns)} patterns")
        
        # Create output PDF directory if it doesn't exist
        os.makedirs(output_pdf_dir, exist_ok=True)
        
        # Store all similarity reports for detailed analysis
        all_similarity_reports = []
        
        # Master dictionary to store tag-to-number mappings from all PDFs
        master_tag_numbers = {}
        
        # Dictionary to store all PDF processing results
        all_pdf_results = {}
        
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
            
            # Update master tag numbers dictionary
            master_tag_numbers.update(pdf_tag_numbers)
            
            # Collect similarity reports from this PDF
            all_similarity_reports.extend(self.similarity_reports)
            
            # Generate per-PDF statistics
            pdf_stats = self.get_processing_stats()
            logger.info(f"PDF {pdf_filename} statistics:")
            for key, value in pdf_stats.items():
                logger.info(f"  {key}: {value}")
        
        # Create intermediate Excel file with JB, MC, Tag/SPARE, SCR, Wire Colors, and Cable Description
        intermediate_excel_path = os.path.join(output_pdf_dir, "intermediate_data.xlsx")
        
        # Pass all_pdf_results to add_wire_colors_and_scr_to_dataframe
        self.add_wire_colors_and_scr_to_dataframe(
            pd.DataFrame(), 
            master_tag_numbers, 
            intermediate_excel_path, 
            all_pdf_results
        )
        
        # If IO List is provided, process and combine both Excel files
        if excel_path:
            final_df, unmatched_io_tags, unmatched_tags = self.process_excel_with_io_list(
                intermediate_excel_path, 
                excel_path, 
                output_excel_path
            )
            logger.info(f"Combined Excel file with IO List saved to: {output_excel_path}")
            
            # For function output
            unmatched_excel_tags = unmatched_io_tags
            unmatched_pdf_tags = unmatched_tags
            
            # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
            unmatched_excel_path = os.path.join(output_pdf_dir, "unmatched_tags.xlsx")
            self._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
            logger.info(f"Unmatched tags Excel file saved to: {unmatched_excel_path}")
        else:
            # If no IO List, just copy the intermediate file to the output path
            
            shutil.copy2(intermediate_excel_path, output_excel_path)
            logger.info(f"Excel file saved to: {output_excel_path}")
            
            # For function output
            unmatched_excel_tags = []
            unmatched_pdf_tags = []
        
        # Generate summary statistics
        self.processing_time = time.time() - start_time
        stats = self.get_processing_stats()
        
        # Save similarity reports and tag numbers as JSON
        reports_path = os.path.join(output_pdf_dir, "processing_reports.json")
        with open(reports_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'similarity_reports': all_similarity_reports,
                'tag_numbers': master_tag_numbers,
                'wire_colors': {tag: self.generate_mc_wire_colors(master_tag_numbers[tag]) for tag in master_tag_numbers},
                'scr_numbers': {tag: self.generate_scr_number(master_tag_numbers[tag]) for tag in master_tag_numbers}
            }, f, indent=2)
        
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
            logger.error(f"Error setting wire color rule: {e}")

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
            logger.error(f"Error setting SCR number rule: {e}")

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

    def add_wire_colors_and_scr_to_dataframe(self, df: pd.DataFrame, tag_to_number: Dict[str, int], 
                                    output_path: str, pdf_results: Dict[str, Dict[int, Tuple]],                       
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
                    
                    # پردازش هر صفحه از این PDF
                    for page_num, page_results in page_results_dict.items():
                        self._process_page_results(new_df_data, page_num, page_results, pdf_name, tag_to_number)
                
                # ایجاد دیتافریم جدید
                new_df = pd.DataFrame(new_df_data)
                
                # اگر دیتافریم خالی نیست، مرتب‌سازی انجام بده
                if not new_df.empty:
                    # مرتب‌سازی بر اساس نام PDF، صفحه، JB و شماره تگ
                    new_df = new_df.sort_values(by=['PDF_Name', 'Page', 'JB', 'Tag_Number'])
                    
                    # تنظیم ترتیب ستون‌های نهایی
                    column_order = [
                        'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                        'BK_Colors', 'WT_Colors', 'Terminal_First', 'Terminal_Second', 'Terminal_Label',
                        'Cable_Description', 'Type', 'Tag_Number_Status'  # اضافه کردن ستون وضعیت
                    ]
                    
                    # فقط ستون‌هایی که وجود دارند را انتخاب کن
                    available_columns = [col for col in column_order if col in new_df.columns]
                    new_df = new_df[available_columns]
                
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
                
                return df  # دیتافریم اصلی را برمی‌گردانیم
                
            except Exception as e:
                logger.error(f"Error in add_wire_colors_and_scr_to_dataframe: {e}")
                
                logger.error(traceback.format_exc())
                raise

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

    def _process_page_results(self, new_df_data: List[Dict], page_num: int, page_results: Any, 
                pdf_name: str, tag_to_number: Dict[str, int]):
        """
        پردازش نتایج یک صفحه و افزودن آن‌ها به لیست داده‌های دیتافریم
        با استفاده از اطلاعات دقیق استخراج شده توسط draw_bounding_boxes
        
        Args:
            new_df_data: لیست دیکشنری‌های داده برای دیتافریم
            page_num: شماره صفحه
            page_results: نتایج پردازش صفحه
            pdf_name: نام فایل PDF
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها
        """
        try:
            logger.info(f"Processing page {page_num} of {pdf_name}")
            
            # استخراج اطلاعات از ساختار page_results
            page_tags = set()
            page_jbs = set()
            page_mcs = set()
            page_cable_descriptions = []
            page_spares = []
            page_tag_to_number = {}
            
            # استخراج داده‌ها از page_results با توجه به ساختار آن
            if isinstance(page_results, tuple) and len(page_results) >= 5:
                page_tags = page_results[0]
                page_jbs = page_results[1]
                page_mcs = page_results[2]
                page_cable_descriptions = page_results[3]
                page_spares = page_results[4]
                if len(page_results) >= 6:
                    page_tag_to_number = page_results[5]
            elif isinstance(page_results, dict):
                page_tags = page_results.get('tags', set())
                page_jbs = page_results.get('jbs', set())
                page_mcs = page_results.get('mcs', set())
                page_cable_descriptions = page_results.get('cable_descriptions', [])
                page_spares = page_results.get('spares', [])
                page_tag_to_number = page_results.get('tag_to_number', {})
            else:
                logger.warning(f"Unexpected type for page_results in PDF {pdf_name}, page {page_num}: {type(page_results)}")
                return
            
            # تبدیل به set اگر لیست هستند
            if not isinstance(page_tags, set):
                page_tags = set(page_tags) if hasattr(page_tags, '__iter__') else set()
            if not isinstance(page_jbs, set):
                page_jbs = set(page_jbs) if hasattr(page_jbs, '__iter__') else set()
            if not isinstance(page_mcs, set):
                page_mcs = set(page_mcs) if hasattr(page_mcs, '__iter__') else set()
            
            # تعیین JB و MC اصلی برای این صفحه
            main_jb = list(page_jbs)[0] if page_jbs else ''
            main_mc = list(page_mcs)[0] if page_mcs else ''
            main_cable_desc = page_cable_descriptions[0] if page_cable_descriptions else ''
            
            logger.info(f"PDF: {pdf_name}, Page {page_num}: JB={main_jb}, MC={main_mc}, Tags={len(page_tags)}, Spares={len(page_spares)}")
            logger.info(f"Tag numbers directly from bounding box: {page_tag_to_number}")
            
            # استخراج عدد پشت "Pair" از Cable_Description
            pair_number = self.extract_pair_number(main_cable_desc)
            logger.info(f"Extracted pair number from cable description: {pair_number}")
            
            # یافتن بزرگترین شماره تگ برای این JB
            max_tag_number = 0
            if page_tag_to_number:
                max_tag_number = max(page_tag_to_number.values())
            
            # مقایسه شماره زوج با بزرگترین شماره تگ
            tag_number_status = "OK"
            if pair_number is not None and max_tag_number > 0:
                if pair_number != max_tag_number:
                    tag_number_status = f"WARNING: Pair number ({pair_number}) != Max Tag number ({max_tag_number})"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
                else:
                    logger.info(f"Page {page_num}, JB {main_jb}: Pair number ({pair_number}) matches max tag number ({max_tag_number})")
            else:
                if pair_number is None:
                    tag_number_status = "WARNING: Could not extract pair number from cable description"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
                elif max_tag_number == 0:
                    tag_number_status = "WARNING: No tag numbers found"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
            
            # پردازش تگ‌های این صفحه - فقط با استفاده از اطلاعات bounding box
            for tag in page_tags:
                try:
                    # استفاده مستقیم از شماره تگ استخراج شده توسط bounding box
                    if tag in page_tag_to_number:
                        tag_num = page_tag_to_number[tag]
                        
                        # استفاده از JB و MC همین صفحه
                        jb = main_jb
                        mc = main_mc
                        cable_desc = main_cable_desc
                        
                        # تولید رنگ‌های سیم و شماره‌های SCR بر اساس شماره تگ
                        tag_num_str = f"{tag_num:02d}"
                        bk_color = f"BK{tag_num_str}"
                        wt_color = f"WT{tag_num_str}"
                        
                        # تولید شماره SCR بر اساس شماره تگ
                        first_scr_num = (tag_num * 2) - 1
                        second_scr_num = tag_num * 2
                        
                        # اضافه کردن به لیست داده‌ها
                        new_df_data.append({
                            'PDF_Name': pdf_name,
                            'Page': page_num,
                            'JB': jb,
                            'MC': mc,
                            'Tag/SPARE': tag,
                            'Tag_Number': tag_num,
                            'BK_Colors': bk_color,
                            'WT_Colors': wt_color,
                            'Terminal_First': str(first_scr_num),
                            'Terminal_Second': str(second_scr_num),
                            'Terminal_Label': 'SCR',
                            'Cable_Description': cable_desc,
                            'Type': 'Tag',
                            'Tag_Number_Status': tag_number_status  # ستون برای مقایسه شماره زوج و تگ
                        })
                        
                        logger.info(f"Added Tag from {pdf_name}: {tag} -> JB: {jb}, MC: {mc}, Tag#: {tag_num}")
                    else:
                        logger.warning(f"Tag {tag} not found in bounding box tag numbers, skipping")
                        
                except Exception as e:
                    logger.error(f"Error processing tag {tag} in PDF {pdf_name}, page {page_num}: {e}")
                    
                    logger.error(traceback.format_exc())
            
            # پردازش اسپیرهای این صفحه - فقط با استفاده از اطلاعات bounding box
            for i, spare in enumerate(page_spares):
                try:
                    # ایجاد شناسه‌های مختلف برای اسپیر برای جستجو در دیکشنری
                    spare_id_options = [
                        f"SPARE_{i+1}",
                        spare,
                        f"SPARE_{page_num}_{i+1}"
                    ]
                    
                    # جستجو برای شناسه اسپیر در page_tag_to_number
                    spare_number = None
                    spare_id_used = None
                    
                    for spare_id in spare_id_options:
                        if spare_id in page_tag_to_number:
                            spare_number = page_tag_to_number[spare_id]
                            spare_id_used = spare_id
                            break
                    
                    # اگر شناسه اسپیر در page_tag_to_number پیدا نشد، از این اسپیر صرف نظر می‌کنیم
                    if spare_number is None:
                        logger.warning(f"Spare {spare} not found in bounding box tag numbers, skipping")
                        continue
                    
                    # استفاده از JB و MC همین صفحه
                    jb = main_jb
                    mc = main_mc
                    cable_desc = main_cable_desc
                    
                    # تولید رنگ‌های سیم و شماره‌های SCR بر اساس شماره اسپیر
                    spare_num_str = f"{spare_number:02d}"
                    bk_color = f"BK{spare_num_str}"
                    wt_color = f"WT{spare_num_str}"
                    
                    # تولید شماره SCR بر اساس شماره اسپیر
                    first_scr_num = (spare_number * 2) - 1
                    second_scr_num = spare_number * 2
                    
                    # اضافه کردن به لیست داده‌ها
                    new_df_data.append({
                        'PDF_Name': pdf_name,
                        'Page': page_num,
                        'JB': jb,
                        'MC': mc,
                        'Tag/SPARE': spare,
                        'Tag_Number': spare_number,
                        'BK_Colors': bk_color,
                        'WT_Colors': wt_color,
                        'Terminal_First': str(first_scr_num),
                        'Terminal_Second': str(second_scr_num),
                        'Terminal_Label': 'SCR',
                        'Cable_Description': cable_desc,
                        'Type': 'SPARE',
                        'Tag_Number_Status': tag_number_status  # ستون برای مقایسه شماره زوج و تگ
                    })
                    
                    logger.info(f"Added Spare from {pdf_name}: {spare} (ID: {spare_id_used}) -> JB: {jb}, MC: {mc}, Tag#: {spare_number}")
                    
                except Exception as e:
                    logger.error(f"Error processing spare {spare} in PDF {pdf_name}, page {page_num}: {e}")
                    
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_name}, page {page_num}: {e}")
            
            logger.error(traceback.format_exc())

    def _get_unique_wire_colors(self, tag: str, wire_colors: Dict[str, List[str]], 
                            used_wire_colors: Dict[str, Dict[str, bool]], 
                            tag_to_number: Dict[str, int], as_list: bool = False) -> Union[str, List[str]]:
        """
        برای هر تگ، رنگ‌های سیم منحصر به فرد را برمی‌گرداند و از تکرار جلوگیری می‌کند.
        
        Args:
            tag: نام تگ
            wire_colors: دیکشنری رنگ‌های سیم
            used_wire_colors: دیکشنری رنگ‌های استفاده شده
            tag_to_number: دیکشنری شماره تگ‌ها
            as_list: اگر True باشد، لیست رنگ‌ها را برمی‌گرداند، در غیر این صورت رشته
        
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
            
    def check_tag_number_consistency(self, tag_to_number: Dict[str, int]) -> Tuple[bool, int, int]:
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
            logger.debug(f"Tag to number dictionary: {tag_to_number}")
            
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
                logger.warning("No tag numbers found in tag_to_number dictionary")
            
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
        
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Return detailed statistics about the processing results.
        """
        return {
            'total_tags': len(self.all_tags),
            'matched_tags': len(self.matched_tags),
            'exact_matches': self.exact_matches,
            'similar_matches': self.similar_matches,
            'total_jbs': len(self.all_jbs),
            'processing_time': f"{self.processing_time:.2f} seconds",
            'match_rate': f"{(len(self.matched_tags) / len(self.all_tags) * 100):.1f}%" if self.all_tags else "0%",
            'exact_match_rate': f"{(self.exact_matches / len(self.matched_tags) * 100):.1f}%" if self.matched_tags else "0%",
            'unmatched_tags': len(self.all_tags - self.matched_tags),
        }
    
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