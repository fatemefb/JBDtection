import cv2
import pytesseract
import numpy as np
from PIL import Image
import pandas as pd
import re
import os
import gc
import fitz  # PyMuPDF for PDF processing
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import Levenshtein

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تعریف کلاس TagJBExtractor
class TagJBExtractor:
    """
    کلاسی برای استخراج تگ‌ها و شناسه‌های JB از نمودارهای PDF و تطبیق آن‌ها با داده‌های اکسل.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, excel_path: Optional[str] = None):
        """
        Initialize the extractor with optional tesseract and excel paths.
        
        Args:
            tesseract_path: Path to tesseract executable (if needed)
            excel_path: Path to Excel file for dynamic tag pattern building (if needed)
        """
        # Set default Tesseract path if none provided
        if tesseract_path:
            if not os.path.exists(tesseract_path):
                raise ValueError(f"Provided Tesseract path does not exist: {tesseract_path}")
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
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
                raise RuntimeError("Tesseract not found in common locations. Please provide tesseract_path.")
        
        # Test Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tesseract: {e}")
        
        # Initialize other attributes
        self.jb_pattern = re.compile(r'JB-\d+', re.IGNORECASE)
        
        if excel_path:
            self.build_tag_pattern_from_excel(excel_path)
        else:
           self.tag_pattern = r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b'
            
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر برای بهبود دقت OCR.
        
        Args:
            image: تصویر ورودی به صورت آرایه numpy
            
        Returns:
            تصویر پردازش‌شده به صورت آرایه numpy
        """
        # تبدیل به سیاه و سفید اگر قبلاً نشده است
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # افزایش وضوح
        scale_factor = 2
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # کاهش نویز و افزایش کنتراست
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        
        # عملیات مورفولوژیک برای بستن فاصله‌ها
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        return gray
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein distance.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        distance = Levenshtein.distance(text1.upper(), text2.upper())
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0
        return 1 - (distance / max_len)
        
    def extract_from_image(self, image: np.ndarray) -> Tuple[Set[str], Set[str]]:
        """
        Extract tags and JB identifiers from an image using two-stage detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (tags, jb_identifiers) as sets
        """
        processed_image = self.preprocess_image(image)
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. -c preserve_interword_spaces=1'
        
        tags = set()
        jb_identifiers = set()
        
        # Stage 1: Exact matching
        full_text = pytesseract.image_to_string(processed_image, config=custom_config)
        
        # Process text for exact matches
        for line in full_text.split("\n"):
            words = line.split()
            for word in words:
                word_clean = word.upper().strip()
                
                # Check for JB identifiers
                if word_clean.startswith("JB-"):
                    jb_identifiers.add(word_clean)
                    
                # Check for tags with exact pattern match
                if re.search(self.tag_pattern, word_clean, re.IGNORECASE):
                    tags.add(word_clean)
        
        # Stage 2: Similarity-based matching
        ocr_data = pytesseract.image_to_data(processed_image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Create a list of potential tag candidates from OCR
        candidates = []
        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                candidates.append({
                    'text': text.strip().upper(),
                    'conf': float(ocr_data['conf'][i])
                })
        
        # Find similar matches for potential tags
        similarity_threshold = 0.8
        stage2_tags = set()
        
        for candidate in candidates:
            candidate_text = candidate['text']
            
            # Skip if already detected as exact match
            if candidate_text in tags:
                continue
            
            # Check for partial matches using tag pattern components
            tag_components = re.findall(r'[A-Z]+|\d+', candidate_text)
            if len(tag_components) >= 2:  # At least prefix and number
                potential_tag = ''.join(tag_components)
                
                # Check similarity with common tag patterns
                for pattern in [
                    r'[A-Z]{2,4}\d{3}-\d{2,3}',
                    r'[A-Z]{2,4}-\d{3}-\d{2,3}',
                    r'[A-Z]{2,4}\d{3}\d{2,3}'
                ]:
                    if re.match(pattern, potential_tag):
                        stage2_tags.add(potential_tag)
                        break
        
        # Combine both stages' tags
        all_tags = tags.union(stage2_tags)
        
        return all_tags, jb_identifiers

    
    def process_pdf(self, pdf_path: str) -> Dict[int, Tuple[Set[str], Set[str]]]:
        """
        پردازش تمام صفحات در یک فایل PDF.
        
        Args:
            pdf_path: مسیر فایل PDF
            
        Returns:
            دیکشنری نگاشت شماره صفحات به تاپل‌های (tags, jb_identifiers)
        """
        results = {}
        
        # باز کردن فایل PDF
        logger.info(f"Opening PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        
        # ایجاد یک دایرکتوری موقت برای ذخیره تصاویر صفحات
        with tempfile.TemporaryDirectory() as temp_dir:
            # پردازش هر صفحه
            for page_num in range(len(pdf_document)):
                logger.info(f"Processing page {page_num + 1}/{len(pdf_document)}")
                
                # دریافت صفحه
                page = pdf_document[page_num]
                
                # تبدیل صفحه به تصویر با وضوح بالاتر
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                
                # بارگذاری تصویر
                image = cv2.imread(image_path)
                
                # استخراج تگ‌ها و شناسه‌های JB
                tags, jb_identifiers = self.extract_from_image(image)
                
                # ذخیره نتایج
                results[page_num + 1] = (tags, jb_identifiers)
                
                logger.info(f"Page {page_num + 1}: Found {len(tags)} tags and {len(jb_identifiers)} JB identifiers")
        
        return results
    
    def extract_tag_prefix(self, tag: str) -> str:
        """
        استخراج بخش پیشوند یک تگ (مثلاً FV، FT، TT).
        
        Args:
            tag: رشته تگ
            
        Returns:
            بخش پیشوند تگ
        """
        # مدیریت فرمت‌های مختلف تگ
        # فرمت با خط تیره: "FT-1234" -> "FT"
        hyphen_match = re.match(r'([A-Z]+)-', tag)
        if hyphen_match:
            return hyphen_match.group(1)
            
        # فرمت بدون جداکننده: "FT1234" -> "FT"
        alpha_match = re.match(r'([A-Z]+)\d', tag)
        if alpha_match:
            return alpha_match.group(1)
            
        # فقط کاراکترهای الفبایی در ابتدا
        match = re.match(r'([A-Z]+)', tag)
        if match:
            return match.group(1)
            
        return ""
        
    def build_tag_pattern_from_excel(self, excel_path: str) -> None:
        """
        ساخت الگوی تگ به صورت پویا از ستون Tag NO در فایل اکسل.
        
        Args:
            excel_path: مسیر فایل اکسل
        """
        try:
            # خواندن فایل اکسل
            df = pd.read_excel(excel_path)
            
            # اطمینان از وجود ستون 'Tag NO'
            if 'Tag No' not in df.columns:
                logger.warning("'Tag No' column not found in Excel. Using default tag pattern.")
                self.tag_pattern = r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b'
                return
                
            # استخراج پیشوندها از ستون Tag NO
            prefixes = set()
            for tag in df['Tag No'].dropna():
                prefix = self.extract_tag_prefix(str(tag).strip().upper())
                if prefix:
                    prefixes.add(prefix)
                    
            if not prefixes:
                logger.warning("No valid prefixes found in Tag NO column. Using default pattern.")
                self.tag_pattern = r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b'
                return
                
            # ایجاد یک الگوی regex جدید با استفاده از پیشوندهای جمع‌آوری شده
            prefix_pattern = '|'.join(prefixes)
            # Use double curly braces for literal curly braces in format string
            self.tag_pattern = r'\b(?:{})[-.]?\d{{3}}-\d{{2,3}}[A-Z]?\d*\b'.format(prefix_pattern)
            
            logger.info(f"Built dynamic tag pattern from {len(prefixes)} prefixes: {prefix_pattern}")
            logger.info(f"Final tag pattern: {self.tag_pattern}")
            
        except Exception as e:
            logger.error(f"Error building tag pattern from Excel: {e}")
            # بازگشت به الگوی پیش‌فرض
            self.tag_pattern = r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}-\d{2,3}[A-Z]?\d*\b'

                
    def create_tag_jb_mapping(self, pdf_results: Dict[int, Tuple[Set[str], Set[str]]]) -> Dict[str, str]:
        """
        ایجاد نگاشت از تگ‌ها به شناسه‌های JB با محدودیت صفحه.
        
        Args:
            pdf_results: نتایج از متد process_pdf
            
        Returns:
            دیکشنری نگاشت تگ‌ها به شناسه‌های JB
        """
        tag_to_jb = {}
        tag_to_page = {}  # برای ردیابی صفحه هر تگ
        similarity_threshold = 0.8

        # مرحله اول: تطبیق دقیق با محدودیت صفحه
        for page_num, (tags, jb_identifiers) in pdf_results.items():
            if not tags or not jb_identifiers:
                continue
                
            # ذخیره صفحه هر تگ
            for tag in tags:
                if tag not in tag_to_page:
                    tag_to_page[tag] = page_num
                
            # اگر فقط یک JB در صفحه وجود دارد
            if len(jb_identifiers) == 1:
                jb = next(iter(jb_identifiers))
                for tag in tags:
                    # فقط اگر تگ قبلاً به JB دیگری متصل نشده باشد
                    if tag not in tag_to_jb:
                        tag_to_jb[tag] = jb

            # اگر دو JB وجود دارد
            elif len(jb_identifiers) == 2:
                # انتخاب JB با فرمت صحیح
                jb_candidates = list(jb_identifiers)
                jb = next((jb for jb in jb_candidates if self.jb_pattern.match(jb)), jb_candidates[0])
                for tag in tags:
                    if tag not in tag_to_jb:
                        tag_to_jb[tag] = jb

        # مرحله دوم: تطبیق بر اساس شباهت با محدودیت صفحه
        unmatched_tags = set()
        for page_num, (tags, jb_identifiers) in pdf_results.items():
            for tag in tags:
                if tag not in tag_to_jb:
                    unmatched_tags.add(tag)

        if unmatched_tags:
            for tag in unmatched_tags.copy():
                tag_page = tag_to_page.get(tag)
                if tag_page:
                    # فقط با تگ‌های همان صفحه مقایسه می‌کنیم
                    page_tags, page_jbs = pdf_results[tag_page]
                    matched_tags = [t for t in page_tags if t in tag_to_jb]
                    
                    if matched_tags:
                        # بررسی شباهت با تگ‌های تطبیق‌یافته در همان صفحه
                        for matched_tag in matched_tags:
                            if self._calculate_similarity(tag, matched_tag) > similarity_threshold:
                                tag_to_jb[tag] = tag_to_jb[matched_tag]
                                unmatched_tags.remove(tag)
                                break
                    elif len(page_jbs) == 1:
                        # اگر تگ مشابهی نیست ولی فقط یک JB در صفحه هست
                        tag_to_jb[tag] = next(iter(page_jbs))
                        unmatched_tags.remove(tag)

        # گزارش نتایج
        logger.info(f"Total tags mapped: {len(tag_to_jb)}")
        logger.info(f"Tags mapped in first stage: {len(tag_to_jb) - len(unmatched_tags)}")
        logger.info(f"Tags mapped in second stage: {len(unmatched_tags)}")
        
        return tag_to_jb
            
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
                r'--oem 3 --psm 6',  # Assume uniform block of text
                r'--oem 3 --psm 11',  # Sparse text with OSD
                r'--oem 3 --psm 12'   # Sparse text without OSD
            ]
            
            for config in ocr_configs:
                column_text = pytesseract.image_to_string(
                    column_image, 
                    config=config + r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456-'
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


    def process_pdf_page(self, page_info: Tuple[fitz.Page, str, int]) -> Tuple[int, Set[str], Set[str]]:
        """
        Process a single PDF page in parallel.
        
        Args:
            page_info: Tuple containing (page object, temp_dir path, page number)
            
        Returns:
            Tuple of (page_number, tags, jb_identifiers)
        """
        page, temp_dir, page_num = page_info
        
        # Create image path
        image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
        
        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        pix.save(image_path)
        
        # Load and process image
        image = cv2.imread(image_path)
        tags, jb_identifiers = self.extract_from_image(image)
        
        # Clean up temporary image file
        try:
            os.remove(image_path)
        except:
            pass
            
        return page_num + 1, tags, jb_identifiers

    def process_pdf(self, pdf_path: str) -> Dict[int, Tuple[Set[str], Set[str]]]:
        """
        Process all pages in a PDF file using multiprocessing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to tuples of (tags, jb_identifiers)
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
                    tags, jb_identifiers = self.extract_from_image(image)
                    
                    # Store results
                    results[page_num + 1] = (tags, jb_identifiers)
                    
                    # Print results immediately for this page
                    print(f"Page {page_num + 1}:")
                    print(f"  Tags found ({len(tags)}): {', '.join(sorted(tags))}")
                    print(f"  JB identifiers found ({len(jb_identifiers)}): {', '.join(sorted(jb_identifiers))}")
                    
                    # Clean up temporary image file
                    try:
                        os.remove(image_path)
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    continue
        
        return results

    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[int, Tuple[Set[str], Set[str]]]:
        """
        Process multiple PDF files in parallel with improved resource management.
        """
        combined_results = {}
        page_offset = 0
        
        # Calculate optimal number of processes
        num_processes = min(cpu_count(), len(pdf_paths), 4)  # محدود کردن به حداکثر 4 پروسس
        logger.info(f"Processing {len(pdf_paths)} PDF files using {num_processes} processes")
        
        try:
            # Use context manager for better resource management
            with Pool(processes=num_processes) as pool:
                # Use imap instead of map for better memory management with large files
                for pdf_result in pool.imap_unordered(self.process_pdf, pdf_paths):
                    if pdf_result:
                        for page_num, (tags, jb_identifiers) in pdf_result.items():
                            combined_results[page_num + page_offset] = (tags, jb_identifiers)
                        page_offset += max(pdf_result.keys()) if pdf_result else 0
                        
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            logger.info("Falling back to sequential processing")
            for pdf_path in pdf_paths:
                try:
                    pdf_result = self.process_pdf(pdf_path)
                    for page_num, (tags, jb_identifiers) in pdf_result.items():
                        combined_results[page_num + page_offset] = (tags, jb_identifiers)
                    page_offset += max(pdf_result.keys()) if pdf_result else 0
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {e}")
                    continue
        
        return combined_results
    def process_excel_chunk(self, chunk_data: Tuple[pd.DataFrame, Dict[str, str]]) -> Tuple[pd.DataFrame, List[str], Set[str]]:
        """
        Process a chunk of Excel data with two-stage tag matching.
        
        Args:
            chunk_data: Tuple containing (DataFrame chunk, tag_to_jb mapping)
            
        Returns:
            Tuple of (processed DataFrame chunk, unmatched excel tags, unmatched pdf tags)
        """
        chunk, tag_to_jb = chunk_data
        unmatched_excel = []
        unmatched_pdf = set()
        similarity_threshold = 0.8
        
        # Process each row in the chunk
        for idx, row in chunk.iterrows():
            tag = str(row['Tag No']).strip().upper()
            
            # Skip empty tags
            if pd.isna(tag) or not tag:
                continue
            
            matched = False
            
            # Try exact match first
            if tag in tag_to_jb:
                chunk.at[idx, 'JB'] = tag_to_jb[tag]
                chunk.at[idx, 'Match_Type'] = 'Exact'
                matched = True
            else:
                # Try similarity matching with all PDF tags
                best_match = None
                highest_similarity = 0
                
                for pdf_tag in tag_to_jb.keys():
                    similarity = self._calculate_similarity(tag, pdf_tag)
                    if similarity > similarity_threshold and similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = pdf_tag
                
                if best_match:
                    chunk.at[idx, 'JB'] = tag_to_jb[best_match]
                    chunk.at[idx, 'Match_Type'] = f'Similar ({highest_similarity:.2f})'
                    matched = True
            
            if not matched:
                unmatched_excel.append(tag)
        
        # Track unmatched PDF tags
        for pdf_tag in tag_to_jb.keys():
            if not any(self._calculate_similarity(pdf_tag, str(excel_tag).strip().upper()) > similarity_threshold 
                    for excel_tag in chunk['Tag No'].dropna()):
                unmatched_pdf.add(pdf_tag)
        
        return chunk, unmatched_excel, unmatched_pdf

    def process_excel(self, excel_path: str, tag_to_jb: Dict[str, str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Process Excel file with improved parallel processing and two-stage matching.
        """
        logger.info(f"Processing Excel file: {excel_path}")
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        if 'Tag No' not in df.columns:
            raise ValueError("Excel file must contain a 'Tag No' column")
        
        # Add new columns
        df['JB'] = None
        df['Match_Type'] = None
        df['Similarity_Score'] = None  # New column for similarity scores
        
        # Calculate optimal chunk size and number of processes
        num_processes = min(cpu_count(), 4)
        chunk_size = max(100, len(df) // (num_processes * 2))
        
        # Split DataFrame into chunks
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        chunk_data = [(chunk, tag_to_jb) for chunk in chunks]
        
        all_results = []
        try:
            with Pool(processes=num_processes) as pool:
                for result in pool.imap(self.process_excel_chunk, chunk_data):
                    all_results.append(result)
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            all_results = [self.process_excel_chunk(data) for data in chunk_data]
        
        # Combine results
        processed_chunks = []
        all_unmatched_excel = []
        all_unmatched_pdf = set()
        
        for processed_chunk, unmatched_excel, unmatched_pdf in all_results:
            processed_chunks.append(processed_chunk)
            all_unmatched_excel.extend(unmatched_excel)
            all_unmatched_pdf.update(unmatched_pdf)
        
        # Combine processed chunks back into single DataFrame
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Extract similarity scores from Match_Type column where available
        final_df['Similarity_Score'] = final_df['Match_Type'].apply(
            lambda x: float(re.search(r'\(([0-9.]+)\)', x).group(1))
            if isinstance(x, str) and 'Similar' in x
            else None
        )
        
        # Sort by similarity score (exact matches first, then by descending similarity)
        final_df['Sort_Score'] = final_df.apply(
            lambda row: 1.0 if row['Match_Type'] == 'Exact'
            else row['Similarity_Score'] if pd.notnull(row['Similarity_Score'])
            else 0.0,
            axis=1
        )
        final_df = final_df.sort_values('Sort_Score', ascending=False)
        
        # Remove temporary sorting column
        final_df = final_df.drop('Sort_Score', axis=1)
        
        # Log results
        exact_matches = len(final_df[final_df['Match_Type'] == 'Exact'])
        similar_matches = len(final_df[final_df['Match_Type'].str.startswith('Similar', na=False)])
        
        logger.info(f"Exact matches: {exact_matches}")
        logger.info(f"Similar matches: {similar_matches}")
        logger.info(f"Total matches: {exact_matches + similar_matches}")
        logger.info(f"Unmatched Excel tags: {len(all_unmatched_excel)}")
        logger.info(f"Unmatched PDF tags: {len(all_unmatched_pdf)}")
        
        # Add summary to the log
        logger.info("\nMatching Summary:")
        logger.info("-" * 50)
        logger.info(f"Total rows processed: {len(final_df)}")
        logger.info(f"Exact matches: {exact_matches} ({exact_matches/len(final_df)*100:.1f}%)")
        logger.info(f"Similar matches: {similar_matches} ({similar_matches/len(final_df)*100:.1f}%)")
        logger.info(f"Unmatched: {len(all_unmatched_excel)} ({len(all_unmatched_excel)/len(final_df)*100:.1f}%)")
        
        # For similar matches, show distribution of similarity scores
        if similar_matches > 0:
            similarity_scores = final_df[pd.notnull(final_df['Similarity_Score'])]['Similarity_Score']
            logger.info("\nSimilarity Score Distribution:")
            logger.info(f"Min: {similarity_scores.min():.3f}")
            logger.info(f"Max: {similarity_scores.max():.3f}")
            logger.info(f"Mean: {similarity_scores.mean():.3f}")
            logger.info(f"Median: {similarity_scores.median():.3f}")
        
        return final_df, all_unmatched_excel, list(all_unmatched_pdf)

    def run(self, pdf_paths: List[str], excel_path: str, output_excel_path: str) -> Tuple[List[str], List[str]]:
        """
        Run the complete process with parallel processing support.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path
            output_excel_path: Output Excel file path
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # Build tag pattern from Excel first
        self.build_tag_pattern_from_excel(excel_path)
        logger.info(f"Using tag pattern: {self.tag_pattern}")
        
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
        
        # Process Excel in parallel
        updated_df, unmatched_excel_tags, unmatched_pdf_tags = self.process_excel(excel_path, tag_to_jb)
        
        # Save updated Excel
        updated_df.to_excel(output_excel_path, index=False)
        logger.info(f"Updated Excel saved to: {output_excel_path}")
        
        return unmatched_excel_tags, unmatched_pdf_tags

    # Update the draw_bounding_boxes method to show both detection stages
    def draw_bounding_boxes(self, image: np.ndarray, tags: Set[str], jb_identifiers: Set[str]) -> np.ndarray:
        """
        Draw bounding boxes for tags and JB identifiers with different colors for detection stages.
        
        Args:
            image: Input image
            tags: Set of detected tags
            jb_identifiers: Set of detected JB identifiers
            
        Returns:
            Annotated image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        custom_config = r'--oem 3 --psm 11'
        ocr_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # First stage detections (exact matches) - Green
        for tag in tags:
            for i, text in enumerate(ocr_data['text']):
                if text.strip().upper() == tag:
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                                ocr_data['width'][i], ocr_data['height'][i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f"Stage 1: {tag}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Second stage detections (similarity matches) - Orange
        similarity_threshold = 0.8
        for i, text in enumerate(ocr_data['text']):
            text_clean = text.strip().upper()
            if text_clean and any(self._calculate_similarity(text_clean, tag) > similarity_threshold for tag in tags):
                if not any(text_clean == tag for tag in tags):  # Not an exact match
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                                ocr_data['width'][i], ocr_data['height'][i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(image, f"Stage 2: {text_clean}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # JB identifiers - Blue
        for jb in jb_identifiers:
            for i, text in enumerate(ocr_data['text']):
                if text.strip().upper() == jb:
                    x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], 
                                ocr_data['width'][i], ocr_data['height'][i])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, jb, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return image

    def create_annotated_pdf(self, pdf_path: str, output_pdf_path: str) -> None:
        """
        ایجاد PDF جدید با bounding box‌های تگ‌ها و JB‌ها.
        
        Args:
            pdf_path: مسیر فایل PDF ورودی
            output_pdf_path: مسیر فایل PDF خروجی
        """
        # باز کردن فایل PDF
        pdf_document = fitz.open(pdf_path)
        
        # ایجاد یک PDF جدید برای ذخیره صفحات پردازش‌شده
        new_pdf = fitz.open()
        
        # ایجاد یک دایرکتوری موقت برای ذخیره تصاویر صفحات
        with tempfile.TemporaryDirectory() as temp_dir:
            # پردازش هر صفحه
            for page_num in range(len(pdf_document)):
                logger.info(f"Annotating page {page_num + 1}/{len(pdf_document)}")
                
                # دریافت صفحه
                page = pdf_document[page_num]
                
                # تبدیل صفحه به تصویر با وضوح بالاتر
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                
                # بارگذاری تصویر
                image = cv2.imread(image_path)
                
                # استخراج تگ‌ها و شناسه‌های JB
                tags, jb_identifiers = self.extract_from_image(image)
                
                # رسم bounding box‌ها روی تصویر
                annotated_image = self.draw_bounding_boxes(image, tags, jb_identifiers)
                
                # ذخیره تصویر پردازش‌شده
                annotated_image_path = os.path.join(temp_dir, f"annotated_page_{page_num + 1}.png")
                cv2.imwrite(annotated_image_path, annotated_image)
                
                # اضافه کردن تصویر پردازش‌شده به PDF جدید
                new_page = new_pdf.new_page(width=pix.width, height=pix.height)
                new_page.insert_image(new_page.rect, filename=annotated_image_path)

                # پاکسازی حافظه پس از پردازش هر صفحه
                gc.collect()
        
        # ذخیره PDF جدید
        new_pdf.save(output_pdf_path)
        new_pdf.close()
        logger.info(f"Annotated PDF saved to: {output_pdf_path}")

    def run_with_annotated_pdf(self, pdf_paths: List[str], excel_path: str, output_excel_path: str, output_pdf_dir: str) -> Tuple[List[str], List[str]]:
        """
        اجرای کامل فرآیند با خروجی PDF‌های حاوی bounding box.
        
        Args:
            pdf_paths: لیست مسیرهای فایل‌های PDF
            excel_path: مسیر فایل اکسل ورودی
            output_excel_path: مسیر فایل اکسل خروجی
            output_pdf_dir: مسیر دایرکتوری برای ذخیره PDF‌های پردازش‌شده
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # اجرای فرآیند اصلی
        unmatched_excel_tags, unmatched_pdf_tags = self.run(pdf_paths, excel_path, output_excel_path)
        
        # ایجاد PDF‌های پردازش‌شده
        os.makedirs(output_pdf_dir, exist_ok=True)
        for pdf_path in pdf_paths:
            pdf_filename = os.path.basename(pdf_path)
            output_pdf_path = os.path.join(output_pdf_dir, f"annotated_{pdf_filename}")
            self.create_annotated_pdf(pdf_path, output_pdf_path)
        
        return unmatched_excel_tags, unmatched_pdf_tags