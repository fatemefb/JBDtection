import cv2
import pytesseract
import numpy as np
from PIL import Image
import pandas as pd
import re
import os
import fitz  # PyMuPDF for PDF processing
from typing import Dict, List, Set, Tuple, Optional
import tempfile
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

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
    
    def extract_from_image(self, image: np.ndarray) -> Tuple[Set[str], Set[str]]:
        """
        استخراج تگ‌ها و شناسه‌های JB از یک تصویر.
        
        Args:
            image: تصویر ورودی به صورت آرایه numpy
            
        Returns:
            تاپلی از (tags, jb_identifiers) به صورت مجموعه‌ها
        """
        # پیش‌پردازش تصویر
        processed_image = self.preprocess_image(image)
        
        # تنظیمات Tesseract
        custom_config = r'''--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. -c preserve_interword_spaces=1'''
        
        # ابتدا JB را بر روی کل تصویر تشخیص می‌دهیم
        full_image_text = pytesseract.image_to_string(processed_image, config=custom_config)
        

        jb_identifiers = set()
        for line in full_image_text.split("\n"):
            words = line.split()
            for word in words:
                # تشخیص JB در ابتدای کلمه
                if word.upper().startswith("JB-"):
                    jb_identifiers.add(word.upper())
                

        for line in full_image_text.split("\n"):
            words = line.split()
            for word in words:
                word_upper = word.upper()
    

        # اکنون تگ‌ها را با تقسیم تصویر تشخیص می‌دهیم
        # تقسیم تصویر به دو نیمه با همپوشانی برای OCR بهتر تگ‌ها
        height, width = processed_image.shape
        overlap = 50
        left_half = processed_image[:, :width//2 + overlap]
        right_half = processed_image[:, width//2 - overlap:]
        
        # پردازش هر نیمه به صورت جداگانه
        left_text = pytesseract.image_to_string(left_half, config=custom_config)
        right_text = pytesseract.image_to_string(right_half, config=custom_config)
        
        # ترکیب نتایج و حذف خطوط خالی
        extracted_text = left_text + "\n" + right_text
        extracted_text = "\n".join([line for line in extracted_text.split("\n") if line.strip()])
        
        # استخراج تگ‌ها
        tags = set()
        
        # پردازش متن خط به خط
        for line in extracted_text.split("\n"):
            words = line.split()
            for word in words:
                # پاکسازی پیشوندهای تگ
                clean_word = word.upper()
                if clean_word.startswith("-P-"):
                    clean_word = clean_word.replace("-P-", "", 1)
                if clean_word.startswith("C-P-"):
                    clean_word = clean_word.replace("C-P-", "", 1)
                
                # بررسی الگوی تگ
                if re.search(self.tag_pattern, clean_word, re.IGNORECASE):
                    tags.add(clean_word)
        
        return tags, jb_identifiers
    
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
        ایجاد نگاشتی از تگ‌ها به شناسه‌های JB بر اساس نتایج پردازش PDF.

        Args:
            pdf_results: نتایج از متد process_pdf

        Returns:
            دیکشنری نگاشت تگ‌ها به شناسه‌های JB
        """
        tag_to_jb = {}

        # پردازش هر صفحه
        for page_num, (tags, jb_identifiers) in pdf_results.items():
            if tags and jb_identifiers:
                # اگر فقط **یک JB** در صفحه وجود داشت، همان را انتخاب کند
                if len(jb_identifiers) == 1:
                    jb = next(iter(jb_identifiers))

                # اگر **دو JB** وجود داشت، اولویت را به JB که با الگو تطابق دارد بدهد
                elif len(jb_identifiers) == 2:
                    jb_candidates = list(jb_identifiers)
                    jb = next((jb for jb in jb_candidates if self.jb_pattern.match(jb)), jb_candidates[0])

                # اگر **بیش از دو JB** در صفحه وجود داشت، هیچ JB انتخاب نشود
                else:
                    continue  # از این صفحه عبور کن

                # نگاشت همه تگ‌های این صفحه به JB انتخاب‌شده
                for tag in tags:
                    tag_to_jb[tag] = jb

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
                    
                    logger.info(f"Page {page_num + 1}: Found {len(tags)} tags and {len(jb_identifiers)} JB identifiers")
                    
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
        Process multiple PDF files in parallel.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Dictionary mapping page numbers to tuples of (tags, jb_identifiers)
        """
        combined_results = {}
        page_offset = 0
        
        # Process PDFs in parallel
        num_processes = min(cpu_count(), len(pdf_paths))
        logger.info(f"Processing {len(pdf_paths)} PDF files using {num_processes} processes")
        
        try:
            with Pool(processes=num_processes) as pool:
                pdf_results = pool.map(self.process_pdf, pdf_paths)
                
            # Combine results from all PDFs
            for pdf_result in pdf_results:
                if pdf_result:  # Only process non-empty results
                    for page_num, (tags, jb_identifiers) in pdf_result.items():
                        combined_results[page_num + page_offset] = (tags, jb_identifiers)
                    page_offset += len(pdf_result)
                    
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            for pdf_path in pdf_paths:
                try:
                    pdf_result = self.process_pdf(pdf_path)
                    for page_num, (tags, jb_identifiers) in pdf_result.items():
                        combined_results[page_num + page_offset] = (tags, jb_identifiers)
                    page_offset += len(pdf_result)
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {e}")
                    continue
        
        return combined_results
    def process_excel_chunk(self, chunk_data: Tuple[pd.DataFrame, Dict[str, str]]) -> Tuple[pd.DataFrame, List[str], Set[str]]:
        """
        Process a chunk of Excel data in parallel.
        
        Args:
            chunk_data: Tuple containing (DataFrame chunk, tag_to_jb mapping)
            
        Returns:
            Tuple of (processed DataFrame chunk, unmatched excel tags, unmatched pdf tags)
        """
        chunk, tag_to_jb = chunk_data
        unmatched_excel = []
        unmatched_pdf = set()
        
        # Process each row in the chunk
        for idx, row in chunk.iterrows():
            tag = str(row['Tag No']).strip().upper()
            
            # Skip empty tags
            if pd.isna(tag) or not tag:
                continue
                
            # Check if tag exists in PDF mapping
            if tag in tag_to_jb:
                chunk.at[idx, 'JB'] = tag_to_jb[tag]
            else:
                unmatched_excel.append(tag)
                
            # Check for new tags in PDF that aren't in Excel
            for pdf_tag in tag_to_jb.keys():
                if pdf_tag not in chunk['Tag No'].values:
                  unmatched_pdf.add(pdf_tag)
    
        return chunk, unmatched_excel, unmatched_pdf

    def process_excel(self, excel_path: str, tag_to_jb: Dict[str, str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Process Excel file using parallel processing.
        
        Args:
            excel_path: Path to Excel file
            tag_to_jb: Mapping of tags to JB identifiers
            
        Returns:
            Tuple of (updated DataFrame, unmatched excel tags, unmatched pdf tags)
        """
        logger.info(f"Processing Excel file: {excel_path}")
        
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        if 'Tag No' not in df.columns:
            raise ValueError("Excel file must contain a 'Tag No' column")
        
        # Add new columns
        df['JB'] = None
        df['New Tag'] = None
        
        # Split DataFrame into chunks for parallel processing
        num_processes = cpu_count()
        chunk_size = max(1, len(df) // num_processes)
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Prepare data for parallel processing
        chunk_data = [(chunk, tag_to_jb) for chunk in chunks]
        
        # Process chunks in parallel
        results = []
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.process_excel_chunk, chunk_data)
        
        # Combine results
        processed_chunks = []
        all_unmatched_excel = []
        all_unmatched_pdf = set()
        
        for processed_chunk, unmatched_excel, unmatched_pdf in results:
            processed_chunks.append(processed_chunk)
            all_unmatched_excel.extend(unmatched_excel)
            all_unmatched_pdf.update(unmatched_pdf)
        
        # Combine processed chunks back into single DataFrame
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
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
        
        # Process Excel in parallel
        updated_df, unmatched_excel_tags, unmatched_pdf_tags = self.process_excel(excel_path, tag_to_jb)
        
        # Save updated Excel
        updated_df.to_excel(output_excel_path, index=False)
        logger.info(f"Updated Excel saved to: {output_excel_path}")
        
        return unmatched_excel_tags, unmatched_pdf_tags