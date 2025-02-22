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
        مقداردهی اولیه استخراج‌کننده با مسیر اختیاری tesseract و مسیر اکسل.
        
        Args:
            tesseract_path: مسیر فایل اجرایی tesseract (در صورت نیاز)
            excel_path: مسیر فایل اکسل برای ساخت الگوی تگ به صورت پویا (در صورت نیاز)
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # نیازی به الگوی پیچیده JB نیست - هر کلمه‌ای که شامل "JB" باشد را تشخیص می‌دهیم
        
        # ساخت الگوی تگ به صورت پویا اگر فایل اکسل ارائه شده باشد
        if excel_path:
            self.build_tag_pattern_from_excel(excel_path)
        else:
            # الگوی تگ پیش‌فرض به عنوان پشتیبان
            self.tag_pattern = r'''\b(?:FT|FV|TT|PT|CP|XP|EHSDI|LAHSD)[-.]?\d{2,4}(?:[A-Z]{1,2}\d*)?\b'''
            
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
        
        # استخراج شناسه‌های JB از تصویر کامل
        jb_identifiers = set()
        for line in full_image_text.split("\n"):
            words = line.split()
            for word in words:
                # تشخیص JB در ابتدای کلمه
                if word.upper().startswith("JB"):
                    jb_identifiers.add(word.upper())
        
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
                self.tag_pattern = r'''\b(?:FT|FV|TT|PT|CP|XP|EHSDI|LAHSD)[-.]?\d{2,4}(?:[A-Z]{1,2}\d*)?\b'''
                return
                
            # استخراج پیشوندها از ستون Tag NO
            prefixes = set()
            for tag in df['Tag No'].dropna():
                prefix = self.extract_tag_prefix(str(tag).strip().upper())
                if prefix:
                    prefixes.add(prefix)
                    
            if not prefixes:
                logger.warning("No valid prefixes found in Tag NO column. Using default pattern.")
                self.tag_pattern = r'''\b(?:FT|FV|TT|PT|CP|XP|EHSDI|LAHSD)[-.]?\d{2,4}(?:[A-Z]{1,2}\d*)?\b'''
                return
                
            # ایجاد یک الگوی regex جدید با استفاده از پیشوندهای جمع‌آوری شده
            prefix_pattern = '|'.join(prefixes)
            self.tag_pattern = r'''\b(?:{})[-.]?\d{{2,4}}(?:[A-Z]{{1,2}}\d*)?\b'''.format(prefix_pattern)
            
            logger.info(f"Built dynamic tag pattern from {len(prefixes)} prefixes: {prefix_pattern}")
            
        except Exception as e:
            logger.error(f"Error building tag pattern from Excel: {e}")
            # بازگشت به الگوی پیش‌فرض
            self.tag_pattern = r'''\b(?:FT|FV|TT|PT|CP|XP|EHSDI|LAHSD)[-.]?\d{2,4}(?:[A-Z]{1,2}\d*)?\b'''
            
    def create_tag_jb_mapping(self, pdf_results: Dict[int, Tuple[Set[str], Set[str]]]) -> Dict[str, str]:
        """
        ایجاد نگاشتی از تگ‌ها به شناسه‌های JB بر اساس نتایج پردازش PDF.
        
        Args:
            pdf_results: نتایج از متد process_pdf
            
        Returns:
            دیکشنری نگاشت تگ‌ها به شناسه‌های JB
        """
        tag_to_jb = {}
        
        # برای هر صفحه
        for page_num, (tags, jb_identifiers) in pdf_results.items():
            # اگر هم تگ و هم شناسه JB در این صفحه وجود دارد
            if tags and jb_identifiers:
                # دریافت اولین شناسه JB (با فرض یک JB در هر صفحه)
                jb = next(iter(jb_identifiers))
                
                # نگاشت هر تگ به این JB
                for tag in tags:
                    tag_to_jb[tag] = jb
        
        return tag_to_jb
        
    def process_excel(self, excel_path: str, tag_to_jb: Dict[str, str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        پردازش فایل اکسل، تطبیق تگ‌ها با شناسه‌های JB و به‌روزرسانی.
        
        Args:
            excel_path: مسیر فایل اکسل
            tag_to_jb: نگاشت تگ‌ها به شناسه‌های JB
            
        Returns:
            تاپلی از (updated_dataframe, unmatched_excel_tags, unmatched_pdf_tags)
        """
        logger.info(f"Processing Excel file: {excel_path}")
        
        # خواندن فایل اکسل
        df = pd.read_excel(excel_path)
        
        # اطمینان از وجود ستون 'Tag NO'
        if 'Tag No' not in df.columns:
            raise ValueError("Excel file must contain a 'Tag NO' column")
        
        # ایجاد ستون JB جدید
        df['JB'] = None
        
        # لیست‌هایی برای ردیابی تگ‌های تطبیق نشده
        unmatched_excel_tags = []
        unmatched_pdf_tags = set(tag_to_jb.keys())
        
        # پردازش هر سطر در اکسل
        for idx, row in df.iterrows():
            tag_no = str(row['Tag No']).strip().upper()
            
            # بررسی اینکه آیا تگ در نگاشت ما وجود دارد
            if tag_no in tag_to_jb:
                df.at[idx, 'JB'] = tag_to_jb[tag_no]
                unmatched_pdf_tags.discard(tag_no)
            else:
                unmatched_excel_tags.append(tag_no)
        
        return df, unmatched_excel_tags, list(unmatched_pdf_tags)
        
    def run(self, pdf_path: str, excel_path: str, output_excel_path: str) -> Tuple[List[str], List[str]]:
        """
        اجرای فرآیند کامل: استخراج از PDF، پردازش اکسل و ذخیره نتایج.
        
        Args:
            pdf_path: مسیر فایل PDF
            excel_path: مسیر فایل اکسل ورودی
            output_excel_path: مسیر ذخیره فایل اکسل خروجی
            
        Returns:
            تاپلی از (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # ساخت الگوی تگ از فایل اکسل ابتدا
        self.build_tag_pattern_from_excel(excel_path)
        logger.info(f"Using tag pattern: {self.tag_pattern}")
        
        # پردازش PDF
        pdf_results = self.process_pdf(pdf_path)
        
        # ایجاد نگاشت تگ به JB
        tag_to_jb = self.create_tag_jb_mapping(pdf_results)
        
        # پردازش اکسل
        updated_df, unmatched_excel_tags, unmatched_pdf_tags = self.process_excel(excel_path, tag_to_jb)
        
        # ذخیره اکسل به‌روزرسانی شده
        updated_df.to_excel(output_excel_path, index=False)
        logger.info(f"Updated Excel saved to: {output_excel_path}")
        
        # بازگرداندن تگ‌های تطبیق نشده
        return unmatched_excel_tags, unmatched_pdf_tags