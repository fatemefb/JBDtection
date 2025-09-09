from LinuxTagJBExtractor import LinuxTagJBExtractor
from TagJBExtractorLogger import LoggedTagJBExtractor
from logger_config import LoggerMixin
import logging
import os
import re
import sys
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from file_utils import standardize_path, copy_to_output_paths

# تنظیم لاگر
logger = logging.getLogger(__name__)

class LoggedLinuxTagJBExtractor(LoggerMixin, LinuxTagJBExtractor):
    """
    نسخه بهبودیافته LinuxTagJBExtractor با قابلیت لاگینگ پیشرفته
    این کلاس قابلیت‌های LinuxTagJBExtractor و LoggedTagJBExtractor را ترکیب می‌کند
    """
    
    def __init__(self, tesseract_path=None, excel_path=None):
        # ابتدا LoggerMixin را مقداردهی می‌کنیم
        LoggerMixin.__init__(self)
        # سپس کلاس اصلی را مقداردهی می‌کنیم
        LinuxTagJBExtractor.__init__(self, tesseract_path, excel_path)
        self.logger.info("LoggedLinuxTagJBExtractor initialized")
    
    # متدهای اصلی را بازنویسی می‌کنیم تا از لاگینگ استفاده کنند
    
    def _detect_gpu(self):
        self.logger.info("Detecting GPU...")
        result = super()._detect_gpu()
        if hasattr(self, 'gpu_available') and self.gpu_available:
            self.logger.info(f"GPU detected: {self.gpu_type}")
            if hasattr(self, 'cuda_device_count'):
                self.logger.info(f"CUDA device count: {self.cuda_device_count}")
        else:
            self.logger.info("No GPU detected")
        return result
    
    def enable_gpu(self):
        self.logger.info("Enabling GPU processing")
        return super().enable_gpu()
    
    def disable_gpu(self):
        self.logger.info("Disabling GPU processing")
        return super().disable_gpu()
    
    def build_tag_vectors_from_excel(self, excel_path):
        self.logger.info(f"Building tag vectors from Excel: {excel_path}")
        result = super().build_tag_vectors_from_excel(excel_path)
        self.logger.info(f"Built tag vectors from Excel: {len(self.tag_patterns)} patterns found")
        return result
    
    def process_pdf(self, pdf_path):
        self.logger.info(f"Processing PDF: {pdf_path}")
        result = super().process_pdf(pdf_path)
        self.logger.info(f"Processed PDF: {pdf_path}, found {len(result)} pages with data")
        return result
    
    def process_multiple_pdfs(self, pdf_paths):
        self.logger.info(f"Processing {len(pdf_paths)} PDFs")
        result = super().process_multiple_pdfs(pdf_paths)
        self.logger.info(f"Processed {len(pdf_paths)} PDFs, found {len(result)} pages with data")
        return result
    
    def extract_from_image(self, image):
        """
        استخراج اطلاعات از تصویر با ثبت لاگ
        
        Args:
            image: تصویر ورودی
            
        Returns:
            نتیجه استخراج شده از تصویر (ممکن است 5 یا 6 مقدار برگرداند)
        """
        self.logger.info("Extracting information from image")
        result = super().extract_from_image(image)
        
        # بررسی تعداد مقادیر برگشتی
        if isinstance(result, tuple):
            if len(result) == 5:
                self.logger.info(f"Extracted 5 values: {len(result[0])} tags, {len(result[1])} JBs, {len(result[2])} MCs, {len(result[3])} cable descriptions, {len(result[4])} spare identifiers")
            elif len(result) == 6:
                self.logger.info(f"Extracted 6 values: {len(result[0])} tags, {len(result[1])} JBs, {len(result[2])} MCs, {len(result[3])} cable descriptions, {len(result[4])} spare identifiers, {len(result[5])} tag-to-number mappings")
            else:
                self.logger.warning(f"Unexpected number of values returned: {len(result)}")
        else:
            self.logger.warning(f"Unexpected return type from extract_from_image: {type(result)}")
        
        return result
    
    def create_annotated_pdf(self, pdf_path, output_pdf_path):
        """
        ایجاد PDF حاشیه‌نویسی شده با ثبت لاگ
        
        Args:
            pdf_path: مسیر فایل PDF ورودی
            output_pdf_path: مسیر فایل PDF خروجی
            
        Returns:
            دیکشنری شماره‌گذاری تگ‌ها
        """
        self.logger.info(f"Creating annotated PDF: {pdf_path} -> {output_pdf_path}")
        result = super().create_annotated_pdf(pdf_path, output_pdf_path)
        self.logger.info(f"Created annotated PDF with {len(result)} tagged elements")
        return result
    
    def run_with_annotated_pdf(self, pdf_paths, excel_path, output_excel_path, output_pdf_dir):
        """
        اجرای کامل پردازش با PDF های حاشیه‌نویسی شده
        
        Args:
            pdf_paths: لیست مسیرهای فایل‌های PDF
            excel_path: مسیر فایل اکسل ورودی
            output_excel_path: مسیر فایل اکسل خروجی
            output_pdf_dir: مسیر دایرکتوری PDF های خروجی
            
        Returns:
            نتیجه پردازش
        """
        self.logger.info(f"شروع پردازش با PDF های حاشیه‌نویسی شده")
        self.logger.info(f"تعداد فایل‌های PDF: {len(pdf_paths)}")
        
        # استاندارد کردن مسیرها
        output_excel_path = standardize_path(output_excel_path)
        output_pdf_dir = standardize_path(output_pdf_dir)
        
        self.logger.info(f"مسیر خروجی اکسل: {output_excel_path}")
        self.logger.info(f"مسیر خروجی PDF: {output_pdf_dir}")
        
        # اجرای متد اصلی
        result = super().run_with_annotated_pdf(pdf_paths, excel_path, output_excel_path, output_pdf_dir)
        
        # ثبت نتایج
        if isinstance(result, tuple) and len(result) == 2:
            unmatched_excel_tags, unmatched_pdf_tags = result
            self.logger.info(f"پردازش با موفقیت انجام شد. تگ‌های تطبیق نیافته در اکسل: {len(unmatched_excel_tags)}, تگ‌های تطبیق نیافته در PDF: {len(unmatched_pdf_tags)}")
        else:
            self.logger.info("پردازش با موفقیت انجام شد.")
        
        return result
    
    def set_wire_color_rule(self, rule):
        """
        تنظیم قانون تولید رنگ‌های سیم
        
        Args:
            rule: قانون تولید رنگ‌های سیم
        """
        try:
            self.logger.info(f"Setting wire color rule: {rule}")
            self.wire_color_rule = rule
            
            # تست قانون با یک نمونه
            test_colors = self.generate_mc_wire_colors(1)
            self.logger.info(f"Test wire colors for tag #1: {test_colors}")
        except Exception as e:
            self.logger.error(f"Error Setting wire color rule: {e}")

    def set_scr_number_rule(self, rule):
        """
        تنظیم قانون تولید شماره SCR
        
        Args:
            rule: قانون تولید شماره SCR
        """
        try:
            self.logger.info(f"Setting SCR number rule: {rule}")
            self.scr_number_rule = rule
            
            # تست قانون با یک نمونه
            test_scr = self.generate_scr_number(1)
            self.logger.info(f"Test SCR number for tag #1: {test_scr}")
        except Exception as e:
            self.logger.error(f"Error Setting SCR number rule: {e}")
    
    def preprocess_image(self, image):
        """
        پیش‌پردازش تصویر با استفاده از GPU در صورت امکان
        
        Args:
            image: تصویر ورودی
            
        Returns:
            تصویر پیش‌پردازش شده
        """
        self.logger.info("Preprocessing image...")
        if self.use_gpu and self.gpu_available:
            self.logger.info(f"Using GPU ({self.gpu_type}) for preprocessing")
        else:
            self.logger.info("Using CPU for preprocessing")
        
        result = super().preprocess_image(image)
        self.logger.info("Image preprocessing completed")
        return result