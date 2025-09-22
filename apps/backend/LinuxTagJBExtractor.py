import os
import re
import cv2
import numpy as np
import logging
import traceback
import sys
from typing import List, Dict, Set, Tuple, Any, Optional, Union
import shutil
import pandas as pd
import json
from datetime import datetime


# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
from apps.backend.utils.file_utils import standardize_path, copy_to_output_paths, ensure_directory_exists, verify_file_exists_with_retries

# Check for GPU support
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    
    # بررسی وضعیت GPU به روش امن
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_info = f"Found {len(gpus)} GPU(s): {gpus}"
        print(f"GPU detected: {gpu_info}")
    else:
        print("No GPU detected, using CPU only.")
        
except (ImportError, AttributeError, TypeError) as e:
    TF_AVAILABLE = False
    print(f"TensorFlow not available or error initializing: {e}")
    
try:
    from DataAnalysisModule import TagJBExtractor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from DataAnalysisModule import TagJBExtractor

# Configure logger
logger = logging.getLogger(__name__)

class LinuxTagJBExtractor(TagJBExtractor):
    """
    Optimized Tag and JB Extractor for Linux systems with GPU support.
    """

    def __init__(self, tesseract_path=None, excel_path=None):
        """
        Initialize the class with GPU support.

        Args:
            tesseract_path: Path to Tesseract OCR executable.
            excel_path: Path to reference Excel file.
        """
        super().__init__(tesseract_path, excel_path)

        self.gpu_available = False
        self.gpu_type = "None"
        self.cuda_device_count = 0
        self.use_gpu = False

        self._detect_gpu()
        self._compile_regex_patterns()

    def _detect_gpu(self):
        """
        Detect available GPU and its type.
        """
        try:
            # Check for NVIDIA GPUs
            import subprocess
            nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if nvidia_smi.returncode == 0:
                self.gpu_available = True
                self.gpu_type = "NVIDIA"
                if TF_AVAILABLE:
                    self.cuda_device_count = len(tf.config.list_physical_devices('GPU'))
                    logger.info(f"NVIDIA GPU detected with {self.cuda_device_count} CUDA devices")
            else:
                logger.info("No NVIDIA GPU detected")

            # Check for AMD GPUs
            if not self.gpu_available:
                rocm_smi = subprocess.run(['rocm-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if rocm_smi.returncode == 0:
                    self.gpu_available = True
                    self.gpu_type = "AMD"
                    logger.info("AMD GPU detected")

            # Check for Intel GPUs
            if not self.gpu_available:
                intel_gpu = subprocess.run(['lspci', '|', 'grep', '-i', 'intel.*graphics'], 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if intel_gpu.returncode == 0 and intel_gpu.stdout:
                    self.gpu_available = True
                    self.gpu_type = "Intel"
                    logger.info("Intel GPU detected")

        except Exception as e:
            logger.error(f"Error detecting GPU: {e}")

        logger.info(f"GPU available: {self.gpu_available}, Type: {self.gpu_type}")

    def _compile_regex_patterns(self):
        """
        Compile regex patterns for faster processing.
        """
        self.tag_patterns = [
            re.compile(r'\b(?:UZSO|UZSC|UY|UHSL|UHSH|TY|TIT|TCV|PIT|PDIT|PCV|LIT|LCV|LA|HZSC|HCV|FIT|FCV|AXA|ASL|AIT)[-.]?\d{3}[-.]?\d{2,3}[A-Z]?\d*\b', re.IGNORECASE),
            re.compile(r'\b[A-Z]{2,4}[-.]?\d{3}[-.]?\d{2,3}[A-Z]?\d*\b', re.IGNORECASE)
        ]

        self.jb_patterns = [
            re.compile(r'\bJB[-.]?\d{3,4}[A-Z]?\b', re.IGNORECASE),
            re.compile(r'\b[A-Z]?JB[-.]?\d{2,4}[A-Z]?\b', re.IGNORECASE)
        ]

        self.mc_patterns = [
            re.compile(r'\bMC[-.]?\d{3,4}[A-Z]?\b', re.IGNORECASE),
            re.compile(r'\b[A-Z]?MC[-.]?\d{2,4}[A-Z]?\b', re.IGNORECASE)
        ]

        self.spare_patterns = [
            re.compile(r'\bSPARE\b', re.IGNORECASE)
        ]

        self.cable_patterns = [
            re.compile(r'(\d+)\s*(?:PAIR|P|PR)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(?:TRIPLE|T|TR)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(?:CORE|C|CR)', re.IGNORECASE),
            re.compile(r'(\d+)P', re.IGNORECASE),
            re.compile(r'(\d+)C', re.IGNORECASE),
            re.compile(r'(\d+)T', re.IGNORECASE)
        ]

    def enable_gpu(self):
        """
        Enable GPU processing.
        """
        if not self.gpu_available:
            logger.warning("No GPU available, using CPU processing")
            return False

        self.use_gpu = True
        logger.info(f"GPU processing enabled: {self.gpu_type}")

        if self.gpu_type == "NVIDIA" and TF_AVAILABLE:
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"TensorFlow configured to use {len(physical_devices)} NVIDIA GPUs")
            except Exception as e:
                logger.error(f"Error configuring TensorFlow for GPU: {e}")

        try:
            if self.gpu_type == "NVIDIA":
                cv2.setUseOptimized(True)
                cv2.cuda.setDevice(0)
                logger.info("OpenCV configured to use NVIDIA GPU")
            elif self.gpu_type in ["AMD", "Intel"]:
                cv2.setUseOptimized(True)
                logger.info(f"OpenCV optimizations enabled for {self.gpu_type} GPU")
        except Exception as e:
            logger.error(f"Error configuring OpenCV for GPU: {e}")

        return True

    def disable_gpu(self):
        """
        Disable GPU processing.
        """
        self.use_gpu = False
        logger.info("GPU processing disabled")
        cv2.setUseOptimized(False)
        return True

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image using GPU if available.

        Args:
            image: Input image.

        Returns:
            Preprocessed image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if self.use_gpu and self.gpu_available:
            try:
                if self.gpu_type == "NVIDIA" and hasattr(cv2, 'cuda'):
                    gpu_gray = cv2.cuda_GpuMat()
                    gpu_gray.upload(gray)

                    scale_factor = 2
                    gpu_gray = cv2.cuda.resize(gpu_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                    gpu_gray = cv2.cuda.GaussianBlur(gpu_gray, (3, 3), 0)

                    gray = gpu_gray.download()

                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                    gray = cv2.medianBlur(gray, 3)
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 31, 2)

                    kernel = np.ones((2, 2), np.uint8)
                    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

                    kernel_dilate = np.ones((1, 1), np.uint8)
                    gray = cv2.dilate(gray, kernel_dilate, iterations=1)

                    return gray
            except Exception as e:
                logger.error(f"Error using GPU for preprocessing: {e}")
                logger.info("Falling back to CPU preprocessing")

        return super().preprocess_image(image)

    def set_patterns(self, jb_examples=None, mc_examples=None, spare_examples=None, 
                     cable_examples=None, wire_color_rule=None, scr_number_rule=None):
        """
        Set custom patterns for detection.

        Args:
            jb_examples: List of JB examples.
            mc_examples: List of MC examples.
            spare_examples: List of SPARE examples.
            cable_examples: List of cable examples.
            wire_color_rule: Rule for generating wire colors.
            scr_number_rule: Rule for generating SCR numbers.
        """
        if jb_examples:
            self.jb_examples = jb_examples
            jb_pattern_str = r'\b(?:' + '|'.join(re.escape(ex) for ex in jb_examples) + r')[-.]?\d+[A-Z]?\b'
            self.jb_patterns.append(re.compile(jb_pattern_str, re.IGNORECASE))
            logger.info(f"Added custom JB pattern: {jb_pattern_str}")

        if mc_examples:
            self.mc_examples = mc_examples
            mc_pattern_str = r'\b(?:' + '|'.join(re.escape(ex) for ex in mc_examples) + r')[-.]?\d+[A-Z]?\b'
            self.mc_patterns.append(re.compile(mc_pattern_str, re.IGNORECASE))
            logger.info(f"Added custom MC pattern: {mc_pattern_str}")

        if spare_examples:
            self.spare_examples = spare_examples
            spare_pattern_str = r'\b(?:' + '|'.join(re.escape(ex) for ex in spare_examples) + r')\b'
            self.spare_patterns.append(re.compile(spare_pattern_str, re.IGNORECASE))
            logger.info(f"Added custom SPARE pattern: {spare_pattern_str}")

        if cable_examples:
            self.cable_examples = cable_examples
            for example in cable_examples:
                if 'PAIR' in example.upper() or 'P' in example.upper():
                    pattern_str = r'(\d+)\s*(?:' + re.escape(example) + r')'
                    self.cable_patterns.append(re.compile(pattern_str, re.IGNORECASE))
                    logger.info(f"Added custom cable pattern: {pattern_str}")

        if wire_color_rule:
            self.set_wire_color_rule(wire_color_rule)

        if scr_number_rule:
            self.set_scr_number_rule(scr_number_rule)

    def run_with_annotated_pdf(self, pdf_paths, excel_path, output_excel_path, output_pdf_dir):
        """
        اجرای پردازش کامل با PDF های حاشیه‌نویسی شده با بهبود مدیریت فایل‌ها

        Args:
            pdf_paths: لیست مسیرهای فایل PDF
            excel_path: مسیر فایل Excel ورودی
            output_excel_path: مسیر فایل Excel خروجی
            output_pdf_dir: مسیر دایرکتوری PDF های خروجی

        Returns:
            تاپل شامل (unmatched_excel_tags, unmatched_pdf_tags)
        """
        self.logger.info(f"شروع پردازش با PDF های حاشیه‌نویسی شده")
        self.logger.info(f"تعداد فایل‌های PDF: {len(pdf_paths)}")
        
        # استاندارد کردن مسیرها
        output_excel_path = standardize_path(output_excel_path)
        output_pdf_dir = standardize_path(output_pdf_dir)
        
        # اطمینان از وجود دایرکتوری‌های خروجی
        ensure_directory_exists(output_pdf_dir)
        ensure_directory_exists(os.path.dirname(output_excel_path))
        
        self.logger.info(f"مسیر خروجی اکسل: {output_excel_path}")
        self.logger.info(f"مسیر خروجی PDF: {output_pdf_dir}")
        
        try:
            # اجرای متد اصلی
            result = super().run_with_annotated_pdf(pdf_paths, excel_path, output_excel_path, output_pdf_dir)
            
            # بررسی نوع خروجی
            if isinstance(result, tuple) and len(result) == 2:
                unmatched_excel_tags, unmatched_pdf_tags = result
            else:
                self.logger.warning(f"Unexpected return type from parent method: {type(result)}")
                unmatched_excel_tags, unmatched_pdf_tags = [], []
            
            # بررسی وجود فایل‌های خروجی
            excel_exists = os.path.exists(output_excel_path)
            if not excel_exists:
                self.logger.warning(f"Excel output file not found at {output_excel_path}, creating empty file")
                self._create_empty_excel(output_excel_path)
            
            # ایجاد فایل گزارش
            report_path = os.path.join(output_pdf_dir, "processing_reports.json")
            self._create_report_file(report_path, unmatched_excel_tags, unmatched_pdf_tags)
            
            # ایجاد فایل Excel تگ‌های تطبیق نیافته
            unmatched_excel_path = os.path.join(os.path.dirname(output_excel_path), 
                                               f"Aryavakav-NGL-UnmatchedTags-{datetime.now().strftime('%Y-%m-%d')}-v1.0.xlsx")
            self._create_unmatched_tags_excel(unmatched_excel_path, unmatched_excel_tags, unmatched_pdf_tags)
            self.logger.info(f"فایل Excel تگ‌های تطبیق نیافته ذخیره شد: {unmatched_excel_path}")
            
            # کپی فایل‌های خروجی به مسیر سرور
            server_output_dir = "/home/devio/JB-outputs"
            files_to_copy = []
            
            # اضافه کردن فایل‌های موجود به لیست کپی
            if os.path.exists(output_excel_path):
                files_to_copy.append(output_excel_path)
            if os.path.exists(unmatched_excel_path):
                files_to_copy.append(unmatched_excel_path)
            if os.path.exists(report_path):
                files_to_copy.append(report_path)
                
            # اضافه کردن فایل‌های PDF حاشیه‌نویسی شده به لیست کپی
            for pdf_path in pdf_paths:
                pdf_name = os.path.basename(pdf_path)
                annotated_pdf_path = os.path.join(output_pdf_dir, f"annotated_{pdf_name}")
                if os.path.exists(annotated_pdf_path):
                    files_to_copy.append(annotated_pdf_path)
            
            # کپی فایل‌ها به مسیر سرور
            copy_result = copy_to_output_paths(files_to_copy, server_output_dir)
            if copy_result['server_success']:
                self.logger.info(f"Output files successfully copied to server directory: {server_output_dir}")
                self.logger.info(f"Server files: {copy_result['server_files']}")
            else:
                self.logger.warning(f"Failed to copy output files to server directory: {copy_result.get('error', 'Unknown error')}")
            
            # ایجاد فایل ZIP از همه خروجی‌ها
            zip_path = os.path.join(os.path.dirname(output_excel_path), 
                                   f"Aryavakav-NGL-Results-{datetime.now().strftime('%Y-%m-%d')}-v1.0.zip")
            self._create_zip_archive(zip_path, files_to_copy)
            self.logger.info(f"Created ZIP archive: {zip_path}")
            
            # ثبت اطلاعات تکمیلی
            self.logger.info(f"پردازش با موفقیت انجام شد")
            self.logger.info(f"تعداد تگ‌های یافت نشده در Excel: {len(unmatched_excel_tags)}")
            self.logger.info(f"تعداد تگ‌های یافت نشده در PDF: {len(unmatched_pdf_tags)}")
            
            return unmatched_excel_tags, unmatched_pdf_tags
            
        except Exception as e:
            self.logger.error(f"خطا در اجرای پردازش: {e}")
            self.logger.error(traceback.format_exc())
            # برگرداندن مقادیر خالی در صورت بروز خطا
            return [], []
    
    def _create_empty_excel(self, file_path):
        """
        ایجاد فایل Excel خالی
        
        Args:
            file_path: مسیر فایل Excel
        """
        try:
            # ایجاد دیتافریم خالی با ستون‌های پیش‌فرض
            columns = ['PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                      'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 
                      'Terminal_Second_Number', 'SCR_Terminal_Number', 'Cable_Code',
                      'Cable_Description', 'Type', 'Tag_Number_Status']
            df = pd.DataFrame(columns=columns)
            
            # ذخیره فایل
            ensure_directory_exists(os.path.dirname(file_path))
            df.to_excel(file_path, index=False)
            self.logger.info(f"Created empty Excel file: {file_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating empty Excel file: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_report_file(self, file_path, unmatched_excel_tags, unmatched_pdf_tags):
        """
        ایجاد فایل گزارش JSON
        
        Args:
            file_path: مسیر فایل گزارش
            unmatched_excel_tags: تگ‌های تطبیق نیافته در Excel
            unmatched_pdf_tags: تگ‌های تطبیق نیافته در PDF
        """
        try:
            # ایجاد دیکشنری گزارش
            report = {
                'timestamp': datetime.now().isoformat(),
                'stats': self.get_processing_stats(),
                'unmatched_excel_tags': list(unmatched_excel_tags) if unmatched_excel_tags else [],
                'unmatched_pdf_tags': list(unmatched_pdf_tags) if unmatched_pdf_tags else [],
                'tag_numbers': getattr(self, 'tag_to_number', {})
            }
            
            # ذخیره فایل
            ensure_directory_exists(os.path.dirname(file_path))
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Reports and tag numbers saved to: {file_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating report file: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_unmatched_tags_excel(self, file_path, unmatched_excel_tags, unmatched_pdf_tags):
        """
        ایجاد فایل Excel تگ‌های تطبیق نیافته
        
        Args:
            file_path: مسیر فایل Excel
            unmatched_excel_tags: تگ‌های تطبیق نیافته در Excel
            unmatched_pdf_tags: تگ‌های تطبیق نیافته در PDF
        """
        try:
            # اطمینان از اینکه file_path یک رشته است
            if not isinstance(file_path, str):
                self.logger.error(f"Invalid file_path type: {type(file_path)}, expected string")
                file_path = str(file_path) if file_path else "/home/devio/JB-outputs/unmatched_tags.xlsx"
            
            # تبدیل به لیست
            excel_tags = list(unmatched_excel_tags) if unmatched_excel_tags else []
            pdf_tags = list(unmatched_pdf_tags) if unmatched_pdf_tags else []
            
            # ایجاد دیتافریم
            df_excel = pd.DataFrame({'IO_List_Tags': excel_tags + [''] * (len(pdf_tags) - len(excel_tags) if len(pdf_tags) > len(excel_tags) else 0)})
            df_pdf = pd.DataFrame({'PDF_Tags': pdf_tags + [''] * (len(excel_tags) - len(pdf_tags) if len(excel_tags) > len(pdf_tags) else 0)})
            
            # ترکیب دیتافریم‌ها
            df = pd.concat([df_excel, df_pdf], axis=1)
            
            # ذخیره فایل
            ensure_directory_exists(os.path.dirname(file_path))
            df.to_excel(file_path, index=False)
            self.logger.info(f"Created unmatched tags Excel file with {len(excel_tags)} IO List tags and {len(pdf_tags)} PDF tags")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating unmatched tags Excel file: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _create_zip_archive(self, zip_path, files_to_zip):
        """
        ایجاد فایل ZIP از فایل‌های خروجی
        
        Args:
            zip_path: مسیر فایل ZIP
            files_to_zip: لیست فایل‌های مورد نظر برای فشرده‌سازی
        """
        try:
            import zipfile
            
            # ایجاد فایل ZIP
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in files_to_zip:
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
                        self.logger.info(f"Added file to ZIP: {file_path}")
                    else:
                        self.logger.warning(f"File not found for ZIP: {file_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating ZIP archive: {e}")
            self.logger.error(traceback.format_exc())
            return False