import os
import re
import cv2
import numpy as np
import logging
import traceback
import sys
import time  # Added missing import for time module
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

    def preprocess_image(self, image: np.ndarray, pdf_type: str = "diagrams") -> np.ndarray:
        """
        Preprocess the image using GPU if available.

        Args:
            image: Input image.
            pdf_type: 'diagrams' or 'table'. For 'table' we always use the
                      base class's table branch (CPU) because table-mode
                      preprocessing is just Gaussian + Otsu — no GPU benefit.

        Returns:
            Preprocessed image.
        """
        # TABLE path: skip GPU branch entirely — the table preprocessing
        # pipeline (Gaussian blur + Otsu threshold) is simpler and works
        # best on CPU. Delegating to the base class ensures consistency.
        if pdf_type == 'table':
            return super().preprocess_image(image, pdf_type='table')

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

        return super().preprocess_image(image, pdf_type=pdf_type)

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

    def run_with_annotated_pdf(self, pdf_paths: 'List[str]', excel_path: str, output_excel_path: str, output_pdf_dir: str, 
                            create_zip: bool = True, zip_path: str = None) -> 'Tuple[List[str], List[str]]':
        """
        Delegate to the base implementation (TagJBExtractor) so Linux mode keeps
        full parity with the main pipeline outputs (including pattern-unmatched
        candidates/details used by the dashboard).
        """
        logger.info("Linux run_with_annotated_pdf delegated to base TagJBExtractor pipeline")
        return super().run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=output_pdf_dir,
            create_zip=create_zip,
            zip_path=zip_path,
        )
    
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
    
    def _create_report_file(self, reports_path: str, unmatched_excel_tags: List[str], unmatched_pdf_tags: List[str]) -> None:
        """
        Create a JSON report file with processing statistics and tag information
        """
        try:
            # Get processing stats
            stats = self.get_processing_stats()
            
            # Create report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'unmatched_excel_tags': unmatched_excel_tags,
                'unmatched_pdf_tags': unmatched_pdf_tags
            }
            
            # Add tag numbers if available
            if hasattr(self, 'master_tag_numbers'):
                report_data['tag_numbers'] = self.master_tag_numbers
                
            # Add similarity reports if available
            if hasattr(self, 'similarity_reports'):
                report_data['similarity_reports'] = self.similarity_reports
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(reports_path), exist_ok=True)
            
            # Write to JSON file
            with open(reports_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Report file created: {reports_path}")
            
        except Exception as e:
            logger.error(f"Error creating report file: {e}")
            logger.error(traceback.format_exc())
    
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
                
    def _create_zip_archive(self, zip_path: str, files_to_add: List[str]) -> None:
        """
        Create a ZIP archive containing the specified files
        """
        import zipfile  # Import zipfile module here
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            
            # Create ZIP file
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in files_to_add:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
                        logger.info(f"Added file to ZIP: {file_path}")
                    else:
                        logger.warning(f"File not found, skipping: {file_path}")
                        
            logger.info(f"ZIP archive created at: {zip_path}")
            
        except Exception as e:
            logger.error(f"Error creating ZIP archive: {e}")
            logger.error(traceback.format_exc())
            
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Calculate and return processing statistics
        """
        stats = {}
        try:
            # Default values
            stats = {
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
            
            # Add processing time if available
            if hasattr(self, 'processing_time'):
                stats['processing_time'] = f"{self.processing_time:.2f} seconds"
            
            # If we have similarity reports, use them for stats
            if hasattr(self, 'similarity_reports') and isinstance(self.similarity_reports, list):
                exact_matches = sum(1 for report in self.similarity_reports if report.get('match_type') == 'exact')
                similar_matches = sum(1 for report in self.similarity_reports if report.get('match_type') == 'similar')
                total_matches = exact_matches + similar_matches
                total_tags = len(self.similarity_reports)
                
                stats['total_tags'] = total_tags
                stats['matched_tags'] = total_matches
                stats['exact_matches'] = exact_matches
                stats['similar_matches'] = similar_matches
                
                # Calculate match rates
                if total_tags > 0:
                    match_rate = (total_matches / total_tags) * 100
                    exact_match_rate = (exact_matches / total_tags) * 100 if total_tags > 0 else 0
                    stats['match_rate'] = f"{match_rate:.1f}%"
                    stats['exact_match_rate'] = f"{exact_match_rate:.1f}%"
                    stats['unmatched_tags'] = total_tags - total_matches
            
            # Add JB stats if available
            if hasattr(self, 'total_jbs'):
                stats['total_jbs'] = self.total_jbs
                
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating stats: {e}")
            return stats
