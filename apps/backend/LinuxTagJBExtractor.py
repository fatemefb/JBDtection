import os
import re
import cv2
import numpy as np
import logging
import traceback
from typing import List, Dict, Set, Tuple, Any, Optional, Union

# Check for GPU support
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

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