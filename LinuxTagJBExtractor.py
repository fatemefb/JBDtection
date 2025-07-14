import os
import sys
import platform
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set
import pytesseract
from DataAnalysisModule import TagJBExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LinuxTagJBExtractor(TagJBExtractor):
    """
    Linux-specific implementation of TagJBExtractor with GPU support.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None, excel_path: Optional[str] = None):
        """
        Initialize the Linux-specific extractor with GPU support detection.
        
        Args:
            tesseract_path: Optional path to Tesseract executable
            excel_path: Optional path to Excel file
        """
        # Linux-specific Tesseract detection
        if tesseract_path is None:
            # Common Linux Tesseract locations
            linux_tesseract_paths = [
                '/usr/bin/tesseract',
                '/usr/local/bin/tesseract',
                '/opt/tesseract/bin/tesseract'
            ]
            
            tesseract_found = False
            for location in linux_tesseract_paths:
                if os.path.exists(location):
                    tesseract_path = location
                    tesseract_found = True
                    logger.info(f"Automatically detected Tesseract at: {tesseract_path}")
                    break
                    
            if not tesseract_found:
                raise RuntimeError("Tesseract not found in common Linux locations. Please provide tesseract_path.")
        
        # Call parent constructor with detected Tesseract path
        super().__init__(tesseract_path=tesseract_path, excel_path=excel_path)
        
        # Initialize GPU variables
        self.gpu_available = False
        self.cuda_device_count = 0
        self.gpu_type = None
        
        # Check for GPU support
        self.check_linux_gpu_support()
        
    def check_linux_gpu_support(self):
        """
        Check for Linux-specific GPU support (CUDA, ROCm, etc.)
        """
        try:
            # Check for NVIDIA GPUs using nvidia-smi
            nvidia_smi_output = os.popen('nvidia-smi -L 2>/dev/null').read()
            if 'GPU' in nvidia_smi_output:
                logger.info("NVIDIA GPU detected via nvidia-smi")
                self.gpu_type = "NVIDIA"
                
                # Get more detailed GPU info
                gpu_info = os.popen('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null').read().strip().split('\n')
                for i, gpu in enumerate(gpu_info):
                    logger.info(f"GPU {i}: {gpu}")
                
                # Check if OpenCV was built with CUDA
                if hasattr(cv2, 'cuda'):
                    self.cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
                    if self.cuda_device_count > 0:
                        self.gpu_available = True
                        logger.info("OpenCV with CUDA support confirmed")
                        # Get GPU info
                        for i in range(self.cuda_device_count):
                            try:
                                gpu_info = cv2.cuda.getDevice()
                                free_memory, total_memory = cv2.cuda.DeviceInfo().freeMemory(), cv2.cuda.DeviceInfo().totalMemory()
                                logger.info(f"CUDA GPU device {i}: {free_memory/(1024*1024):.2f}MB free / {total_memory/(1024*1024):.2f}MB total")
                            except Exception as e:
                                logger.warning(f"Error getting CUDA device info: {e}")
                        
                        logger.info(f"GPU acceleration enabled with {self.cuda_device_count} CUDA device(s)")
                    else:
                        logger.warning("NVIDIA GPU detected, but OpenCV was not built with CUDA support")
            
            # Check for AMD GPUs using rocm-smi
            if not self.gpu_available:
                rocm_smi_output = os.popen('rocm-smi --showproductname 2>/dev/null').read()
                if 'GPU' in rocm_smi_output:
                    logger.info("AMD GPU detected via rocm-smi")
                    self.gpu_type = "AMD"
                    
                    # Get more detailed GPU info
                    gpu_info = os.popen('rocm-smi --showmeminfo vram 2>/dev/null').read().strip().split('\n')
                    for line in gpu_info:
                        if 'GPU' in line and 'vram' in line.lower():
                            logger.info(f"AMD GPU Memory: {line}")
                    
                    # Check if OpenCV was built with OpenCL support for AMD GPUs
                    if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                        logger.info("OpenCV with OpenCL support confirmed (useful for AMD GPUs)")
                        cv2.ocl.setUseOpenCL(True)
                        self.gpu_available = True
                        logger.info(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
                    else:
                        logger.warning("AMD GPU detected, but OpenCV OpenCL support not confirmed")
            
            # If no dedicated GPU found, check for integrated graphics
            if not self.gpu_available:
                # Check for Intel integrated graphics
                intel_gpu = os.popen('lspci | grep -i vga | grep -i intel 2>/dev/null').read()
                if intel_gpu:
                    logger.info("Intel integrated graphics detected")
                    self.gpu_type = "Intel"
                    # Check for OpenCL support
                    if hasattr(cv2, 'ocl') and cv2.ocl.haveOpenCL():
                        logger.info("OpenCV with OpenCL support confirmed (useful for Intel GPUs)")
                        cv2.ocl.setUseOpenCL(True)
                        self.gpu_available = True
                        logger.info(f"OpenCL enabled: {cv2.ocl.useOpenCL()}")
                
                if not self.gpu_available:
                    logger.info("No GPU support detected, using CPU processing")
        
        except Exception as e:
            logger.warning(f"Error checking Linux GPU support: {e}")
            logger.info("Defaulting to CPU processing")
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize file paths for Linux.
        
        Args:
            path: Input file path
            
        Returns:
            Normalized path for Linux
        """
        return Path(path).as_posix()
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image with GPU acceleration if available.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use GPU acceleration if available
        if self.gpu_available:
            try:
                if self.gpu_type == "NVIDIA" and self.cuda_device_count > 0:
                    # NVIDIA GPU processing with CUDA
                    # Upload image to GPU
                    gpu_gray = cv2.cuda_GpuMat()
                    gpu_gray.upload(gray)
                    
                    # Enhance resolution
                    scale_factor = 2
                    gpu_resized = cv2.cuda.resize(gpu_gray, (0, 0), fx=scale_factor, fy=scale_factor, 
                                                interpolation=cv2.INTER_CUBIC)
                    
                    # Apply Gaussian blur on GPU
                    gpu_blurred = cv2.cuda.GaussianBlur(gpu_resized, (3, 3), 0)
                    
                    # Download result back to CPU for operations not supported on GPU
                    gray = gpu_blurred.download()
                
                elif (self.gpu_type == "AMD" or self.gpu_type == "Intel") and hasattr(cv2, 'ocl') and cv2.ocl.useOpenCL():
                    # AMD/Intel GPU processing with OpenCL
                    # Convert to UMat for OpenCL processing
                    umat_image = cv2.UMat(gray)
                    
                    # Enhance resolution
                    scale_factor = 2
                    umat_resized = cv2.resize(umat_image, (0, 0), fx=scale_factor, fy=scale_factor, 
                                            interpolation=cv2.INTER_CUBIC)
                    
                    # Apply Gaussian blur
                    umat_blurred = cv2.GaussianBlur(umat_resized, (3, 3), 0)
                    
                    # Download result back to CPU
                    gray = umat_blurred.get()
                
                # Continue with CPU operations
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                
                # Apply adaptive thresholding
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 31, 2)
                
                # Morphological operations
                kernel = np.ones((2, 2), np.uint8)
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                
                # Dilate slightly
                kernel_dilate = np.ones((1, 1), np.uint8)
                gray = cv2.dilate(gray, kernel_dilate, iterations=1)
                
                logger.debug(f"Image preprocessing completed with {self.gpu_type} GPU acceleration")
                return gray
                
            except Exception as e:
                logger.warning(f"GPU processing failed, falling back to CPU: {e}")
                # Fall back to CPU processing
        
        # CPU processing (same as parent class)
        return super().preprocess_image(image)
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[int, Tuple[Set[str], Set[str], Set[str], List[str], List[str]]]:
        """
        Process multiple PDF files with Linux-specific path handling and optimized batch sizing.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Dictionary mapping page numbers to tuples of extracted data
        """
        # Normalize all paths for Linux
        normalized_paths = [self._normalize_path(path) for path in pdf_paths]
        
        # Optimize batch size based on GPU memory if available
        num_processes = min(os.cpu_count() or 1, len(pdf_paths))
        if self.gpu_available:
            # Get available GPU memory in MB
            try:
                if self.gpu_type == "NVIDIA" and hasattr(cv2, 'cuda'):
                    free_memory = cv2.cuda.DeviceInfo().freeMemory() / (1024 * 1024)
                    # Estimate memory needed per PDF (rough estimate)
                    memory_per_pdf = 500  # MB
                    optimal_batch = max(1, int(free_memory / memory_per_pdf))
                    num_processes = min(num_processes, optimal_batch)
                    logger.info(f"GPU-optimized batch size: {num_processes} processes")
            except Exception as e:
                logger.warning(f"Error calculating GPU-optimized batch size: {e}")
        
        # Call parent implementation with normalized paths
        return super().process_multiple_pdfs(normalized_paths)
    
    def run_with_annotated_pdf(self, pdf_paths: List[str], excel_path: str, 
                             output_excel_path: str, output_pdf_dir: str) -> Tuple[List[str], List[str]]:
        """
        Run complete process with Linux-specific path handling.
        
        Args:
            pdf_paths: List of PDF file paths
            excel_path: Input Excel file path
            output_excel_path: Output Excel file path
            output_pdf_dir: Directory path for storing processed PDFs
            
        Returns:
            Tuple of (unmatched_excel_tags, unmatched_pdf_tags)
        """
        # Normalize all paths for Linux
        normalized_pdf_paths = [self._normalize_path(path) for path in pdf_paths]
        normalized_excel_path = self._normalize_path(excel_path)
        normalized_output_excel_path = self._normalize_path(output_excel_path)
        normalized_output_pdf_dir = self._normalize_path(output_pdf_dir)
        
        # Call parent implementation with normalized paths
        return super().run_with_annotated_pdf(
            normalized_pdf_paths,
            normalized_excel_path,
            normalized_output_excel_path,
            normalized_output_pdf_dir
        )
    