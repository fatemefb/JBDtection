import os
import logging
import numpy as np
import cv2
import re
import gc
import time
import json
import traceback
import tempfile
from typing import Dict, List, Tuple, Set, Any, Union
import pandas as pd
import fitz  # PyMuPDF

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import base class
from DataAnalysisModule import TagJBExtractor

# Try to import GPU libraries, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available for GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch is not available")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    logger.info("TensorFlow is available for GPU acceleration")
except ImportError:
    TF_AVAILABLE = False
    logger.info("TensorFlow is not available")


class LinuxTagJBExtractor(TagJBExtractor):
    """
    نسخه بهینه‌سازی شده TagJBExtractor برای لینوکس با پشتیبانی از GPU
    این کلاس، کلاس پایه TagJBExtractor را با بهینه‌سازی‌هایی برای محیط‌های لینوکس
    و شتاب‌دهی GPU برای پردازش تصویر گسترش می‌دهد.
    """
    
    def __init__(self, tesseract_path=None, excel_path=None):
        """
        مقداردهی اولیه LinuxTagJBExtractor با تشخیص GPU
        
        Args:
            tesseract_path: مسیر اجرایی Tesseract
            excel_path: مسیر فایل اکسل با تگ‌های مرجع
        """
        super().__init__(tesseract_path, excel_path)
        
        # بررسی در دسترس بودن GPU
        self.gpu_available = False
        self.gpu_type = None
        self.cuda_device_count = 0
        
        # تلاش برای تشخیص GPU NVIDIA با CUDA
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.gpu_available = True
                    self.gpu_type = "NVIDIA"
                    self.cuda_device_count = torch.cuda.device_count()
                    logger.info(f"GPU NVIDIA شناسایی شد: {torch.cuda.get_device_name(0)}")
                    logger.info(f"تعداد دستگاه‌های CUDA: {self.cuda_device_count}")
                    
                    # تنظیم PyTorch برای استفاده از GPU
                    self.device = torch.device("cuda:0")
                else:
                    logger.info("GPU NVIDIA شناسایی نشد")
            except Exception as e:
                logger.warning(f"خطا در تشخیص GPU NVIDIA: {e}")
        
        # اگر NVIDIA GPU در دسترس نبود، TensorFlow را بررسی کنیم
        if not self.gpu_available and TF_AVAILABLE:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.gpu_available = True
                    self.gpu_type = "TensorFlow-compatible"
                    logger.info(f"GPU سازگار با TensorFlow شناسایی شد: {len(gpus)} دستگاه")
                    
                    # تنظیم TensorFlow برای استفاده از GPU
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    self.device = "tf-gpu"
                else:
                    logger.info("هیچ GPU سازگار با TensorFlow شناسایی نشد")
            except Exception as e:
                logger.warning(f"خطا در تشخیص GPU سازگار با TensorFlow: {e}")
        
        # اگر هیچ GPU شناسایی نشد، از CPU استفاده می‌کنیم
        if not self.gpu_available:
            logger.info("هیچ GPU شناسایی نشد، از پردازش CPU استفاده می‌شود")
            self.device = "cpu"
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر بهبود یافته با شتاب‌دهی GPU در صورت وجود
        
        Args:
            image: تصویر ورودی به صورت آرایه numpy
            
        Returns:
            تصویر پیش‌پردازش شده
        """
        # استفاده از شتاب‌دهی GPU در صورت وجود
        if self.gpu_available:
            try:
                if self.gpu_type == "NVIDIA" and TORCH_AVAILABLE:
                    # استفاده از PyTorch برای پردازش تصویر با شتاب‌دهی GPU
                    return self._preprocess_with_pytorch(image)
                elif self.gpu_type == "TensorFlow-compatible" and TF_AVAILABLE:
                    # استفاده از TensorFlow برای پردازش تصویر با شتاب‌دهی GPU
                    return self._preprocess_with_tensorflow(image)
                else:
                    # استفاده از پردازش CPU از کلاس والد
                    return super().preprocess_image(image)
            except Exception as e:
                logger.warning(f"پردازش GPU با خطا مواجه شد، استفاده از CPU: {e}")
                # بازگشت به پردازش CPU
                return super().preprocess_image(image)
        else:
            # استفاده از پردازش CPU استاندارد از کلاس والد
            return super().preprocess_image(image)
    
    def _preprocess_with_pytorch(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر با استفاده از PyTorch روی GPU
        
        Args:
            image: تصویر ورودی
            
        Returns:
            تصویر پردازش شده
        """
        try:
            # تبدیل به تصویر خاکستری در صورت نیاز
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # تبدیل به تنسور PyTorch و انتقال به GPU
            img_tensor = torch.from_numpy(gray).float().to(self.device)
            
            # اعمال عملیات پیش‌پردازش روی GPU
            # افزایش وضوح
            scale_factor = 2
            height, width = img_tensor.shape
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.view(1, 1, height, width),
                scale_factor=scale_factor,
                mode='bicubic',
                align_corners=False
            ).squeeze()
            
            # تبدیل به CPU و numpy برای عملیات OpenCV
            img_np = img_tensor.cpu().numpy().astype(np.uint8)
            
            # اعمال CLAHE برای کنتراست بهتر
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            
            # اعمال آستانه‌گذاری تطبیقی
            img_np = cv2.adaptiveThreshold(
                img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 2
            )
            
            # عملیات مورفولوژیکی
            kernel = np.ones((2, 2), np.uint8)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            
            # اتساع اندکی
            kernel_dilate = np.ones((1, 1), np.uint8)
            img_np = cv2.dilate(img_np, kernel_dilate, iterations=1)
            
            return img_np
            
        except Exception as e:
            logger.error(f"خطای پیش‌پردازش PyTorch: {e}")
            # بازگشت به پردازش CPU
            return super().preprocess_image(image)
    
    def _preprocess_with_tensorflow(self, image: np.ndarray) -> np.ndarray:
        """
        پیش‌پردازش تصویر با استفاده از TensorFlow روی GPU
        
        Args:
            image: تصویر ورودی
            
        Returns:
            تصویر پردازش شده
        """
        try:
            # تبدیل به تصویر خاکستری در صورت نیاز
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # تبدیل به تنسور TensorFlow
            img_tensor = tf.convert_to_tensor(gray, dtype=tf.float32)
            img_tensor = tf.expand_dims(img_tensor, axis=0)  # افزودن بعد batch
            img_tensor = tf.expand_dims(img_tensor, axis=-1)  # افزودن بعد کانال
            
            # افزایش وضوح
            scale_factor = 2
            img_tensor = tf.image.resize(
                img_tensor, 
                [int(gray.shape[0] * scale_factor), int(gray.shape[1] * scale_factor)],
                method='bicubic'
            )
            
            # تبدیل به numpy برای عملیات OpenCV
            img_np = tf.squeeze(img_tensor).numpy().astype(np.uint8)
            
            # اعمال CLAHE برای کنتراست بهتر
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_np = clahe.apply(img_np)
            
            # اعمال آستانه‌گذاری تطبیقی
            img_np = cv2.adaptiveThreshold(
                img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 2
            )
            
            # عملیات مورفولوژیکی
            kernel = np.ones((2, 2), np.uint8)
            img_np = cv2.morphologyEx(img_np, cv2.MORPH_CLOSE, kernel)
            
            # اتساع اندکی
            kernel_dilate = np.ones((1, 1), np.uint8)
            img_np = cv2.dilate(img_np, kernel_dilate, iterations=1)
            
            return img_np
            
        except Exception as e:
            logger.error(f"خطای پیش‌پردازش TensorFlow: {e}")
            # بازگشت به پردازش CPU
            return super().preprocess_image(image)
    
    def extract_from_image(self, image: np.ndarray):
        """
        بازنویسی extract_from_image با نسخه بهینه‌سازی شده با GPU
        
        Args:
            image: تصویر ورودی
            
        Returns:
            اطلاعات استخراج شده از تصویر
        """
        # استفاده از پیش‌پردازش بهینه‌سازی شده با GPU
        processed_image = self.preprocess_image(image)
        
        # فراخوانی متد والد با تصویر پردازش شده
        result = super().extract_from_image(processed_image)
        
        return result
    
    def process_pdf(self, pdf_path: str):
        """
        بازنویسی process_pdf با نسخه بهینه‌سازی شده برای لینوکس
        
        Args:
            pdf_path: مسیر فایل PDF
            
        Returns:
            نتایج پردازش
        """
        logger.info(f"پردازش PDF با استخراج کننده بهینه‌سازی شده برای لینوکس: {pdf_path}")
        
        # فراخوانی متد والد
        return super().process_pdf(pdf_path)
    
    def process_multiple_pdfs(self, pdf_paths):
        """
        بازنویسی با پردازش موازی بهینه‌سازی شده برای لینوکس
        
        Args:
            pdf_paths: لیست مسیرهای PDF
            
        Returns:
            نتایج پردازش ترکیب شده
        """
        logger.info(f"پردازش {len(pdf_paths)} PDF با پردازش موازی بهینه‌سازی شده برای لینوکس")
        
        # فراخوانی متد والد
        return super().process_multiple_pdfs(pdf_paths)

    # پیاده‌سازی متدهای مورد نیاز از کد ارائه شده
    def draw_bounding_boxes(self, image, tags, jb_identifiers, mc_identifiers, cable_descriptions, spare_identifiers, tag_to_number=None):
        """
        کشیدن کادرهای مرزی دور تگ‌ها، JB‌ها، MC‌ها و کابل‌ها در تصویر.
        همچنین شماره‌های تگ را تعیین می‌کند.
        
        Args:
            image: تصویر ورودی
            tags: مجموعه تگ‌ها
            jb_identifiers: مجموعه شناسه‌های JB
            mc_identifiers: مجموعه شناسه‌های MC
            cable_descriptions: لیست توضیحات کابل
            spare_identifiers: لیست شناسه‌های SPARE
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها (اختیاری)
            
        Returns:
            تصویر حاشیه‌گذاری شده و دیکشنری نگاشت تگ‌ها به شماره‌ها
        """
        # اگر tag_to_number ارائه نشده، یک دیکشنری خالی ایجاد کن
        if tag_to_number is None:
            tag_to_number = {}
            
        # استخراج متن از تصویر با استفاده از OCR
        ocr_data = self.extract_text_with_positions(image)
        
        # مجموعه‌ای برای ردیابی مناطق پردازش شده
        processed_regions = set()
        processed_identifiers = set()
        
        # لیست تمام components برای مرتب‌سازی و نمایش
        all_components = []
        
        # ذخیره موقعیت‌های MC برای جستجوی مکانی
        mc_positions = {}
        
        # پردازش MC identifiers اول برای پیدا کردن موقعیت‌ها
        for mc in mc_identifiers:
            for i, text in enumerate(ocr_data['text']):
                text_clean = text.strip().upper()
                if text_clean == mc:
                    region_key = (ocr_data['left'][i], ocr_data['top'][i],
                                ocr_data['width'][i], ocr_data['height'][i])
                    processed_regions.add(region_key)
                    processed_identifiers.add(text_clean)
                    x, y, w, h = region_key
                    all_components.append({
                        'type': 'MC',
                        'text': mc,
                        'position': (x, y, w, h),
                        'color': (0, 255, 0)
                    })
                    
                    # ذخیره موقعیت MC برای استفاده بعدی
                    mc_positions[mc] = (x, y)
        
        # پردازش Cable descriptions
        for cd in cable_descriptions:
            if not cd:
                continue
                
            # بررسی الگوهای خاص در داده OCR
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
        cv2.putText(image, "Numbering: Tags are unique by name, Spares have individual numbers", 
                    (legend_x_pos, legend_y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # بررسی تطابق شماره تگ با شماره زوج
        is_consistent, max_tag_number, extracted_pair_number = self.check_tag_number_consistency(tag_to_number)
        
        # افزودن وضعیت تطابق به تصویر
        status_y_pos = image.shape[0] - 90
        status_x_pos = 10
        
        if is_consistent:
            status_message = f"Tag numbering OK: Max tag #{max_tag_number} matches pair number {extracted_pair_number}"
            status_color = (0, 255, 0)  # سبز برای موفقیت
        else:
            status_message = f"WARNING: Max tag #{max_tag_number} doesn't match pair number {extracted_pair_number} - CHECK NUMBERING!"
            status_color = (0, 0, 255)  # قرمز برای هشدار
        
        # افزودن پیام وضعیت با پس‌زمینه برای وضوح بیشتر
        text_size = cv2.getTextSize(status_message, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (status_x_pos - 5, status_y_pos - 20), 
                    (status_x_pos + text_size[0] + 5, status_y_pos + 5), 
                    (255, 255, 255), -1)  # پس‌زمینه سفید
        cv2.putText(image, status_message, (status_x_pos, status_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return image, tag_to_number