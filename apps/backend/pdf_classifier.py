import os
import random
import logging
import numpy as np
from typing import Tuple, List
from PIL import Image, ImageOps
import fitz  # PyMuPDF
from tensorflow import keras

logger = logging.getLogger(__name__)

class PDFClassifier:
    """دسته‌بندی هوشمند فایل‌های PDF بدون نیاز به ابزارهای جانبی"""
    
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.model = None
        self.class_names = []
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading Keras model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path, compile=False)
            
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                # تمیز کردن نام کلاس‌ها و تبدیل به حروف کوچک
                self.class_names = [line.strip().split(' ', 1)[-1].lower() for line in f.readlines()]
            logger.info(f"✅ Model loaded successfully. Classes: {self.class_names}")
        except Exception as e:
            logger.error(f"❌ Error loading classification model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        return (image_array.astype(np.float32) / 127.5) - 1
    
    def predict_image(self, image: Image.Image) -> Tuple[str, float]:
        preprocessed = self._preprocess_image(image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = preprocessed
        
        prediction = self.model.predict(data, verbose=0)
        index = np.argmax(prediction)
        return self.class_names[index], float(prediction[0][index])
    
    def sample_pages_smart(self, pdf_path: str, num_samples: int = 3) -> List[Image.Image]:
        """نمونه برداری هوشمند و فوق‌سریع از صفحات PDF با fitz"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return []
                
            sampled_page_numbers = []
            if total_pages <= num_samples:
                sampled_page_numbers = list(range(total_pages))
            else:
                mid_point = total_pages // 2
                sampled_page_numbers.append(random.choice(range(0, mid_point)))
                sampled_page_numbers.extend(random.sample(range(mid_point, total_pages), 2))
            
            sampled_page_numbers.sort()
            sampled_images = []
            
            for p_num in sampled_page_numbers:
                page = doc[p_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                sampled_images.append(img)
                
            doc.close()
            return sampled_images
        except Exception as e:
            logger.error(f"❌ Error during smart sampling on {pdf_path}: {e}")
            return []

    def classify_pdf(self, pdf_path: str) -> str:
        """بر اساس رای‌گیری اکثریت، مشخص می‌کند سند نقشه است یا جدول"""
        sampled_images = self.sample_pages_smart(pdf_path, num_samples=3)
        if not sampled_images:
            return "diagrams"  # حالت پیش‌فرض در صورت بروز خطا
            
        vote_counts = {}
        for img in sampled_images:
            label, _ = self.predict_image(img)
            vote_counts[label] = vote_counts.get(label, 0) + 1
            
        # برگرداندن کلاسی که بیشترین رای را آورده است
        final_label = max(vote_counts, key=vote_counts.get)
        return final_label