from PIL import Image
from typing import Dict, List, Tuple
import os
from dataclasses import dataclass, asdict
import json

@dataclass
class ImageSection:
    JB: str
    tags: List[str]

class DiagramProcessor:
    def __init__(self):
        """
        تنظیم نسبت‌های نسبی برای مناطق مختلف تصویر
        """
        # نسبت‌های نسبی برای JB (x_start%, x_end%, y_start%, y_end%)
        self.jb_relative = (0.35, 0.65, 0.05, 0.2)
        
        # نسبت‌های نسبی برای TAG ها
        self.left_tags_relative = (0.05, 0.3, 0.1, 0.9)  # سمت چپ
        self.right_tags_relative = (0.7, 0.95, 0.1, 0.9)  # سمت راست
        
        self.output_dir = "output"
        self.image_database: Dict[str, ImageSection] = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def get_absolute_coordinates(self, img_size: Tuple[int, int], relative_coords: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """
        تبدیل مختصات نسبی به مختصات مطلق
        """
        width, height = img_size
        x_start, x_end, y_start, y_end = relative_coords
        return (
            int(width * x_start),
            int(height * y_start),
            int(width * x_end),
            int(height * y_end)
        )

    def process_image(self, image_path: str) -> Dict:
        """
        پردازش تصویر و استخراج بخش‌های مختلف
        """
        try:
            # باز کردن تصویر
            img = Image.open(image_path)
            
            # تبدیل مختصات نسبی به مطلق
            jb_coords = self.get_absolute_coordinates(img.size, self.jb_relative)
            left_tags_coords = self.get_absolute_coordinates(img.size, self.left_tags_relative)
            right_tags_coords = self.get_absolute_coordinates(img.size, self.right_tags_relative)
            
            # برش بخش‌ها
            jb_section = img.crop(jb_coords)
            left_tags = img.crop(left_tags_coords)
            right_tags = img.crop(right_tags_coords)
            
            # ذخیره بخش‌ها
            base_filename = os.path.basename(image_path)
            
            jb_path = os.path.join(self.output_dir, f"JB_{base_filename}")
            left_tags_path = os.path.join(self.output_dir, f"LEFT_TAGS_{base_filename}")
            right_tags_path = os.path.join(self.output_dir, f"RIGHT_TAGS_{base_filename}")
            
            jb_section.save(jb_path)
            left_tags.save(left_tags_path)
            right_tags.save(right_tags_path)
            
            # اضافه کردن به دیتابیس
            self.image_database[jb_path] = ImageSection(
                JB=jb_path,
                tags=[left_tags_path, right_tags_path]
            )
            
            return {
                'jb': jb_path,
                'tags': [left_tags_path, right_tags_path],
                'coordinates': {
                    'jb': jb_coords,
                    'left_tags': left_tags_coords,
                    'right_tags': right_tags_coords
                }
            }
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None