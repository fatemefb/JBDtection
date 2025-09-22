import os
import re
import zipfile
import datetime
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple

# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
logger = logging.getLogger(__name__)

# مسیر پایه برای ذخیره فایل‌ها
BASE_OUTPUT_DIR = "/home/devio/JB-outputs"

def ensure_directory_exists(directory_path: str) -> str:
    """
    اطمینان از وجود دایرکتوری و ایجاد آن در صورت نیاز
    
    Args:
        directory_path: مسیر دایرکتوری
        
    Returns:
        مسیر دایرکتوری استاندارد شده
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path

def get_project_output_dir(project_name: str) -> str:
    """
    ایجاد مسیر خروجی برای پروژه
    
    Args:
        project_name: نام پروژه
        
    Returns:
        مسیر خروجی پروژه
    """
    # استاندارد‌سازی نام پروژه (حذف کاراکترهای غیرمجاز)
    safe_project_name = re.sub(r'[^\w\-]', '_', project_name)
    
    # ایجاد مسیر خروجی
    project_dir = os.path.join(BASE_OUTPUT_DIR, safe_project_name)
    return ensure_directory_exists(project_dir)

def get_log_dir(project_name: str) -> str:
    """
    ایجاد مسیر لاگ برای پروژه
    
    Args:
        project_name: نام پروژه
        
    Returns:
        مسیر لاگ پروژه
    """
    # استاندارد‌سازی نام پروژه
    safe_project_name = re.sub(r'[^\w\-]', '_', project_name)
    
    # ایجاد مسیر لاگ با ساختار /home/devio/JB-outputs/logs/{project_name}/{date}/
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join(BASE_OUTPUT_DIR, "logs", safe_project_name, today)
    return ensure_directory_exists(log_dir)

def generate_document_filename(project_name: str, doc_type: str, extension: str, version: str = "1.0") -> str:
    """
    تولید نام فایل بر اساس استاندارد ARYAVAKAV برای اسناد
    
    فرمت: [Company/Project]-[DocType]-[YYYY-MM-DD]-vX.Y.ext
    
    Args:
        project_name: نام پروژه
        doc_type: نوع سند
        extension: پسوند فایل (بدون نقطه)
        version: نسخه سند (پیش‌فرض: 1.0)
        
    Returns:
        نام فایل استاندارد
    """
    # استاندارد‌سازی نام پروژه و نوع سند
    safe_project_name = re.sub(r'[^\w\-]', '_', project_name)
    safe_doc_type = re.sub(r'[^\w\-]', '_', doc_type)
    
    # تاریخ امروز
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # تولید نام فایل
    filename = f"Aryavakav-{safe_project_name}-{safe_doc_type}-{today}-v{version}.{extension}"
    return filename

def generate_log_filename(project_name: str, data_type: str, extension: str) -> str:
    """
    تولید نام فایل بر اساس استاندارد ARYAVAKAV برای لاگ‌ها
    
    فرمت: [project]_[datatype]_[YYYY-MM-DD].ext
    
    Args:
        project_name: نام پروژه
        data_type: نوع داده
        extension: پسوند فایل (بدون نقطه)
        
    Returns:
        نام فایل استاندارد
    """
    # استاندارد‌سازی نام پروژه و نوع داده
    safe_project_name = re.sub(r'[^\w\-]', '_', project_name)
    safe_data_type = re.sub(r'[^\w\-]', '_', data_type)
    
    # تاریخ امروز
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # تولید نام فایل
    filename = f"{safe_project_name}_{safe_data_type}_{today}.{extension}"
    return filename

def create_zip_archive(project_name: str, files_to_zip: List[str], doc_type: str = "Results") -> str:
    """
    ایجاد فایل ZIP از فایل‌های خروجی
    
    Args:
        project_name: نام پروژه
        files_to_zip: لیست مسیرهای فایل‌های مورد نظر برای فشرده‌سازی
        doc_type: نوع سند (پیش‌فرض: Results)
        
    Returns:
        مسیر فایل ZIP ایجاد شده
    """
    try:
        # ایجاد مسیر خروجی پروژه
        project_dir = get_project_output_dir(project_name)
        
        # تولید نام فایل ZIP
        zip_filename = generate_document_filename(project_name, doc_type, "zip")
        zip_path = os.path.join(project_dir, zip_filename)
        
        # ایجاد فایل ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_zip:
                if os.path.exists(file_path):
                    # فیلتر کردن فایل‌های JSON
                    if file_path.lower().endswith('.json'):
                        logger.info(f"Skipping JSON file: {file_path}")
                        continue
                        
                    # افزودن فایل با نام نسبی (بدون مسیر کامل)
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname)
                    logger.info(f"Added file to ZIP: {file_path}")
                else:
                    logger.warning(f"File not found for ZIP: {file_path}")
        
        logger.info(f"Created ZIP archive: {zip_path}")
        return zip_path
    
    except Exception as e:
        logger.error(f"Error creating ZIP archive: {e}")
        return ""

def get_download_url(file_path: str) -> str:
    """
    تبدیل مسیر فایل به URL قابل دانلود
    
    Args:
        file_path: مسیر فایل
        
    Returns:
        URL قابل دانلود
    """
    # تبدیل مسیر فایل به URL نسبی
    if file_path.startswith(BASE_OUTPUT_DIR):
        relative_path = file_path[len(BASE_OUTPUT_DIR):].lstrip('/')
        return f"/downloads/{relative_path}"
    
    # اگر مسیر فایل در مسیر پایه نیست، مسیر اصلی را برگردان
    return file_path