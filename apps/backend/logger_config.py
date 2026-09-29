import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from apps.backend.utils.file_naming import get_log_dir, generate_log_filename, BASE_OUTPUT_DIR

# تنظیم مسیر پایه لاگ‌ها
BASE_LOG_DIR = os.path.join(BASE_OUTPUT_DIR, "logs")

def get_logger(name: str, username: Optional[str] = None, project_name: Optional[str] = None) -> logging.Logger:
    """
    ایجاد و پیکربندی لاگر با قابلیت نوشتن در فایل و نمایش در کنسول
    
    Args:
        name: نام لاگر
        username: نام کاربری (اختیاری)
        project_name: نام پروژه (اختیاری)
        
    Returns:
        لاگر پیکربندی شده
    """
    # ایجاد لاگر
    logger = logging.getLogger(name)
    
    # اگر لاگر قبلاً پیکربندی شده، آن را برگردان
    if logger.handlers:
        return logger
    
    # تنظیم سطح لاگ
    logger.setLevel(logging.DEBUG)
    
    # فرمت لاگ
    if username:
        log_format = f'%(asctime)s - {username} - %(levelname)s - %(message)s'
    else:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    # ایجاد هندلر کنسول
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # اضافه کردن هندلر کنسول به لاگر
    logger.addHandler(console_handler)
    
    # ایجاد هندلر فایل
    try:
        # تعیین مسیر لاگ بر اساس پروژه
        if project_name:
            log_dir = get_log_dir(project_name)
            log_filename = generate_log_filename(project_name, name, "log")
        else:
            # اگر نام پروژه مشخص نشده، از مسیر پیش‌فرض استفاده کن
            today = datetime.now().strftime("%Y-%m-%d")
            log_dir = os.path.join(BASE_LOG_DIR, "general", today)
            os.makedirs(log_dir, exist_ok=True)
            log_filename = f"{name}_{today}.log"
        
        log_path = os.path.join(log_dir, log_filename)
        
        # ایجاد هندلر فایل با چرخش روزانه
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_path, when='midnight', backupCount=30
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # اضافه کردن هندلر فایل به لاگر
        logger.addHandler(file_handler)
        
        logger.info(f"Logger initialized. Log file: {log_path}")
        
    except Exception as e:
        logger.error(f"Error setting up file handler: {e}")
        # در صورت خطا، فقط از هندلر کنسول استفاده می‌کنیم
        pass
    
    return logger

class LoggerMixin:
    """
    میکسین برای اضافه کردن قابلیت لاگ به کلاس‌ها
    """
    def __init__(self, logger_name: str = None, username: str = None, project_name: str = None):
        """
        تنظیم لاگر برای کلاس
        
        Args:
            logger_name: نام لاگر (اگر None باشد، از نام کلاس استفاده می‌شود)
            username: نام کاربری (اختیاری)
            project_name: نام پروژه (اختیاری)
        """
        if logger_name is None:
            logger_name = self.__class__.__name__
        
        self.logger = get_logger(logger_name, username, project_name)