import os
import shutil
import logging
import platform
import subprocess
from typing import List, Dict, Any, Optional, Union
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# تنظیم لاگر
logger = logging.getLogger(__name__)

def standardize_path(path: str) -> str:
    """
    استاندارد‌سازی مسیر فایل با توجه به سیستم عامل
    
    Args:
        path: مسیر فایل
        
    Returns:
        مسیر استاندارد شده
    """
    if path:
        # تبدیل مسیر به مسیر مطلق
        path = os.path.abspath(os.path.expanduser(path))
        
        # استاندارد‌سازی جداکننده‌های مسیر
        path = os.path.normpath(path)
    
    return path

def copy_to_output_paths(files_to_copy: List[str], server_output_dir: str = None, user_output_dir: str = None) -> Dict[str, Any]:
    """
    کپی فایل‌های خروجی به مسیرهای سرور و کاربر
    
    Args:
        files_to_copy: لیست فایل‌های مورد نظر برای کپی
        server_output_dir: مسیر خروجی سرور (اختیاری)
        user_output_dir: مسیر خروجی کاربر (اختیاری)
        
    Returns:
        دیکشنری نتایج کپی
    """
    result = {
        'server_success': False,
        'server_files': [],
        'user_path_success': False,
        'user_files': [],
        'method_used': 'none'
    }
    
    # استاندارد‌سازی مسیرها
    if server_output_dir:
        server_output_dir = standardize_path(server_output_dir)
    if user_output_dir:
        user_output_dir = standardize_path(user_output_dir)
    
    # کپی به مسیر سرور
    if server_output_dir:
        try:
            os.makedirs(server_output_dir, exist_ok=True)
            server_files = []
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    dest_path = os.path.join(server_output_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    server_files.append(dest_path)
                    logger.info(f"Copied to server: {file_path} -> {dest_path}")
                else:
                    logger.warning(f"File not found for server copy: {file_path}")
            
            result['server_success'] = True
            result['server_files'] = server_files
            
        except Exception as e:
            logger.error(f"Error copying to server directory: {e}")
            result['error'] = str(e)
    
    # کپی به مسیر کاربر
    if user_output_dir:
        try:
            os.makedirs(user_output_dir, exist_ok=True)
            user_files = []
            
            # روش کپی: محلی
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    dest_path = os.path.join(user_output_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    user_files.append(dest_path)
                    logger.info(f"Copied to user path: {file_path} -> {dest_path}")
                else:
                    logger.warning(f"File not found for user path copy: {file_path}")
            
            result['user_path_success'] = True
            result['user_files'] = user_files
            result['method_used'] = 'local'
            
        except Exception as e:
            logger.error(f"Error copying to user directory: {e}")
            result['error'] = str(e)
    
    return result

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    دریافت اطلاعات فایل
    
    Args:
        file_path: مسیر فایل
        
    Returns:
        دیکشنری اطلاعات فایل
    """
    file_info = {
        'exists': False,
        'size': 0,
        'is_file': False,
        'is_dir': False,
        'permissions': '',
        'last_modified': '',
        'path': file_path
    }
    
    try:
        if os.path.exists(file_path):
            file_info['exists'] = True
            file_info['size'] = os.path.getsize(file_path)
            file_info['is_file'] = os.path.isfile(file_path)
            file_info['is_dir'] = os.path.isdir(file_path)
            
            # دریافت مجوزها
            file_info['permissions'] = oct(os.stat(file_path).st_mode)[-3:]
            
            # دریافت زمان آخرین تغییر
            import datetime
            file_info['last_modified'] = datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        file_info['error'] = str(e)
    
    return file_info

def ensure_directory_exists(directory_path: str) -> bool:
    """
    اطمینان از وجود دایرکتوری و ایجاد آن در صورت نیاز
    
    Args:
        directory_path: مسیر دایرکتوری
        
    Returns:
        نتیجه عملیات (True در صورت موفقیت)
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {directory_path}: {e}")
        return False

def delete_file(file_path: str) -> bool:
    """
    حذف فایل
    
    Args:
        file_path: مسیر فایل
        
    Returns:
        نتیجه عملیات (True در صورت موفقیت)
    """
    try:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                logger.info(f"Deleted directory: {file_path}")
                return True
        else:
            logger.warning(f"File not found for deletion: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False