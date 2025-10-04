import os
import shutil
import logging
import platform
import subprocess
from typing import List, Dict, Any, Optional, Union
import sys
# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
logger = logging.getLogger(__name__)

# اضافه کردن import برای استفاده از توابع file_naming
from utils.file_naming import get_unique_filename

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
    کپی فایل‌های خروجی به مسیرهای سرور و کاربر با ورژن‌بندی خودکار
    
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
            # اطمینان از وجود دایرکتوری مقصد
            os.makedirs(server_output_dir, exist_ok=True)
            server_files = []
            
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    # استفاده از get_unique_filename برای جلوگیری از بازنویسی فایل‌های قبلی
                    base_filename = os.path.basename(file_path)
                    unique_filename = get_unique_filename(server_output_dir, base_filename)
                    dest_path = os.path.join(server_output_dir, unique_filename)
                    
                    # بررسی یکسان نبودن مسیر مبدأ و مقصد
                    if os.path.normpath(file_path) != os.path.normpath(dest_path):
                        shutil.copy2(file_path, dest_path)
                        server_files.append(dest_path)
                        logger.info(f"Copied to server with version control: {file_path} -> {dest_path}")
                    else:
                        # اگر مسیرها یکسان هستند، فقط به لیست اضافه کن بدون کپی
                        server_files.append(file_path)
                        logger.info(f"Source and destination are the same, skipping copy: {file_path}")
                else:
                    logger.warning(f"File not found for server copy: {file_path}")
            
            result['server_success'] = len(server_files) > 0
            result['server_files'] = server_files
            
        except Exception as e:
            logger.error(f"Error copying to server directory: {e}")
            result['error'] = str(e)
    
    # کپی به مسیر کاربر
    if user_output_dir and user_output_dir != server_output_dir:
        try:
            # اطمینان از وجود دایرکتوری مقصد
            os.makedirs(user_output_dir, exist_ok=True)
            user_files = []
            
            # روش کپی: محلی
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    # استفاده از get_unique_filename برای جلوگیری از بازنویسی فایل‌های قبلی
                    base_filename = os.path.basename(file_path)
                    unique_filename = get_unique_filename(user_output_dir, base_filename)
                    dest_path = os.path.join(user_output_dir, unique_filename)
                    
                    # بررسی یکسان نبودن مسیر مبدأ و مقصد
                    if os.path.normpath(file_path) != os.path.normpath(dest_path):
                        shutil.copy2(file_path, dest_path)
                        user_files.append(dest_path)
                        logger.info(f"Copied to user path with version control: {file_path} -> {dest_path}")
                    else:
                        # اگر مسیرها یکسان هستند، فقط به لیست اضافه کن بدون کپی
                        user_files.append(file_path)
                        logger.info(f"Source and destination are the same, skipping copy: {file_path}")
                else:
                    logger.warning(f"File not found for user path copy: {file_path}")
            
            result['user_path_success'] = len(user_files) > 0
            result['user_files'] = user_files
            result['method_used'] = 'local'
            
        except Exception as e:
            logger.error(f"Error copying to user directory: {e}")
            result['error'] = str(e)
    elif user_output_dir == server_output_dir:
        # اگر مسیرهای سرور و کاربر یکسان هستند، نتایج سرور را برای کاربر هم استفاده کن
        result['user_path_success'] = result['server_success']
        result['user_files'] = result['server_files']
        result['method_used'] = 'same_as_server'
        logger.info("User output directory is the same as server directory, using server results")
    
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

def verify_file_exists_with_retries(file_path: str, max_retries: int = 3, retry_delay: int = 1) -> bool:
    """
    تأیید وجود فایل با چند بار تلاش
    
    Args:
        file_path: مسیر فایل
        max_retries: حداکثر تعداد تلاش‌ها
        retry_delay: تأخیر بین تلاش‌ها (ثانیه)
        
    Returns:
        آیا فایل وجود دارد
    """
    import time
    
    for attempt in range(max_retries):
        if os.path.exists(file_path) and os.path.isfile(file_path):
            logger.info(f"File exists after {attempt+1} attempts: {file_path}")
            return True
        
        if attempt < max_retries - 1:
            logger.warning(f"File not found (attempt {attempt+1}/{max_retries}), retrying in {retry_delay} seconds: {file_path}")
            time.sleep(retry_delay)
    
    logger.error(f"File not found after {max_retries} attempts: {file_path}")
    return False

def filter_files_for_zip(files_list: List[str], exclude_extensions: List[str] = None) -> List[str]:
    """
    فیلتر کردن فایل‌ها برای اضافه کردن به ZIP
    
    Args:
        files_list: لیست فایل‌های ورودی
        exclude_extensions: لیست پسوندهایی که باید حذف شوند (مثلاً ['.json', '.tmp'])
        
    Returns:
        لیست فایل‌های فیلتر شده
    """
    if exclude_extensions is None:
        exclude_extensions = ['.json']  # به صورت پیش‌فرض، فایل‌های JSON را حذف می‌کنیم
    
    filtered_files = []
    
    for file_path in files_list:
        # بررسی وجود فایل
        if not os.path.exists(file_path):
            logger.warning(f"File not found for filtering: {file_path}")
            continue
            
        # بررسی پسوند فایل
        _, extension = os.path.splitext(file_path.lower())
        if extension in exclude_extensions:
            logger.info(f"Excluding file with extension {extension}: {file_path}")
            continue
            
        # اضافه کردن فایل به لیست فیلتر شده
        filtered_files.append(file_path)
        logger.debug(f"File passed filtering: {file_path}")
    
    logger.info(f"Filtered {len(files_list) - len(filtered_files)} files out of {len(files_list)}")
    return filtered_files

def create_zip_with_filtered_files(zip_path: str, files_to_zip: List[str], exclude_extensions: List[str] = None) -> bool:
    """
    ایجاد فایل ZIP با فیلتر کردن فایل‌های خاص
    
    Args:
        zip_path: مسیر فایل ZIP خروجی
        files_to_zip: لیست فایل‌های مورد نظر برای فشرده‌سازی
        exclude_extensions: لیست پسوندهایی که باید حذف شوند
        
    Returns:
        نتیجه عملیات (True در صورت موفقیت)
    """
    try:
        # فیلتر کردن فایل‌ها
        filtered_files = filter_files_for_zip(files_to_zip, exclude_extensions)
        
        if not filtered_files:
            logger.warning("No files left after filtering for ZIP creation")
            return False
            
        # اطمینان از وجود دایرکتوری مقصد
        zip_dir = os.path.dirname(zip_path)
        ensure_directory_exists(zip_dir)
        
        # استفاده از get_unique_filename برای جلوگیری از بازنویسی فایل ZIP قبلی
        unique_zip_filename = get_unique_filename(zip_dir, os.path.basename(zip_path))
        unique_zip_path = os.path.join(zip_dir, unique_zip_filename)
        
        # ایجاد فایل ZIP
        import zipfile
        with zipfile.ZipFile(unique_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in filtered_files:
                # افزودن فایل با نام نسبی (بدون مسیر کامل)
                arcname = os.path.basename(file_path)
                zipf.write(file_path, arcname)
                logger.info(f"Added file to ZIP: {file_path}")
        
        logger.info(f"Created ZIP archive with {len(filtered_files)} files: {unique_zip_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating ZIP with filtered files: {e}")
        return False