from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from typing import List, Tuple, Dict, Set, Optional, Any, Union
import os
import re
import gc
import json
import math
import numpy as np
import pandas as pd
import cv2
import fitz 
import traceback
import tempfile
import platform
import shutil  
from pathlib import Path
import pytesseract
import time
from datetime import datetime  
from multiprocessing import Pool, cpu_count
import subprocess  
import tkinter as tk
from tkinter import filedialog 
from logger_config import get_logger, LoggerMixin
from TagJBExtractorLogger import LoggedTagJBExtractor
from LinuxTagJBExtractorLogger import LoggedLinuxTagJBExtractor
from DataAnalysisModule import TagJBExtractor
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static')
)

# تنظیم کلید محرمانه برای session
app.secret_key = 'jb_detection_system_secret_key'

# Configure upload folder for temporary files
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# مسیر پشتیبان‌گیری روی سرور (مسیر ثابت)
SERVER_BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
os.makedirs(SERVER_BACKUP_DIR, exist_ok=True)

# تنظیم مسیر پیش‌فرض Tesseract بر اساس سیستم عامل
system = platform.system().lower()
if system == 'windows':
    DEFAULT_TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif system == 'linux':
    # مسیرهای معمول Tesseract در لینوکس
    linux_tesseract_paths = [
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
        '/opt/tesseract/bin/tesseract'
    ]
    
    DEFAULT_TESSERACT_PATH = None
    for path in linux_tesseract_paths:
        if os.path.exists(path):
            DEFAULT_TESSERACT_PATH = path
            break
    
    if DEFAULT_TESSERACT_PATH is None:
        DEFAULT_TESSERACT_PATH = '/usr/bin/tesseract'  # مسیر پیش‌فرض اگر پیدا نشد
elif system == 'darwin':  # macOS
    DEFAULT_TESSERACT_PATH = '/usr/local/bin/tesseract'
else:
    DEFAULT_TESSERACT_PATH = 'tesseract'  # مسیر پیش‌فرض برای سایر سیستم‌های عامل

# کاربران مجاز (در یک پروژه واقعی این اطلاعات باید در دیتابیس ذخیره شوند)
VALID_USERS = {
    'admin': 'admin123',
    'user': 'user123'
}

# ایجاد لاگر برای فایل اصلی
logger = get_logger('app')



def get_platform_specific_extractor(tesseract_path=None, excel_path=None):
    """
    بر اساس سیستم عامل، کلاس مناسب استخراج کننده را برمی‌گرداند
    """
    system = platform.system().lower()
    
    if system == 'linux':
        try:
            logger.info("استفاده از استخراج کننده مخصوص لینوکس با پشتیبانی از GPU و قابلیت لاگینگ")
            return LoggedLinuxTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری LoggedLinuxTagJBExtractor: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
       
    elif system == 'windows':
        try:
            # در صورتی که پیاده‌سازی مخصوص ویندوز داشته باشید، می‌توانید اینجا import کنید
            # from WindowsTagJBExtractorLogger import LoggedWindowsTagJBExtractor
            # logger.info("استفاده از استخراج کننده مخصوص ویندوز با قابلیت لاگینگ")
            # return LoggedWindowsTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
            
            # فعلاً از پیاده‌سازی عمومی استفاده می‌کنیم
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ در ویندوز")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده ویندوز: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    
    elif system == 'darwin':  # macOS
        try:
            # در صورتی که پیاده‌سازی مخصوص macOS داشته باشید، می‌توانید اینجا import کنید
            # from MacTagJBExtractorLogger import LoggedMacTagJBExtractor
            # logger.info("استفاده از استخراج کننده مخصوص macOS با قابلیت لاگینگ")
            # return LoggedMacTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
            
            # فعلاً از پیاده‌سازی عمومی استفاده می‌کنیم
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ در macOS")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده macOS: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    else:
        # سیستم عامل ناشناخته، از پیاده‌سازی عمومی استفاده می‌کنیم
        logger.info(f"سیستم عامل ناشناخته '{system}'، استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
        return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)

def standardize_path(path: str) -> str:
    """
    استانداردسازی مسیر با حفظ فرمت اصلی (ویندوزی یا لینوکسی)
    
    Args:
        path: مسیر ورودی
        
    Returns:
        مسیر استاندارد شده با همان فرمت اصلی
    """
    if not path:
        return ""
    
    # تمیز کردن مسیر
    path = path.strip()
    
    # تشخیص نوع مسیر (ویندوزی یا لینوکسی)
    windows_path = is_windows_path(path)
    
    # استانداردسازی مسیر با استفاده از os.path.normpath
    normalized_path = os.path.normpath(path)
    
    # اگر مسیر ویندوزی بود، اطمینان حاصل کن که با فرمت ویندوزی برگردانده شود
    if windows_path:
        # تبدیل اسلش‌های لینوکسی به بک‌اسلش ویندوزی
        normalized_path = normalized_path.replace('/', '\\')
    else:
        # تبدیل بک‌اسلش‌های ویندوزی به اسلش لینوکسی
        normalized_path = normalized_path.replace('\\', '/')
    
    return normalized_path

def is_windows_path(path: str) -> bool:
    """
    تشخیص اینکه آیا مسیر ورودی در فرمت ویندوزی است یا خیر
    
    Args:
        path: مسیر ورودی
        
    Returns:
        True اگر مسیر ویندوزی باشد، False در غیر این صورت
    """
    # مسیر ویندوزی معمولاً شامل بک‌اسلش یا درایو (مثل C:) است
    return '\\' in path or (':' in path and '/' not in path) or path.startswith('//') or path.startswith('\\\\')


def create_samba_share(windows_path):
    """
    ایجاد یک اشتراک سمبا برای دسترسی به مسیر ویندوزی از لینوکس
    
    این تابع یک پوشه در لینوکس ایجاد می‌کند و آن را به عنوان یک اشتراک سمبا
    تنظیم می‌کند تا کلاینت‌های ویندوزی بتوانند به آن دسترسی داشته باشند.
    
    Args:
        windows_path: مسیر ویندوزی که می‌خواهیم به آن دسترسی داشته باشیم
        
    Returns:
        tuple: (linux_path, share_name) - مسیر لینوکسی و نام اشتراک
    """
    if platform.system().lower() == 'windows':
        # در ویندوز نیازی به ایجاد اشتراک نیست
        return windows_path, None
    
    try:
        # ایجاد یک نام منحصر به فرد برای اشتراک
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        share_name = f"jbdetection_share_{timestamp}"
        
        # ایجاد یک پوشه در لینوکس برای اشتراک گذاری
        linux_path = os.path.join(tempfile.gettempdir(), share_name)
        os.makedirs(linux_path, exist_ok=True)
        
        # تنظیم اشتراک سمبا
        smb_conf_path = "/etc/samba/smb.conf"
        
        # بررسی دسترسی به فایل پیکربندی سمبا
        if not os.access(smb_conf_path, os.W_OK):
            logger.warning(f"دسترسی نوشتن به {smb_conf_path} وجود ندارد. از مسیر موقت استفاده می‌شود.")
            # استفاده از یک فایل پیکربندی موقت
            smb_conf_path = os.path.join(tempfile.gettempdir(), "smb.conf")
        
        # ایجاد پیکربندی اشتراک
        share_config = f"""
[{share_name}]
   path = {linux_path}
   browseable = yes
   read only = no
   guest ok = yes
   create mask = 0777
   directory mask = 0777
"""
        
        # افزودن اشتراک به پیکربندی سمبا
        with open(smb_conf_path, 'a') as f:
            f.write(share_config)
        
        # راه‌اندازی مجدد سرویس سمبا
        try:
            subprocess.run(['systemctl', 'restart', 'smbd'], check=True)
            logger.info(f"سرویس سمبا با موفقیت راه‌اندازی مجدد شد. اشتراک {share_name} ایجاد شد.")
        except subprocess.CalledProcessError:
            logger.warning("خطا در راه‌اندازی مجدد سرویس سمبا. تلاش با روش دیگر...")
            try:
                subprocess.run(['service', 'smbd', 'restart'], check=True)
                logger.info(f"سرویس سمبا با موفقیت راه‌اندازی مجدد شد. اشتراک {share_name} ایجاد شد.")
            except subprocess.CalledProcessError:
                logger.error("خطا در راه‌اندازی مجدد سرویس سمبا.")
        
        return linux_path, share_name
        
    except Exception as e:
        logger.error(f"خطا در ایجاد اشتراک سمبا: {e}")
        # در صورت خطا، فقط یک پوشه موقت برمی‌گردانیم
        linux_path = os.path.join(tempfile.gettempdir(), f"jbdetection_temp_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(linux_path, exist_ok=True)
        return linux_path, None

def copy_to_output_paths(server_files: List[str], output_path: str) -> Tuple[bool, bool, str, str, str]:
    """
    کپی فایل‌های خروجی به مسیر تعیین شده توسط کاربر و همچنین به یک پوشه در سرور
    
    Args:
        server_files: لیست مسیرهای فایل در سرور
        output_path: مسیر خروجی تعیین شده توسط کاربر
        
    Returns:
        Tuple of (server_success, output_success, server_output_path, final_output_path, error_message)
    """
    try:
        logger.info(f"کپی فایل‌ها به مسیر خروجی: {output_path}")
        
        # استانداردسازی مسیر با حفظ فرمت اصلی
        output_path = standardize_path(output_path)
        
        # ایجاد مسیر خروجی در سرور برای پشتیبان‌گیری
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        server_output_path = os.path.join(SERVER_BACKUP_DIR, f"output_{timestamp}")
        os.makedirs(server_output_path, exist_ok=True)
        
        # کپی فایل‌ها به پوشه خروجی سرور
        for src_file in server_files:
            if os.path.isfile(src_file):
                filename = os.path.basename(src_file)
                dst_file = os.path.join(server_output_path, filename)
                shutil.copy2(src_file, dst_file)
                logger.info(f"فایل در سرور کپی شد: {src_file} -> {dst_file}")
        
        # مقادیر پیش‌فرض برای بازگشت
        output_success = False
        error_message = ""
        
        # تشخیص نوع مسیر ورودی (ویندوزی یا لینوکسی)
        windows_path = is_windows_path(output_path)
        
        # اگر مسیر ورودی ویندوزی است
        if windows_path:
            try:
                # تلاش برای دسترسی به مسیر ویندوزی
                # برای این مثال، فرض می‌کنیم مسیر ویندوزی قابل دسترسی نیست
                # و فقط مسیر را برمی‌گردانیم
                output_success = False
                error_message = f"مسیر ویندوزی '{output_path}' قابل دسترسی نیست. فایل‌ها فقط در سرور ذخیره شدند."
                logger.warning(error_message)
                
                # نمایش مسیر انتخاب شده توسط کاربر بدون تغییر
                final_output_path = output_path
                
            except Exception as e:
                error_message = f"خطا در کپی فایل‌ها به مسیر ویندوزی: {str(e)}"
                logger.error(error_message)
                final_output_path = output_path  # حفظ مسیر اصلی برای نمایش
        
        # اگر مسیر ورودی لینوکسی است
        else:
            try:
                # ایجاد پوشه اگر وجود ندارد
                os.makedirs(output_path, exist_ok=True)
                
                # کپی فایل‌ها به مسیر لینوکسی
                for src_file in server_files:
                    if os.path.isfile(src_file):
                        filename = os.path.basename(src_file)
                        dst_file = os.path.join(output_path, filename)
                        shutil.copy2(src_file, dst_file)
                        logger.info(f"فایل در مسیر لینوکسی کپی شد: {src_file} -> {dst_file}")
                
                output_success = True
                final_output_path = output_path  # حفظ مسیر اصلی برای نمایش
            except Exception as e:
                error_message = f"خطا در کپی فایل‌ها به مسیر لینوکسی: {str(e)}"
                logger.error(error_message)
                final_output_path = output_path  # حفظ مسیر اصلی برای نمایش
        
        return True, output_success, server_output_path, final_output_path, error_message
    
    except Exception as e:
        error_message = f"خطا در کپی فایل‌ها: {str(e)}"
        logger.error(error_message)
        return False, False, "", output_path, error_message  # برگرداندن مسیر اصلی حتی در صورت خطا

@app.route('/')
def home():
    # اگر کاربر قبلاً وارد شده باشد، مستقیماً به داشبورد هدایت می‌شود
    if 'username' in session:
        return redirect(url_for('dashboard'))
    # در غیر این صورت به صفحه ورود هدایت می‌شود
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # بررسی اعتبار نام کاربری و رمز عبور
    if username in VALID_USERS and VALID_USERS[username] == password:
        session['username'] = username
        # تنظیم لاگر با نام کاربری جدید
        global logger
        logger = get_logger('app', username)
        logger.info(f"کاربر {username} وارد سیستم شد")
        return jsonify({'status': 'success'})
    else:
        logger.warning(f"تلاش ناموفق برای ورود با نام کاربری: {username}")
        return jsonify({'status': 'error', 'message': 'نام کاربری یا رمز عبور اشتباه است'})

@app.route('/logout')
def logout():
    username = session.get('username', 'anonymous')
    # حذف اطلاعات کاربر از session
    session.pop('username', None)
    logger.info(f"کاربر {username} از سیستم خارج شد")
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    # بررسی اینکه آیا کاربر وارد شده است یا خیر
    if 'username' not in session:
        return redirect(url_for('home'))
    # نمایش صفحه داشبورد
    username = session.get('username')
    logger.info(f"کاربر {username} به داشبورد دسترسی پیدا کرد")
    return render_template('JB.html', username=username)

@app.route('/select-folder', methods=['GET'])
def select_folder():
    """
    انتخاب پوشه با استفاده از دیالوگ گرافیکی
    ابتدا از tkinter استفاده می‌کند و اگر با خطا مواجه شود، از روش‌های دیگر استفاده می‌کند
    """
    try:
        # روش اول: استفاده از tkinter
        import tkinter as tk
        from tkinter import filedialog
        
        # ایجاد پنجره اصلی و مخفی کردن آن
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  # اطمینان از اینکه دیالوگ در بالای همه پنجره‌ها باشد
        
        # باز کردن دیالوگ انتخاب پوشه
        folder_path = filedialog.askdirectory()
        
        # بستن پنجره اصلی
        root.destroy()
        
        if folder_path:
            app.logger.info(f"Folder selected: {folder_path}")
            return jsonify({
                'status': 'success',
                'folder_path': folder_path
            })
        else:
            app.logger.info("Folder selection cancelled")
            return jsonify({
                'status': 'cancelled',
                'message': 'انتخاب پوشه لغو شد'
            })
    
    except Exception as e:
        app.logger.error(f"Error in tkinter folder selection: {str(e)}")
        # روش دوم: استفاده از zenity در لینوکس
        try:
            import subprocess
            app.logger.info("Trying zenity for folder selection")
            result = subprocess.run(['zenity', '--file-selection', '--directory'], 
                                   capture_output=True, text=True)
            if result.returncode == 0:
                folder_path = result.stdout.strip()
                app.logger.info(f"Folder selected with zenity: {folder_path}")
                return jsonify({
                    'status': 'success',
                    'folder_path': folder_path
                })
            else:
                app.logger.info("Zenity folder selection cancelled")
                return jsonify({
                    'status': 'cancelled',
                    'message': 'انتخاب پوشه لغو شد'
                })
        
        except Exception as e2:
            app.logger.error(f"Error in zenity folder selection: {str(e2)}")
            # روش سوم: استفاده از kdialog در KDE
            try:
                app.logger.info("Trying kdialog for folder selection")
                result = subprocess.run(['kdialog', '--getexistingdirectory'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    folder_path = result.stdout.strip()
                    app.logger.info(f"Folder selected with kdialog: {folder_path}")
                    return jsonify({
                        'status': 'success',
                        'folder_path': folder_path
                    })
                else:
                    app.logger.info("KDialog folder selection cancelled")
                    return jsonify({
                        'status': 'cancelled',
                        'message': 'انتخاب پوشه لغو شد'
                    })
            
            except Exception as e3:
                app.logger.error(f"Error in kdialog folder selection: {str(e3)}")
                # همه روش‌ها با شکست مواجه شدند
                error_message = f"Could not open folder selection dialog. Errors: tkinter: {str(e)}, zenity: {str(e2)}, kdialog: {str(e3)}"
                app.logger.error(error_message)
                return jsonify({
                    'status': 'error',
                    'message': error_message
                })

@app.route('/system-info')
def system_info():
    """
    ارائه اطلاعات سیستم و GPU به کاربر
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    
    system_info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'tesseract_path': DEFAULT_TESSERACT_PATH
    }
    
    # بررسی وضعیت GPU
    try:
        extractor = get_platform_specific_extractor(tesseract_path=DEFAULT_TESSERACT_PATH)
        
        # اگر استخراج کننده مخصوص لینوکس باشد، اطلاعات GPU را اضافه می‌کنیم
        if hasattr(extractor, 'gpu_available'):
            system_info['gpu_available'] = extractor.gpu_available
            if extractor.gpu_available:
                system_info['gpu_type'] = extractor.gpu_type
                if extractor.gpu_type == "NVIDIA" and hasattr(extractor, 'cuda_device_count'):
                    system_info['cuda_device_count'] = extractor.cuda_device_count
    except Exception as e:
        logger.error(f"خطا در دریافت اطلاعات GPU: {e}", extra={'user': username})
        system_info['gpu_error'] = str(e)
    
    logger.info(f"کاربر {username} اطلاعات سیستم را درخواست کرد", extra={'system_info': system_info})
    
    return jsonify({
        'status': 'success',
        'system_info': system_info
    })


@app.route('/process', methods=['POST'])
def process_files():
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    logger.info(f"کاربر {username} درخواست پردازش فایل‌ها را ارسال کرد")
        
    try:
        # Get PDF and Excel files
        pdf_files = request.files.getlist('pdf_files')
        excel_file = request.files['excel_file']
        output_dir = request.form.get('output_path')
        
        # اعتبارسنجی مسیر خروجی
        if not output_dir:
            logger.warning(f"کاربر {username} مسیر خروجی معتبری وارد نکرد")
            return jsonify({
                'status': 'error',
                'message': 'لطفاً یک مسیر خروجی معتبر وارد کنید'
            }), 400
        
        # ایجاد یک مسیر منحصر به فرد برای پشتیبان‌گیری در سرور
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        server_output_dir = os.path.join(SERVER_BACKUP_DIR, f"{username}_{timestamp}")
        os.makedirs(server_output_dir, exist_ok=True)
        logger.info(f"مسیر پشتیبان‌گیری در سرور ایجاد شد: {server_output_dir}")
        
        # استانداردسازی مسیر خروجی با حفظ فرمت اصلی
        display_output_dir = standardize_path(output_dir)
        
        # حفظ مسیر اصلی برای نمایش به کاربر
        original_output_dir = output_dir

        # گزینه استفاده از GPU (اگر در دسترس باشد)
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        
        # دریافت الگوها از فرم
        jb_examples = request.form.get('jb_examples', '').strip()
        mc_examples = request.form.get('mc_examples', '').strip()
        spare_examples = request.form.get('spare_examples', '').strip()
        cable_examples = request.form.get('cable_examples', '').strip()
        wire_color_rule = request.form.get('wire_color_rule', '').strip()
        scr_number_rule = request.form.get('scr_number_rule', '').strip()
        
        # ذخیره فایل‌ها در سرور
        pdf_paths = []
        for pdf in pdf_files:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(temp_path)
            pdf_paths.append(temp_path)
            logger.info(f"فایل PDF ذخیره شد: {pdf.filename} در مسیر {temp_path}")
        
        # ذخیره فایل اکسل
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(excel_path)
        logger.info(f"فایل Excel ذخیره شد: {excel_file.filename} در مسیر {excel_path}")
        
        # تنظیم مسیرهای خروجی در سرور
        server_output_excel_path = os.path.join(server_output_dir, 'output.xlsx')
        server_output_pdf_dir = os.path.join(server_output_dir, 'annotated_pdfs')
        
        # ایجاد پوشه PDF های حاشیه‌نویسی شده در سرور
        os.makedirs(server_output_pdf_dir, exist_ok=True)
        
        # حالا که excel_path را داریم، می‌توانیم extractor را ایجاد کنیم
        logger.info("در حال راه‌اندازی استخراج کننده مناسب برای سیستم عامل...")
        extractor = get_platform_specific_extractor(
            tesseract_path=DEFAULT_TESSERACT_PATH,
            excel_path=excel_path
        )
        
        # تنظیم الگوها در extractor
        if hasattr(extractor, 'set_patterns'):
            extractor.set_patterns(
                jb_examples=jb_examples,
                mc_examples=mc_examples,
                spare_examples=spare_examples,
                cable_examples=cable_examples,
                wire_color_rule=wire_color_rule,
                scr_number_rule=scr_number_rule
            )
        
        # نمایش اطلاعات GPU اگر در دسترس باشد
        gpu_info = {}
        if hasattr(extractor, 'gpu_available'):
            gpu_info['gpu_available'] = extractor.gpu_available
            if extractor.gpu_available:
                gpu_info['gpu_type'] = extractor.gpu_type
                if use_gpu:
                    logger.info(f"پردازش با استفاده از {extractor.gpu_type} GPU فعال شد")
                    if hasattr(extractor, 'enable_gpu'):
                        extractor.enable_gpu()
                else:
                    logger.info("پردازش GPU غیرفعال شده است (توسط کاربر)")
        
        # پردازش فایل‌ها در سرور
        logger.info(f"شروع پردازش {len(pdf_paths)} فایل PDF و Excel...")
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=server_output_excel_path,
            output_pdf_dir=server_output_pdf_dir
        )
        
        # لیست فایل‌های خروجی در سرور
        server_output_files = [server_output_excel_path]
        annotated_pdfs = []
        
        for f in os.listdir(server_output_pdf_dir):
            if f.startswith('annotated_'):
                pdf_path = os.path.join(server_output_pdf_dir, f)
                server_output_files.append(pdf_path)
                annotated_pdfs.append(f)
        
        # کپی فایل‌های خروجی به مسیر ویندوزی کلاینت و همچنین به یک پوشه در سرور
        server_copy_success, client_copy_success, server_output_path, client_output_path, error_message = copy_to_output_paths(
            server_files=server_output_files,
            output_path=output_dir  # اصلاح شد: windows_output_dir به output_dir تغییر کرد
        )
        
        # پاکسازی فایل‌های موقت
        for path in pdf_paths:
            os.remove(path)
        os.remove(excel_path)
        
        # تنظیم مسیرهای نمایشی (همیشه ویندوزی)
        display_output_excel_path = os.path.join(display_output_dir, 'output.xlsx')
        display_output_pdf_dir = os.path.join(display_output_dir, 'annotated_pdfs')

        # آماده‌سازی پاسخ
        response = {
            'status': 'success',
            'message': 'Processing completed successfully',
            'details': {
                'input_files': {
                    'pdf_count': len(pdf_paths),
                    'pdf_names': [os.path.basename(p) for p in pdf_paths],
                    'excel_file': excel_file.filename
                },
                'output_files': {
                    # مسیرهای ویندوزی برای نمایش به کاربر
                    'excel_path': display_output_excel_path,
                    'annotated_pdfs_dir': display_output_pdf_dir,
                    'annotated_pdfs': annotated_pdfs
                },
                'results': {
                    'unmatched_excel_tags': unmatched_excel_tags,
                    'unmatched_pdf_tags': unmatched_pdf_tags,
                    'unmatched_excel_count': len(unmatched_excel_tags),
                    'unmatched_pdf_count': len(unmatched_pdf_tags)
                },
                'system': {
                    # همیشه ویندوز نمایش دهید حتی در سرور لینوکس
                    'platform': "Windows",
                    'gpu_info': gpu_info
                },
                'backup': {
                    'server_backup_path': server_output_dir,
                    'server_output_path': server_output_path,
                    'client_copy_success': client_copy_success,
                    'user_output_dir': original_output_dir,  
                    'display_output_dir': display_output_dir
                }
            }
        }
        
        # اضافه کردن پیام خطا اگر کپی به کلاینت ناموفق بود
        if not client_copy_success and error_message:
            response['details']['backup']['client_error'] = error_message
        
        logger.info(f"پردازش با موفقیت به پایان رسید", extra={'user': username, 'results': {
            'unmatched_excel_count': len(unmatched_excel_tags),
            'unmatched_pdf_count': len(unmatched_pdf_tags)
        }})
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"خطا در پردازش فایل‌ها: {str(e)}", extra={'user': username})
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': {
                'error_type': type(e).__name__,
                'error_description': str(e)
            }
        }), 500

if __name__ == '__main__':
    # Print startup message
    print("=" * 50)
    print("JB Detection System")
    print("=" * 50)
    print(f"سیستم عامل: {platform.system()}")
    print(f"مسیر Tesseract: {DEFAULT_TESSERACT_PATH}")
    
    # ایجاد پوشه پشتیبان‌گیری اگر وجود ندارد
    os.makedirs(SERVER_BACKUP_DIR, exist_ok=True)
    print(f"مسیر پشتیبان‌گیری: {SERVER_BACKUP_DIR}")
    
    # بررسی وضعیت GPU
    try:
        extractor = get_platform_specific_extractor(tesseract_path=DEFAULT_TESSERACT_PATH)
        if hasattr(extractor, 'gpu_available') and extractor.gpu_available:
            print(f"GPU در دسترس: {extractor.gpu_type}")
            if extractor.gpu_type == "NVIDIA":
                print(f"تعداد دستگاه‌های CUDA: {extractor.cuda_device_count}")
        else:
            print("GPU در دسترس نیست، از پردازش CPU استفاده می‌شود")
    except Exception as e:
        print(f"خطا در بررسی وضعیت GPU: {e}")
    
    logger.info("سرور راه‌اندازی شد")
    print("در حال راه‌اندازی سرور...")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)