from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory
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
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
from tkinter import filedialog 
from logger_config import get_logger, LoggerMixin
from TagJBExtractorLogger import LoggedTagJBExtractor
from LinuxTagJBExtractorLogger import LoggedLinuxTagJBExtractor
from DataAnalysisModule import TagJBExtractor
from werkzeug.utils import secure_filename
from file_naming import (
    BASE_OUTPUT_DIR,
    get_project_output_dir,
    get_log_dir,
    generate_document_filename,
    generate_log_filename,
    create_zip_archive,
    get_download_url
)

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

# اطمینان از وجود دایرکتوری پایه
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

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
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ در ویندوز")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده ویندوز: {e}")
            logger.info("استفاده از استخراج کننده عمومی با قابلیت لاگینگ")
            return LoggedTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    elif system == 'darwin':  # macOS
        try:
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
        'tesseract_path': DEFAULT_TESSERACT_PATH,
        'output_base_dir': BASE_OUTPUT_DIR
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
        # دریافت نام پروژه (الزامی)
        project_name = request.form.get('project_name')
        if not project_name:
            logger.warning(f"کاربر {username} نام پروژه را وارد نکرد")
            return jsonify({
                'status': 'error',
                'message': 'لطفاً نام پروژه را وارد کنید'
            }), 400
        
        # Get PDF and Excel files
        pdf_files = request.files.getlist('pdf_files')
        excel_file = request.files['excel_file']
        
        # گزینه استفاده از GPU (اگر در دسترس باشد)
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        
        # دریافت الگوها از فرم
        jb_examples = request.form.get('jb_examples', '').strip()
        mc_examples = request.form.get('mc_examples', '').strip()
        spare_examples = request.form.get('spare_examples', '').strip()
        cable_examples = request.form.get('cable_examples', '').strip()
        wire_color_rule = request.form.get('wire_color_rule', '').strip()
        scr_number_rule = request.form.get('scr_number_rule', '').strip()
        
        # ایجاد دایرکتوری خروجی پروژه
        project_output_dir = get_project_output_dir(project_name)
        logger.info(f"دایرکتوری خروجی پروژه: {project_output_dir}")
        
        # ایجاد دایرکتوری لاگ پروژه
        log_dir = get_log_dir(project_name)
        logger.info(f"دایرکتوری لاگ پروژه: {log_dir}")
        
        # تنظیم نام فایل‌های خروجی
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel_filename = generate_document_filename(project_name, "Excel", "xlsx")
        output_excel_path = os.path.join(project_output_dir, output_excel_filename)
        
        # ایجاد دایرکتوری برای PDF های حاشیه‌نویسی شده
        annotated_pdf_dir = os.path.join(project_output_dir, "annotated_pdfs")
        os.makedirs(annotated_pdf_dir, exist_ok=True)
        
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
        
        # پردازش فایل‌ها
        logger.info(f"شروع پردازش {len(pdf_paths)} فایل PDF و Excel...")
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=annotated_pdf_dir
        )
        
        # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
        unmatched_excel_filename = generate_document_filename(project_name, "UnmatchedTags", "xlsx")
        unmatched_excel_path = os.path.join(project_output_dir, unmatched_excel_filename)
        
        # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
        if hasattr(extractor, '_create_unmatched_tags_excel'):
            extractor._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
            logger.info(f"فایل Excel تگ‌های تطبیق نیافته ذخیره شد: {unmatched_excel_path}")
        
        # ایجاد فایل گزارش
        report_filename = generate_document_filename(project_name, "Report", "json")
        report_path = os.path.join(project_output_dir, report_filename)
        
        # ذخیره گزارش پردازش
        with open(report_path, 'w') as f:
            json.dump({
                'project_name': project_name,
                'processing_date': datetime.now().isoformat(),
                'user': username,
                'results': {
                    'unmatched_excel_tags': len(unmatched_excel_tags),
                    'unmatched_pdf_tags': len(unmatched_pdf_tags),
                    'pdf_count': len(pdf_paths),
                    'pdf_names': [os.path.basename(p) for p in pdf_paths]
                }
            }, f, indent=2)
        
        # لیست فایل‌های خروجی
        output_files = [output_excel_path, unmatched_excel_path, report_path]
        
        # اضافه کردن PDF های حاشیه‌نویسی شده
        for f in os.listdir(annotated_pdf_dir):
            if f.startswith('annotated_'):
                output_files.append(os.path.join(annotated_pdf_dir, f))
        
        # ایجاد فایل ZIP
        zip_path = create_zip_archive(project_name, output_files)
        
        # ایجاد URL دانلود
        download_url = get_download_url(zip_path)
        
        # پاکسازی فایل‌های موقت
        for path in pdf_paths:
            os.remove(path)
        os.remove(excel_path)
        
        # آماده‌سازی پاسخ
        response = {
            'status': 'success',
            'message': 'Processing completed successfully',
            'details': {
                'project_name': project_name,
                'input_files': {
                    'pdf_count': len(pdf_paths),
                    'pdf_names': [os.path.basename(p) for p in pdf_paths],
                    'excel_file': excel_file.filename
                },
                'output_files': {
                    'excel_path': output_excel_path,
                    'unmatched_excel_path': unmatched_excel_path,
                    'report_path': report_path,
                    'zip_path': zip_path,
                    'download_url': download_url
                },
                'results': {
                    'unmatched_excel_tags': unmatched_excel_tags,
                    'unmatched_pdf_tags': unmatched_pdf_tags,
                    'unmatched_excel_count': len(unmatched_excel_tags),
                    'unmatched_pdf_count': len(unmatched_pdf_tags)
                },
                'system': {
                    'platform': platform.system(),
                    'gpu_info': gpu_info
                }
            }
        }
        
        logger.info(f"پردازش با موفقیت به پایان رسید", extra={'user': username, 'results': {
            'unmatched_excel_count': len(unmatched_excel_tags),
            'unmatched_pdf_count': len(unmatched_pdf_tags)
        }})
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"خطا در پردازش فایل‌ها: {str(e)}", extra={'user': username})
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': {
                'error_type': type(e).__name__,
                'error_description': str(e)
            }
        }), 500

@app.route('/api/process', methods=['POST'])
def api_process():
    """
    API endpoint برای پردازش فایل‌ها
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    logger.info(f"کاربر {username} درخواست API پردازش فایل‌ها را ارسال کرد")
    
    try:
        # دریافت داده‌ها از درخواست
        data = request.json
        pdf_paths = data.get('pdf_paths', [])
        excel_path = data.get('excel_path')
        project_name = data.get('project_name')
        
        # اعتبارسنجی داده‌ها
        if not pdf_paths or not excel_path:
            return jsonify({
                'status': 'error',
                'message': 'مسیرهای PDF و Excel الزامی هستند'
            }), 400
        
        if not project_name:
            return jsonify({
                'status': 'error',
                'message': 'نام پروژه الزامی است'
            }), 400
        
        # ایجاد دایرکتوری خروجی پروژه
        project_output_dir = get_project_output_dir(project_name)
        
        # ایجاد دایرکتوری لاگ پروژه
        log_dir = get_log_dir(project_name)
        
        # تنظیم نام فایل‌های خروجی
        output_excel_filename = generate_document_filename(project_name, "Excel", "xlsx")
        output_excel_path = os.path.join(project_output_dir, output_excel_filename)
        
        # ایجاد دایرکتوری برای PDF های حاشیه‌نویسی شده
        annotated_pdf_dir = os.path.join(project_output_dir, "annotated_pdfs")
        os.makedirs(annotated_pdf_dir, exist_ok=True)
        
        # ایجاد استخراج کننده
        extractor = get_platform_specific_extractor(
            tesseract_path=DEFAULT_TESSERACT_PATH,
            excel_path=excel_path
        )
        
        # پردازش فایل‌ها
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=annotated_pdf_dir
        )
        
        # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
        unmatched_excel_filename = generate_document_filename(project_name, "UnmatchedTags", "xlsx")
        unmatched_excel_path = os.path.join(project_output_dir, unmatched_excel_filename)
        
        # ایجاد فایل اکسل برای تگ‌های تطبیق نیافته
        if hasattr(extractor, '_create_unmatched_tags_excel'):
            extractor._create_unmatched_tags_excel(unmatched_excel_tags, unmatched_pdf_tags, unmatched_excel_path)
        
        # لیست فایل‌های خروجی
        output_files = [output_excel_path, unmatched_excel_path]
        
        # اضافه کردن PDF های حاشیه‌نویسی شده
        for f in os.listdir(annotated_pdf_dir):
            if f.startswith('annotated_'):
                output_files.append(os.path.join(annotated_pdf_dir, f))
        
        # ایجاد فایل ZIP
        zip_path = create_zip_archive(project_name, output_files)
        
        # ایجاد URL دانلود
        download_url = get_download_url(zip_path)
        
        # آماده‌سازی پاسخ
        response = {
            'status': 'success',
            'message': 'پردازش با موفقیت انجام شد',
            'details': {
                'project_name': project_name,
                'output_files': {
                    'excel_path': output_excel_path,
                    'unmatched_excel_path': unmatched_excel_path,
                    'zip_path': zip_path,
                    'download_url': download_url
                },
                'results': {
                    'unmatched_excel_tags': unmatched_excel_tags,
                    'unmatched_pdf_tags': unmatched_pdf_tags,
                    'unmatched_excel_count': len(unmatched_excel_tags),
                    'unmatched_pdf_count': len(unmatched_pdf_tags)
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"خطا در API پردازش فایل‌ها: {str(e)}", extra={'user': username})
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/downloads/<path:filename>')
def download_file(filename):
    """
    دانلود فایل از مسیر خروجی
    """
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'لطفاً ابتدا وارد سیستم شوید'
        }), 401
    
    username = session.get('username')
    logger.info(f"کاربر {username} درخواست دانلود فایل {filename} را ارسال کرد")
    
    # بررسی امنیتی مسیر فایل
    safe_path = os.path.normpath(os.path.join(BASE_OUTPUT_DIR, filename))
    if not safe_path.startswith(BASE_OUTPUT_DIR):
        logger.warning(f"تلاش برای دسترسی به فایل خارج از مسیر مجاز: {filename}")
        return jsonify({
            'status': 'error',
            'message': 'دسترسی غیرمجاز'
        }), 403
    
    # بررسی وجود فایل
    if not os.path.exists(safe_path):
        logger.warning(f"فایل درخواستی یافت نشد: {filename}")
        return jsonify({
            'status': 'error',
            'message': 'فایل یافت نشد'
        }), 404
    
    # ارسال فایل
    directory = os.path.dirname(safe_path)
    file_name = os.path.basename(safe_path)
    logger.info(f"دانلود فایل {file_name} از مسیر {directory}")
    return send_from_directory(directory, file_name, as_attachment=True)

if __name__ == '__main__':
    # Print startup message
    print("=" * 50)
    print("JB Detection System")
    print("=" * 50)
    print(f"سیستم عامل: {platform.system()}")
    print(f"مسیر Tesseract: {DEFAULT_TESSERACT_PATH}")
    
    # ایجاد پوشه پشتیبان‌گیری اگر وجود ندارد
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"مسیر خروجی: {BASE_OUTPUT_DIR}")
    
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