from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_from_directory, Response
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
import zipfile
from datetime import datetime  
from multiprocessing import Pool, cpu_count
import subprocess  
import tkinter as tk
import sys
import threading
import uuid

# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tkinter import filedialog 
from logger_config import get_logger, LoggerMixin
from TagJBExtractorLogger import LoggedTagJBExtractor
from LinuxTagJBExtractorLogger import LoggedLinuxTagJBExtractor
from DataAnalysisModule import TagJBExtractor
from werkzeug.utils import secure_filename
from apps.backend.utils.file_naming import (
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

# تعریف مسیر پایه برای فایل‌های خروجی در کانتینر
OUTPUT_DIRS = {
    "v1": "/home/devio/JB-outputs",
    "v2": "/home/devio/JB-outputs"
}

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

# Dictionary to store processing jobs
processing_jobs = {}

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

# تابع پردازش در پس‌زمینه
def background_processing_job(job_id, pdf_paths, excel_path, project_name, jb_examples, mc_examples, 
                             spare_examples, cable_examples, wire_color_rule, scr_number_rule):
    try:
        extractor = get_platform_specific_extractor(
            tesseract_path=DEFAULT_TESSERACT_PATH,
            excel_path=excel_path
        )

        # تنظیم الگوها اگر نیاز باشد
        if hasattr(extractor, 'set_patterns'):
            extractor.set_patterns(
                jb_examples=jb_examples,
                mc_examples=mc_examples,
                spare_examples=spare_examples,
                cable_examples=cable_examples,
                wire_color_rule=wire_color_rule,
                scr_number_rule=scr_number_rule
            )

        # ایجاد دایرکتوری خروجی پروژه
        project_output_dir = get_project_output_dir(project_name)
        annotated_pdf_dir = os.path.join(project_output_dir, "annotated_pdfs")
        os.makedirs(annotated_pdf_dir, exist_ok=True)

        total_files = len(pdf_paths)
        results = []

        for idx, pdf_path in enumerate(pdf_paths):
            unmatched_excel, unmatched_pdf = extractor.run_with_annotated_pdf(
                pdf_paths=[pdf_path],
                excel_path=excel_path,
                output_excel_path=generate_document_filename(project_name, "Excel", "xlsx"),
                output_pdf_dir=annotated_pdf_dir
            )
            results.append({
                "pdf": os.path.basename(pdf_path),
                "unmatched_excel": unmatched_excel,
                "unmatched_pdf": unmatched_pdf
            })

            # بروزرسانی progress
            processing_jobs[job_id]["progress"] = int((idx + 1) / total_files * 100)

        processing_jobs[job_id]["status"] = "done"
        processing_jobs[job_id]["results"] = results

    except Exception as e:
        processing_jobs[job_id]["status"] = "error"
        processing_jobs[job_id]["results"] = str(e)

    finally:
        # پاکسازی فایل‌های موقت
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(excel_path):
            os.remove(excel_path)

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
        
        # ایجاد یک job_id منحصر به فرد
        job_id = str(uuid.uuid4())
        
        # ایجاد یک ورودی در دیکشنری processing_jobs
        processing_jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "results": None
        }
        
        # شروع یک thread برای پردازش در پس‌زمینه
        thread = threading.Thread(
            target=background_processing_job,
            args=(job_id, pdf_paths, excel_path, project_name, jb_examples, mc_examples, 
                  spare_examples, cable_examples, wire_color_rule, scr_number_rule)
        )
        thread.daemon = True
        thread.start()
        
        # پاسخ اولیه شامل job_id
        return jsonify({
            "status": "accepted", 
            "job_id": job_id,
            "message": "پردازش در پس‌زمینه آغاز شد"
        })
        
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

@app.route('/progress/<job_id>')
def get_progress(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    def generate():
        while job["status"] == "processing":
            yield f"data: {job['progress']}\n\n"
            time.sleep(1)
        yield f"data: {job['status']}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/job-result/<job_id>')
def get_job_result(job_id):
    job = processing_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Job not found"}), 404
    
    if job["status"] == "processing":
        return jsonify({"status": "processing", "progress": job["progress"]}), 202
    
    return jsonify({
        "status": job["status"],
        "results": job["results"]
    })
    
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
        annotated_pdfs = []
        for f in os.listdir(annotated_pdf_dir):
            if f.startswith('annotated_'):
                pdf_path = os.path.join(annotated_pdf_dir, f)
                output_files.append(pdf_path)
                annotated_pdfs.append(pdf_path)
        
        # ایجاد فایل ZIP
        zip_path = create_zip_archive(project_name, output_files)
        
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        
        # تشخیص پورت فعلی برای تعیین نسخه API
        current_port = request.host.split(':')[-1] if ':' in request.host else '5000'
        
        # ایجاد URL دانلود
        server_name = request.host.split(':')[0]
        download_url = f"http://{server_name}:{current_port}/download?file={project_name}/{os.path.basename(zip_path)}"
        
        # آماده‌سازی پاسخ
        response = {
            "status": "success",
            "message": "Processing completed successfully",
            "details": {
                "output_files": {
                    "excel_path": output_excel_path,
                    "annotated_pdfs": annotated_pdfs,
                    "zip_path": zip_path,
                    "download_url": download_url
                },
                "results": {
                    "unmatched_pdf_tags": unmatched_pdf_tags,
                    "unmatched_excel_tags": unmatched_excel_tags
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

@app.route('/api/status', methods=['GET'])
def api_status():
    """
    API endpoint برای بررسی وضعیت سرور
    """
    try:
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
            
        return jsonify({
            'status': 'online',
            'version': version,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/download', methods=['GET'])
def download_file():
    """
    دانلود فایل با مسیر نسبی مشخص شده
    
    پارامترها:
        file: مسیر نسبی فایل برای دانلود (نسبت به دایرکتوری خروجی)
        
    Returns:
        فایل برای دانلود
    """
    try:
        file_path = request.args.get('file')
        
        if not file_path:
            return jsonify({"error": "No file path provided"}), 400
        
        # حذف اسلش اضافی از ابتدای مسیر
        if file_path.startswith('/'):
            file_path = file_path[1:]
        
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        
        # مسیر کامل فایل در سیستم میزبان
        base_dir = OUTPUT_DIRS[version]
        full_path = os.path.join(base_dir, file_path)
        
        # بررسی امنیتی: اطمینان از اینکه فایل درخواستی در مسیر مجاز قرار دارد
        abs_path = os.path.abspath(full_path)
        if not abs_path.startswith(base_dir):
            return jsonify({"error": "Access denied"}), 403
        
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return jsonify({"error": f"File not found: {abs_path}"}), 404
        
        # تعیین نام فایل برای دانلود
        filename = os.path.basename(abs_path)
        directory = os.path.dirname(abs_path)
        
        username = session.get('username', 'anonymous')
        logger.info(f"کاربر {username} درخواست دانلود فایل {filename} را ارسال کرد")
        
        # ارسال فایل برای دانلود
        return send_from_directory(directory, filename, as_attachment=True)
        
    except Exception as e:
        username = session.get('username', 'anonymous')
        logger.error(f"خطا در دانلود فایل: {str(e)}", extra={'user': username})
        return jsonify({"error": str(e)}), 500

@app.route('/download-all-pdfs', methods=['GET'])
def download_all_pdfs():
    """
    دانلود همه فایل‌های PDF یک پروژه به صورت فشرده
    
    پارامترها:
        project: نام پروژه
        
    Returns:
        فایل ZIP حاوی همه PDF‌ها
    """
    try:
        project_name = request.args.get('project')
        
        if not project_name:
            return jsonify({"error": "No project name provided"}), 400
        
        # تعیین نسخه سرویس (v1 یا v2) بر اساس پورت درخواست
        version = "v1"
        if request.host.endswith(':5001'):
            version = "v2"
        
        # مسیر دایرکتوری پروژه
        base_dir = OUTPUT_DIRS[version]
        project_dir = os.path.join(base_dir, project_name)
        
        if not os.path.exists(project_dir) or not os.path.isdir(project_dir):
            return jsonify({"error": f"Project directory not found: {project_dir}"}), 404
        
        # یافتن همه فایل‌های PDF در دایرکتوری پروژه
        pdf_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        if not pdf_files:
            return jsonify({"error": "No PDF files found for this project"}), 404
        
        # ایجاد فایل ZIP موقت
        zip_filename = f"{project_name}_PDFs.zip"
        zip_path = os.path.join(base_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pdf_file in pdf_files:
                # افزودن فایل با نام نسبی (بدون مسیر کامل)
                arcname = os.path.basename(pdf_file)
                zipf.write(pdf_file, arcname)
        
        username = session.get('username', 'anonymous')
        logger.info(f"کاربر {username} درخواست دانلود همه PDF های پروژه {project_name} را ارسال کرد")
        
        # ارسال فایل ZIP برای دانلود
        return send_from_directory(base_dir, zip_filename, as_attachment=True)
        
    except Exception as e:
        username = session.get('username', 'anonymous')
        logger.error(f"خطا در دانلود همه PDF ها: {str(e)}", extra={'user': username})
        return jsonify({"error": str(e)}), 500

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