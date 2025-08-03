from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import tempfile
import logging
import platform
from pathlib import Path

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Import DataAnalysisModule here
from DataAnalysisModule import TagJBExtractor

def get_platform_specific_extractor(tesseract_path=None, excel_path=None):
    """
    بر اساس سیستم عامل، کلاس مناسب استخراج کننده را برمی‌گرداند
    """
    system = platform.system().lower()
    
    if system == 'linux':
        try:
            from LinuxTagJBExtractor import LinuxTagJBExtractor
            logger.info("استفاده از استخراج کننده مخصوص لینوکس با پشتیبانی از GPU")
            return LinuxTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری LinuxTagJBExtractor: {e}")
            from DataAnalysisModule import TagJBExtractor
            logger.info("استفاده از استخراج کننده عمومی")
            return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    elif system == 'windows':
        try:
            # در صورتی که پیاده‌سازی مخصوص ویندوز داشته باشید، می‌توانید اینجا import کنید
            # from WindowsTagJBExtractor import WindowsTagJBExtractor
            # logger.info("استفاده از استخراج کننده مخصوص ویندوز")
            # return WindowsTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
            
            # فعلاً از پیاده‌سازی عمومی استفاده می‌کنیم
            from DataAnalysisModule import TagJBExtractor
            logger.info("استفاده از استخراج کننده عمومی در ویندوز")
            return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده ویندوز: {e}")
            from DataAnalysisModule import TagJBExtractor
            logger.info("استفاده از استخراج کننده عمومی")
            return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    elif system == 'darwin':  # macOS
        try:
            # در صورتی که پیاده‌سازی مخصوص macOS داشته باشید، می‌توانید اینجا import کنید
            # from MacTagJBExtractor import MacTagJBExtractor
            # logger.info("استفاده از استخراج کننده مخصوص macOS")
            # return MacTagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
            
            # فعلاً از پیاده‌سازی عمومی استفاده می‌کنیم
            from DataAnalysisModule import TagJBExtractor
            logger.info("استفاده از استخراج کننده عمومی در macOS")
            return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
        except ImportError as e:
            logger.warning(f"خطا در بارگذاری استخراج کننده macOS: {e}")
            from DataAnalysisModule import TagJBExtractor
            logger.info("استفاده از استخراج کننده عمومی")
            return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)
    
    else:
        # سیستم عامل ناشناخته، از پیاده‌سازی عمومی استفاده می‌کنیم
        from DataAnalysisModule import TagJBExtractor
        logger.info(f"سیستم عامل ناشناخته '{system}'، استفاده از استخراج کننده عمومی")
        return TagJBExtractor(tesseract_path=tesseract_path, excel_path=excel_path)

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
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'نام کاربری یا رمز عبور اشتباه است'})

@app.route('/logout')
def logout():
    # حذف اطلاعات کاربر از session
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    # بررسی اینکه آیا کاربر وارد شده است یا خیر
    if 'username' not in session:
        return redirect(url_for('home'))
    # نمایش صفحه داشبورد
    return render_template('JB.html', username=session['username'])

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
        logger.error(f"خطا در دریافت اطلاعات GPU: {e}")
        system_info['gpu_error'] = str(e)
    
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
        
    try:
        # Get PDF and Excel files
        pdf_files = request.files.getlist('pdf_files')
        excel_file = request.files['excel_file']
        output_dir = request.form.get('output_path', '')
        
        # گزینه استفاده از GPU (اگر در دسترس باشد)
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        
        # دریافت الگوها از فرم
        jb_examples = request.form.get('jb_examples', '').strip()
        mc_examples = request.form.get('mc_examples', '').strip()
        spare_examples = request.form.get('spare_examples', '').strip()
        cable_examples = request.form.get('cable_examples', '').strip()
        wire_color_rule = request.form.get('wire_color_rule', '').strip()
        scr_number_rule = request.form.get('scr_number_rule', '').strip()
        
        # ایجاد مسیرهای خروجی
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ذخیره فایل‌ها
        pdf_paths = []
        for pdf in pdf_files:
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
            pdf.save(temp_path)
            pdf_paths.append(temp_path)
            logger.info(f"Saved PDF: {pdf.filename}")
        
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_file.filename)
        excel_file.save(excel_path)
        logger.info(f"Saved Excel file: {excel_file.filename}")
        
        output_excel_path = os.path.join(output_dir, 'output.xlsx')
        output_pdf_dir = os.path.join(output_dir, 'annotated_pdfs')
        os.makedirs(output_pdf_dir, exist_ok=True)
        
        # حالا که excel_path را داریم، می‌توانیم extractor را ایجاد کنیم
        logger.info("Initializing platform-specific extractor...")
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
        logger.info("Starting PDF and Excel processing...")
        unmatched_excel_tags, unmatched_pdf_tags = extractor.run_with_annotated_pdf(
            pdf_paths=pdf_paths,
            excel_path=excel_path,
            output_excel_path=output_excel_path,
            output_pdf_dir=output_pdf_dir
        )
        
        # لیست PDF های حاشیه‌نویسی شده
        annotated_pdfs = [f for f in os.listdir(output_pdf_dir) if f.startswith('annotated_')]
        
        # پاکسازی فایل‌های موقت
        for path in pdf_paths:
            os.remove(path)
        os.remove(excel_path)
        
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
                    'excel_path': output_excel_path,
                    'annotated_pdfs_dir': output_pdf_dir,
                    'annotated_pdfs': annotated_pdfs
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
        
        logger.info("Processing completed successfully")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
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
    
    print("در حال راه‌اندازی سرور...")
    print("=" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)