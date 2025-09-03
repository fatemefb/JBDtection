import os
import logging
import json
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import traceback

class JsonFormatter(logging.Formatter):
    """
    فرمت‌کننده برای تبدیل پیام‌های لاگ به فرمت JSON
    """
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # اضافه کردن اطلاعات کاربر اگر موجود باشد
        if hasattr(record, 'user'):
            log_data['user'] = record.user
        
        # اضافه کردن اطلاعات خطا اگر موجود باشد
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exc().split('\n')
            }
            
        # اضافه کردن اطلاعات اضافی اگر موجود باشد
        if hasattr(record, 'extra_data'):
            log_data['extra_data'] = record.extra_data
            
        return json.dumps(log_data, ensure_ascii=False)

class UserLogFilter(logging.Filter):
    """
    فیلتر برای اضافه کردن نام کاربر به رکوردهای لاگ
    """
    def __init__(self, username=None):
        super().__init__()
        self.username = username
        
    def filter(self, record):
        record.user = self.username or "anonymous"
        return True

def setup_logger(name=None, level=logging.INFO):
    """
    تنظیم لاگر با چرخش روزانه فایل‌ها
    
    Args:
        name: نام لاگر (اگر None باشد، لاگر ریشه استفاده می‌شود)
        level: سطح لاگینگ
        
    Returns:
        لاگر تنظیم شده
    """
    # ایجاد پوشه logs اگر وجود نداشته باشد
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # مسیر فایل لاگ
    log_file = os.path.join(logs_dir, 'app.log')
    
    # دریافت لاگر
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # حذف هندلرهای قبلی اگر وجود داشته باشند
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # ایجاد هندلر با چرخش روزانه
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,  # نگهداری لاگ‌های 30 روز اخیر
        encoding='utf-8'
    )
    
    # تنظیم فرمت JSON
    file_handler.setFormatter(JsonFormatter())
    
    # اضافه کردن هندلر به لاگر
    logger.addHandler(file_handler)
    
    # اضافه کردن هندلر کنسول برای نمایش لاگ‌ها در خروجی استاندارد
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter())
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name=None, username=None):
    """
    دریافت لاگر با فیلتر کاربر
    
    Args:
        name: نام لاگر
        username: نام کاربری که به لاگ‌ها اضافه می‌شود
        
    Returns:
        لاگر با فیلتر کاربر
    """
    logger = setup_logger(name)
    
    # اضافه کردن فیلتر کاربر
    for handler in logger.handlers:
        handler.addFilter(UserLogFilter(username))
    
    return logger

class LoggerMixin:
    """
    میکسین برای اضافه کردن لاگر به کلاس‌ها
    """
    def __init__(self, *args, **kwargs):
        self.logger = get_logger(self.__class__.__name__)
        super().__init__(*args, **kwargs)