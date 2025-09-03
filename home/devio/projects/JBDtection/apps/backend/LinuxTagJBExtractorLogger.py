from LinuxTagJBExtractor import LinuxTagJBExtractor
from logger_config import LoggerMixin

class LoggedLinuxTagJBExtractor(LoggerMixin, LinuxTagJBExtractor):
    """
    نسخه بهبودیافته LinuxTagJBExtractor با قابلیت لاگینگ پیشرفته
    """
    
    def __init__(self, tesseract_path=None, excel_path=None):
        # ابتدا LoggerMixin را مقداردهی می‌کنیم
        LoggerMixin.__init__(self)
        # سپس کلاس اصلی را مقداردهی می‌کنیم
        LinuxTagJBExtractor.__init__(self, tesseract_path, excel_path)
        self.logger.info("LoggedLinuxTagJBExtractor initialized")
    
    # متدهای اصلی را بازنویسی می‌کنیم تا از لاگینگ استفاده کنند
    
    def _detect_gpu(self):
        self.logger.info("Detecting GPU...")
        result = super()._detect_gpu()
        if hasattr(self, 'gpu_available') and self.gpu_available:
            self.logger.info(f"GPU detected: {self.gpu_type}")
            if hasattr(self, 'cuda_device_count'):
                self.logger.info(f"CUDA device count: {self.cuda_device_count}")
        else:
            self.logger.info("No GPU detected")
        return result
    
    def enable_gpu(self):
        self.logger.info("Enabling GPU processing")
        return super().enable_gpu()
    
    def disable_gpu(self):
        self.logger.info("Disabling GPU processing")
        return super().disable_gpu()