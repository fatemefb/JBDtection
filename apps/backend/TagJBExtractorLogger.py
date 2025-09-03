from DataAnalysisModule import TagJBExtractor
from logger_config import LoggerMixin

class LoggedTagJBExtractor(LoggerMixin, TagJBExtractor):
    """
    نسخه بهبودیافته TagJBExtractor با قابلیت لاگینگ پیشرفته
    """
    
    def __init__(self, tesseract_path=None, excel_path=None):
        # ابتدا LoggerMixin را مقداردهی می‌کنیم
        LoggerMixin.__init__(self)
        # سپس کلاس اصلی را مقداردهی می‌کنیم
        TagJBExtractor.__init__(self, tesseract_path, excel_path)
        self.logger.info("LoggedTagJBExtractor initialized")
    
    # متدهای اصلی را بازنویسی می‌کنیم تا از لاگینگ استفاده کنند
    
    def build_tag_vectors_from_excel(self, excel_path):
        self.logger.info(f"Building tag vectors from Excel: {excel_path}")
        result = super().build_tag_vectors_from_excel(excel_path)
        self.logger.info(f"Built tag vectors from Excel: {len(self.tag_patterns)} patterns found")
        return result
    
    def process_pdf(self, pdf_path):
        self.logger.info(f"Processing PDF: {pdf_path}")
        result = super().process_pdf(pdf_path)
        self.logger.info(f"Processed PDF: {pdf_path}, found {len(result)} pages with data")
        return result
    
    def process_multiple_pdfs(self, pdf_paths):
        self.logger.info(f"Processing {len(pdf_paths)} PDFs")
        result = super().process_multiple_pdfs(pdf_paths)
        self.logger.info(f"Processed {len(pdf_paths)} PDFs, found {len(result)} pages with data")
        return result
    
    def create_annotated_pdf(self, pdf_path, output_pdf_path):
        self.logger.info(f"Creating annotated PDF: {pdf_path} -> {output_pdf_path}")
        result = super().create_annotated_pdf(pdf_path, output_pdf_path)
        self.logger.info(f"Created annotated PDF with {len(result)} tagged elements")
        return result
    
    def run_with_annotated_pdf(self, pdf_paths, excel_path, output_excel_path, output_pdf_dir):
        self.logger.info(f"Running full process with annotated PDFs")
        self.logger.info(f"Input: {len(pdf_paths)} PDFs, Excel: {excel_path}")
        self.logger.info(f"Output: Excel: {output_excel_path}, PDF dir: {output_pdf_dir}")
        
        result = super().run_with_annotated_pdf(
            pdf_paths, excel_path, output_excel_path, output_pdf_dir
        )
        
        unmatched_excel_tags, unmatched_pdf_tags = result
        self.logger.info(f"Process completed. Unmatched Excel tags: {len(unmatched_excel_tags)}, Unmatched PDF tags: {len(unmatched_pdf_tags)}")
        return result