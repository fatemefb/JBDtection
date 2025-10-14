from DataAnalysisModule import TagJBExtractor
from logger_config import LoggerMixin
import re
import logging
import traceback
import pandas as pd
import numpy as np
import sys 
import os
# اصلاح مسیرهای import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from typing import List, Dict, Set, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

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
    
    def extract_from_image(self, image):
        """
        استخراج اطلاعات از تصویر با ثبت لاگ
        
        Args:
            image: تصویر ورودی
            
        Returns:
            نتیجه استخراج شده از تصویر (ممکن است 5 یا 6 مقدار برگرداند)
        """
        self.logger.info("Extracting information from image")
        result = super().extract_from_image(image)
        
        # بررسی تعداد مقادیر برگشتی
        if isinstance(result, tuple):
            if len(result) == 5:
                self.logger.info(f"Extracted 5 values: {len(result[0])} tags, {len(result[1])} JBs, {len(result[2])} MCs, {len(result[3])} cable descriptions, {len(result[4])} spare identifiers")
            elif len(result) == 6:
                self.logger.info(f"Extracted 6 values: {len(result[0])} tags, {len(result[1])} JBs, {len(result[2])} MCs, {len(result[3])} cable descriptions, {len(result[4])} spare identifiers, {len(result[5])} tag-to-number mappings")
            else:
                self.logger.warning(f"Unexpected number of values returned: {len(result)}")
        else:
            self.logger.warning(f"Unexpected return type from extract_from_image: {type(result)}")
        
        return result
    
    def create_annotated_pdf(self, pdf_path, output_pdf_path):
        """
        ایجاد PDF حاشیه‌نویسی شده با ثبت لاگ
        
        Args:
            pdf_path: مسیر فایل PDF ورودی
            output_pdf_path: مسیر فایل PDF خروجی
            
        Returns:
            دیکشنری شماره‌گذاری تگ‌ها
        """
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
        
    def set_wire_color_rule(self, rule):
        """
        تنظیم قانون تولید رنگ‌های سیم
        
        Args:
            rule: قانون تولید رنگ‌های سیم
        """
        try:
            logger.info(f"Setting wire color rule: {rule}")
            self.wire_color_rule = rule
            
            # تست قانون با یک نمونه
            test_colors = self.generate_mc_wire_colors(1)
            logger.info(f"Test wire colors for tag #1: {test_colors}")
        except Exception as e:
            logger.error(f"Error Setting wire color rule: {e}")

    def set_scr_number_rule(self, rule):
        """
        تنظیم قانون تولید شماره SCR
        
        Args:
            rule: قانون تولید شماره SCR
        """
        try:
            logger.info(f"Setting SCR number rule: {rule}")
            self.scr_number_rule = rule
            
            # تست قانون با یک نمونه
            test_scr = self.generate_scr_number(1)
            logger.info(f"Test SCR number for tag #1: {test_scr}")
        except Exception as e:
            logger.error(f"Error Setting SCR number rule: {e}")

    def generate_mc_wire_colors(self, tag_number):
        """
        تولید رنگ‌های سیم بر اساس شماره تگ و قانون تعریف شده
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            لیست رنگ‌های سیم
        """
        try:
            if not hasattr(self, 'wire_color_rule') or not self.wire_color_rule:
                return []
                
            # جداسازی قانون‌ها با کاما
            color_rules = [rule.strip() for rule in self.wire_color_rule.split(',')]
            
            # تولید رنگ‌ها با استفاده از قانون
            colors = []
            for rule in color_rules:
                # جایگزینی {number} با شماره تگ
                if '{number' in rule:
                    # بررسی فرمت اختیاری
                    format_match = re.search(r'\{number:([^}]+)\}', rule)
                    if format_match:
                        format_spec = format_match.group(1)
                        formatted_number = format(tag_number, format_spec)
                        color = rule.replace(format_match.group(0), formatted_number)
                    else:
                        color = rule.replace('{number}', str(tag_number))
                else:
                    # جایگزینی ساده عبارات ریاضی
                    # مثال: BK{number*2-1} -> BK1 برای tag_number=1
                    expr_match = re.search(r'\{([^}]+)\}', rule)
                    if expr_match:
                        expr = expr_match.group(1).replace('number', str(tag_number))
                        try:
                            result = eval(expr)
                            color = rule.replace(expr_match.group(0), str(result))
                        except Exception as e:
                            logger.error(f"Error evaluating expression {expr}: {e}")
                            color = rule
                    else:
                        color = rule
                
                colors.append(color)
            
            return ', '.join(colors)
        except Exception as e:
            logger.error(f"Error generating wire colors: {e}")
            return ""

    def generate_scr_number(self, tag_number):
        """
        تولید شماره SCR بر اساس شماره تگ و قانون تعریف شده
        
        Args:
            tag_number: شماره تگ
            
        Returns:
            شماره SCR
        """
        try:
            if not hasattr(self, 'scr_number_rule') or not self.scr_number_rule:
                return ''
                    
            # جایگزینی {number} با شماره تگ
            if '{number' in self.scr_number_rule:
                # بررسی فرمت اختیاری
                format_match = re.search(r'\{number:([^}]+)\}', self.scr_number_rule)
                if format_match:
                    format_spec = format_match.group(1)
                    formatted_number = format(tag_number, format_spec)
                    scr_number = self.scr_number_rule.replace(format_match.group(0), formatted_number)
                else:
                    scr_number = self.scr_number_rule.replace('{number}', str(tag_number))
            else:
                # جایگزینی ساده عبارات ریاضی
                # مثال: {number*2-1} {number*2} SCR -> "1 2 SCR" برای tag_number=1
                def replace_expr(match):
                    expr = match.group(1).replace('number', str(tag_number))
                    try:
                        result = eval(expr)
                        return str(result)
                    except Exception as e:
                        logger.error(f"Error evaluating expression {expr}: {e}")
                        return match.group(0)
                    
                scr_number = re.sub(r'\{([^}]+)\}', replace_expr, self.scr_number_rule)
                
            return scr_number
        except Exception as e:
            logger.error(f"Error generating SCR number: {e}")
            return ''

    def add_wire_colors_and_scr_to_dataframe(self, df: pd.DataFrame, tag_to_number: 'Dict[str, int]', 
                                    output_path: str, pdf_results: 'Dict[str, Dict[int, Tuple[Any, ...]]]',                       
                                    pdf_name: str = None) -> pd.DataFrame:
            """
            رنگ‌های سیم MC و شماره‌های SCR را به دیتافریم اضافه می‌کند و یک فایل اکسل جدید ایجاد می‌کند.
            استفاده مستقیم از شماره‌های تگ استخراج شده توسط bounding box.
            
            Args:
                df: دیتافریم ورودی حاوی اطلاعات تگ
                tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها از bounding box
                output_path: مسیر فایل اکسل خروجی
                pdf_results: نتایج پردازش PDF ها (دیکشنری با کلید نام PDF و مقدار نتایج صفحات)
                pdf_name: نام فایل PDF (اختیاری)
                
            Returns:
                دیتافریم به‌روزرسانی شده با ستون‌های جدید
            """
            try:
                # ایجاد دیتافریم جدید برای فایل اکسل خروجی
                new_df_data = []
                
                # بررسی ساختار pdf_results
                logger.info(f"pdf_results structure: {type(pdf_results)}")
                
                # پردازش هر PDF به صورت جداگانه
                for pdf_name, page_results_dict in pdf_results.items():
                    logger.info(f"Processing PDF: {pdf_name}")
                    
                    if page_results_dict is None:
                        logger.warning(f"page_results_dict for PDF {pdf_name} is None, skipping.")
                        continue

                    # پردازش هر صفحه از این PDF
                    for page_num, page_results in page_results_dict.items():
                        self._process_page_results(new_df_data, page_num, page_results, pdf_name, tag_to_number)
                
                # ایجاد دیتافریم جدید
                new_df = pd.DataFrame(new_df_data)
                
                # اگر دیتافریم خالی نیست، مرتب‌سازی انجام بده
                if not new_df.empty:
                    # مرتب‌سازی بر اساس نام PDF، صفحه، JB و شماره تگ
                    new_df = new_df.sort_values(by=['PDF_Name', 'Page', 'JB', 'Tag_Number'])
                    
                    # تنظیم ترتیب ستون‌های نهایی
                    column_order = [
                        'PDF_Name', 'Page', 'JB', 'MC', 'Tag/SPARE', 'Tag_Number', 
                        'Wire_Code_1', 'Wire_Code_2', 'Terminal_First_Number', 'Terminal_Second_Number','Cable_Code', 'SCR_Terminal_Number',
                        'Cable_Description', 'Type', 'Tag_Number_Status'  # اضافه کردن ستون وضعیت
                    ]
                    
                    # فقط ستون‌هایی که وجود دارند را انتخاب کن
                    available_columns = [col for col in column_order if col in new_df.columns]
                    new_df = new_df[available_columns]
                
                # ذخیره دیتافریم به عنوان فایل اکسل
                new_df.to_excel(output_path, index=False)
                
                # آمار کلی
                total_tags = len(new_df[new_df['Type'] == 'Tag'].drop_duplicates(subset=['Tag/SPARE'])) if not new_df.empty else 0
                total_spares = len(new_df[new_df['Type'] == 'SPARE'].drop_duplicates(subset=['Tag/SPARE'])) if not new_df.empty else 0
                
                # تعداد هشدارها
                warnings_count = len(new_df[new_df['Tag_Number_Status'].str.contains('WARNING', na=False)]) if 'Tag_Number_Status' in new_df.columns else 0
                
                logger.info(f"Created Excel file with {len(new_df)} rows:")
                logger.info(f"Total rows: {len(new_df)} ({total_tags} unique tags, {total_spares} unique spares)")
                logger.info(f"Number of warnings: {warnings_count}")
                logger.info(f"Output file: {output_path}")
                
                return new_df  # دیتافریم اصلی را برمی‌گردانیم
                
            except Exception as e:
                logger.error(f"Error in add_wire_colors_and_scr_to_dataframe: {e}")
                
                logger.error(traceback.format_exc())
                raise

    def extract_pair_number(self, cable_description):
        """
        استخراج عدد پشت 'Pair' از توضیحات کابل
        
        Args:
            cable_description: توضیحات کابل (مثلاً "12 pair", "12P", "12 PAIR", "12 CORE", "12C")
            
        Returns:
            عدد استخراج شده یا None اگر هیچ عددی پیدا نشد
        """
        
        
        if not cable_description:
            return None
            
        # الگوهای مختلف برای استخراج شماره زوج
        pair_patterns = [
            r'(\d+)\s*(?:pair|P|PR)',  # مثل "12 pair", "12P", "12 P"
            r'(\d+)P',                 # مثل "12P"
            r'(\d+)\s*PAIR',           # مثل "12 PAIR"
            r'(\d+)\s*CORE',           # مثل "12 CORE"
            r'(\d+)\s*C',              # مثل "12 C"
            r'(\d+)C'                  # مثل "12C"
        ]
        
        desc_str = str(cable_description).upper()  # تبدیل به رشته و حروف بزرگ برای جستجوی بهتر
        
        for pattern in pair_patterns:
            match = re.search(pattern, desc_str, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
                    
        return None

    def _process_page_results(self, new_df_data: 'List[Dict]', page_num: int, page_results: Any, 
                pdf_name: str, tag_to_number: 'Dict[str, int]'):
        """
        پردازش نتایج یک صفحه و افزودن آن‌ها به لیست داده‌های دیتافریم
        با استفاده از اطلاعات دقیق استخراج شده توسط draw_bounding_boxes
        
        Args:
            new_df_data: لیست دیکشنری‌های داده برای دیتافریم
            page_num: شماره صفحه
            page_results: نتایج پردازش صفحه
            pdf_name: نام فایل PDF
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها
        """
        try:
            logger.info(f"Processing page {page_num} of {pdf_name}")
            
            # استخراج اطلاعات از ساختار page_results
            page_tags = set()
            page_jbs = set()
            page_mcs = set()
            page_cable_descriptions = []
            page_spares = []
            page_tag_to_number = {}
            page_raw_cable_descriptions = []  # متغیر جدید

            
            # استخراج داده‌ها از page_results با توجه به ساختار آن
            if isinstance(page_results, tuple) and len(page_results) >= 5:
                page_tags = page_results[0]
                page_jbs = page_results[1]
                page_mcs = page_results[2]
                page_cable_descriptions = page_results[3]
                page_spares = page_results[4]
                page_raw_cable_descriptions = page_results[5]
                if len(page_results) >= 6:
                    page_tag_to_number = page_results[6]
            elif isinstance(page_results, dict):
                page_tags = page_results.get('tags', set())
                page_jbs = page_results.get('jbs', set())
                page_mcs = page_results.get('mcs', set())
                page_cable_descriptions = page_results.get('cable_descriptions', [])
                page_spares = page_results.get('spares', [])
                page_raw_cable_descriptions = page_results.get('raw_cable_descriptions', [])
                page_tag_to_number = page_results.get('tag_to_number', {})
            else:
                logger.warning(f"Unexpected type for page_results in PDF {pdf_name}, page {page_num}: {type(page_results)}")
                return
            
            # تبدیل به Set اگر لیست هستند
            if not isinstance(page_tags, set):
                page_tags = set(page_tags) if hasattr(page_tags, '__iter__') else set()
            if not isinstance(page_jbs, set):
                page_jbs = set(page_jbs) if hasattr(page_jbs, '__iter__') else set()
            if not isinstance(page_mcs, set):
                page_mcs = set(page_mcs) if hasattr(page_mcs, '__iter__') else set()
            
            # تعیین JB و MC اصلی برای این صفحه
            main_jb = list(page_jbs)[0] if page_jbs else ''
            main_mc = list(page_mcs)[0] if page_mcs else ''
            main_cable_desc = page_cable_descriptions[0] if page_cable_descriptions else ''
            main_raw_cable_desc = page_raw_cable_descriptions[0] if page_raw_cable_descriptions else ''
            logger.info(f"PDF: {pdf_name}, Page {page_num}: JB={main_jb}, MC={main_mc}, Tags={len(page_tags)}, Spares={len(page_spares)}, Cable Desc='{main_cable_desc}', Raw='{main_raw_cable_desc}'")
            logger.info(f"Tag numbers directly from bounding box: {page_tag_to_number}")
            # استخراج عدد پشت "Pair" از Cable_Description
            pair_number = self.extract_pair_number(main_cable_desc)
            logger.info(f"Extracted pair number from cable description: {pair_number}")
            
            # یافتن بزرگترین شماره تگ برای این JB
            max_tag_number = 0
            if page_tag_to_number:
                max_tag_number = max(page_tag_to_number.values())
            
            # مقایسه شماره زوج با بزرگترین شماره تگ
            tag_number_status = "OK"
            if pair_number is not None and max_tag_number > 0:
                if pair_number != max_tag_number:
                    tag_number_status = f"WARNING: Pair number ({pair_number}) != Max Tag number ({max_tag_number})"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
                else:
                    logger.info(f"Page {page_num}, JB {main_jb}: Pair number ({pair_number}) matches max tag number ({max_tag_number})")
            else:
                if pair_number is None:
                    tag_number_status = "WARNING: Could not extract pair number from cable description"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
                elif max_tag_number == 0:
                    tag_number_status = "WARNING: No tag numbers found"
                    logger.warning(f"Page {page_num}, JB {main_jb}: {tag_number_status}")
            
            # پردازش تگ‌های این صفحه - فقط با استفاده از اطلاعات bounding box
            for tag in page_tags:
                try:
                    # استفاده مستقیم از شماره تگ استخراج شده توسط bounding box
                    if tag in page_tag_to_number:
                        tag_num = page_tag_to_number[tag]
                        
                        # استفاده از JB و MC همین صفحه
                        jb = main_jb
                        mc = main_mc
                        cable_desc = main_cable_desc
                        raw_cable_desc = main_raw_cable_desc  # متغیر جدید

                        
                        # تولید رنگ‌های سیم و شماره‌های SCR بر اساس شماره تگ
                        tag_num_str = f"{tag_num:02d}"
                        bk_color = f"BK{tag_num_str}"
                        wt_color = f"WT{tag_num_str}"
                        
                        # تولید شماره SCR بر اساس شماره تگ
                        first_scr_num = (tag_num * 2) - 1
                        second_scr_num = tag_num * 2
                        
                        # اضافه کردن به لیست داده‌ها
                        new_df_data.append({
                            'PDF_Name': pdf_name,
                            'Page': page_num,
                            'JB': jb,
                            'MC': mc,
                            'Tag/SPARE': tag,
                            'Tag_Number': tag_num,
                            'Wire_Code_1': bk_color,
                            'Wire_Code_2': wt_color,
                            'Terminal_First_Number': str(first_scr_num),
                            'Terminal_Second_Number': str(second_scr_num),
                            'SCR_Terminal_Number': 'SCR',
                            'Cable_Code': cable_desc,
                            'Cable_Description': raw_cable_desc,  # ستون جدید                       
                            'Type': 'Tag',
                            'Tag_Number_Status': tag_number_status  # ستون برای مقایسه شماره زوج و تگ
                        })
                        
                        logger.info(f"Added Tag from {pdf_name}: {tag} -> JB: {jb}, MC: {mc}, Tag#: {tag_num}")
                    else:
                        logger.warning(f"Tag {tag} not found in bounding box tag numbers, skipping")
                        
                except Exception as e:
                    logger.error(f"Error processing tag {tag} in PDF {pdf_name}, page {page_num}: {e}")
                    
                    logger.error(traceback.format_exc())
            
            # پردازش اسپیرهای این صفحه - فقط با استفاده از اطلاعات bounding box
            for i, spare in enumerate(page_spares):
                try:
                    # ایجاد شناسه‌های مختلف برای اسپیر برای جستجو در دیکشنری
                    spare_id_options = [
                        f"SPARE_{i+1}",
                        spare,
                        f"SPARE_{page_num}_{i+1}"
                    ]
                    
                    # جستجو برای شناسه اسپیر در page_tag_to_number
                    spare_number = None
                    spare_id_used = None
                    
                    for spare_id in spare_id_options:
                        if spare_id in page_tag_to_number:
                            spare_number = page_tag_to_number[spare_id]
                            spare_id_used = spare_id
                            break
                    
                    # اگر شناسه اسپیر در page_tag_to_number پیدا نشد، از این اسپیر صرف نظر می‌کنیم
                    if spare_number is None:
                        logger.warning(f"Spare {spare} not found in bounding box tag numbers, skipping")
                        continue
                    
                    # استفاده از JB و MC همین صفحه
                    jb = main_jb
                    mc = main_mc
                    cable_desc = main_cable_desc
                    raw_cable_desc = main_raw_cable_desc 
                    
                    # تولید رنگ‌های سیم و شماره‌های SCR بر اساس شماره اسپیر
                    spare_num_str = f"{spare_number:02d}"
                    bk_color = f"BK{spare_num_str}"
                    wt_color = f"WT{spare_num_str}"
                    
                    # تولید شماره SCR بر اساس شماره اسپیر
                    first_scr_num = (spare_number * 2) - 1
                    second_scr_num = spare_number * 2
                    
                    # اضافه کردن به لیست داده‌ها
                    new_df_data.append({
                        'PDF_Name': pdf_name,
                        'Page': page_num,
                        'JB': jb,
                        'MC': mc,
                        'Tag/SPARE': spare,
                        'Tag_Number': spare_number,
                        'Wire_Code_1': bk_color,
                        'Wire_Code_2': wt_color,
                        'Terminal_First_Number': str(first_scr_num),
                        'Terminal_Second_Number': str(second_scr_num),
                        'SRC_Terminal_Number': 'SCR',
                        'Cable_Code': cable_desc,
                        'Cable_Description': raw_cable_desc,  
                        'Type': 'SPARE',
                        'Tag_Number_Status': tag_number_status  # ستون برای مقایسه شماره زوج و تگ
                    })
                    
                    logger.info(f"Added Spare from {pdf_name}: {spare} (ID: {spare_id_used}) -> JB: {jb}, MC: {mc}, Tag#: {spare_number}")
                    
                except Exception as e:
                    logger.error(f"Error processing spare {spare} in PDF {pdf_name}, page {page_num}: {e}")
                    
                    logger.error(traceback.format_exc())
                    
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_name}, page {page_num}: {e}")
            
            logger.error(traceback.format_exc())

    def _get_unique_wire_colors(self, tag: str, wire_colors: 'Dict[str, List[str]]', 
                            used_wire_colors: 'Dict[str, Dict[str, bool]]', 
                            tag_to_number: 'Dict[str, int]', as_list: bool = False) -> 'Union[str, List[str]]':
        """
        برای هر تگ، رنگ‌های سیم منحصر به فرد را برمی‌گرداند و از تکرار جلوگیری می‌کند.
        
        Args:
            tag: نام تگ
            wire_colors: دیکشنری رنگ‌های سیم
            used_wire_colors: دیکشنری رنگ‌های استفاده شده
            tag_to_number: دیکشنری شماره تگ‌ها
            as_List: اگر True باشد، لیست رنگ‌ها را برمی‌گرداند، در غیر این صورت رشته
        
        Returns:
            رشته رنگ‌های سیم با کاما جدا شده یا لیست رنگ‌ها
        """
        if tag in wire_colors:
            colors = wire_colors[tag]
            # حذف تکراری‌ها
            unique_colors = list(dict.fromkeys(colors))
            return unique_colors if as_list else ', '.join(unique_colors)
        
        # اگر رنگ برای این تگ تعریف نشده، تولید کن
        if tag not in used_wire_colors:
            used_wire_colors[tag] = {}
        
        # تولید رنگ پیش‌فرض
        tag_num = tag_to_number.get(tag, 1)
        tag_num_str = f"{tag_num:02d}"
        default_colors = [f"BK{tag_num_str}", f"WT{tag_num_str}"]
        
        return default_colors if as_list else ', '.join(default_colors)
            
    def check_tag_number_consistency(self, tag_to_number: 'Dict[str, int]') -> 'Tuple[bool, int, int]':
        """
        بررسی می‌کند که آیا بزرگترین شماره تگ با شماره زوج در توضیحات کابل مطابقت دارد یا خیر.
        
        Args:
            tag_to_number: دیکشنری نگاشت تگ‌ها به شماره‌های آن‌ها از bounding box
                
        Returns:
            Tuple of (is_consistent, max_tag_number, extracted_pair_number)
        """
        try:
            # اگر tag_to_number خالی است، مقادیر پیش‌فرض را برگردان
            if not tag_to_number:
                logger.warning("tag_to_number is empty, returning default values")
                return True, 0, 0
            
            # اطلاعات دیباگ برای بررسی مقادیر tag_to_number
            logger.debug(f"Tag to number Dictionary: {tag_to_number}")
            
            # پیدا کردن بزرگترین شماره تگ - با بررسی دقیق‌تر
            # ابتدا همه مقادیر را به عنوان لیست استخراج می‌کنیم
            all_tag_numbers = list(tag_to_number.values())
            logger.debug(f"All tag numbers: {all_tag_numbers}")
            
            # بزرگترین شماره تگ را پیدا می‌کنیم
            if all_tag_numbers:
                max_tag_number = max(all_tag_numbers)
                logger.debug(f"Maximum tag number from all tags: {max_tag_number}")
            else:
                max_tag_number = 0
                logger.warning("No tag numbers found in tag_to_number Dictionary")
            
            # استخراج شماره زوج از توضیحات کابل
            cable_descriptions = []
            
            # بررسی کنیم آیا در صفحه فعلی توضیحات کابل داریم
            if hasattr(self, 'cable_descriptions') and self.cable_descriptions:
                cable_descriptions.extend(self.cable_descriptions)
                logger.debug(f"Cable descriptions from current page: {self.cable_descriptions}")
            
            # مقدار پیش‌فرض برای شماره زوج
            extracted_pair_number = 0
            matched_description = ""
            
            # استخراج شماره زوج از توضیحات کابل
            for desc in cable_descriptions:
                if not desc:
                    continue
                
                # الگوهای مختلف برای استخراج شماره زوج
                pair_patterns = [
                    r'(\d+)\s*(?:pair|P|PR)',  # مثل "12 pair", "12P", "12 P"
                    r'(\d+)P',                 # مثل "12P"
                    r'(\d+)\s*PAIR',           # مثل "12 PAIR"
                    r'(\d+)\s*CORE',           # مثل "12 CORE"
                    r'(\d+)\s*C',              # مثل "12 C"
                    r'(\d+)C'                  # مثل "12C"
                ]
                
                desc_str = str(desc).upper()  # تبدیل به رشته و حروف بزرگ برای جستجوی بهتر
                logger.debug(f"Processing cable description: {desc_str}")
                
                for pattern in pair_patterns:
                    match = re.search(pattern, desc_str, re.IGNORECASE)
                    if match:
                        try:
                            pair_number = int(match.group(1))
                            logger.debug(f"Found potential pair number {pair_number} with pattern {pattern} in: {desc_str}")
                            if pair_number > extracted_pair_number:
                                extracted_pair_number = pair_number
                                matched_description = desc_str
                                logger.info(f"Found pair number {pair_number} in cable description: {desc_str}")
                        except ValueError:
                            logger.debug(f"Could not convert {match.group(1)} to integer")
                            continue
            
            # اگر هیچ شماره زوجی پیدا نشد، سعی کنیم از سایر منابع استخراج کنیم
            if extracted_pair_number == 0:
                # بررسی اطلاعات در page_results
                if hasattr(self, 'page_results') and self.page_results:
                    logger.debug(f"Checking page_results for cable descriptions")
                    for page_num, page_data in self.page_results.items():
                        if isinstance(page_data, tuple) and len(page_data) >= 4:
                            page_cable_descriptions = page_data[3]
                            logger.debug(f"Cable descriptions from page {page_num}: {page_cable_descriptions}")
                            for desc in page_cable_descriptions:
                                desc_str = str(desc).upper()
                                for pattern in pair_patterns:
                                    match = re.search(pattern, desc_str, re.IGNORECASE)
                                    if match:
                                        try:
                                            pair_number = int(match.group(1))
                                            logger.debug(f"Found potential pair number {pair_number} with pattern {pattern} in page {page_num}: {desc_str}")
                                            if pair_number > extracted_pair_number:
                                                extracted_pair_number = pair_number
                                                matched_description = desc_str
                                                logger.info(f"Found pair number {pair_number} in page {page_num} cable description: {desc_str}")
                                        except ValueError:
                                            continue
            
            # اگر هنوز هیچ شماره زوجی پیدا نشد، از تعداد تگ‌ها استفاده کنیم
            if extracted_pair_number == 0 and tag_to_number:
                # تعداد تگ‌ها (بدون SPARE) را به عنوان تخمینی از شماره زوج استفاده کنیم
                non_spare_tags = [tag for tag in tag_to_number.keys() if not str(tag).upper().startswith('SPARE')]
                tag_count = len(non_spare_tags)
                if tag_count > 0:
                    extracted_pair_number = tag_count
                    logger.info(f"No pair number found in cable descriptions, using tag count: {tag_count}")
            
            # بررسی تطابق - با دقت بیشتر
            if extracted_pair_number == 0:
                logger.warning("Could not extract pair number, skipping consistency check")
                return True, max_tag_number, 0
            
            # بررسی دقیق تطابق - بدون تلرانس
            is_consistent = max_tag_number == extracted_pair_number
            
            # اگر مطابقت نداشت، پیام هشدار مناسب را ثبت کن
            if not is_consistent:
                if max_tag_number > extracted_pair_number:
                    logger.warning(f"WARNING: Maximum tag number ({max_tag_number}) is GREATER than cable pair number ({extracted_pair_number}) from '{matched_description}'")
                else:
                    logger.warning(f"WARNING: Maximum tag number ({max_tag_number}) is LESS than cable pair number ({extracted_pair_number}) from '{matched_description}'")
            else:
                logger.info(f"Tag number consistency check PASSED: max_tag_number={max_tag_number}, pair_number={extracted_pair_number} from '{matched_description}'")
            
            # برای اطمینان از صحت مقادیر، اطلاعات دیباگ بیشتری اضافه می‌کنیم
            logger.debug(f"Final values: is_consistent={is_consistent}, max_tag_number={max_tag_number}, extracted_pair_number={extracted_pair_number}")
            
            return is_consistent, max_tag_number, extracted_pair_number
                
        except Exception as e:
            logger.error(f"Error checking tag number consistency: {e}")
            
            logger.error(traceback.format_exc())
            return False, 0, 0
        
    def get_processing_stats(self) -> 'Dict[str, Any]':
        """
        Return detailed statistics about the processing results.
        """
        return {
            'total_tags': len(self.all_tags),
            'matched_tags': len(self.matched_tags),
            'exact_matches': self.exact_matches,
            'similar_matches': self.similar_matches,
            'total_jbs': len(self.all_jbs),
            'processing_time': f"{self.processing_time:.2f} seconds",
            'match_rate': f"{(len(self.matched_tags) / len(self.all_tags) * 100):.1f}%" if self.all_tags else "0%",
            'exact_match_rate': f"{(self.exact_matches / len(self.matched_tags) * 100):.1f}%" if self.matched_tags else "0%",
            'unmatched_tags': len(self.all_tags - self.matched_tags),
        }
    
    def reset_stats(self):
        """
        Reset all statistics counters.
        """
        self.all_tags.clear()
        self.matched_tags.clear()
        self.all_jbs.clear()
        self.exact_matches = 0
        self.similar_matches = 0
        self.processing_time = 0