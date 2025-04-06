"""
ماژول لاگ‌گیری برای خزشگر هوشمند داده‌های حقوقی

این ماژول یک سیستم لاگ‌گیری یکپارچه برای کل پروژه فراهم می‌کند.
"""

import os
import logging
import logging.handlers
from datetime import datetime


def get_logger(name, log_level=None):
    """
    ایجاد و پیکربندی یک لاگر برای استفاده در ماژول‌های مختلف

    Args:
        name: نام لاگر (معمولاً نام ماژول)
        log_level: سطح لاگ‌گیری (اختیاری، پیش‌فرض از متغیر محیطی)

    Returns:
        logging.Logger: لاگر پیکربندی شده
    """
    # دریافت سطح لاگ‌گیری از متغیر محیطی یا پارامتر
    level_str = log_level or os.getenv('LOG_LEVEL', 'INFO')
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    level = level_map.get(level_str.upper(), logging.INFO)

    # تنظیم فرمت لاگ‌ها
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)

    # ایجاد لاگر
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # حذف هندلرهای قبلی (برای اجتناب از تکرار)
    if logger.hasHandlers():
        logger.handlers.clear()

    # افزودن هندلر کنسول
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ایجاد پوشه لاگ‌ها در صورت نیاز
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # افزودن هندلر فایل
    log_file = os.path.join(log_dir, f'crawler_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10485760, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_crawler_logger():
    """
    دریافت لاگر اختصاصی برای خزشگر با تنظیمات پیشرفته

    Returns:
        logging.Logger: لاگر پیکربندی شده خزشگر
    """
    logger = get_logger('crawler')

    # افزودن هندلر فایل اختصاصی برای لاگ‌های خزشگر
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    crawler_log_file = os.path.join(log_dir, f'crawler_requests_{datetime.now().strftime("%Y%m%d")}.log')

    request_handler = logging.handlers.RotatingFileHandler(
        crawler_log_file, maxBytes=10485760, backupCount=10, encoding='utf-8'
    )
    request_format = '%(asctime)s [%(levelname)s] REQUEST: %(message)s'
    request_handler.setFormatter(logging.Formatter(request_format))
    logger.addHandler(request_handler)

    return logger