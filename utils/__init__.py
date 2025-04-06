"""
پکیج utils:
این پکیج شامل توابع و ابزارهای کمکی برای پروژه خزشگر هوشمند داده‌های حقوقی می‌باشد.
ماژول‌های موجود در این پوشه عبارتند از:
    - logger: سیستم لاگ‌گیری یکپارچه
    - http: مدیریت درخواست‌های HTTP و تعامل با وب (شامل پردازش robots.txt و استفاده از Selenium)
    - text: توابع پردازش و نرمال‌سازی متن و استخراج اطلاعات از HTML
    - ml: ابزارها و توابع کمکی برای عملیات یادگیری ماشین (مانند بارگذاری/ذخیره مدل، ارزیابی و به‌روزرسانی مدل‌ها)
"""

from .logger import get_logger, get_crawler_logger
from .http import RequestManager, RobotsTxtParser, make_request, normalize_url
from .text import (
    clean_html,
    extract_text_from_tags,
    normalize_persian_text,
    tokenize_persian_text,
    calculate_text_hash,
    extract_main_content,
    extract_title,
    extract_date,
    extract_author,
    extract_links,
    is_similar_content
)
from .ml import MLUtils

__all__ = [
    "get_logger",
    "get_crawler_logger",
    "RequestManager",
    "RobotsTxtParser",
    "make_request",
    "normalize_url",
    "clean_html",
    "extract_text_from_tags",
    "normalize_persian_text",
    "tokenize_persian_text",
    "calculate_text_hash",
    "extract_main_content",
    "extract_title",
    "extract_date",
    "extract_author",
    "extract_links",
    "is_similar_content",
    "MLUtils"
]
