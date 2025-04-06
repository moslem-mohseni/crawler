"""
پکیج core:
این پکیج شامل ماژول‌های اصلی سیستم خزش هوشمند است که وظایف زیر را پوشش می‌دهد:
    - مدیریت خزش و صف‌بندی کارها (crawler)
    - شناسایی ساختار وبسایت و استخراج الگوهای URL (structure_discovery)
    - طبقه‌بندی محتوا و حوزه‌های تخصصی (classifier)
    - استخراج دقیق محتوا از صفحات وب (content_extractor)
    - ذخیره‌سازی و ایندکس‌گذاری داده‌های استخراج‌شده (storage)

با وارد کردن این پکیج، سایر بخش‌های پروژه می‌توانند به راحتی به کلاس‌ها و توابع اصلی دسترسی پیدا کنند.
"""

from .crawler import Crawler, CrawlJob, CrawlState, URLPriorityPolicyManager
from .structure_discovery import StructureDiscovery, URLPattern, HTMLPatternFinder, URLStructureDiscovery
from .classifier import BaseClassifier, DomainClassifier, ContentTypeClassifier, TextClassifier
from .content_extractor import ContentExtractor
from .storage import StorageManager

__all__ = [
    "Crawler",
    "CrawlJob",
    "CrawlState",
    "URLPriorityPolicyManager",
    "StructureDiscovery",
    "URLPattern",
    "HTMLPatternFinder",
    "URLStructureDiscovery",
    "BaseClassifier",
    "DomainClassifier",
    "ContentTypeClassifier",
    "TextClassifier",
    "ContentExtractor",
    "StorageManager"
]
