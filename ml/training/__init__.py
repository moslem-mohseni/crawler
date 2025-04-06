"""
پکیج ml/training:
این پوشه شامل اسکریپت‌های آموزشی مدل‌های طبقه‌بندی حوزه تخصصی و تشخیص نوع محتوا برای خزشگر داده‌های حقوقی است.
اسکریپت‌های موجود در این پوشه:
    - train_domain_model.py: آموزش مدل طبقه‌بندی حوزه‌های تخصصی
    - train_content_model.py: آموزش مدل تشخیص نوع محتوا
"""

from .train_domain_model import main as train_domain_model_main
from .train_content_model import main as train_content_model_main

__all__ = [
    "train_domain_model_main",
    "train_content_model_main"
]
