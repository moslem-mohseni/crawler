"""
پکیج ml:
این پکیج شامل ماژول‌های یادگیری ماشین پروژه می‌باشد که وظایف زیر را پوشش می‌دهد:
  - مدل‌های آموزش دیده (ذخیره شده در پوشه ml/models)
  - اسکریپت‌های آموزشی (موجود در پوشه ml/training)
  - ابزارها و توابع استخراج ویژگی (موجود در پوشه ml/features)

با وارد کردن این پکیج، سایر بخش‌های پروژه (مانند core، utils و config) می‌توانند به راحتی به اجزای کلیدی یادگیری ماشین دسترسی پیدا کنند.
"""

from .models import MODEL_DIR
from .training import train_domain_model_main, train_content_model_main
from .features import (
    PersianTextPreprocessor,
    PersianTfidfVectorizer,
    KeywordFeatureExtractor,
    ContentTypeFeatureExtractor,
    StructuralFeatureExtractor,
    LegalFeatureExtractor,
    ContentTypeFeatures,
    DomainFeatures
)

__all__ = [
    "MODEL_DIR",
    "train_domain_model_main",
    "train_content_model_main",
    "PersianTextPreprocessor",
    "PersianTfidfVectorizer",
    "KeywordFeatureExtractor",
    "ContentTypeFeatureExtractor",
    "StructuralFeatureExtractor",
    "LegalFeatureExtractor",
    "ContentTypeFeatures",
    "DomainFeatures"
]
