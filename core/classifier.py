"""
ماژول طبقه‌بندی متون برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس‌های طبقه‌بندی کننده است که از مدل‌های آموزش‌دیده
برای تشخیص حوزه‌های تخصصی و نوع محتوا استفاده می‌کنند.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any

from sklearn.base import BaseEstimator

from utils.logger import get_logger
from ml.features import (DomainFeatures, ContentTypeFeatures,
                        LEGAL_DOMAINS_KEYWORDS, CONTENT_TYPE_KEYWORDS)

# تنظیم لاگر
logger = get_logger(__name__)

# مسیر پیش‌فرض مدل‌های ذخیره شده
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class BaseClassifier:
    """کلاس پایه برای طبقه‌بندی کننده‌ها"""

    def __init__(self, model_path: Optional[str] = None):
        """
        مقداردهی اولیه طبقه‌بندی کننده پایه

        Args:
            model_path: مسیر فایل مدل آموزش‌دیده (اختیاری)
        """
        self.model = None
        self.feature_extractor = None
        self.label_transformer = None  # encoder یا binarizer
        self.model_path = model_path
        self.model_loaded = False

        # بارگذاری مدل در صورت ارائه مسیر
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        بارگذاری مدل از فایل

        Args:
            model_path: مسیر فایل مدل

        Returns:
            bool: آیا بارگذاری موفق بود؟
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"فایل مدل {model_path} یافت نشد")
                return False

            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)

            self.model = model_package.get('model')
            self.label_transformer = model_package.get('label_encoder') or model_package.get('mlb')

            # بارگذاری استخراج‌کننده ویژگی
            feature_extractor_path = model_package.get('feature_extractor_path')
            if feature_extractor_path and os.path.exists(feature_extractor_path):
                # بارگذاری استخراج‌کننده ویژگی مناسب برهر زیرکلاس
                self._load_feature_extractor(feature_extractor_path)
            else:
                logger.warning("مسیر استخراج‌کننده ویژگی یافت نشد")
                self._create_default_feature_extractor()

            self.model_path = model_path
            self.model_loaded = True
            logger.info(f"مدل با موفقیت از {model_path} بارگذاری شد")
            return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل: {str(e)}")
            self.model_loaded = False
            return False

    def _load_feature_extractor(self, path: str):
        """
        بارگذاری استخراج‌کننده ویژگی (در زیرکلاس‌ها پیاده‌سازی می‌شود)

        Args:
            path: مسیر فایل استخراج‌کننده ویژگی
        """
        raise NotImplementedError("این متد باید در زیرکلاس‌ها پیاده‌سازی شود")

    def _create_default_feature_extractor(self):
        """
        ایجاد استخراج‌کننده ویژگی پیش‌فرض (در زیرکلاس‌ها پیاده‌سازی می‌شود)
        """
        raise NotImplementedError("این متد باید در زیرکلاس‌ها پیاده‌سازی شود")

    def predict(self, text: str) -> Dict:
        """
        پیش‌بینی برای یک متن

        Args:
            text: متن ورودی

        Returns:
            Dict: نتیجه پیش‌بینی
        """
        raise NotImplementedError("این متد باید در زیرکلاس‌ها پیاده‌سازی شود")

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        پیش‌بینی برای مجموعه‌ای از متون

        Args:
            texts: لیست متون ورودی

        Returns:
            List[Dict]: لیست نتایج پیش‌بینی
        """
        return [self.predict(text) for text in texts]

    def is_ready(self) -> bool:
        """
        بررسی آماده بودن طبقه‌بندی کننده

        Returns:
            bool: آیا طبقه‌بندی کننده آماده است؟
        """
        return (self.model is not None and
                self.feature_extractor is not None and
                self.label_transformer is not None and
                self.model_loaded)


class DomainClassifier(BaseClassifier):
    """کلاس طبقه‌بندی کننده حوزه‌های تخصصی"""

    def __init__(self, model_path: Optional[str] = None):
        """
        مقداردهی اولیه طبقه‌بندی کننده حوزه تخصصی

        Args:
            model_path: مسیر فایل مدل آموزش‌دیده (اختیاری)
        """
        # استفاده از مسیر پیش‌فرض در صورت عدم ارائه مسیر
        if model_path is None:
            model_path = self._find_latest_model('domain_classifier_')

        super().__init__(model_path)

    def _find_latest_model(self, prefix: str) -> Optional[str]:
        """
        یافتن جدیدترین فایل مدل با پیشوند مشخص

        Args:
            prefix: پیشوند نام فایل

        Returns:
            str: مسیر فایل یا None در صورت عدم وجود
        """
        if not os.path.exists(MODEL_DIR):
            return None

        # یافتن تمام فایل‌های با پیشوند مشخص
        matching_files = [f for f in os.listdir(MODEL_DIR)
                         if f.startswith(prefix) and f.endswith('.pkl')]

        if not matching_files:
            return None

        # مرتب‌سازی بر اساس زمان تغییر فایل (نزولی)
        matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
                           reverse=True)

        # انتخاب جدیدترین فایل
        return os.path.join(MODEL_DIR, matching_files[0])

    def _load_feature_extractor(self, path: str):
        """
        بارگذاری استخراج‌کننده ویژگی حوزه تخصصی

        Args:
            path: مسیر فایل استخراج‌کننده ویژگی
        """
        try:
            from ml.features import DomainFeatures
            self.feature_extractor = DomainFeatures.load(path)
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده ویژگی حوزه تخصصی: {str(e)}")
            self._create_default_feature_extractor()

    def _create_default_feature_extractor(self):
        """ایجاد استخراج‌کننده ویژگی پیش‌فرض حوزه تخصصی"""
        logger.info("ایجاد استخراج‌کننده ویژگی پیش‌فرض حوزه تخصصی")
        self.feature_extractor = DomainFeatures()

    def predict(self, text: str) -> Dict:
        """
        پیش‌بینی حوزه‌های تخصصی برای یک متن

        Args:
            text: متن ورودی

        Returns:
            Dict: نتیجه پیش‌بینی شامل حوزه‌های تخصصی و امتیازات
        """
        if not self.is_ready():
            raise ValueError("طبقه‌بندی کننده آماده نیست. ابتدا یک مدل بارگذاری کنید.")

        # استخراج ویژگی‌ها
        features = self.feature_extractor.transform([text])

        # پیش‌بینی برچسب‌های چندتایی
        y_pred = self.model.predict(features)[0]

        # دریافت احتمالات (در صورت پشتیبانی)
        probabilities = {}
        domains = []

        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(features)

            # برای هر حوزه تخصصی، احتمال پیش‌بینی را ذخیره می‌کنیم
            for i, label in enumerate(self.label_transformer.classes_):
                prob = y_prob[0, i]
                probabilities[label] = float(prob)

                # اگر برچسب مثبت است، حوزه را اضافه می‌کنیم
                if y_pred[i] == 1:
                    domains.append({
                        'domain': label,
                        'probability': float(prob)
                    })
        else:
            # در صورتی که احتمالات در دسترس نیست، فقط برچسب‌ها را برمی‌گردانیم
            domains = [{'domain': self.label_transformer.classes_[i], 'probability': 1.0}
                      for i, val in enumerate(y_pred) if val == 1]

        # مرتب‌سازی حوزه‌ها بر اساس احتمال
        domains.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'domains': [d['domain'] for d in domains],
            'domain_details': domains,
            'probabilities': probabilities
        }


class ContentTypeClassifier(BaseClassifier):
    """کلاس طبقه‌بندی کننده نوع محتوا"""

    def __init__(self, model_path: Optional[str] = None):
        """
        مقداردهی اولیه طبقه‌بندی کننده نوع محتوا

        Args:
            model_path: مسیر فایل مدل آموزش‌دیده (اختیاری)
        """
        # استفاده از مسیر پیش‌فرض در صورت عدم ارائه مسیر
        if model_path is None:
            model_path = self._find_latest_model('content_type_classifier_')

        super().__init__(model_path)

    def _find_latest_model(self, prefix: str) -> Optional[str]:
        """
        یافتن جدیدترین فایل مدل با پیشوند مشخص

        Args:
            prefix: پیشوند نام فایل

        Returns:
            str: مسیر فایل یا None در صورت عدم وجود
        """
        if not os.path.exists(MODEL_DIR):
            return None

        # یافتن تمام فایل‌های با پیشوند مشخص
        matching_files = [f for f in os.listdir(MODEL_DIR)
                         if f.startswith(prefix) and f.endswith('.pkl')]

        if not matching_files:
            return None

        # مرتب‌سازی بر اساس زمان تغییر فایل (نزولی)
        matching_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
                           reverse=True)

        # انتخاب جدیدترین فایل
        return os.path.join(MODEL_DIR, matching_files[0])

    def _load_feature_extractor(self, path: str):
        """
        بارگذاری استخراج‌کننده ویژگی نوع محتوا

        Args:
            path: مسیر فایل استخراج‌کننده ویژگی
        """
        try:
            from ml.features import ContentTypeFeatures
            self.feature_extractor = ContentTypeFeatures.load(path)
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده ویژگی نوع محتوا: {str(e)}")
            self._create_default_feature_extractor()

    def _create_default_feature_extractor(self):
        """ایجاد استخراج‌کننده ویژگی پیش‌فرض نوع محتوا"""
        logger.info("ایجاد استخراج‌کننده ویژگی پیش‌فرض نوع محتوا")
        self.feature_extractor = ContentTypeFeatures()

    def predict(self, text: str) -> Dict:
        """
        پیش‌بینی نوع محتوا برای یک متن

        Args:
            text: متن ورودی

        Returns:
            Dict: نتیجه پیش‌بینی شامل نوع محتوا و احتمالات
        """
        if not self.is_ready():
            raise ValueError("طبقه‌بندی کننده آماده نیست. ابتدا یک مدل بارگذاری کنید.")

        # استخراج ویژگی‌ها
        features = self.feature_extractor.transform([text])

        # پیش‌بینی نوع محتوا
        y_pred = self.model.predict(features)[0]
        content_type = self.label_transformer.inverse_transform([y_pred])[0]

        # دریافت احتمالات (در صورت پشتیبانی)
        probabilities = {}
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(features)[0]

            # ذخیره احتمال هر نوع محتوا
            for i, label in enumerate(self.label_transformer.classes_):
                probabilities[label] = float(y_prob[i])
        else:
            # در صورتی که احتمالات در دسترس نیست، امتیاز تصمیم را برمی‌گردانیم
            if hasattr(self.model, 'decision_function'):
                decision_scores = self.model.decision_function(features)[0]

                if len(self.label_transformer.classes_) == 2:
                    # طبقه‌بندی باینری
                    probabilities[self.label_transformer.classes_[0]] = 1.0 / (1.0 + np.exp(-decision_scores))
                    probabilities[self.label_transformer.classes_[1]] = 1.0 / (1.0 + np.exp(decision_scores))
                else:
                    # طبقه‌بندی چندکلاسه - نرمال‌سازی امتیازات تصمیم
                    exp_scores = np.exp(decision_scores - np.max(decision_scores))
                    softmax = exp_scores / exp_scores.sum()

                    for i, label in enumerate(self.label_transformer.classes_):
                        probabilities[label] = float(softmax[i])
            else:
                # فقط برچسب پیش‌بینی شده
                probabilities = {label: 1.0 if label == content_type else 0.0
                               for label in self.label_transformer.classes_}

        # تحلیل اضافی برای محتواهای سؤال و پاسخ
        analysis = {}
        if content_type == 'question':
            # بررسی نوع سؤال
            if '؟' in text:
                analysis['has_question_mark'] = True

            question_indicators = ['آیا', 'چرا', 'چگونه', 'چطور', 'چیست', 'کیست', 'کجاست']
            found_indicators = [ind for ind in question_indicators if ind in text.lower()]

            if found_indicators:
                analysis['question_indicators'] = found_indicators

        elif content_type == 'answer':
            # بررسی استناد به قانون
            law_references = ['ماده', 'قانون', 'آیین‌نامه', 'بخشنامه', 'دستورالعمل']
            found_references = [ref for ref in law_references if ref in text]

            if found_references:
                analysis['law_references'] = found_references

        return {
            'content_type': content_type,
            'probabilities': probabilities,
            'analysis': analysis
        }


class TextClassifier:
    """کلاس اصلی طبقه‌بندی متون حقوقی"""

    def __init__(self, domain_model_path: Optional[str] = None, content_type_model_path: Optional[str] = None):
        """
        مقداردهی اولیه طبقه‌بندی کننده متون

        Args:
            domain_model_path: مسیر فایل مدل حوزه تخصصی (اختیاری)
            content_type_model_path: مسیر فایل مدل نوع محتوا (اختیاری)
        """
        # طبقه‌بندی کننده‌های تخصصی
        self.domain_classifier = DomainClassifier(domain_model_path)
        self.content_type_classifier = ContentTypeClassifier(content_type_model_path)

    def classify_text(self, text: str) -> Dict:
        """
        طبقه‌بندی کامل یک متن

        Args:
            text: متن ورودی

        Returns:
            Dict: نتیجه طبقه‌بندی شامل حوزه تخصصی و نوع محتوا
        """
        results = {
            'text_summary': text[:100] + '...' if len(text) > 100 else text
        }

        # طبقه‌بندی نوع محتوا
        if self.content_type_classifier.is_ready():
            try:
                content_type_results = self.content_type_classifier.predict(text)
                results['content_type'] = content_type_results
            except Exception as e:
                logger.error(f"خطا در طبقه‌بندی نوع محتوا: {str(e)}")
                results['content_type_error'] = str(e)

        # طبقه‌بندی حوزه تخصصی
        if self.domain_classifier.is_ready():
            try:
                domain_results = self.domain_classifier.predict(text)
                results['domains'] = domain_results
            except Exception as e:
                logger.error(f"خطا در طبقه‌بندی حوزه تخصصی: {str(e)}")
                results['domains_error'] = str(e)

        return results

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """
        طبقه‌بندی دسته‌ای متون

        Args:
            texts: لیست متون ورودی

        Returns:
            List[Dict]: لیست نتایج طبقه‌بندی
        """
        return [self.classify_text(text) for text in texts]

    def is_ready(self) -> Dict:
        """
        بررسی آماده بودن طبقه‌بندی کننده‌ها

        Returns:
            Dict: وضعیت آمادگی طبقه‌بندی کننده‌ها
        """
        return {
            'domain_classifier': self.domain_classifier.is_ready(),
            'content_type_classifier': self.content_type_classifier.is_ready(),
            'all_ready': (self.domain_classifier.is_ready() and
                          self.content_type_classifier.is_ready())
        }


# ایجاد نمونه طبقه‌بندی کننده برای استفاده سریع
default_classifier = TextClassifier()

def classify_text(text: str) -> Dict:
    """
    تابع کمکی برای طبقه‌بندی سریع متن

    Args:
        text: متن ورودی

    Returns:
        Dict: نتیجه طبقه‌بندی
    """
    return default_classifier.classify_text(text)

def predict_domain(text: str) -> Dict:
    """
    تابع کمکی برای پیش‌بینی حوزه تخصصی متن

    Args:
        text: متن ورودی

    Returns:
        Dict: نتیجه پیش‌بینی حوزه تخصصی
    """
    if default_classifier.domain_classifier.is_ready():
        return default_classifier.domain_classifier.predict(text)
    else:
        raise ValueError("طبقه‌بندی کننده حوزه تخصصی آماده نیست")

def predict_content_type(text: str) -> Dict:
    """
    تابع کمکی برای پیش‌بینی نوع محتوا

    Args:
        text: متن ورودی

    Returns:
        Dict: نتیجه پیش‌بینی نوع محتوا
    """
    if default_classifier.content_type_classifier.is_ready():
        return default_classifier.content_type_classifier.predict(text)
    else:
        raise ValueError("طبقه‌بندی کننده نوع محتوا آماده نیست")