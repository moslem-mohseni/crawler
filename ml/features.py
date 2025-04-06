"""
ماژول استخراج ویژگی‌ها برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس‌ها و توابع استخراج ویژگی‌های متنی مناسب برای
پردازش متون حقوقی فارسی، طبقه‌بندی حوزه تخصصی و تشخیص نوع محتوا است.
"""

import os
import re
import json
import pickle
import numpy as np
import scipy.sparse as sp
from collections import Counter
from typing import List, Dict, Tuple, Union, Optional, Any

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

# واردکردن ماژول‌های پروژه
from utils.logger import get_logger
from utils.text import (normalize_persian_text, tokenize_persian_text,
                        PERSIAN_STOP_WORDS, extract_main_content, calculate_text_hash)

# تنظیم لاگر
logger = get_logger(__name__)

# مسیر فایل‌های مربوط به آموزش مدل
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# لیست عبارات و کلمات کلیدی در حوزه‌های مختلف حقوقی
LEGAL_DOMAINS_KEYWORDS = {
    'criminal': [
        'جرم', 'مجازات', 'زندان', 'حبس', 'قصاص', 'دیه', 'تعزیر', 'قانون مجازات',
        'جزا', 'بزهکار', 'متهم', 'مجرم', 'شاکی', 'بزه', 'جنایت', 'سرقت', 'قتل',
        'جرح', 'ضرب', 'حدود', 'کلاهبرداری', 'اختلاس', 'ارتشا', 'قاچاق'
    ],
    'civil': [
        'قرارداد', 'عقد', 'معامله', 'ارث', 'وصیت', 'قانون مدنی', 'مالکیت', 'بیع',
        'اجاره', 'وقف', 'نکاح', 'طلاق', 'مهریه', 'حضانت', 'عقود', 'تعهد', 'تملیک',
        'تملک', 'خسارت', 'مسئولیت مدنی', 'ضمان', 'رهن', 'اسناد', 'شرط', 'الزام'
    ],
    'commercial': [
        'تجارت', 'شرکت', 'سهام', 'تاجر', 'قانون تجارت', 'ورشکستگی', 'چک', 'سفته',
        'برات', 'اوراق بهادار', 'بورس', 'قرارداد تجاری', 'حق العمل کاری', 'ضمانت نامه',
        'حمل و نقل', 'بیمه', 'داوری تجاری', 'مالیات', 'مناقصه', 'مزایده'
    ],
    'administrative': [
        'استخدام', 'کارگر', 'کارفرما', 'حقوق کار', 'قانون کار', 'تأمین اجتماعی',
        'بیمه', 'مالیات', 'تخلفات اداری', 'دیوان عدالت اداری', 'قانون شهرداری',
        'امور اداری', 'استخدام دولتی', 'ترفیع', 'انفصال', 'کارمند', 'خدمات کشوری'
    ],
    'constitutional': [
        'قانون اساسی', 'حقوق اساسی', 'دولت', 'مجلس', 'قوه قضاییه', 'قوه مجریه',
        'قوه مقننه', 'انتخابات', 'نظام', 'حکومت', 'جمهوری', 'رهبر', 'ریاست جمهوری',
        'وزیر', 'وزارت', 'نمایندگان', 'شورای نگهبان'
    ]
}

# عبارات تشخیص نوع محتوا
CONTENT_TYPE_KEYWORDS = {
    'question': [
        'سوال', 'پرسش', 'سؤال', 'چرا', 'چگونه', 'آیا',
        'چطور', 'چیست', 'کیست', 'کجاست', 'کدام',
        '؟', 'لطفا پاسخ دهید', 'لطفا راهنمایی کنید'
    ],
    'answer': [
        'پاسخ', 'جواب', 'در پاسخ به', 'طبق قانون', 'بر اساس قانون',
        'با استناد به', 'با توجه به قانون', 'به موجب ماده', 'طبق ماده',
        'با احترام', 'باید گفت', 'باید عرض کنم'
    ],
    'article': [
        'مقاله', 'مقدمه', 'چکیده', 'نتیجه‌گیری', 'بررسی', 'تحلیل',
        'پژوهش', 'مطالعه', 'یافته‌ها', 'منابع', 'مآخذ', 'نتایج',
        'ادبیات تحقیق', 'روش‌شناسی', 'روش تحقیق'
    ],
    'profile': [
        'سوابق', 'تحصیلات', 'تخصص', 'وکیل', 'مشاور حقوقی', 'قاضی',
        'حقوقدان', 'دانشگاه', 'مدرک', 'دکتری', 'کارشناسی', 'سابقه فعالیت',
        'زمینه فعالیت', 'حوزه تخصصی'
    ]
}


class PersianTextPreprocessor(BaseEstimator, TransformerMixin):
    """کلاس پیش‌پردازش متن فارسی برای استخراج ویژگی"""

    def __init__(self, normalize=True, remove_stopwords=True, remove_punctuation=True, min_tokens=2):
        """
        مقداردهی اولیه پیش‌پردازشگر متن فارسی

        Args:
            normalize: آیا متن نرمال‌سازی شود؟
            remove_stopwords: آیا ایست‌واژه‌ها حذف شوند؟
            remove_punctuation: آیا علائم نگارشی حذف شوند؟
            min_tokens: حداقل تعداد توکن‌های مورد نیاز
        """
        self.normalize = normalize
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.min_tokens = min_tokens

    def fit(self, X, y=None):
        """
        آموزش پیش‌پردازشگر (در این مورد نیازی به آموزش نیست)

        Args:
            X: داده‌های ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        return self

    def transform(self, X):
        """
        اعمال پیش‌پردازش روی داده‌های ورودی

        Args:
            X: لیست متون ورودی

        Returns:
            list: لیست متون پیش‌پردازش شده
        """
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        """
        پیش‌پردازش یک متن

        Args:
            text: متن ورودی

        Returns:
            str: متن پیش‌پردازش شده
        """
        if not text:
            return ""

        # استخراج محتوای اصلی اگر متن HTML باشد
        if "<html" in text.lower() or "<body" in text.lower():
            text = extract_main_content(text)

        # نرمال‌سازی متن
        if self.normalize:
            text = normalize_persian_text(text)

        # توکن‌سازی
        tokens = tokenize_persian_text(
            text,
            remove_stop_words=self.remove_stopwords,
            remove_punctuation=self.remove_punctuation
        )

        # بررسی حداقل تعداد توکن‌ها
        if len(tokens) < self.min_tokens:
            return ""

        # ترکیب توکن‌ها
        return " ".join(tokens)

    def get_params(self, deep=True):
        """
        دریافت پارامترهای تنظیم شده

        Args:
            deep: آیا پارامترهای زیرمجموعه‌ها هم بازگردانده شوند؟

        Returns:
            dict: دیکشنری پارامترها
        """
        return {
            "normalize": self.normalize,
            "remove_stopwords": self.remove_stopwords,
            "remove_punctuation": self.remove_punctuation,
            "min_tokens": self.min_tokens
        }

    def set_params(self, **parameters):
        """
        تنظیم پارامترها

        Args:
            **parameters: پارامترهای جدید

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PersianTfidfVectorizer:
    """کلاس استخراج ویژگی TF-IDF برای متون فارسی"""

    def __init__(self, max_features=10000, min_df=2, max_df=0.95, ngram_range=(1, 2)):
        """
        مقداردهی اولیه استخراج‌کننده TF-IDF فارسی

        Args:
            max_features: حداکثر تعداد ویژگی‌ها
            min_df: حداقل تعداد اسناد برای هر ویژگی
            max_df: حداکثر درصد اسناد برای هر ویژگی
            ngram_range: محدوده n-gram
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range

        # ایجاد پیش‌پردازشگر و بردارساز
        self.preprocessor = PersianTextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            analyzer='word',
            tokenizer=lambda x: x.split()  # متن از قبل توکن‌سازی شده
        )

        self.fitted = False

    def fit(self, texts, y=None):
        """
        آموزش استخراج‌کننده TF-IDF

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # پیش‌پردازش متون
        preprocessed_texts = self.preprocessor.fit_transform(texts)

        # آموزش بردارساز
        self.vectorizer.fit(preprocessed_texts)

        self.fitted = True
        return self

    def transform(self, texts):
        """
        تبدیل متون به بردارهای TF-IDF

        Args:
            texts: لیست متون ورودی

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌های TF-IDF
        """
        if not self.fitted:
            raise ValueError("استخراج‌کننده TF-IDF آموزش داده نشده است. ابتدا متد fit را فراخوانی کنید.")

        # پیش‌پردازش متون
        preprocessed_texts = self.preprocessor.transform(texts)

        # استخراج ویژگی‌ها
        return self.vectorizer.transform(preprocessed_texts)

    def fit_transform(self, texts, y=None):
        """
        آموزش و تبدیل همزمان

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌های TF-IDF
        """
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names_out(self):
        """
        دریافت نام ویژگی‌ها

        Returns:
            List[str]: لیست نام ویژگی‌ها
        """
        if not self.fitted:
            raise ValueError("استخراج‌کننده TF-IDF آموزش داده نشده است. ابتدا متد fit را فراخوانی کنید.")

        return self.vectorizer.get_feature_names_out()

    def save(self, file_path):
        """
        ذخیره مدل استخراج‌کننده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'max_features': self.max_features,
                    'min_df': self.min_df,
                    'max_df': self.max_df,
                    'ngram_range': self.ngram_range,
                    'preprocessor': self.preprocessor,
                    'vectorizer': self.vectorizer,
                    'fitted': self.fitted
                }, f)
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره استخراج‌کننده TF-IDF: {str(e)}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        بارگذاری مدل استخراج‌کننده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            PersianTfidfVectorizer: نمونه بارگذاری شده یا None در صورت خطا
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            vectorizer = cls(
                max_features=data['max_features'],
                min_df=data['min_df'],
                max_df=data['max_df'],
                ngram_range=data['ngram_range']
            )

            vectorizer.preprocessor = data['preprocessor']
            vectorizer.vectorizer = data['vectorizer']
            vectorizer.fitted = data['fitted']

            return vectorizer
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده TF-IDF: {str(e)}")
            return None


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """استخراج‌کننده ویژگی‌های مبتنی بر کلیدواژه‌های حوزه حقوقی"""

    def __init__(self, domain_keywords=None, normalize=True):
        """
        مقداردهی اولیه استخراج‌کننده کلیدواژه

        Args:
            domain_keywords: دیکشنری حوزه‌ها و کلیدواژه‌های آنها
            normalize: آیا متن نرمال‌سازی شود؟
        """
        self.domain_keywords = domain_keywords or LEGAL_DOMAINS_KEYWORDS
        self.normalize = normalize
        self.domain_names = list(self.domain_keywords.keys())

    def fit(self, X, y=None):
        """
        آموزش استخراج‌کننده (در این مورد نیازی به آموزش نیست)

        Args:
            X: داده‌های ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        return self

    def transform(self, X):
        """
        استخراج ویژگی‌های کلیدواژه

        Args:
            X: لیست متون ورودی

        Returns:
            numpy.ndarray: ماتریس ویژگی‌های کلیدواژه
        """
        # ماتریس ویژگی‌ها: هر ردیف یک متن، هر ستون فراوانی کلیدواژه‌های یک حوزه
        features = np.zeros((len(X), len(self.domain_names)))

        for i, text in enumerate(X):
            if not text:
                continue

            # نرمال‌سازی متن
            if self.normalize:
                text = normalize_persian_text(text)

            # محاسبه فراوانی کلیدواژه‌ها برای هر حوزه
            for j, domain in enumerate(self.domain_names):
                keywords = self.domain_keywords[domain]

                # محاسبه فراوانی هر کلیدواژه
                domain_score = 0
                for keyword in keywords:
                    # الگو برای تطبیق دقیق‌تر کلیدواژه‌ها
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    count = len(re.findall(pattern, text, re.IGNORECASE))
                    domain_score += count

                # نرمال‌سازی امتیاز بر اساس تعداد کلیدواژه‌ها
                if domain_score > 0:
                    features[i, j] = domain_score / len(keywords)

        # نرمال‌سازی سطری
        row_sums = features.sum(axis=1)
        non_zero_rows = row_sums > 0
        if np.any(non_zero_rows):
            features[non_zero_rows] = features[non_zero_rows] / row_sums[non_zero_rows].reshape(-1, 1)

        return features

    def get_feature_names_out(self):
        """
        دریافت نام ویژگی‌ها

        Returns:
            List[str]: لیست نام ویژگی‌ها
        """
        return self.domain_names


class ContentTypeFeatureExtractor(BaseEstimator, TransformerMixin):
    """استخراج‌کننده ویژگی‌های مبتنی بر نوع محتوا"""

    def __init__(self, type_keywords=None, normalize=True):
        """
        مقداردهی اولیه استخراج‌کننده نوع محتوا

        Args:
            type_keywords: دیکشنری انواع محتوا و کلیدواژه‌های آنها
            normalize: آیا متن نرمال‌سازی شود؟
        """
        self.type_keywords = type_keywords or CONTENT_TYPE_KEYWORDS
        self.normalize = normalize
        self.content_types = list(self.type_keywords.keys())

    def fit(self, X, y=None):
        """
        آموزش استخراج‌کننده (در این مورد نیازی به آموزش نیست)

        Args:
            X: داده‌های ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        return self

    def transform(self, X):
        """
        استخراج ویژگی‌های نوع محتوا

        Args:
            X: لیست متون ورودی

        Returns:
            numpy.ndarray: ماتریس ویژگی‌های نوع محتوا
        """
        # ماتریس ویژگی‌ها: هر ردیف یک متن، هر ستون فراوانی کلیدواژه‌های یک نوع محتوا
        features = np.zeros((len(X), len(self.content_types)))

        # ویژگی‌های ساختاری
        structural_features = np.zeros((len(X), 6))

        for i, text in enumerate(X):
            if not text:
                continue

            # نرمال‌سازی متن
            if self.normalize:
                text = normalize_persian_text(text)

            # محاسبه ویژگی‌های ساختاری
            # 1. طول متن
            text_length = len(text)
            structural_features[i, 0] = text_length / 1000  # نرمال‌سازی

            # 2. تعداد پاراگراف‌ها
            paragraphs = [p for p in text.split('\n') if p.strip()]
            structural_features[i, 1] = len(paragraphs)

            # 3. تعداد علامت سوال
            question_marks = text.count('؟')
            structural_features[i, 2] = question_marks

            # 4. میانگین طول جملات
            sentences = re.split(r'[.!؟]+', text)
            sentences = [s for s in sentences if s.strip()]
            avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
            structural_features[i, 3] = avg_sentence_length / 100  # نرمال‌سازی

            # 5. نسبت علامت سوال به تعداد جملات
            structural_features[i, 4] = question_marks / max(len(sentences), 1)

            # 6. آیا با سوال شروع می‌شود؟
            if sentences:
                first_sentence = sentences[0].strip().lower()
                is_question = any(q in first_sentence for q in ['آیا', 'چرا', 'چگونه', 'چطور', 'چیست', 'کیست', 'کجاست'])
                structural_features[i, 5] = 1 if is_question or (first_sentence and first_sentence[-1] == '؟') else 0

            # محاسبه فراوانی کلیدواژه‌ها برای هر نوع محتوا
            for j, content_type in enumerate(self.content_types):
                keywords = self.type_keywords[content_type]

                # محاسبه فراوانی هر کلیدواژه
                type_score = 0
                for keyword in keywords:
                    # الگو برای تطبیق دقیق‌تر کلیدواژه‌ها
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    count = len(re.findall(pattern, text, re.IGNORECASE))
                    type_score += count

                # نرمال‌سازی امتیاز بر اساس تعداد کلیدواژه‌ها
                if type_score > 0:
                    features[i, j] = type_score / len(keywords)

        # نرمال‌سازی سطری برای ویژگی‌های کلیدواژه
        row_sums = features.sum(axis=1)
        non_zero_rows = row_sums > 0
        if np.any(non_zero_rows):
            features[non_zero_rows] = features[non_zero_rows] / row_sums[non_zero_rows].reshape(-1, 1)

        # ترکیب ویژگی‌های کلیدواژه و ساختاری
        combined_features = np.hstack((features, structural_features))

        return combined_features

    def get_feature_names_out(self):
        """
        دریافت نام ویژگی‌ها

        Returns:
            List[str]: لیست نام ویژگی‌ها
        """
        return self.content_types + [
            'text_length', 'paragraphs_count', 'question_marks',
            'avg_sentence_length', 'question_mark_ratio', 'starts_with_question'
        ]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """استخراج‌کننده ویژگی‌های ساختاری متن"""

    def __init__(self, normalize=True):
        """
        مقداردهی اولیه استخراج‌کننده ویژگی‌های ساختاری

        Args:
            normalize: آیا متن نرمال‌سازی شود؟
        """
        self.normalize = normalize
        self.feature_names = [
            'length', 'word_count', 'sentence_count', 'paragraph_count',
            'avg_sentence_length', 'avg_word_length', 'punct_ratio',
            'uppercase_ratio', 'digit_ratio', 'unique_words_ratio',
            'question_marks', 'exclamation_marks'
        ]

    def fit(self, X, y=None):
        """
        آموزش استخراج‌کننده (در این مورد نیازی به آموزش نیست)

        Args:
            X: داده‌های ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        return self

    def transform(self, X):
        """
        استخراج ویژگی‌های ساختاری

        Args:
            X: لیست متون ورودی

        Returns:
            numpy.ndarray: ماتریس ویژگی‌های ساختاری
        """
        # ماتریس ویژگی‌ها: هر ردیف یک متن، هر ستون یک ویژگی ساختاری
        features = np.zeros((len(X), len(self.feature_names)))

        for i, text in enumerate(X):
            if not text:
                continue

            # نرمال‌سازی متن
            if self.normalize:
                text = normalize_persian_text(text)

            # 1. طول متن
            text_length = len(text)
            features[i, 0] = text_length

            # 2. تعداد کلمات
            words = text.split()
            word_count = len(words)
            features[i, 1] = word_count

            # 3. تعداد جملات
            sentences = re.split(r'[.!؟]+', text)
            sentences = [s for s in sentences if s.strip()]
            sentence_count = len(sentences)
            features[i, 2] = sentence_count

            # 4. تعداد پاراگراف‌ها
            paragraphs = [p for p in text.split('\n') if p.strip()]
            paragraph_count = len(paragraphs)
            features[i, 3] = paragraph_count

            # 5. میانگین طول جملات
            if sentence_count > 0:
                avg_sentence_length = sum(len(s) for s in sentences) / sentence_count
                features[i, 4] = avg_sentence_length

            # 6. میانگین طول کلمات
            if word_count > 0:
                avg_word_length = sum(len(w) for w in words) / word_count
                features[i, 5] = avg_word_length

            # 7. نسبت علائم نگارشی
            punct_count = sum(1 for c in text if c in '.,;:؛،?!؟-()[]{}«»')
            if text_length > 0:
                features[i, 6] = punct_count / text_length

            # 8. نسبت حروف بزرگ (برای متون انگلیسی)
            uppercase_count = sum(1 for c in text if c.isupper())
            if text_length > 0:
                features[i, 7] = uppercase_count / text_length

            # 9. نسبت اعداد
            digit_count = sum(1 for c in text if c.isdigit())
            if text_length > 0:
                features[i, 8] = digit_count / text_length

            # 10. نسبت کلمات یکتا
            if word_count > 0:
                unique_words_ratio = len(set(words)) / word_count
                features[i, 9] = unique_words_ratio

            # 11. تعداد علامت سوال
            features[i, 10] = text.count('؟')

            # 12. تعداد علامت تعجب
            features[i, 11] = text.count('!')

        # نرمال‌سازی ویژگی‌ها
        if len(X) > 0:
            features_max = np.max(features, axis=0)
            features_max[features_max == 0] = 1  # جلوگیری از تقسیم بر صفر
            features = features / features_max

        return features

    def get_feature_names_out(self):
        """
        دریافت نام ویژگی‌ها

        Returns:
            List[str]: لیست نام ویژگی‌ها
        """
        return self.feature_names


class LegalFeatureExtractor:
    """کلاس اصلی استخراج ویژگی‌های متنی حقوقی"""

    def __init__(self, use_tfidf=True, use_keywords=True, use_structural=True):
        """
        مقداردهی اولیه استخراج‌کننده ویژگی‌های حقوقی

        Args:
            use_tfidf: آیا از ویژگی‌های TF-IDF استفاده شود؟
            use_keywords: آیا از ویژگی‌های کلیدواژه استفاده شود؟
            use_structural: آیا از ویژگی‌های ساختاری استفاده شود؟
        """
        self.use_tfidf = use_tfidf
        self.use_keywords = use_keywords
        self.use_structural = use_structural

        # ایجاد استخراج‌کننده‌های ویژگی
        self.tfidf_vectorizer = PersianTfidfVectorizer() if use_tfidf else None
        self.keyword_extractor = KeywordFeatureExtractor() if use_keywords else None
        self.structural_extractor = StructuralFeatureExtractor() if use_structural else None

        self.fitted = False

    def fit(self, texts, y=None):
        """
        آموزش استخراج‌کننده ویژگی‌های حقوقی

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # آموزش استخراج‌کننده‌های ویژگی
        if self.use_tfidf and self.tfidf_vectorizer:
            self.tfidf_vectorizer.fit(texts)

        if self.use_keywords and self.keyword_extractor:
            self.keyword_extractor.fit(texts)

        if self.use_structural and self.structural_extractor:
            self.structural_extractor.fit(texts)

        self.fitted = True
        return self

    def transform(self, texts):
        """
        استخراج ویژگی‌های متنی

        Args:
            texts: لیست متون ورودی

        Returns:
            scipy.sparse.csr_matrix یا numpy.ndarray: ماتریس ویژگی‌ها
        """
        if not self.fitted:
            raise ValueError("استخراج‌کننده ویژگی‌ها آموزش داده نشده است. ابتدا متد fit را فراخوانی کنید.")

        features_list = []

        # استخراج ویژگی‌های TF-IDF
        if self.use_tfidf and self.tfidf_vectorizer:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
            features_list.append(tfidf_features)

        # استخراج ویژگی‌های کلیدواژه
        if self.use_keywords and self.keyword_extractor:
            keyword_features = self.keyword_extractor.transform(texts)
            features_list.append(sp.csr_matrix(keyword_features))

        # استخراج ویژگی‌های ساختاری
        if self.use_structural and self.structural_extractor:
            structural_features = self.structural_extractor.transform(texts)
            features_list.append(sp.csr_matrix(structural_features))

        # ترکیب ویژگی‌ها
        if not features_list:
            return sp.csr_matrix((len(texts), 0))

        if len(features_list) == 1:
            return features_list[0]

        return sp.hstack(features_list, format='csr')

    def fit_transform(self, texts, y=None):
        """
        آموزش و استخراج همزمان

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            scipy.sparse.csr_matrix یا numpy.ndarray: ماتریس ویژگی‌ها
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, file_path):
        """
        ذخیره مدل استخراج‌کننده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        if not self.fitted:
            logger.warning("استخراج‌کننده ویژگی‌ها آموزش داده نشده است")

        try:
            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # ذخیره اجزای مختلف در فایل‌های جداگانه
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]

            # ذخیره TF-IDF Vectorizer
            if self.use_tfidf and self.tfidf_vectorizer:
                tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
                self.tfidf_vectorizer.save(tfidf_path)

            # ذخیره کل مدل
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'use_tfidf': self.use_tfidf,
                    'use_keywords': self.use_keywords,
                    'use_structural': self.use_structural,
                    'keyword_extractor': self.keyword_extractor,
                    'structural_extractor': self.structural_extractor,
                    'fitted': self.fitted
                }, f)

            logger.info(f"استخراج‌کننده ویژگی‌ها با موفقیت در {file_path} ذخیره شد")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره استخراج‌کننده ویژگی‌ها: {str(e)}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        بارگذاری مدل استخراج‌کننده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            LegalFeatureExtractor: نمونه بارگذاری شده یا None در صورت خطا
        """
        try:
            # بارگذاری مدل اصلی
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # ایجاد نمونه جدید
            extractor = cls(
                use_tfidf=data['use_tfidf'],
                use_keywords=data['use_keywords'],
                use_structural=data['use_structural']
            )

            # بارگذاری اجزای مختلف
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]

            # بارگذاری TF-IDF Vectorizer
            if data['use_tfidf']:
                tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
                extractor.tfidf_vectorizer = PersianTfidfVectorizer.load(tfidf_path)

            # بارگذاری سایر اجزا
            extractor.keyword_extractor = data['keyword_extractor']
            extractor.structural_extractor = data['structural_extractor']
            extractor.fitted = data['fitted']

            logger.info(f"استخراج‌کننده ویژگی‌ها با موفقیت از {file_path} بارگذاری شد")
            return extractor
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده ویژگی‌ها: {str(e)}")
            return None


class ContentTypeFeatures:
    """کلاس استخراج ویژگی‌های تشخیص نوع محتوا"""

    def __init__(self):
        """مقداردهی اولیه استخراج‌کننده نوع محتوا"""
        # ایجاد استخراج‌کننده‌های ویژگی
        self.tfidf_vectorizer = PersianTfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        self.content_type_extractor = ContentTypeFeatureExtractor()
        self.structural_extractor = StructuralFeatureExtractor()

        self.fitted = False

    def fit(self, texts, y=None):
        """
        آموزش استخراج‌کننده نوع محتوا

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # آموزش استخراج‌کننده‌های ویژگی
        self.tfidf_vectorizer.fit(texts)
        self.content_type_extractor.fit(texts)
        self.structural_extractor.fit(texts)

        self.fitted = True
        return self

    def transform(self, texts):
        """
        استخراج ویژگی‌های نوع محتوا

        Args:
            texts: لیست متون ورودی

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌ها
        """
        if not self.fitted:
            raise ValueError("استخراج‌کننده نوع محتوا آموزش داده نشده است. ابتدا متد fit را فراخوانی کنید.")

        # استخراج ویژگی‌ها
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        content_type_features = self.content_type_extractor.transform(texts)
        structural_features = self.structural_extractor.transform(texts)

        # ترکیب ویژگی‌ها
        combined_features = sp.hstack([
            tfidf_features,
            sp.csr_matrix(content_type_features),
            sp.csr_matrix(structural_features)
        ], format='csr')

        return combined_features

    def fit_transform(self, texts, y=None):
        """
        آموزش و استخراج همزمان

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌ها
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, file_path):
        """
        ذخیره مدل استخراج‌کننده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        try:
            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # ذخیره TF-IDF Vectorizer
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]
            tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
            self.tfidf_vectorizer.save(tfidf_path)

            # ذخیره کل مدل
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'content_type_extractor': self.content_type_extractor,
                    'structural_extractor': self.structural_extractor,
                    'fitted': self.fitted
                }, f)

            logger.info(f"استخراج‌کننده نوع محتوا با موفقیت در {file_path} ذخیره شد")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره استخراج‌کننده نوع محتوا: {str(e)}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        بارگذاری مدل استخراج‌کننده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            ContentTypeFeatures: نمونه بارگذاری شده یا None در صورت خطا
        """
        try:
            # بارگذاری مدل اصلی
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # ایجاد نمونه جدید
            extractor = cls()

            # بارگذاری TF-IDF Vectorizer
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]
            tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
            extractor.tfidf_vectorizer = PersianTfidfVectorizer.load(tfidf_path)

            # بارگذاری سایر اجزا
            extractor.content_type_extractor = data['content_type_extractor']
            extractor.structural_extractor = data['structural_extractor']
            extractor.fitted = data['fitted']

            logger.info(f"استخراج‌کننده نوع محتوا با موفقیت از {file_path} بارگذاری شد")
            return extractor
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده نوع محتوا: {str(e)}")
            return None


class DomainFeatures:
    """کلاس استخراج ویژگی‌های تشخیص حوزه تخصصی"""

    def __init__(self):
        """مقداردهی اولیه استخراج‌کننده حوزه تخصصی"""
        # ایجاد استخراج‌کننده‌های ویژگی
        self.tfidf_vectorizer = PersianTfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 3)
        )
        self.keyword_extractor = KeywordFeatureExtractor()

        self.fitted = False

    def fit(self, texts, y=None):
        """
        آموزش استخراج‌کننده حوزه تخصصی

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # آموزش استخراج‌کننده‌های ویژگی
        self.tfidf_vectorizer.fit(texts)
        self.keyword_extractor.fit(texts)

        self.fitted = True
        return self

    def transform(self, texts):
        """
        استخراج ویژگی‌های حوزه تخصصی

        Args:
            texts: لیست متون ورودی

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌ها
        """
        if not self.fitted:
            raise ValueError("استخراج‌کننده حوزه تخصصی آموزش داده نشده است. ابتدا متد fit را فراخوانی کنید.")

        # استخراج ویژگی‌ها
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        keyword_features = self.keyword_extractor.transform(texts)

        # ترکیب ویژگی‌ها
        combined_features = sp.hstack([
            tfidf_features,
            sp.csr_matrix(keyword_features)
        ], format='csr')

        return combined_features

    def fit_transform(self, texts, y=None):
        """
        آموزش و استخراج همزمان

        Args:
            texts: لیست متون ورودی
            y: برچسب‌ها (اختیاری)

        Returns:
            scipy.sparse.csr_matrix: ماتریس ویژگی‌ها
        """
        self.fit(texts)
        return self.transform(texts)

    def save(self, file_path):
        """
        ذخیره مدل استخراج‌کننده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        try:
            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # ذخیره TF-IDF Vectorizer
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]
            tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
            self.tfidf_vectorizer.save(tfidf_path)

            # ذخیره کل مدل
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'keyword_extractor': self.keyword_extractor,
                    'fitted': self.fitted
                }, f)

            logger.info(f"استخراج‌کننده حوزه تخصصی با موفقیت در {file_path} ذخیره شد")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره استخراج‌کننده حوزه تخصصی: {str(e)}")
            return False

    @classmethod
    def load(cls, file_path):
        """
        بارگذاری مدل استخراج‌کننده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            DomainFeatures: نمونه بارگذاری شده یا None در صورت خطا
        """
        try:
            # بارگذاری مدل اصلی
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # ایجاد نمونه جدید
            extractor = cls()

            # بارگذاری TF-IDF Vectorizer
            data_dir = os.path.dirname(file_path)
            base_name = os.path.basename(file_path).split('.')[0]
            tfidf_path = os.path.join(data_dir, f"{base_name}_tfidf.pkl")
            extractor.tfidf_vectorizer = PersianTfidfVectorizer.load(tfidf_path)

            # بارگذاری سایر اجزا
            extractor.keyword_extractor = data['keyword_extractor']
            extractor.fitted = data['fitted']

            logger.info(f"استخراج‌کننده حوزه تخصصی با موفقیت از {file_path} بارگذاری شد")
            return extractor
        except Exception as e:
            logger.error(f"خطا در بارگذاری استخراج‌کننده حوزه تخصصی: {str(e)}")
            return None


# توابع کمکی برای استفاده مستقیم

def extract_features(texts, feature_type='general'):
    """
    استخراج ویژگی‌های متنی با توجه به نوع مورد نیاز

    Args:
        texts: لیست متون ورودی
        feature_type: نوع ویژگی ('general', 'domain', 'content_type')

    Returns:
        scipy.sparse.csr_matrix: ماتریس ویژگی‌ها
    """
    if feature_type == 'domain':
        extractor = DomainFeatures()
    elif feature_type == 'content_type':
        extractor = ContentTypeFeatures()
    else:
        extractor = LegalFeatureExtractor()

    return extractor.fit_transform(texts)


def load_feature_extractor(feature_type, model_path=None):
    """
    بارگذاری استخراج‌کننده ویژگی ذخیره شده

    Args:
        feature_type: نوع ویژگی ('general', 'domain', 'content_type')
        model_path: مسیر فایل مدل (اختیاری)

    Returns:
        BaseEstimator: استخراج‌کننده ویژگی بارگذاری شده یا None در صورت خطا
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"{feature_type}_features.pkl")

    if not os.path.exists(model_path):
        logger.warning(f"فایل مدل یافت نشد: {model_path}")
        return None

    if feature_type == 'domain':
        return DomainFeatures.load(model_path)
    elif feature_type == 'content_type':
        return ContentTypeFeatures.load(model_path)
    else:
        return LegalFeatureExtractor.load(model_path)