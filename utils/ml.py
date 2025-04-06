"""
ماژول ml:
این ماژول شامل توابع و ابزارهای کمکی برای انجام عملیات یادگیری ماشین در پروژه است.
ویژگی‌های اصلی شامل:
    - بارگذاری و ذخیره‌سازی مدل‌های یادگیری ماشین
    - ارزیابی عملکرد مدل‌ها با استفاده از معیارهای استاندارد
    - پشتیبانی از آموزش تدریجی (در صورت نیاز)
    - پوشش توابع استخراج ویژگی (در صورت نیاز به فراخوانی توابع موجود در ml/features)
تمام این توابع با استفاده از سیستم لاگ‌گیری یکپارچه (get_logger) ثبت می‌شوند.
"""

import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.logger import get_logger

logger = get_logger(__name__)


class MLUtils:
    """
    کلاس ابزارهای کمکی ML:
    این کلاس شامل توابع استاتیک برای عملیات رایج یادگیری ماشین مانند بارگذاری/ذخیره مدل،
    ارزیابی مدل، و به‌روزرسانی مدل‌ها است.
    """

    @staticmethod
    def load_model(model_path: str):
        """
        بارگذاری مدل از فایل pickle.

        Args:
            model_path (str): مسیر فایل مدل

        Returns:
            مدل بارگذاری‌شده یا None در صورت بروز خطا
        """
        if not os.path.exists(model_path):
            logger.error(f"فایل مدل یافت نشد: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"مدل با موفقیت از {model_path} بارگذاری شد")
            return model
        except Exception as e:
            logger.error(f"خطا در بارگذاری مدل از {model_path}: {str(e)}")
            return None

    @staticmethod
    def save_model(model, model_path: str):
        """
        ذخیره مدل به فایل pickle.

        Args:
            model: مدل یادگیری ماشین
            model_path (str): مسیر فایل جهت ذخیره

        Returns:
            bool: True در صورت موفقیت، False در صورت بروز خطا
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"مدل با موفقیت در {model_path} ذخیره شد")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره مدل در {model_path}: {str(e)}")
            return False

    @staticmethod
    def evaluate_model(model, X, y, average='binary'):
        """
        ارزیابی عملکرد مدل با استفاده از معیارهای دقت، precision، recall و f1.

        Args:
            model: مدل یادگیری ماشین
            X: ویژگی‌های ورودی
            y: برچسب‌های واقعی
            average: نوع میانگین‌گیری ('binary', 'micro', 'macro', 'weighted')

        Returns:
            dict: شامل معیارهای ارزیابی مدل
        """
        try:
            predictions = model.predict(X)
            acc = accuracy_score(y, predictions)
            prec = precision_score(y, predictions, average=average, zero_division=0)
            rec = recall_score(y, predictions, average=average, zero_division=0)
            f1 = f1_score(y, predictions, average=average, zero_division=0)

            metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1
            }
            logger.info(f"ارزیابی مدل: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"خطا در ارزیابی مدل: {str(e)}")
            return {}

    @staticmethod
    def update_model(model, X_new, y_new):
        """
        به‌روزرسانی مدل با داده‌های جدید به‌صورت تدریجی.
        توجه: این تابع تنها برای مدل‌هایی که از آموزش تدریجی (partial_fit) پشتیبانی می‌کنند، قابل استفاده است.

        Args:
            model: مدل یادگیری ماشین (باید متد partial_fit داشته باشد)
            X_new: ویژگی‌های جدید
            y_new: برچسب‌های جدید

        Returns:
            مدل به‌روز شده یا None در صورت بروز خطا
        """
        if not hasattr(model, 'partial_fit'):
            logger.error("مدل از آموزش تدریجی پشتیبانی نمی‌کند")
            return None

        try:
            model.partial_fit(X_new, y_new)
            logger.info("مدل به‌روزرسانی تدریجی با داده‌های جدید انجام شد")
            return model
        except Exception as e:
            logger.error(f"خطا در به‌روزرسانی مدل: {str(e)}")
            return None

    @staticmethod
    def extract_features(text, feature_extractor):
        """
        پوششی برای استخراج ویژگی‌ها از متن.
        این تابع انتظار دارد که feature_extractor دارای متد transform باشد.

        Args:
            text (str): متن ورودی
            feature_extractor: شیء استخراج‌کننده ویژگی (مثلاً از ml/features)

        Returns:
            ویژگی‌های استخراج‌شده یا None در صورت بروز خطا
        """
        try:
            features = feature_extractor.transform([text])
            logger.info("ویژگی‌ها با موفقیت استخراج شدند")
            return features
        except Exception as e:
            logger.error(f"خطا در استخراج ویژگی‌ها: {str(e)}")
            return None

    @staticmethod
    def log_model_event(event_message: str):
        """
        ثبت رویدادهای مهم مربوط به مدل‌های یادگیری ماشین.

        Args:
            event_message (str): پیام رویداد
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[ML EVENT - {timestamp}] {event_message}")
