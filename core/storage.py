"""
ماژول ذخیره‌سازی و ایندکس‌گذاری:
این ماژول شامل کلاس StorageManager است که وظیفه مدیریت ذخیره‌سازی داده‌های استخراج‌شده
(مانند محتواهای استخراج‌شده توسط ContentExtractor) را در پایگاه داده بر عهده دارد.
این کلاس به‌صورت یکپارچه با مدل‌های پروژه (مانند ContentItem)، سیستم پایگاه داده و سیستم لاگ‌گیری عمل می‌کند.
"""

import json
from datetime import datetime

from utils.logger import get_logger
from database.connection import DatabaseConnection
from models.content import ContentItem


class StorageManager:
    """
    کلاس مدیریت ذخیره‌سازی:
    این کلاس متدهایی برای ذخیره، به‌روزرسانی و بازیابی محتوا از پایگاه داده ارائه می‌دهد.
    همچنین با استفاده از هش مبتنی بر محتوای استخراج‌شده (با استفاده از ContentItem.calculate_similarity_hash)
    از درج داده‌های تکراری جلوگیری می‌کند.
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.db = DatabaseConnection()

    def get_content_by_hash(self, similarity_hash: str):
        """
        بازیابی یک رکورد محتوا بر اساس هش مشابهت.

        Args:
            similarity_hash (str): هش محتوای مورد نظر.

        Returns:
            ContentItem یا None: نمونه محتوا در صورت یافتن، در غیر این صورت None.
        """
        session = self.db.get_session()
        try:
            result = session.query(ContentItem).filter(ContentItem.similarity_hash == similarity_hash).first()
            return result
        except Exception as e:
            self.logger.error(f"خطا در بازیابی محتوا با هش {similarity_hash}: {str(e)}")
            return None
        finally:
            session.close()

    def store_content(self, content_data: dict):
        """
        ذخیره یا به‌روزرسانی یک رکورد محتوا در پایگاه داده.
        این متد ابتدا با محاسبه هش محتوا (با استفاده از متد calculate_similarity_hash در ContentItem)
        بررسی می‌کند که آیا محتوای مشابهی قبلاً ذخیره شده است یا خیر.
        در صورت وجود، رکورد موجود به‌روز شده و در غیر این صورت، رکورد جدید درج می‌شود.

        Args:
            content_data (dict): دیکشنری شامل اطلاعات محتوا، مانند:
                {
                    "url": <str>,
                    "title": <str>,
                    "content": <str>,
                    "date": <str>,
                    "author": <str>,
                    "entities": <dict>,
                    "content_type": <str>  # (مانند 'question', 'article', 'profile', 'other')
                }

        Returns:
            ContentItem یا None: نمونه ذخیره‌شده در صورت موفقیت، یا None در صورت بروز خطا.
        """
        # اطمینان از وجود متن محتوا
        content_text = content_data.get("content", "")
        if not content_text:
            self.logger.error("متن محتوا خالی است؛ ذخیره‌سازی انجام نشد.")
            return None

        # محاسبه هش مشابهت با استفاده از متد موجود در مدل
        similarity_hash = ContentItem.calculate_similarity_hash(content_text)
        content_data["similarity_hash"] = similarity_hash

        # آماده‌سازی داده‌های متادیتا به‌صورت دیکشنری
        meta = {
            "date": content_data.get("date", ""),
            "author": content_data.get("author", ""),
            "entities": content_data.get("entities", {})
        }

        session = self.db.get_session()
        try:
            existing_content = session.query(ContentItem).filter(ContentItem.similarity_hash == similarity_hash).first()
            if existing_content:
                self.logger.info(f"محتوا با هش {similarity_hash} قبلاً ذخیره شده است. به‌روزرسانی رکورد.")
                # به‌روزرسانی فیلدها
                existing_content.title = content_data.get("title", existing_content.title)
                existing_content.content = content_text  # به‌روزرسانی متن محتوا
                existing_content.content_type = content_data.get("content_type", existing_content.content_type)
                # به‌روزرسانی داده‌های متادیتا به صورت دیکشنری
                existing_content.meta_data = meta
                existing_content.updated_at = datetime.now()
                session.commit()
                session.refresh(existing_content)
                return existing_content
            else:
                self.logger.info(f"درج محتوای جدید با هش {similarity_hash}.")
                # استفاده از متد create مدل برای ایجاد نمونه جدید
                new_content = ContentItem.create(
                    content=content_text,
                    url=content_data.get("url", ""),
                    title=content_data.get("title", ""),
                    content_type=content_data.get("content_type", "other"),
                    meta_data=meta,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                # درج رکورد جدید
                session.add(new_content)
                session.commit()
                session.refresh(new_content)
                return new_content
        except Exception as e:
            session.rollback()
            self.logger.error(f"خطا در ذخیره‌سازی محتوا: {str(e)}")
            return None
        finally:
            session.close()

    def bulk_store_contents(self, contents: list):
        """
        ذخیره‌سازی دسته‌ای چندین محتوای استخراج‌شده.
        برای هر محتوا، ابتدا بررسی تکراری بودن انجام شده و در صورت عدم وجود، رکورد جدید درج می‌شود.

        Args:
            contents (list): لیستی از دیکشنری‌های محتوا.

        Returns:
            dict: شامل تعداد موفق و ناموفق بودن ذخیره‌سازی.
        """
        success_count = 0
        failure_count = 0
        for content_data in contents:
            result = self.store_content(content_data)
            if result:
                success_count += 1
            else:
                failure_count += 1
        self.logger.info(f"ذخیره‌سازی دسته‌ای: {success_count} موفق، {failure_count} ناموفق")
        return {"success": success_count, "failure": failure_count}

    def index_content(self, content_item):
        """
        (اختیاری) به‌روزرسانی ایندکس‌های معنایی برای یک محتوای ذخیره‌شده.
        این متد می‌تواند شامل پردازش‌های اضافی برای بهبود جستجو و بازیابی اطلاعات باشد.

        Args:
            content_item (ContentItem): نمونه محتوای ذخیره‌شده.

        Returns:
            bool: True در صورت موفقیت، False در صورت بروز خطا.
        """
        try:
            # به عنوان مثال، می‌توان از Elasticsearch یا سیستم‌های جستجوی متن استفاده کرد.
            # در اینجا به سادگی یک لاگ ثبت می‌شود.
            self.logger.info(f"ایندکس‌بندی محتوای با آیدی {content_item.id} و عنوان {content_item.title}")
            return True
        except Exception as e:
            self.logger.error(f"خطا در ایندکس‌بندی محتوا: {str(e)}")
            return False
