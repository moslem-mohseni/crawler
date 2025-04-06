"""
ماژول پایه برای مدل‌های داده در خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس پایه برای تمام مدل‌های داده است که ویژگی‌های مشترک
و متدهای کمکی را فراهم می‌کند.
"""

import uuid
import json
from datetime import datetime
from sqlalchemy import Column, String, DateTime, inspect
from sqlalchemy.ext.declarative import declared_attr

from database.connection import Base


class BaseModel(Base):
    """کلاس پایه برای تمام مدل‌های داده در سیستم"""

    __abstract__ = True

    @declared_attr
    def __tablename__(cls):
        """نام جدول را به صورت خودکار از نام کلاس استخراج می‌کند"""
        return cls.__name__.lower()

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @classmethod
    def generate_id(cls, prefix=""):
        """
        تولید یک شناسه منحصر به فرد

        Args:
            prefix: پیشوند اختیاری برای شناسه

        Returns:
            str: یک شناسه منحصر به فرد
        """
        unique_id = str(uuid.uuid4())
        if prefix:
            return f"{prefix}_{unique_id}"
        return unique_id

    def to_dict(self):
        """
        تبدیل مدل به دیکشنری

        Returns:
            dict: یک دیکشنری از تمام ویژگی‌های مدل
        """
        result = {}
        for c in inspect(self).mapper.column_attrs:
            value = getattr(self, c.key)

            # تبدیل تاریخ به رشته
            if isinstance(value, datetime):
                value = value.isoformat()

            # تبدیل JSON ذخیره شده به دیکشنری
            if c.key.endswith('_json') and value is not None:
                try:
                    if isinstance(value, str):
                        value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

            result[c.key] = value

        return result

    def from_dict(self, data):
        """
        بارگذاری داده‌ها از دیکشنری به مدل

        Args:
            data: دیکشنری داده‌ها

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        for key, value in data.items():
            if hasattr(self, key):
                # تبدیل دیکشنری به JSON برای فیلدهای JSON
                if key.endswith('_json') and isinstance(value, (dict, list)):
                    value = json.dumps(value)

                setattr(self, key, value)

        return self

    def update(self, **kwargs):
        """
        به‌روزرسانی چندین فیلد مدل به صورت یکجا

        Args:
            **kwargs: کلید-مقدار برای فیلدهایی که باید به‌روزرسانی شوند

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # به‌روزرسانی زمان تغییر
        self.updated_at = datetime.utcnow()
        return self

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        attrs = []
        for c in inspect(self).mapper.column_attrs:
            if c.key != 'content':  # محتوای بزرگ را نمایش نمی‌دهد
                value = getattr(self, c.key)
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                attrs.append(f"{c.key}={value}")

        return f"<{self.__class__.__name__}({', '.join(attrs)})>"