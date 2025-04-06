"""
ماژول مدیریت اتصال به پایگاه داده برای خزشگر هوشمند داده‌های حقوقی

این ماژول مسئول ایجاد و مدیریت اتصال به پایگاه داده MySQL است و از
SQLAlchemy برای ایجاد واسط استفاده می‌کند.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
import logging

from utils.logger import get_logger

# بارگذاری متغیرهای محیطی
load_dotenv()

# تنظیم لاگر
logger = get_logger(__name__)

# پایه برای تعریف مدل‌ها
Base = declarative_base()


class DatabaseConnection:
    """کلاس مدیریت اتصال به پایگاه داده"""

    _instance = None

    def __new__(cls):
        """پیاده‌سازی الگوی Singleton برای اطمینان از وجود تنها یک نمونه از کلاس"""
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """مقداردهی اولیه اتصال به پایگاه داده"""
        if self._initialized:
            return

        try:
            # دریافت پارامترهای اتصال از متغیرهای محیطی
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '3306')
            db_user = os.getenv('DB_USER', 'root')
            db_password = os.getenv('DB_PASSWORD', '')
            db_name = os.getenv('DB_NAME', 'legal_crawler')

            # ساخت رشته اتصال
            connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4'

            # ایجاد موتور اتصال
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )

            # ایجاد کارخانه نشست
            session_factory = sessionmaker(bind=self.engine)
            self.SessionLocal = scoped_session(session_factory)

            logger.info(f"اتصال به پایگاه داده {db_name} در {db_host} با موفقیت برقرار شد")
            self._initialized = True

        except Exception as e:
            logger.error(f"خطا در اتصال به پایگاه داده: {str(e)}")
            raise

    def get_session(self):
        """
        ایجاد و بازگرداندن یک نشست پایگاه داده

        Returns:
            Session: یک نشست SQLAlchemy
        """
        return self.SessionLocal()

    def close_session(self, session):
        """
        بستن نشست پایگاه داده

        Args:
            session: نشست SQLAlchemy که باید بسته شود
        """
        session.close()

    def get_engine(self):
        """
        بازگرداندن موتور SQLAlchemy

        Returns:
            Engine: موتور SQLAlchemy
        """
        return self.engine

    def create_tables(self):
        """ایجاد تمام جدول‌های تعریف شده در مدل‌ها"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("تمام جداول با موفقیت ایجاد شدند")
        except Exception as e:
            logger.error(f"خطا در ایجاد جداول: {str(e)}")
            raise


# تابع کمکی برای دریافت نشست پایگاه داده
def get_db():
    """
    تابع کمکی برای استفاده در عملیات پایگاه داده

    Yields:
        Session: یک نشست SQLAlchemy
    """
    db = DatabaseConnection()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()