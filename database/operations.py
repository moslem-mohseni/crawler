"""
ماژول عملیات پایگاه داده برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس‌های پایه و توابع کمکی برای اجرای عملیات CRUD
بر روی جداول پایگاه داده است.
"""

from sqlalchemy.exc import SQLAlchemyError
from utils.logger import get_logger
from database.connection import DatabaseConnection

# تنظیم لاگر
logger = get_logger(__name__)


class BaseDBOperations:
    """کلاس پایه برای عملیات پایگاه داده"""

    def __init__(self):
        """مقداردهی اولیه با ایجاد اتصال به پایگاه داده"""
        self.db = DatabaseConnection()

    def create(self, model_instance):
        """
        ایجاد یک رکورد جدید در پایگاه داده

        Args:
            model_instance: نمونه‌ای از یک کلاس مدل SQLAlchemy

        Returns:
            model_instance: نمونه افزوده شده به پایگاه داده، یا None در صورت خطا
        """
        session = self.db.get_session()
        try:
            session.add(model_instance)
            session.commit()
            session.refresh(model_instance)
            return model_instance
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"خطا در ایجاد رکورد: {str(e)}")
            return None
        finally:
            session.close()

    def bulk_create(self, model_instances):
        """
        ایجاد چندین رکورد به صورت یکجا در پایگاه داده

        Args:
            model_instances: لیستی از نمونه‌های کلاس مدل SQLAlchemy

        Returns:
            bool: True در صورت موفقیت، False در صورت شکست
        """
        session = self.db.get_session()
        try:
            session.add_all(model_instances)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"خطا در ایجاد دسته‌ای رکوردها: {str(e)}")
            return False
        finally:
            session.close()

    def get_by_id(self, model_class, id_value):
        """
        بازیابی یک رکورد با شناسه مشخص

        Args:
            model_class: کلاس مدل SQLAlchemy
            id_value: مقدار شناسه برای جستجو

        Returns:
            model_instance: نمونه یافت شده، یا None در صورت عدم وجود
        """
        session = self.db.get_session()
        try:
            result = session.query(model_class).filter(model_class.id == id_value).first()
            return result
        except SQLAlchemyError as e:
            logger.error(f"خطا در بازیابی رکورد با شناسه {id_value}: {str(e)}")
            return None
        finally:
            session.close()

    def get_all(self, model_class, skip=0, limit=100, **filters):
        """
        بازیابی تمام رکوردهای یک مدل با اعمال فیلترهای اختیاری

        Args:
            model_class: کلاس مدل SQLAlchemy
            skip: تعداد رکوردهایی که باید از ابتدا رد شوند
            limit: حداکثر تعداد رکوردهایی که باید بازگردانده شوند
            **filters: فیلترهای اختیاری به صورت keyword arguments

        Returns:
            list: لیستی از نمونه‌های یافت شده
        """
        session = self.db.get_session()
        try:
            query = session.query(model_class)

            # اعمال فیلترها
            for attr, value in filters.items():
                if hasattr(model_class, attr):
                    query = query.filter(getattr(model_class, attr) == value)

            # اعمال پارامترهای صفحه‌بندی
            results = query.offset(skip).limit(limit).all()
            return results
        except SQLAlchemyError as e:
            logger.error(f"خطا در بازیابی رکوردها: {str(e)}")
            return []
        finally:
            session.close()

    def update(self, model_class, id_value, **update_values):
        """
        به‌روزرسانی یک رکورد با شناسه مشخص

        Args:
            model_class: کلاس مدل SQLAlchemy
            id_value: مقدار شناسه برای جستجو
            **update_values: مقادیر جدید به صورت keyword arguments

        Returns:
            model_instance: نمونه به‌روزرسانی شده، یا None در صورت خطا
        """
        session = self.db.get_session()
        try:
            instance = session.query(model_class).filter(model_class.id == id_value).first()
            if not instance:
                logger.warning(f"رکورد با شناسه {id_value} یافت نشد")
                return None

            # به‌روزرسانی فیلدها
            for attr, value in update_values.items():
                if hasattr(instance, attr):
                    setattr(instance, attr, value)

            session.commit()
            session.refresh(instance)
            return instance
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"خطا در به‌روزرسانی رکورد با شناسه {id_value}: {str(e)}")
            return None
        finally:
            session.close()

    def delete(self, model_class, id_value):
        """
        حذف یک رکورد با شناسه مشخص

        Args:
            model_class: کلاس مدل SQLAlchemy
            id_value: مقدار شناسه برای جستجو

        Returns:
            bool: True در صورت موفقیت، False در صورت شکست
        """
        session = self.db.get_session()
        try:
            instance = session.query(model_class).filter(model_class.id == id_value).first()
            if not instance:
                logger.warning(f"رکورد با شناسه {id_value} برای حذف یافت نشد")
                return False

            session.delete(instance)
            session.commit()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"خطا در حذف رکورد با شناسه {id_value}: {str(e)}")
            return False
        finally:
            session.close()

    def execute_raw_query(self, query, params=None):
        """
        اجرای یک کوئری SQL خام

        Args:
            query: رشته SQL برای اجرا
            params: پارامترهای کوئری (اختیاری)

        Returns:
            list: نتایج اجرای کوئری یا None در صورت خطا
        """
        engine = self.db.get_engine()
        try:
            with engine.connect() as connection:
                if params:
                    result = connection.execute(query, params)
                else:
                    result = connection.execute(query)

                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"خطا در اجرای کوئری SQL: {str(e)}")
            return None

    def count(self, model_class, **filters):
        """
        شمارش تعداد رکوردهای یک مدل با اعمال فیلترهای اختیاری

        Args:
            model_class: کلاس مدل SQLAlchemy
            **filters: فیلترهای اختیاری به صورت keyword arguments

        Returns:
            int: تعداد رکوردهای یافت شده
        """
        session = self.db.get_session()
        try:
            query = session.query(model_class)

            # اعمال فیلترها
            for attr, value in filters.items():
                if hasattr(model_class, attr):
                    query = query.filter(getattr(model_class, attr) == value)

            return query.count()
        except SQLAlchemyError as e:
            logger.error(f"خطا در شمارش رکوردها: {str(e)}")
            return 0
        finally:
            session.close()


# کلاس‌های عملیات تخصصی برای هر مدل را می‌توان در اینجا اضافه کرد
class DomainOperations(BaseDBOperations):
    """کلاس عملیات تخصصی برای جدول حوزه‌های تخصصی"""

    def get_domains_by_confidence(self, min_confidence=0.7, limit=10):
        """
        بازیابی حوزه‌های تخصصی با اطمینان بالا

        Args:
            min_confidence: حداقل میزان اطمینان (0.0 تا 1.0)
            limit: حداکثر تعداد نتایج

        Returns:
            list: لیستی از حوزه‌های یافت شده
        """
        # این متد بعداً با پیاده‌سازی مدل‌ها تکمیل خواهد شد
        pass


class ContentOperations(BaseDBOperations):
    """کلاس عملیات تخصصی برای جدول محتوا"""

    def get_by_similarity_hash(self, similarity_hash):
        """
        بازیابی محتوا با هش مشابهت مشخص

        Args:
            similarity_hash: هش مشابهت محتوا

        Returns:
            model_instance: نمونه یافت شده، یا None در صورت عدم وجود
        """
        # این متد بعداً با پیاده‌سازی مدل‌ها تکمیل خواهد شد
        pass

    def search_by_keyword(self, keyword, limit=10):
        """
        جستجوی محتوا بر اساس کلیدواژه

        Args:
            keyword: کلیدواژه برای جستجو
            limit: حداکثر تعداد نتایج

        Returns:
            list: لیستی از محتواهای یافت شده
        """
        # این متد بعداً با پیاده‌سازی مدل‌ها تکمیل خواهد شد
        pass


class ExpertOperations(BaseDBOperations):
    """کلاس عملیات تخصصی برای جدول متخصصان"""

    def get_top_experts(self, domain_id=None, limit=10):
        """
        بازیابی برترین متخصصان بر اساس امتیاز

        Args:
            domain_id: شناسه حوزه تخصصی (اختیاری)
            limit: حداکثر تعداد نتایج

        Returns:
            list: لیستی از متخصصان یافت شده
        """
        # این متد بعداً با پیاده‌سازی مدل‌ها تکمیل خواهد شد
        pass