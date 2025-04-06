#!/usr/bin/env python
"""
اسکریپت راه‌اندازی اولیه پایگاه داده برای خزشگر هوشمند داده‌های حقوقی

این اسکریپت جداول پایگاه داده را ایجاد و برخی داده‌های پایه را بارگذاری می‌کند.
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

# افزودن مسیر پروژه به سیستم
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# بارگذاری متغیرهای محیطی
load_dotenv()

from utils.logger import get_logger
from database.connection import DatabaseConnection
from database.schema import create_tables, drop_tables, recreate_tables
from models.domain import Domain
from models.content import ContentItem, Answer, DomainContent
from models.expert import Expert, ExpertDomain

# تنظیم لاگر
logger = get_logger(__name__)


def parse_arguments():
    """
    پردازش آرگومان‌های خط فرمان

    Returns:
        argparse.Namespace: آرگومان‌های پردازش شده
    """
    parser = argparse.ArgumentParser(description='اسکریپت راه‌اندازی اولیه پایگاه داده')

    parser.add_argument('--recreate', action='store_true',
                        help='حذف و ایجاد مجدد تمام جداول (هشدار: تمام داده‌ها حذف می‌شوند)')

    parser.add_argument('--drop', action='store_true',
                        help='فقط حذف تمام جداول (هشدار: تمام داده‌ها حذف می‌شوند)')

    parser.add_argument('--seed', action='store_true',
                        help='بارگذاری داده‌های اولیه')

    parser.add_argument('--seed-file', type=str, default='config/initial_data.json',
                        help='مسیر فایل داده‌های اولیه (پیش‌فرض: config/initial_data.json)')

    return parser.parse_args()


def init_database(recreate=False, drop=False):
    """
    ایجاد جداول پایگاه داده

    Args:
        recreate: آیا جداول حذف و دوباره ایجاد شوند؟
        drop: آیا فقط جداول حذف شوند؟

    Returns:
        bool: آیا عملیات موفق بود؟
    """
    try:
        db_conn = DatabaseConnection()

        if drop:
            logger.warning("در حال حذف تمام جداول...")
            result = drop_tables()
            logger.info("حذف جداول کامل شد")
            return result

        if recreate:
            logger.warning("در حال بازسازی تمام جداول...")
            result = recreate_tables()
            logger.info("بازسازی جداول کامل شد")
            return result

        logger.info("در حال ایجاد جداول...")
        result = create_tables()
        logger.info("ایجاد جداول کامل شد")
        return result

    except Exception as e:
        logger.error(f"خطا در راه‌اندازی پایگاه داده: {str(e)}")
        return False


def load_initial_data(seed_file):
    """
    بارگذاری داده‌های اولیه به پایگاه داده

    Args:
        seed_file: مسیر فایل داده‌های اولیه (JSON)

    Returns:
        bool: آیا عملیات موفق بود؟
    """
    try:
        # بررسی وجود فایل
        if not os.path.exists(seed_file):
            logger.error(f"فایل داده‌های اولیه یافت نشد: {seed_file}")
            return False

        # خواندن فایل داده‌های اولیه
        with open(seed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # اتصال به پایگاه داده
        db_conn = DatabaseConnection()
        session = db_conn.get_session()

        try:
            # بارگذاری حوزه‌های تخصصی
            if 'domains' in data:
                logger.info(f"در حال بارگذاری {len(data['domains'])} حوزه تخصصی...")

                for domain_data in data['domains']:
                    domain = Domain.create(
                        name=domain_data['name'],
                        description=domain_data.get('description'),
                        keywords=domain_data.get('keywords'),
                        id=domain_data.get('id')
                    )
                    session.add(domain)

            # بارگذاری متخصصان
            if 'experts' in data:
                logger.info(f"در حال بارگذاری {len(data['experts'])} متخصص...")

                for expert_data in data['experts']:
                    expert = Expert.create(
                        name=expert_data['name'],
                        bio=expert_data.get('bio'),
                        expertise=expert_data.get('expertise'),
                        profile_url=expert_data.get('profile_url'),
                        avatar_url=expert_data.get('avatar_url')
                    )
                    session.add(expert)

            # ذخیره تغییرات برای ایجاد شناسه‌ها
            session.commit()

            # بارگذاری روابط متخصص-حوزه
            if 'expert_domains' in data:
                logger.info(f"در حال بارگذاری {len(data['expert_domains'])} رابطه متخصص-حوزه...")

                for relation in data['expert_domains']:
                    expert_domain = ExpertDomain.create(
                        expert_id=relation['expert_id'],
                        domain_id=relation['domain_id'],
                        confidence_score=relation.get('confidence_score', 0.8)
                    )
                    session.add(expert_domain)

            # ذخیره نهایی تمام تغییرات
            session.commit()
            logger.info("بارگذاری داده‌های اولیه با موفقیت انجام شد")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"خطا در بارگذاری داده‌های اولیه: {str(e)}")
            return False

        finally:
            session.close()

    except Exception as e:
        logger.error(f"خطا در خواندن فایل داده‌های اولیه: {str(e)}")
        return False


def main():
    """تابع اصلی برنامه"""
    # پردازش آرگومان‌های خط فرمان
    args = parse_arguments()

    # مقداردهی اولیه پایگاه داده
    if args.drop:
        init_database(drop=True)
    elif args.recreate:
        init_database(recreate=True)
    else:
        init_database()

    # بارگذاری داده‌های اولیه در صورت نیاز
    if args.seed:
        load_initial_data(args.seed_file)


# اجرای برنامه در صورت فراخوانی مستقیم
if __name__ == '__main__':
    main()