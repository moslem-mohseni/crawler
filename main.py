#!/usr/bin/env python
"""
نقطه ورود اصلی برنامه خزشگر هوشمند داده‌های حقوقی

این اسکریپت تنظیمات اولیه، ایجاد جداول پایگاه داده، بارگذاری پیکربندی‌ها و راه‌اندازی
ماژول‌های اصلی (مانند خزشگر و استخراج‌کننده ساختار وبسایت) را بر عهده دارد. پس از راه‌اندازی،
برنامه به صورت خودکار عملیات خزش (crawl) را آغاز می‌کند.
"""

import os
import sys
import argparse
import time
import signal
from dotenv import load_dotenv
from sqlalchemy import text

# بارگذاری متغیرهای محیطی
load_dotenv()

# افزودن مسیر پروژه به سیستم
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.settings import DB_CONFIG  # استفاده از تنظیمات پایگاه داده و سایر پیکربندی‌ها
from core.crawler import Crawler
from database.connection import DatabaseConnection
from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)

# متغیر نگهداری نمونه خزشگر برای امکان توقف مناسب
crawler_instance = None


def signal_handler(sig, frame):
    """مدیریت سیگنال‌های سیستم عامل برای توقف مناسب"""
    logger.info(f"سیگنال {sig} دریافت شد. در حال توقف مناسب خزشگر...")

    if crawler_instance and crawler_instance.is_running():
        crawler_instance.stop(wait=True, save_checkpoint=True)

    logger.info("خزشگر با موفقیت متوقف شد.")
    sys.exit(0)


def parse_arguments():
    """پردازش آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(
        description="نقطه ورود برنامه خزشگر هوشمند داده‌های حقوقی"
    )
    parser.add_argument(
        "--crawl-type", choices=["full", "incremental"], default="full",
        help="نوع خزش: 'full' برای خزش کامل یا 'incremental' برای خزش تدریجی (پیش‌فرض: full)"
    )
    parser.add_argument(
        "--base-url", type=str, default=os.getenv("BASE_URL", "https://www.bonyadvokala.com/"),
        help="آدرس وبسایت هدف برای خزش (پیش‌فرض از متغیر محیطی BASE_URL یا https://www.bonyadvokala.com/)"
    )
    parser.add_argument(
        "--max-threads", type=int, default=int(os.getenv("MAX_THREADS", "4")),
        help="حداکثر تعداد نخ‌های همزمان (پیش‌فرض: 4)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=int(os.getenv("MAX_DEPTH", "5")),
        help="حداکثر عمق خزش (پیش‌فرض: 5)"
    )
    parser.add_argument(
        "--delay", type=float, default=float(os.getenv("CRAWL_DELAY", "1.0")),
        help="تأخیر بین درخواست‌ها (ثانیه، پیش‌فرض: 1.0)"
    )
    parser.add_argument(
        "--no-robots", action="store_true",
        help="عدم رعایت محدودیت‌های robots.txt"
    )
    return parser.parse_args()


def initialize_database():
    """ایجاد جداول پایگاه داده در صورت نیاز"""
    try:
        logger.info("در حال اتصال به پایگاه داده و بررسی جداول...")
        db_conn = DatabaseConnection()

        # بررسی اتصال
        session = db_conn.get_session()
        session.execute(text("SELECT 1"))  # استفاده از text() برای کوئری SQL
        session.close()

        # ایجاد جداول
        db_conn.create_tables()
        logger.info("جداول پایگاه داده با موفقیت ایجاد یا بررسی شدند")
        return True
    except Exception as e:
        logger.error(f"خطا در اتصال به پایگاه داده یا ایجاد جداول: {str(e)}")
        return False


def start_crawling(args):
    """
    راه‌اندازی و شروع فرآیند خزش

    Args:
        args: آرگومان‌های پارس شده خط فرمان
    """
    global crawler_instance

    logger.info(f"شروع فرآیند خزش ({args.crawl_type}) برای آدرس: {args.base_url}")

    # نمونه‌سازی خزشگر از ماژول core/crawler
    crawler_instance = Crawler(
        base_url=args.base_url,
        max_threads=args.max_threads,
        max_depth=args.max_depth,
        politeness_delay=args.delay,
        respect_robots=not args.no_robots
    )

    # کشف ساختار وبسایت
    force_discovery = args.crawl_type == "full"
    logger.info(f"در حال کشف ساختار وبسایت {'(اجباری)' if force_discovery else '(در صورت نیاز)'}...")

    try:
        structure_discovered = crawler_instance.discover_site_structure(force=force_discovery)

        if not structure_discovered:
            logger.warning("شناسایی ساختار وبسایت با مشکلاتی همراه بود اما ادامه می‌دهیم...")
    except Exception as e:
        logger.error(f"خطا در شناسایی ساختار وبسایت: {str(e)}")
        return False

    # بارگذاری نقطه بازیابی در خزش تدریجی
    load_checkpoint = args.crawl_type == "incremental"

    # افزودن URL اولیه و شروع خزش
    try:
        logger.info(f"در حال راه‌اندازی خزشگر با {args.max_threads} نخ...")
        crawler_instance.start(
            initial_urls=[args.base_url],
            load_checkpoint=load_checkpoint
        )

        # انتظار برای تکمیل یا وقفه توسط کاربر
        logger.info("خزشگر با موفقیت شروع به کار کرد. منتظر تکمیل یا سیگنال توقف...")
        try:
            # حلقه انتظار با نمایش آمار دوره‌ای
            while crawler_instance.is_running():
                stats = crawler_instance.get_stats()
                logger.info(f"آمار فعلی خزش: {stats['successful_urls']} URL موفق، "
                            f"{stats['current_queue_size']} کار در صف، "
                            f"{stats['active_threads']}/{args.max_threads} نخ فعال")

                time.sleep(30)  # نمایش آمار هر 30 ثانیه

        except KeyboardInterrupt:
            logger.info("خزش توسط کاربر متوقف شد")

        # توقف مناسب خزشگر
        crawler_instance.stop(wait=True, save_checkpoint=True)
        return True

    except Exception as e:
        logger.error(f"خطا در فرآیند خزش: {str(e)}")
        if crawler_instance and crawler_instance.is_running():
            crawler_instance.stop(wait=True, save_checkpoint=True)
        return False


def main():
    """تابع اصلی برنامه"""
    # تنظیم گیرنده سیگنال‌ها برای توقف مناسب
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # پردازش آرگومان‌ها
    args = parse_arguments()

    # ثبت شروع اجرای برنامه
    logger.info("=" * 50)
    logger.info("برنامه خزشگر هوشمند داده‌های حقوقی راه‌اندازی شد")
    logger.info(f"پیکربندی: URL={args.base_url}, نوع={args.crawl_type}, نخ‌ها={args.max_threads}, عمق={args.max_depth}")
    logger.info("=" * 50)

    # بررسی و ایجاد جداول پایگاه داده
    if not initialize_database():
        logger.critical("خروج به دلیل مشکل در پایگاه داده")
        sys.exit(1)

    # شروع فرآیند خزش با استفاده از تنظیمات ورودی
    success = start_crawling(args)

    # ثبت پایان اجرای برنامه
    status = "با موفقیت" if success else "با خطا"
    logger.info(f"برنامه خزشگر {status} به پایان رسید")


if __name__ == "__main__":
    main()
