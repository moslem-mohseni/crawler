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

from config.settings import DB_CONFIG  # استفاده از تنظیمات پایگاه داده و سایر پیکربندی‌ها
from core.crawler import Crawler
from database.connection import DatabaseConnection
from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="نقطه ورود برنامه خزشگر هوشمند داده‌های حقوقی"
    )
    parser.add_argument(
        "--crawl-type", choices=["full", "incremental"], default="full",
        help="نوع خزش: 'full' برای خزش کامل یا 'incremental' برای خزش تدریجی (پیش‌فرض: full)"
    )
    parser.add_argument(
        "--base-url", type=str, default=os.getenv("BASE_URL", "https://example.com"),
        help="آدرس وبسایت هدف برای خزش (پیش‌فرض از متغیر محیطی BASE_URL یا https://example.com)"
    )
    return parser.parse_args()


def initialize_database():
    """ایجاد جداول پایگاه داده در صورت نیاز"""
    try:
        db_conn = DatabaseConnection()
        db_conn.create_tables()
        logger.info("جداول پایگاه داده با موفقیت ایجاد شدند")
    except Exception as e:
        logger.error(f"خطا در ایجاد جداول پایگاه داده: {str(e)}")
        sys.exit(1)


def start_crawling(base_url: str, crawl_type: str):
    """
    راه‌اندازی و شروع فرآیند خزش

    Args:
        base_url (str): آدرس وبسایت هدف
        crawl_type (str): نوع خزش ("full" یا "incremental")
    """
    logger.info(f"شروع فرآیند خزش ({crawl_type}) برای آدرس: {base_url}")

    # نمونه‌سازی خزشگر از ماژول core/crawler
    crawler = Crawler(base_url=base_url)

    # کشف ساختار وبسایت (در صورت نیاز، force می‌تواند true شود)
    if not crawler.discover_site_structure(force=False):
        logger.error("شناسایی ساختار وبسایت با خطا مواجه شد")
        sys.exit(1)

    # افزودن اولین کار خزش (آدرس پایه)
    crawler.add_job(url=base_url, depth=0, job_type='page')

    # شروع فرآیند خزش؛ در اینجا از متد run استفاده می‌شود (که می‌تواند شامل
    # ایجاد نخ‌ها، پردازش صف کارها و نظارت بر عملکرد باشد)
    try:
        crawler.run()  # فرض بر این است که متد run تمامی کارها را مدیریت می‌کند
    except KeyboardInterrupt:
        logger.info("خزش متوقف شد توسط کاربر")
    except Exception as e:
        logger.error(f"خطا در فرآیند خزش: {str(e)}")
    finally:
        logger.info("پایان فرآیند خزش")


def main():
    args = parse_arguments()

    # ثبت شروع اجرای برنامه
    logger.info("برنامه خزشگر هوشمند داده‌های حقوقی راه‌اندازی شد")

    # ایجاد جداول پایگاه داده
    initialize_database()

    # شروع فرآیند خزش با استفاده از تنظیمات ورودی
    start_crawling(base_url=args.base_url, crawl_type=args.crawl_type)

    # ثبت پایان اجرای برنامه
    logger.info("برنامه خزشگر به پایان رسید")


if __name__ == "__main__":
    main()
