#!/usr/bin/env python
"""
اسکریپت خزش کامل (Full Crawl) برای سیستم خزشگر هوشمند داده‌های حقوقی

این اسکریپت تنظیمات اولیه را از آرگومان‌های خط فرمان دریافت کرده،
ساختار وبسایت را کشف می‌کند، سپس با استفاده از کلاس Crawler، فرآیند خزش کامل را آغاز می‌کند.
"""

import os
import sys
import argparse
import time
from dotenv import load_dotenv

from utils.logger import get_logger
from core.crawler import Crawler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Full Crawl Script for Legal Data Crawler")
    parser.add_argument('--base-url', type=str, required=True, help="آدرس پایه وبسایت برای شروع خزش")
    parser.add_argument('--max-depth', type=int, default=5, help="حداکثر عمق خزش (پیش‌فرض: 5)")
    parser.add_argument('--max-threads', type=int, default=4, help="حداکثر تعداد نخ‌های همزمان (پیش‌فرض: 4)")
    parser.add_argument('--politeness-delay', type=float, default=1.0, help="تأخیر بین درخواست‌ها (پیش‌فرض: 1.0 ثانیه)")
    parser.add_argument('--respect-robots', action='store_true', help="رعایت قوانین robots.txt")
    return parser.parse_args()


def main():
    # بارگذاری متغیرهای محیطی
    load_dotenv()

    # پردازش آرگومان‌ها
    args = parse_arguments()

    # تنظیم لاگر
    logger = get_logger("full_crawl")
    logger.info("شروع فرآیند خزش کامل")

    # ایجاد نمونه‌ای از خزشگر
    crawler = Crawler(
        base_url=args.base_url,
        max_depth=args.max_depth,
        max_threads=args.max_threads,
        politeness_delay=args.politeness_delay,
        respect_robots=args.respect_robots
    )

    # کشف ساختار وبسایت
    logger.info("در حال کشف ساختار وبسایت...")
    if not crawler.discover_site_structure(force=True):
        logger.error("کشف ساختار وبسایت با مشکل مواجه شد. متوقف می‌شود.")
        sys.exit(1)

    # افزودن آدرس پایه به صف خزش
    crawler.add_job(args.base_url, depth=0, job_type='page')
    logger.info(f"آدرس پایه {args.base_url} به صف خزش اضافه شد.")

    # شروع فرآیند خزش
    try:
        logger.info("شروع اجرای خزش...")
        crawler.start()  # شروع نخ‌ها یا اجرای فرآیند خزش
    except Exception as e:
        logger.error(f"خطا در اجرای خزش: {str(e)}")
        sys.exit(1)

    # منتظر ماندن تا پایان کار نخ‌ها (join)
    crawler.join()  # فرض بر این است که متدی برای انتظار پایان همه نخ‌ها موجود است

    logger.info("خزش کامل به پایان رسید.")


if __name__ == "__main__":
    main()
