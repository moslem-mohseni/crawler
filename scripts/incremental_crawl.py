#!/usr/bin/env python
"""
اسکریپت خزش تدریجی (Incremental Crawl) برای سیستم خزشگر داده‌های حقوقی

این اسکریپت از نقطه بازیابی موجود (checkpoint) ادامه می‌دهد یا در صورت عدم وجود،
فرآیند خزش را از ابتدا آغاز می‌کند.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

from utils.logger import get_logger
from core.crawler import Crawler


def parse_arguments():
    parser = argparse.ArgumentParser(description="Incremental Crawl Script for Legal Data Crawler")
    parser.add_argument('--base-url', type=str, required=True, help="آدرس پایه وبسایت برای شروع خزش")
    parser.add_argument('--max-depth', type=int, default=5, help="حداکثر عمق خزش (پیش‌فرض: 5)")
    parser.add_argument('--max-threads', type=int, default=4, help="حداکثر تعداد نخ‌های همزمان (پیش‌فرض: 4)")
    parser.add_argument('--politeness-delay', type=float, default=1.0, help="تأخیر بین درخواست‌ها (پیش‌فرض: 1.0 ثانیه)")
    parser.add_argument('--respect-robots', action='store_true', help="رعایت قوانین robots.txt")
    parser.add_argument('--checkpoint', type=str, default=None, help="مسیر فایل نقطه بازیابی (اختیاری)")
    return parser.parse_args()


def main():
    # بارگذاری متغیرهای محیطی
    load_dotenv()

    # پردازش آرگومان‌های خط فرمان
    args = parse_arguments()

    # تنظیم لاگر
    logger = get_logger("incremental_crawl")
    logger.info("شروع فرآیند خزش تدریجی")

    # ایجاد نمونه‌ای از خزشگر با تنظیمات دریافتی
    crawler = Crawler(
        base_url=args.base_url,
        max_depth=args.max_depth,
        max_threads=args.max_threads,
        politeness_delay=args.politeness_delay,
        respect_robots=args.respect_robots
    )

    # بارگذاری نقطه بازیابی در صورت وجود
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"بارگذاری نقطه بازیابی از فایل {args.checkpoint}")
        if crawler.load_checkpoint(args.checkpoint):
            logger.info("نقطه بازیابی با موفقیت بارگذاری شد.")
        else:
            logger.warning("بارگذاری نقطه بازیابی با مشکل مواجه شد؛ ادامه از ابتدا.")
            crawler.add_job(args.base_url, depth=0, job_type='page')
    else:
        logger.info("فایل نقطه بازیابی مشخص نشده یا موجود نیست؛ ادامه خزش از ابتدا.")
        crawler.add_job(args.base_url, depth=0, job_type='page')

    # شروع اجرای خزش تدریجی
    try:
        logger.info("شروع اجرای خزش تدریجی...")
        crawler.start()  # شروع اجرای نخ‌ها
    except Exception as e:
        logger.error(f"خطا در اجرای خزش: {str(e)}")
        sys.exit(1)

    # انتظار برای پایان کار نخ‌ها
    crawler.join()  # انتظار پایان تمام نخ‌ها

    logger.info("خزش تدریجی به پایان رسید.")

    # ذخیره وضعیت جدید (checkpoint) پس از پایان خزش
    if crawler.checkpoint_file:
        logger.info(f"ذخیره نقطه بازیابی در {crawler.checkpoint_file}")
        crawler.save_checkpoint(crawler.checkpoint_file)


if __name__ == "__main__":
    main()
