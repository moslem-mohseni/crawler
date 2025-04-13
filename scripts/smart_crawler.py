#!/usr/bin/env python
"""
اسکریپت خزش هوشمند خودکار برای خزشگر داده‌های حقوقی

این اسکریپت به صورت هوشمند و خودکار، سرویس‌ها را راه‌اندازی کرده،
دیتابیس را بررسی و مدیریت می‌کند، و با استراتژی زمانی هوشمند، خزش سایت هدف را انجام می‌دهد.
تنظیم فواصل زمانی به صورت تطبیقی از کوتاه (در شروع) به طولانی (در ادامه) انجام می‌شود.
"""

import os
import sys
import time
import random
import signal
import argparse
import threading
import queue
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlparse

# بارگذاری متغیرهای محیطی
from dotenv import load_dotenv

load_dotenv()

# افزودن مسیر پروژه به سیستم (فقط اگر مورد نیاز باشد)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# واردسازی تنظیمات پروژه
from config.settings import (
    DB_CONFIG, CRAWLER_CONFIG, BASE_DIR, DATA_DIR, LOGS_DIR,
    load_defaults, load_domain_config, get_user_agent_list
)

# واردسازی ماژول‌های پایگاه داده
from database.connection import DatabaseConnection
from database.schema import create_tables

# واردسازی ماژول‌های هسته خزشگر
from core.crawler import Crawler, CrawlState
from core.structure_discovery import StructureDiscovery
from core.content_extractor import ContentExtractor
from core.classifier import TextClassifier
from core.storage import StorageManager

# واردسازی مدل‌ها
from models.content import ContentItem

# واردسازی ابزارهای کمکی
from utils.logger import get_logger, get_crawler_logger
from utils.text import extract_links

# تنظیم لاگرها
logger = get_logger("smart_crawler")
crawler_logger = get_crawler_logger()

# بارگذاری تنظیمات پیش‌فرض
DEFAULT_CONFIG = load_defaults()


class SmartCrawlManager:
    """مدیریت هوشمند خزش با استراتژی زمانی تطبیقی"""

    def __init__(self, base_url: str, config_dir: str = None,
                 max_threads: int = None, initial_delay: float = None,
                 respect_robots: bool = None, database_retry_attempts: int = 5):
        """
        مقداردهی اولیه مدیریت هوشمند خزش

        Args:
            base_url: آدرس پایه وبسایت هدف
            config_dir: مسیر دایرکتوری پیکربندی (اختیاری)
            max_threads: حداکثر تعداد نخ‌های همزمان (بازنویسی تنظیمات)
            initial_delay: تأخیر اولیه بین درخواست‌ها (بازنویسی تنظیمات)
            respect_robots: آیا محدودیت‌های robots.txt رعایت شود؟ (بازنویسی تنظیمات)
            database_retry_attempts: تعداد تلاش‌های مجدد برای اتصال به دیتابیس
        """
        self.base_url = base_url
        self.config_dir = config_dir or os.path.join(BASE_DIR, 'config')

        # استفاده از تنظیمات پیکربندی با امکان بازنویسی
        self.max_threads = max_threads if max_threads is not None else CRAWLER_CONFIG['max_threads']
        self.initial_delay = initial_delay if initial_delay is not None else CRAWLER_CONFIG['politeness_delay']
        self.respect_robots = respect_robots if respect_robots is not None else CRAWLER_CONFIG['respect_robots']
        self.db_retry_attempts = database_retry_attempts

        # فرکانس‌های خزش (به دقیقه)
        self.crawl_frequency = {
            'initial': 1,  # هر 1 دقیقه
            'active': 30,  # هر 30 دقیقه
            'steady': 180,  # هر 3 ساعت
            'maintenance': 1440  # هر 24 ساعت (روزی یکبار)
        }

        # شمارشگرها و حالت‌ها
        self.urls_processed = 0
        self.urls_new_content = 0
        self.crawl_phase = 'initial'  # initial, active, steady, maintenance
        self.last_phase_change = datetime.now()

        # مسیر فایل نقطه بازیابی
        self.domain = self._extract_domain(base_url)
        self.checkpoint_file = os.path.join(self.config_dir, f"{self.domain}_smart_crawl_state.json")

        # اتصال به دیتابیس و سایر منابع
        self.db_conn = None
        self.crawler = None
        self.structure_discovery = None
        self.content_extractor = None
        self.classifier = None
        self.storage_manager = None

        # صف کار و وضعیت خزش
        self.crawl_queue = queue.PriorityQueue()
        self.crawl_state = CrawlState(checkpoint_file=self.checkpoint_file)

        # متغیرهای کنترل
        self.running = False
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()

        # آمار و اطلاعات
        self.stats = {
            'startup_time': datetime.now(),
            'last_crawl_time': '',
            'total_urls_found': 0,
            'total_urls_processed': 0,
            'total_content_items': 0,
            'total_errors': 0,
            'pages_by_type': {},
            'current_phase': self.crawl_phase,
            'phase_transitions': []
        }

        # تنظیم گیرنده سیگنال‌ها برای پایان دادن مناسب
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _extract_domain(self, url: str) -> str:
        """
        استخراج دامنه از URL

        Args:
            url: آدرس URL

        Returns:
            دامنه استخراج شده
        """
        parsed = urlparse(url)
        return parsed.netloc.replace(".", "_")

    def _signal_handler(self, signum, frame):
        """
        مدیریت سیگنال‌های سیستم عامل

        Args:
            signum: نوع سیگنال
            frame: فریم اجرایی
        """
        logger.info(f"سیگنال {signum} دریافت شد. در حال توقف مناسب...")
        self.stop()
        sys.exit(0)

    def verify_database_connection(self) -> bool:
        """
        بررسی اتصال به پایگاه داده

        Returns:
            آیا اتصال موفقیت‌آمیز است؟
        """
        logger.info("در حال بررسی اتصال به پایگاه داده...")

        for attempt in range(1, self.db_retry_attempts + 1):
            try:
                # تلاش برای ایجاد اتصال
                logger.info(f"تلاش {attempt} از {self.db_retry_attempts} برای اتصال به پایگاه داده")

                # استفاده از DatabaseConnection موجود در پروژه
                self.db_conn = DatabaseConnection()

                # بررسی اتصال با اجرای یک کوئری ساده
                session = self.db_conn.get_session()
                session.execute("SELECT 1")
                session.close()

                logger.info("✅ اتصال به پایگاه داده با موفقیت برقرار شد.")
                return True

            except Exception as e:
                logger.error(f"خطا در اتصال به پایگاه داده: {str(e)}")

                if attempt < self.db_retry_attempts:
                    # تأخیر نمایی قبل از تلاش مجدد
                    delay = 2 ** attempt
                    logger.info(f"تلاش مجدد در {delay} ثانیه...")
                    time.sleep(delay)
                else:
                    logger.critical("❌ اتصال به پایگاه داده پس از چندین تلاش امکان‌پذیر نبود.")
                    return False

    def verify_database_tables(self) -> bool:
        """
        بررسی وجود جداول لازم در پایگاه داده و ایجاد آنها در صورت نیاز

        Returns:
            آیا جداول به درستی وجود دارند یا ایجاد شده‌اند؟
        """
        logger.info("در حال بررسی جداول پایگاه داده...")

        try:
            # بررسی وجود جداول با فراخوانی مستقیم تابع create_tables
            result = create_tables()

            if result:
                logger.info("✅ جداول پایگاه داده با موفقیت بررسی و در صورت نیاز ایجاد شدند.")
                return True
            else:
                logger.error("❌ خطا در بررسی یا ایجاد جداول پایگاه داده.")
                return False

        except Exception as e:
            logger.error(f"خطا در بررسی یا ایجاد جداول: {str(e)}")
            return False

    def initialize_services(self) -> bool:
        """
        راه‌اندازی سرویس‌های مورد نیاز

        Returns:
            آیا راه‌اندازی موفقیت‌آمیز بود؟
        """
        logger.info("در حال راه‌اندازی سرویس‌های مورد نیاز...")

        try:
            # راه‌اندازی خزشگر
            logger.info("راه‌اندازی خزشگر...")
            self.crawler = Crawler(
                base_url=self.base_url,
                config_dir=self.config_dir,
                max_threads=self.max_threads,
                politeness_delay=self.initial_delay,
                respect_robots=self.respect_robots
            )

            # راه‌اندازی کشف ساختار
            logger.info("راه‌اندازی سیستم کشف ساختار...")
            self.structure_discovery = StructureDiscovery(self.base_url, self.config_dir)

            # راه‌اندازی استخراج محتوا
            logger.info("راه‌اندازی سیستم استخراج محتوا...")
            self.content_extractor = ContentExtractor()

            # راه‌اندازی طبقه‌بندی کننده
            logger.info("راه‌اندازی سیستم طبقه‌بندی...")
            self.classifier = TextClassifier()

            # راه‌اندازی مدیریت ذخیره‌سازی
            logger.info("راه‌اندازی سیستم ذخیره‌سازی...")
            self.storage_manager = StorageManager()

            # بررسی آمادگی سیستم
            classifier_status = self.classifier.is_ready()
            if classifier_status.get('all_ready', False):
                logger.info("✅ طبقه‌بندی‌کننده با موفقیت راه‌اندازی شد.")
            else:
                logger.warning("⚠️ طبقه‌بندی‌کننده به طور کامل راه‌اندازی نشد. برخی قابلیت‌ها ممکن است محدود باشد.")

            logger.info("✅ تمام سرویس‌ها با موفقیت راه‌اندازی شدند.")
            return True

        except Exception as e:
            logger.error(f"خطا در راه‌اندازی سرویس‌ها: {str(e)}")
            return False

    def discover_site_structure(self, force=False) -> bool:
        """
        کشف ساختار وبسایت

        Args:
            force: آیا کشف ساختار با وجود کشف قبلی انجام شود؟

        Returns:
            آیا کشف ساختار موفقیت‌آمیز بود؟
        """
        logger.info("در حال کشف ساختار وبسایت...")

        try:
            # استفاده از ماژول کشف ساختار موجود
            success = self.structure_discovery.discover_structure(force=force)

            if success:
                logger.info("✅ ساختار وبسایت با موفقیت کشف شد.")
            else:
                logger.warning("⚠️ کشف ساختار وبسایت با مشکلاتی مواجه شد.")

            return success

        except Exception as e:
            logger.error(f"خطا در کشف ساختار وبسایت: {str(e)}")
            return False

    def load_state(self) -> bool:
        """
        بارگذاری وضعیت قبلی خزش (در صورت وجود)

        Returns:
            آیا بارگذاری موفقیت‌آمیز بود؟
        """
        logger.info("در حال بررسی وضعیت قبلی خزش...")

        try:
            # بررسی وجود فایل وضعیت
            if os.path.exists(self.checkpoint_file):
                logger.info(f"فایل وضعیت یافت شد: {self.checkpoint_file}")

                # بارگذاری وضعیت
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # به‌روزرسانی وضعیت جاری
                self.crawl_phase = state.get('crawl_phase', 'initial')
                self.urls_processed = state.get('urls_processed', 0)
                self.urls_new_content = state.get('urls_new_content', 0)

                # به‌روزرسانی آمار
                if 'stats' in state:
                    temp_stats = state['stats']
                    if isinstance(temp_stats, dict):
                        # حفظ برخی مقادیر اصلی و به‌روزرسانی بقیه
                        temp_stats['startup_time'] = datetime.now()
                        self.stats.update(temp_stats)

                # به‌روزرسانی زمان آخرین تغییر فاز
                phase_change_str = state.get('last_phase_change')
                if phase_change_str:
                    try:
                        self.last_phase_change = datetime.fromisoformat(phase_change_str)
                    except (ValueError, TypeError):
                        self.last_phase_change = datetime.now()

                # بارگذاری وضعیت خزش
                self.crawl_state.load_checkpoint(self.checkpoint_file)

                logger.info(f"✅ وضعیت خزش بارگذاری شد. فاز فعلی: {self.crawl_phase}, "
                            f"URLهای پردازش شده: {self.urls_processed}")
                return True
            else:
                logger.info("فایل وضعیت قبلی یافت نشد. شروع خزش از ابتدا...")
                return False

        except Exception as e:
            logger.error(f"خطا در بارگذاری وضعیت: {str(e)}")
            return False

    def save_state(self) -> bool:
        """
        ذخیره وضعیت فعلی خزش

        Returns:
            آیا ذخیره‌سازی موفقیت‌آمیز بود؟
        """
        logger.info("در حال ذخیره وضعیت خزش...")

        try:
            # آماده‌سازی داده‌های وضعیت
            state = {
                'crawl_phase': self.crawl_phase,
                'urls_processed': self.urls_processed,
                'urls_new_content': self.urls_new_content,
                'last_phase_change': self.last_phase_change.isoformat(),
                'stats': self.stats,
                'saved_at': datetime.now().isoformat()
            }

            # ایجاد دایرکتوری در صورت نیاز
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

            # ذخیره وضعیت
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            # ذخیره وضعیت خزش
            self.crawl_state.save_checkpoint(self.checkpoint_file)

            logger.info(f"✅ وضعیت خزش در {self.checkpoint_file} ذخیره شد.")
            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره وضعیت: {str(e)}")
            return False

    def update_crawl_phase(self) -> None:
        """به‌روزرسانی فاز خزش بر اساس پیشرفت و معیارهای تعیین شده"""
        current_time = datetime.now()
        time_in_phase = (current_time - self.last_phase_change).total_seconds() / 60  # به دقیقه

        with self.stats_lock:
            # منطق تغییر فاز
            if self.crawl_phase == 'initial':
                # معیار پیشرفت از initial به active
                if self.urls_processed > 100 or time_in_phase > 60:  # 100 URL یا 1 ساعت
                    self._change_phase('active')

            elif self.crawl_phase == 'active':
                # معیار پیشرفت از active به steady
                if self.urls_processed > 1000 or time_in_phase > 240:  # 1000 URL یا 4 ساعت
                    percentage_new = self.urls_new_content / max(1, self.urls_processed) * 100
                    if percentage_new < 20:  # کمتر از 20٪ محتوای جدید
                        self._change_phase('steady')

            elif self.crawl_phase == 'steady':
                # معیار پیشرفت از steady به maintenance
                if self.urls_processed > 5000 or time_in_phase > 1440:  # 5000 URL یا 24 ساعت
                    percentage_new = self.urls_new_content / max(1, self.urls_processed) * 100
                    if percentage_new < 5:  # کمتر از 5٪ محتوای جدید
                        self._change_phase('maintenance')

    def _change_phase(self, new_phase: str) -> None:
        """
        تغییر فاز خزش و به‌روزرسانی آمار

        Args:
            new_phase: فاز جدید
        """
        old_phase = self.crawl_phase
        self.crawl_phase = new_phase
        self.last_phase_change = datetime.now()

        # به‌روزرسانی آمار
        self.stats['current_phase'] = new_phase
        self.stats['phase_transitions'].append({
            'from': old_phase,
            'to': new_phase,
            'time': self.last_phase_change.isoformat(),
            'urls_processed': self.urls_processed,
            'urls_new_content': self.urls_new_content
        })

        # ریست شمارشگرها
        self.urls_processed = 0
        self.urls_new_content = 0

        logger.info(f"🔄 تغییر فاز خزش از '{old_phase}' به '{new_phase}'")
        logger.info(f"⏱️ زمان خواب بین خزش‌ها: {self.get_current_sleep_time()} دقیقه")

    def get_current_sleep_time(self) -> float:
        """
        محاسبه زمان خواب فعلی بر اساس فاز خزش

        Returns:
            زمان خواب به دقیقه
        """
        # استفاده از فرکانس‌های از پیش تعریف شده
        base_time = self.crawl_frequency[self.crawl_phase]

        # اضافه کردن کمی تصادفی‌سازی (±20٪) برای جلوگیری از رفتار قابل پیش‌بینی
        jitter = random.uniform(0.8, 1.2)
        return base_time * jitter

    def extract_and_store_content(self, url: str, html_content: str) -> bool:
        """
        استخراج محتوا از HTML و ذخیره آن در پایگاه داده

        Args:
            url: آدرس URL صفحه
            html_content: محتوای HTML صفحه

        Returns:
            آیا ذخیره‌سازی موفقیت‌آمیز بود؟
        """
        logger.info(f"در حال استخراج و ذخیره محتوای {url}...")

        try:
            # استخراج محتوا با استفاده از ContentExtractor
            extracted_data = self.content_extractor.extract(html_content, url)

            # طبقه‌بندی محتوا
            classifier_status = self.classifier.is_ready()
            if classifier_status.get('all_ready', False):
                classification_result = self.classifier.classify_text(extracted_data['content'])

                # افزودن نتایج طبقه‌بندی به داده‌های استخراج شده
                if 'content_type' in classification_result:
                    content_type_info = classification_result.get('content_type', {})
                    extracted_data['content_type'] = content_type_info.get('content_type', 'other')

                if 'domains' in classification_result:
                    domains_info = classification_result.get('domains', {})
                    extracted_data['domains'] = domains_info.get('domains', [])
            else:
                # طبقه‌بندی پیش‌فرض
                extracted_data['content_type'] = 'other'
                extracted_data['domains'] = []

            # بررسی وجود محتوای مشابه
            hash_value = ContentItem.calculate_similarity_hash(extracted_data['content'])
            existing_content = self.storage_manager.get_content_by_hash(hash_value)

            if existing_content:
                logger.info(f"محتوای مشابه برای {url} قبلاً ذخیره شده است.")
                return False

            # ذخیره محتوا
            content_data = {
                'url': url,
                'title': extracted_data.get('title', ''),
                'content': extracted_data.get('content', ''),
                'content_type': extracted_data.get('content_type', 'other'),
                'meta_data': {
                    'date': extracted_data.get('date', ''),
                    'author': extracted_data.get('author', ''),
                    'entities': extracted_data.get('entities', {})
                }
            }

            stored_content = self.storage_manager.store_content(content_data)

            if stored_content:
                with self.stats_lock:
                    self.urls_new_content += 1
                    self.stats['total_content_items'] += 1

                    # به‌روزرسانی آمار نوع محتوا
                    content_type = content_data.get('content_type', 'other')
                    if 'pages_by_type' not in self.stats:
                        self.stats['pages_by_type'] = {}

                    self.stats['pages_by_type'][content_type] = self.stats['pages_by_type'].get(content_type, 0) + 1

                logger.info(f"✅ محتوای {url} با موفقیت ذخیره شد.")
                return True
            else:
                logger.warning(f"⚠️ ذخیره محتوای {url} با مشکل مواجه شد.")
                return False

        except Exception as e:
            with self.stats_lock:
                self.stats['total_errors'] += 1

            logger.error(f"خطا در استخراج یا ذخیره محتوای {url}: {str(e)}")
            return False

    def process_url(self, url: str, depth: int = 0) -> Dict[str, Any]:
        """
        پردازش یک URL: دریافت محتوا، استخراج لینک‌ها، ذخیره محتوا

        Args:
            url: آدرس URL برای پردازش
            depth: عمق URL در گراف خزش

        Returns:
            نتایج پردازش
        """
        logger.info(f"در حال پردازش {url} (عمق: {depth})...")

        # بررسی قبلاً پردازش شدن
        if self.crawl_state.was_visited(url):
            logger.info(f"URL {url} قبلاً پردازش شده است.")
            return {'success': False, 'reason': 'already_visited'}

        try:
            # دریافت صفحه
            response = self.crawler.request_manager.get(url)

            if not response.get('html'):
                logger.warning(f"⚠️ محتوای HTML برای {url} دریافت نشد.")
                self.crawl_state.add_failed(url, error="No HTML content")
                return {'success': False, 'reason': 'no_html_content'}

            # استخراج و ذخیره محتوا
            content_stored = self.extract_and_store_content(url, response['html'])

            # استخراج لینک‌های جدید
            links = extract_links(response['html'], url, internal_only=True)

            # به‌روزرسانی آمار
            with self.stats_lock:
                self.urls_processed += 1
                self.stats['total_urls_processed'] += 1
                self.stats['total_urls_found'] += len(links)
                self.stats['last_crawl_time'] = datetime.now().isoformat()

            # افزودن لینک‌های جدید به صف خزش
            new_links_added = 0
            for link in links:
                if not self.crawl_state.was_visited(link) and not self.crawl_state.is_in_progress(link):
                    # تشخیص نوع لینک
                    pattern = self.structure_discovery.get_url_pattern(link)
                    job_type = 'page'

                    if pattern:
                        if pattern.is_list:
                            job_type = 'list'
                        elif pattern.is_detail:
                            job_type = 'detail'

                    # محاسبه اولویت
                    priority = self._calculate_url_priority(link, job_type, depth + 1)

                    # افزودن به صف
                    self.crawler.add_job(link, depth=depth + 1, priority=priority, job_type=job_type)
                    new_links_added += 1

            # ثبت URL به عنوان بازدید شده
            self.crawl_state.add_visited(url)

            logger.info(f"✅ پردازش {url} با موفقیت انجام شد. {new_links_added} لینک جدید یافت شد.")

            return {
                'success': True,
                'url': url,
                'new_links': new_links_added,
                'content_stored': content_stored
            }

        except Exception as e:
            with self.stats_lock:
                self.stats['total_errors'] += 1

            logger.error(f"خطا در پردازش {url}: {str(e)}")
            self.crawl_state.add_failed(url, error=str(e))

            return {'success': False, 'reason': str(e)}

    def _calculate_url_priority(self, url: str, job_type: str, depth: int) -> int:
        """
        محاسبه اولویت URL برای خزش

        Args:
            url: آدرس URL
            job_type: نوع کار ('page', 'list', 'detail')
            depth: عمق URL در گراف خزش

        Returns:
            اولویت محاسبه شده (مقادیر کمتر، اولویت بالاتر)
        """
        # اولویت پایه بر اساس عمق
        priority = depth * 10

        # تنظیم اولویت بر اساس نوع صفحه
        if job_type == 'list':
            priority -= 20  # اولویت بالاتر برای صفحات لیستی
        elif job_type == 'detail':
            priority -= 10  # اولویت متوسط برای صفحات جزئیات

        # بررسی الگوهای URL مهم
        important_patterns = [
            '/legal/', '/law/', '/cases/', '/judgments/', '/attorneys/',
            '/حقوقی/', '/قانون/', '/قوانین/', '/پرونده/', '/دادگاه/', '/وکلا/'
        ]

        for pattern in important_patterns:
            if pattern in url:
                priority -= 5  # افزایش اولویت برای URL‌های مهم
                break

        return priority

    def crawl_worker(self):
        """تابع کارگر خزش که در نخ‌های جداگانه اجرا می‌شود"""
        logger.info(f"نخ کارگر خزش آغاز به کار کرد: {threading.current_thread().name}")

        while not self.stop_event.is_set():
            try:
                # دریافت کار از صف با زمان انتظار
                try:
                    job = self.crawler.job_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # پردازش URL
                self.process_url(job.url, job.depth)

                # اعلام تکمیل کار به صف
                self.crawler.job_queue.task_done()

            except Exception as e:
                logger.error(f"خطا در نخ کارگر خزش: {str(e)}")

    def start_crawl_workers(self) -> bool:
        """
        راه‌اندازی نخ‌های کارگر خزش

        Returns:
            آیا راه‌اندازی موفقیت‌آمیز بود؟
        """
        logger.info(f"در حال راه‌اندازی {self.max_threads} نخ کارگر خزش...")

        try:
            # راه‌اندازی نخ‌های کارگر
            for i in range(self.max_threads):
                thread = threading.Thread(
                    target=self.crawl_worker,
                    name=f"CrawlWorker-{i + 1}",
                    daemon=True
                )
                thread.start()
                logger.info(f"نخ کارگر {i + 1} آغاز به کار کرد.")

            logger.info(f"✅ {self.max_threads} نخ کارگر با موفقیت راه‌اندازی شدند.")
            return True

        except Exception as e:
            logger.error(f"خطا در راه‌اندازی نخ‌های کارگر: {str(e)}")
            return False

    def run(self):
        """اجرای اصلی مدیریت هوشمند خزش"""
        logger.info("🚀 آغاز اجرای مدیریت هوشمند خزش...")

        # بررسی اتصال به دیتابیس
        if not self.verify_database_connection():
            logger.critical("خروج به دلیل عدم اتصال به دیتابیس.")
            return

        # بررسی جداول دیتابیس
        if not self.verify_database_tables():
            logger.critical("خروج به دلیل مشکل در جداول دیتابیس.")
            return

        # راه‌اندازی سرویس‌ها
        if not self.initialize_services():
            logger.critical("خروج به دلیل مشکل در راه‌اندازی سرویس‌ها.")
            return

        # کشف ساختار وبسایت
        if not self.discover_site_structure():
            logger.warning("⚠️ کشف ساختار وبسایت با مشکل مواجه شد. ادامه با محدودیت...")

        # بارگذاری وضعیت قبلی
        self.load_state()

        # افزودن URL اولیه به صف خزش (اگر صف خالی است)
        if self.crawler.job_queue.empty():
            logger.info(f"افزودن URL اولیه به صف خزش: {self.base_url}")
            self.crawler.add_job(self.base_url, depth=0, job_type='page')

        # راه‌اندازی نخ‌های کارگر
        if not self.start_crawl_workers():
            logger.critical("خروج به دلیل مشکل در راه‌اندازی نخ‌های کارگر.")
            return

        # علامت‌گذاری شروع اجرا
        self.running = True

        try:
            # حلقه اصلی برنامه
            logger.info("⏱️ ورود به حلقه اصلی مدیریت هوشمند خزش...")

            while self.running and not self.stop_event.is_set():
                # گزارش وضعیت
                current_queue_size = self.crawler.job_queue.qsize()
                stats = self.crawl_state.get_stats()

                logger.info(f"📊 وضعیت خزش - فاز: {self.crawl_phase}, "
                            f"صف: {current_queue_size}, "
                            f"URLهای پردازش شده: {stats['successful_urls']}, "
                            f"محتوای جدید: {self.urls_new_content}")

                # به‌روزرسانی فاز خزش
                self.update_crawl_phase()

                # ذخیره وضعیت فعلی
                self.save_state()

                # استراتژی خواب
                sleep_minutes = self.get_current_sleep_time()
                logger.info(f"💤 استراحت برای {sleep_minutes:.2f} دقیقه...")

                # خواب با امکان وقفه
                for _ in range(int(sleep_minutes * 60)):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)

                # بررسی خالی بودن صف
                if current_queue_size == 0:
                    logger.info("صف خزش خالی است. افزودن URL اولیه برای بررسی تغییرات...")
                    self.crawler.add_job(self.base_url, depth=0, job_type='page')

            logger.info("خروج از حلقه اصلی مدیریت هوشمند خزش.")

        except KeyboardInterrupt:
            logger.info("دریافت سیگنال توقف از کاربر...")
        except Exception as e:
            logger.error(f"خطا در حلقه اصلی مدیریت هوشمند خزش: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """توقف مناسب تمام فعالیت‌ها"""
        if not self.running:
            return

        logger.info("در حال توقف مناسب مدیریت هوشمند خزش...")

        # علامت‌گذاری برای توقف
        self.stop_event.set()
        self.running = False

        # ذخیره وضعیت نهایی
        self.save_state()

        # توقف خزشگر
        if self.crawler:
            self.crawler.stop(wait=True, save_checkpoint=True)

        logger.info("✅ مدیریت هوشمند خزش با موفقیت متوقف شد.")


def parse_arguments():
    """پردازش آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(
        description="مدیریت هوشمند خزش داده‌های حقوقی"
    )
    parser.add_argument(
        "--base-url", type=str,
        default=os.getenv("INITIAL_URL", "https://www.bonyadvokala.com/"),
        help="آدرس پایه وبسایت هدف"
    )
    parser.add_argument(
        "--max-threads", type=int,
        default=int(os.getenv("MAX_THREADS", "4")),
        help="حداکثر تعداد نخ‌های همزمان"
    )
    parser.add_argument(
        "--delay", type=float,
        default=float(os.getenv("CRAWL_DELAY", "1.0")),
        help="تأخیر اولیه بین درخواست‌ها (ثانیه)"
    )
    parser.add_argument(
        "--no-robots", action="store_true",
        help="عدم رعایت محدودیت‌های robots.txt"
    )
    return parser.parse_args()


def main():
    """تابع اصلی برنامه"""
    # پردازش آرگومان‌ها
    args = parse_arguments()

    # ایجاد مدیریت هوشمند خزش
    crawler_manager = SmartCrawlManager(
        base_url=args.base_url,
        max_threads=args.max_threads,
        initial_delay=args.delay,
        respect_robots=not args.no_robots
    )

    # اجرای مدیریت هوشمند خزش
    crawler_manager.run()


if __name__ == "__main__":
    main()