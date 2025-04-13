"""
ماژول خزش هوشمند برای خزشگر داده‌های حقوقی

این ماژول شامل کلاس‌ها و توابع مربوط به مدیریت فرآیند خزش و استخراج داده از وبسایت‌ها است.
"""

import os
import time
import threading
import queue
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime
import xml.etree.ElementTree as ET

from utils.logger import get_logger, get_crawler_logger
from utils.http import RequestManager, normalize_url
from core.structure_discovery import StructureDiscovery
from utils.text import extract_links, extract_main_content, extract_title, extract_date, extract_author

# تنظیم لاگرها
logger = get_logger(__name__)
crawler_logger = get_crawler_logger()


class CrawlJob:
    """کلاس نمایش‌دهنده یک کار خزش"""

    def __init__(self, url, depth=0, priority=0, parent_url=None, referrer=None, job_type='page'):
        """
        مقداردهی اولیه کار خزش

        Args:
            url: آدرس URL برای خزش
            depth: عمق فعلی در گراف خزش
            priority: اولویت برای خزش (مقادیر پایین‌تر اولویت بالاتر دارند)
            parent_url: آدرس URL والد
            referrer: آدرس URL ارجاع‌دهنده
            job_type: نوع کار ('page', 'list', 'detail', 'api', 'sitemap', etc.)
        """
        self.url = url
        self.depth = depth
        self.priority = priority
        self.parent_url = parent_url
        self.referrer = referrer
        self.job_type = job_type
        self.created_at = datetime.now()

        # از URL، دامنه و مسیر استخراج می‌شود
        parsed = urlparse(url)
        self.domain = parsed.netloc
        self.path = parsed.path

    def __lt__(self, other):
        """
        مقایسه کمتر بودن برای استفاده در صف اولویت
        اولویت کمتر (عدد کوچکتر) به معنای اولویت بالاتر است

        Args:
            other: کار خزش دیگر برای مقایسه

        Returns:
            bool: آیا این کار از نظر اولویت کمتر است؟
        """
        return self.priority < other.priority

    def __eq__(self, other):
        """
        مقایسه برابری دو کار

        Args:
            other: کار خزش دیگر برای مقایسه

        Returns:
            bool: آیا دو کار برابرند؟
        """
        if not isinstance(other, CrawlJob):
            return False
        return self.url == other.url

    def __str__(self):
        """
        نمایش رشته‌ای کار

        Returns:
            str: رشته نمایشی
        """
        return f"CrawlJob(url='{self.url}', depth={self.depth}, priority={self.priority}, type='{self.job_type}')"

    def __repr__(self):
        """
        نمایش رشته‌ای کار برای توسعه‌دهندگان

        Returns:
            str: رشته نمایشی
        """
        return self.__str__()

    def get_info(self):
        """
        دریافت اطلاعات کار به صورت دیکشنری

        Returns:
            dict: اطلاعات کار
        """
        return {
            'url': self.url,
            'depth': self.depth,
            'priority': self.priority,
            'job_type': self.job_type,
            'domain': self.domain,
            'path': self.path,
            'parent_url': self.parent_url,
            'created_at': self.created_at.isoformat()
        }

    def is_high_priority(self):
        """
        بررسی اینکه آیا کار اولویت بالایی دارد

        Returns:
            bool: آیا کار اولویت بالایی دارد؟
        """
        # اولویت کمتر از صفر به معنای اولویت بالا است
        return self.priority < 0

    def is_sitemap(self):
        """
        بررسی اینکه آیا کار از نوع نقشه سایت است

        Returns:
            bool: آیا کار از نوع نقشه سایت است؟
        """
        return self.job_type == 'sitemap'

    def is_list_page(self):
        """
        بررسی اینکه آیا کار از نوع صفحه لیستی است

        Returns:
            bool: آیا کار از نوع صفحه لیستی است؟
        """
        return self.job_type == 'list'

    def is_detail_page(self):
        """
        بررسی اینکه آیا کار از نوع صفحه جزئیات است

        Returns:
            bool: آیا کار از نوع صفحه جزئیات است؟
        """
        return self.job_type == 'detail'


class CrawlState:
    """کلاس مدیریت وضعیت خزش"""

    def __init__(self, max_urls=10000, checkpoint_file=None):
        """
        مقداردهی اولیه وضعیت خزش

        Args:
            max_urls: حداکثر تعداد URL برای ذخیره‌سازی در تاریخچه
            checkpoint_file: مسیر فایل برای ذخیره نقاط بازیابی
        """
        self.visited_urls = set()
        self.url_history = {}  # نگاشت URL به زمان و وضعیت بازدید
        self.failed_urls = {}  # نگاشت URL به تعداد تلاش‌ها و خطاهای مربوطه
        self.in_progress = set()  # URLهای در حال پردازش
        self.checkpointed = False  # آیا نقطه بازیابی ایجاد شده است؟
        self.max_urls = max_urls
        self.checkpoint_file = checkpoint_file

        # آمار و اطلاعات
        self.stats = {
            'total_urls': 0,
            'successful_urls': 0,
            'failed_urls': 0,
            'skipped_urls': 0,
            'start_time': datetime.now(),
            'last_update_time': datetime.now()
        }

        # قفل‌ها برای همگام‌سازی چندنخی
        self.state_lock = threading.RLock()

    def add_visited(self, url, status_code=200, content_type=None):
        """
        افزودن یک URL به لیست بازدید شده

        Args:
            url: آدرس URL بازدید شده
            status_code: کد وضعیت HTTP
            content_type: نوع محتوای دریافت شده

        Returns:
            None
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            self.visited_urls.add(normalized_url)

            self.url_history[normalized_url] = {
                'visited_at': datetime.now(),
                'status_code': status_code,
                'content_type': content_type
            }

            if normalized_url in self.in_progress:
                self.in_progress.remove(normalized_url)

            # به‌روزرسانی آمار
            self.stats['total_urls'] += 1
            self.stats['successful_urls'] += 1
            self.stats['last_update_time'] = datetime.now()

            # کنترل اندازه تاریخچه
            if len(self.url_history) > self.max_urls:
                # حذف قدیمی‌ترین URL‌ها
                oldest = sorted(self.url_history.items(), key=lambda x: x[1]['visited_at'])[:100]
                for old_url, _ in oldest:
                    del self.url_history[old_url]

    def add_failed(self, url, error=None, status_code=None):
        """
        افزودن یک URL به لیست شکست‌خورده

        Args:
            url: آدرس URL ناموفق
            error: پیام خطا (اختیاری)
            status_code: کد وضعیت HTTP (اختیاری)

        Returns:
            None
        """
        with self.state_lock:
            normalized_url = normalize_url(url)

            if normalized_url in self.failed_urls:
                self.failed_urls[normalized_url]['attempts'] += 1
                self.failed_urls[normalized_url]['last_error'] = error
                self.failed_urls[normalized_url]['last_status_code'] = status_code
                self.failed_urls[normalized_url]['last_attempt'] = datetime.now()
            else:
                self.failed_urls[normalized_url] = {
                    'attempts': 1,
                    'first_attempt': datetime.now(),
                    'last_attempt': datetime.now(),
                    'last_error': error,
                    'last_status_code': status_code
                }

            if normalized_url in self.in_progress:
                self.in_progress.remove(normalized_url)

            # به‌روزرسانی آمار
            self.stats['total_urls'] += 1
            self.stats['failed_urls'] += 1
            self.stats['last_update_time'] = datetime.now()

    def add_in_progress(self, url):
        """
        افزودن یک URL به لیست در حال پردازش

        Args:
            url: آدرس URL

        Returns:
            None
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            self.in_progress.add(normalized_url)

    def was_visited(self, url):
        """
        بررسی آیا یک URL قبلاً بازدید شده است

        Args:
            url: آدرس URL برای بررسی

        Returns:
            bool: آیا URL قبلاً بازدید شده است؟
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            return normalized_url in self.visited_urls

    def is_in_progress(self, url):
        """
        بررسی آیا یک URL در حال پردازش است

        Args:
            url: آدرس URL برای بررسی

        Returns:
            bool: آیا URL در حال پردازش است؟
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            return normalized_url in self.in_progress

    def was_failed(self, url):
        """
        بررسی آیا یک URL قبلاً ناموفق بوده است

        Args:
            url: آدرس URL برای بررسی

        Returns:
            bool: آیا URL قبلاً ناموفق بوده است؟
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            return normalized_url in self.failed_urls

    def should_retry(self, url, max_retries=3):
        """
        بررسی آیا یک URL باید مجدداً امتحان شود

        Args:
            url: آدرس URL برای بررسی
            max_retries: حداکثر تعداد تلاش‌ها

        Returns:
            bool: آیا URL باید مجدداً امتحان شود؟
        """
        with self.state_lock:
            normalized_url = normalize_url(url)
            if normalized_url not in self.failed_urls:
                return True

            return self.failed_urls[normalized_url]['attempts'] < max_retries

    def get_stats(self):
        """
        دریافت آمار فعلی خزش

        Returns:
            dict: دیکشنری آمار
        """
        with self.state_lock:
            # محاسبه زمان سپری شده
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            rate = self.stats['successful_urls'] / max(elapsed, 1) * 60

            stats = self.stats.copy()
            stats['elapsed_seconds'] = int(elapsed)
            stats['urls_per_minute'] = int(rate)

            return stats

    def save_checkpoint(self, checkpoint_file=None):
        """
        ذخیره وضعیت فعلی در یک نقطه بازیابی

        Args:
            checkpoint_file: مسیر فایل برای ذخیره‌سازی (اختیاری)

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        file_path = checkpoint_file or self.checkpoint_file
        if not file_path:
            logger.warning("مسیر فایل نقطه بازیابی مشخص نشده است")
            return False

        try:
            with self.state_lock:
                checkpoint_data = {
                    'visited_urls': list(self.visited_urls),
                    'failed_urls': self.failed_urls,
                    'stats': self.stats,
                    'checkpoint_time': datetime.now().isoformat()
                }

                # ایجاد پوشه در صورت نیاز
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False)

                self.checkpointed = True
                logger.info(f"نقطه بازیابی در {file_path} ذخیره شد")
                return True

        except Exception as e:
            logger.error(f"خطا در ذخیره نقطه بازیابی: {str(e)}")
            return False

    def load_checkpoint(self, checkpoint_file=None):
        """
        بارگذاری وضعیت از یک نقطه بازیابی

        Args:
            checkpoint_file: مسیر فایل برای بارگذاری (اختیاری)

        Returns:
            bool: آیا بارگذاری موفق بود؟
        """
        file_path = checkpoint_file or self.checkpoint_file
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"فایل نقطه بازیابی یافت نشد: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            with self.state_lock:
                self.visited_urls = set(checkpoint_data.get('visited_urls', []))
                self.failed_urls = checkpoint_data.get('failed_urls', {})
                loaded_stats = checkpoint_data.get('stats', {})

                # ترکیب آمار قدیمی با جدید
                self.stats = {
                    'total_urls': loaded_stats.get('total_urls', 0),
                    'successful_urls': loaded_stats.get('successful_urls', 0),
                    'failed_urls': loaded_stats.get('failed_urls', 0),
                    'skipped_urls': loaded_stats.get('skipped_urls', 0),
                    'start_time': datetime.fromisoformat(loaded_stats.get('start_time'))
                    if isinstance(loaded_stats.get('start_time'), str)
                    else datetime.now(),
                    'last_update_time': datetime.now()
                }

                if 'url_history' in checkpoint_data:
                    self.url_history = checkpoint_data['url_history']

                logger.info(f"نقطه بازیابی از {file_path} بارگذاری شد")
                logger.info(f"{len(self.visited_urls)} URL بازدید شده و {len(self.failed_urls)} URL ناموفق بازیابی شد")

                self.checkpointed = True
                return True

        except Exception as e:
            logger.error(f"خطا در بارگذاری نقطه بازیابی: {str(e)}")
            return False


class URLPriorityPolicyManager:
    """کلاس مدیریت سیاست‌های اولویت‌بندی URL"""

    def __init__(self):
        """مقداردهی اولیه مدیریت سیاست‌ها"""
        self.policies = []

    def add_policy(self, name, condition_func, priority_func, weight=1.0, enabled=True):
        """
        افزودن یک سیاست اولویت‌بندی جدید

        Args:
            name: نام سیاست
            condition_func: تابع شرط برای اعمال سیاست (url, job) -> bool
            priority_func: تابع محاسبه اولویت (url, job) -> int
            weight: وزن این سیاست
            enabled: آیا سیاست فعال است؟

        Returns:
            None
        """
        self.policies.append({
            'name': name,
            'condition': condition_func,
            'priority': priority_func,
            'weight': weight,
            'enabled': enabled
        })

    def calculate_priority(self, url, job=None):
        """
        محاسبه اولویت یک URL با استفاده از سیاست‌های فعال

        Args:
            url: آدرس URL برای محاسبه اولویت
            job: شیء کار خزش (اختیاری)

        Returns:
            int: اولویت محاسبه شده
        """
        priority = 0
        total_weight = 0

        for policy in self.policies:
            if not policy['enabled']:
                continue

            if policy['condition'](url, job):
                priority_value = policy['priority'](url, job)
                priority += priority_value * policy['weight']
                total_weight += policy['weight']

        if total_weight > 0:
            priority = priority / total_weight

        return int(priority)

    def get_default_policies(self):
        """
        ایجاد سیاست‌های پیش‌فرض

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # سیاست عمق: اولویت پایین‌تر برای URLهای عمیق‌تر
        self.add_policy(
            name="depth_policy",
            condition_func=lambda url, job: job is not None,
            priority_func=lambda url, job: job.depth * 10,
            weight=1.0
        )

        # سیاست لیست: اولویت بالاتر برای صفحات لیستی
        self.add_policy(
            name="list_policy",
            condition_func=lambda url, job: job is not None and job.job_type == 'list',
            priority_func=lambda url, job: -20,
            weight=1.5
        )

        # سیاست جزئیات: اولویت متوسط برای صفحات جزئیات
        self.add_policy(
            name="detail_policy",
            condition_func=lambda url, job: job is not None and job.job_type == 'detail',
            priority_func=lambda url, job: -10,
            weight=1.0
        )

        # سیاست sitemap: اولویت بالا برای سایت‌مپ‌ها
        self.add_policy(
            name="sitemap_policy",
            condition_func=lambda url, job: job is not None and job.job_type == 'sitemap',
            priority_func=lambda url, job: -30,  # اولویت بسیار بالا
            weight=2.0
        )

        # سیاست مسیر کوتاه: اولویت بالاتر برای URL‌های با مسیر کوتاهتر
        self.add_policy(
            name="path_length_policy",
            condition_func=lambda url, job: True,
            priority_func=lambda url, job: urlparse(url).path.count('/') * 5,
            weight=0.8
        )

        return self


class Crawler:
    """کلاس اصلی خزشگر هوشمند"""

    def __init__(self, base_url, config_dir=None, max_threads=4, max_depth=5, politeness_delay=1.0,
                 respect_robots=False, use_db_storage=True):
        """
        مقداردهی اولیه خزشگر

        Args:
            base_url: آدرس پایه وبسایت
            config_dir: مسیر پوشه پیکربندی (اختیاری)
            max_threads: حداکثر تعداد نخ‌های همزمان
            max_depth: حداکثر عمق خزش
            politeness_delay: تأخیر بین درخواست‌ها برای رعایت ادب (ثانیه)
            respect_robots: آیا محدودیت‌های robots.txt رعایت شود؟
            use_db_storage: آیا داده‌ها در دیتابیس ذخیره شوند؟
        """
        self.base_url = base_url
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config'
        )

        # ایجاد پوشه پیکربندی در صورت عدم وجود
        os.makedirs(self.config_dir, exist_ok=True)

        # پارامترهای خزش
        self.max_threads = int(os.getenv('MAX_THREADS', max_threads))
        self.max_depth = int(os.getenv('MAX_DEPTH', max_depth))
        self.politeness_delay = float(os.getenv('CRAWL_DELAY', politeness_delay))
        self.respect_robots = respect_robots
        self.use_db_storage = use_db_storage

        # تنظیم دامنه اصلی
        self.domain = urlparse(base_url).netloc

        # مسیر فایل‌های پیکربندی و نقاط بازیابی
        self.checkpoint_file = os.path.join(self.config_dir, f"{self.domain}_crawl_state.json")

        # کشف ساختار وبسایت
        self.structure_discovery = StructureDiscovery(base_url, self.config_dir)

        # مدیریت سیاست‌های اولویت‌بندی
        self.priority_manager = URLPriorityPolicyManager()
        self.priority_manager.get_default_policies()

        # صف کار و وضعیت خزش
        self.job_queue = queue.PriorityQueue()
        self.crawl_state = CrawlState(checkpoint_file=self.checkpoint_file)

        # مدیریت درخواست‌ها
        self.request_manager = RequestManager(
            base_url=base_url,
            default_delay=self.politeness_delay,
            respect_robots=False  # همیشه محدودیت‌های robots.txt را نادیده می‌گیریم
        )

        # متغیرهای کنترل خزش
        self.running = False
        self.threads = []
        self.stop_event = threading.Event()

        # آمار و اطلاعات اضافی
        self.stats_lock = threading.Lock()
        self.max_queue_size = 0
        self.last_job_time = datetime.now()
        self.checkpoint_interval = 300  # ۵ دقیقه
        self.last_checkpoint_time = datetime.now()

        # آمار ذخیره‌سازی
        self.storage_stats = {
            'stored_content_count': 0,
            'failed_storage_count': 0,
            'stored_by_type': {}
        }

        # اضافه کردن ماژول‌های استخراج، طبقه‌بندی و ذخیره‌سازی
        if self.use_db_storage:
            try:
                from core.content_extractor import ContentExtractor
                from core.classifier import TextClassifier
                from core.storage import StorageManager

                self.content_extractor = ContentExtractor()
                self.classifier = TextClassifier()
                self.storage_manager = StorageManager()

                logger.info("ماژول‌های استخراج، طبقه‌بندی و ذخیره‌سازی با موفقیت بارگذاری شدند")
            except Exception as e:
                logger.error(f"خطا در بارگذاری ماژول‌های استخراج، طبقه‌بندی یا ذخیره‌سازی: {str(e)}")
                self.use_db_storage = False

    def extract_sitemap_from_robots(self):
        """
        استخراج آدرس sitemap از فایل robots.txt

        Returns:
            list: لیست آدرس‌های sitemap یا None در صورت خطا
        """
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            logger.info(f"در حال استخراج sitemap از {robots_url}...")

            response = self.request_manager.get(robots_url, use_selenium=False)

            if not response.get('html'):
                logger.warning(f"فایل robots.txt در {robots_url} یافت نشد")
                return None

            # جستجوی خط‌های sitemap در فایل robots.txt
            sitemap_urls = []
            for line in response.get('html').splitlines():
                line = line.strip()
                if line.lower().startswith('sitemap:'):
                    sitemap_url = line.split(':', 1)[1].strip()
                    logger.info(f"آدرس sitemap یافت شد: {sitemap_url}")
                    sitemap_urls.append(sitemap_url)

            return sitemap_urls
        except Exception as e:
            logger.error(f"خطا در استخراج sitemap: {str(e)}")
            return None

    def discover_site_structure(self, force=False):
        """
        کشف ساختار وبسایت

        Args:
            force: آیا کشف مجدد انجام شود حتی اگر قبلاً انجام شده؟

        Returns:
            bool: آیا عملیات موفق بود؟
        """
        if force or not self.structure_discovery.is_discovered():
            return self.structure_discovery.discover_structure()
        return True

    def add_job(self, url, depth=0, priority=None, parent_url=None, job_type=None):
        """
        افزودن یک کار جدید به صف

        Args:
            url: آدرس URL برای خزش
            depth: عمق URL در گراف خزش
            priority: اولویت (اختیاری، محاسبه خودکار در صورت عدم ارائه)
            parent_url: آدرس URL والد (اختیاری)
            job_type: نوع کار ('page', 'list', 'detail', 'sitemap', etc.)

        Returns:
            bool: آیا کار با موفقیت اضافه شد؟
        """
        # نرمالایز کردن URL
        normalized_url = normalize_url(url)

        # اگر نوع کار sitemap است، بدون بررسی اضافه می‌کنیم
        if job_type != 'sitemap':
            # بررسی آیا این URL قبلاً بازدید شده یا در صف است
            if self.crawl_state.was_visited(normalized_url) or self.crawl_state.is_in_progress(normalized_url):
                self.crawl_state.stats['skipped_urls'] += 1
                return False

            # بررسی محدودیت عمق
            if depth > self.max_depth:
                self.crawl_state.stats['skipped_urls'] += 1
                return False

            # بررسی محدودیت دامنه (فقط URL‌های داخلی)
            if self.domain != urlparse(normalized_url).netloc:
                self.crawl_state.stats['skipped_urls'] += 1
                return False

        # تشخیص نوع کار اگر ارائه نشده
        if job_type is None:
            pattern = self.structure_discovery.get_url_pattern(normalized_url)
            if pattern:
                if pattern.is_list:
                    job_type = 'list'
                elif pattern.is_detail:
                    job_type = 'detail'
                else:
                    job_type = 'page'
            else:
                job_type = 'page'

        # محاسبه اولویت اگر ارائه نشده
        if priority is None:
            temp_job = CrawlJob(normalized_url, depth, 0, parent_url, parent_url, job_type)
            priority = self.priority_manager.calculate_priority(normalized_url, temp_job)

        # ایجاد کار جدید
        job = CrawlJob(normalized_url, depth, priority, parent_url, parent_url, job_type)

        # افزودن به صف
        self.job_queue.put(job)

        # به‌روزرسانی آمارها
        with self.stats_lock:
            queue_size = self.job_queue.qsize()
            if queue_size > self.max_queue_size:
                self.max_queue_size = queue_size

        return True

    def process_job(self, job):
        """
        پردازش یک کار خزش

        Args:
            job: کار خزش برای پردازش

        Returns:
            dict: نتیجه پردازش
        """
        url = job.url

        # ثبت URL به عنوان در حال پردازش
        self.crawl_state.add_in_progress(url)

        try:
            # لاگ اطلاعات کار
            crawler_logger.info(f"خزش {url} (عمق {job.depth}, اولویت {job.priority}, نوع {job.job_type})")

            # اگر کار از نوع sitemap است، به طور ویژه پردازش می‌کنیم
            if job.job_type == 'sitemap':
                return self._process_sitemap_job(job)

            # درخواست صفحه
            use_selenium = job.job_type in ['list', 'detail']  # استفاده از سلنیوم برای صفحات پیچیده‌تر
            response = self.request_manager.get(url, use_selenium=use_selenium)

            # بررسی موفقیت
            if not response.get('html') or (response.get('status_code') and response.get('status_code') >= 400):
                error_message = f"خطا در دریافت {url}: {response.get('error') or response.get('status_code')}"
                crawler_logger.error(error_message)

                self.crawl_state.add_failed(
                    url,
                    error=response.get('error'),
                    status_code=response.get('status_code')
                )

                return {
                    'success': False,
                    'url': url,
                    'error': error_message,
                    'status_code': response.get('status_code')
                }

            # پردازش موفق
            html_content = response.get('html')
            soup = response.get('soup')
            final_url = response.get('url')  # URL نهایی پس از ریدایرکت

            # ذخیره نتایج استخراج و طبقه‌بندی
            storage_result = None
            extracted_data = None
            classification_result = None

            # استخراج و طبقه‌بندی محتوا با استفاده از ماژول‌های جدید
            if self.use_db_storage and hasattr(self, 'content_extractor') and hasattr(self, 'classifier'):
                try:
                    # استخراج محتوا
                    extracted_data = self.content_extractor.extract(html_content, final_url, job_type=job.job_type)

                    # طبقه‌بندی محتوا
                    if 'content' in extracted_data and extracted_data['content']:
                        classification_result = self.classifier.classify_text(extracted_data['content'])

                        # افزودن نتایج طبقه‌بندی به داده‌های استخراج شده
                        if 'content_type' in classification_result:
                            extracted_data['content_type'] = classification_result['content_type'].get('content_type',
                                                                                                       'other')

                        if 'domains' in classification_result:
                            extracted_data['domains'] = classification_result['domains'].get('domains', [])

                    # ذخیره محتوا در دیتابیس
                    if hasattr(self, 'storage_manager'):
                        storage_result = self.storage_manager.store_content(extracted_data)

                        if storage_result:
                            logger.info(f"محتوای {url} با موفقیت در دیتابیس ذخیره شد (ID: {storage_result.id})")

                            # به‌روزرسانی آمار ذخیره‌سازی
                            with self.stats_lock:
                                self.storage_stats['stored_content_count'] += 1
                                content_type = extracted_data.get('content_type', 'other')
                                self.storage_stats['stored_by_type'][content_type] = self.storage_stats[
                                                                                         'stored_by_type'].get(
                                    content_type, 0) + 1
                        else:
                            logger.warning(f"محتوای {url} در دیتابیس ذخیره نشد")
                            with self.stats_lock:
                                self.storage_stats['failed_storage_count'] += 1

                except Exception as e:
                    logger.error(f"خطا در استخراج، طبقه‌بندی یا ذخیره‌سازی محتوای {url}: {str(e)}")
                    with self.stats_lock:
                        self.storage_stats['failed_storage_count'] += 1

            # اگر استخراج با ماژول‌های جدید انجام نشد، از روش قدیمی استفاده می‌کنیم
            if extracted_data is None:
                extracted_data = self._extract_page_data(final_url, html_content, soup, job.job_type)

            # استخراج لینک‌های جدید
            new_links = []

            if job.depth < self.max_depth:
                links = extract_links(html_content, final_url, internal_only=True)

                # افزودن لینک‌های جدید به صف
                for link in links:
                    normalized_link = normalize_url(link)

                    # بررسی محدودیت‌ها
                    if self.domain != urlparse(normalized_link).netloc:
                        continue

                    if (self.crawl_state.was_visited(normalized_link) or
                            self.crawl_state.is_in_progress(normalized_link)):
                        continue

                    # تشخیص نوع لینک
                    pattern = self.structure_discovery.get_url_pattern(normalized_link)
                    link_job_type = 'page'

                    if pattern:
                        if pattern.is_list:
                            link_job_type = 'list'
                        elif pattern.is_detail:
                            link_job_type = 'detail'

                    # محاسبه اولویت
                    temp_job = CrawlJob(normalized_link, job.depth + 1, 0, final_url, final_url, link_job_type)
                    priority = self.priority_manager.calculate_priority(normalized_link, temp_job)

                    # ایجاد و افزودن کار جدید
                    new_job = CrawlJob(
                        normalized_link,
                        job.depth + 1,
                        priority,
                        final_url,
                        final_url,
                        link_job_type
                    )

                    self.job_queue.put(new_job)
                    new_links.append(normalized_link)

                    # به‌روزرسانی آمار صف
                    with self.stats_lock:
                        queue_size = self.job_queue.qsize()
                        if queue_size > self.max_queue_size:
                            self.max_queue_size = queue_size

            # ثبت URL به عنوان بازدید شده
            self.crawl_state.add_visited(
                url,
                status_code=response.get('status_code'),
                content_type='text/html'
            )

            # به‌روزرسانی زمان آخرین کار
            self.last_job_time = datetime.now()

            result = {
                'success': True,
                'url': url,
                'final_url': final_url,
                'job_type': job.job_type,
                'depth': job.depth,
                'status_code': response.get('status_code'),
                'extracted_data': extracted_data,
                'new_links': new_links,
                'new_links_count': len(new_links)
            }

            # افزودن اطلاعات طبقه‌بندی و ذخیره‌سازی به نتیجه
            if classification_result:
                result['classification'] = classification_result

            if storage_result:
                result['stored'] = True
                result['stored_id'] = storage_result.id

            return result

        except Exception as e:
            error_message = f"خطا در پردازش {url}: {str(e)}"
            logger.error(error_message)
            crawler_logger.error(error_message)

            self.crawl_state.add_failed(url, error=str(e))

            return {
                'success': False,
                'url': url,
                'error': error_message
            }

    def worker(self):
        """تابع اجرایی نخ‌های کارگر"""
        while not self.stop_event.is_set():
            try:
                # گرفتن یک کار از صف با زمان انتظار
                try:
                    job = self.job_queue.get(timeout=5)
                except queue.Empty:
                    # اگر صف خالی است، ادامه می‌دهیم بدون فراخوانی task_done
                    continue

                try:
                    # پردازش کار
                    result = self.process_job(job)

                    # لاگ نتیجه
                    if result['success']:
                        crawler_logger.info(
                            f"خزش موفق {job.url} - استخراج {result.get('new_links_count', 0) if 'new_links_count' in result else result.get('extracted_urls', 0)} لینک جدید"
                        )
                    else:
                        crawler_logger.warning(f"خزش ناموفق {job.url} - {result.get('error')}")

                    # بررسی زمان ذخیره نقطه بازیابی
                    now = datetime.now()
                    if (now - self.last_checkpoint_time).total_seconds() > self.checkpoint_interval:
                        self.crawl_state.save_checkpoint()
                        self.last_checkpoint_time = now
                finally:
                    # اعلام تکمیل کار به صف - فقط یک بار فراخوانی می‌شود، در بلوک finally
                    self.job_queue.task_done()

            except Exception as e:
                logger.error(f"خطا در نخ کارگر: {str(e)}")
                # در صورت خطا، نباید task_done را فراخوانی کنیم زیرا قبلاً این کار را انجام داده‌ایم

    def start(self, initial_urls=None, load_checkpoint=True):
        """
        شروع فرآیند خزش

        Args:
            initial_urls: لیست آدرس‌های URL اولیه (اختیاری)
            load_checkpoint: آیا نقطه بازیابی بارگذاری شود؟

        Returns:
            bool: آیا شروع خزش موفق بود؟
        """
        if self.running:
            logger.warning("خزشگر از قبل در حال اجراست")
            return False

        # کشف ساختار سایت
        self.discover_site_structure()

        # بارگذاری نقطه بازیابی
        if load_checkpoint:
            self.crawl_state.load_checkpoint()

        # ابتدا تلاش برای استخراج sitemap
        sitemap_urls = self.extract_sitemap_from_robots()

        # افزودن URL‌های اولیه
        if sitemap_urls:
            logger.info(f"استفاده از {len(sitemap_urls)} sitemap به عنوان نقاط شروع...")
            for sitemap_url in sitemap_urls:
                self.add_job(sitemap_url, depth=0, job_type='sitemap')
        elif not initial_urls:
            logger.info("استفاده از آدرس پایه به عنوان نقطه شروع...")
            initial_urls = [self.base_url]
            for url in initial_urls:
                self.add_job(url, depth=0, job_type='page')
        else:
            logger.info(f"استفاده از {len(initial_urls)} آدرس اولیه...")
            for url in initial_urls:
                self.add_job(url, depth=0, job_type='page')

        # راه‌اندازی نخ‌های کارگر
        self.stop_event.clear()
        self.threads = []

        for i in range(self.max_threads):
            thread = threading.Thread(
                target=self.worker,
                name=f"CrawlerWorker-{i + 1}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)

        self.running = True
        logger.info(f"خزشگر با {self.max_threads} نخ شروع به کار کرد")

        return True

    def stop(self, wait=True, save_checkpoint=True):
        """
        توقف فرآیند خزش

        Args:
            wait: آیا منتظر اتمام کارهای در حال انجام بمانیم؟
            save_checkpoint: آیا نقطه بازیابی ذخیره شود؟

        Returns:
            bool: آیا توقف خزش موفق بود؟
        """
        if not self.running:
            logger.warning("خزشگر در حال اجرا نیست")
            return False

        # علامت‌گذاری برای توقف
        self.stop_event.set()

        # انتظار برای اتمام کارها
        if wait and self.threads:
            for thread in self.threads:
                thread.join(timeout=10)

        # ذخیره نقطه بازیابی
        if save_checkpoint:
            self.crawl_state.save_checkpoint()

        # بستن منابع
        self.request_manager.close()

        self.running = False
        logger.info("خزشگر متوقف شد")

        return True

    def get_stats(self):
        """
        دریافت آمار خزش و ذخیره‌سازی

        Returns:
            dict: دیکشنری آمار
        """
        stats = self.crawl_state.get_stats()

        # افزودن آمار اضافی
        with self.stats_lock:
            stats['max_queue_size'] = self.max_queue_size
            stats['current_queue_size'] = self.job_queue.qsize()
            stats['active_threads'] = sum(1 for thread in self.threads if thread.is_alive())

            # افزودن آمار ذخیره‌سازی
            stats['storage'] = self.storage_stats

            # افزودن آمار استخراج محتوا
            if self.use_db_storage and hasattr(self, 'content_extractor'):
                try:
                    stats['extraction'] = self.content_extractor.get_stats()
                except Exception:
                    pass

            # افزودن آمار ذخیره‌سازی از StorageManager
            if self.use_db_storage and hasattr(self, 'storage_manager'):
                try:
                    storage_stats = self.storage_manager.get_stats()
                    stats['storage_manager'] = storage_stats
                except Exception:
                    pass

        return stats

    def is_running(self):
        """
        بررسی وضعیت اجرای خزشگر

        Returns:
            bool: آیا خزشگر در حال اجراست؟
        """
        return self.running and any(thread.is_alive() for thread in self.threads)

    def wait_for_completion(self, timeout=None):
        """
        انتظار برای تکمیل تمام کارها

        Args:
            timeout: زمان انتظار حداکثر (ثانیه، None برای نامحدود)

        Returns:
            bool: آیا همه کارها تکمیل شدند؟
        """
        if not self.running:
            return True

        start_time = time.time()

        try:
            # انتظار برای تکمیل همه کارهای صف
            self.job_queue.join()
            return True
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"وقفه در انتظار برای تکمیل: {str(e)}")
            return False

    def __del__(self):
        """آزادسازی منابع هنگام حذف شیء"""
        if self.running:
            self.stop(wait=False)

    def join(self):
        """انتظار برای اتمام کارها"""
        try:
            self.job_queue.join()
        except (KeyboardInterrupt, Exception) as e:
            logger.error(f"وقفه در انتظار برای اتمام کارها: {str(e)}")
            return False
        return True

    def _extract_page_data(self, url, html_content, soup, job_type):
        """
        استخراج داده‌های صفحه بر اساس نوع آن

        Args:
            url: آدرس URL صفحه
            html_content: محتوای HTML صفحه
            soup: شیء BeautifulSoup
            job_type: نوع صفحه ('page', 'list', 'detail')

        Returns:
            dict: داده‌های استخراج شده
        """
        # استخراج اطلاعات پایه
        data = {
            'url': url,
            'type': job_type,
            'title': extract_title(html_content),
            'date': extract_date(html_content),
            'author': extract_author(html_content)
        }

        # دریافت سلکتورهای HTML متناسب با نوع صفحه
        selectors = self.structure_discovery.get_html_selectors(url, job_type)

        if job_type == 'list':
            # استخراج محتوای لیستی
            items = []

            if soup and selectors and 'container' in selectors and 'item' in selectors:
                container_selector = selectors['container']
                item_selector = selectors['item']

                container = soup.select_one(container_selector)
                if container:
                    for item_element in container.select(item_selector):
                        item_data = {}

                        # استخراج عنوان
                        if 'title' in selectors:
                            title_element = item_element.select_one(selectors['title'])
                            if title_element:
                                item_data['title'] = title_element.get_text().strip()

                        # استخراج لینک
                        if 'link' in selectors:
                            link_element = item_element.select_one(selectors['link'])
                            if link_element and link_element.has_attr('href'):
                                href = link_element['href']
                                item_data['link'] = urljoin(url, href)

                        # استخراج خلاصه
                        if 'summary' in selectors:
                            summary_element = item_element.select_one(selectors['summary'])
                            if summary_element:
                                item_data['summary'] = summary_element.get_text().strip()

                        items.append(item_data)

            data['items'] = items
            data['items_count'] = len(items)

            # استخراج اطلاعات صفحه‌بندی
            if soup and selectors and 'pagination' in selectors:
                pagination_selector = selectors['pagination']
                pagination_element = soup.select_one(pagination_selector)

                if pagination_element:
                    data['has_pagination'] = True

                    # استخراج لینک‌های صفحه‌بندی
                    pagination_links = []

                    if 'pagination_links' in selectors:
                        links_selector = selectors['pagination_links']
                        for link in pagination_element.select(links_selector):
                            if link.has_attr('href'):
                                href = link['href']
                                pagination_links.append(urljoin(url, href))

                    data['pagination_links'] = pagination_links
            else:
                data['has_pagination'] = False

        elif job_type == 'detail':
            # استخراج محتوای صفحه جزئیات
            if soup and selectors:
                # استخراج محتوای اصلی
                if 'content' in selectors:
                    content_selector = selectors['content']
                    content_element = soup.select_one(content_selector)

                    if content_element:
                        data['content'] = content_element.get_text().strip()
                        data['content_html'] = str(content_element)
                else:
                    # استخراج محتوای اصلی با روش‌های عمومی
                    data['content'] = extract_main_content(html_content)

                # استخراج تاریخ از سلکتور اختصاصی
                if 'date' in selectors and not data.get('date'):
                    date_selector = selectors['date']
                    date_element = soup.select_one(date_selector)

                    if date_element:
                        data['date'] = date_element.get_text().strip()

                # استخراج نویسنده از سلکتور اختصاصی
                if 'author' in selectors and not data.get('author'):
                    author_selector = selectors['author']
                    author_element = soup.select_one(author_selector)

                    if author_element:
                        data['author'] = author_element.get_text().strip()
        else:
            # صفحه عمومی - استخراج محتوای اصلی
            data['content'] = extract_main_content(html_content)

        return data

    def _process_sitemap_job(self, job):
        """
        پردازش یک کار از نوع sitemap

        Args:
            job: کار خزش برای پردازش

        Returns:
            dict: نتیجه پردازش
        """
        url = job.url

        try:
            # درخواست فایل sitemap
            response = self.request_manager.get(url, use_selenium=False)

            if not response.get('html'):
                self.crawl_state.add_failed(url, error="محتوای HTML دریافت نشد")
                return {
                    'success': False,
                    'url': url,
                    'error': "محتوای HTML دریافت نشد"
                }

            # پردازش فایل sitemap
            try:
                root = ET.fromstring(response.get('html'))

                # تعیین namespace
                namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                # استخراج URLs
                urls = []

                # بررسی اگر این یک فایل sitemap index است
                sitemap_tags = root.findall('.//sm:sitemap/sm:loc', namespaces)
                if sitemap_tags:
                    # این یک sitemap index است، باید هر فایل sitemap را جداگانه پردازش کنیم
                    logger.info(f"فایل {url} یک sitemap index با {len(sitemap_tags)} sitemap است")
                    for sitemap_tag in sitemap_tags:
                        sitemap_url = sitemap_tag.text.strip()
                        self.add_job(sitemap_url, depth=job.depth + 1, job_type='sitemap')
                        urls.append(sitemap_url)
                else:
                    # این یک sitemap عادی است، آدرس‌های URL را استخراج می‌کنیم
                    url_tags = root.findall('.//sm:url/sm:loc', namespaces)
                    logger.info(f"فایل {url} یک sitemap با {len(url_tags)} آدرس است")
                    for url_tag in url_tags:
                        page_url = url_tag.text.strip()
                        self.add_job(page_url, depth=0, job_type='page')
                        urls.append(page_url)

                # ثبت URL به عنوان بازدید شده
                self.crawl_state.add_visited(url, status_code=response.get('status_code'), content_type='application/xml')

                return {
                    'success': True,
                    'url': url,
                    'final_url': response.get('url'),
                    'job_type': 'sitemap',
                    'depth': job.depth,
                    'extracted_urls': len(urls),
                    'sitemap_type': 'index' if sitemap_tags else 'regular'
                }

            except ET.ParseError:
                # اگر XML نامعتبر است، ممکن است یک فایل متنی ساده باشد
                lines = response.get('html').splitlines()
                urls = []

                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):  # نادیده گرفتن خطوط توضیح
                        urls.append(line)
                        self.add_job(line, depth=0, job_type='page')

                # ثبت URL به عنوان بازدید شده
                self.crawl_state.add_visited(url, status_code=response.get('status_code'), content_type='text/plain')

                return {
                    'success': True,
                    'url': url,
                    'final_url': response.get('url'),
                    'job_type': 'sitemap',
                    'depth': job.depth,
                    'extracted_urls': len(urls),
                    'sitemap_type': 'text'
                }

        except Exception as e:
            error_message = f"خطا در پردازش sitemap {url}: {str(e)}"
            logger.error(error_message)

            self.crawl_state.add_failed(url, error=str(e))

            return {
                'success': False,
                'url': url,
                'error': error_message
            }
