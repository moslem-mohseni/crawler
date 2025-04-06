"""
ماژول مدیریت درخواست‌های HTTP برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس‌ها و توابع پیشرفته برای مدیریت درخواست‌های HTTP است.
"""

import os
import time
import random
import urllib.robotparser
from urllib.parse import urlparse, urljoin
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)

# لیست User-Agent های متداول
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
]


class RobotsTxtParser:
    """کلاس پردازش فایل robots.txt"""

    _instances = {}  # ذخیره‌سازی نمونه‌ها برای هر دامنه

    def __new__(cls, base_url):
        """پیاده‌سازی الگوی Singleton برای هر دامنه"""
        parsed_url = urlparse(base_url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

        if domain not in cls._instances:
            cls._instances[domain] = super(RobotsTxtParser, cls).__new__(cls)
            cls._instances[domain].domain = domain
            cls._instances[domain].initialized = False

        return cls._instances[domain]

    def __init__(self, base_url):
        """مقداردهی اولیه با بررسی فایل robots.txt"""
        if self.initialized:
            return

        try:
            self.parser = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(self.domain, '/robots.txt')
            self.parser.set_url(robots_url)
            self.parser.read()
            logger.info(f"فایل robots.txt برای دامنه {self.domain} با موفقیت بارگذاری شد")
            self.initialized = True
        except Exception as e:
            logger.error(f"خطا در بارگذاری robots.txt برای {self.domain}: {str(e)}")
            # در صورت خطا، فرض می‌کنیم همه مسیرها مجاز هستند
            self.parser = None
            self.initialized = True

    def can_fetch(self, url, user_agent="*"):
        """
        بررسی اجازه دسترسی به یک URL

        Args:
            url: آدرس برای بررسی
            user_agent: User-Agent برای بررسی (پیش‌فرض: *)

        Returns:
            bool: آیا دسترسی مجاز است؟
        """
        if not self.parser:
            return True  # در صورت عدم دسترسی به robots.txt، همه چیز مجاز است

        return self.parser.can_fetch(user_agent, url)

    def crawl_delay(self, user_agent="*"):
        """
        دریافت تأخیر مجاز برای خزش

        Args:
            user_agent: User-Agent برای بررسی (پیش‌فرض: *)

        Returns:
            float: تأخیر توصیه شده برای خزش (ثانیه)
        """
        if not self.parser:
            return None

        delay = self.parser.crawl_delay(user_agent)
        if delay:
            return float(delay)

        # بررسی request rate
        rrate = self.parser.request_rate(user_agent)
        if rrate:
            return float(rrate.seconds) / float(rrate.requests)

        return None


class RequestManager:
    """کلاس مدیریت درخواست‌های HTTP"""

    def __init__(self, base_url=None, default_delay=1, respect_robots=True, use_selenium=False):
        """
        مقداردهی اولیه مدیریت درخواست‌ها

        Args:
            base_url: آدرس پایه وبسایت (اختیاری)
            default_delay: تأخیر پیش‌فرض بین درخواست‌ها (ثانیه)
            respect_robots: آیا محدودیت‌های robots.txt رعایت شود؟
            use_selenium: آیا از سلنیوم برای بارگذاری صفحات استفاده شود؟
        """
        self.base_url = base_url
        self.default_delay = float(os.getenv('CRAWL_DELAY', default_delay))
        self.respect_robots = respect_robots
        self.use_selenium = use_selenium
        self.last_request_time = 0

        # ایجاد نشست HTTP
        self.session = self._create_session()

        # پردازنده robots.txt در صورت نیاز
        self.robots_parser = None
        if self.respect_robots and self.base_url:
            self.robots_parser = RobotsTxtParser(self.base_url)

        # راه‌اندازی سلنیوم در صورت نیاز
        self.driver = None
        if self.use_selenium:
            self._setup_selenium()

    def _create_session(self):
        """
        ایجاد یک نشست HTTP با قابلیت تلاش مجدد

        Returns:
            requests.Session: نشست HTTP
        """
        session = requests.Session()

        # تنظیم استراتژی تلاش مجدد
        retry_strategy = Retry(
            total=5,  # تعداد کل تلاش‌های مجدد
            backoff_factor=1,  # ضریب تأخیر برای تلاش مجدد
            status_forcelist=[429, 500, 502, 503, 504],  # کدهای وضعیت برای تلاش مجدد
            allowed_methods=["HEAD", "GET", "POST"]  # متدهای مجاز برای تلاش مجدد
        )

        # تنظیم آداپتور با استراتژی تلاش مجدد
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # تنظیم User-Agent
        session.headers.update({"User-Agent": self._get_random_user_agent()})

        return session

    def _setup_selenium(self):
        """راه‌اندازی مرورگر سلنیوم"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument(f"user-agent={self._get_random_user_agent()}")

            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(30)  # تنظیم محدودیت زمانی بارگذاری صفحه
            logger.info("مرورگر سلنیوم با موفقیت راه‌اندازی شد")
        except Exception as e:
            logger.error(f"خطا در راه‌اندازی سلنیوم: {str(e)}")
            self.use_selenium = False

    def _get_random_user_agent(self):
        """
        انتخاب تصادفی یک User-Agent

        Returns:
            str: User-Agent انتخاب شده
        """
        return random.choice(USER_AGENTS)

    def _respect_crawl_delay(self, url=None, user_agent=None):
        """
        رعایت تأخیر مناسب بین درخواست‌ها

        Args:
            url: آدرس درخواست (برای بررسی robots.txt)
            user_agent: User-Agent استفاده شده
        """
        now = time.time()

        # محاسبه تأخیر مناسب
        delay = self.default_delay

        if self.robots_parser and url:
            robots_delay = self.robots_parser.crawl_delay(user_agent)
            if robots_delay:
                delay = max(delay, robots_delay)

        # محاسبه زمان انتظار
        elapsed = now - self.last_request_time
        wait_time = max(0, delay - elapsed)

        if wait_time > 0:
            logger.debug(f"انتظار {wait_time:.2f} ثانیه برای رعایت محدودیت خزش")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _check_robots_permission(self, url, user_agent):
        """
        بررسی مجوز دسترسی در robots.txt

        Args:
            url: آدرس درخواست
            user_agent: User-Agent استفاده شده

        Returns:
            bool: آیا دسترسی مجاز است؟
        """
        if not self.respect_robots or not self.robots_parser:
            return True

        return self.robots_parser.can_fetch(url, user_agent)

    def get(self, url, use_selenium=None, params=None, timeout=30, verify_ssl=True):
        """
        ارسال درخواست GET

        Args:
            url: آدرس درخواست
            use_selenium: آیا از سلنیوم استفاده شود؟ (جایگزین تنظیم پیش‌فرض)
            params: پارامترهای درخواست
            timeout: محدودیت زمانی (ثانیه)
            verify_ssl: آیا گواهی SSL بررسی شود؟

        Returns:
            dict: شیء پاسخ شامل 'html', 'url', 'status_code' و غیره
        """
        # تنظیم User-Agent
        user_agent = self._get_random_user_agent()
        self.session.headers.update({"User-Agent": user_agent})

        # بررسی مجوز robots.txt
        if not self._check_robots_permission(url, user_agent):
            logger.warning(f"دسترسی به {url} توسط robots.txt منع شده است")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': 'Disallowed by robots.txt',
                'headers': None,
                'soup': None
            }

        # رعایت تأخیر بین درخواست‌ها
        self._respect_crawl_delay(url, user_agent)

        # تعیین روش درخواست
        should_use_selenium = use_selenium if use_selenium is not None else self.use_selenium

        try:
            if should_use_selenium and self.driver:
                return self._get_with_selenium(url)
            else:
                return self._get_with_requests(url, params, timeout, verify_ssl)
        except Exception as e:
            logger.error(f"خطا در ارسال درخواست به {url}: {str(e)}")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': str(e),
                'headers': None,
                'soup': None
            }

    def _get_with_requests(self, url, params=None, timeout=30, verify_ssl=True):
        """
        ارسال درخواست GET با کتابخانه requests

        Args:
            url: آدرس درخواست
            params: پارامترهای درخواست
            timeout: محدودیت زمانی (ثانیه)
            verify_ssl: آیا گواهی SSL بررسی شود؟

        Returns:
            dict: شیء پاسخ
        """
        logger.info(f"ارسال درخواست GET به {url}")

        start_time = time.time()
        response = self.session.get(url, params=params, timeout=timeout, verify=verify_ssl)
        response_time = time.time() - start_time

        logger.info(f"دریافت پاسخ از {url} با کد وضعیت {response.status_code} در {response_time:.2f} ثانیه")

        # تنظیم کدگذاری در صورت نیاز
        if response.encoding == 'ISO-8859-1':
            response.encoding = response.apparent_encoding

        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        return {
            'html': response.text,
            'url': response.url,
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'cookies': dict(response.cookies),
            'response_time': response_time,
            'soup': soup
        }

    def _get_with_selenium(self, url):
        """
        بارگذاری صفحه با سلنیوم

        Args:
            url: آدرس صفحه

        Returns:
            dict: شیء پاسخ
        """
        logger.info(f"بارگذاری {url} با سلنیوم")

        if not self.driver:
            logger.error("سلنیوم راه‌اندازی نشده است")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': 'Selenium not initialized',
                'headers': None,
                'soup': None
            }

        start_time = time.time()

        try:
            self.driver.get(url)

            # انتظار برای بارگذاری کامل صفحه
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # اجازه دادن به JavaScript برای اجرا
            time.sleep(2)

            response_time = time.time() - start_time
            logger.info(f"بارگذاری {url} با سلنیوم در {response_time:.2f} ثانیه")

            html = self.driver.page_source
            current_url = self.driver.current_url

            # ایجاد شیء BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            return {
                'html': html,
                'url': current_url,
                'status_code': 200,  # فرض می‌کنیم موفق بوده
                'headers': {},  # سلنیوم هدرهای پاسخ را ارائه نمی‌دهد
                'cookies': self.driver.get_cookies(),
                'response_time': response_time,
                'soup': soup
            }

        except TimeoutException:
            logger.error(f"زمان بارگذاری {url} با سلنیوم به پایان رسید")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': 'Timeout',
                'headers': None,
                'soup': None
            }
        except Exception as e:
            logger.error(f"خطا در بارگذاری {url} با سلنیوم: {str(e)}")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': str(e),
                'headers': None,
                'soup': None
            }

    def post(self, url, data=None, json=None, timeout=30, verify_ssl=True):
        """
        ارسال درخواست POST

        Args:
            url: آدرس درخواست
            data: داده‌های فرم
            json: داده‌های JSON
            timeout: محدودیت زمانی (ثانیه)
            verify_ssl: آیا گواهی SSL بررسی شود؟

        Returns:
            dict: شیء پاسخ
        """
        # تنظیم User-Agent
        user_agent = self._get_random_user_agent()
        self.session.headers.update({"User-Agent": user_agent})

        # رعایت تأخیر بین درخواست‌ها
        self._respect_crawl_delay(url, user_agent)

        try:
            logger.info(f"ارسال درخواست POST به {url}")

            start_time = time.time()
            response = self.session.post(url, data=data, json=json, timeout=timeout, verify=verify_ssl)
            response_time = time.time() - start_time

            logger.info(f"دریافت پاسخ از {url} با کد وضعیت {response.status_code} در {response_time:.2f} ثانیه")

            # تنظیم کدگذاری در صورت نیاز
            if response.encoding == 'ISO-8859-1':
                response.encoding = response.apparent_encoding

            # ایجاد شیء BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            return {
                'html': response.text,
                'url': response.url,
                'status_code': response.status_code,
                'headers': dict(response.headers),
                'cookies': dict(response.cookies),
                'response_time': response_time,
                'soup': soup
            }

        except Exception as e:
            logger.error(f"خطا در ارسال درخواست POST به {url}: {str(e)}")
            return {
                'html': None,
                'url': url,
                'status_code': None,
                'error': str(e),
                'headers': None,
                'soup': None
            }

    def close(self):
        """بستن نشست و آزادسازی منابع"""
        self.session.close()

        if self.driver:
            try:
                self.driver.quit()
                logger.info("مرورگر سلنیوم با موفقیت بسته شد")
            except Exception as e:
                logger.error(f"خطا در بستن مرورگر سلنیوم: {str(e)}")

    def __del__(self):
        """فراخوانی خودکار close() هنگام حذف شیء"""
        self.close()


# توابع کمکی برای سادگی استفاده

def make_request(url, method='get', **kwargs):
    """
    ارسال یک درخواست HTTP ساده

    Args:
        url: آدرس درخواست
        method: متد درخواست ('get' یا 'post')
        **kwargs: پارامترهای اضافی برای ارسال به RequestManager

    Returns:
        dict: شیء پاسخ
    """
    manager = RequestManager()

    try:
        if method.lower() == 'get':
            return manager.get(url, **kwargs)
        elif method.lower() == 'post':
            return manager.post(url, **kwargs)
        else:
            logger.error(f"متد نامعتبر: {method}")
            return None
    finally:
        manager.close()


def normalize_url(url, base_url=None):
    """
    نرمال‌سازی یک URL

    Args:
        url: آدرس برای نرمال‌سازی
        base_url: آدرس پایه برای URL‌های نسبی

    Returns:
        str: URL نرمال‌سازی شده
    """
    # تبدیل URL نسبی به مطلق
    if base_url and not url.startswith(('http://', 'https://')):
        url = urljoin(base_url, url)

    # پردازش URL
    parsed = urlparse(url)

    # بازسازی URL با حذف پارامترهای غیرضروری
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"