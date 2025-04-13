"""
ماژول شناسایی ساختار وبسایت برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس‌ها و الگوریتم‌های شناسایی خودکار ساختار وبسایت و الگوهای URL است.
"""

from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)

import re
import os
import json
import time
from urllib.parse import urlparse, parse_qs, urljoin
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from bs4 import BeautifulSoup

from utils.http import RequestManager
from database.operations import BaseDBOperations
from models.domain import Domain


class URLPattern:
    """کلاس الگوی URL برای شناسایی و دسته‌بندی آدرس‌ها"""

    def __init__(self, pattern, is_list=False, is_detail=False, sample_urls=None, weight=1.0):
        """
        مقداردهی اولیه الگوی URL

        Args:
            pattern: الگوی URL (عبارت منظم یا الگوی رشته)
            is_list: آیا این الگو مربوط به صفحات لیستی است؟
            is_detail: آیا این الگو مربوط به صفحات جزئیات است؟
            sample_urls: نمونه‌های URL برای این الگو
            weight: وزن اهمیت این الگو
        """
        self.pattern = pattern
        self.is_list = is_list
        self.is_detail = is_detail
        self.sample_urls = sample_urls or []
        self.weight = weight
        self.url_count = 0

        # تبدیل الگو به عبارت منظم در صورت نیاز
        self.regex = self._pattern_to_regex(pattern)

    def _pattern_to_regex(self, pattern):
        """
        تبدیل الگوی ساده به عبارت منظم

        Args:
            pattern: الگوی URL

        Returns:
            re.Pattern: عبارت منظم کامپایل شده
        """
        # اگر الگو از قبل عبارت منظم است
        if pattern.startswith('^') and (pattern.endswith('$') or pattern.endswith('.*$')):
            return re.compile(pattern)

        # تبدیل یک الگوی ساده به عبارت منظم
        # جایگزینی متغیرها با .*?
        regex_pattern = pattern.replace('*', '.*?')

        # افزودن ابتدا و انتهای رشته
        if not regex_pattern.startswith('^'):
            regex_pattern = '^' + regex_pattern

        if not regex_pattern.endswith('$') and not regex_pattern.endswith('.*?$'):
            regex_pattern = regex_pattern + '$'

        return re.compile(regex_pattern)

    def matches(self, url):
        """
        بررسی تطابق یک URL با این الگو

        Args:
            url: آدرس URL برای بررسی

        Returns:
            bool: آیا URL با این الگو مطابقت دارد؟
        """
        return bool(self.regex.search(url))

    def add_sample_url(self, url):
        """
        افزودن یک نمونه URL به این الگو

        Args:
            url: آدرس URL برای افزودن

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if url not in self.sample_urls:
            self.sample_urls.append(url)
        self.url_count += 1
        return self

    def to_dict(self):
        """
        تبدیل الگو به دیکشنری برای ذخیره‌سازی

        Returns:
            dict: دیکشنری حاوی اطلاعات الگو
        """
        return {
            'pattern': self.pattern,
            'is_list': self.is_list,
            'is_detail': self.is_detail,
            'sample_urls': self.sample_urls[:5],  # حداکثر 5 نمونه ذخیره می‌شود
            'weight': self.weight,
            'url_count': self.url_count
        }

    @classmethod
    def from_dict(cls, data):
        """
        ایجاد الگو از دیکشنری ذخیره شده

        Args:
            data: دیکشنری حاوی اطلاعات الگو

        Returns:
            URLPattern: نمونه جدید ایجاد شده
        """
        pattern = cls(
            pattern=data['pattern'],
            is_list=data.get('is_list', False),
            is_detail=data.get('is_detail', False),
            sample_urls=data.get('sample_urls', []),
            weight=data.get('weight', 1.0)
        )
        pattern.url_count = data.get('url_count', 0)
        return pattern

    def __str__(self):
        """
        نمایش رشته‌ای الگو

        Returns:
            str: رشته نمایشی
        """
        return f"URLPattern('{self.pattern}', list={self.is_list}, detail={self.is_detail}, count={self.url_count})"

    def __repr__(self):
        """
        نمایش رشته‌ای الگو

        Returns:
            str: رشته نمایشی
        """
        return self.__str__()


class HTMLPatternFinder:
    """کلاس شناسایی الگوهای HTML"""

    def __init__(self):
        """مقداردهی اولیه تحلیلگر الگوهای HTML"""
        self.content_xpaths = {}
        self.list_selectors = {}
        self.detail_selectors = {}
        self.current_patterns = {}

    def analyze_html_structure(self, html_content, url):
        """
        تحلیل ساختار HTML یک صفحه

        Args:
            html_content: محتوای HTML صفحه
            url: آدرس URL صفحه

        Returns:
            dict: اطلاعات ساختاری استخراج شده
        """
        try:
            # ایجاد شیء BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            url_type = self._detect_page_type(url, soup)

            # استخراج اطلاعات ساختاری بر اساس نوع صفحه
            if url_type == 'list':
                selectors = self._analyze_list_page(soup)
                self.list_selectors[url] = selectors
            elif url_type == 'detail':
                selectors = self._analyze_detail_page(soup)
                self.detail_selectors[url] = selectors
            else:
                selectors = self._analyze_generic_page(soup)

            return {
                'url': url,
                'type': url_type,
                'selectors': selectors
            }

        except Exception as e:
            logger.error(f"خطا در تحلیل ساختار HTML برای {url}: {str(e)}")
            return {
                'url': url,
                'type': 'unknown',
                'selectors': {},
                'error': str(e)
            }

    def _detect_page_type(self, url, soup):
        """
        تشخیص نوع صفحه (لیست، جزئیات، عمومی)

        Args:
            url: آدرس URL صفحه
            soup: شیء BeautifulSoup صفحه

        Returns:
            str: نوع صفحه ('list', 'detail', 'generic')
        """
        # بررسی URL
        list_patterns = [
            r'/category/', r'/tag/', r'/archive/', r'/blog/', r'/articles/',
            r'/questions/', r'/list/', r'/search/', r'/page/\d+', r'/\?page=\d+'
        ]

        for pattern in list_patterns:
            if re.search(pattern, url):
                return 'list'

        # بررسی ساختار HTML

        # صفحات لیستی معمولاً دارای عناصر تکراری هستند
        list_candidates = [
            soup.find_all('div', class_=re.compile(r'(post|article|item|card)s?')),
            soup.find_all('li', class_=re.compile(r'(post|article|item|card)s?')),
            soup.find_all('article')
        ]

        for candidates in list_candidates:
            if len(candidates) >= 3:  # حداقل 3 آیتم تکراری
                return 'list'

        # صفحات جزئیات معمولاً شامل عناصر خاصی هستند
        detail_indicators = [
            soup.find('article', class_=re.compile(r'(post|article|content)')),
            soup.find('div', class_=re.compile(r'(post|article|content)-detail')),
            soup.find('div', id=re.compile(r'(post|article|content)-detail')),
            soup.find('div', class_=re.compile(r'single')),
            soup.find('section', class_=re.compile(r'(post|article|content)'))
        ]

        if any(indicator for indicator in detail_indicators):
            return 'detail'

        # بررسی اضافی برای صفحات جزئیات
        h1_tags = soup.find_all('h1')
        if h1_tags and len(h1_tags) == 1:
            content_tags = soup.find_all(['p', 'div'], class_=re.compile(r'(content|text|body)'))
            if content_tags and any(len(tag.get_text()) > 500 for tag in content_tags):
                return 'detail'

        # در صورت عدم شناسایی مشخص، عمومی فرض می‌کنیم
        return 'generic'

    def _analyze_list_page(self, soup):
        """
        تحلیل ساختار صفحه لیستی

        Args:
            soup: شیء BeautifulSoup صفحه

        Returns:
            dict: اطلاعات ساختاری صفحه لیستی
        """
        selectors = {}

        # یافتن کانتینر اصلی لیست
        container_candidates = [
            ('div', re.compile(r'(posts|articles|items|cards|list)s?-container')),
            ('div', re.compile(r'(posts|articles|items|cards|list)s?')),
            ('ul', re.compile(r'(posts|articles|items|cards|list)s?')),
            ('section', re.compile(r'(posts|articles|items|cards|list)s?')),
        ]

        container = None
        for tag, class_pattern in container_candidates:
            container = soup.find(tag, class_=class_pattern)
            if container:
                selectors['container'] = f"{tag}.{container.get('class', [''])[0]}"
                break

        # اگر کانتینر مشخصی یافت نشد، با روش‌های دیگر پیدا می‌کنیم
        if not container:
            # بررسی تگ‌های article
            articles = soup.find_all('article')
            if len(articles) >= 2:
                container = articles[0].parent
                selectors['container'] = f"{container.name}"
                if container.get('class'):
                    selectors['container'] += f".{container.get('class', [''])[0]}"
                selectors['item'] = 'article'

        # بررسی آیتم‌های لیست
        if container:
            item_candidates = [
                ('div', re.compile(r'(post|article|item|card)s?-item')),
                ('div', re.compile(r'(post|article|item|card)s?')),
                ('article', re.compile(r'.*')),
                ('li', re.compile(r'(post|article|item|card)s?')),
            ]

            for tag, class_pattern in item_candidates:
                items = container.find_all(tag, class_=class_pattern)
                if items and len(items) > 1:
                    selectors['item'] = f"{tag}"
                    if items[0].get('class'):
                        selectors['item'] += f".{items[0].get('class', [''])[0]}"
                    break

        # یافتن عناصر داخل هر آیتم
        if container and 'item' in selectors:
            items = container.select(selectors['item'])
            if items:
                # بررسی عنوان
                title_candidates = [
                    ('h2', None), ('h3', None), ('h4', None),
                    ('div', re.compile(r'title')), ('span', re.compile(r'title')),
                    ('a', None)
                ]

                for tag, class_pattern in title_candidates:
                    title_elements = [item.find(tag, class_=class_pattern) for item in items[:3] if item]
                    if title_elements and all(title_elements):
                        selectors['title'] = f"{tag}"
                        if title_elements[0] and title_elements[0].get('class') and class_pattern:
                            selectors['title'] += f".{title_elements[0].get('class', [''])[0]}"
                        break

                # بررسی لینک
                link_elements = [item.find('a') for item in items[:3] if item]
                if link_elements and all(link_elements):
                    selectors['link'] = 'a'

                # بررسی خلاصه
                summary_candidates = [
                    ('p', None), ('div', re.compile(r'(summary|excerpt)')),
                    ('span', re.compile(r'(summary|excerpt)')), ('div', re.compile(r'content'))
                ]

                for tag, class_pattern in summary_candidates:
                    summary_elements = [item.find(tag, class_=class_pattern) for item in items[:3] if item]
                    if summary_elements and all(summary_elements):
                        selectors['summary'] = f"{tag}"
                        if summary_elements[0] and summary_elements[0].get('class') and class_pattern:
                            selectors['summary'] += f".{summary_elements[0].get('class', [''])[0]}"
                        break

        # یافتن اطلاعات صفحه‌بندی
        pagination_candidates = [
            ('div', re.compile(r'pagination')),
            ('ul', re.compile(r'pagination')),
            ('nav', re.compile(r'pagination')),
            ('div', re.compile(r'paging')),
        ]

        for tag, class_pattern in pagination_candidates:
            pagination = soup.find(tag, class_=class_pattern)
            if pagination:
                selectors['pagination'] = f"{tag}.{pagination.get('class', [''])[0]}"
                pagination_links = pagination.find_all('a')
                if pagination_links:
                    selectors['pagination_links'] = 'a'
                break

        return selectors

    def _analyze_detail_page(self, soup):
        """
        تحلیل ساختار صفحه جزئیات

        Args:
            soup: شیء BeautifulSoup صفحه

        Returns:
            dict: اطلاعات ساختاری صفحه جزئیات
        """
        selectors = {}

        # یافتن کانتینر اصلی محتوا
        container_candidates = [
            ('article', None),
            ('div', re.compile(r'(post|article|content)-content')),
            ('div', re.compile(r'(post|article|content)-detail')),
            ('div', re.compile(r'(post|article|content)')),
            ('section', re.compile(r'(post|article|content)')),
        ]

        container = None
        for tag, class_pattern in container_candidates:
            container = soup.find(tag, class_=class_pattern)
            if container:
                selectors['container'] = f"{tag}"
                if container.get('class') and class_pattern:
                    selectors['container'] += f".{container.get('class', [''])[0]}"
                break

        # اگر کانتینر مشخصی یافت نشد، روش‌های دیگر را امتحان می‌کنیم
        if not container:
            # بررسی محتوای اصلی
            content_candidates = [
                ('div', 'content'),
                ('div', 'entry-content'),
                ('div', 'post-content'),
                ('div', 'article-content')
            ]

            for tag, class_name in content_candidates:
                container = soup.find(tag, class_=class_name)
                if container:
                    selectors['container'] = f"{tag}.{class_name}"
                    break

        # یافتن عناصر داخل کانتینر
        if container:
            # بررسی عنوان
            title_candidates = [
                ('h1', None), ('h2', None),
                ('div', re.compile(r'title')), ('span', re.compile(r'title'))
            ]

            for tag, class_pattern in title_candidates:
                title_element = soup.find(tag, class_=class_pattern)
                if title_element:
                    selectors['title'] = f"{tag}"
                    if title_element.get('class') and class_pattern:
                        selectors['title'] += f".{title_element.get('class', [''])[0]}"
                    break

            # بررسی محتوای اصلی
            content_candidates = [
                ('div', re.compile(r'content')), ('div', re.compile(r'body')),
                ('div', re.compile(r'text'))
            ]

            for tag, class_pattern in content_candidates:
                content_element = container.find(tag, class_=class_pattern)
                if content_element:
                    selectors['content'] = f"{tag}"
                    if content_element.get('class') and class_pattern:
                        selectors['content'] += f".{content_element.get('class', [''])[0]}"
                    break

            # اگر محتوای مشخصی یافت نشد، فرض می‌کنیم خود کانتینر، محتوا است
            if 'content' not in selectors:
                selectors['content'] = selectors['container']

            # بررسی تاریخ
            date_candidates = [
                ('time', None),
                ('span', re.compile(r'date')), ('div', re.compile(r'date')),
                ('span', re.compile(r'time')), ('div', re.compile(r'time')),
                ('span', re.compile(r'published')), ('div', re.compile(r'published'))
            ]

            for tag, class_pattern in date_candidates:
                date_element = container.find(tag, class_=class_pattern)
                if date_element:
                    selectors['date'] = f"{tag}"
                    if date_element.get('class') and class_pattern:
                        selectors['date'] += f".{date_element.get('class', [''])[0]}"
                    break

            # بررسی نویسنده
            author_candidates = [
                ('span', re.compile(r'author')), ('div', re.compile(r'author')),
                ('a', re.compile(r'author')), ('div', re.compile(r'writer')),
                ('span', re.compile(r'writer'))
            ]

            for tag, class_pattern in author_candidates:
                author_element = container.find(tag, class_=class_pattern)
                if author_element:
                    selectors['author'] = f"{tag}"
                    if author_element.get('class') and class_pattern:
                        selectors['author'] += f".{author_element.get('class', [''])[0]}"
                    break

        return selectors

    def _analyze_generic_page(self, soup):
        """
        تحلیل ساختار صفحه عمومی

        Args:
            soup: شیء BeautifulSoup صفحه

        Returns:
            dict: اطلاعات ساختاری صفحه عمومی
        """
        selectors = {}

        # بررسی عنوان صفحه
        title_element = soup.find('h1')
        if title_element:
            selectors['title'] = 'h1'
            if title_element.get('class'):
                selectors['title'] += f".{title_element.get('class', [''])[0]}"

        # بررسی ناوبری
        nav_candidates = [
            ('nav', None),
            ('div', re.compile(r'nav')),
            ('div', re.compile(r'menu')),
            ('ul', re.compile(r'menu'))
        ]

        for tag, class_pattern in nav_candidates:
            nav_element = soup.find(tag, class_=class_pattern)
            if nav_element:
                selectors['navigation'] = f"{tag}"
                if nav_element.get('class') and class_pattern:
                    selectors['navigation'] += f".{nav_element.get('class', [''])[0]}"
                break

        # بررسی لینک‌های مهم
        important_links = soup.find_all('a', href=True)
        if important_links:
            link_patterns = defaultdict(int)
            for link in important_links:
                if 'href' in link.attrs:
                    href = link['href']
                    if href.startswith(('http', '/')):
                        path = urlparse(href).path.split('/')
                        if len(path) > 1:
                            link_patterns[path[1]] += 1

            top_sections = [section for section, count in
                           sorted(link_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
                           if section and section not in ('wp-content', 'wp-includes', 'js', 'css', 'img', 'images')]

            if top_sections:
                selectors['main_sections'] = top_sections

        return selectors

    def get_xpaths_from_selectors(self, selectors, page_type):
        """
        تبدیل سلکتورهای CSS به XPath

        Args:
            selectors: دیکشنری سلکتورهای استخراج شده
            page_type: نوع صفحه ('list', 'detail', 'generic')

        Returns:
            dict: دیکشنری مسیرهای XPath
        """
        xpaths = {}

        if page_type == 'list':
            if 'container' in selectors:
                container = selectors['container']
                xpaths['container'] = self._css_to_xpath(container)

                if 'item' in selectors:
                    item = selectors['item']
                    xpaths['item'] = f"{xpaths['container']}//{self._css_to_xpath(item, strip_xpath=True)}"

                    if 'title' in selectors:
                        title = selectors['title']
                        xpaths['title'] = f"{xpaths['item']}//{self._css_to_xpath(title, strip_xpath=True)}"

                    if 'link' in selectors:
                        link = selectors['link']
                        xpaths['link'] = f"{xpaths['item']}//{self._css_to_xpath(link, strip_xpath=True)}"

                    if 'summary' in selectors:
                        summary = selectors['summary']
                        xpaths['summary'] = f"{xpaths['item']}//{self._css_to_xpath(summary, strip_xpath=True)}"

            if 'pagination' in selectors:
                pagination = selectors['pagination']
                xpaths['pagination'] = self._css_to_xpath(pagination)

                if 'pagination_links' in selectors:
                    pagination_links = selectors['pagination_links']
                    xpaths['pagination_links'] = f"{xpaths['pagination']}//{self._css_to_xpath(pagination_links, strip_xpath=True)}"

        elif page_type == 'detail':
            if 'container' in selectors:
                container = selectors['container']
                xpaths['container'] = self._css_to_xpath(container)

                if 'title' in selectors:
                    title = selectors['title']
                    xpaths['title'] = self._css_to_xpath(title)

                if 'content' in selectors:
                    content = selectors['content']
                    xpaths['content'] = f"{xpaths['container']}//{self._css_to_xpath(content, strip_xpath=True)}"

                if 'date' in selectors:
                    date = selectors['date']
                    xpaths['date'] = f"{xpaths['container']}//{self._css_to_xpath(date, strip_xpath=True)}"

                if 'author' in selectors:
                    author = selectors['author']
                    xpaths['author'] = f"{xpaths['container']}//{self._css_to_xpath(author, strip_xpath=True)}"

        return xpaths

    def _css_to_xpath(self, css_selector, strip_xpath=False):
        """
        تبدیل سلکتور CSS به XPath

        Args:
            css_selector: سلکتور CSS
            strip_xpath: آیا بخش اول مسیر XPath حذف شود؟

        Returns:
            str: مسیر XPath
        """
        if not css_selector:
            return ""

        # تجزیه سلکتور CSS
        parts = css_selector.split('.')
        tag = parts[0]

        if len(parts) > 1:
            cls = parts[1]
            xpath = f"//{tag}[contains(@class, '{cls}')]"
        else:
            xpath = f"//{tag}"

        if strip_xpath:
            xpath = xpath[2:]  # حذف // از ابتدای مسیر

        return xpath

    def save_patterns(self, file_path):
        """
        ذخیره الگوهای شناسایی شده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        try:
            data = {
                'list_selectors': self.list_selectors,
                'detail_selectors': self.detail_selectors,
                'content_xpaths': self.content_xpaths
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"الگوهای HTML با موفقیت در {file_path} ذخیره شدند")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره‌سازی الگوهای HTML: {str(e)}")
            return False

    def load_patterns(self, file_path):
        """
        بارگذاری الگوهای شناسایی شده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            bool: آیا بارگذاری موفق بود؟
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"فایل الگوهای HTML یافت نشد: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.list_selectors = data.get('list_selectors', {})
            self.detail_selectors = data.get('detail_selectors', {})
            self.content_xpaths = data.get('content_xpaths', {})

            logger.info(f"الگوهای HTML با موفقیت از {file_path} بارگذاری شدند")
            return True
        except Exception as e:
            logger.error(f"خطا در بارگذاری الگوهای HTML: {str(e)}")
            return False


class URLStructureDiscovery:
    """کلاس شناسایی ساختار URL‌ها"""

    def __init__(self, base_url=None):
        """
        مقداردهی اولیه کشف ساختار URL

        Args:
            base_url: آدرس پایه وبسایت (اختیاری)
        """
        self.base_url = base_url
        self.urls = []
        self.patterns = []
        self.url_parts = defaultdict(set)
        self.common_parameters = {}
        self.important_sections = []

    def add_url(self, url):
        """
        افزودن یک URL به مجموعه داده‌ها

        Args:
            url: آدرس URL برای افزودن

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        # نرمال‌سازی URL
        if self.base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(self.base_url, url)

        if url not in self.urls:
            self.urls.append(url)
            self._update_url_parts(url)

        return self

    def add_urls(self, urls):
        """
        افزودن چندین URL به مجموعه داده‌ها

        Args:
            urls: لیست آدرس‌های URL برای افزودن

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        for url in urls:
            self.add_url(url)
        return self

    def _update_url_parts(self, url):
        """
        به‌روزرسانی اطلاعات آماری بخش‌های URL

        Args:
            url: آدرس URL برای تحلیل
        """
        try:
            parsed = urlparse(url)

            # تجزیه مسیر
            path_parts = parsed.path.strip('/').split('/')
            for i, part in enumerate(path_parts):
                if part:
                    self.url_parts[f"path_{i}"].add(part)

            # تجزیه پارامترها
            params = parse_qs(parsed.query)
            for key, values in params.items():
                if key not in self.common_parameters:
                    self.common_parameters[key] = []

                for value in values:
                    if value not in self.common_parameters[key]:
                        self.common_parameters[key].append(value)

        except Exception as e:
            logger.error(f"خطا در تحلیل بخش‌های URL {url}: {str(e)}")

    def discover_patterns(self, clustering=True, min_samples=2, eps=0.3):
        """
        کشف الگوهای URL با استفاده از روش‌های خوشه‌بندی

        Args:
            clustering: آیا از خوشه‌بندی استفاده شود؟
            min_samples: حداقل تعداد نمونه در هر خوشه
            eps: فاصله حداکثر برای الگوریتم DBSCAN

        Returns:
            list: لیست الگوهای کشف شده
        """
        if not self.urls:
            logger.warning("هیچ URL‌ی برای کشف الگو یافت نشد")
            return []

        # شناسایی بخش‌های مهم URL
        self._identify_important_sections()

        if clustering:
            patterns = self._discover_patterns_with_clustering(min_samples, eps)
        else:
            patterns = self._discover_patterns_with_heuristics()

        # ذخیره الگوهای کشف شده
        self.patterns = patterns

        return patterns

    def _identify_important_sections(self):
        """شناسایی بخش‌های مهم و تکرارشونده در URL‌ها"""
        # بخش‌های مسیر URL
        path_sections = []
        for key, values in self.url_parts.items():
            if key.startswith('path_'):
                # اگر تعداد مقادیر یکتا کم است (ثابت)
                if 1 < len(values) < 10:
                    path_sections.append({
                        'position': int(key.split('_')[1]),
                        'values': list(values),
                        'is_variable': False
                    })
                # اگر تعداد مقادیر یکتا زیاد است (متغیر)
                elif len(values) >= 10:
                    # بررسی الگوهای عددی یا شناسه‌ای
                    numeric_count = sum(1 for v in values if v.isdigit())
                    if numeric_count > len(values) * 0.7:
                        path_sections.append({
                            'position': int(key.split('_')[1]),
                            'values': ['<id>'],
                            'is_variable': True,
                            'type': 'numeric'
                        })
                    else:
                        # بررسی الگوهای رشته‌ای (slug)
                        slug_count = sum(1 for v in values if '-' in v)
                        if slug_count > len(values) * 0.7:
                            path_sections.append({
                                'position': int(key.split('_')[1]),
                                'values': ['<slug>'],
                                'is_variable': True,
                                'type': 'slug'
                            })
                        else:
                            path_sections.append({
                                'position': int(key.split('_')[1]),
                                'values': ['<variable>'],
                                'is_variable': True,
                                'type': 'unknown'
                            })

        # مرتب‌سازی بخش‌ها بر اساس موقعیت
        path_sections.sort(key=lambda x: x['position'])

        # ذخیره بخش‌های مهم
        self.important_sections = path_sections

    def _discover_patterns_with_clustering(self, min_samples=2, eps=0.3):
        """
        کشف الگوهای URL با استفاده از خوشه‌بندی

        Args:
            min_samples: حداقل تعداد نمونه در هر خوشه
            eps: فاصله حداکثر برای الگوریتم DBSCAN

        Returns:
            list: لیست الگوهای کشف شده
        """
        # ایجاد ویژگی‌های بردار برای هر URL
        url_features = self._extract_url_features()

        if url_features is None or url_features.size == 0 or len(url_features) < min_samples:
            logger.warning("تعداد URL‌ها برای خوشه‌بندی کافی نیست یا ویژگی‌ها استخراج نشدند")
            return self._discover_patterns_with_heuristics()

        try:
            # اجرای الگوریتم خوشه‌بندی DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(url_features)

            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            logger.info(f"خوشه‌بندی URL‌ها به {n_clusters} خوشه")

            # تبدیل خوشه‌ها به الگوها
            patterns = []

            # گروه‌بندی URL‌ها بر اساس برچسب خوشه
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  # نادیده گرفتن نقاط نویز
                    clusters[label].append(self.urls[i])

            # ایجاد الگو برای هر خوشه
            for label, cluster_urls in clusters.items():
                pattern = self._generate_pattern_from_cluster(cluster_urls)
                if pattern:
                    patterns.append(pattern)

            # افزودن الگوهای اضافی برای نقاط بدون خوشه
            unclustered = [self.urls[i] for i, label in enumerate(labels) if label == -1]
            if unclustered:
                extra_patterns = self._discover_patterns_with_heuristics(urls=unclustered)
                patterns.extend(extra_patterns)

            return patterns

        except Exception as e:
            logger.error(f"خطا در خوشه‌بندی URL‌ها: {str(e)}")
            return self._discover_patterns_with_heuristics()

    def _extract_url_features(self):
        """
        استخراج ویژگی‌های URL برای خوشه‌بندی

        Returns:
            numpy.ndarray: ماتریس ویژگی‌ها
        """
        try:
            # تعیین حداکثر تعداد بخش‌ها
            max_parts = max(len(urlparse(url).path.strip('/').split('/')) for url in self.urls)
            features = np.zeros((len(self.urls), max_parts + 1))  # +1 برای دامنه

            for i, url in enumerate(self.urls):
                parsed = urlparse(url)

                # شاخص برای دامنه
                features[i, 0] = hash(parsed.netloc) % 1000000

                # شاخص برای بخش‌های مسیر
                parts = parsed.path.strip('/').split('/')
                for j, part in enumerate(parts):
                    if part:
                        # اگر عدد است، یک مقدار خاص
                        if part.isdigit():
                            features[i, j + 1] = -1
                        # اگر slug است (حاوی خط تیره)
                        elif isinstance(part, str) and '-' in part:
                            features[i, j + 1] = -2
                        else:
                            features[i, j + 1] = hash(part) % 1000000

            return features
        except Exception as e:
            logger.error(f"خطا در استخراج ویژگی‌های URL: {str(e)}")
            return None

    def _generate_pattern_from_cluster(self, cluster_urls):
        """
        ایجاد یک الگوی URL از خوشه

        Args:
            cluster_urls: لیست URL‌های یک خوشه

        Returns:
            URLPattern: الگوی ایجاد شده یا None در صورت خطا
        """
        try:
            # استخراج الگوی مشترک
            common_pattern = self._find_common_pattern(cluster_urls)

            # تشخیص نوع صفحه
            is_list = self._is_list_pattern(common_pattern)
            is_detail = self._is_detail_pattern(common_pattern)

            # ایجاد الگو
            pattern = URLPattern(
                pattern=common_pattern,
                is_list=is_list,
                is_detail=is_detail,
                sample_urls=cluster_urls[:5],  # حداکثر 5 نمونه
                weight=len(cluster_urls) / len(self.urls)  # وزن نسبی
            )

            return pattern
        except Exception as e:
            logger.error(f"خطا در ایجاد الگو از خوشه: {str(e)}")
            return None

    def _find_common_pattern(self, urls):
        """
        یافتن الگوی مشترک بین URL‌ها

        Args:
            urls: لیست URL‌ها

        Returns:
            str: الگوی مشترک
        """
        if not urls:
            return ""

        # تجزیه URL‌ها
        parsed_urls = [urlparse(url) for url in urls]

        # استخراج الگوی مشترک در مسیر
        path_parts = [
            [part for part in url.path.strip('/').split('/') if part]
            for url in parsed_urls
        ]

        # یافتن طول حداکثر مسیر
        max_length = max(len(parts) for parts in path_parts) if path_parts else 0

        # ایجاد الگو برای هر بخش
        pattern_parts = []
        for i in range(max_length):
            # جمع‌آوری تمام مقادیر در این موقعیت
            values = [parts[i] if i < len(parts) else None for parts in path_parts]
            values = [v for v in values if v is not None]

            # اگر همه مقادیر یکسان هستند
            if len(values) > 0 and all(v is not None for v in values) and len(set(values)) == 1:
                pattern_parts.append(values[0])
            # اگر مقادیر متفاوت اما همه عددی هستند
            elif values and all(v.isdigit() for v in values if v):
                pattern_parts.append("*")
            # اگر مقادیر متفاوت و اکثراً slug هستند
            elif sum(1 for v in values if isinstance(v, str) and '-' in v) > len(values) * 0.7:
                pattern_parts.append("*")
            # اگر مقادیر متفاوت هستند
            else:
                pattern_parts.append("*")

        # ایجاد الگوی کامل
        domain = parsed_urls[0].netloc if parsed_urls else ""
        path = "/".join(pattern_parts)

        return f"{domain}/{path}"

    def _discover_patterns_with_heuristics(self, urls=None):
        """
        کشف الگوهای URL با استفاده از اکتشاف و قواعد

        Args:
            urls: لیست URL‌ها (اختیاری، پیش‌فرض: self.urls)

        Returns:
            list: لیست الگوهای کشف شده
        """
        if urls is None:
            urls = self.urls

        if not urls:
            return []

        patterns = []

        # گروه‌بندی URL‌ها بر اساس تعداد بخش‌های مسیر
        grouped_by_length = defaultdict(list)
        for url in urls:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.strip('/').split('/') if p]
            grouped_by_length[len(path_parts)].append(url)

        # پردازش هر گروه
        for length, group_urls in grouped_by_length.items():
            # اگر گروه کوچک است، هر URL یک الگوی جداگانه می‌شود
            if len(group_urls) < 3:
                for url in group_urls:
                    parsed = urlparse(url)
                    path = parsed.path

                    is_list = self._is_list_pattern(path)
                    is_detail = self._is_detail_pattern(path)

                    pattern = URLPattern(
                        pattern=url,
                        is_list=is_list,
                        is_detail=is_detail,
                        sample_urls=[url],
                        weight=1 / len(urls)
                    )
                    patterns.append(pattern)
            else:
                # گروه‌بندی بیشتر بر اساس بخش‌های ثابت
                subgroups = self._group_by_fixed_parts(group_urls)

                for subgroup in subgroups:
                    if subgroup:
                        common_pattern = self._find_common_pattern(subgroup)

                        is_list = self._is_list_pattern(common_pattern)
                        is_detail = self._is_detail_pattern(common_pattern)

                        pattern = URLPattern(
                            pattern=common_pattern,
                            is_list=is_list,
                            is_detail=is_detail,
                            sample_urls=subgroup[:5],
                            weight=len(subgroup) / len(urls)
                        )
                        patterns.append(pattern)

        return patterns

    def _group_by_fixed_parts(self, urls):
        """
        گروه‌بندی URL‌ها بر اساس بخش‌های ثابت

        Args:
            urls: لیست URL‌ها

        Returns:
            list: لیست گروه‌های URL
        """
        if not urls:
            return []

        # تجزیه URL‌ها
        parsed_urls = [(url, urlparse(url)) for url in urls]

        # استخراج بخش‌های مسیر
        url_parts = []
        for url, parsed in parsed_urls:
            parts = parsed.path.strip('/').split('/')
            url_parts.append((url, parts))

        # یافتن بخش‌های ثابت و متغیر
        fixed_indices = set()

        if url_parts:
            # تعیین طول بیشینه
            max_length = max(len(parts) for _, parts in url_parts)

            # بررسی هر موقعیت
            for i in range(max_length):
                values = [parts[i] if i < len(parts) else None for _, parts in url_parts]
                values = [v for v in values if v is not None]

                if values and len(set(values)) == 1:
                    fixed_indices.add(i)

        # گروه‌بندی بر اساس بخش‌های ثابت
        groups = defaultdict(list)

        for url, parts in url_parts:
            key = tuple(parts[i] if i < len(parts) else None for i in fixed_indices)
            groups[key].append(url)

        return list(groups.values())

    def _is_list_pattern(self, pattern):
        """
        تشخیص الگوی صفحات لیستی

        Args:
            pattern: الگوی URL

        Returns:
            bool: آیا الگو مربوط به صفحات لیستی است؟
        """
        list_indicators = [
            '/category/', '/tag/', '/archive/', '/blog/', '/articles/',
            '/questions/', '/list/', '/search/', '/page/', r'\?page=',
            'archive', 'category', 'tag', 'blog', 'articles'
        ]

        return any(indicator in pattern for indicator in list_indicators)

    def _is_detail_pattern(self, pattern):
        """
        تشخیص الگوی صفحات جزئیات

        Args:
            pattern: الگوی URL

        Returns:
            bool: آیا الگو مربوط به صفحات جزئیات است؟
        """
        # اگر URL شامل عدد یا slug است، احتمالاً صفحه جزئیات است
        has_id = '*' in pattern

        # اگر URL شامل شناسه‌گرهای صفحات جزئیات است
        detail_indicators = [
            '/post/', '/article/', '/question/', '/view/', '/show/',
            '/single/', r'/\d+/', r'/[^/]+/'
        ]

        has_detail_indicator = any(re.search(indicator, pattern) for indicator in detail_indicators)

        # نباید شاخص‌های صفحات لیستی داشته باشد
        is_not_list = not self._is_list_pattern(pattern)

        return (has_id or has_detail_indicator) and is_not_list

    def save_patterns(self, file_path):
        """
        ذخیره الگوهای کشف شده در فایل

        Args:
            file_path: مسیر فایل برای ذخیره‌سازی

        Returns:
            bool: آیا ذخیره‌سازی موفق بود؟
        """
        try:
            patterns_data = [pattern.to_dict() for pattern in self.patterns]

            data = {
                'base_url': self.base_url,
                'important_sections': self.important_sections,
                'patterns': patterns_data
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            logger.info(f"الگوهای URL با موفقیت در {file_path} ذخیره شدند")
            return True
        except Exception as e:
            logger.error(f"خطا در ذخیره‌سازی الگوهای URL: {str(e)}")
            return False

    def load_patterns(self, file_path):
        """
        بارگذاری الگوهای کشف شده از فایل

        Args:
            file_path: مسیر فایل برای بارگذاری

        Returns:
            bool: آیا بارگذاری موفق بود؟
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"فایل الگوهای URL یافت نشد: {file_path}")
                return False

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.base_url = data.get('base_url')
            self.important_sections = data.get('important_sections', [])

            patterns_data = data.get('patterns', [])
            self.patterns = [URLPattern.from_dict(pattern_data) for pattern_data in patterns_data]

            logger.info(f"الگوهای URL با موفقیت از {file_path} بارگذاری شدند")
            return True
        except Exception as e:
            logger.error(f"خطا در بارگذاری الگوهای URL: {str(e)}")
            return False

    def match_url(self, url):
        """
        تطبیق یک URL با الگوهای موجود

        Args:
            url: آدرس URL برای تطبیق

        Returns:
            URLPattern: الگوی مطابقت‌یافته یا None
        """
        for pattern in self.patterns:
            if pattern.matches(url):
                return pattern
        return None

    def get_pattern_for_url(self, url):
        """
        یافتن الگوی مناسب برای یک URL

        Args:
            url: آدرس URL

        Returns:
            URLPattern: الگوی مناسب یا None
        """
        # ابتدا بررسی می‌کنیم آیا الگوی موجودی مطابقت دارد
        existing_pattern = self.match_url(url)
        if existing_pattern:
            return existing_pattern

        # اگر الگوی موجودی یافت نشد، یک الگوی جدید ایجاد می‌کنیم
        parsed = urlparse(url)

        # تشخیص نوع صفحه
        is_list = self._is_list_pattern(url)
        is_detail = self._is_detail_pattern(url)

        # ایجاد الگوی جدید
        pattern = URLPattern(
            pattern=url,
            is_list=is_list,
            is_detail=is_detail,
            sample_urls=[url],
            weight=0.1  # وزن کم برای الگوهای جدید
        )

        return pattern

class StructureDiscovery:
    """کلاس اصلی شناسایی ساختار وبسایت"""

    def __init__(self, base_url, config_dir=None):
        """
        مقداردهی اولیه شناسایی ساختار

        Args:
            base_url: آدرس پایه وبسایت
            config_dir: مسیر پوشه پیکربندی (اختیاری)
        """
        self.base_url = base_url
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config'
        )

        # ایجاد پوشه پیکربندی در صورت عدم وجود
        os.makedirs(self.config_dir, exist_ok=True)

        # مسیر فایل‌های پیکربندی
        self.domain = urlparse(base_url).netloc
        self.domain_id = f"domain_{self.domain.replace('.', '_')}"
        self.url_patterns_file = os.path.join(self.config_dir, f"{self.domain}_url_patterns.json")
        self.html_patterns_file = os.path.join(self.config_dir, f"{self.domain}_html_patterns.json")

        # مدیریت درخواست‌ها
        self.request_manager = RequestManager(base_url=base_url)

        # کشف‌کنندگان الگو
        self.url_discoverer = URLStructureDiscovery(base_url=base_url)
        self.html_discoverer = HTMLPatternFinder()

        # دسترسی به دیتابیس
        self.db_ops = BaseDBOperations()

        # وضعیت شناسایی
        self.discovered = False

        # بارگذاری الگوهای ذخیره شده در صورت وجود
        self._load_patterns()

    def _load_patterns(self):
        """بارگذاری الگوهای ذخیره شده از دیتابیس یا فایل"""
        # ابتدا تلاش برای بارگذاری از دیتابیس
        db_loaded = self._load_patterns_from_db()

        # اگر از دیتابیس بارگذاری نشد، از فایل تلاش می‌کنیم
        file_loaded = False
        if not db_loaded:
            file_loaded = self._load_patterns_from_files()

        self.discovered = db_loaded or file_loaded

        if self.discovered:
            logger.info("الگوهای ساختاری با موفقیت بارگذاری شدند")
        else:
            logger.info("نیاز به کشف الگوهای ساختاری جدید")

    def _load_patterns_from_db(self):
        """بارگذاری الگوهای ذخیره شده از دیتابیس"""
        try:
            # بررسی وجود دامنه در دیتابیس
            domain = self.db_ops.get_by_id(Domain, self.domain_id)

            if not domain:
                logger.info(f"دامنه {self.domain_id} در دیتابیس یافت نشد")
                return False

            # بارگذاری الگوهای URL از دیتابیس
            if domain.keywords:
                # تبدیل JSON به داده پایتون
                if isinstance(domain.keywords, str):
                    domain_data = json.loads(domain.keywords)
                else:
                    domain_data = domain.keywords

                # اگر داده شامل الگوهای URL است
                if 'patterns' in domain_data:
                    patterns = []
                    for pattern_data in domain_data['patterns']:
                        pattern = URLPattern.from_dict(pattern_data)
                        patterns.append(pattern)

                    self.url_discoverer.patterns = patterns

                # اگر داده شامل بخش‌های مهم است
                if 'important_sections' in domain_data:
                    self.url_discoverer.important_sections = domain_data['important_sections']

                # بارگذاری الگوهای HTML (بسته به ساختار داده)
                if 'html_patterns' in domain_data:
                    # بازسازی الگوهای HTML
                    self.html_discoverer.list_selectors = domain_data.get('html_patterns', {}).get('list_selectors', {})
                    self.html_discoverer.detail_selectors = domain_data.get('html_patterns', {}).get('detail_selectors',
                                                                                                     {})

                logger.info(f"الگوهای ساختاری برای دامنه {self.domain} با موفقیت از دیتابیس بارگذاری شدند")
                return True
            else:
                logger.info(f"کلیدواژه‌های دامنه {self.domain_id} در دیتابیس خالی است")
                return False

        except Exception as e:
            logger.error(f"خطا در بارگذاری الگوها از دیتابیس: {str(e)}")
            return False

    def _load_patterns_from_files(self):
        """بارگذاری الگوهای ذخیره شده از فایل‌ها"""
        # بارگذاری الگوهای URL
        url_patterns_loaded = self.url_discoverer.load_patterns(self.url_patterns_file)

        # بارگذاری الگوهای HTML
        html_patterns_loaded = self.html_discoverer.load_patterns(self.html_patterns_file)

        return url_patterns_loaded and html_patterns_loaded

    def is_discovered(self):
        """
        بررسی وضعیت شناسایی ساختار

        Returns:
            bool: آیا ساختار شناسایی شده است؟
        """
        return self.discovered

    def discover_structure(self, max_pages=50, save=True):
        """
        کشف خودکار ساختار وبسایت

        Args:
            max_pages: حداکثر تعداد صفحات برای بررسی
            save: آیا الگوهای کشف شده ذخیره شوند؟

        Returns:
            bool: آیا عملیات موفق بود؟
        """
        try:
            logger.info(f"شروع کشف ساختار برای {self.base_url}")

            # بررسی صفحه اصلی
            logger.info("بررسی صفحه اصلی...")
            response = self.request_manager.get(self.base_url)

            if not response.get('html'):
                logger.error("خطا در دسترسی به صفحه اصلی")
                return False

            # استخراج لینک‌های صفحه اصلی
            links = self._extract_links(response.get('html'), self.base_url)

            # افزودن لینک‌های استخراج شده به مجموعه داده‌ها
            self.url_discoverer.add_urls(links)

            # تحلیل ساختار HTML صفحه اصلی
            self.html_discoverer.analyze_html_structure(response.get('html'), self.base_url)

            # بررسی صفحات بیشتر
            pages_visited = 1
            urls_to_visit = links[:max_pages]
            visited_urls = {self.base_url}

            for url in urls_to_visit:
                if pages_visited >= max_pages:
                    break

                if url in visited_urls:
                    continue

                logger.info(f"بررسی صفحه {pages_visited}/{max_pages}: {url}")

                try:
                    # درخواست صفحه
                    response = self.request_manager.get(url)

                    if not response.get('html'):
                        continue

                    # استخراج لینک‌های جدید
                    new_links = self._extract_links(response.get('html'), url)

                    # افزودن لینک‌های استخراج شده به مجموعه داده‌ها
                    self.url_discoverer.add_urls(new_links)

                    # تحلیل ساختار HTML صفحه
                    self.html_discoverer.analyze_html_structure(response.get('html'), url)

                    # افزودن URL‌های جدید به لیست بازدید
                    for link in new_links:
                        if link not in visited_urls and link not in urls_to_visit:
                            urls_to_visit.append(link)

                    visited_urls.add(url)
                    pages_visited += 1

                    # تأخیر بین درخواست‌ها
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"خطا در بررسی {url}: {str(e)}")

            # کشف الگوهای URL
            logger.info("کشف الگوهای URL...")
            url_patterns = self.url_discoverer.discover_patterns()

            # ذخیره الگوهای کشف شده
            if save:
                logger.info("ذخیره الگوهای کشف شده...")
                self.save_patterns()

            self.discovered = True

            logger.info(f"کشف ساختار با موفقیت انجام شد. {len(url_patterns)} الگوی URL شناسایی شد.")
            return True

        except Exception as e:
            logger.error(f"خطا در کشف ساختار: {str(e)}")
            return False

    def save_patterns(self):
        """ذخیره الگوهای کشف شده در دیتابیس و فایل"""
        # ذخیره در فایل (برای سازگاری با گذشته)
        self._save_patterns_to_files()

        # ذخیره در دیتابیس
        return self._save_patterns_to_db()

    def _save_patterns_to_files(self):
        """ذخیره الگوهای کشف شده در فایل‌ها"""
        # ذخیره الگوهای URL در فایل
        url_saved = self.url_discoverer.save_patterns(self.url_patterns_file)

        # ذخیره الگوهای HTML در فایل
        html_saved = self.html_discoverer.save_patterns(self.html_patterns_file)

        if url_saved and html_saved:
            logger.info(f"الگوهای ساختاری با موفقیت در فایل‌ها ذخیره شدند")
            return True
        else:
            logger.warning(f"مشکل در ذخیره الگوهای ساختاری در فایل‌ها")
            return False

    def _save_patterns_to_db(self):
        """ذخیره الگوهای کشف شده در دیتابیس"""
        try:
            # آماده‌سازی داده‌های الگوها
            patterns_data = [pattern.to_dict() for pattern in self.url_discoverer.patterns]

            # تهیه دیکشنری داده‌های دامنه
            domain_data = {
                'patterns': patterns_data,
                'important_sections': self.url_discoverer.important_sections,
                'html_patterns': {
                    'list_selectors': self.html_discoverer.list_selectors,
                    'detail_selectors': self.html_discoverer.detail_selectors
                }
            }

            # بررسی وجود دامنه در دیتابیس
            domain = self.db_ops.get_by_id(Domain, self.domain_id)

            if domain:
                # به‌روزرسانی دامنه موجود
                self.db_ops.update(
                    Domain,
                    self.domain_id,
                    keywords=json.dumps(domain_data, ensure_ascii=False)
                )
                logger.info(f"الگوهای ساختاری برای دامنه {self.domain_id} در دیتابیس به‌روزرسانی شدند")
            else:
                # ایجاد دامنه جدید
                new_domain = Domain.create(
                    id=self.domain_id,
                    name=self.domain,
                    description=f"Domain patterns for {self.domain}",
                    keywords=json.dumps(domain_data, ensure_ascii=False),
                    auto_detected=True
                )
                self.db_ops.create(new_domain)
                logger.info(f"دامنه جدید {self.domain_id} با الگوهای ساختاری در دیتابیس ایجاد شد")

            return True

        except Exception as e:
            logger.error(f"خطا در ذخیره الگوها در دیتابیس: {str(e)}")
            return False

    def _extract_links(self, html_content, base_url):
        """
        استخراج لینک‌های داخلی از یک صفحه

        Args:
            html_content: محتوای HTML
            base_url: آدرس پایه برای تکمیل لینک‌های نسبی

        Returns:
            list: لیست لینک‌های استخراج شده
        """
        if not html_content:
            return []

        try:
            # ایجاد شیء BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # استخراج تمام تگ‌های a با href
            links = []
            base_domain = urlparse(self.base_url).netloc

            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']

                # نادیده گرفتن لینک‌های خاص
                if href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    continue

                # تکمیل لینک‌های نسبی
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(base_url, href)

                # فقط لینک‌های داخلی را نگه می‌داریم
                parsed_href = urlparse(href)
                if parsed_href.netloc == base_domain:
                    # حذف پارامترهای غیرضروری و بخش‌های فرگمنت
                    clean_href = f"{parsed_href.scheme}://{parsed_href.netloc}{parsed_href.path}"

                    # حذف لینک‌های تکراری
                    if clean_href not in links:
                        links.append(clean_href)

            return links
        except Exception as e:
            logger.error(f"خطا در استخراج لینک‌ها: {str(e)}")
            return []

    def get_url_pattern(self, url):
        """
        دریافت الگوی URL برای یک آدرس

        Args:
            url: آدرس URL

        Returns:
            URLPattern: الگوی URL یا None
        """
        return self.url_discoverer.get_pattern_for_url(url)

    def get_html_selectors(self, url, url_type=None):
        """
        دریافت سلکتورهای HTML برای یک آدرس

        Args:
            url: آدرس URL
            url_type: نوع URL (اختیاری، 'list', 'detail')

        Returns:
            dict: سلکتورهای HTML
        """
        # تشخیص نوع URL
        if not url_type:
            pattern = self.get_url_pattern(url)
            if pattern:
                if pattern.is_list:
                    url_type = 'list'
                elif pattern.is_detail:
                    url_type = 'detail'
                else:
                    url_type = 'generic'
            else:
                url_type = 'generic'

        # دریافت سلکتورها بر اساس نوع
        if url_type == 'list':
            # یافتن سلکتورهای لیستی مشابه
            for pattern_url, selectors in self.html_discoverer.list_selectors.items():
                if self._url_similarity(url, pattern_url) > 0.7:
                    return selectors
        elif url_type == 'detail':
            # یافتن سلکتورهای جزئیات مشابه
            for pattern_url, selectors in self.html_discoverer.detail_selectors.items():
                if self._url_similarity(url, pattern_url) > 0.7:
                    return selectors

        # اگر سلکتور مناسبی یافت نشد، خالی برمی‌گردانیم
        return {}

    def _url_similarity(self, url1, url2):
        """
        محاسبه میزان شباهت بین دو URL

        Args:
            url1: آدرس اول
            url2: آدرس دوم

        Returns:
            float: میزان شباهت (0.0 تا 1.0)
        """
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)

        # اگر دامنه متفاوت است، شباهت کم است
        if parsed1.netloc != parsed2.netloc:
            return 0.1

        # تجزیه مسیرها
        path1 = parsed1.path.strip('/').split('/')
        path2 = parsed2.path.strip('/').split('/')

        # اگر طول مسیرها متفاوت است، شباهت کمتر است
        if len(path1) != len(path2):
            return 0.3

        # محاسبه تعداد بخش‌های مشترک
        common_parts = sum(1 for p1, p2 in zip(path1, path2) if p1 == p2)

        if not path1:
            return 1.0 if not path2 else 0.5

        return common_parts / max(len(path1), 1)

    def close(self):
        """بستن منابع و اتصالات"""
        if self.request_manager:
            self.request_manager.close()

