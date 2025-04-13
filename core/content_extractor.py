"""
ماژول استخراج محتوا برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس ContentExtractor است که وظیفه استخراج دقیق و ساختارمند
اطلاعات از صفحات HTML را بر عهده دارد. این اطلاعات شامل عنوان، محتوای اصلی،
تاریخ انتشار، نویسنده و موجودیت‌های نام‌دار (با استفاده از مدل‌های NLP) می‌باشد.

این کلاس به گونه‌ای طراحی شده است که با سایر بخش‌های پروژه مانند خزش (crawler)،
طبقه‌بندی (classifier) و ذخیره‌سازی (storage) یکپارچه عمل کند.
"""

import os
import re
import threading
import concurrent.futures
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup, Tag

from utils.logger import get_logger
from utils.text import normalize_persian_text


# پویانمایی import برای جلوگیری از وابستگی دوری
def import_classifier():
    from core.classifier import TextClassifier
    return TextClassifier()


def import_storage_manager():
    from core.storage import StorageManager
    return StorageManager()


# تلاش برای import spacy
try:
    import spacy
except ImportError:
    spacy = None


class ContentExtractor:
    """
    کلاس استخراج محتوا:
    این کلاس وظیفه استخراج اطلاعات ساختاری از محتوای HTML یک صفحه وب را بر عهده دارد.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, nlp_model_path: Optional[str] = None,
                use_classifier: bool = True,
                auto_store: bool = False) -> 'ContentExtractor':
        """پیاده‌سازی الگوی Singleton برای بهینه‌سازی منابع"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ContentExtractor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, nlp_model_path: Optional[str] = None,
                 use_classifier: bool = True,
                 auto_store: bool = False) -> None:
        """
        مقداردهی اولیه استخراج‌کننده محتوا.
        در صورت ارائه مسیر مدل NLP، تلاش می‌شود مدل مربوطه بارگذاری شود.
        در غیر این صورت، سعی بر بارگذاری مدل پیش‌فرض زبان فارسی (fa_core_news_sm) خواهد شد.

        Args:
            nlp_model_path: مسیر فایل مدل NLP برای تحلیل موجودیت‌ها.
            use_classifier: آیا از طبقه‌بندی کننده استفاده شود؟
            auto_store: آیا محتوا به صورت خودکار در پایگاه داده ذخیره شود؟
        """
        if getattr(self, '_initialized', False):
            return

        self.logger = get_logger(__name__)
        self.nlp = None
        self.classifier = None
        self.storage_manager = None
        self.use_classifier = use_classifier
        self.auto_store = auto_store

        # آمار استخراج
        self.stats: Dict[str, Union[int, float, datetime, None, str]] = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'last_extraction_time': None,
            'start_time': datetime.now()
        }

        # بارگذاری مدل NLP
        self._load_nlp_model(nlp_model_path)

        # بارگذاری طبقه‌بندی کننده در صورت نیاز
        if self.use_classifier:
            try:
                self.classifier = import_classifier()
                self.logger.info("طبقه‌بندی کننده با موفقیت بارگذاری شد")
            except Exception as e:
                self.logger.error(f"خطا در بارگذاری طبقه‌بندی کننده: {str(e)}")
                self.use_classifier = False

        # مقداردهی مدیر ذخیره‌سازی در صورت نیاز به ذخیره خودکار
        if self.auto_store:
            try:
                self.storage_manager = import_storage_manager()
                self.logger.info("مدیر ذخیره‌سازی با موفقیت بارگذاری شد")
            except Exception as e:
                self.logger.error(f"خطا در بارگذاری مدیر ذخیره‌سازی: {str(e)}")
                self.auto_store = False

        self._initialized = True

    def _load_nlp_model(self, nlp_model_path: Optional[str] = None) -> None:
        """
        بارگذاری مدل NLP برای استخراج موجودیت‌ها.

        Args:
            nlp_model_path: مسیر فایل مدل NLP.
        """
        if spacy is not None:
            try:
                if nlp_model_path and os.path.exists(nlp_model_path):
                    self.nlp = spacy.load(nlp_model_path)
                    self.logger.info(f"مدل NLP از {nlp_model_path} بارگذاری شد")
                else:
                    try:
                        self.nlp = spacy.load("fa_core_news_sm")
                        self.logger.info("مدل پیش‌فرض fa_core_news_sm بارگذاری شد")
                    except Exception as e:
                        self.logger.warning("مدل NLP فارسی یافت نشد. تحلیل موجودیت‌ها غیرفعال خواهد شد")
                        self.nlp = None
            except Exception as e:
                self.logger.error(f"خطا در بارگذاری مدل NLP: {str(e)}")
                self.nlp = None
        else:
            self.logger.warning("کتابخانه spacy نصب نشده است. تحلیل موجودیت‌ها غیرفعال خواهد شد")

    def extract(self, html_content: str, url: str, job_type: Optional[str] = None) -> Dict[str, Any]:
        """
        استخراج اطلاعات ساختاری از محتوای HTML صفحه.
        این متد صفحه را پردازش کرده و اطلاعاتی مانند عنوان، محتوای اصلی،
        تاریخ انتشار، نویسنده و موجودیت‌های استخراج‌شده را برمی‌گرداند.

        Args:
            html_content: محتوای HTML صفحه.
            url: آدرس صفحه.
            job_type: نوع صفحه (مثلاً 'page', 'list', 'detail').

        Returns:
            دیکشنری شامل اطلاعات استخراج‌شده.
            شامل کلیدهای "url"، "title"، "content"، "date"، "author" و "entities".
        """
        self.stats['total_extractions'] += 1
        self.stats['last_extraction_time'] = datetime.now()

        try:
            # اطمینان از وجود محتوای HTML
            if not html_content:
                self.logger.warning(f"محتوای HTML خالی برای URL {url}")
                self.stats['failed_extractions'] += 1
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "date": "",
                    "author": "",
                    "entities": {},
                    "error": "محتوای HTML خالی"
                }

            # پارس HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # حذف عناصر جانبی و ناخواسته
            self._clean_soup(soup)

            # استخراج اطلاعات اصلی با استفاده از پردازش موازی
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # استخراج عنوان
                title_future = executor.submit(self._extract_title, soup)

                # استخراج محتوای اصلی
                content_future = executor.submit(self._extract_main_content, soup, job_type)

                # استخراج تاریخ
                date_future = executor.submit(self._extract_date, soup)

                # استخراج نویسنده
                author_future = executor.submit(self._extract_author, soup)

                # دریافت نتایج
                title = title_future.result()
                main_content = content_future.result()
                date = date_future.result()
                author = author_future.result()

            # استخراج موجودیت‌ها (غیر موازی به دلیل وابستگی به محتوا)
            entities = self._extract_entities(main_content) if self.nlp else {}

            # ایجاد دیکشنری نتیجه
            extracted_data = {
                "url": url,
                "title": title,
                "content": main_content,
                "date": date,
                "author": author,
                "entities": entities,
                "job_type": job_type,
                "extraction_time": datetime.now().isoformat()
            }

            # استخراج داده‌های اضافی بر اساس نوع صفحه
            if job_type == 'list':
                extracted_data["list_items"] = self._extract_list_items(soup)
            elif job_type == 'detail':
                extracted_data["related_links"] = self._extract_related_links(soup, url)

            self.stats['successful_extractions'] += 1
            self.logger.info(f"اطلاعات استخراج‌شده از {url} با موفقیت به‌دست آمد")

            return extracted_data

        except Exception as e:
            self.stats['failed_extractions'] += 1
            self.logger.error(f"خطا در استخراج اطلاعات از {url}: {str(e)}")

            return {
                "url": url,
                "title": "",
                "content": "",
                "date": "",
                "author": "",
                "entities": {},
                "error": str(e)
            }

    def extract_and_classify(self, html_content: str, url: str, job_type: Optional[str] = None) -> Dict[str, Any]:
        """
        استخراج و طبقه‌بندی محتوا در یک فرایند.

        Args:
            html_content: محتوای HTML صفحه.
            url: آدرس صفحه.
            job_type: نوع صفحه.

        Returns:
            دیکشنری شامل اطلاعات استخراج و طبقه‌بندی شده.
        """
        # استخراج محتوا
        extracted_data = self.extract(html_content, url, job_type)

        # طبقه‌بندی محتوا در صورت موفقیت استخراج و فعال بودن طبقه‌بندی کننده
        if self.use_classifier and self.classifier and 'error' not in extracted_data:
            try:
                classification = self.classifier.classify_text(extracted_data['content'])

                # افزودن نتایج طبقه‌بندی به داده‌های استخراج شده
                extracted_data.update({
                    'classification_results': classification,
                    'content_type': classification.get('content_type', {}).get('content_type', 'other'),
                    'domains': classification.get('domains', {}).get('domains', [])
                })

                self.logger.info(f"محتوای {url} با موفقیت طبقه‌بندی شد")
            except Exception as e:
                self.logger.error(f"خطا در طبقه‌بندی محتوای {url}: {str(e)}")
                extracted_data['classification_error'] = str(e)

        # ذخیره‌سازی خودکار در صورت نیاز
        if self.auto_store and self.storage_manager and 'error' not in extracted_data:
            try:
                stored_item = self.storage_manager.store_content(extracted_data)
                if stored_item:
                    extracted_data['stored_id'] = stored_item.id
                    self.logger.info(f"محتوای {url} با موفقیت در پایگاه داده ذخیره شد (ID: {stored_item.id})")
            except Exception as e:
                self.logger.error(f"خطا در ذخیره‌سازی محتوای {url}: {str(e)}")
                extracted_data['storage_error'] = str(e)

        return extracted_data

    def bulk_extract(self, urls: List[str], html_contents: List[str],
                    job_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        استخراج دسته‌ای محتوا از چندین صفحه.

        Args:
            urls: لیست آدرس‌های صفحات.
            html_contents: لیست محتواهای HTML.
            job_types: لیست انواع صفحات.

        Returns:
            لیست دیکشنری‌های شامل اطلاعات استخراج شده.
        """
        if not urls or not html_contents:
            return []

        if job_types is None:
            job_types = ['page'] * len(urls)

        results = []

        # استفاده از ThreadPoolExecutor برای پردازش موازی
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(self.extract_and_classify, html, url, job_type): url
                for url, html, job_type in zip(urls, html_contents, job_types)
            }

            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    self.logger.error(f"خطا در استخراج محتوای {url}: {str(e)}")
                    results.append({
                        "url": url,
                        "error": str(e)
                    })

        return results

    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """
        حذف عناصر ناخواسته از HTML.

        Args:
            soup: شیء BeautifulSoup.
        """
        # حذف اسکریپت‌ها و استایل‌ها
        for tag in soup.find_all(['script', 'style', 'iframe', 'noscript']):
            tag.decompose()

        # حذف عناصر جانبی
        for tag in soup.find_all(['header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # حذف تگ‌های مبتنی بر کلاس‌های معمول تبلیغات و محتوای اضافی
        ad_classes = ['ads', 'advertisement', 'banner', 'popup', 'social', 'sharing', 'footer', 'menu']
        for cls in ad_classes:
            for tag in soup.find_all(class_=lambda x: x and cls in x.lower()):
                tag.decompose()

    @staticmethod
    def _extract_main_content(soup: BeautifulSoup, job_type: Optional[str] = None) -> str:
        """
        استخراج محتوای اصلی صفحه با یافتن بزرگترین بلوک متنی.

        Args:
            soup: شیء BeautifulSoup صفحه.
            job_type: نوع صفحه.

        Returns:
            متن استخراج‌شده به عنوان محتوای اصلی.
        """
        # استراتژی‌های مختلف بر اساس نوع صفحه
        if job_type == 'detail':
            # روش‌های خاص برای صفحات جزئیات
            content_candidates = [
                soup.find('article'),
                soup.find('div', class_=re.compile(r'(content|article|post|body|text|main)')),
                soup.find('main'),
                soup.find('section', class_=re.compile(r'(content|article)'))
            ]

            for candidate in content_candidates:
                if candidate and len(candidate.get_text(strip=True)) > 200:
                    return candidate.get_text(separator=" ", strip=True)

        # روش‌های عمومی
        candidates = soup.find_all(['article', 'div', 'section'])

        # ایجاد لیست کاندیداها با امتیازدهی
        scored_candidates = []
        for candidate in candidates:
            if not candidate or not isinstance(candidate, Tag):
                continue

            text = candidate.get_text(separator=" ", strip=True)
            score = len(text)

            # افزایش امتیاز برای محتوای با پاراگراف‌های متعدد
            p_tags = candidate.find_all('p')
            if p_tags and len(p_tags) > 2:
                score += len(p_tags) * 50

            # افزایش امتیاز برای محتوای با تگ‌های معنایی
            if candidate.find_all(['h1', 'h2', 'h3']):
                score += 100

            # کاهش امتیاز برای محتوای با لینک‌های زیاد
            a_tags = candidate.find_all('a')
            if a_tags:
                a_text = sum(len(a.get_text(strip=True)) for a in a_tags)
                a_ratio = a_text / max(1, len(text))
                if a_ratio > 0.5:
                    score -= 200

            scored_candidates.append((score, text))

        # مرتب‌سازی و انتخاب بهترین کاندیدا
        if scored_candidates:
            scored_candidates.sort(reverse=True)
            return scored_candidates[0][1]

        # در صورت عدم یافتن محتوای مناسب، استفاده از کل متن صفحه
        return soup.get_text(separator=" ", strip=True)

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """
        استخراج عنوان صفحه از تگ‌های <title> یا <h1>.

        Args:
            soup: شیء BeautifulSoup صفحه.

        Returns:
            عنوان استخراج‌شده.
        """
        # استراتژی 1: تگ title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
            # حذف بخش‌های ثابت سایت از عنوان (مانند نام سایت)
            title = re.sub(r'\s*[|]\s*.+$', '', title)
            title = re.sub(r'\s*[-]\s*.+$', '', title)
            return title

        # استراتژی 2: تگ h1 اصلی
        h1_tags = soup.find_all('h1')
        if h1_tags:
            for h1 in h1_tags:
                if h1.get_text(strip=True):
                    return h1.get_text(strip=True)

        # استراتژی 3: جستجو در کلاس‌های معمول عنوان
        title_classes = ['title', 'heading', 'post-title', 'article-title', 'main-title']
        for cls in title_classes:
            title_elem = soup.find(class_=re.compile(cls, re.I))
            if title_elem and title_elem.get_text(strip=True):
                return title_elem.get_text(strip=True)

        # استراتژی 4: اولین h2 اگر h1 یافت نشد
        h2 = soup.find('h2')
        if h2 and h2.get_text(strip=True):
            return h2.get_text(strip=True)

        return ""

    @staticmethod
    def _extract_date(soup: BeautifulSoup) -> str:
        """
        استخراج تاریخ انتشار از تگ‌های <time> یا متا.

        Args:
            soup: شیء BeautifulSoup صفحه.

        Returns:
            تاریخ استخراج‌شده.
        """
        # استراتژی 1: تگ time
        time_tags = soup.find_all('time')
        for tag in time_tags:
            if tag.has_attr('datetime'):
                return tag['datetime'].strip()
            elif tag.get_text(strip=True):
                return tag.get_text(strip=True)

        # استراتژی 2: متاتگ‌های مربوط به تاریخ
        meta_tags = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'property': 'article:modified_time'}),
            ('meta', {'name': 'date'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'publish_date'})
        ]

        for tag_name, attrs in meta_tags:
            tag = soup.find(tag_name, attrs)
            if tag and tag.get('content'):
                return tag['content'].strip()

        # استراتژی 3: جستجو در کلاس‌های معمول تاریخ
        date_classes = ['date', 'time', 'published', 'pubdate', 'timestamp']
        for cls in date_classes:
            date_elem = soup.find(class_=re.compile(cls, re.I))
            if date_elem and date_elem.get_text(strip=True):
                return date_elem.get_text(strip=True)

        # استراتژی 4: جستجوی الگوهای تاریخ در متن
        html_text = soup.get_text()
        date_patterns = [
            r'تاریخ(?:\s*انتشار)?[:]\s*(\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+[آ-یa-zA-Z]+\s+\d{4})',
            r'(\d{4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}\s+[آ-یa-zA-Z]+\s+\d{4})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, html_text)
            if match:
                return match.group(1).strip()

        return ""

    @staticmethod
    def _extract_author(soup: BeautifulSoup) -> str:
        """
        استخراج نام نویسنده از تگ‌های مختلف.

        Args:
            soup: شیء BeautifulSoup صفحه.

        Returns:
            نام نویسنده استخراج‌شده.
        """
        # استراتژی 1: متاتگ نویسنده
        meta_author = soup.find('meta', {'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content'].strip()

        # استراتژی 2: تگ‌های با کلاس یا شناسه مرتبط با نویسنده
        author_classes = ['author', 'writer', 'byline', 'by']

        for cls in author_classes:
            author_tag = soup.find(class_=re.compile(cls, re.I))
            if author_tag and author_tag.get_text(strip=True):
                # پاکسازی متن استخراج شده
                author_text = author_tag.get_text(strip=True)
                # حذف پیشوندهای متداول
                author_text = re.sub(r'^(?:نویسنده|نگارنده|نوشته)[:]\s*', '', author_text, flags=re.I)
                return author_text

        # استراتژی 3: جستجوی الگوهای نویسنده در متن
        html_text = soup.get_text()
        author_patterns = [
            r'نویسنده[:]\s*([آ-یA-Za-z\s]+)',
            r'نگارنده[:]\s*([آ-یA-Za-z\s]+)',
            r'نوشته[:]\s*([آ-یA-Za-z\s]+)'
        ]

        for pattern in author_patterns:
            match = re.search(pattern, html_text)
            if match:
                author = match.group(1).strip()
                # اعتبارسنجی طول نام
                if 2 < len(author) < 50:
                    return author

        return ""

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        استخراج موجودیت‌های نام‌دار از متن با استفاده از مدل NLP.

        Args:
            text: متن ورودی.

        Returns:
            دیکشنری شامل موجودیت‌های استخراج‌شده به تفکیک برچسب.
        """
        entities: Dict[str, List[str]] = {}

        if not self.nlp or not text:
            return entities

        try:
            # پیش‌پردازش و محدود کردن طول متن برای عملکرد بهتر
            if len(text) > 10000:
                text = text[:10000]  # محدود کردن طول برای بهبود کارایی

            # نرمال‌سازی متن
            normalized_text = normalize_persian_text(text)

            # پردازش متن با مدل spaCy
            doc = self.nlp(normalized_text)

            # استخراج و دسته‌بندی موجودیت‌ها
            for ent in doc.ents:
                label = ent.label_

                if label not in entities:
                    entities[label] = []

                # نرمال‌سازی و افزودن موجودیت
                entity_text = ent.text.strip()

                # اضافه کردن موجودیت اگر تکراری نباشد
                if entity_text and entity_text not in entities[label]:
                    entities[label].append(entity_text)

            # حذف تکرارهای موجودیت و مرتب‌سازی
            for label in entities:
                entities[label] = sorted(list(set(entities[label])))

        except Exception as e:
            self.logger.error(f"خطا در استخراج موجودیت‌ها: {str(e)}")

        return entities

    @staticmethod
    def _extract_list_items(soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        استخراج آیتم‌های لیستی در صفحات لیست.

        Args:
            soup: شیء BeautifulSoup صفحه.

        Returns:
            لیست آیتم‌های استخراج شده (هر آیتم شامل عنوان، لینک و خلاصه).
        """
        items = []

        # یافتن کانتینر لیست
        list_containers = [
            soup.find('ul', class_=re.compile(r'(list|items|posts|articles)')),
            soup.find('div', class_=re.compile(r'(list|items|posts|articles)')),
            soup.find('section', class_=re.compile(r'(list|items|posts|articles)'))
        ]

        container = next((c for c in list_containers if c is not None), None)

        if not container:
            # در صورت عدم یافتن کانتینر مشخص، جستجوی مستقیم آیتم‌ها
            item_elements = soup.find_all(['article', 'div', 'li'],
                class_=re.compile(r'(item|post|article)'))
        else:
            # جستجو در کانتینر
            item_elements = container.find_all(['article', 'div', 'li'])

        # پردازش آیتم‌ها
        for item in item_elements:
            item_data = {}

            # استخراج عنوان و لینک
            title_elem = item.find(['h2', 'h3', 'h4', 'a'])
            if title_elem:
                item_data['title'] = title_elem.get_text(strip=True)

                # استخراج لینک
                link = None
                if title_elem.name == 'a' and title_elem.has_attr('href'):
                    link = title_elem['href']
                else:
                    a_tag = title_elem.find('a')
                    if a_tag and a_tag.has_attr('href'):
                        link = a_tag['href']

                item_data['link'] = link

            # استخراج خلاصه
            summary_elem = item.find(['p', 'div'], class_=re.compile(r'(summary|excerpt|desc)'))
            if summary_elem:
                item_data['summary'] = summary_elem.get_text(strip=True)

            # اضافه کردن آیتم به لیست در صورت وجود حداقل عنوان یا لینک
            if item_data.get('title') or item_data.get('link'):
                items.append(item_data)

        return items

    @staticmethod
    def _extract_related_links(soup: BeautifulSoup, current_url: str) -> List[Dict[str, str]]:
        """
        استخراج لینک‌های مرتبط در صفحات جزئیات.

        Args:
            soup: شیء BeautifulSoup صفحه.
            current_url: آدرس فعلی برای تکمیل لینک‌های نسبی.

        Returns:
            لیست لینک‌های مرتبط (هر لینک شامل عنوان و آدرس).
        """
        related_links = []

        # استراتژی 1: جستجو در کانتینرهای معمول لینک‌های مرتبط
        related_containers = [
            soup.find('div', class_=re.compile(r'(related|similar|suggested)')),
            soup.find('section', class_=re.compile(r'(related|similar|suggested)')),
            soup.find('ul', class_=re.compile(r'(related|similar|suggested)'))
        ]

        container = next((c for c in related_containers if c is not None), None)

        if container:
            # جستجوی لینک‌ها در کانتینر
            for a_tag in container.find_all('a', href=True):
                href = a_tag['href']

                # رد کردن لینک‌های خاص
                if href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                    continue

                # تکمیل لینک‌های نسبی
                if not href.startswith(('http://', 'https://')):
                    parsed_current = urlparse(current_url)
                    base_url = f"{parsed_current.scheme}://{parsed_current.netloc}"
                    href = urljoin(base_url, href)

                related_links.append({
                    'title': a_tag.get_text(strip=True),
                    'url': href
                })

        # استراتژی 2: جستجوی عمومی در بخش پایین صفحه
        if not related_links:
            # محدود کردن جستجو به نیمه پایین صفحه
            body = soup.find('body')
            if body:
                all_elements = body.find_all()

                # فرض کنیم لینک‌های مرتبط در نیمه دوم صفحه قرار دارند
                start_index = len(all_elements) // 2

                for element in all_elements[start_index:]:
                    if element.name == 'a' and element.has_attr('href'):
                        href = element['href']

                        # حذف لینک‌های نامرتبط
                        if href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                            continue

                        # تکمیل لینک‌های نسبی
                        if not href.startswith(('http://', 'https://')):
                            parsed_current = urlparse(current_url)
                            base_url = f"{parsed_current.scheme}://{parsed_current.netloc}"
                            href = urljoin(base_url, href)

                        # افزودن لینک در صورت وجود متن
                        if element.get_text(strip=True):
                            related_links.append({
                                'title': element.get_text(strip=True),
                                'url': href
                            })

        # حذف لینک‌های تکراری و محدود کردن به 10 لینک
        unique_links = []
        seen_urls = set()

        for link in related_links:
            if link['url'] not in seen_urls:
                seen_urls.add(link['url'])
                unique_links.append(link)

                if len(unique_links) >= 10:
                    break

        return unique_links

    def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار استخراج محتوا.

        Returns:
            آمار استخراج محتوا
        """
        # محاسبه زمان اجرا
        runtime_seconds = (datetime.now() - self.stats['start_time']).total_seconds()

        # محاسبه نرخ موفقیت
        success_rate = 0
        if self.stats['total_extractions'] > 0:
            success_rate = self.stats['successful_extractions'] / self.stats['total_extractions']

        # آمار به‌روزرسانی شده
        stats = self.stats.copy()
        stats['runtime_seconds'] = runtime_seconds
        stats['success_rate'] = success_rate

        return stats