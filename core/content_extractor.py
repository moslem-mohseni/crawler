"""
ماژول استخراج محتوا برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل کلاس ContentExtractor است که وظیفه استخراج دقیق و ساختارمند
اطلاعات از صفحات HTML را بر عهده دارد. این اطلاعات شامل عنوان، محتوای اصلی،
تاریخ انتشار، نویسنده و موجودیت‌های نام‌دار (با استفاده از مدل‌های NLP) می‌باشد.

این کلاس به گونه‌ای طراحی شده است که با سایر بخش‌های پروژه مانند خزش (crawler)
و طبقه‌بندی (classifier) یکپارچه عمل کند.
"""

import os
from bs4 import BeautifulSoup
from utils.logger import get_logger

try:
    import spacy
except ImportError:
    spacy = None


class ContentExtractor:
    """
    کلاس استخراج محتوا:
    این کلاس وظیفه استخراج اطلاعات ساختاری از محتوای HTML یک صفحه وب را بر عهده دارد.
    """

    def __init__(self, nlp_model_path=None):
        """
        مقداردهی اولیه استخراج‌کننده محتوا.
        در صورت ارائه مسیر مدل NLP، تلاش می‌شود مدل مربوطه بارگذاری شود.
        در غیر این صورت، سعی بر بارگذاری مدل پیش‌فرض زبان فارسی (fa_core_news_sm) خواهد شد.

        Args:
            nlp_model_path (str, optional): مسیر فایل مدل NLP برای تحلیل موجودیت‌ها.
        """
        self.logger = get_logger(__name__)
        self.nlp = None

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

    def extract(self, html_content, url, job_type=None):
        """
        استخراج اطلاعات ساختاری از محتوای HTML صفحه.
        این متد صفحه را پردازش کرده و اطلاعاتی مانند عنوان، محتوای اصلی،
        تاریخ انتشار، نویسنده و موجودیت‌های استخراج‌شده را برمی‌گرداند.

        Args:
            html_content (str): محتوای HTML صفحه.
            url (str): آدرس صفحه.
            job_type (str, optional): نوع صفحه (مثلاً 'page', 'list', 'detail').

        Returns:
            dict: دیکشنری شامل اطلاعات استخراج‌شده.
                  شامل کلیدهای "url"، "title"، "content"، "date"، "author" و "entities".
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # حذف عناصر جانبی و ناخواسته (header, footer, nav, aside)
        for tag in soup.find_all(['header', 'footer', 'nav', 'aside']):
            tag.decompose()

        title = self._extract_title(soup)
        main_content = self._extract_main_content(soup)
        date = self._extract_date(soup)
        author = self._extract_author(soup)
        entities = self._extract_entities(main_content) if self.nlp else {}

        extracted_data = {
            "url": url,
            "title": title,
            "content": main_content,
            "date": date,
            "author": author,
            "entities": entities
        }

        self.logger.info(f"اطلاعات استخراج‌شده از {url} با موفقیت به‌دست آمد")
        return extracted_data

    def _extract_main_content(self, soup):
        """
        استخراج محتوای اصلی صفحه با یافتن بزرگترین بلوک متنی.

        Args:
            soup (BeautifulSoup): شیء BeautifulSoup صفحه.

        Returns:
            str: متن استخراج‌شده به عنوان محتوای اصلی.
        """
        candidates = soup.find_all(['article', 'div', 'section'])
        max_text = ""
        for candidate in candidates:
            text = candidate.get_text(separator=" ", strip=True)
            if len(text) > len(max_text):
                max_text = text
        if max_text:
            return max_text
        else:
            # استفاده از کل متن صفحه در صورت عدم یافتن بلوک مشخص
            return soup.get_text(separator="\n", strip=True)

    def _extract_title(self, soup):
        """
        استخراج عنوان صفحه از تگ‌های <title> یا <h1>.

        Args:
            soup (BeautifulSoup): شیء BeautifulSoup صفحه.

        Returns:
            str: عنوان استخراج‌شده.
        """
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _extract_date(self, soup):
        """
        استخراج تاریخ انتشار از تگ‌های <time> یا متا.

        Args:
            soup (BeautifulSoup): شیء BeautifulSoup صفحه.

        Returns:
            str: تاریخ استخراج‌شده.
        """
        time_tag = soup.find('time')
        if time_tag:
            return time_tag.get_text(strip=True)
        meta_date = soup.find('meta', {'property': 'article:published_time'})
        if meta_date and meta_date.get('content'):
            return meta_date['content'].strip()
        return ""

    def _extract_author(self, soup):
        """
        استخراج نویسنده صفحه از تگ‌های دارای کلاس 'author' یا تگ متا.

        Args:
            soup (BeautifulSoup): شیء BeautifulSoup صفحه.

        Returns:
            str: نام نویسنده استخراج‌شده.
        """
        author_tag = soup.find(class_=lambda x: x and "author" in x.lower())
        if author_tag:
            return author_tag.get_text(strip=True)
        meta_author = soup.find('meta', {'name': 'author'})
        if meta_author and meta_author.get('content'):
            return meta_author['content'].strip()
        return ""

    def _extract_entities(self, text):
        """
        استخراج موجودیت‌های نام‌دار از متن با استفاده از مدل NLP.

        Args:
            text (str): متن ورودی.

        Returns:
            dict: دیکشنری شامل موجودیت‌های استخراج‌شده به تفکیک برچسب.
        """
        entities = {}
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                label = ent.label_
                if label not in entities:
                    entities[label] = []
                entities[label].append(ent.text.strip())
            # حذف تکرارهای موجودیت
            for label in entities:
                entities[label] = list(set(entities[label]))
        except Exception as e:
            self.logger.error(f"خطا در استخراج موجودیت‌ها: {str(e)}")
        return entities
