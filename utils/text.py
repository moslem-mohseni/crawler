"""
ماژول پردازش متن برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل توابع و کلاس‌های مختلف برای پردازش و نرمال‌سازی متن است.
"""

import re
import string
import unicodedata
from bs4 import BeautifulSoup
import hashlib

from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)

# کاراکترهای ویژه فارسی که باید یکدست شوند
PERSIAN_CHARS_MAP = {
    'ك': 'ک',  # کاف عربی به کاف فارسی
    'ي': 'ی',  # یای عربی به یای فارسی
    '١': '1',  # اعداد عربی به انگلیسی
    '٢': '2',
    '٣': '3',
    '٤': '4',
    '٥': '5',
    '٦': '6',
    '٧': '7',
    '٨': '8',
    '٩': '9',
    '٠': '0',
    'ة': 'ه',  # تای گرد به های معمولی
    'ئ': 'ی',  # همزه‌دار به ساده
    'إ': 'ا',
    'أ': 'ا',
    'آ': 'ا',
    'ؤ': 'و',
    '‌': ' ',  # نیم‌فاصله به فاصله
}

# لیست کلمات ایست فارسی
PERSIAN_STOP_WORDS = [
    'از', 'به', 'با', 'در', 'بر', 'را', 'که', 'این', 'آن', 'و', 'یا', 'اما', 'ولی',
    'برای', 'تا', 'هر', 'چه', 'چرا', 'اگر', 'مگر', 'پس', 'نیز', 'حتی', 'همه', 'هیچ',
    'خود', 'باید', 'شاید', 'چون', 'زیرا', 'بنابراین', 'سپس', 'گرچه', 'درباره', 'بدون',
    'توسط', 'علاوه', 'بین', 'همچنین', 'بسیار', 'برخی', 'می', 'های', 'ها', 'ی', 'است',
    'نیست', 'بود', 'شد', 'شود', 'کرد', 'کند', 'شده', 'می‌شود', 'می‌کند', 'دارد', 'ندارد'
]


def clean_html(html_content):
    """
    حذف تگ‌های HTML و استخراج متن خالص

    Args:
        html_content: محتوای HTML

    Returns:
        str: متن استخراج شده بدون تگ‌های HTML
    """
    if not html_content:
        return ""

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # حذف تگ‌های اسکریپت و استایل
        for tag in soup.find_all(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()

        # استخراج متن
        text = soup.get_text(separator=' ')

        # نرمال‌سازی فضای خالی
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        logger.error(f"خطا در تمیزسازی HTML: {str(e)}")
        # بازگشت متن خام در صورت خطا
        return re.sub(r'<.*?>', '', html_content)


def extract_text_from_tags(html_content, tags, class_=None, id=None):
    """
    استخراج متن از تگ‌های خاص در HTML

    Args:
        html_content: محتوای HTML
        tags: لیست تگ‌های مورد نظر (مثلاً ['h1', 'p'])
        class_: کلاس CSS برای محدود کردن جستجو (اختیاری)
        id: شناسه HTML برای محدود کردن جستجو (اختیاری)

    Returns:
        str: متن استخراج شده از تگ‌های مشخص شده
    """
    if not html_content:
        return ""

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # جستجو با اعمال فیلترها
        results = []
        for tag in tags:
            if class_ and id:
                elements = soup.find_all(tag, class_=class_, id=id)
            elif class_:
                elements = soup.find_all(tag, class_=class_)
            elif id:
                elements = soup.find_all(tag, id=id)
            else:
                elements = soup.find_all(tag)

            for element in elements:
                results.append(element.get_text().strip())

        return " ".join(results)
    except Exception as e:
        logger.error(f"خطا در استخراج متن از تگ‌ها: {str(e)}")
        return ""


def normalize_persian_text(text):
    """
    نرمال‌سازی متن فارسی

    Args:
        text: متن ورودی

    Returns:
        str: متن نرمال‌سازی شده
    """
    if not text:
        return ""

    # جایگزینی کاراکترهای ویژه
    for k, v in PERSIAN_CHARS_MAP.items():
        text = text.replace(k, v)

    # یکسان‌سازی فاصله‌ها
    text = re.sub(r'\s+', ' ', text).strip()

    # حذف اعراب
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c))

    return text


def tokenize_persian_text(text, remove_stop_words=True, remove_punctuation=True):
    """
    تقسیم متن فارسی به توکن‌ها

    Args:
        text: متن ورودی
        remove_stop_words: آیا کلمات ایست حذف شوند؟
        remove_punctuation: آیا علائم نگارشی حذف شوند؟

    Returns:
        list: لیست توکن‌ها
    """
    if not text:
        return []

    # نرمال‌سازی متن
    text = normalize_persian_text(text)

    # حذف علائم نگارشی
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation + '،؛؟»«!')
        text = text.translate(translator)

    # تقسیم به توکن‌ها
    tokens = text.split()

    # حذف کلمات ایست
    if remove_stop_words:
        tokens = [t for t in tokens if t not in PERSIAN_STOP_WORDS]

    return tokens


def calculate_text_hash(text, method='md5'):
    """
    محاسبه هش متن برای مقایسه مشابهت

    Args:
        text: متن ورودی
        method: روش هش‌گذاری ('md5', 'sha1', 'sha256')

    Returns:
        str: هش محاسبه شده
    """
    if not text:
        return None

    # نرمال‌سازی متن قبل از هش‌گذاری
    text = normalize_persian_text(text)
    text = ' '.join(text.split())  # یکسان‌سازی فاصله‌ها

    # انتخاب روش هش‌گذاری
    if method == 'md5':
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    elif method == 'sha1':
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    elif method == 'sha256':
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    else:
        # پیش‌فرض: md5
        return hashlib.md5(text.encode('utf-8')).hexdigest()


def extract_main_content(html_content):
    """
    استخراج محتوای اصلی صفحه با حذف بخش‌های فرعی

    Args:
        html_content: محتوای HTML

    Returns:
        str: محتوای اصلی استخراج شده
    """
    if not html_content:
        return ""

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # حذف تگ‌های غیرضروری
        for tag in soup.find_all(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        # یافتن بخش اصلی محتوا (با ابتکار)
        main_tags = soup.find_all(['article', 'main', 'div'],
                                  class_=re.compile(r'(content|article|post|body|main)', re.I))

        if main_tags:
            # انتخاب بزرگترین بخش محتوا
            main_content = max(main_tags, key=lambda tag: len(tag.get_text()))

            # استخراج متن
            text = main_content.get_text(separator=' ')

            # نرمال‌سازی فضای خالی
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        else:
            # برگشت به روش ساده در صورت عدم یافتن بخش‌های خاص
            return clean_html(html_content)

    except Exception as e:
        logger.error(f"خطا در استخراج محتوای اصلی: {str(e)}")
        # بازگشت به روش ساده در صورت خطا
        return clean_html(html_content)


def extract_title(html_content):
    """
    استخراج عنوان صفحه

    Args:
        html_content: محتوای HTML

    Returns:
        str: عنوان استخراج شده
    """
    if not html_content:
        return ""

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # بررسی تگ title
        title_tag = soup.title
        if title_tag:
            title = title_tag.string.strip()

            # حذف بخش‌های تکراری (مانند نام سایت)
            title = re.sub(r'\s*\|.*$', '', title)
            title = re.sub(r'\s*-.*$', '', title)

            return title

        # بررسی تگ‌های h1
        h1_tags = soup.find_all('h1')
        if h1_tags:
            return h1_tags[0].get_text().strip()

        # بررسی سایر تگ‌های عنوان
        for tag in soup.find_all(['h2', 'header']):
            text = tag.get_text().strip()
            if text and len(text) < 200:  # عنوان‌ها معمولاً کوتاه هستند
                return text

        return ""
    except Exception as e:
        logger.error(f"خطا در استخراج عنوان: {str(e)}")
        return ""


def extract_date(html_content):
    """
    استخراج تاریخ انتشار از HTML

    Args:
        html_content: محتوای HTML

    Returns:
        str: تاریخ استخراج شده یا None در صورت عدم یافتن
    """
    if not html_content:
        return None

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # بررسی تگ‌های time
        time_tags = soup.find_all('time')
        if time_tags:
            for tag in time_tags:
                if tag.has_attr('datetime'):
                    return tag['datetime']
                else:
                    return tag.get_text().strip()

        # بررسی متا تگ‌ها
        meta_tags = soup.find_all('meta', attrs={'name': re.compile(r'(publish|date|time)', re.I)})
        if meta_tags:
            for tag in meta_tags:
                if tag.has_attr('content'):
                    return tag['content']

        # بررسی الگوهای متنی تاریخ فارسی
        text = soup.get_text()
        date_patterns = [
            r'تاریخ انتشار:?\s*(\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4})',
            r'تاریخ:?\s*(\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4})',
            r'(\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4})',
            r'(\d{2,4}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{1,2})'
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return None
    except Exception as e:
        logger.error(f"خطا در استخراج تاریخ: {str(e)}")
        return None


def extract_author(html_content):
    """
    استخراج نام نویسنده از HTML

    Args:
        html_content: محتوای HTML

    Returns:
        str: نام نویسنده یا None در صورت عدم یافتن
    """
    if not html_content:
        return None

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # بررسی متا تگ‌ها
        meta_tags = soup.find_all('meta', attrs={'name': re.compile(r'author', re.I)})
        if meta_tags:
            for tag in meta_tags:
                if tag.has_attr('content'):
                    return tag['content']

        # بررسی تگ‌های با کلاس یا شناسه مربوط به نویسنده
        author_tags = soup.find_all(['a', 'span', 'div'], class_=re.compile(r'(author|نویسنده)', re.I))
        if author_tags:
            return author_tags[0].get_text().strip()

        # بررسی الگوهای متنی نویسنده فارسی
        text = soup.get_text()
        author_patterns = [
            r'نویسنده:?\s*([^\n\.،]*)',
            r'نگارنده:?\s*([^\n\.،]*)',
            r'نگارش:?\s*([^\n\.،]*)',
            r'تهیه کننده:?\s*([^\n\.،]*)'
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                author = match.group(1).strip()
                if 5 < len(author) < 50:  # محدودیت طول منطقی
                    return author

        return None
    except Exception as e:
        logger.error(f"خطا در استخراج نویسنده: {str(e)}")
        return None


def extract_links(html_content, base_url=None, internal_only=False):
    """
    استخراج لینک‌ها از HTML

    Args:
        html_content: محتوای HTML
        base_url: آدرس پایه برای تکمیل لینک‌های نسبی
        internal_only: فقط لینک‌های داخلی استخراج شوند؟

    Returns:
        list: لیست لینک‌های استخراج شده
    """
    if not html_content:
        return []

    try:
        # ایجاد شیء BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # دریافت تمام تگ‌های a با href
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']

            # حذف تگ‌های خاص
            if href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue

            # تکمیل آدرس‌های نسبی
            if base_url and not href.startswith(('http://', 'https://')):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)

            # فیلتر کردن لینک‌های خارجی
            if internal_only and base_url:
                from urllib.parse import urlparse
                base_domain = urlparse(base_url).netloc
                href_domain = urlparse(href).netloc

                if href_domain and href_domain != base_domain:
                    continue

            links.append(href)

        return links
    except Exception as e:
        logger.error(f"خطا در استخراج لینک‌ها: {str(e)}")
        return []


def is_similar_content(text1, text2, threshold=0.8):
    """
    بررسی شباهت بین دو متن

    Args:
        text1: متن اول
        text2: متن دوم
        threshold: آستانه شباهت (0.0 تا 1.0)

    Returns:
        bool: آیا دو متن مشابه هستند؟
    """
    if not text1 or not text2:
        return False

    # نرمال‌سازی متون
    text1 = normalize_persian_text(text1)
    text2 = normalize_persian_text(text2)

    # حذف فضاهای خالی اضافی
    text1 = ' '.join(text1.split())
    text2 = ' '.join(text2.split())

    # مقایسه طول متون
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    if len_ratio < threshold:
        return False

    # محاسبه هش متون
    hash1 = calculate_text_hash(text1)
    hash2 = calculate_text_hash(text2)

    # بررسی شباهت کامل
    if hash1 == hash2:
        return True

    # در نسخه‌های پیشرفته‌تر، می‌توان از الگوریتم‌های شباهت متن استفاده کرد
    # مانند Levenshtein distance یا Jaccard similarity
    # اما این نسخه ساده فقط بر اساس هش عمل می‌کند

    return False
