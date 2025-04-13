"""
ماژول ذخیره‌سازی و ایندکس‌گذاری:
این ماژول شامل کلاس StorageManager است که وظیفه مدیریت ذخیره‌سازی داده‌های استخراج‌شده
(مانند محتواهای استخراج شده توسط ContentExtractor) را در پایگاه داده بر عهده دارد.
این کلاس به‌صورت یکپارچه با مدل‌های پروژه (مانند ContentItem)، سیستم پایگاه داده و سیستم لاگ‌گیری عمل می‌کند.
"""

from datetime import datetime
import threading
from typing import Dict, List, Optional, Any, Union

from utils.logger import get_logger
from database.connection import DatabaseConnection
from database.operations import BaseDBOperations
from models.content import ContentItem, DomainContent
from models.domain import Domain

logger = get_logger(__name__)
StatsDict = Dict[str, Union[int, dict, datetime, float, None, Dict[str, int]]]

class StorageManager:
    """
    کلاس مدیریت ذخیره‌سازی:
    این کلاس متدهایی برای ذخیره، به‌روزرسانی و بازیابی محتوا از پایگاه داده ارائه می‌دهد.
    همچنین با استفاده از هش مبتنی بر محتوای استخراج‌شده (با استفاده از ContentItem.calculate_similarity_hash)
    از درج داده‌های تکراری جلوگیری می‌کند.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """پیاده‌سازی الگوی Singleton برای اطمینان از وجود تنها یک نمونه از کلاس"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StorageManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        """مقداردهی اولیه وضعیت ذخیره‌سازی"""
        if self._initialized:
            return

        self.logger = get_logger(__name__)
        self.db = DatabaseConnection()
        self.db_ops = BaseDBOperations()

        # آمار و اطلاعات ذخیره‌سازی
        self.stats: StatsDict = {
            'total_items_stored': 0,
            'total_duplicates_skipped': 0,
            'items_by_content_type': {},
            'items_by_domain': {},
            'start_time': datetime.now(),
            'last_store_time': None
        }

        self._initialized = True

    def get_content_by_hash(self, similarity_hash: str) -> Optional[ContentItem]:
        """
        بازیابی یک رکورد محتوا بر اساس هش مشابهت.

        Args:
            similarity_hash (str): هش محتوای مورد نظر.

        Returns:
            ContentItem یا None: نمونه محتوا در صورت یافتن، در غیر این صورت None.
        """
        try:
            result = self.db_ops.get_all(
                ContentItem,
                limit=1,
                similarity_hash=similarity_hash
            )
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"خطا در بازیابی محتوا با هش {similarity_hash}: {str(e)}")
            return None

    def get_content_by_url(self, url: str) -> Optional[ContentItem]:
        """
        بازیابی یک رکورد محتوا بر اساس URL.

        Args:
            url (str): آدرس URL محتوا.

        Returns:
            ContentItem یا None: نمونه محتوا در صورت یافتن، در غیر این صورت None.
        """
        try:
            result = self.db_ops.get_all(
                ContentItem,
                limit=1,
                url=url
            )
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"خطا در بازیابی محتوا با URL {url}: {str(e)}")
            return None

    def store_content(self, content_data: Dict[str, Any]) -> Optional[ContentItem]:
        """
        ذخیره یا به‌روزرسانی یک رکورد محتوا در پایگاه داده.
        این متد ابتدا با محاسبه هش محتوا (با استفاده از متد calculate_similarity_hash در ContentItem)
        بررسی می‌کند که آیا محتوای مشابهی قبلاً ذخیره شده است یا خیر.
        در صورت وجود، رکورد موجود به‌روز شده و در غیر این صورت، رکورد جدید درج می‌شود.

        Args:
            content_data (dict): دیکشنری شامل اطلاعات محتوا، مانند:
                {
                    "url": <str>,
                    "title": <str>,
                    "content": <str>,
                    "date": <str>,
                    "author": <str>,
                    "entities": <dict>,
                    "content_type": <str>,  # (مانند 'question', 'article', 'profile', 'other')
                    "domains": <list>  # لیست شناسه‌های حوزه‌های تخصصی
                }

        Returns:
            ContentItem یا None: نمونه ذخیره‌شده در صورت موفقیت، یا None در صورت بروز خطا.
        """
        # اطمینان از وجود متن محتوا
        content_text = content_data.get("content", "")
        if not content_text:
            self.logger.error("متن محتوا خالی است؛ ذخیره‌سازی انجام نشد.")
            return None

        # محاسبه هش مشابهت
        similarity_hash = ContentItem.calculate_similarity_hash(content_text)
        content_data["similarity_hash"] = similarity_hash

        # آماده‌سازی داده‌های متادیتا به‌صورت دیکشنری
        meta = {
            "date": content_data.get("date", ""),
            "author": content_data.get("author", ""),
            "entities": content_data.get("entities", {})
        }

        # استخراج حوزه‌های تخصصی
        domains = content_data.get("domains", [])

        try:
            # بررسی وجود محتوای مشابه با هش یکسان
            existing_content = self.get_content_by_hash(similarity_hash)

            if existing_content:
                self.logger.info(f"محتوا با هش {similarity_hash} قبلاً ذخیره شده است. به‌روزرسانی رکورد.")

                # به‌روزرسانی فیلدها
                updated_content = self.db_ops.update(
                    ContentItem,
                    existing_content.id,
                    title=content_data.get("title", existing_content.title),
                    content=content_text,
                    content_type=content_data.get("content_type", existing_content.content_type),
                    meta_data=meta,
                    updated_at=datetime.now()
                )

                # به‌روزرسانی آمار
                self.stats['total_duplicates_skipped'] += 1
                self.stats['last_store_time'] = datetime.now()

                # به‌روزرسانی روابط با حوزه‌های تخصصی
                if domains:
                    self._update_domain_relationships(existing_content.id, domains)

                return updated_content
            else:
                self.logger.info(f"درج محتوای جدید با هش {similarity_hash}.")

                # استفاده از متد create مدل برای ایجاد نمونه جدید
                new_content = ContentItem.create(
                    content=content_text,
                    url=content_data.get("url", ""),
                    title=content_data.get("title", ""),
                    content_type=content_data.get("content_type", "other"),
                    meta_data=meta,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

                # درج رکورد جدید
                stored_content = self.db_ops.create(new_content)

                if stored_content:
                    # ایجاد روابط با حوزه‌های تخصصی
                    if domains:
                        self._create_domain_relationships(stored_content.id, domains)

                    # به‌روزرسانی آمار
                    self.stats['total_items_stored'] += 1
                    self.stats['last_store_time'] = datetime.now()

                    # به‌روزرسانی آمار نوع محتوا
                    content_type = content_data.get("content_type", "other")
                    self.stats['items_by_content_type'][content_type] = self.stats['items_by_content_type'].get(content_type, 0) + 1

                    # به‌روزرسانی آمار حوزه‌های تخصصی
                    for domain_id in domains:
                        self.stats['items_by_domain'][domain_id] = self.stats['items_by_domain'].get(domain_id, 0) + 1

                return stored_content

        except Exception as e:
            self.logger.error(f"خطا در ذخیره‌سازی محتوا: {str(e)}")
            return None

    def _create_domain_relationships(self, content_id: int, domain_ids: List[str]) -> None:
        """
        ایجاد روابط بین محتوا و حوزه‌های تخصصی.

        Args:
            content_id (int): شناسه محتوا
            domain_ids (List[str]): لیست شناسه‌های حوزه‌های تخصصی
        """
        for domain_id in domain_ids:
            try:
                # بررسی وجود حوزه تخصصی
                domain = self.db_ops.get_by_id(Domain, domain_id)
                if not domain:
                    self.logger.warning(f"حوزه تخصصی با شناسه {domain_id} یافت نشد")
                    continue

                # ایجاد رابطه جدید
                domain_content = DomainContent.create(
                    domain_id=domain_id,
                    content_id=content_id,
                    relevance_score=0.8  # مقدار پیش‌فرض
                )

                self.db_ops.create(domain_content)
                self.logger.debug(f"رابطه بین محتوا {content_id} و حوزه {domain_id} ایجاد شد")

            except Exception as e:
                self.logger.error(f"خطا در ایجاد رابطه بین محتوا {content_id} و حوزه {domain_id}: {str(e)}")

    def _update_domain_relationships(self, content_id: int, domain_ids: List[str]) -> None:
        """
        به‌روزرسانی روابط بین محتوا و حوزه‌های تخصصی.

        Args:
            content_id (int): شناسه محتوا
            domain_ids (List[str]): لیست شناسه‌های حوزه‌های تخصصی
        """
        try:
            # دریافت روابط موجود
            existing_relationships = self.db_ops.get_all(
                DomainContent,
                content_id=content_id
            )

            existing_domain_ids = [rel.domain_id for rel in existing_relationships]

            # حذف روابط قدیمی که در لیست جدید نیستند
            for rel in existing_relationships:
                if rel.domain_id not in domain_ids:
                    self.db_ops.delete(DomainContent, rel.id)
                    self.logger.debug(f"رابطه بین محتوا {content_id} و حوزه {rel.domain_id} حذف شد")

            # ایجاد روابط جدید
            for domain_id in domain_ids:
                if domain_id not in existing_domain_ids:
                    self._create_domain_relationships(content_id, [domain_id])

        except Exception as e:
            self.logger.error(f"خطا در به‌روزرسانی روابط محتوا {content_id}: {str(e)}")

    def bulk_store_contents(self, contents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        ذخیره‌سازی دسته‌ای چندین محتوای استخراج‌شده.
        برای هر محتوا، ابتدا بررسی تکراری بودن انجام شده و در صورت عدم وجود، رکورد جدید درج می‌شود.

        Args:
            contents (list): لیستی از دیکشنری‌های محتوا.

        Returns:
            dict: شامل تعداد موفق و ناموفق بودن ذخیره‌سازی.
        """
        success_count = 0
        failure_count = 0
        duplicate_count = 0

        for content_data in contents:
            result = self.store_content(content_data)
            if result:
                if hasattr(result, '_is_updated') and result._is_updated:
                    duplicate_count += 1
                else:
                    success_count += 1
            else:
                failure_count += 1

        self.logger.info(f"ذخیره‌سازی دسته‌ای: {success_count} موفق، {duplicate_count} به‌روزرسانی، {failure_count} ناموفق")
        return {
            "success": success_count,
            "updated": duplicate_count,
            "failure": failure_count
        }

    def extract_and_store(self, extracted_data: Dict[str, Any], classification_data: Optional[Dict[str, Any]] = None) -> Optional[ContentItem]:
        """
        تلفیق داده‌های استخراج شده و طبقه‌بندی و ذخیره‌سازی آن‌ها.

        Args:
            extracted_data (dict): داده‌های استخراج شده از ContentExtractor
            classification_data (dict, optional): داده‌های طبقه‌بندی از Classifier

        Returns:
            ContentItem یا None: نمونه ذخیره‌شده در صورت موفقیت، یا None در صورت بروز خطا.
        """
        # ترکیب داده‌های استخراج شده و طبقه‌بندی
        content_data = extracted_data.copy()

        if classification_data:
            # افزودن نوع محتوا
            if 'content_type' in classification_data:
                content_data['content_type'] = classification_data['content_type'].get('content_type', 'other')

            # افزودن حوزه‌های تخصصی
            if 'domains' in classification_data:
                domain_list = classification_data['domains'].get('domains', [])
                content_data['domains'] = domain_list

        # ذخیره‌سازی محتوای ترکیب شده
        return self.store_content(content_data)

    def index_content(self, content_item: ContentItem) -> bool:
        """
        (اختیاری) به‌روزرسانی ایندکس‌های معنایی برای یک محتوای ذخیره‌شده.
        این متد می‌تواند شامل پردازش‌های اضافی برای بهبود جستجو و بازیابی اطلاعات باشد.

        Args:
            content_item (ContentItem): نمونه محتوای ذخیره‌شده.

        Returns:
            bool: True در صورت موفقیت، False در صورت بروز خطا.
        """
        try:
            # به عنوان مثال، می‌توان از Elasticsearch یا سیستم‌های جستجوی متن استفاده کرد.
            # در اینجا به سادگی یک لاگ ثبت می‌شود.
            self.logger.info(f"ایندکس‌بندی محتوای با آیدی {content_item.id} و عنوان {content_item.title}")

            # ایجاد تغییری ساده در وضعیت محتوا برای نشان دادن ایندکس‌گذاری
            self.db_ops.update(ContentItem, content_item.id, indexed_at=datetime.now())
            return True
        except Exception as e:
            self.logger.error(f"خطا در ایندکس‌بندی محتوا: {str(e)}")
            return False

    def get_stats(self) -> StatsDict:

        """
        دریافت آمار ذخیره‌سازی.

        Returns:
            Dict[str, Any]: آمار ذخیره‌سازی
        """
        # به‌روزرسانی برخی آمارها از دیتابیس
        try:
            # دریافت تعداد کل محتواها
            total_count = self.db_ops.count(ContentItem)

            # دریافت تعداد محتواها بر اساس نوع
            content_types = {
                'question': self.db_ops.count(ContentItem, content_type='question'),
                'article': self.db_ops.count(ContentItem, content_type='article'),
                'profile': self.db_ops.count(ContentItem, content_type='profile'),
                'other': self.db_ops.count(ContentItem, content_type='other')
            }

            # ترکیب آمار فعلی با آمار محاسبه شده
            stats = self.stats.copy()
            stats['total_db_items'] = total_count
            stats['content_types_db'] = content_types
            stats['runtime_seconds'] = (datetime.now() - self.stats['start_time']).total_seconds()

            return stats
        except Exception as e:
            self.logger.error(f"خطا در دریافت آمار ذخیره‌سازی: {str(e)}")
            return self.stats
