"""
ماژول مدل‌های محتوا برای خزشگر هوشمند داده‌های حقوقی

این ماژول مدل‌های داده مربوط به محتوا و روابط آن را تعریف می‌کند.
"""

import hashlib
from sqlalchemy import Column, BigInteger, String, Text, Integer, Float, Enum, JSON, ForeignKey, DateTime, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import BaseModel


class ContentItem(BaseModel):
    """مدل داده برای محتوای استخراج شده"""

    __tablename__ = 'content_items'

    # فیلدهای اصلی
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    content_type = Column(Enum('question', 'article', 'profile', 'other'), nullable=False)
    meta_data = Column(JSON, nullable=True)
    url = Column(String(255), nullable=False, unique=True)
    view_count = Column(Integer, default=0)
    similarity_hash = Column(String(64), nullable=True)
    status = Column(Enum('active', 'archived', 'processing', 'error'), default='active')

    # زمان‌های خاص
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    indexed_at = Column(DateTime, default=None, nullable=True)

    # روابط با سایر مدل‌ها
    domains = relationship("DomainContent", back_populates="content")
    answers = relationship("Answer", back_populates="question")

    @classmethod
    def create(cls, content, url, title=None, content_type='article', meta_data=None, created_at=None, updated_at=None):
        """
        ایجاد یک نمونه جدید از محتوا

        Args:
            content: متن اصلی محتوا
            url: آدرس اصلی محتوا
            title: عنوان محتوا (اختیاری)
            content_type: نوع محتوا ('question', 'article', 'profile', 'other')
            meta_data: داده‌های متا (دیکشنری)
            created_at: زمان ایجاد اصلی محتوا
            updated_at: زمان به‌روزرسانی اصلی محتوا

        Returns:
            ContentItem: نمونه جدید ایجاد شده
        """
        # محاسبه هش محتوا برای تشخیص تکرار
        hash_value = cls.calculate_similarity_hash(content)

        return cls(
            title=title,
            content=content,
            content_type=content_type,
            meta_data=meta_data,
            url=url,
            similarity_hash=hash_value,
            created_at=created_at,
            updated_at=updated_at
        )

    @staticmethod
    def calculate_similarity_hash(content, method='md5'):
        """
        محاسبه هش محتوا برای تشخیص محتوای مشابه

        Args:
            content: متن محتوا
            method: روش هش‌گذاری (پیش‌فرض: md5)

        Returns:
            str: هش محاسبه شده
        """
        if not content:
            return None

        # حذف فضاهای خالی اضافی و یکسان‌سازی
        normalized_content = ' '.join(content.split())

        if method == 'md5':
            return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()
        elif method == 'sha256':
            return hashlib.sha256(normalized_content.encode('utf-8')).hexdigest()

        # پیش‌فرض به md5
        return hashlib.md5(normalized_content.encode('utf-8')).hexdigest()

    def update_content(self, new_content, update_hash=True):
        """
        به‌روزرسانی محتوا و هش مربوطه

        Args:
            new_content: محتوای جدید
            update_hash: آیا هش مشابهت نیز به‌روزرسانی شود؟

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        self.content = new_content
        if update_hash:
            self.similarity_hash = self.calculate_similarity_hash(new_content)
        return self

    @hybrid_property
    def content_summary(self):
        """
        خلاصه محتوا (حداکثر 150 کاراکتر)

        Returns:
            str: خلاصه محتوا
        """
        if not self.content:
            return ""
        if len(self.content) <= 150:
            return self.content
        return self.content[:147] + "..."

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<ContentItem(id={self.id}, title='{self.title if self.title else ''}', type='{self.content_type}', status='{self.status}')>"


class Answer(BaseModel):
    """مدل داده برای پاسخ‌های متخصصان به پرسش‌ها"""

    __tablename__ = 'answers'

    # فیلدهای اصلی
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    content_id = Column(BigInteger, ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False)
    expert_id = Column(BigInteger, ForeignKey('experts.id', ondelete='CASCADE'), nullable=False)
    text = Column(Text, nullable=False)
    rating = Column(Float, default=0)
    status = Column(Enum('active', 'archived', 'hidden'), default='active')

    # زمان‌های خاص
    created_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)

    # روابط با سایر مدل‌ها
    question = relationship("ContentItem", back_populates="answers")
    expert = relationship("Expert", back_populates="answers")

    @classmethod
    def create(cls, content_id, expert_id, text, rating=0, created_at=None):
        """
        ایجاد یک نمونه جدید از پاسخ

        Args:
            content_id: شناسه محتوای پرسش
            expert_id: شناسه متخصص
            text: متن پاسخ
            rating: امتیاز پاسخ (0.0 تا 5.0)
            created_at: زمان ایجاد اصلی پاسخ

        Returns:
            Answer: نمونه جدید ایجاد شده
        """
        return cls(
            content_id=content_id,
            expert_id=expert_id,
            text=text,
            rating=rating,
            created_at=created_at
        )

    @hybrid_property
    def text_summary(self):
        """
        خلاصه متن پاسخ (حداکثر 100 کاراکتر)

        Returns:
            str: خلاصه متن
        """
        if not self.text:
            return ""
        if len(self.text) <= 100:
            return self.text
        return self.text[:97] + "..."

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<Answer(id={self.id}, expert_id={self.expert_id}, rating={self.rating}, status='{self.status}')>"


class DomainContent(BaseModel):
    """مدل داده برای ارتباط بین حوزه‌های تخصصی و محتوا"""

    __tablename__ = 'domain_content'

    # فیلدهای اصلی
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    domain_id = Column(String(50), ForeignKey('domains.id', ondelete='CASCADE'), nullable=False)
    content_id = Column(BigInteger, ForeignKey('content_items.id', ondelete='CASCADE'), nullable=False)
    relevance_score = Column(Float, default=0)

    # روابط با سایر مدل‌ها
    domain = relationship("Domain", back_populates="contents")
    content = relationship("ContentItem", back_populates="domains")

    @classmethod
    def create(cls, domain_id, content_id, relevance_score=0.5):
        """
        ایجاد یک نمونه جدید از ارتباط حوزه-محتوا

        Args:
            domain_id: شناسه حوزه تخصصی
            content_id: شناسه محتوا
            relevance_score: میزان ارتباط (0.0 تا 1.0)

        Returns:
            DomainContent: نمونه جدید ایجاد شده
        """
        return cls(
            domain_id=domain_id,
            content_id=content_id,
            relevance_score=relevance_score
        )

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<DomainContent(domain_id='{self.domain_id}', content_id={self.content_id}, relevance={self.relevance_score})>"