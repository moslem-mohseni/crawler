"""
ماژول مدل حوزه‌های تخصصی برای خزشگر هوشمند داده‌های حقوقی

این ماژول مدل داده مربوط به حوزه‌های تخصصی حقوقی را تعریف می‌کند.
"""

from sqlalchemy import Column, String, Text, Boolean, Float, JSON
from sqlalchemy.orm import relationship

from models.base import BaseModel


class Domain(BaseModel):
    """مدل داده برای حوزه‌های تخصصی حقوقی"""

    __tablename__ = 'domains'

    # فیلدهای اصلی
    id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)
    auto_detected = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)

    # روابط با سایر مدل‌ها
    contents = relationship("DomainContent", back_populates="domain")
    experts = relationship("ExpertDomain", back_populates="domain")

    @classmethod
    def create(cls, name, description=None, keywords=None, auto_detected=False, confidence=1.0, id=None):
        """
        ایجاد یک نمونه جدید از حوزه تخصصی

        Args:
            name: نام حوزه تخصصی
            description: توضیحات حوزه
            keywords: کلیدواژه‌های مرتبط (لیست یا دیکشنری)
            auto_detected: آیا این حوزه به صورت خودکار شناسایی شده؟
            confidence: میزان اطمینان از تشخیص خودکار (0.0 تا 1.0)
            id: شناسه اختیاری، در صورت عدم ارائه یک شناسه تصادفی تولید می‌شود

        Returns:
            Domain: نمونه جدید ایجاد شده
        """
        domain_id = id if id else cls.generate_id("domain")

        return cls(
            id=domain_id,
            name=name,
            description=description,
            keywords=keywords,
            auto_detected=auto_detected,
            confidence=confidence
        )

    def add_keyword(self, keyword, score=None):
        """
        افزودن یک کلیدواژه جدید به حوزه

        Args:
            keyword: کلیدواژه جدید
            score: امتیاز اهمیت کلیدواژه (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if self.keywords is None:
            self.keywords = []

        if isinstance(self.keywords, list):
            if score is None:
                if keyword not in self.keywords:
                    self.keywords.append(keyword)
            else:
                # تبدیل لیست به دیکشنری در صورت نیاز به ذخیره امتیاز
                keywords_dict = {k: 1.0 for k in self.keywords} if self.keywords else {}
                keywords_dict[keyword] = score
                self.keywords = keywords_dict
        elif isinstance(self.keywords, dict):
            self.keywords[keyword] = score if score is not None else 1.0

        return self

    def remove_keyword(self, keyword):
        """
        حذف یک کلیدواژه از حوزه

        Args:
            keyword: کلیدواژه برای حذف

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if self.keywords is None:
            return self

        if isinstance(self.keywords, list) and keyword in self.keywords:
            self.keywords.remove(keyword)
        elif isinstance(self.keywords, dict) and keyword in self.keywords:
            del self.keywords[keyword]

        return self

    def get_top_keywords(self, limit=10):
        """
        دریافت مهم‌ترین کلیدواژه‌های حوزه

        Args:
            limit: حداکثر تعداد کلیدواژه‌ها

        Returns:
            list: لیست کلیدواژه‌ها یا لیست توپل‌های (کلیدواژه، امتیاز)
        """
        if not self.keywords:
            return []

        if isinstance(self.keywords, list):
            return self.keywords[:limit]
        elif isinstance(self.keywords, dict):
            # مرتب‌سازی بر اساس امتیاز نزولی
            sorted_keywords = sorted(self.keywords.items(), key=lambda x: x[1], reverse=True)
            return sorted_keywords[:limit]

        return []

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<Domain(id='{self.id}', name='{self.name}', auto_detected={self.auto_detected}, confidence={self.confidence})>"