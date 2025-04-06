"""
ماژول مدل متخصصان برای خزشگر هوشمند داده‌های حقوقی

این ماژول مدل داده مربوط به متخصصان حقوقی و روابط آن‌ها را تعریف می‌کند.
"""

from sqlalchemy import Column, BigInteger, String, Text, Float, Integer, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from models.base import BaseModel


class Expert(BaseModel):
    """مدل داده برای متخصصان حقوقی"""

    __tablename__ = 'experts'

    # فیلدهای اصلی
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    bio = Column(Text, nullable=True)
    expertise = Column(JSON, nullable=True)
    rating = Column(Float, default=0)
    answers_count = Column(Integer, default=0)
    profile_url = Column(String(255), unique=True, nullable=True)
    avatar_url = Column(String(255), nullable=True)

    # روابط با سایر مدل‌ها
    domains = relationship("ExpertDomain", back_populates="expert")
    answers = relationship("Answer", back_populates="expert")

    @classmethod
    def create(cls, name, profile_url=None, bio=None, expertise=None, avatar_url=None):
        """
        ایجاد یک نمونه جدید از متخصص

        Args:
            name: نام کامل متخصص
            profile_url: آدرس پروفایل متخصص
            bio: بیوگرافی متخصص
            expertise: تخصص‌های متخصص (لیست یا دیکشنری)
            avatar_url: آدرس تصویر پروفایل

        Returns:
            Expert: نمونه جدید ایجاد شده
        """
        return cls(
            name=name,
            bio=bio,
            expertise=expertise,
            profile_url=profile_url,
            avatar_url=avatar_url
        )

    def add_expertise(self, expertise_name, level=None):
        """
        افزودن یک تخصص جدید به متخصص

        Args:
            expertise_name: نام تخصص
            level: سطح تخصص (اختیاری)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if self.expertise is None:
            self.expertise = []

        if isinstance(self.expertise, list):
            if level is None:
                if expertise_name not in self.expertise:
                    self.expertise.append(expertise_name)
            else:
                # تبدیل لیست به دیکشنری در صورت نیاز به ذخیره سطح
                expertise_dict = {e: 1.0 for e in self.expertise} if self.expertise else {}
                expertise_dict[expertise_name] = level
                self.expertise = expertise_dict
        elif isinstance(self.expertise, dict):
            self.expertise[expertise_name] = level if level is not None else 1.0

        return self

    def remove_expertise(self, expertise_name):
        """
        حذف یک تخصص از متخصص

        Args:
            expertise_name: نام تخصص برای حذف

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if self.expertise is None:
            return self

        if isinstance(self.expertise, list) and expertise_name in self.expertise:
            self.expertise.remove(expertise_name)
        elif isinstance(self.expertise, dict) and expertise_name in self.expertise:
            del self.expertise[expertise_name]

        return self

    def update_answers_count(self, value=None):
        """
        به‌روزرسانی تعداد پاسخ‌ها

        Args:
            value: مقدار جدید (در صورت عدم ارائه، تعداد پاسخ‌های موجود شمارش می‌شود)

        Returns:
            self: خود آبجکت برای استفاده زنجیره‌ای
        """
        if value is not None:
            self.answers_count = value
        else:
            # شمارش پاسخ‌های فعال
            self.answers_count = sum(1 for answer in self.answers if answer.status == 'active')

        return self

    @hybrid_property
    def bio_summary(self):
        """
        خلاصه بیوگرافی (حداکثر 100 کاراکتر)

        Returns:
            str: خلاصه بیوگرافی
        """
        if not self.bio:
            return ""
        if len(self.bio) <= 100:
            return self.bio
        return self.bio[:97] + "..."

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<Expert(id={self.id}, name='{self.name}', rating={self.rating}, answers_count={self.answers_count})>"


class ExpertDomain(BaseModel):
    """مدل داده برای ارتباط بین متخصصان و حوزه‌های تخصصی"""

    __tablename__ = 'expert_domain'

    # فیلدهای اصلی
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    expert_id = Column(BigInteger, ForeignKey('experts.id', ondelete='CASCADE'), nullable=False)
    domain_id = Column(String(50), ForeignKey('domains.id', ondelete='CASCADE'), nullable=False)
    confidence_score = Column(Float, default=0)

    # روابط با سایر مدل‌ها
    expert = relationship("Expert", back_populates="domains")
    domain = relationship("Domain", back_populates="experts")

    @classmethod
    def create(cls, expert_id, domain_id, confidence_score=0.5):
        """
        ایجاد یک نمونه جدید از ارتباط متخصص-حوزه

        Args:
            expert_id: شناسه متخصص
            domain_id: شناسه حوزه تخصصی
            confidence_score: میزان اطمینان از ارتباط (0.0 تا 1.0)

        Returns:
            ExpertDomain: نمونه جدید ایجاد شده
        """
        return cls(
            expert_id=expert_id,
            domain_id=domain_id,
            confidence_score=confidence_score
        )

    def __repr__(self):
        """
        نمایش رشته‌ای مدل

        Returns:
            str: رشته نمایشی
        """
        return f"<ExpertDomain(expert_id={self.expert_id}, domain_id='{self.domain_id}', confidence={self.confidence_score})>"