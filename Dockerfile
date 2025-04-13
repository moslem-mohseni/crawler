# استفاده از تصویر پایه پایتون 3.11-slim
FROM python:3.11-slim

# تنظیم متغیرهای محیطی
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tehran

# نصب ابزارهای ضروری
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# تنظیم دایرکتوری کاری در کانتینر
WORKDIR /app

# کپی فایل requirements.txt برای نصب وابستگی‌ها
COPY requirements.txt ./

# به‌روزرسانی pip و نصب وابستگی‌ها
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# کپی کل کدهای پروژه به کانتینر
COPY . .

# ساخت پوشه‌های مورد نیاز
RUN mkdir -p /app/logs /app/config /app/data

# تعیین حقوق دسترسی
RUN chmod +x main.py

# تعریف مسیر ذخیره‌سازی داده‌ها
VOLUME ["/app/data", "/app/config", "/app/logs"]

# تعریف پورت (در صورت نیاز برای رابط وب)
# EXPOSE 8080

# سلامت سنجی
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# اجرای برنامه (نقطه ورود اصلی پروژه)
ENTRYPOINT ["python", "main.py"]

# پارامترهای پیش‌فرض (قابل بازنویسی)
CMD ["--crawl-type", "incremental", "--max-threads", "4"]