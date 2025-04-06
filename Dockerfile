# استفاده از تصویر پایه پایتون 3.11-slim
FROM python:3.11-slim

# تنظیم دایرکتوری کاری در کانتینر
WORKDIR /app

# کپی فایل requirements.txt برای نصب وابستگی‌ها
COPY requirements.txt .

# به‌روزرسانی pip و نصب وابستگی‌ها
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# کپی کل کدهای پروژه به کانتینر
COPY . .

# اجرای برنامه (نقطه ورود اصلی پروژه)
CMD ["python", "main.py"]
