"""
ماژول تنظیمات برای خزشگر هوشمند داده‌های حقوقی

این ماژول حاوی تنظیمات و پیکربندی‌های پیش‌فرض برای کل پروژه است و مسئول
بارگذاری و مدیریت تنظیمات از فایل‌های پیکربندی و متغیرهای محیطی است.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

# مسیرهای پایه پروژه
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = os.path.join(BASE_DIR, 'config')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'ml', 'models')

# ایجاد دایرکتوری‌های مورد نیاز
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# تنظیمات پایگاه داده
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'name': os.getenv('DB_NAME', 'legal_crawler'),
    'charset': 'utf8mb4',
    'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 20)),
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', 3600)),
}

# تنظیمات لاگ‌گیری
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_size': int(os.getenv('LOG_FILE_SIZE', 10 * 1024 * 1024)),  # 10 مگابایت
    'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
}

# تنظیمات خزشگر
CRAWLER_CONFIG = {
    'max_threads': int(os.getenv('MAX_THREADS', 4)),
    'max_depth': int(os.getenv('MAX_DEPTH', 5)),
    'politeness_delay': float(os.getenv('CRAWL_DELAY', 1.0)),
    'respect_robots': os.getenv('RESPECT_ROBOTS', 'True').lower() in ('true', '1', 't'),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'timeout': int(os.getenv('REQUEST_TIMEOUT', 30)),
    'checkpoint_interval': int(os.getenv('CHECKPOINT_INTERVAL', 300)),  # 5 دقیقه
    'use_selenium': os.getenv('USE_SELENIUM', 'False').lower() in ('true', '1', 't'),
}

# تنظیمات هوش مصنوعی و یادگیری ماشین
ML_CONFIG = {
    'content_model_path': os.getenv('CONTENT_MODEL_PATH', os.path.join(MODELS_DIR, 'content_type_classifier.pkl')),
    'domain_model_path': os.getenv('DOMAIN_MODEL_PATH', os.path.join(MODELS_DIR, 'domain_classifier.pkl')),
    'features_path': os.getenv('FEATURES_PATH', os.path.join(MODELS_DIR, 'features')),
    'min_confidence': float(os.getenv('MIN_CONFIDENCE', 0.6)),
    'model_update_interval': int(os.getenv('MODEL_UPDATE_INTERVAL', 24 * 60 * 60)),  # 24 ساعت
    'auto_train': os.getenv('AUTO_TRAIN', 'False').lower() in ('true', '1', 't'),
}


def load_defaults():
    """
    بارگذاری تنظیمات پیش‌فرض از فایل JSON

    Returns:
        dict: دیکشنری تنظیمات پیش‌فرض
    """
    default_path = os.path.join(CONFIG_DIR, 'defaults.json')

    try:
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"فایل تنظیمات پیش‌فرض یافت نشد: {default_path}")
            return {}
    except Exception as e:
        print(f"خطا در بارگذاری تنظیمات پیش‌فرض: {str(e)}")
        return {}


def load_domain_config(domain):
    """
    بارگذاری تنظیمات مخصوص دامنه از فایل JSON

    Args:
        domain: نام دامنه

    Returns:
        dict: دیکشنری تنظیمات دامنه
    """
    domain_config_path = os.path.join(CONFIG_DIR, f"{domain}_config.json")

    try:
        if os.path.exists(domain_config_path):
            with open(domain_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        print(f"خطا در بارگذاری تنظیمات دامنه {domain}: {str(e)}")
        return {}


def get_user_agent_list():
    """
    دریافت لیست User-Agent ها از فایل پیکربندی یا مقادیر پیش‌فرض

    Returns:
        list: لیست User-Agent ها
    """
    user_agents_path = os.path.join(CONFIG_DIR, 'user_agents.json')

    default_user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    ]

    try:
        if os.path.exists(user_agents_path):
            with open(user_agents_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return default_user_agents
    except Exception as e:
        print(f"خطا در بارگذاری لیست User-Agent ها: {str(e)}")
        return default_user_agents


def get_connection_string():
    """
    ایجاد رشته اتصال پایگاه داده بر اساس تنظیمات

    Returns:
        str: رشته اتصال SQLAlchemy
    """
    return (f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['name']}?"
            f"charset={DB_CONFIG['charset']}")


# بارگذاری تنظیمات پیش‌فرض
DEFAULT_CONFIG = load_defaults()

# نمایش پیغام راه‌اندازی
print(f"تنظیمات خزشگر هوشمند داده‌های حقوقی بارگذاری شد")