"""
پکیج config:
این پکیج شامل تنظیمات و پیکربندی‌های مورد نیاز پروژه است.
با وارد کردن این پکیج، می‌توانید به راحتی به تمام توابع و متغیرهای تنظیمات دسترسی داشته باشید.
"""

from .settings import (
    BASE_DIR,
    CONFIG_DIR,
    LOGS_DIR,
    DATA_DIR,
    MODELS_DIR,
    DB_CONFIG,
    LOG_CONFIG,
    CRAWLER_CONFIG,
    ML_CONFIG,
    load_defaults,
    load_domain_config,
    get_user_agent_list,
    get_connection_string,
    DEFAULT_CONFIG,
)

__all__ = [
    "BASE_DIR",
    "CONFIG_DIR",
    "LOGS_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "DB_CONFIG",
    "LOG_CONFIG",
    "CRAWLER_CONFIG",
    "ML_CONFIG",
    "load_defaults",
    "load_domain_config",
    "get_user_agent_list",
    "get_connection_string",
    "DEFAULT_CONFIG",
]
