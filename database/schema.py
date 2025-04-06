"""
ماژول تعریف ساختار جداول پایگاه داده برای خزشگر هوشمند داده‌های حقوقی

این ماژول شامل تعاریف DDL برای ایجاد جداول پایگاه داده و
همچنین توابع کمکی برای اجرای این دستورات است.
"""

import os
from sqlalchemy import text
from utils.logger import get_logger
from database.connection import DatabaseConnection

# تنظیم لاگر
logger = get_logger(__name__)

# تعاریف DDL برای جداول پایگاه داده

CREATE_DOMAINS_TABLE = """
CREATE TABLE IF NOT EXISTS domains (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    keywords JSON,
    auto_detected BOOLEAN DEFAULT FALSE,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

CREATE_CONTENT_ITEMS_TABLE = """
CREATE TABLE IF NOT EXISTS content_items (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255),
    content TEXT NOT NULL,
    content_type ENUM('question', 'article', 'profile', 'other') NOT NULL,
    meta_data JSON,
    url VARCHAR(255) NOT NULL UNIQUE,
    view_count INT DEFAULT 0,
    similarity_hash VARCHAR(64),
    status ENUM('active', 'archived', 'processing', 'error') DEFAULT 'active',
    created_at TIMESTAMP NULL,
    updated_at TIMESTAMP NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

CREATE_EXPERTS_TABLE = """
CREATE TABLE IF NOT EXISTS experts (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    bio TEXT,
    expertise JSON,
    rating FLOAT DEFAULT 0,
    answers_count INT DEFAULT 0,
    profile_url VARCHAR(255) UNIQUE,
    avatar_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

CREATE_ANSWERS_TABLE = """
CREATE TABLE IF NOT EXISTS answers (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    content_id BIGINT NOT NULL,
    expert_id BIGINT NOT NULL,
    text TEXT NOT NULL,
    rating FLOAT DEFAULT 0,
    status ENUM('active', 'archived', 'hidden') DEFAULT 'active',
    created_at TIMESTAMP NULL,
    updated_at TIMESTAMP NULL,
    FOREIGN KEY (content_id) REFERENCES content_items(id) ON DELETE CASCADE,
    FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

CREATE_DOMAIN_CONTENT_TABLE = """
CREATE TABLE IF NOT EXISTS domain_content (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    domain_id VARCHAR(50) NOT NULL,
    content_id BIGINT NOT NULL,
    relevance_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE,
    FOREIGN KEY (content_id) REFERENCES content_items(id) ON DELETE CASCADE,
    UNIQUE KEY unique_domain_content (domain_id, content_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

CREATE_EXPERT_DOMAIN_TABLE = """
CREATE TABLE IF NOT EXISTS expert_domain (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    expert_id BIGINT NOT NULL,
    domain_id VARCHAR(50) NOT NULL,
    confidence_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
    FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE CASCADE,
    UNIQUE KEY unique_expert_domain (expert_id, domain_id)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_persian_ci;
"""

# لیست تمام دستورات ایجاد جداول
CREATE_TABLES_STATEMENTS = [
    CREATE_DOMAINS_TABLE,
    CREATE_CONTENT_ITEMS_TABLE,
    CREATE_EXPERTS_TABLE,
    CREATE_ANSWERS_TABLE,
    CREATE_DOMAIN_CONTENT_TABLE,
    CREATE_EXPERT_DOMAIN_TABLE
]


def create_tables():
    """ایجاد تمام جداول پایگاه داده"""
    db_conn = DatabaseConnection()
    engine = db_conn.get_engine()

    try:
        with engine.connect() as connection:
            # اجرای هر یک از دستورات ایجاد جدول
            for create_statement in CREATE_TABLES_STATEMENTS:
                connection.execute(text(create_statement))

        logger.info("تمام جداول با موفقیت ایجاد شدند")
        return True

    except Exception as e:
        logger.error(f"خطا در ایجاد جداول: {str(e)}")
        return False


def drop_tables():
    """حذف تمام جداول پایگاه داده (استفاده با احتیاط)"""
    db_conn = DatabaseConnection()
    engine = db_conn.get_engine()

    # دستورات حذف جداول به ترتیب مناسب (جداول با کلید خارجی ابتدا حذف می‌شوند)
    drop_statements = [
        "DROP TABLE IF EXISTS expert_domain;",
        "DROP TABLE IF EXISTS domain_content;",
        "DROP TABLE IF EXISTS answers;",
        "DROP TABLE IF EXISTS experts;",
        "DROP TABLE IF EXISTS content_items;",
        "DROP TABLE IF EXISTS domains;"
    ]

    try:
        with engine.connect() as connection:
            for drop_statement in drop_statements:
                connection.execute(text(drop_statement))

        logger.info("تمام جداول با موفقیت حذف شدند")
        return True

    except Exception as e:
        logger.error(f"خطا در حذف جداول: {str(e)}")
        return False


def recreate_tables():
    """حذف و ایجاد مجدد تمام جداول (استفاده با احتیاط)"""
    drop_tables()
    create_tables()
    logger.info("تمام جداول با موفقیت بازسازی شدند")