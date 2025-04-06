#!/usr/bin/env python
"""
اسکریپت آموزش مدل طبقه‌بندی حوزه‌های تخصصی حقوقی

این اسکریپت داده‌های آموزشی را بارگذاری کرده، ویژگی‌های متنی استخراج می‌کند،
سپس مدل‌های طبقه‌بندی چندبرچسبی (با استفاده از OneVsRestClassifier) را آموزش داده،
بهینه‌سازی می‌کند و ارزیابی می‌نماید. در نهایت، مدل بهینه همراه با استخراج‌کننده ویژگی و
تبدیل‌کننده برچسب‌ها ذخیره می‌شود.
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
from datetime import datetime
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

# افزودن مسیر پروژه به سیستم
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from utils.text import normalize_persian_text, tokenize_persian_text
from ml.features import DomainFeatures, LEGAL_DOMAINS_KEYWORDS

logger = get_logger(__name__)

# مسیر ذخیره‌سازی مدل‌ها
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    بارگذاری داده‌های آموزشی از فایل

    Args:
        data_path: مسیر فایل داده‌ها (JSON)
        test_size: نسبت داده‌های تست
        random_state: مقدار اولیه تصادفی‌سازی

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, mlb)
    """
    logger.info(f"بارگذاری داده‌ها از {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []
        labels = []

        for item in data:
            texts.append(item.get('text', ''))
            item_labels = item.get('domains', [])
            # اطمینان از صحت برچسب‌ها
            valid_labels = [label for label in item_labels if label in LEGAL_DOMAINS_KEYWORDS]
            labels.append(valid_labels)

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"تعداد کل نمونه‌ها: {len(texts)}")
        logger.info(f"تعداد حوزه‌های تخصصی: {len(mlb.classes_)} - {mlb.classes_}")
        logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
        logger.info(f"تعداد نمونه‌های تست: {len(X_test)}")
        return X_train, X_test, y_train, y_test, mlb

    except Exception as e:
        logger.error(f"خطا در بارگذاری داده‌ها: {str(e)}")
        raise


def create_synthetic_data(num_samples: int = 1000, random_state: int = 42) -> Tuple:
    """
    ایجاد داده‌های مصنوعی برای آموزش مدل

    Args:
        num_samples: تعداد نمونه‌ها
        random_state: مقدار اولیه تصادفی‌سازی

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, mlb)
    """
    logger.info(f"ایجاد {num_samples} نمونه داده مصنوعی")
    np.random.seed(random_state)
    domains = list(LEGAL_DOMAINS_KEYWORDS.keys())
    texts, labels = [], []

    for _ in range(num_samples):
        num_domains = np.random.randint(1, 4)
        sample_domains = np.random.choice(domains, size=num_domains, replace=False)
        text_parts = []
        for domain in sample_domains:
            keywords = LEGAL_DOMAINS_KEYWORDS[domain]
            num_keywords = min(np.random.randint(3, 10), len(keywords))
            selected_keywords = np.random.choice(keywords, size=num_keywords, replace=False)
            text_parts.append(' '.join(selected_keywords))
        texts.append(' '.join(text_parts))
        labels.append(list(sample_domains))

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=random_state
    )
    logger.info(f"تعداد کل نمونه‌ها: {len(texts)}")
    logger.info(f"حوزه‌های تخصصی: {mlb.classes_}")
    logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
    logger.info(f"تعداد نمونه‌های تست: {len(X_test)}")
    return X_train, X_test, y_train, y_test, mlb


def extract_features(texts: list, feature_extractor=None) -> Tuple:
    """
    استخراج ویژگی‌های متنی با استفاده از DomainFeatures

    Args:
        texts: لیست متون
        feature_extractor: استخراج‌کننده ویژگی (اختیاری)

    Returns:
        Tuple: (features, feature_extractor)
    """
    if feature_extractor is None:
        logger.info("ایجاد استخراج‌کننده ویژگی جدید برای حوزه‌ها")
        feature_extractor = DomainFeatures()
        features = feature_extractor.fit_transform(texts)
    else:
        logger.info("استخراج ویژگی‌ها با استخراج‌کننده موجود")
        features = feature_extractor.transform(texts)
    return features, feature_extractor


def train_models(X_train, y_train, model_type: str = 'svm') -> Dict:
    """
    آموزش مدل‌های طبقه‌بندی حوزه‌های تخصصی

    Args:
        X_train: ویژگی‌های آموزش
        y_train: برچسب‌های آموزش (باینری چندبرچسبی)
        model_type: نوع مدل ('svm', 'logistic', 'forest')

    Returns:
        Dict: شامل مدل بهینه، پارامترها، نتایج متقاطع و نوع مدل
    """
    logger.info(f"آموزش مدل {model_type} برای طبقه‌بندی حوزه‌های تخصصی")
    if model_type == 'svm':
        base_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            dual=False,
            max_iter=10000,
            random_state=42
        )
        param_grid = {'estimator__C': [0.1, 1.0, 10.0]}
    elif model_type == 'logistic':
        base_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=10000,
            random_state=42,
            solver='liblinear'
        )
        param_grid = {'estimator__C': [0.1, 1.0, 10.0],
                      'estimator__solver': ['liblinear', 'saga']}
    elif model_type == 'forest':
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        param_grid = {'estimator__n_estimators': [50, 100, 200],
                      'estimator__max_depth': [None, 10, 20]}
    else:
        raise ValueError(f"نوع مدل {model_type} نامعتبر است")

    model = OneVsRestClassifier(base_model)
    logger.info("بهینه‌سازی پارامترهای مدل با GridSearchCV")
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    logger.info(f"بهترین پارامترها: {grid_search.best_params_}")
    logger.info(f"بهترین امتیاز GridSearch: {grid_search.best_score_:.4f}")

    logger.info("ارزیابی متقاطع مدل بهینه")
    cv_results = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
        return_train_score=True
    )
    mean_results = {key: np.mean(value) for key, value in cv_results.items()}
    logger.info(f"میانگین دقت آموزش: {mean_results['train_accuracy']:.4f}")
    logger.info(f"میانگین دقت اعتبارسنجی: {mean_results['test_accuracy']:.4f}")
    logger.info(f"میانگین F1 آموزش: {mean_results['train_f1_macro']:.4f}")
    logger.info(f"میانگین F1 اعتبارسنجی: {mean_results['test_f1_macro']:.4f}")

    logger.info("آموزش مجدد مدل بهینه روی کل داده‌های آموزش")
    best_model.fit(X_train, y_train)

    return {
        'model': best_model,
        'params': grid_search.best_params_,
        'cv_results': mean_results,
        'type': model_type
    }


def evaluate_model(model, X_test, y_test, mlb) -> Dict:
    """
    ارزیابی مدل طبقه‌بندی حوزه‌های تخصصی روی داده‌های تست

    Args:
        model: مدل آموزش دیده
        X_test: ویژگی‌های تست
        y_test: برچسب‌های تست
        mlb: تبدیل‌کننده برچسب‌های چندبرچسبی

    Returns:
        Dict: شامل دقت، precision، recall، f1 و گزارش طبقه‌بندی
    """
    logger.info("ارزیابی مدل روی داده‌های تست")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    logger.info(f"دقت: {accuracy:.4f}")
    logger.info(f"دقت (Precision): {precision:.4f}")
    logger.info(f"فراخوانی (Recall): {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
    for domain, metrics in report.items():
        if domain in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            continue
        logger.info(f"حوزه {domain}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }


def save_model(model_data: Dict, feature_extractor, mlb, model_path: str = None) -> str:
    """
    ذخیره مدل طبقه‌بندی حوزه‌های تخصصی به همراه استخراج‌کننده ویژگی و تبدیل‌کننده برچسب‌ها

    Args:
        model_data: دیکشنری مدل آموزش دیده
        feature_extractor: استخراج‌کننده ویژگی (DomainFeatures)
        mlb: تبدیل‌کننده برچسب‌های چندبرچسبی
        model_path: مسیر ذخیره مدل (اختیاری)

    Returns:
        str: مسیر فایل مدل ذخیره شده
    """
    if model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"domain_classifier_{timestamp}.pkl")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    feature_extractor_path = os.path.join(MODEL_DIR, "domain_features.pkl")
    feature_extractor.save(feature_extractor_path)

    model_package = {
        'model': model_data['model'],
        'params': model_data['params'],
        'cv_results': model_data['cv_results'],
        'type': model_data['type'],
        'mlb': mlb,
        'domain_names': list(mlb.classes_),
        'feature_extractor_path': feature_extractor_path,
        'timestamp': datetime.now().isoformat()
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)

    logger.info(f"مدل طبقه‌بندی حوزه با موفقیت در {model_path} ذخیره شد")
    return model_path


def parse_arguments():
    """پردازش آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(description='آموزش مدل طبقه‌بندی حوزه‌های تخصصی حقوقی')
    parser.add_argument('--data', type=str, help='مسیر فایل داده‌های آموزشی (JSON)')
    parser.add_argument('--synthetic', type=int, default=0, help='تعداد نمونه‌های مصنوعی (پیش‌فرض: 0)')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'logistic', 'forest'],
                        help='نوع مدل (پیش‌فرض: svm)')
    parser.add_argument('--output', type=str, help='مسیر خروجی برای ذخیره مدل')
    parser.add_argument('--test-size', type=float, default=0.2, help='نسبت داده‌های تست (پیش‌فرض: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='مقدار اولیه تصادفی‌سازی (پیش‌فرض: 42)')
    return parser.parse_args()


def main():
    """تابع اصلی برنامه"""
    args = parse_arguments()

    try:
        if args.data:
            X_train, X_test, y_train, y_test, mlb = load_data(
                args.data,
                test_size=args.test_size,
                random_state=args.random_state
            )
        elif args.synthetic > 0:
            X_train, X_test, y_train, y_test, mlb = create_synthetic_data(
                num_samples=args.synthetic,
                random_state=args.random_state
            )
        else:
            logger.error("هیچ منبع داده‌ای مشخص نشده است")
            sys.exit(1)

        features_train, feature_extractor = extract_features(X_train)
        features_test, _ = extract_features(X_test, feature_extractor)

        model_data = train_models(features_train, y_train, model_type=args.model)
        evaluation = evaluate_model(model_data['model'], features_test, y_test, mlb)
        save_model(model_data, feature_extractor, mlb, args.output)

        logger.info("آموزش مدل طبقه‌بندی حوزه‌های تخصصی با موفقیت به پایان رسید")
    except Exception as e:
        logger.error(f"خطا در آموزش مدل: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
