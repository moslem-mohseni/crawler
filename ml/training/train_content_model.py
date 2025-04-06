#!/usr/bin/env python
"""
اسکریپت آموزش مدل تشخیص نوع محتوا برای خزشگر داده‌های حقوقی

این اسکریپت داده‌های آموزشی (JSON) را بارگذاری کرده، با استفاده از
ContentTypeFeatures ویژگی‌های متنی استخراج می‌کند، سپس با بهره‌گیری از
مدل‌های طبقه‌بندی (SVM، Logistic Regression یا RandomForest) مدل تشخیص نوع محتوا را آموزش داده،
بهینه‌سازی و ارزیابی می‌کند. در نهایت، مدل بهینه به همراه استخراج‌کننده ویژگی و
تبدیل‌کننده برچسب‌ها ذخیره شده و در صورت ارائه، امکان پیش‌بینی نوع محتوا نیز فراهم می‌شود.
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
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# افزودن مسیر پروژه به سیستم
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger
from utils.text import normalize_persian_text, tokenize_persian_text
from ml.features import ContentTypeFeatures, CONTENT_TYPE_KEYWORDS
from ml.train_model_utils import save_model_to_file, load_model_from_file, save_metrics_to_json  # توابع مشترک

logger = get_logger(__name__)

# مسیر ذخیره‌سازی مدل‌ها (پوشه ml/models)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    بارگذاری داده‌های آموزشی از فایل JSON

    Args:
        data_path: مسیر فایل داده‌های آموزشی
        test_size: نسبت داده‌های تست
        random_state: مقدار اولیه تصادفی‌سازی

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    logger.info(f"بارگذاری داده‌ها از {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = []
        labels = []
        for item in data:
            texts.append(item.get('text', ''))
            # دریافت نوع محتوا؛ در صورت نامعتبر، مقدار "other" در نظر گرفته می‌شود
            content_type = item.get('content_type', 'other')
            if content_type not in CONTENT_TYPE_KEYWORDS and content_type != 'other':
                content_type = 'other'
            labels.append(content_type)
        # تبدیل برچسب‌ها به اعداد با استفاده از LabelEncoder (چون نوع محتوا تک برچسبی است)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"تعداد کل نمونه‌ها: {len(texts)}")
        logger.info(f"انواع محتوا: {label_encoder.classes_}")
        logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
        logger.info(f"تعداد نمونه‌های تست: {len(X_test)}")
        return X_train, X_test, y_train, y_test, label_encoder
    except Exception as e:
        logger.error(f"خطا در بارگذاری داده‌ها: {str(e)}")
        raise


def create_synthetic_data(num_samples: int = 1000, random_state: int = 42) -> Tuple:
    """
    ایجاد داده‌های مصنوعی برای آموزش مدل تشخیص نوع محتوا

    Args:
        num_samples: تعداد نمونه‌ها
        random_state: مقدار اولیه تصادفی‌سازی

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, label_encoder)
    """
    logger.info(f"ایجاد {num_samples} نمونه داده مصنوعی")
    np.random.seed(random_state)
    # تعریف انواع محتوا از کلیدواژه‌های موجود و افزودن نوع "other"
    content_types = list(CONTENT_TYPE_KEYWORDS.keys())
    content_types.append("other")
    texts, labels = [], []
    for _ in range(num_samples):
        # انتخاب تصادفی یک نوع محتوا
        content_type = np.random.choice(content_types)
        if content_type == "other":
            text = "این یک متن عمومی است که هیچ ویژگی خاصی ندارد."
        else:
            keywords = CONTENT_TYPE_KEYWORDS[content_type]
            num_keywords = min(np.random.randint(3, 8), len(keywords))
            selected_keywords = np.random.choice(keywords, size=num_keywords, replace=False)
            sentences = []
            for keyword in selected_keywords:
                if content_type == "question":
                    sentences.append(f"{keyword} مربوط به موضوع حقوقی چیست؟")
                elif content_type == "answer":
                    sentences.append(f"{keyword} در این مورد حقوقی قابل استناد است.")
                elif content_type == "article":
                    sentences.append(f"در این مقاله به بررسی {keyword} می‌پردازیم.")
                elif content_type == "profile":
                    sentences.append(f"{keyword} از ویژگی‌های تخصصی این وکیل است.")
            text = " ".join(sentences)
        texts.append(text)
        labels.append(content_type)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info(f"تعداد کل نمونه‌ها: {len(texts)}")
    logger.info(f"انواع محتوا: {label_encoder.classes_}")
    logger.info(f"تعداد نمونه‌های آموزش: {len(X_train)}")
    logger.info(f"تعداد نمونه‌های تست: {len(X_test)}")
    return X_train, X_test, y_train, y_test, label_encoder


def extract_features(texts: list, feature_extractor=None) -> Tuple:
    """
    استخراج ویژگی‌های متنی با استفاده از ContentTypeFeatures

    Args:
        texts: لیست متون
        feature_extractor: استخراج‌کننده ویژگی (اختیاری)

    Returns:
        Tuple: (features, feature_extractor)
    """
    if feature_extractor is None:
        logger.info("ایجاد استخراج‌کننده ویژگی جدید برای نوع محتوا")
        feature_extractor = ContentTypeFeatures()
        logger.info("استخراج ویژگی‌ها از متون")
        features = feature_extractor.fit_transform(texts)
    else:
        logger.info("استخراج ویژگی‌ها با استخراج‌کننده موجود")
        features = feature_extractor.transform(texts)
    return features, feature_extractor


def train_models(X_train, y_train, model_type: str = 'svm') -> Dict:
    """
    آموزش مدل‌های طبقه‌بندی نوع محتوا

    Args:
        X_train: ویژگی‌های آموزش
        y_train: برچسب‌های آموزش
        model_type: نوع مدل ('svm', 'logistic', 'forest')

    Returns:
        Dict: شامل مدل بهینه، پارامترها، نتایج ارزیابی متقاطع و نوع مدل
    """
    logger.info(f"آموزش مدل {model_type} برای تشخیص نوع محتوا")
    if model_type == 'svm':
        base_model = LinearSVC(
            C=1.0,
            class_weight='balanced',
            dual=False,
            max_iter=10000,
            random_state=42
        )
        param_grid = {'C': [0.1, 1.0, 10.0]}
    elif model_type == 'logistic':
        base_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=10000,
            random_state=42,
            solver='liblinear'
        )
        param_grid = {'C': [0.1, 1.0, 10.0],
                      'solver': ['liblinear', 'saga']}
    elif model_type == 'forest':
        base_model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        param_grid = {'n_estimators': [50, 100, 200],
                      'max_depth': [None, 10, 20]}
    else:
        raise ValueError(f"نوع مدل {model_type} نامعتبر است")
    # ایجاد مدل طبقه‌بندی تک‌برچسبی
    from sklearn.multiclass import OneVsRestClassifier
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


def evaluate_model(model, X_test, y_test, label_encoder) -> Dict:
    """
    ارزیابی مدل تشخیص نوع محتوا روی داده‌های تست

    Args:
        model: مدل آموزش دیده
        X_test: ویژگی‌های تست
        y_test: برچسب‌های تست
        label_encoder: رمزگذار برچسب‌ها

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
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    for content_type, metrics in report.items():
        if content_type in ['micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            continue
        logger.info(f"نوع محتوا {content_type}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    # محاسبه ماتریس اغتشاش
    conf_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_idx, pred_idx in zip(y_test, y_pred):
        conf_matrix[true_idx, pred_idx] += 1
    logger.info("ماتریس اغتشاش:")
    confusion_str = "واقعی \\ پیش‌بینی\t" + "\t".join(class_names) + "\n"
    for i, class_name in enumerate(class_names):
        row = [str(conf_matrix[i, j]) for j in range(len(class_names))]
        confusion_str += f"{class_name}\t\t" + "\t".join(row) + "\n"
    logger.info("\n" + confusion_str)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': conf_matrix.tolist()
    }


def save_model(model_data: Dict, feature_extractor, label_encoder, model_path: str = None) -> str:
    """
    ذخیره مدل تشخیص نوع محتوا به همراه استخراج‌کننده ویژگی و رمزگذار برچسب‌ها

    Args:
        model_data: دیکشنری داده‌های مدل آموزش دیده
        feature_extractor: استخراج‌کننده ویژگی (ContentTypeFeatures)
        label_encoder: رمزگذار برچسب‌ها (LabelEncoder)
        model_path: مسیر ذخیره مدل (اختیاری)

    Returns:
        str: مسیر فایل مدل ذخیره شده
    """
    if model_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"content_type_classifier_{timestamp}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    feature_extractor_path = os.path.join(MODEL_DIR, "content_type_features.pkl")
    feature_extractor.save(feature_extractor_path)
    model_package = {
        'model': model_data['model'],
        'params': model_data['params'],
        'cv_results': model_data['cv_results'],
        'type': model_data['type'],
        'label_encoder': label_encoder,
        'content_types': list(label_encoder.classes_),
        'feature_extractor_path': feature_extractor_path,
        'timestamp': datetime.now().isoformat()
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    logger.info(f"مدل تشخیص نوع محتوا با موفقیت در {model_path} ذخیره شد")
    return model_path


def load_model(model_path: str) -> Tuple:
    """
    بارگذاری مدل ذخیره شده

    Args:
        model_path: مسیر فایل مدل

    Returns:
        Tuple: (model, feature_extractor, label_encoder)
    """
    logger.info(f"بارگذاری مدل از {model_path}")
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    model = model_package['model']
    label_encoder = model_package['label_encoder']
    feature_extractor_path = model_package.get('feature_extractor_path')
    if feature_extractor_path and os.path.exists(feature_extractor_path):
        feature_extractor = ContentTypeFeatures.load(feature_extractor_path)
    else:
        logger.warning("استخراج‌کننده ویژگی یافت نشد، ایجاد یک نمونه جدید")
        feature_extractor = ContentTypeFeatures()
    return model, feature_extractor, label_encoder


def predict_content_type(text: str, model, feature_extractor, label_encoder) -> Dict:
    """
    پیش‌بینی نوع محتوا برای یک متن ورودی

    Args:
        text: متن ورودی
        model: مدل آموزش دیده
        feature_extractor: استخراج‌کننده ویژگی
        label_encoder: رمزگذار برچسب‌ها

    Returns:
        Dict: شامل نوع محتوا و احتمالات مربوطه
    """
    features = feature_extractor.transform([text])
    pred_idx = model.predict(features)[0]
    pred_type = label_encoder.inverse_transform([pred_idx])[0]
    probabilities = {}
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        for i, content_type in enumerate(label_encoder.classes_):
            probabilities[content_type] = float(proba[i])
    else:
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(features)[0]
            if len(label_encoder.classes_) == 2:
                probabilities[label_encoder.classes_[0]] = 1.0 / (1.0 + np.exp(-decision_scores))
                probabilities[label_encoder.classes_[1]] = 1.0 / (1.0 + np.exp(decision_scores))
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores))
                softmax = exp_scores / exp_scores.sum()
                for i, content_type in enumerate(label_encoder.classes_):
                    probabilities[content_type] = float(softmax[i])
    return {
        'content_type': pred_type,
        'probabilities': probabilities
    }


def parse_arguments():
    """پردازش آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(description='آموزش مدل تشخیص نوع محتوا برای خزشگر داده‌های حقوقی')
    parser.add_argument('--data', type=str, help='مسیر فایل داده‌های آموزشی (JSON)')
    parser.add_argument('--synthetic', type=int, default=0, help='تعداد نمونه‌های مصنوعی برای تولید (پیش‌فرض: 0)')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'logistic', 'forest'],
                        help='نوع مدل (پیش‌فرض: svm)')
    parser.add_argument('--output', type=str, help='مسیر خروجی برای ذخیره مدل')
    parser.add_argument('--test-size', type=float, default=0.2, help='نسبت داده‌های تست (پیش‌فرض: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='مقدار اولیه تصادفی‌سازی (پیش‌فرض: 42)')
    parser.add_argument('--predict', type=str, help='متنی برای پیش‌بینی (اختیاری)')
    parser.add_argument('--load', type=str, help='مسیر مدل ذخیره شده برای بارگذاری')
    return parser.parse_args()


def main():
    """تابع اصلی برنامه آموزش مدل تشخیص نوع محتوا"""
    args = parse_arguments()
    try:
        if args.load:
            model, feature_extractor, label_encoder = load_model(args.load)
            if args.predict:
                result = predict_content_type(args.predict, model, feature_extractor, label_encoder)
                print(f"نوع محتوا: {result['content_type']}")
                print("احتمالات:")
                for content_type, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {content_type}: {prob:.4f}")
                sys.exit(0)
        if args.data:
            X_train, X_test, y_train, y_test, label_encoder = load_data(
                args.data,
                test_size=args.test_size,
                random_state=args.random_state
            )
        elif args.synthetic > 0:
            X_train, X_test, y_train, y_test, label_encoder = create_synthetic_data(
                num_samples=args.synthetic,
                random_state=args.random_state
            )
        else:
            logger.error("هیچ منبع داده‌ای مشخص نشده است")
            sys.exit(1)
        X_train_features, feature_extractor = extract_features(X_train)
        X_test_features, _ = extract_features(X_test, feature_extractor)
        model_data = train_models(X_train_features, y_train, model_type=args.model)
        evaluation = evaluate_model(model_data['model'], X_test_features, y_test, label_encoder)
        save_model(model_data, feature_extractor, label_encoder, args.output)
        if args.predict:
            result = predict_content_type(args.predict, model_data['model'], feature_extractor, label_encoder)
            print(f"نوع محتوا: {result['content_type']}")
            print("احتمالات:")
            for content_type, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {content_type}: {prob:.4f}")
        logger.info("آموزش مدل تشخیص نوع محتوا با موفقیت به پایان رسید")
    except Exception as e:
        logger.error(f"خطا در آموزش مدل: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
