"""
توابع کمکی و مشترک برای اسکریپت‌های آموزش مدل در خزشگر داده‌های حقوقی

این ماژول شامل توابع مشترکی است که در اسکریپت‌های مختلف آموزش مدل استفاده می‌شوند
تا از تکرار کد جلوگیری شود و یکپارچگی بیشتری بین مدل‌ها ایجاد گردد.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any

from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# افزودن مسیر پروژه به سیستم
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.logger import get_logger

# تنظیم لاگر
logger = get_logger(__name__)

# مسیر پیش‌فرض ذخیره‌سازی مدل‌ها
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model_to_file(model_data: Dict, file_path: str) -> bool:
    """
    ذخیره مدل و داده‌های مرتبط در فایل

    Args:
        model_data: دیکشنری حاوی مدل و داده‌های مرتبط
        file_path: مسیر فایل برای ذخیره‌سازی

    Returns:
        bool: آیا ذخیره‌سازی موفق بود؟
    """
    try:
        # ایجاد دایرکتوری در صورت نیاز
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # افزودن زمان ذخیره‌سازی
        model_data['saved_at'] = datetime.now().isoformat()

        # ذخیره در فایل
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"مدل با موفقیت در {file_path} ذخیره شد")
        return True

    except Exception as e:
        logger.error(f"خطا در ذخیره مدل: {str(e)}")
        return False


def load_model_from_file(file_path: str) -> Optional[Dict]:
    """
    بارگذاری مدل از فایل

    Args:
        file_path: مسیر فایل مدل

    Returns:
        Dict: دیکشنری حاوی مدل و داده‌های مرتبط یا None در صورت خطا
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"فایل مدل {file_path} یافت نشد")
            return None

        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        logger.info(f"مدل با موفقیت از {file_path} بارگذاری شد")
        return model_data

    except Exception as e:
        logger.error(f"خطا در بارگذاری مدل: {str(e)}")
        return None


def save_metrics_to_json(metrics: Dict, file_path: str) -> bool:
    """
    ذخیره معیارهای ارزیابی مدل در فایل JSON

    Args:
        metrics: دیکشنری معیارهای ارزیابی
        file_path: مسیر فایل برای ذخیره‌سازی

    Returns:
        bool: آیا ذخیره‌سازی موفق بود؟
    """
    try:
        # تبدیل داده‌های نامپای به لیست برای سازگاری با JSON
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                # بازگشتی برای دیکشنری‌های تو در تو
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_metrics[key][k] = v.tolist()
                    elif isinstance(v, np.floating):
                        serializable_metrics[key][k] = float(v)
                    elif isinstance(v, np.integer):
                        serializable_metrics[key][k] = int(v)
                    else:
                        serializable_metrics[key][k] = v
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, np.floating):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_metrics[key] = int(value)
            else:
                serializable_metrics[key] = value

        # افزودن زمان ذخیره‌سازی
        serializable_metrics['saved_at'] = datetime.now().isoformat()

        # ایجاد دایرکتوری در صورت نیاز
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # ذخیره در فایل JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"معیارهای ارزیابی مدل با موفقیت در {file_path} ذخیره شد")
        return True

    except Exception as e:
        logger.error(f"خطا در ذخیره معیارهای ارزیابی: {str(e)}")
        return False


def plot_confusion_matrix(y_true, y_pred, class_names, title=None, output_path=None, figsize=(10, 8), normalize=False,
                          cmap=plt.cm.Blues):
    """
    رسم ماتریس اغتشاش

    Args:
        y_true: برچسب‌های واقعی
        y_pred: برچسب‌های پیش‌بینی شده
        class_names: نام‌های کلاس‌ها
        title: عنوان نمودار (اختیاری)
        output_path: مسیر ذخیره نمودار (اختیاری)
        figsize: اندازه نمودار
        normalize: آیا مقادیر نرمال‌سازی شوند؟
        cmap: نقشه رنگ

    Returns:
        matplotlib.figure.Figure: شیء نمودار
    """
    # محاسبه ماتریس اغتشاش
    cm = confusion_matrix(y_true, y_pred)

    # نرمال‌سازی
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # ایجاد نمودار
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # نمایش برچسب‌های محور
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='برچسب واقعی',
           xlabel='برچسب پیش‌بینی شده')

    # چرخش برچسب‌های محور X برای خوانایی بهتر
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # نمایش مقادیر در ماتریس
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # عنوان نمودار
    if title:
        ax.set_title(title)

    fig.tight_layout()

    # ذخیره نمودار در صورت نیاز
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"ماتریس اغتشاش در {output_path} ذخیره شد")

    return fig


def plot_roc_curve(model, X_test, y_test, class_names, output_path=None, figsize=(10, 8)):
    """
    رسم منحنی ROC برای مدل‌های چندکلاسه

    Args:
        model: مدل آموزش دیده
        X_test: ویژگی‌های تست
        y_test: برچسب‌های تست
        class_names: نام‌های کلاس‌ها
        output_path: مسیر ذخیره نمودار (اختیاری)
        figsize: اندازه نمودار

    Returns:
        matplotlib.figure.Figure: شیء نمودار
    """
    # بررسی پشتیبانی از پیش‌بینی احتمالات
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        logger.warning("مدل از predict_proba یا decision_function پشتیبانی نمی‌کند")
        return None

    # باینری‌سازی برچسب‌ها برای مدل چندکلاسه
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=range(n_classes))

    # محاسبه امتیازات پیش‌بینی
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
        # برای مدل باینری، تغییر شکل خروجی
        if y_score.ndim == 1:
            y_score = np.column_stack([1 - y_score, y_score])

    # ایجاد نمودار
    fig, ax = plt.subplots(figsize=figsize)

    # منحنی ROC برای هر کلاس
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    # خط مرجع
    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    # تنظیم نمودار
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('نرخ مثبت کاذب')
    ax.set_ylabel('نرخ مثبت حقیقی')
    ax.set_title('منحنی ROC چندکلاسه')
    ax.legend(loc="lower right")

    # ذخیره نمودار در صورت نیاز
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"منحنی ROC در {output_path} ذخیره شد")

    return fig


def plot_precision_recall_curve(model, X_test, y_test, class_names, output_path=None, figsize=(10, 8)):
    """
    رسم منحنی Precision-Recall برای مدل‌های چندکلاسه

    Args:
        model: مدل آموزش دیده
        X_test: ویژگی‌های تست
        y_test: برچسب‌های تست
        class_names: نام‌های کلاس‌ها
        output_path: مسیر ذخیره نمودار (اختیاری)
        figsize: اندازه نمودار

    Returns:
        matplotlib.figure.Figure: شیء نمودار
    """
    # بررسی پشتیبانی از پیش‌بینی احتمالات
    if not hasattr(model, "predict_proba") and not hasattr(model, "decision_function"):
        logger.warning("مدل از predict_proba یا decision_function پشتیبانی نمی‌کند")
        return None

    # باینری‌سازی برچسب‌ها برای مدل چندکلاسه
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=range(n_classes))

    # محاسبه امتیازات پیش‌بینی
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = model.decision_function(X_test)
        # برای مدل باینری، تغییر شکل خروجی
        if y_score.ndim == 1:
            y_score = np.column_stack([1 - y_score, y_score])

    # ایجاد نمودار
    fig, ax = plt.subplots(figsize=figsize)

    # منحنی Precision-Recall برای هر کلاس
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        avg_precision = np.mean(precision)
        ax.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP = {avg_precision:.2f})')

    # تنظیم نمودار
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('فراخوانی (Recall)')
    ax.set_ylabel('دقت (Precision)')
    ax.set_title('منحنی Precision-Recall چندکلاسه')
    ax.legend(loc="lower left")

    # ذخیره نمودار در صورت نیاز
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"منحنی Precision-Recall در {output_path} ذخیره شد")

    return fig


def plot_learning_curve(estimator, X, y, cv=5, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5),
                        title=None, output_path=None, figsize=(10, 6)):
    """
    رسم منحنی یادگیری برای بررسی بیش‌برازش/کم‌برازش

    Args:
        estimator: مدل آموزش داده نشده
        X: داده‌های ویژگی
        y: برچسب‌ها
        cv: تعداد تقسیمات اعتبارسنجی متقاطع
        n_jobs: تعداد کارهای همزمان
        train_sizes: اندازه‌های مختلف مجموعه آموزش
        title: عنوان نمودار (اختیاری)
        output_path: مسیر ذخیره نمودار (اختیاری)
        figsize: اندازه نمودار

    Returns:
        matplotlib.figure.Figure: شیء نمودار
    """
    # محاسبه منحنی یادگیری
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    # میانگین و انحراف معیار نمرات
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # ایجاد نمودار
    fig, ax = plt.subplots(figsize=figsize)

    # نمودار میانگین و محدوده انحراف معیار برای آموزش
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="نمره آموزش")

    # نمودار میانگین و محدوده انحراف معیار برای اعتبارسنجی
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="نمره اعتبارسنجی")

    # تنظیم نمودار
    ax.set_xlabel('تعداد نمونه‌های آموزش')
    ax.set_ylabel('نمره')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('منحنی یادگیری')
    ax.legend(loc="best")
    ax.grid(True)

    # ذخیره نمودار در صورت نیاز
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"منحنی یادگیری در {output_path} ذخیره شد")

    return fig


def plot_feature_importance(model, feature_names, top_n=20, title=None, output_path=None, figsize=(12, 8)):
    """
    رسم اهمیت ویژگی‌ها

    Args:
        model: مدل آموزش دیده
        feature_names: نام ویژگی‌ها
        top_n: تعداد ویژگی‌های برتر برای نمایش
        title: عنوان نمودار (اختیاری)
        output_path: مسیر ذخیره نمودار (اختیاری)
        figsize: اندازه نمودار

    Returns:
        matplotlib.figure.Figure: شیء نمودار یا None در صورت عدم پشتیبانی مدل
    """
    # بررسی پشتیبانی از اهمیت ویژگی‌ها
    if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
        logger.warning("مدل از feature_importances_ یا coef_ پشتیبانی نمی‌کند")
        return None

    # دریافت مقادیر اهمیت ویژگی‌ها
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # برای مدل‌های خطی مانند SVM یا رگرسیون لجستیک
        if model.coef_.ndim > 1:
            # در مدل‌های چندکلاسه، میانگین قدرمطلق ضرایب را می‌گیریم
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            importances = np.abs(model.coef_)

    # اگر تعداد نام‌های ویژگی با تعداد مقادیر اهمیت برابر نباشد
    if len(feature_names) != len(importances):
        logger.warning(
            f"تعداد نام‌های ویژگی ({len(feature_names)}) با تعداد مقادیر اهمیت ({len(importances)}) مطابقت ندارد")
        feature_names = [f"ویژگی {i}" for i in range(len(importances))]

    # ایجاد دیتافریم برای مرتب‌سازی
    features_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # مرتب‌سازی بر اساس اهمیت
    features_df = features_df.sort_values('importance', ascending=False)

    # انتخاب ویژگی‌های برتر
    if top_n > 0:
        features_df = features_df.head(top_n)

    # ایجاد نمودار
    fig, ax = plt.subplots(figsize=figsize)

    # رسم نمودار میله‌ای افقی
    ax.barh(features_df['feature'], features_df['importance'])

    # تنظیم نمودار
    ax.set_xlabel('اهمیت')
    ax.set_ylabel('ویژگی')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('اهمیت ویژگی‌ها')
    ax.invert_yaxis()  # نمایش ویژگی با بالاترین اهمیت در بالا

    # ذخیره نمودار در صورت نیاز
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"نمودار اهمیت ویژگی‌ها در {output_path} ذخیره شد")

    return fig


def calculate_class_weights(y, strategy='balanced'):
    """
    محاسبه وزن‌های کلاس برای داده‌های نامتوازن

    Args:
        y: برچسب‌های کلاس
        strategy: استراتژی وزن‌دهی ('balanced', 'balanced_subsample')

    Returns:
        Dict: دیکشنری از وزن‌های کلاس
    """
    if strategy not in ['balanced', 'balanced_subsample']:
        raise ValueError(f"استراتژی {strategy} پشتیبانی نمی‌شود")

    # محاسبه فراوانی کلاس‌ها
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    # محاسبه وزن‌ها
    weights = {}

    if strategy == 'balanced':
        for i, c in enumerate(unique_classes):
            weights[c] = n_samples / (n_classes * class_counts[i])
    elif strategy == 'balanced_subsample':
        # برای Random Forest با bootstrap=True
        for i, c in enumerate(unique_classes):
            weights[c] = 1 / class_counts[i]

    return weights


def generate_model_report(model, X_train, y_train, X_test, y_test, class_names, feature_names=None,
                          output_dir=None, model_name=None):
    """
    تولید گزارش کامل از مدل شامل معیارها و نمودارها

    Args:
        model: مدل آموزش دیده
        X_train: ویژگی‌های آموزش
        y_train: برچسب‌های آموزش
        X_test: ویژگی‌های تست
        y_test: برچسب‌های تست
        class_names: نام‌های کلاس‌ها
        feature_names: نام ویژگی‌ها (اختیاری)
        output_dir: دایرکتوری خروجی برای ذخیره گزارش (اختیاری)
        model_name: نام مدل برای استفاده در نام فایل‌ها (اختیاری)

    Returns:
        Dict: دیکشنری حاوی نتایج ارزیابی
    """
    if model_name is None:
        model_name = model.__class__.__name__

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODEL_DIR, 'reports', f"{model_name}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # ارزیابی مدل روی داده‌های تست
    y_pred = model.predict(X_test)

    # گزارش طبقه‌بندی
    classification_metrics = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # تلفیق نتایج
    results = {
        'model_name': model_name,
        'accuracy': classification_metrics['accuracy'],
        'macro_precision': classification_metrics['macro avg']['precision'],
        'macro_recall': classification_metrics['macro avg']['recall'],
        'macro_f1': classification_metrics['macro avg']['f1-score'],
        'class_metrics': classification_metrics,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    # ذخیره نتایج در فایل JSON
    json_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    save_metrics_to_json(results, json_path)

    # رسم ماتریس اغتشاش
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, class_names=class_names,
                          title=f"ماتریس اغتشاش - {model_name}", output_path=cm_path)

    # رسم منحنی ROC
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(model, X_test, y_test, class_names=class_names, output_path=roc_path)

    # رسم منحنی Precision-Recall
    pr_path = os.path.join(output_dir, f"{model_name}_precision_recall_curve.png")
    plot_precision_recall_curve(model, X_test, y_test, class_names=class_names, output_path=pr_path)

    # رسم اهمیت ویژگی‌ها (اگر ویژگی‌ها ارائه شده باشند)
    if feature_names is not None:
        fi_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
        plot_feature_importance(model, feature_names, output_path=fi_path)

    logger.info(f"گزارش کامل مدل {model_name} در {output_dir} ذخیره شد")

    return results