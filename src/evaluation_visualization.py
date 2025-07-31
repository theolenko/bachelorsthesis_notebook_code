# evaluation_visualization.py
"""
Collection of visualization functions for machine learning model evaluation.

This module provides standardized plotting functions for common evaluation metrics
in binary classification. All functions are model-independent and can be used with
any classifier that follows scikit-learn conventions.

Functions:
- plot_learning_curve: Visualizes learning curves for train/validation
- plot_confusion_matrix: Creates a formatted confusion matrix
- plot_roc_curve: Shows ROC curve with AUC score
- plot_precision_recall_curve: Visualizes precision-recall trade-off
- evaluate_thresholds: Evaluates different thresholds for classification

Examples:
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_prob = model.predict_proba(X_test)[:, 1]
    >>> plot_roc_curve(y_test, y_prob)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, cv, scoring='recall'):
    """Displays learning curves for training and validation data to detect over/underfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
    plt.xlabel('Training Size'); plt.ylabel(scoring)
    plt.legend(); plt.tight_layout(); plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=('irrelevant','relevant')):
    """Creates a confusion matrix for analyzing true/false positives/negatives."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout(); plt.show()

def plot_roc_curve(y_true, y_prob):
    """Visualizes ROC curve with AUC score for evaluating classification performance."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

def plot_precision_recall_curve(y_true, y_prob):
    """Shows precision-recall curve with average precision score, particularly important for imbalanced data."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'AP={ap:.3f}'); plt.tight_layout(); plt.show()

def evaluate_thresholds(y_true, y_prob, thresholds=np.arange(0.1,0.95,0.05)):
    """Evaluates different thresholds for binary classification using precision, recall, and F1."""
    return [
        {
            'threshold': t,
            'precision': precision_score(y_true, (y_prob>=t).astype(int)),
            'recall':    recall_score(y_true, (y_prob>=t).astype(int)),
            'f1':        f1_score(y_true, (y_prob>=t).astype(int))
        }
        for t in thresholds
    ]
