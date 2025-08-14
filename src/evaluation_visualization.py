# evaluation_visualization.py
"""
Pure visualization functions for machine learning model evaluation.

This module provides standardized plotting functions for common evaluation metrics
in binary classification. For threshold optimization, use the optimize_threshold module.
All functions are model-independent and focus solely on visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score, fbeta_score,
    make_scorer, accuracy_score, classification_report
)
from sklearn.model_selection import learning_curve


# Default metrics for all evaluation functions
# Note: average_precision is equivalent to Area Under Precision-Recall Curve (AUPRC/AUC-PR)
DEFAULT_METRICS = ['accuracy', 'f1', 'f2', 'precision', 'recall', 'roc_auc', 'auc_pr']

def extended_classification_report(y_true, y_pred, y_prob=None):
    """
    Generates an extended classification report including additional metrics
    like F2-score and AUC values.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for the positive class
        Required for AUC-ROC and AUC-PR calculations
        
    Returns:
    --------
    str
        Formatted classification report including all metrics
    """
    # Get standard classification report
    standard_report = classification_report(y_true, y_pred)
    
    # Calculate metrics based on DEFAULT_METRICS
    additional_metrics = {}
    for metric in DEFAULT_METRICS:
        if metric == 'accuracy':
            additional_metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            # f1 is already in standard report
            continue
        elif metric == 'f2':
            additional_metrics['F2-score'] = fbeta_score(y_true, y_pred, beta=2)
        elif metric == 'precision' or metric == 'recall':
            # precision and recall are already in standard report
            continue
        elif metric == 'roc_auc' and y_prob is not None:
            additional_metrics['AUC-ROC'] = roc_auc_score(y_true, y_prob)
        elif metric == 'auc_pr' and y_prob is not None:
            additional_metrics['AUC-PR'] = average_precision_score(y_true, y_prob)
    
    # Format additional metrics
    additional_report = "\nAdditional Metrics:\n"
    for metric, value in additional_metrics.items():
        additional_report += f"{metric:<10} {value:.3f}\n"
    
    return standard_report + additional_report


def get_metrics_dict(y_true, y_pred, y_prob=None):
    """
    Extract structured metrics using same calculations as extended_classification_report.
    
    This function provides machine-readable structured output that complements
    the human-readable format from extended_classification_report.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities for the positive class
        Required for AUC-ROC and AUC-PR calculations
        
    Returns:
    --------
    dict
        Structured metrics dictionary containing:
        - 'classification_report': Standard sklearn classification report as dict
        - 'additional_metrics': Dict with accuracy, f2_score, and AUC metrics
        - 'class_1_metrics': Quick access to positive class metrics
    """
    # Get classification report as dict (same as in extended_classification_report)
    standard_dict = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate additional metrics (same logic as extended_classification_report)
    additional_metrics = {}
    additional_metrics['accuracy'] = accuracy_score(y_true, y_pred)
    additional_metrics['f2_score'] = fbeta_score(y_true, y_pred, beta=2)
    
    if y_prob is not None:
        additional_metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        additional_metrics['auc_pr'] = average_precision_score(y_true, y_prob)
    
    # Combine for comprehensive results
    return {
        'classification_report': standard_dict,
        'additional_metrics': additional_metrics,
        'class_1_metrics': {
            'precision': standard_dict['1']['precision'],
            'recall': standard_dict['1']['recall'], 
            'f1_score': standard_dict['1']['f1-score']
        }
    }


def plot_learning_curves(estimator, X, y, cv, 
                        metrics=DEFAULT_METRICS,
                        train_sizes=np.linspace(0.1, 1.0, 5), 
                        figsize=(20, 15)):
    """
    Displays multiple learning curves for different metrics to analyze model performance.
    
    Parameters:
    -----------
    estimator : estimator object
        A scikit-learn estimator with fit/predict methods
    X : array-like
        Training data
    y : array-like
        Target values
    cv : int or cross-validation generator
        Cross-validation splitting strategy
    metrics : list of str, default=DEFAULT_METRICS
        Metrics to evaluate. Can include: 'accuracy', 'f1', 'f2', 'precision', 'recall', 
        'roc_auc', 'auc_pr'
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
        Points at which to evaluate training set size
    figsize : tuple, default=(20, 15)
        Size of the figure in inches
        
    Returns:
    --------
    fig : matplotlib figure
        Figure containing the learning curves
    
    Notes:
    ------
    - Requires cross-validation (cv parameter) for meaningful results
    - For imbalanced datasets, use f2
    - Memory usage increases with train_sizes points
    """
    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.ravel()

    # Define custom scorers dictionary
    # Define custom F2 scorer function
    def f2_score(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=2)
    
    scorers = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'f2': make_scorer(f2_score),
        'roc_auc': 'roc_auc',
        'auc_pr': 'average_precision'
    }
    
    for idx, metric in enumerate(metrics):
        # Get learning curves using the appropriate scorer
        scorer = scorers.get(metric, metric)  # fallback to metric name if not in scorers
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scorer,
            train_sizes=train_sizes, n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        axes[idx].plot(train_sizes_abs, train_mean, 'o-', label='Training', color='blue')
        axes[idx].plot(train_sizes_abs, val_mean, 'o-', label='Cross-validation', color='orange')
        
        # Add standard deviation bands
        axes[idx].fill_between(train_sizes_abs, train_mean - train_std,
                             train_mean + train_std, alpha=0.1, color='blue')
        axes[idx].fill_between(train_sizes_abs, val_mean - val_std,
                             val_mean + val_std, alpha=0.1, color='orange')
        
        axes[idx].set_title(f'Learning Curve - {metric.upper()}')
        axes[idx].set_xlabel('Training Examples')
        axes[idx].set_ylabel('Score')
        axes[idx].grid(True)
        axes[idx].legend(loc='best')

    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=('irrelevant', 'relevant'), 
                         normalize=False, title=None, ax=None):
    """
    Creates a confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : tuple, default=('irrelevant', 'relevant')
        Labels for the classes
    normalize : bool, default=False
        Whether to normalize the confusion matrix
    title : str, optional
        Title for the plot
    ax : matplotlib axis, optional
        Axis to plot on
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    
    Notes:
    ------
    - For imbalanced datasets, consider using normalize=True
    - Labels should match your actual class names
    """
    if ax is None:
        ax = plt.gca()

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    return ax

def plot_roc_and_pr_curves(y_true, y_prob, ax=None):
    """
    Plots ROC curve and Precision-Recall curve side by side.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities for the positive class
    ax : tuple of matplotlib axes, optional
        Two axes to plot on
    
    Returns:
    --------
    fig : matplotlib figure
        Figure containing both curves
    
    Notes:
    ------
    - For imbalanced datasets, Precision-Recall curve is more informative than ROC
    - Requires probability predictions (not class predictions)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax[0].plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax[0].plot([0,1], [0,1], 'k--')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('ROC Curve')
    ax[0].legend(loc='lower right')
    ax[0].grid(True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax[1].step(recall, precision, where='post')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title(f'Precision-Recall Curve (AP={ap:.3f})')
    ax[1].grid(True)

    plt.tight_layout()
    return plt.gcf()

def plot_threshold_curves(threshold_results, metrics=DEFAULT_METRICS):
    """
    Visualizes how different metrics change with threshold.
    
    Parameters:
    -----------
    threshold_results : list of dict
        Threshold evaluation results from optimize_threshold module
    metrics : list of str, optional
        Specific metrics to plot. If None, all available metrics will be plotted.
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    
    Notes:
    ------
    - Input data should come from optimize_threshold.get_threshold_evaluation_data()
    - Look for intersection points of precision and recall
    - AUC metrics are shown as horizontal lines as they don't depend on threshold
    """
    if not threshold_results:
        raise ValueError("threshold_results cannot be empty")
        
    if metrics is None:
        metrics = list(threshold_results[0].keys())
        metrics.remove('threshold')
    
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        if metric in ['roc_auc', 'auc_pr']:
            # AUC metrics are constant across thresholds (plot as horizontal lines)
            metric_label = 'AUC-ROC' if metric == 'roc_auc' else 'AUC-PR'
            plt.axhline(y=threshold_results[0][metric], 
                       label=f'{metric_label} = {threshold_results[0][metric]:.3f}',
                       linestyle='--', alpha=0.5)
        else:
            plt.plot([r['threshold'] for r in threshold_results],
                     [r[metric] for r in threshold_results],
                     'o-', label=metric)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Scores vs. Classification Threshold')
    plt.legend()
    plt.grid(True)
    
    return plt.gca()


def quick_f2_score_default_threshold(y_true, y_prob, threshold=0.5):
    """
    Quick calculation of F2-score with default threshold for comparison purposes.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities for the positive class
    threshold : float, default=0.5
        Classification threshold
        
    Returns:
    --------
    float
        F2-score with the specified threshold
    """
    y_pred_default = (y_prob >= threshold).astype(int)
    return fbeta_score(y_true, y_pred_default, beta=2)