# evaluation_visualization.py
"""
Collection of visualization functions for machine learning model evaluation.

This module provides standardized plotting functions for common evaluation metrics
in binary classification. All functions are model-independent and can be used with
any classifier that follows scikit-learn conventions.
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
    metrics : list of str, default=['accuracy', 'f1', 'precision', 'recall']
        Metrics to evaluate. Can include: 'accuracy', 'f1', 'f2', 'precision', 'recall', 
        'roc_auc', 'average_precision'
    train_sizes : array-like, default=np.linspace(0.1, 1.0, 5)
        Points at which to evaluate training set size
    figsize : tuple, default=(15, 10)
        Size of the figure in inches
        
    Returns:
    --------
    fig : matplotlib figure
        Figure containing the learning curves
    
    Notes:
    ------
    - Requires cross-validation (cv parameter) for meaningful results
    - For imbalanced datasets, consider using 'f1' or 'average_precision' metrics
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

def evaluate_thresholds(y_true, y_prob, thresholds=np.linspace(0.1, 0.9, 17),
                       metrics=DEFAULT_METRICS):
    """
    Evaluates model performance at different classification thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities for the positive class
    thresholds : array-like, default=np.linspace(0.1, 0.9, 17)
        Thresholds to evaluate
    metrics : list of str, default=['precision', 'recall', 'f1', 'f2']
        Metrics to calculate
    
    Returns:
    --------
    list of dict
        Evaluation results for each threshold
    
    Notes:
    ------
    - Useful for finding optimal threshold for imbalanced datasets
    - Requires probability predictions
    - Consider your use case when choosing threshold (precision vs recall trade-off)
    """
    results = []
    
    # Calculate area under curves (AUC) metrics
    # These don't depend on threshold as they consider all possible thresholds
    auc_roc = roc_auc_score(y_true, y_prob)  # Area under ROC curve
    auc_pr = average_precision_score(y_true, y_prob)  # Area under Precision-Recall curve
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        result = {'threshold': t}
        
        if 'accuracy' in metrics:
            result['accuracy'] = accuracy_score(y_true, y_pred)
        if 'precision' in metrics:
            result['precision'] = precision_score(y_true, y_pred)
        if 'recall' in metrics:
            result['recall'] = recall_score(y_true, y_pred)
        if 'f1' in metrics:
            result['f1'] = f1_score(y_true, y_pred)
        if 'f2' in metrics:
            result['f2'] = fbeta_score(y_true, y_pred, beta=2)
        if 'roc_auc' in metrics:
            result['roc_auc'] = auc_roc
        if 'auc_pr' in metrics:
            result['auc_pr'] = auc_pr  # Also known as average precision (AP)
            
        results.append(result)
    
    return results

def plot_threshold_curves(results, metrics=None):
    """
    Visualizes how different metrics change with threshold.
    
    Parameters:
    -----------
    results : list of dict
        Output from evaluate_thresholds function
    metrics : list of str, optional
        Specific metrics to plot. If None, all metrics will be plotted.
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    
    Notes:
    ------
    - Useful for finding optimal threshold
    - Look for intersection points of precision and recall
    - AUC metrics are shown as horizontal lines as they don't depend on threshold
    """
    if metrics is None:
        metrics = list(results[0].keys())
        metrics.remove('threshold')
    
    plt.figure(figsize=(12, 6))
    for metric in metrics:
        if metric in ['roc_auc', 'auc_pr']:
            # AUC metrics are constant across thresholds (plot as horizontal lines)
            metric_label = 'AUC-ROC' if metric == 'roc_auc' else 'AUC-PR'
            plt.axhline(y=results[0][metric], 
                       label=f'{metric_label} = {results[0][metric]:.3f}',
                       linestyle='--', alpha=0.5)
        else:
            plt.plot([r['threshold'] for r in results],
                     [r[metric] for r in results],
                     'o-', label=metric)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metric Scores vs. Classification Threshold')
    plt.legend()
    plt.grid(True)
    
    return plt.gca()