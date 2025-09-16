# optimize_threshold.py
"""
Threshold optimization module using sklearn's TunedThresholdClassifierCV.

This module provides a clean interface for threshold optimization that uses
proper nested cross-validation to prevent overfitting. It uses sklearn's 
robust implementation for academic and production environments.


Usage Examples:
    
    # Robust threshold optimization with nested CV:
    tuned_model, results = optimize_threshold_with_cv(
        base_estimator=best_model, 
        X=X_dev, y=y_dev, 
        scoring='f2',
        coarse_to_fine=True  # Two-stage optimization for better results
    )
    
    # Get data for plotting:
    threshold_data = get_threshold_evaluation_data(tuned_model, X_dev, y_dev)
    plot_threshold_curves(threshold_data)
"""

import numpy as np
from sklearn.model_selection import TunedThresholdClassifierCV, StratifiedKFold
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.base import clone, is_classifier

class DummyTuned:
    """A lightweight drop-in replacement for TunedThresholdClassifierCV.
    Stores only threshold + score + estimator and mimics predict/predict_proba.
    """
    def __init__(self, best_threshold, best_score, estimator):
        self.best_threshold_ = best_threshold
        self.best_score_ = best_score
        self.estimator_ = estimator

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.best_threshold_).astype(int)



def optimize_threshold_with_cv(base_estimator, 
                               X, y, 
                               scoring='f2', 
                               cv=None, 
                               thresholds=50, 
                               coarse_to_fine=True, 
                               fine_range_factor=0.15,
                               n_jobs=-1, 
                               random_state=42,
                               retrain=True):
    """
    Optimizes classification threshold using TunedThresholdClassifierCV with proper nested CV.
    
    This function wraps sklearn's TunedThresholdClassifierCV to provide threshold optimization
    with nested cross-validation, preventing overfitting that can occur with manual threshold
    tuning on the same data used for hyperparameter optimization.
    
    Parameters:
    -----------
    base_estimator : estimator object
        A fitted sklearn estimator (e.g., result from RandomizedSearchCV.best_estimator_)
        Must have predict_proba or decision_function method
    X : array-like
        Feature data for threshold optimization
    y : array-like  
        Target labels for threshold optimization
    scoring : str or callable, default='f2'
        Scoring metric for threshold optimization.
        - 'f2': F2-score (emphasizes recall)
        - 'f1': F1-score (balanced precision/recall)
        - 'precision': Precision score
        - 'recall': Recall score
        - Custom scorer from make_scorer()
    cv : int or cross-validation generator, default=None
        Cross-validation strategy for threshold tuning. If None, uses 3-fold StratifiedKFold
    thresholds : int or array-like, default=50
        Number of thresholds to evaluate per stage, or explicit threshold values.
        If coarse_to_fine=True: thresholds used for both coarse and fine stages.
    coarse_to_fine : bool, default=True
        Whether to use two-stage optimization:
        - Stage 1: Coarse scan over entire score range
        - Stage 2: Fine scan around optimum from stage 1
        If False, uses single-stage optimization with sklearn's default threshold selection.
    fine_range_factor : float, default=0.15
        For coarse_to_fine mode: fraction of total score range to use for fine scan.
        E.g., 0.15 means fine scan covers 15% of the total score range around coarse optimum.
    n_jobs : int, default=-1
        Number of parallel jobs for cross-validation
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuned_estimator : TunedThresholdClassifierCV
        Fitted estimator with optimized threshold
    results : dict
        Dictionary containing:
        - 'best_threshold': Optimized threshold value
        - 'cv_score': Cross-validation score with optimal threshold
        - 'estimator': The tuned estimator
        
    Examples:
    ---------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> 
    >>> # Step 1: Hyperparameter optimization
    >>> rf = RandomForestClassifier()
    >>> param_dist = {'n_estimators': [10, 50, 100]}
    >>> search = RandomizedSearchCV(rf, param_dist, cv=5)
    >>> search.fit(X_train, y_train)
    >>> 
    >>> # Step 2: Threshold optimization (separate from hyperparameters)
    >>> tuned_model, results = optimize_threshold_with_cv(
    ...     search.best_estimator_, X_train, y_train, scoring='f2'
    ... )
    >>> 
    >>> # Step 3: Evaluate on test set
    >>> y_pred = tuned_model.predict(X_test)
    
    Notes:
    ------
    - Uses nested cross-validation to prevent overfitting
    - More robust than manual threshold optimization
    - coarse_to_fine=True provides better optimization for models with wide score ranges (e.g., SVM)
    - Integrates seamlessly with sklearn pipelines
    - Recommended for academic/production use
    """
    
    # Handle default CV
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    
    # Handle scoring parameter
    if scoring == 'f2':
        scorer = make_scorer(fbeta_score, beta=2)
    elif scoring == 'f1':
        scorer = 'f1'
    elif scoring == 'precision':
        scorer = 'precision'
    elif scoring == 'recall':
        scorer = 'recall'
    else:
        scorer = scoring  # Assume it's already a valid scorer
    
    # Only clone if retrain=True, otherwise reuse the fitted estimator (important for heavy models like EuroBERT)
    base_estimator_copy = clone(base_estimator) if retrain else base_estimator

    # Determine threshold strategy
    if coarse_to_fine and isinstance(thresholds, int):
        # Two-stage coarse-to-fine optimization
        tuned_estimator = _optimize_coarse_to_fine(
            base_estimator_copy, X, y, scorer, cv, thresholds, 
            fine_range_factor, n_jobs, random_state, retrain=retrain
        )
    else:
        # Standard single-stage optimization
        tuned_estimator = TunedThresholdClassifierCV(
            estimator=base_estimator_copy,
            scoring=scorer,
            cv=cv,
            thresholds=thresholds,
            n_jobs=n_jobs
        )
        tuned_estimator.fit(X, y)
    
    # Prepare results
    results = {
        'best_threshold': tuned_estimator.best_threshold_,
        'cv_score': tuned_estimator.best_score_,
        'estimator': tuned_estimator
    }
    
    return tuned_estimator, results


def get_threshold_evaluation_data(tuned_estimator, X, y, thresholds=None):
    """
    Generates threshold evaluation data for visualization purposes.
    
    This function extracts threshold evaluation results from a fitted
    TunedThresholdClassifierCV for plotting threshold curves.
    
    Parameters:
    -----------
    tuned_estimator : TunedThresholdClassifierCV
        Fitted threshold-optimized estimator
    X : array-like
        Feature data
    y : array-like
        Target labels
    thresholds : array-like, optional
        Custom threshold values for evaluation. If None, auto-generates appropriate range
        
    Returns:
    --------
    threshold_results : list of dict
        List of dictionaries with threshold evaluation results,
        compatible with plot_threshold_curves from evaluation_visualization
    """
    
    # Get decision scores from the base estimator (raw scores before threshold optimization)
    # This is correct for threshold visualization - we need the original model scores
    if hasattr(tuned_estimator.estimator_, 'predict_proba'):
        scores = tuned_estimator.estimator_.predict_proba(X)[:, 1]
    elif hasattr(tuned_estimator.estimator_, 'decision_function'):
        scores = tuned_estimator.estimator_.decision_function(X)
    else:
        raise ValueError("Base estimator must have predict_proba or decision_function method")
    
    # Use custom thresholds or derive from score range
    if thresholds is None:
        if hasattr(tuned_estimator.estimator_, 'predict_proba'):
            # For probability outputs, use 0-1 range
            thresholds = np.linspace(0.1, 0.9, 50)
        else:
            # For decision function outputs, use score range, cut off edge cases
            score_min, score_max = scores.min(), scores.max()
            thresholds = np.linspace(score_min * 0.8, score_max * 0.8, 50)

    # Generate threshold evaluation results directly
    threshold_results = _evaluate_thresholds_for_viz(y, scores, thresholds)
    
    return threshold_results


def get_threshold_evaluation_data_cv(base_estimator, X, y, cv=None, thresholds=None):
    """
    Generates threshold evaluation data using cross-validation predict for unbiased visualization.
    
    This approach prevents overfitting in threshold visualization by using cross-validation
    to generate out-of-fold predictions, providing a more realistic view of threshold performance.
    
    Parameters:
    -----------
    base_estimator : estimator object
        Base sklearn estimator (before threshold optimization)
    X : array-like
        Feature data
    y : array-like
        Target labels
    cv : cross-validation generator, optional
        Cross-validation strategy. If None, uses 3-fold StratifiedKFold
    thresholds : array-like, optional
        Custom threshold values for evaluation. If None, auto-generates appropriate range
        
    Returns:
    --------
    threshold_results : list of dict
        List of dictionaries with threshold evaluation results,
        compatible with plot_threshold_curves from evaluation_visualization
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    
    # Handle default CV
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Get cross-validation predictions (out-of-fold scores)
    if hasattr(base_estimator, 'predict_proba'):
        # For classifiers with predict_proba
        cv_probas = cross_val_predict(base_estimator, X, y, cv=cv, method='predict_proba')
        scores = cv_probas[:, 1]  # Positive class probabilities
    elif hasattr(base_estimator, 'decision_function'):
        # For classifiers with decision_function (like SVM)
        scores = cross_val_predict(base_estimator, X, y, cv=cv, method='decision_function')
    else:
        raise ValueError("Base estimator must have predict_proba or decision_function method")
    
    # Use custom thresholds or derive from score range
    if thresholds is None:
        if hasattr(base_estimator, 'predict_proba'):
            # For probability outputs, use 0-1 range
            thresholds = np.linspace(0.1, 0.9, 50)
        else:
            # For decision function outputs, use score range, cut off edge cases
            score_min, score_max = scores.min(), scores.max()
            thresholds = np.linspace(score_min * 0.8, score_max * 0.8, 50)

    # Generate threshold evaluation results using CV predictions
    threshold_results = _evaluate_thresholds_for_viz(y, scores, thresholds)
    
    return threshold_results


def _evaluate_thresholds_for_viz(y_true, y_scores, thresholds):
    """
    Internal function to evaluate thresholds for visualization purposes.
    This replaces the removed evaluate_thresholds from evaluation_visualization.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, fbeta_score, roc_auc_score, average_precision_score
    )
    
    results = []
    
    # Calculate AUC metrics (threshold-independent)
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        result = {
            'threshold': t,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'roc_auc': auc_roc,
            'auc_pr': auc_pr
        }
        results.append(result)
    
    return results


def _optimize_coarse_to_fine(base_estimator, 
                             X, 
                             y, 
                             scorer, 
                             cv, 
                             n_thresholds, 
                            fine_range_factor, 
                            n_jobs, 
                            random_state,
                            retrain=True):
    """
    Two-stage coarse-to-fine threshold optimization.
    
    Stage 1: Coarse scan over entire score range
    Stage 2: Fine scan around optimum from stage 1
    """
    
    if retrain:
        # Stage 1: Coarse scan over entire possible range
        # Get score range by fitting estimator on full data temporarily
        temp_estimator = clone(base_estimator)
        temp_estimator.fit(X, y)
        if hasattr(temp_estimator, 'predict_proba'):
            temp_scores = temp_estimator.predict_proba(X)[:, 1]
            score_min, score_max = -0.1, 1.1
        elif hasattr(temp_estimator, 'decision_function'):
            temp_scores = temp_estimator.decision_function(X)
            score_min = temp_scores.min() - 0.1 * abs(temp_scores.min())
            score_max = temp_scores.max() + 0.1 * abs(temp_scores.max())
        else:
            raise ValueError("Estimator must have predict_proba or decision_function method")
    else:
        # Shortcut for big models as EuroBERT
        if hasattr(base_estimator, 'predict_proba'):
            temp_scores = base_estimator.predict_proba(X)[:, 1]
            score_min, score_max = 0.0, 1.0
        elif hasattr(base_estimator, 'decision_function'):
            temp_scores = base_estimator.decision_function(X)
            score_min, score_max = temp_scores.min(), temp_scores.max()
        else:
            raise ValueError("Estimator must have predict_proba or decision_function method")

    
    # Stage 1: Coarse thresholds
    coarse_thresholds = np.linspace(score_min, score_max, n_thresholds)
    
    if retrain:
        coarse_estimator = TunedThresholdClassifierCV(
            estimator=clone(base_estimator),
            scoring=scorer,
            cv=cv,
            thresholds=coarse_thresholds,
            n_jobs=n_jobs
        )
        coarse_estimator.fit(X, y)
        coarse_optimum = coarse_estimator.best_threshold_
    else:
        # shortcut: no retraining, just use scores of fitted model
        scores = base_estimator.predict_proba(X)[:, 1]
        from sklearn.metrics import fbeta_score
        best_t, best_s = None, -1
        for t in coarse_thresholds:
            preds = (scores >= t).astype(int)
            s = fbeta_score(y, preds, beta=2, zero_division=0)
            if s > best_s:
                best_s, best_t = s, t
        coarse_optimum = best_t
    total_range = score_max - score_min
    fine_range = total_range * fine_range_factor
    
    # Define fine search bounds
    fine_min = max(score_min, coarse_optimum - fine_range / 2)
    fine_max = min(score_max, coarse_optimum + fine_range / 2)
    
    # Ensure we have a meaningful range for fine search
    if fine_max - fine_min < (score_max - score_min) * 0.01:  # At least 1% of total range
        fine_range = (score_max - score_min) * 0.05  # Use 5% of total range
        fine_min = max(score_min, coarse_optimum - fine_range / 2)
        fine_max = min(score_max, coarse_optimum + fine_range / 2)
    
    fine_thresholds = np.linspace(fine_min, fine_max, n_thresholds)
    
    # Final optimization with fine thresholds
    if retrain:
        fine_estimator = TunedThresholdClassifierCV(
            estimator=clone(base_estimator),
            scoring=scorer,
            cv=cv,
            thresholds=fine_thresholds,
            n_jobs=n_jobs
        )
        fine_estimator.fit(X, y)
        return fine_estimator
    else:
        # shortcut: manual threshold scan without retraining
        scores = base_estimator.predict_proba(X)[:, 1]
        from sklearn.metrics import fbeta_score
        best_t, best_s = None, -1
        for t in fine_thresholds:
            preds = (scores >= t).astype(int)
            s = fbeta_score(y, preds, beta=2, zero_division=0)
            if s > best_s:
                best_s, best_t = s, t

    return DummyTuned(best_t, best_s, base_estimator)



