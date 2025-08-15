"""
Advanced Hyperparameter Optimization Module using Optuna

Provides model-specific optimization strategies with intelligent objective function
selection based on model capabilities and pruning requirements.

KEY FEATURES:
- Model-specific objective functions for optimal pruning
- Comprehensive logging and progress tracking
- Parameter validation and filtering
- Both Bayesian (TPE) and exhaustive (Grid) search
- Full reproducibility with random state control

OPTIMIZATION STRATEGIES:
- "logistic_regression": Manual CV with fold-level pruning
- "svm": Manual CV optimized for SVM characteristics  
- "naive_bayes": Manual CV for NB models
- "xgboost": XGBoost-specific optimization with built-in CV
- "mlp": Neural network optimization with per-epoch pruning
- "manual_cv": Generic manual CV (fallback)

USAGE:
>>> best_model, best_params, study = optimize_with_optuna(
...     estimator=pipeline,
...     param_space=param_space,
...     X=X_train, y=y_train,
...     cv=5,
...     model_type="logistic_regression",  # Model-specific optimization
...     n_trials=100
... )
"""
import os
from datetime import datetime
import numpy as np
import logging

import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler, GridSampler
from optuna.logging import set_verbosity, INFO, WARNING
from optuna.integration import XGBoostPruningCallback

from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import xgboost as xgb
from xgboost.callback import EarlyStopping



def sample_class_weight_value(
    trial, low=1, high=50, step=1, param_name="positive_class_weight"
):
    pos_weight = trial.suggest_int(param_name, low, high, step=step)
    return pos_weight



# LOGGING FUNCTIONS 


def setup_logging(model_name: str, verbose: bool = False, log_to_file: bool = True):
    """
    Setup logging for Optuna optimization with file and optional console output.
    Handles cleanup to prevent duplicate handlers across runs.
    """
    if not log_to_file:
        return None
        
    log_file_path = f"optuna_log_{model_name}.txt"
    logger = logging.getLogger(f"optuna_{model_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with append mode
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler only if verbose is True
    if verbose:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_optimization_start(logger, model_name, n_trials, direction, random_state, param_space, sampler_type="TPE"):
    """Log optimization session initialization and configuration."""
    if logger is None:
        return
        
    logger.info("=" * 80)
    logger.info(f"NEW OPTIMIZATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Starting Optuna optimization for model: {model_name}")
    logger.info(f"Sampler: {sampler_type}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Direction: {direction}")
    logger.info(f"Random state: {random_state}")
    logger.info(f"Parameter space: {list(param_space.keys())}")
    logger.info("-" * 50)


def log_optimization_completion(logger, study, valid_params):
    """Log optimization results, statistics, and session completion."""
    if logger is None:
        return
        
    logger.info("-" * 50)
    logger.info(f"Optimization completed after {len(study.trials)} trials")
    logger.info(f"Best F2 score: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logger.info(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    logger.info(f"Model training completed with parameters: {valid_params}")
    logger.info(f"SESSION COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    logger.info("")


def cleanup_logger(logger):
    """Clean up logger handlers to prevent resource leaks."""
    if logger is None:
        return
        
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)



# OBJECTIVE FUNCTIONS


# MODEL-SPECIFIC OBJECTIVE FUNCTIONS

def logistic_regression_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    Logistic Regression specific objective with manual CV and fold-level pruning.
    Optimized for LogReg characteristics and regularization patterns.
    """
    return manual_cv_objective(trial, estimator, param_space, X, y, cv, random_state, logger)


def svm_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    SVM specific objective with manual CV optimized for SVM training patterns.
    TODO: Add SVM-specific optimizations (kernel caching, etc.)
    """
    return manual_cv_objective(trial, estimator, param_space, X, y, cv, random_state, logger)


def naive_bayes_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    Naive Bayes specific objective with manual CV.
    TODO: Add NB-specific optimizations (feature independence assumptions, etc.)
    """
    return manual_cv_objective(trial, estimator, param_space, X, y, cv, random_state, logger)


def xgboost_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    Optuna objective for XGBoost inside an sklearn Pipeline (fold-level pruning).

    Why this design:
    - Keep F2 as the monitored metric (consistency with your other models).
    - Use XGBoost EarlyStopping (per fold) for speed.
    - Do *fold-level* Optuna pruning to avoid duplicate step IDs and to restore clear logs
      like "Trial N: Pruned at fold k".
    """

    # 1) Clone estimator and sample all params just like in your manual objective
    model = clone(estimator)
    trial_params = {}
    for name, sampler in param_space.items():
        val = sampler(trial)
        model.set_params(**{name: val})
        trial_params[name] = val
    if logger:
        logger.info(f"Trial {trial.number}: Testing parameters: {trial_params}")

    # 2) Ensure ndarray for splitter
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # 3) CV splitter (deterministic)
    splitter = cv if hasattr(cv, "split") else StratifiedKFold(
        n_splits=int(cv), shuffle=True, random_state=random_state
    )

    # 4) Split pipeline into preprocessor and classifier
    if hasattr(model, "steps") and len(model.steps) > 0:
        preprocessor = model[:-1]
        clf_template = model.steps[-1][1]
    else:
        preprocessor = None
        clf_template = model

    # 5) Separate xgb vs. preprocessing params
    xgb_params, other_params = {}, {}
    for k, v in trial_params.items():
        if k.startswith("clf__"):
            xgb_params[k[5:]] = v            # drop 'clf__'
        else:
            other_params[k] = v
    if preprocessor is not None and other_params:
        preprocessor.set_params(**other_params)

    # 6) Map class_weight -> scale_pos_weight if present
    if "class_weight" in xgb_params:
        cw = xgb_params.pop("class_weight")
        if isinstance(cw, dict):
            xgb_params["scale_pos_weight"] = float(cw.get(1, 1.0))
        elif isinstance(cw, (int, float)):
            xgb_params["scale_pos_weight"] = float(cw)

    # 7) Fit-time defaults (don’t override trial choices)
    early_stopping_rounds = int(xgb_params.pop("early_stopping_rounds", 50))
    xgb_defaults = dict(
        objective="binary:logistic",
        tree_method=xgb_params.get("tree_method", "hist"),
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_params = {**xgb_defaults, **xgb_params}

    # 8) sklearn-style F2 metric (used by EarlyStopping)
    def f2(y_true, y_pred, sample_weight=None):
        y_hat = (y_pred >= 0.5).astype(int)
        return fbeta_score(y_true, y_hat, beta=2, zero_division=0,
                           sample_weight=sample_weight)

    # IMPORTANT CHANGE 
    # We do NOT use XGBoostPruningCallback here. Instead we prune after each fold
    # (unique step == fold index) to avoid duplicate-step warnings and to get
    # clean "Trial X: Pruned at fold Y" logs.

    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X_arr, y_arr)):
        X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
        X_va, y_va = X_arr[va_idx], y_arr[va_idx]

        # Fit/transform per fold to avoid leakage
        if preprocessor is not None:
            P = clone(preprocessor)
            X_tr_proc = P.fit_transform(X_tr, y_tr)
            X_va_proc = P.transform(X_va)
        else:
            X_tr_proc, X_va_proc = X_tr, X_va

        # Early stopping callback: maximize F2 on the validation set
        es_cb = EarlyStopping(
            rounds=early_stopping_rounds,
            maximize=True,          # F2 is to be maximized
            save_best=True,
            data_name="validation_0",
            metric_name="f2",
        )

        # Set eval_metric + EarlyStopping on the estimator (XGBoost >= 2.1 expects this)
        clf = clone(clf_template).set_params(
            **xgb_params,
            eval_metric=f2,
            callbacks=[es_cb],
        )

        # Fit with eval_set; do NOT pass eval_metric/callbacks to fit()
        clf.fit(X_tr_proc, y_tr, eval_set=[(X_va_proc, y_va)], verbose=False)

        # Score this fold with F2 at threshold 0.5
        y_proba = clf.predict_proba(X_va_proc)[:, 1]
        y_hat = (y_proba >= 0.5).astype(int)
        s = fbeta_score(y_va, y_hat, beta=2, zero_division=0)
        fold_scores.append(float(s))

        # Fold-level pruning & logging (unique steps => no warnings) 
        trial.report(float(np.mean(fold_scores)), step=fold_idx)
        if trial.should_prune():
            if logger:
                logger.info(f"Trial {trial.number}: Pruned at fold {fold_idx} (mean F2={np.mean(fold_scores):.4f})")
            raise optuna.TrialPruned()

    mean_score = float(np.mean(fold_scores))
    if logger:
        logger.info(f"Trial {trial.number}: CV F2 Score = {mean_score:.4f} (±{np.std(fold_scores):.4f})")
    return mean_score



#def mlp_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    MLP/Neural Network specific objective with per-epoch pruning capabilities.
    TODO: Implement per-epoch reporting with MLPClassifier partial_fit or validation curves.
    """
    return simple_cv_objective(trial, estimator, param_space, X, y, cv, random_state, logger)


# GENERIC OBJECTIVE FUNCTIONS (FALLBACK, default for logreg, svm, nb)

def manual_cv_objective(trial, estimator, param_space, X, y, cv, random_state, logger=None):
    """
    Generic manual CV objective for models without intermediate reporting.
    Implements manual cross-validation with fold-by-fold result reporting for pruning.
    """
    model = clone(estimator)

    # Set parameters and log trial configuration
    trial_params = {}
    for name, sampler in param_space.items():
        val = sampler(trial)
        model.set_params(**{name: val})
        trial_params[name] = val
    
    if logger:
        logger.info(f"Trial {trial.number}: Testing parameters: {trial_params}")

    # Ensure arrays are in numpy format
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # Use provided CV splitter or create default StratifiedKFold
    splitter = cv if hasattr(cv, "split") else StratifiedKFold(
        n_splits=int(cv), shuffle=True, random_state=random_state
    )

    # Manual CV loop with intermediate reporting for pruning
    fold_scores = []
    for step, (tr_idx, va_idx) in enumerate(splitter.split(X_arr, y_arr)):
        m = clone(model)
        m.fit(X_arr[tr_idx], y_arr[tr_idx])

        y_hat = m.predict(X_arr[va_idx])
        s = fbeta_score(y_arr[va_idx], y_hat, beta=2, zero_division=0)
        fold_scores.append(s)

        # Report intermediate result and check for pruning
        trial.report(float(np.mean(fold_scores)), step=step)
        if trial.should_prune():
            if logger:
                logger.info(f"Trial {trial.number}: Pruned at fold {step}")
            raise optuna.TrialPruned()

    mean_score = float(np.mean(fold_scores))
    if logger:
        logger.info(f"Trial {trial.number}: CV F2 Score = {mean_score:.4f} (±{np.std(fold_scores):.4f})")
    
    return mean_score


def get_objective_function(model_type: str):
    """
    Get model-specific objective function for optimal optimization strategy.
    
    Returns the appropriate objective function based on model type, enabling
    model-specific optimizations and pruning strategies.
    
    Available model types:
    - "logistic_regression": LogReg with manual CV and fold-level pruning
    - "svm": SVM with manual CV optimized for SVM characteristics
    - "naive_bayes": Naive Bayes with manual CV 
    - "xgboost": XGBoost with built-in CV and early stopping
    - "mlp": Neural networks with per-epoch pruning capabilities
    - "manual_cv": Generic manual CV (fallback)
    """
    
    objective_map = {
        # Model-specific objectives
        "logistic_regression": logistic_regression_objective,
        "svm": svm_objective,
        "naive_bayes": naive_bayes_objective,
        "xgboost": xgboost_objective,
        #"mlp": mlp_objective,
        #tbd
        
        # Generic fallback objectives
        "manual_cv": manual_cv_objective,
        
        # TODO: Add more model-specific objectives
        # "lightgbm": lightgbm_objective,
        # "random_forest": random_forest_objective,
        # "gradient_boosting": gradient_boosting_objective,
    }
    
    if model_type not in objective_map:
        available_types = list(objective_map.keys())
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available options: {available_types}"
        )
    
    return objective_map[model_type]


def validate_and_filter_params(estimator, best_params):
    """
    Filter hyperparameters to ensure compatibility with the estimator pipeline.
    Handles both simple estimators and sklearn Pipeline objects with nested parameters.
    """
    valid_pipeline_params = {}
    cloned_estimator = clone(estimator)
    estimator_params = cloned_estimator.get_params().keys()
    
    for param_name, param_value in best_params.items():
        # Include parameters that are directly valid for the estimator
        if param_name in estimator_params:
            valid_pipeline_params[param_name] = param_value
        # Include nested parameters for pipeline components (e.g., clf__C, select__k)
        elif hasattr(cloned_estimator, 'steps') and any(param_name.startswith(step + '__') for step, _ in cloned_estimator.steps):
            valid_pipeline_params[param_name] = param_value
    
    return valid_pipeline_params



# MAIN OPTIMIZATION FUNCTIONS


def optimize_with_optuna_tpe(
    estimator,
    param_space: dict,
    X, y,
    cv,
    model_type: str = "manual_cv",
    n_trials: int = 250,
    direction: str = "maximize",
    random_state: int = 42,
    # Pruner parameters
    n_warmup_steps: int = 1,
    # Logging
    verbose: bool = False,
    log_to_file: bool = True,
    model_name: str = "optuna_model"
):
    """
    Hyperparameter optimization using Optuna with model-specific objective functions.
    
    Provides unified interface for optimization that automatically selects the most
    appropriate strategy based on model type. Supports both traditional ML models
    and modern models with built-in early stopping capabilities.
    
    Parameters:
    -----------
    model_type : str, default="manual_cv"
        Model-specific optimization strategy:
        - "logistic_regression": LogReg with manual CV and fold-level pruning
        - "svm": SVM with manual CV optimized for SVM characteristics  
        - "naive_bayes": Naive Bayes with manual CV
        - "xgboost": XGBoost with built-in CV and early stopping
        - "mlp": Neural networks with per-epoch pruning
        - "manual_cv": Generic manual CV (fallback)
    
    Returns:
    --------
    tuple[sklearn estimator, dict, optuna.Study]
        - best_model: Trained model with optimal hyperparameters
        - best_params: Dictionary of optimal hyperparameters
        - study: Complete Optuna study object with all trial results
    """
    #set pruner parameter dynamically, trials with no prunning and no tpe(instead random search)
    n_startup_trials = max(20, int(0.1 * n_trials))

    # Configure Optuna logging verbosity
    if verbose:
        set_verbosity(INFO)
    else:
        set_verbosity(WARNING)

    # Setup session logging
    logger = setup_logging(model_name, verbose, log_to_file)
    log_optimization_start(logger, model_name, n_trials, direction, random_state, param_space)

    # Get model-specific objective function
    objective_func = get_objective_function(model_type)
    
    def objective(trial):
        return objective_func(trial, estimator, param_space, X, y, cv, random_state, logger)

    # Configure advanced TPE sampler with multivariate optimization
    sampler = TPESampler(
        seed=random_state, 
        multivariate=True,  # Consider parameter interactions
        group=True,         # Group related parameters
        n_startup_trials=n_startup_trials,
    )
    
    # Configure pruner for early termination of unpromising trials
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps
    )
    
    # Create and configure study
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)

    # Process and validate results
    best_params = study.best_params
    valid_pipeline_params = validate_and_filter_params(estimator, best_params)
    
    # Train final model with optimal parameters
    best_model = clone(estimator).set_params(**valid_pipeline_params)
    best_model.fit(X, y)

    # Complete logging and cleanup
    log_optimization_completion(logger, study, valid_pipeline_params)
    cleanup_logger(logger)

    return best_model, best_params, study


def grid_search_with_optuna(
    estimator,
    param_grid: dict,
    X, y,
    cv,
    random_state: int = 42,
    # Logging
    verbose: bool = False,
    log_to_file: bool = True,
    model_name: str = "grid_model"
):
    """
    Exhaustive grid search using Optuna's GridSampler.
    
    Evaluates ALL parameter combinations in the grid - no pruning, no model-specific 
    optimizations. Simple and straightforward grid search.
    
    Parameters:
    -----------
    param_grid : dict
        Grid of parameter values, e.g.: {"clf__C": [0.1, 1, 10], "select__k": [100, 200]}
        Will evaluate ALL combinations (in this example: 3 * 2 = 6 combinations)
    
    Returns same as optimize_with_optuna: (best_model, best_params, study)
    """
    
    # Configure Optuna logging verbosity
    if verbose:
        set_verbosity(INFO)
    else:
        set_verbosity(WARNING)

    # Calculate total number of trials (all grid combinations)
    n_trials = int(np.prod([len(values) for values in param_grid.values()]))
    
    # Setup session logging
    logger = setup_logging(model_name, verbose, log_to_file)
    log_optimization_start(logger, model_name, n_trials, "maximize", random_state, param_grid, "Grid")

    # Simple objective function - no model-specific optimizations needed for grid search
    def objective(trial):
        model = clone(estimator)
        
        # Set parameters from grid
        trial_params = {}
        for param_name in param_grid.keys():
            val = trial.suggest_categorical(param_name, param_grid[param_name])

            if param_name.endswith("class_weight") and isinstance(val, (int, float)):
                val = {0: 1, 1: int(val)}

            model.set_params(**{param_name: val})
            trial_params[param_name] = val
            
        #to surpress warnings with dicts and clf__class_weight
        if logger:
            logger.info(f"Trial {trial.number}: Testing parameters: {trial_params}")
        
        # Simple cross-validation evaluation, maximize f2 score ! 
        scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        mean_score = float(np.mean(scores))
        
        if logger:
            logger.info(f"Trial {trial.number}: CV F2 Score = {mean_score:.4f} (±{np.std(scores):.4f})")
        
        return mean_score

    # Configure GridSampler for exhaustive search
    sampler = GridSampler(param_grid,
                          seed=random_state)
    
    # Create study without pruner (grid search evaluates all combinations)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=NopPruner()

        # No pruner: we want to evaluate ALL combinations
    )
    
    # Run exhaustive grid search (will automatically stop after all combinations)
    study.optimize(objective)

    # Process and validate results
    best_params = study.best_params

    #to surpress warnings with dicts and clf__class_weight
    best_params = {
    k: ({0: 1, 1: int(v)} if k.endswith("class_weight") and isinstance(v, (int, float)) else v)
    for k, v in best_params.items()
}

    valid_pipeline_params = validate_and_filter_params(estimator, best_params)
    
    # Train final model with optimal parameters
    best_model = clone(estimator).set_params(**valid_pipeline_params)
    best_model.fit(X, y)

    # Complete logging and cleanup
    log_optimization_completion(logger, study, valid_pipeline_params)
    cleanup_logger(logger)

    return best_model, best_params, study
