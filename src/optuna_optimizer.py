import optuna
from optuna.pruners import MedianPruner
from optuna.logging import set_verbosity, INFO, WARNING
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, fbeta_score
import logging
import os
from datetime import datetime


def sample_class_weight_value(trial, low=1, high=100, param_name="positive_class_weight"):
    """
    Custom function to sample class_weight with flexible range
    
    Parameters:
    -----------
    trial : optuna.Trial
        The Optuna trial object
    low : int, default=1
        Lower bound for positive class weight
    high : int, default=100
        Upper bound for positive class weight
    param_name : str, default="positive_class_weight"
        Name of the parameter for Optuna tracking
        
    Returns:
    --------
    dict : {0: 1, 1: positive_weight}
        Class weight dictionary for sklearn
    """
    pos_weight = trial.suggest_int(param_name, low, high)
    return {0: 1, 1: pos_weight}


def optimize_with_optuna(
    estimator,
    param_space: dict,
    X, y,
    cv,
    n_trials: int = 500,
    direction: str = "maximize",
    n_jobs: int = 1,
    random_state: int = 42,
    # Pruner parameters
    n_startup_trials: int = 5,
    n_warmup_steps: int = 0,
    # Logging
    verbose: bool = False,
    log_to_file: bool = True,
    model_name: str = "optuna_model"
):
    """
    Generic Optuna optimization for arbitrary sklearn estimators/pipelines. Optimize fpr max f2 score

    Arguments:
    ----------
    estimator          : sklearn estimator or Pipeline (unfitted)
    param_space        : dict, mapping param_name -> sampler function
                         e.g. {
                           "clf__C": lambda tr: tr.suggest_float("clf__C", 1e-3, 1e2, log=True),
                           "select__k": lambda tr: tr.suggest_int("select__k", 100, 5000),
                           "clf__class_weight": lambda tr: create_class_weight(tr),
                           ...
                         }
    X, y               : Your data
    cv                 : Cross-validation splitter, e.g. StratifiedKFold(...)
    n_trials           : Number of Optuna trials
    direction          : "maximize" or "minimize"
    n_jobs             : Parallel jobs for CV
    random_state       : Optuna seed
    n_startup_trials   : Pruner: how many trials before pruning starts
    n_warmup_steps     : Pruner: how many CV splits to ignore
    verbose            : Optuna logging INFO on/off and console output control
    log_to_file        : Whether to save optimization log to file (append mode)
    model_name         : Name for the log file (e.g., "tfidf_logreg_basic")

    Returns:
    --------
    best_model  : Copy of `estimator` fitted on (X,y) with best params  
    best_params : The parameters that Optuna found  
    study       : The complete optuna.Study object for visualizations etc.  
    
    Example:
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from sklearn.feature_selection import SelectKBest, chi2
    >>> from sklearn.linear_model import LogisticRegression
    >>> 
    >>> # Create pipeline
    >>> pipeline = Pipeline([
    ...     ("tfidf", TfidfVectorizer()),
    ...     ("select", SelectKBest(score_func=chi2)),
    ...     ("clf", LogisticRegression(solver="liblinear", random_state=42))
    ... ])
    >>> 
    >>> # Define parameter space
    >>> param_space = {
    ...     "clf__C": lambda tr: tr.suggest_float("clf__C", 1e-3, 1e3, log=True),
    ...     "select__k": lambda tr: tr.suggest_int("select__k", 50, 500),
    ...     "clf__class_weight": lambda tr: create_class_weight(tr)
    ... }
    >>> 
    >>> # Optimize
    >>> best_model, best_params, study = optimize_with_optuna(
    ...     pipeline, param_space, X_train, y_train, cv=cv
    ... )
    """

    # Optional: Optuna logging
    if verbose:
        set_verbosity(INFO)
    else:
        set_verbosity(WARNING)  # Suppress most output

    # Setup file logging if requested
    log_file_path = None
    if log_to_file:
        # Use simple naming scheme that allows appending for retraining
        log_file_path = f"optuna_log_{model_name}.txt"
        
        # Create custom logger for this optimization
        logger = logging.getLogger(f"optuna_{model_name}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler with append mode to continue writing to existing files
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
        
        # Log initial setup with session separator
        logger.info("=" * 80)
        logger.info(f"NEW OPTIMIZATION SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Starting Optuna optimization for model: {model_name}")
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Direction: {direction}")
        logger.info(f"Random state: {random_state}")
        logger.info(f"Parameter space: {list(param_space.keys())}")
        logger.info("-" * 50)

    # Our F2 scorer
    f2_scorer = make_scorer(fbeta_score, beta=2)

    def objective(trial):
        model = clone(estimator)
        
        # Collect parameters for logging
        trial_params = {}
        
        # Suggest and set parameters
        for name, sampler in param_space.items():
            param_value = sampler(trial)
            model.set_params(**{name: param_value})
            trial_params[name] = param_value
        
        # Log trial parameters if file logging is enabled
        if log_to_file and 'logger' in locals():
            logger.info(f"Trial {trial.number}: Testing parameters: {trial_params}")
        
        # CV evaluation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=f2_scorer,
            n_jobs=n_jobs
        )
        
        mean_score = scores.mean()
        
        # Log trial results if file logging is enabled
        if log_to_file and 'logger' in locals():
            logger.info(f"Trial {trial.number}: CV F2 Score = {mean_score:.4f} (±{scores.std():.4f})")
        
        # Optional pruning after each CV fold
        trial.report(mean_score, step=0)
        if trial.should_prune():
            if log_to_file and 'logger' in locals():
                logger.info(f"Trial {trial.number}: Pruned")
            raise optuna.TrialPruned()
        
        return mean_score

    # Configure sampler & pruner
    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner  = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps
    )
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner
    )
    # Optimize!
    study.optimize(objective, n_trials=n_trials)

    # Take best result and fit pipeline finally
    best_params = study.best_params
    
    # Log optimization completion if file logging is enabled
    if log_to_file and 'logger' in locals():
        logger.info("-" * 50)
        logger.info(f"Optimization completed after {len(study.trials)} trials")
        logger.info(f"Best F2 score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        logger.info(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        logger.info("-" * 50)
    
    # Filter parameters to only include valid pipeline parameters
    # This prevents errors when lambda functions create additional internal parameters
    valid_pipeline_params = {}
    estimator_params = clone(estimator).get_params().keys()
    
    for param_name, param_value in best_params.items():
        # Only include parameters that are actually valid for the pipeline
        if param_name in estimator_params:
            valid_pipeline_params[param_name] = param_value
        # Also include nested parameters (e.g., clf__C, select__k)
        elif any(param_name.startswith(step + '__') for step, _ in clone(estimator).steps):
            valid_pipeline_params[param_name] = param_value
    
    best_model = clone(estimator).set_params(**valid_pipeline_params)
    best_model.fit(X, y)

    # Final logging and cleanup
    if log_to_file and 'logger' in locals():
        logger.info(f"Model training completed with best parameters: {valid_pipeline_params}")
        logger.info(f"SESSION COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("")  # Empty line for session separation
        
        # Cleanup handlers to prevent memory leaks
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    return best_model, best_params, study