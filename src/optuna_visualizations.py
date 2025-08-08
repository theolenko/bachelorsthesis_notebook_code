# optuna_visualizations.py
"""
Comprehensive visualization functions for Optuna study analysis.

This module provides standardized plotting functions for analyzing Optuna optimization studies,
including parameter distributions, convergence analysis, trial status, and performance correlations.
All functions work with Optuna study objects and generate publication-ready plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any


def plot_optuna_study_analysis(study, figsize_large=(18, 5), figsize_medium=(15, 5), figsize_small=(12, 5)):
    """
    Complete Optuna study analysis with all visualizations.
    
    Parameters:
    -----------
    study : optuna.Study
        The completed Optuna study object
    figsize_large : tuple, default=(18, 5)
        Figure size for 3-subplot plots
    figsize_medium : tuple, default=(15, 5)
        Figure size for 2-subplot plots
    figsize_small : tuple, default=(12, 5)
        Figure size for small plots
        
    Returns:
    --------
    None
        Displays all plots and prints analysis summary
        
    Notes:
    ------
    - Requires a completed Optuna study with trials
    - Automatically handles different parameter types (continuous, categorical, discrete)
    - Generates 6 different analysis sections with multiple plots each
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE OPTUNA STUDY ANALYSIS & VISUALIZATIONS")
    print("="*60)
    
    # 1. Basic Optimization History & Parameter Importance
    plot_optimization_history_and_importance(study, figsize=figsize_medium)
    
    # 2. Parameter Value Distributions
    plot_parameter_distributions(study, figsize=figsize_large)
    
    # 3. Performance vs Parameter Correlations
    plot_performance_correlations(study, figsize=figsize_large)
    
    # 4. Convergence Analysis
    plot_convergence_analysis(study, figsize=figsize_medium)
    
    # 5. Trial Status Analysis
    plot_trial_status_analysis(study, figsize=figsize_small)
    
    # 6. Best Trials Summary
    print_best_trials_summary(study, top_n=10)
    
    # 7. Study Statistics
    print_study_statistics(study)


def plot_optimization_history_and_importance(study, figsize=(15, 5)):
    """
    Plot optimization history and parameter importance.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    figsize : tuple, default=(15, 5)
        Figure size
    """
    print("\n1. Optimization History & Parameter Importance:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Optimization history
    trial_numbers = [trial.number for trial in study.trials]
    trial_values = [trial.value for trial in study.trials if trial.value is not None]
    ax1.plot(trial_numbers[:len(trial_values)], trial_values, alpha=0.7)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Optuna Optimization History')
    ax1.grid(True, alpha=0.3)
    
    # Parameter importance
    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
        if importance:
            params = list(importance.keys())
            values = list(importance.values())
            ax2.barh(params, values)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance')
        else:
            ax2.text(0.5, 0.5, 'No parameter importance\ndata available', 
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Parameter Importance (N/A)')
    except Exception as e:
        ax2.text(0.5, 0.5, 'Parameter importance\ncalculation failed', 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Parameter Importance (Error)')
    
    plt.tight_layout()
    plt.show()


def plot_parameter_distributions(study, figsize=(18, 5)):
    """
    Plot distributions of parameter values tested during optimization.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    figsize : tuple, default=(18, 5)
        Figure size
    """
    print("\n2. Parameter Value Distributions:")
    
    # Extract all parameter names from successful trials
    successful_trials = [trial for trial in study.trials if trial.value is not None]
    if not successful_trials:
        print("No successful trials found for parameter distribution analysis.")
        return
    
    # Get all unique parameter names
    all_param_names = set()
    for trial in successful_trials:
        all_param_names.update(trial.params.keys())
    
    param_names = sorted(list(all_param_names))
    n_params = len(param_names)
    
    if n_params == 0:
        print("No parameters found in trials.")
        return
    
    # Create subplots (max 3 per row)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        # Extract parameter values
        param_values = []
        for trial in successful_trials:
            if param_name in trial.params:
                value = trial.params[param_name]
                # Handle different parameter types
                if isinstance(value, dict):
                    # For class_weight parameters, extract minority class weight
                    if 1 in value:
                        param_values.append(value[1])
                    else:
                        param_values.append(1)  # balanced case
                else:
                    param_values.append(value)
        
        if param_values:
            # First check if we have boolean values specifically
            has_boolean = any(isinstance(val, bool) for val in param_values)
            
            if has_boolean:
                # Treat as categorical even if it could be converted to float
                value_counts = pd.Series(param_values).value_counts()
                categories = [str(cat) for cat in value_counts.index]
                x_positions = range(len(categories))
                
                axes[i].bar(x_positions, value_counts.values, alpha=0.7, edgecolor='black')
                axes[i].set_xticks(x_positions)
                axes[i].set_xticklabels(categories)
            else:
                # Determine if parameter is numeric (excluding booleans)
                numeric_values = []
                for val in param_values:
                    try:
                        if not isinstance(val, bool):  # Explicitly exclude booleans
                            numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        break
                
                if len(numeric_values) == len(param_values):
                    # Numeric parameter - histogram
                    # Use log scale for specific parameters that need it
                    log_scale_params = ['clf__alpha', 'clf__C', 'alpha', 'C', 'clf__learning_rate']
                    if param_name in log_scale_params:
                       axes[i].set_xscale('log')
                       # Distribute bins evenly in log space
                       bins = np.logspace(
                           np.log10(min(numeric_values)),
                           np.log10(max(numeric_values)),
                           30  # fixed number of bins
                       )
                       axes[i].hist(numeric_values, bins=bins, alpha=0.7, 
                                  edgecolor='black')
                    else:
                        # Other numeric parameters
                        bins = min(30, len(set(numeric_values)))
                        axes[i].hist(numeric_values, bins=bins, alpha=0.7, 
                                   edgecolor='black')
                else:
                    # Categorical parameter - bar plot
                    value_counts = pd.Series(param_values).value_counts()
                    categories = [str(cat) for cat in value_counts.index]
                    x_positions = range(len(categories))
                    
                    axes[i].bar(x_positions, value_counts.values, alpha=0.7, edgecolor='black')
                    axes[i].set_xticks(x_positions)
                    axes[i].set_xticklabels(categories)
        
        axes[i].set_xlabel(param_name.replace('__', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution: {param_name}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_performance_correlations(study, figsize=(18, 5)):
    """
    Plot performance vs individual parameter values.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    figsize : tuple, default=(18, 5)
        Figure size
    """
    print("\n3. Performance vs Individual Parameters:")
    
    successful_trials = [trial for trial in study.trials if trial.value is not None]
    if not successful_trials:
        print("No successful trials found for correlation analysis.")
        return
    
    # Get parameter names and objective values
    all_param_names = set()
    for trial in successful_trials:
        all_param_names.update(trial.params.keys())
    
    param_names = sorted(list(all_param_names))
    scores = [trial.value for trial in successful_trials]
    
    n_params = len(param_names)
    if n_params == 0:
        print("No parameters found in trials.")
        return
    
    # Create subplots
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        # Extract parameter values
        param_values = []
        param_scores = []
        
        for j, trial in enumerate(successful_trials):
            if param_name in trial.params:
                value = trial.params[param_name]
                # Handle different parameter types
                if isinstance(value, dict):
                    # For class_weight parameters
                    if 1 in value:
                        param_values.append(value[1])
                    else:
                        param_values.append(1)  # balanced case
                else:
                    param_values.append(value)
                param_scores.append(scores[j])
        
        if param_values and param_scores:
            # Check if numeric
            try:
                numeric_values = [float(val) for val in param_values]
                axes[i].scatter(numeric_values, param_scores, alpha=0.6)
                
                # Use log scale for specific parameters that need it
                log_scale_params = ['clf__alpha', 'clf__C', 'alpha', 'C', 'clf__learning_rate']
                if param_name in log_scale_params:
                    axes[i].set_xscale('log')
                    
            except (ValueError, TypeError):
                # Categorical parameter - create proper categorical scatter plot
                unique_vals = list(set(param_values))
                # Sort boolean values for better display
                if all(isinstance(v, bool) or str(v).lower() in ['true', 'false'] for v in unique_vals):
                    unique_vals = sorted(unique_vals, key=lambda x: str(x))
                
                x_positions = []
                y_values = []
                labels = []
                
                for k, val in enumerate(unique_vals):
                    val_scores = [s for v, s in zip(param_values, param_scores) if v == val]
                    x_positions.extend([k] * len(val_scores))
                    y_values.extend(val_scores)
                    labels.append(str(val))
                
                axes[i].scatter(x_positions, y_values, alpha=0.6)
                axes[i].set_xticks(range(len(unique_vals)))
                axes[i].set_xticklabels(labels)
                
                # Add jitter for better visibility
                if len(unique_vals) <= 5:
                    for k, val in enumerate(unique_vals):
                        val_scores = [s for v, s in zip(param_values, param_scores) if v == val]
                        if len(val_scores) > 1:
                            jitter = np.random.normal(0, 0.05, len(val_scores))
                            axes[i].scatter([k] * len(val_scores) + jitter, val_scores, alpha=0.6)
        
        axes[i].set_xlabel(param_name.replace('__', ' ').title())
        axes[i].set_ylabel('Objective Score')
        axes[i].set_title(f'Score vs {param_name}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_convergence_analysis(study, figsize=(15, 5)):
    """
    Plot convergence analysis showing best score over time and rolling average.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    figsize : tuple, default=(15, 5)
        Figure size
    """
    print("\n4. Convergence Analysis:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    successful_trials = [trial for trial in study.trials if trial.value is not None]
    scores = [trial.value for trial in successful_trials]
    
    # Running best score
    running_best = []
    current_best = -np.inf if study.direction.name == 'MAXIMIZE' else np.inf
    
    for trial in study.trials:
        if trial.value is not None:
            if study.direction.name == 'MAXIMIZE':
                if trial.value > current_best:
                    current_best = trial.value
            else:
                if trial.value < current_best:
                    current_best = trial.value
        
        if current_best == -np.inf or current_best == np.inf:
            running_best.append(np.nan)
        else:
            running_best.append(current_best)
    
    ax1.plot(range(len(running_best)), running_best, 'b-', linewidth=2)
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Best Score So Far')
    ax1.set_title('Convergence: Best Score Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Rolling average
    window_size = min(50, max(10, len(scores) // 10))
    if len(scores) > window_size:
        rolling_mean = pd.Series(scores).rolling(window=window_size, center=True).mean()
        ax2.plot(range(len(rolling_mean)), rolling_mean, 'g-', linewidth=2)
        ax2.scatter(range(len(scores)), scores, alpha=0.3, s=10)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel(f'Objective Score (Rolling Mean, window={window_size})')
        ax2.set_title('Optimization Progress: Rolling Average')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f'Need >{window_size} trials\nfor rolling average', 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Rolling Average (Insufficient Data)')
    
    plt.tight_layout()
    plt.show()


def plot_trial_status_analysis(study, figsize=(12, 5)):
    """
    Plot trial status distribution.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    figsize : tuple, default=(12, 5)
        Figure size
    """
    print("\n5. Trial Status Overview:")
    
    trial_states = {}
    for trial in study.trials:
        state = trial.state.name
        trial_states[state] = trial_states.get(state, 0) + 1
    
    if not trial_states:
        print("No trial status data available.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(trial_states)))
    ax1.pie(trial_states.values(), labels=trial_states.keys(), autopct='%1.1f%%', colors=colors)
    ax1.set_title('Trial Status Distribution')
    
    # Bar chart
    bars = ax2.bar(trial_states.keys(), trial_states.values(), color=colors)
    ax2.set_ylabel('Number of Trials')
    ax2.set_title('Trial Status Counts')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def print_best_trials_summary(study, top_n=10):
    """
    Print summary of best trials with their parameters.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    top_n : int, default=10
        Number of top trials to display
    """
    print(f"\n6. Top {top_n} Best Trials:")
    
    valid_trials = [trial for trial in study.trials if trial.value is not None]
    if not valid_trials:
        print("No successful trials found.")
        return
    
    # Sort trials by objective value
    reverse_sort = study.direction.name == 'MAXIMIZE'
    top_trials = sorted(valid_trials, key=lambda x: x.value, reverse=reverse_sort)[:top_n]
    
    if not top_trials:
        print("No trials to display.")
        return
    
    # Get all parameter names for consistent formatting
    all_params = set()
    for trial in top_trials:
        all_params.update(trial.params.keys())
    param_names = sorted(list(all_params))
    
    # Print header
    header = f"{'Rank':<4} {'Trial#':<7} {'Score':<10}"
    for param in param_names:
        short_name = param.split('__')[-1] if '__' in param else param
        header += f" {short_name:<12}"
    print(header)
    print("-" * len(header))
    
    # Print trials
    for i, trial in enumerate(top_trials, 1):
        row = f"{i:<4} {trial.number:<7} {trial.value:<10.4f}"
        
        for param in param_names:
            value = trial.params.get(param, 'N/A')
            if isinstance(value, dict):
                # Handle class_weight parameters
                if 1 in value:
                    display_val = f"{value[1]:.2f}" if isinstance(value[1], float) else str(value[1])
                else:
                    display_val = "balanced"
            elif isinstance(value, float):
                display_val = f"{value:.4f}"
            else:
                display_val = str(value)
            
            row += f" {display_val:<12}"
        
        print(row)


def print_study_statistics(study):
    """
    Print comprehensive study statistics.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    """
    print(f"\n7. Study Statistics Summary:")
    print(f"{'='*40}")
    
    total_trials = len(study.trials)
    successful_trials = len([t for t in study.trials if t.value is not None])
    failed_trials = len([t for t in study.trials if t.value is None])
    
    print(f"• Total trials: {total_trials}")
    print(f"• Successful trials: {successful_trials}")
    print(f"• Failed trials: {failed_trials}")
    
    if successful_trials > 0:
        success_rate = (successful_trials / total_trials) * 100
        print(f"• Success rate: {success_rate:.1f}%")
        
        # Best trial info
        if study.best_trial:
            print(f"• Best trial: #{study.best_trial.number}")
            print(f"• Best objective value: {study.best_value:.4f}")
            
        # Score statistics
        scores = [t.value for t in study.trials if t.value is not None]
        if scores:
            print(f"• Score statistics:")
            print(f"  - Mean: {np.mean(scores):.4f}")
            print(f"  - Std:  {np.std(scores):.4f}")
            print(f"  - Min:  {np.min(scores):.4f}")
            print(f"  - Max:  {np.max(scores):.4f}")
    
    print(f"• Study direction: {study.direction.name}")
    print(f"• Sampler: {study.sampler.__class__.__name__}")


# Convenience function for quick analysis
def quick_optuna_analysis(study, show_distributions=True, show_correlations=True, 
                         show_convergence=True, show_status=True, show_summary=True):
    """
    Quick analysis with selected visualizations.
    
    Parameters:
    -----------
    study : optuna.Study
        The Optuna study object
    show_distributions : bool, default=True
        Show parameter distributions
    show_correlations : bool, default=True
        Show performance correlations
    show_convergence : bool, default=True
        Show convergence analysis
    show_status : bool, default=True
        Show trial status
    show_summary : bool, default=True
        Show best trials summary
    """
    print("\n" + "="*50)
    print("QUICK OPTUNA STUDY ANALYSIS")
    print("="*50)
    
    # Always show basic history and importance
    plot_optimization_history_and_importance(study)
    
    if show_distributions:
        plot_parameter_distributions(study)
    
    if show_correlations:
        plot_performance_correlations(study)
    
    if show_convergence:
        plot_convergence_analysis(study)
    
    if show_status:
        plot_trial_status_analysis(study)
    
    if show_summary:
        print_best_trials_summary(study, top_n=5)
        print_study_statistics(study)
