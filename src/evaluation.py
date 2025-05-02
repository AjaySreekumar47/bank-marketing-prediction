"""
Bank Marketing Term Deposit Prediction - Model Evaluation

This module contains functions for evaluating model performance
and visualizing results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate model performance on test data.
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    class_names : list, optional
        Names of classes
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Class names
    if class_names is None:
        class_names = ['No', 'Yes']
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # AUC scores
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6), cmap='Blues', normalize=False):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of classes
    figsize : tuple, optional
        Figure size
    cmap : str, optional
        Colormap for the plot
    normalize : bool, optional
        Whether to normalize the confusion matrix
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if class_names is None:
        class_names = ['No', 'Yes']
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    
    # Set labels
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_roc_curve(y_test, y_prob, figsize=(8, 6)):
    """
    Plot the ROC curve.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    return fig


def plot_precision_recall_curve(y_test, y_prob, figsize=(8, 6)):
    """
    Plot the Precision-Recall curve.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Set labels and limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True)
    
    return fig


def plot_feature_importance(feature_importance, top_n=20, figsize=(12, 8)):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    feature_importance : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int, optional
        Number of top features to plot
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Sort and get top N features
    top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importance
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
    
    # Set labels
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    return fig


def plot_threshold_metrics(y_test, y_prob, thresholds=None, figsize=(10, 6)):
    """
    Plot precision, recall, and F1 score as a function of threshold.
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    thresholds : array-like, optional
        Thresholds to evaluate. If None, 100 thresholds between 0 and 1 are used.
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    
    # Initialize arrays for metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metrics
    ax.plot(thresholds, precision_scores, label='Precision')
    ax.plot(thresholds, recall_scores, label='Recall')
    ax.plot(thresholds, f1_scores, label='F1 Score')
    
    # Find optimal threshold for F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Add vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, color='r', linestyle='--')
    ax.text(optimal_threshold+0.02, 0.5, f'Optimal Threshold = {optimal_threshold:.2f}', 
            rotation=0, verticalalignment='center')
    
    # Set labels and limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall, and F1 Score vs. Threshold')
    ax.legend(loc="best")
    ax.grid(True)
    
    return fig, optimal_threshold


def compare_models(models_dict, X_test, y_test, metric='roc_auc', figsize=(10, 8)):
    """
    Compare multiple models using specified metric.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model names and trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    metric : str, optional
        Metric to use for comparison: 'roc_auc' or 'pr_auc'
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, model in models_dict.items():
        # Get predictions
        y_prob = model.predict_proba(X_test)[:, 1]
        
        if metric == 'roc_auc':
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
            
            if name == list(models_dict.keys())[0]:
                # Add diagonal line (random classifier)
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                
            # Set labels
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curves')
            
        elif metric == 'pr_auc':
            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            auc_score = average_precision_score(y_test, y_prob)
            
            # Plot precision-recall curve
            ax.plot(recall, precision, lw=2, label=f'{name} (AUC = {auc_score:.3f})')
            
            # Set labels
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves')
    
    # Set limits and add grid
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.grid(True)
    
    return fig


def calibration_plot(y_test, y_prob, n_bins=10, figsize=(8, 6)):
    """
    Create a calibration plot (reliability diagram).
    
    Parameters:
    -----------
    y_test : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int, optional
        Number of bins
    figsize : tuple, optional
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create bins and find bin edges
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.minimum(bin_indices, n_bins - 1)  # Ensure we don't go out of bounds
    
    # Calculate fraction of positives and mean predicted probability in each bin
    bin_sums = np.bincount(bin_indices, weights=y_test, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)
    
    # Handle divide by zero
    nonzero = bin_counts > 0
    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)
    
    fraction_of_positives[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    mean_predicted_value[nonzero] = bin_probs[nonzero] / bin_counts[nonzero]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot perfectly calibrated line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    
    # Set labels and limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration plot (Reliability diagram)')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig
