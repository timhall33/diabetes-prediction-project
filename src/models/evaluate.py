"""
Model evaluation utilities for diabetes prediction.

This module provides functions for:
- Computing classification and regression metrics
- Generating confusion matrices and classification reports
- Creating publication-quality evaluation visualizations
- Comparing models across feature sets
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

# Set publication-quality defaults
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
})

# Diabetes status labels
DIABETES_LABELS = {0: 'No Diabetes', 1: 'Prediabetes', 2: 'Diabetes'}
DIABETES_COLORS = {0: '#2166ac', 1: '#fdb863', 2: '#b2182b'}


# =============================================================================
# Classification Metrics
# =============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Predicted probabilities (n_samples, n_classes)
    class_names : dict, optional
        Mapping of class indices to names

    Returns
    -------
    dict
        Dictionary of metrics
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }

    # Per-class metrics
    for cls, name in class_names.items():
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        metrics[f'precision_{name}'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'recall_{name}'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics[f'f1_{name}'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # ROC AUC (one-vs-rest)
    if y_prob is not None:
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except ValueError:
            # Handle case where not all classes are present
            metrics['roc_auc_ovr'] = np.nan
            metrics['roc_auc_ovo'] = np.nan

    # Confusion matrix (store as nested dict for JSON serialization)
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def get_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Get classification report as a formatted DataFrame.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : dict, optional
        Mapping of class indices to names

    Returns
    -------
    pd.DataFrame
        Formatted classification report
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    report = classification_report(
        y_true, y_pred,
        target_names=[class_names[i] for i in sorted(class_names.keys())],
        output_dict=True,
    )

    df = pd.DataFrame(report).T
    df = df.round(3)
    return df


# =============================================================================
# Regression Metrics
# =============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    dict
        Dictionary of metrics
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle any NaN/inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
    }

    # MAPE (handle zeros)
    mask_nonzero = y_true != 0
    if mask_nonzero.sum() > 0:
        metrics['mape'] = mean_absolute_percentage_error(y_true[mask_nonzero], y_pred[mask_nonzero])
    else:
        metrics['mape'] = np.nan

    # Additional metrics
    residuals = y_true - y_pred
    metrics['residual_mean'] = residuals.mean()
    metrics['residual_std'] = residuals.std()
    metrics['max_error'] = np.abs(residuals).max()

    return metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot confusion matrix as heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : dict, optional
        Mapping of class indices to names
    normalize : bool
        Whether to normalize by row (true labels)
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    cmap : str
        Colormap name
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    labels = [class_names[i] for i in sorted(class_names.keys())]

    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    return ax


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = 'ROC Curves (One-vs-Rest)',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot ROC curves for each class (one-vs-rest).

    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities (n_samples, n_classes)
    class_names : dict, optional
        Mapping of class indices to names
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_classes = y_prob.shape[1]

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        auc = roc_auc_score(y_true_binary, y_prob[:, i])

        label = f'{class_names[i]} (AUC = {auc:.3f})'
        ax.plot(fpr, tpr, label=label, color=DIABETES_COLORS[i], linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return ax


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    title: str = 'Precision-Recall Curves',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot precision-recall curves for each class.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities (n_samples, n_classes)
    class_names : dict, optional
        Mapping of class indices to names
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    n_classes = y_prob.shape[1]

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])

        # Calculate prevalence (baseline)
        prevalence = y_true_binary.mean()

        label = f'{class_names[i]} (Prevalence = {prevalence:.2f})'
        ax.plot(recall, precision, label=label, color=DIABETES_COLORS[i], linewidth=2)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_idx: int = 2,  # Default to diabetes class
    class_names: Optional[Dict[int, str]] = None,
    n_bins: int = 10,
    title: str = 'Calibration Curve',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot calibration curve for a specific class.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities (n_samples, n_classes)
    class_idx : int
        Class index to plot calibration for
    class_names : dict, optional
        Mapping of class indices to names
    n_bins : int
        Number of calibration bins
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if class_names is None:
        class_names = DIABETES_LABELS

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    y_true_binary = (y_true == class_idx).astype(int)
    prob_class = y_prob[:, class_idx]

    prob_true, prob_pred = calibration_curve(y_true_binary, prob_class, n_bins=n_bins)

    ax.plot(prob_pred, prob_true, 's-', color=DIABETES_COLORS[class_idx],
            label=f'{class_names[class_idx]}', linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{title} - {class_names[class_idx]}')
    ax.legend(loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Residual Analysis',
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot residual analysis for regression.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Predicted vs Actual
    ax = axes[0]
    ax.scatter(y_pred, y_true, alpha=0.3, s=10)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residuals vs Predicted
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted')
    ax.grid(True, alpha=0.3)

    # Residual distribution
    ax = axes[2]
    ax.hist(residuals, bins=50, edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='Zero')
    ax.axvline(x=residuals.mean(), color='g', linestyle='--', label=f'Mean ({residuals.mean():.3f})')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = 'Predicted vs Actual',
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Axes:
    """
    Plot predicted vs actual values for regression.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_pred, y_true, alpha=0.3, s=20)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)

    # Add R^2 annotation
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Predicted HbA1c (%)')
    ax.set_ylabel('Actual HbA1c (%)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    return ax


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    sort_by: str = 'f1_macro',
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Create comparison table of multiple models.

    Parameters
    ----------
    results : dict
        Dictionary of model_name -> metrics dict
    metrics : list, optional
        Metrics to include in table
    sort_by : str
        Metric to sort by
    ascending : bool
        Sort order

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    df = pd.DataFrame(results).T

    if metrics:
        df = df[metrics]

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    return df.round(4)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot bar chart comparing models across metrics.

    Parameters
    ----------
    results : dict
        Dictionary of model_name -> metrics dict
    metrics : list, optional
        Metrics to compare
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = ['accuracy', 'f1_macro', 'roc_auc_ovr']

    df = pd.DataFrame(results).T
    df = df[[m for m in metrics if m in df.columns]]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df.index))
    width = 0.8 / len(df.columns)

    for i, metric in enumerate(df.columns):
        offset = width * (i - len(df.columns) / 2 + 0.5)
        bars = ax.bar(x + offset, df[metric], width, label=metric, alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def plot_feature_set_comparison(
    results_with_labs: Dict[str, Dict[str, float]],
    results_without_labs: Dict[str, Dict[str, float]],
    metric: str = 'f1_macro',
    title: str = 'With Labs vs Without Labs Comparison',
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Compare model performance between with-labs and without-labs feature sets.

    Parameters
    ----------
    results_with_labs : dict
        Results for with-labs models
    results_without_labs : dict
        Results for without-labs models
    metric : str
        Metric to compare
    title : str
        Plot title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(results_with_labs.keys())

    with_labs_scores = [results_with_labs[m].get(metric, 0) for m in models]
    without_labs_scores = [results_without_labs[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, with_labs_scores, width, label='With Labs', color='#2166ac', alpha=0.8)
    bars2 = ax.bar(x + width/2, without_labs_scores, width, label='Without Labs', color='#b2182b', alpha=0.8)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


# =============================================================================
# Full Evaluation Report
# =============================================================================

def generate_classification_report_figures(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Model',
    save_dir: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12),
) -> plt.Figure:
    """
    Generate comprehensive classification evaluation figure.

    Parameters
    ----------
    model : BaseEstimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name of the model for title
    save_dir : str, optional
        Directory to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, ax=axes[0, 0], title=f'{model_name} - Confusion Matrix')

    # Normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, ax=axes[0, 1], normalize=True,
                         title=f'{model_name} - Normalized Confusion Matrix')

    # ROC curves
    if y_prob is not None:
        plot_roc_curves(y_test, y_prob, ax=axes[1, 0], title=f'{model_name} - ROC Curves')
        plot_precision_recall_curves(y_test, y_prob, ax=axes[1, 1], title=f'{model_name} - PR Curves')
    else:
        axes[1, 0].text(0.5, 0.5, 'Probabilities not available', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'Probabilities not available', ha='center', va='center')

    plt.tight_layout()

    if save_dir:
        save_path = Path(save_dir) / f'{model_name.lower().replace(" ", "_")}_evaluation.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_regression_report_figures(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = 'Model',
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Generate comprehensive regression evaluation figure.

    Parameters
    ----------
    model : BaseEstimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name of the model for title
    save_dir : str, optional
        Directory to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_pred = model.predict(X_test)

    fig = plot_residuals(y_test, y_pred, title=f'{model_name} - Residual Analysis')

    if save_dir:
        save_path = Path(save_dir) / f'{model_name.lower().replace(" ", "_")}_residuals.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
