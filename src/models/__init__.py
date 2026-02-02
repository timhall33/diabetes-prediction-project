"""Model training, evaluation, and tuning utilities."""

from .train import (
    # Data splitting
    split_data,
    get_class_weights,

    # Model definitions
    get_baseline_models,
    get_advanced_models,

    # Training
    train_with_cv,

    # MLflow tracking
    setup_mlflow,
    log_experiment,

    # Hyperparameter tuning
    create_optuna_objective,
    tune_model,

    # Model I/O
    save_model,
    load_model,

    # Constants
    LIGHTGBM_AVAILABLE,
)

from .evaluate import (
    # Classification metrics
    compute_classification_metrics,
    get_classification_report_df,

    # Regression metrics
    compute_regression_metrics,

    # Visualization
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_calibration_curve,
    plot_residuals,
    plot_predicted_vs_actual,

    # Model comparison
    compare_models_table,
    plot_model_comparison,
    plot_feature_set_comparison,

    # Full reports
    generate_classification_report_figures,
    generate_regression_report_figures,

    # Constants
    DIABETES_LABELS,
    DIABETES_COLORS,
)

__all__ = [
    # Training
    'split_data',
    'get_class_weights',
    'get_baseline_models',
    'get_advanced_models',
    'train_with_cv',
    'setup_mlflow',
    'log_experiment',
    'create_optuna_objective',
    'tune_model',
    'save_model',
    'load_model',
    'LIGHTGBM_AVAILABLE',

    # Evaluation
    'compute_classification_metrics',
    'get_classification_report_df',
    'compute_regression_metrics',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_calibration_curve',
    'plot_residuals',
    'plot_predicted_vs_actual',
    'compare_models_table',
    'plot_model_comparison',
    'plot_feature_set_comparison',
    'generate_classification_report_figures',
    'generate_regression_report_figures',
    'DIABETES_LABELS',
    'DIABETES_COLORS',
]
