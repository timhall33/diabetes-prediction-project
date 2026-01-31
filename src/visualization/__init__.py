"""Visualization utilities for EDA and model evaluation."""

from .eda import (
    # Style configuration
    set_publication_style,
    get_diabetes_palette,
    DIABETES_COLORS,
    DIABETES_LABELS,

    # Target distribution
    plot_target_distribution,
    plot_cohort_flow,

    # Feature distributions
    plot_feature_by_status,
    plot_feature_panel,

    # Correlation analysis
    plot_correlation_heatmap,
    plot_top_correlations,

    # Interaction plots
    plot_scatter_by_status,
    plot_interaction_grid,

    # Temporal analysis
    plot_prevalence_by_year,
    plot_feature_by_year,

    # Dimensionality reduction
    plot_pca,

    # Risk factors
    plot_risk_factors,
    calculate_effect_sizes,
)

__all__ = [
    'set_publication_style',
    'get_diabetes_palette',
    'DIABETES_COLORS',
    'DIABETES_LABELS',
    'plot_target_distribution',
    'plot_cohort_flow',
    'plot_feature_by_status',
    'plot_feature_panel',
    'plot_correlation_heatmap',
    'plot_top_correlations',
    'plot_scatter_by_status',
    'plot_interaction_grid',
    'plot_prevalence_by_year',
    'plot_feature_by_year',
    'plot_pca',
    'plot_risk_factors',
    'calculate_effect_sizes',
]
