"""
EDA Visualization Module for Diabetes Prediction Project.

Publication-quality visualizations for exploring NHANES data.
Uses a consistent, colorblind-friendly color palette.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from scipy import stats

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Colorblind-friendly palette for diabetes status
DIABETES_COLORS = {
    0: '#2166ac',  # Blue - No Diabetes
    1: '#fdb863',  # Orange - Prediabetes
    2: '#b2182b',  # Red - Diabetes
}

DIABETES_LABELS = {
    0: 'No Diabetes',
    1: 'Prediabetes',
    2: 'Diabetes',
}

# Alternative categorical palette
CATEGORY_PALETTE = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377']

# Figure defaults
FIGURE_DPI = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': FIGURE_DPI,
        'font.size': TICK_FONTSIZE,
        'axes.titlesize': TITLE_FONTSIZE,
        'axes.labelsize': LABEL_FONTSIZE,
        'xtick.labelsize': TICK_FONTSIZE,
        'ytick.labelsize': TICK_FONTSIZE,
        'legend.fontsize': LEGEND_FONTSIZE,
        'figure.titlesize': TITLE_FONTSIZE + 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def get_diabetes_palette():
    """Return list of colors in order [No Diabetes, Prediabetes, Diabetes]."""
    return [DIABETES_COLORS[0], DIABETES_COLORS[1], DIABETES_COLORS[2]]


# =============================================================================
# TARGET DISTRIBUTION VISUALIZATIONS
# =============================================================================

def plot_target_distribution(
    y: pd.Series,
    title: str = 'Diabetes Status Distribution',
    figsize: Tuple[int, int] = (10, 6),
    show_percentages: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a horizontal bar chart showing diabetes status distribution.

    Parameters
    ----------
    y : pd.Series
        Target variable (0, 1, 2 for diabetes status)
    title : str
        Figure title
    figsize : tuple
        Figure dimensions
    show_percentages : bool
        Whether to show percentage labels
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    # Calculate counts and percentages
    counts = y.value_counts().sort_index()
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bars
    bars = ax.barh(
        [DIABETES_LABELS[i] for i in counts.index],
        counts.values,
        color=[DIABETES_COLORS[i] for i in counts.index],
        edgecolor='white',
        linewidth=1.5
    )

    # Add count and percentage labels
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        width = bar.get_width()
        label = f'{count:,} ({pct}%)' if show_percentages else f'{count:,}'
        ax.text(
            width + total * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            fontsize=LABEL_FONTSIZE,
            fontweight='bold'
        )

    ax.set_xlabel('Number of Participants', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold', pad=20)
    ax.set_xlim(0, max(counts.values) * 1.2)

    # Add total annotation
    ax.text(
        0.98, 0.02,
        f'Total N = {total:,}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=TICK_FONTSIZE,
        style='italic'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_cohort_flow(
    steps: List[Dict],
    title: str = 'Study Population Selection',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a cohort flow diagram showing inclusion/exclusion steps.

    Parameters
    ----------
    steps : list of dict
        Each dict has keys: 'label', 'n', 'excluded' (optional), 'reason' (optional)
        Example: [
            {'label': 'NHANES 2015-2018', 'n': 19225},
            {'label': 'Adults (≥18 years)', 'n': 11878, 'excluded': 7347, 'reason': 'Age < 18'},
            ...
        ]
    title : str
        Figure title
    figsize : tuple
        Figure dimensions
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) + 1)
    ax.axis('off')

    box_width = 4
    box_height = 0.7
    x_center = 5

    for i, step in enumerate(steps):
        y = len(steps) - i

        # Main box
        rect = mpatches.FancyBboxPatch(
            (x_center - box_width/2, y - box_height/2),
            box_width,
            box_height,
            boxstyle='round,pad=0.05,rounding_size=0.1',
            facecolor='#f0f0f0' if i == 0 else '#e6f2ff',
            edgecolor='#333333',
            linewidth=2
        )
        ax.add_patch(rect)

        # Box text
        ax.text(
            x_center, y,
            f"{step['label']}\n(N = {step['n']:,})",
            ha='center', va='center',
            fontsize=LABEL_FONTSIZE,
            fontweight='bold' if i == len(steps) - 1 else 'normal'
        )

        # Arrow and exclusion box
        if i > 0 and 'excluded' in step:
            # Arrow down
            ax.annotate(
                '',
                xy=(x_center, y + box_height/2 + 0.05),
                xytext=(x_center, y + 1 - box_height/2 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2)
            )

            # Exclusion box to the right
            excl_x = x_center + box_width/2 + 1.5
            excl_y = y + 0.5

            excl_rect = mpatches.FancyBboxPatch(
                (excl_x - 1.2, excl_y - 0.25),
                2.4,
                0.5,
                boxstyle='round,pad=0.03,rounding_size=0.1',
                facecolor='#fff0f0',
                edgecolor='#cc0000',
                linewidth=1.5
            )
            ax.add_patch(excl_rect)

            reason = step.get('reason', 'Excluded')
            ax.text(
                excl_x, excl_y,
                f"−{step['excluded']:,}\n{reason}",
                ha='center', va='center',
                fontsize=TICK_FONTSIZE - 1,
                color='#cc0000'
            )

            # Arrow to exclusion box
            ax.annotate(
                '',
                xy=(excl_x - 1.2, excl_y),
                xytext=(x_center + box_width/2, y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#cc0000', lw=1.5, ls='--')
            )

    ax.set_title(title, fontsize=TITLE_FONTSIZE + 2, fontweight='bold', pad=20, y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


# =============================================================================
# FEATURE DISTRIBUTION BY STATUS
# =============================================================================

def plot_feature_by_status(
    df: pd.DataFrame,
    feature: str,
    target: str = 'DIABETES_STATUS',
    plot_type: str = 'violin',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_stats: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature distribution by diabetes status.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing feature and target columns
    feature : str
        Column name of feature to plot
    target : str
        Column name of target variable
    plot_type : str
        'violin', 'box', or 'kde'
    title : str, optional
        Figure title (defaults to feature name)
    xlabel : str, optional
        X-axis label (defaults to feature name)
    figsize : tuple
        Figure dimensions
    show_stats : bool
        Whether to show statistical test results
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    # Prepare data
    plot_df = df[[feature, target]].dropna()

    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'violin':
        sns.violinplot(
            data=plot_df,
            x=target,
            y=feature,
            palette=get_diabetes_palette(),
            ax=ax,
            inner='box',
            linewidth=1.5
        )
    elif plot_type == 'box':
        sns.boxplot(
            data=plot_df,
            x=target,
            y=feature,
            palette=get_diabetes_palette(),
            ax=ax,
            linewidth=1.5
        )
    elif plot_type == 'kde':
        for status in sorted(plot_df[target].unique()):
            subset = plot_df[plot_df[target] == status][feature]
            sns.kdeplot(
                subset,
                ax=ax,
                color=DIABETES_COLORS[int(status)],
                label=DIABETES_LABELS[int(status)],
                linewidth=2,
                fill=True,
                alpha=0.3
            )
        ax.legend()

    # Update x-axis labels
    if plot_type != 'kde':
        ax.set_xticklabels([DIABETES_LABELS[int(i)] for i in sorted(plot_df[target].unique())])
        ax.set_xlabel('')
    else:
        ax.set_xlabel(xlabel or feature)

    ax.set_ylabel(xlabel or feature)
    ax.set_title(title or f'{feature} by Diabetes Status', fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Statistical test annotation
    if show_stats and plot_type != 'kde':
        groups = [plot_df[plot_df[target] == s][feature].dropna() for s in sorted(plot_df[target].unique())]
        if len(groups) >= 2:
            stat, p_value = stats.kruskal(*groups)
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            ax.text(
                0.98, 0.98,
                f'Kruskal-Wallis p {sig}',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=TICK_FONTSIZE,
                style='italic',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_feature_panel(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'DIABETES_STATUS',
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4, 4),
    suptitle: str = 'Feature Distributions by Diabetes Status',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a multi-panel figure showing multiple features by diabetes status.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    features : list of str
        Column names of features to plot
    target : str
        Column name of target variable
    ncols : int
        Number of columns in grid
    figsize_per_plot : tuple
        Size per subplot
    suptitle : str
        Overall figure title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    n_features = len(features)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, (ax, feature) in enumerate(zip(axes, features)):
        plot_df = df[[feature, target]].dropna()

        sns.violinplot(
            data=plot_df,
            x=target,
            y=feature,
            palette=get_diabetes_palette(),
            ax=ax,
            inner='quartile',
            linewidth=1
        )

        ax.set_xticklabels([DIABETES_LABELS[int(i)] for i in sorted(plot_df[target].unique())])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(feature, fontsize=LABEL_FONTSIZE)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=TITLE_FONTSIZE + 2, fontweight='bold', y=1.02)

    # Add legend
    handles = [mpatches.Patch(color=DIABETES_COLORS[i], label=DIABETES_LABELS[i]) for i in [0, 1, 2]]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


# =============================================================================
# CORRELATION VISUALIZATIONS
# =============================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = 'spearman',
    figsize: Tuple[int, int] = (14, 12),
    title: str = 'Feature Correlation Matrix',
    cluster: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a correlation heatmap with optional hierarchical clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features
    features : list of str, optional
        Columns to include (defaults to all numeric)
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    figsize : tuple
        Figure dimensions
    title : str
        Figure title
    cluster : bool
        Whether to apply hierarchical clustering
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    # Select features
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df[features].corr(method=method)

    if cluster:
        # Use clustermap
        g = sns.clustermap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            figsize=figsize,
            linewidths=0.5,
            dendrogram_ratio=(0.1, 0.1),
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            tree_kws={'linewidths': 1.5}
        )
        g.fig.suptitle(title, fontsize=TITLE_FONTSIZE + 2, fontweight='bold', y=1.02)
        fig = g.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(title, fontsize=TITLE_FONTSIZE + 2, fontweight='bold')
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_top_correlations(
    df: pd.DataFrame,
    target: str,
    n_top: int = 20,
    method: str = 'spearman',
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, pd.Series]:
    """
    Create a horizontal bar chart of top features by correlation with target.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    target : str
        Target column name
    n_top : int
        Number of top features to show
    method : str
        Correlation method
    figsize : tuple
        Figure dimensions
    title : str, optional
        Figure title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple of (matplotlib.Figure, pd.Series of correlations)
    """
    set_publication_style()

    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corrwith(df[target], method=method)
    correlations = correlations.drop(target, errors='ignore')

    # Sort by absolute value and get top N
    top_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index)[:n_top]
    top_corr = top_corr.sort_values()  # Sort by value for plotting

    fig, ax = plt.subplots(figsize=figsize)

    # Color by positive/negative
    colors = ['#b2182b' if v > 0 else '#2166ac' for v in top_corr.values]

    bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor='white', linewidth=0.5)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel(f'{method.capitalize()} Correlation', fontsize=LABEL_FONTSIZE)
    ax.set_title(
        title or f'Top {n_top} Features Correlated with {target}',
        fontsize=TITLE_FONTSIZE,
        fontweight='bold'
    )

    # Add value labels
    for bar, val in zip(bars, top_corr.values):
        x_pos = val + 0.01 if val > 0 else val - 0.01
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                ha=ha, va='center', fontsize=TICK_FONTSIZE - 1)

    ax.set_xlim(-1, 1)

    # Add legend
    handles = [
        mpatches.Patch(color='#b2182b', label='Positive'),
        mpatches.Patch(color='#2166ac', label='Negative')
    ]
    ax.legend(handles=handles, loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig, correlations


# =============================================================================
# INTERACTION PLOTS
# =============================================================================

def plot_scatter_by_status(
    df: pd.DataFrame,
    x: str,
    y: str,
    target: str = 'DIABETES_STATUS',
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    sample_frac: Optional[float] = None,
    add_regression: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a scatter plot colored by diabetes status.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing x, y, and target columns
    x : str
        X-axis column
    y : str
        Y-axis column
    target : str
        Target column for coloring
    title : str, optional
        Figure title
    figsize : tuple
        Figure dimensions
    alpha : float
        Point transparency
    sample_frac : float, optional
        Fraction of data to sample for large datasets
    add_regression : bool
        Whether to add regression lines per group
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    plot_df = df[[x, y, target]].dropna()

    if sample_frac and len(plot_df) > 1000:
        plot_df = plot_df.sample(frac=sample_frac, random_state=42)

    fig, ax = plt.subplots(figsize=figsize)

    for status in sorted(plot_df[target].unique()):
        subset = plot_df[plot_df[target] == status]
        ax.scatter(
            subset[x],
            subset[y],
            c=DIABETES_COLORS[int(status)],
            label=DIABETES_LABELS[int(status)],
            alpha=alpha,
            s=30,
            edgecolors='none'
        )

        if add_regression:
            # Add regression line
            z = np.polyfit(subset[x], subset[y], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset[x].min(), subset[x].max(), 100)
            ax.plot(x_line, p(x_line), color=DIABETES_COLORS[int(status)],
                   linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlabel(x, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(y, fontsize=LABEL_FONTSIZE)
    ax.set_title(title or f'{y} vs {x} by Diabetes Status', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.legend(title='Status', framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_interaction_grid(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'DIABETES_STATUS',
    figsize_per_plot: Tuple[float, float] = (4, 4),
    sample_frac: float = 0.3,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grid of scatter plots for feature pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    features : list of str
        Features to plot (creates pairwise combinations)
    target : str
        Target column for coloring
    figsize_per_plot : tuple
        Size per subplot
    sample_frac : float
        Fraction to sample for performance
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    n = len(features)

    # Sample data
    plot_df = df[features + [target]].dropna()
    if sample_frac and len(plot_df) > 1000:
        plot_df = plot_df.sample(frac=sample_frac, random_state=42)

    fig, axes = plt.subplots(n, n, figsize=(figsize_per_plot[0] * n, figsize_per_plot[1] * n))

    for i, feat_i in enumerate(features):
        for j, feat_j in enumerate(features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: KDE
                for status in sorted(plot_df[target].unique()):
                    subset = plot_df[plot_df[target] == status][feat_i]
                    sns.kdeplot(subset, ax=ax, color=DIABETES_COLORS[int(status)],
                              fill=True, alpha=0.3, linewidth=1.5)
            else:
                # Off-diagonal: scatter
                for status in sorted(plot_df[target].unique()):
                    subset = plot_df[plot_df[target] == status]
                    ax.scatter(subset[feat_j], subset[feat_i],
                              c=DIABETES_COLORS[int(status)], alpha=0.4, s=10, edgecolors='none')

            # Labels only on edges
            if i == n - 1:
                ax.set_xlabel(feat_j, fontsize=TICK_FONTSIZE)
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(feat_i, fontsize=TICK_FONTSIZE)
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])

    # Add legend
    handles = [mpatches.Patch(color=DIABETES_COLORS[i], label=DIABETES_LABELS[i]) for i in [0, 1, 2]]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Feature Interactions by Diabetes Status', fontsize=TITLE_FONTSIZE + 2, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def plot_prevalence_by_year(
    df: pd.DataFrame,
    year_col: str,
    target: str = 'DIABETES_STATUS',
    figsize: Tuple[int, int] = (10, 6),
    title: str = 'Diabetes Prevalence by Survey Year',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot diabetes prevalence trends across survey years.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing year and target columns
    year_col : str
        Column with survey year
    target : str
        Target column
    figsize : tuple
        Figure dimensions
    title : str
        Figure title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    # Calculate prevalence by year
    prevalence = df.groupby(year_col)[target].apply(
        lambda x: x.value_counts(normalize=True) * 100
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(prevalence.index))
    width = 0.25

    for i, status in enumerate([0, 1, 2]):
        if status in prevalence.columns:
            bars = ax.bar(
                x + (i - 1) * width,
                prevalence[status],
                width,
                label=DIABETES_LABELS[status],
                color=DIABETES_COLORS[status],
                edgecolor='white',
                linewidth=1
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=TICK_FONTSIZE - 1)

    ax.set_xlabel('Survey Year', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Prevalence (%)', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prevalence.index)
    ax.legend(title='Status')
    ax.set_ylim(0, max(prevalence.max()) * 1.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def plot_feature_by_year(
    df: pd.DataFrame,
    feature: str,
    year_col: str,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature distribution across survey years.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing feature and year columns
    feature : str
        Feature column to plot
    year_col : str
        Column with survey year
    figsize : tuple
        Figure dimensions
    title : str, optional
        Figure title
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    fig, ax = plt.subplots(figsize=figsize)

    plot_df = df[[feature, year_col]].dropna()

    sns.boxplot(
        data=plot_df,
        x=year_col,
        y=feature,
        palette=CATEGORY_PALETTE[:len(plot_df[year_col].unique())],
        ax=ax,
        linewidth=1.5
    )

    ax.set_xlabel('Survey Year', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(feature, fontsize=LABEL_FONTSIZE)
    ax.set_title(title or f'{feature} by Survey Year', fontsize=TITLE_FONTSIZE, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

def plot_pca(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'DIABETES_STATUS',
    n_components: int = 2,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, dict]:
    """
    Create PCA visualization with variance explained.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    features : list of str
        Feature columns to use for PCA
    target : str
        Target column for coloring
    n_components : int
        Number of PCA components
    figsize : tuple
        Figure dimensions
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple of (matplotlib.Figure, dict with PCA results)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    set_publication_style()

    # Prepare data - drop rows where target is missing
    plot_df = df[features + [target]].copy()
    plot_df = plot_df[plot_df[target].notna()]

    # Convert features to numeric, coercing errors to NaN
    for feat in features:
        plot_df[feat] = pd.to_numeric(plot_df[feat], errors='coerce')

    X = plot_df[features].values.astype(np.float64)
    y = plot_df[target].values

    # Handle any remaining NaN/inf values in features
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Standardize and fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, len(features)))
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Scatter plot
    ax1 = axes[0]
    for status in sorted(np.unique(y)):
        mask = y == status
        ax1.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=DIABETES_COLORS[int(status)],
            label=DIABETES_LABELS[int(status)],
            alpha=0.5,
            s=30,
            edgecolors='none'
        )

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=LABEL_FONTSIZE)
    ax1.set_title('PCA: First Two Components', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax1.legend()

    # Right: Variance explained
    ax2 = axes[1]
    cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
    ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1),
           pca.explained_variance_ratio_ * 100,
           color='#4477AA', edgecolor='white', alpha=0.8)
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'o-', color='#CC6677', linewidth=2)
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Principal Component', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('Variance Explained (%)', fontsize=LABEL_FONTSIZE)
    ax2.set_title('Variance Explained by Component', fontsize=TITLE_FONTSIZE, fontweight='bold')
    ax2.legend(['Cumulative', 'Individual'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    results = {
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'feature_names': features
    }

    return fig, results


# =============================================================================
# RISK FACTOR SUMMARY
# =============================================================================

def plot_risk_factors(
    effect_sizes: pd.DataFrame,
    title: str = 'Risk Factors for Diabetes',
    figsize: Tuple[int, int] = (10, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a forest plot showing effect sizes for risk factors.

    Parameters
    ----------
    effect_sizes : pd.DataFrame
        DataFrame with columns: 'feature', 'effect_size', 'ci_lower', 'ci_upper'
    title : str
        Figure title
    figsize : tuple
        Figure dimensions
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    set_publication_style()

    # Sort by effect size
    effect_sizes = effect_sizes.sort_values('effect_size')

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(effect_sizes))

    # Color by direction
    colors = ['#b2182b' if e > 0 else '#2166ac' for e in effect_sizes['effect_size']]

    # Error bars
    xerr = [
        effect_sizes['effect_size'] - effect_sizes['ci_lower'],
        effect_sizes['ci_upper'] - effect_sizes['effect_size']
    ]

    ax.errorbar(
        effect_sizes['effect_size'],
        y_pos,
        xerr=xerr,
        fmt='o',
        capsize=3,
        capthick=1.5,
        elinewidth=1.5,
        markersize=8,
        color='black'
    )

    # Colored markers
    ax.scatter(effect_sizes['effect_size'], y_pos, c=colors, s=100, zorder=3, edgecolors='black', linewidths=0.5)

    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(effect_sizes['feature'])
    ax.set_xlabel('Standardized Effect Size', fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

    # Add legend
    handles = [
        mpatches.Patch(color='#b2182b', label='Increases Risk'),
        mpatches.Patch(color='#2166ac', label='Decreases Risk')
    ]
    ax.legend(handles=handles, loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')

    return fig


def calculate_effect_sizes(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    comparison: Tuple[int, int] = (0, 2),
) -> pd.DataFrame:
    """
    Calculate Cohen's d effect sizes comparing two groups.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing features and target
    features : list of str
        Features to calculate effect sizes for
    target : str
        Target column
    comparison : tuple
        Two target values to compare (default: No Diabetes vs Diabetes)

    Returns
    -------
    pd.DataFrame with columns: feature, effect_size, ci_lower, ci_upper
    """
    results = []

    g1 = df[df[target] == comparison[0]]
    g2 = df[df[target] == comparison[1]]

    for feat in features:
        try:
            x1 = g1[feat].dropna()
            x2 = g2[feat].dropna()

            if len(x1) < 10 or len(x2) < 10:
                continue

            # Cohen's d
            n1, n2 = len(x1), len(x2)
            var1, var2 = x1.var(), x2.var()
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

            if pooled_std == 0:
                continue

            d = (x2.mean() - x1.mean()) / pooled_std

            # Standard error of Cohen's d
            se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

            results.append({
                'feature': feat,
                'effect_size': d,
                'ci_lower': d - 1.96 * se,
                'ci_upper': d + 1.96 * se
            })
        except Exception:
            continue

    return pd.DataFrame(results)
