"""
Model training utilities for diabetes prediction.

This module provides functions for:
- Data splitting (train/val/test with stratification)
- Cross-validation with proper handling of class imbalance
- Model training with experiment tracking via MLflow
- Hyperparameter tuning with Optuna
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Try importing LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    test_size : float
        Proportion of data for test set
    val_size : float
        Proportion of data for validation set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify splits by target class

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: separate test set
    stratify_col = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    # Second split: separate validation from training
    # Adjust val_size for the remaining data
    val_adjusted = val_size / (1 - test_size)
    stratify_col = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted, random_state=random_state, stratify=stratify_col
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute balanced class weights for handling class imbalance.

    Parameters
    ----------
    y : pd.Series
        Target variable

    Returns
    -------
    dict
        Dictionary mapping class labels to weights
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


# =============================================================================
# Model Definitions
# =============================================================================

def get_baseline_models(task: str = 'classification') -> Dict[str, BaseEstimator]:
    """
    Get baseline models for establishing benchmarks.

    Parameters
    ----------
    task : str
        'classification' or 'regression'

    Returns
    -------
    dict
        Dictionary of model name -> model instance
    """
    if task == 'classification':
        return {
            'Dummy (Stratified)': DummyClassifier(strategy='stratified', random_state=42),
            'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        }
    else:
        return {
            'Dummy (Mean)': DummyRegressor(strategy='mean'),
            'Dummy (Median)': DummyRegressor(strategy='median'),
            'Linear Regression (Ridge)': Ridge(alpha=1.0, random_state=42),
            'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        }


def get_advanced_models(
    task: str = 'classification',
    class_weights: Optional[Dict[int, float]] = None,
    random_state: int = 42,
) -> Dict[str, BaseEstimator]:
    """
    Get advanced models for full training.

    Parameters
    ----------
    task : str
        'classification' or 'regression'
    class_weights : dict, optional
        Class weights for handling imbalance
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary of model name -> model instance
    """
    if task == 'classification':
        models = {
            'Logistic Regression (L2)': LogisticRegression(
                penalty='l2', C=1.0, max_iter=1000, solver='lbfgs',
                class_weight=class_weights, random_state=random_state
            ),
            'Logistic Regression (L1)': LogisticRegression(
                penalty='l1', C=1.0, max_iter=1000, solver='saga',
                class_weight=class_weights, random_state=random_state
            ),
            'MLP Classifier': MLPClassifier(
                hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001,
                max_iter=500, early_stopping=True, random_state=random_state
            ),
        }

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=7, learning_rate=0.05,
                num_leaves=31, class_weight=class_weights,
                random_state=random_state, verbose=-1, n_jobs=-1
            )
    else:
        models = {
            'Ridge': Ridge(alpha=1.0, random_state=random_state),
            'Lasso': Lasso(alpha=0.01, max_iter=1000, random_state=random_state),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=random_state),
            'MLP Regressor': MLPRegressor(
                hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001,
                max_iter=500, early_stopping=True, random_state=random_state
            ),
        }

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=300, max_depth=7, learning_rate=0.05,
                num_leaves=31, random_state=random_state, verbose=-1, n_jobs=-1
            )

    return models


# =============================================================================
# Cross-Validation Training
# =============================================================================

def train_with_cv(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
    scoring: Union[str, List[str]] = None,
    task: str = 'classification',
    return_train_score: bool = True,
) -> Dict[str, Any]:
    """
    Train model with cross-validation and return scores.

    Parameters
    ----------
    model : BaseEstimator
        Sklearn-compatible model
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    cv : int
        Number of CV folds
    scoring : str or list
        Scoring metric(s). If None, uses defaults for task type
    task : str
        'classification' or 'regression'
    return_train_score : bool
        Whether to return training scores

    Returns
    -------
    dict
        Cross-validation results with scores and timing
    """
    # Default scoring metrics
    if scoring is None:
        if task == 'classification':
            scoring = ['accuracy', 'f1_macro', 'roc_auc_ovr']
        else:
            scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

    # Set up CV splitter
    if task == 'classification':
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Handle NaN values by converting to numpy
    X_np = X.values if isinstance(X, pd.DataFrame) else X
    y_np = y.values if isinstance(y, pd.Series) else y

    # Run cross-validation
    cv_results = cross_validate(
        model, X_np, y_np,
        cv=cv_splitter, scoring=scoring,
        return_train_score=return_train_score,
        return_estimator=False, n_jobs=-1
    )

    # Process results
    results = {
        'fit_time': cv_results['fit_time'].mean(),
        'fit_time_std': cv_results['fit_time'].std(),
        'score_time': cv_results['score_time'].mean(),
    }

    # Add test scores
    for metric in scoring:
        key = f'test_{metric}'
        results[f'{metric}_mean'] = cv_results[key].mean()
        results[f'{metric}_std'] = cv_results[key].std()
        results[f'{metric}_all'] = cv_results[key].tolist()

    # Add train scores if requested
    if return_train_score:
        for metric in scoring:
            key = f'train_{metric}'
            results[f'{metric}_train_mean'] = cv_results[key].mean()

    return results


# =============================================================================
# MLflow Experiment Tracking
# =============================================================================

def setup_mlflow(experiment_name: str = 'diabetes-prediction', tracking_uri: str = 'mlruns'):
    """Set up MLflow experiment tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_experiment(
    model_name: str,
    model: BaseEstimator,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_set: str,
    task: str,
    tags: Optional[Dict[str, str]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
) -> str:
    """
    Log experiment to MLflow.

    Parameters
    ----------
    model_name : str
        Name of the model
    model : BaseEstimator
        Trained model
    params : dict
        Model parameters
    metrics : dict
        Performance metrics
    feature_set : str
        'with_labs' or 'without_labs'
    task : str
        'classification' or 'regression'
    tags : dict, optional
        Additional tags
    artifacts : dict, optional
        Additional artifacts to log (name -> data)
    save_model : bool
        Whether to save the model artifact

    Returns
    -------
    str
        Run ID
    """
    with mlflow.start_run() as run:
        # Set tags
        mlflow.set_tag('model_name', model_name)
        mlflow.set_tag('feature_set', feature_set)
        mlflow.set_tag('task', task)
        mlflow.set_tag('timestamp', datetime.now().isoformat())

        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Save model
        if save_model:
            mlflow.sklearn.log_model(model, 'model')

        # Log additional artifacts
        if artifacts:
            for name, data in artifacts.items():
                if isinstance(data, dict):
                    mlflow.log_dict(data, f'{name}.json')
                elif isinstance(data, pd.DataFrame):
                    mlflow.log_text(data.to_csv(index=False), f'{name}.csv')

        return run.info.run_id


# =============================================================================
# Hyperparameter Tuning with Optuna
# =============================================================================

def create_optuna_objective(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = 'classification',
    cv: int = 3,
    class_weights: Optional[Dict[int, float]] = None,
    scoring: str = None,
) -> callable:
    """
    Create an Optuna objective function for hyperparameter tuning.

    Parameters
    ----------
    model_type : str
        Type of model to tune
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    task : str
        'classification' or 'regression'
    cv : int
        Number of CV folds
    class_weights : dict, optional
        Class weights for classification
    scoring : str, optional
        Scoring metric

    Returns
    -------
    callable
        Objective function for Optuna
    """
    # Default scoring
    if scoring is None:
        scoring = 'f1_macro' if task == 'classification' else 'neg_root_mean_squared_error'

    # CV splitter
    if task == 'classification':
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_np = y_train.values if isinstance(y_train, pd.Series) else y_train

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""

        # Define hyperparameter search spaces
        if model_type == 'logistic_regression':
            params = {
                'C': trial.suggest_float('C', 1e-4, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga',
                'max_iter': 1000,
                'class_weight': class_weights,
                'random_state': 42,
            }
            model = LogisticRegression(**params)

        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM not available")

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            }

            if task == 'classification':
                params['class_weight'] = class_weights
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

        elif model_type == 'mlp':
            # Layer configurations
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256))

            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 500,
                'early_stopping': True,
                'random_state': 42,
            }

            if task == 'classification':
                model = MLPClassifier(**params)
            else:
                model = MLPRegressor(**params)

        elif model_type == 'ridge':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True),
                'random_state': 42,
            }
            model = Ridge(**params)

        elif model_type == 'lasso':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 10, log=True),
                'max_iter': 1000,
                'random_state': 42,
            }
            model = Lasso(**params)

        elif model_type == 'elasticnet':
            params = {
                'alpha': trial.suggest_float('alpha', 1e-5, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': 1000,
                'random_state': 42,
            }
            model = ElasticNet(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Cross-validate
        try:
            scores = cross_val_score(model, X_np, y_np, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            return scores.mean()
        except Exception as e:
            # Return worst score on failure
            return -1e10 if scoring.startswith('neg') else 0.0

    return objective


def tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task: str = 'classification',
    n_trials: int = 50,
    timeout: Optional[int] = None,
    cv: int = 3,
    class_weights: Optional[Dict[int, float]] = None,
    scoring: str = None,
    study_name: Optional[str] = None,
    show_progress: bool = True,
) -> Tuple[BaseEstimator, Dict[str, Any], optuna.Study]:
    """
    Tune model hyperparameters with Optuna.

    Parameters
    ----------
    model_type : str
        Type of model ('logistic_regression', 'lightgbm', 'mlp', 'ridge', etc.)
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    task : str
        'classification' or 'regression'
    n_trials : int
        Number of Optuna trials
    timeout : int, optional
        Max seconds for tuning
    cv : int
        Number of CV folds
    class_weights : dict, optional
        Class weights for classification
    scoring : str, optional
        Scoring metric
    study_name : str, optional
        Name for Optuna study
    show_progress : bool
        Whether to show progress bar

    Returns
    -------
    best_model : BaseEstimator
        Best model with tuned hyperparameters
    best_params : dict
        Best hyperparameters
    study : optuna.Study
        Optuna study object with all trials
    """
    # Create objective
    objective = create_optuna_objective(
        model_type=model_type,
        X_train=X_train,
        y_train=y_train,
        task=task,
        cv=cv,
        class_weights=class_weights,
        scoring=scoring,
    )

    # Create study
    direction = 'maximize' if scoring is None or not scoring.startswith('neg') else 'maximize'
    if scoring and scoring.startswith('neg'):
        direction = 'maximize'  # cross_val_score already negates, so maximize

    study = optuna.create_study(
        study_name=study_name or f"{model_type}_{task}",
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Run optimization
    verbosity = optuna.logging.INFO if show_progress else optuna.logging.WARNING
    optuna.logging.set_verbosity(verbosity)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress)

    # Get best parameters
    best_params = study.best_params

    # Create best model
    best_model = _create_model_from_params(
        model_type=model_type,
        params=best_params,
        task=task,
        class_weights=class_weights,
    )

    # Fit best model on full training data
    X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_np = y_train.values if isinstance(y_train, pd.Series) else y_train
    best_model.fit(X_np, y_np)

    return best_model, best_params, study


def _create_model_from_params(
    model_type: str,
    params: Dict[str, Any],
    task: str,
    class_weights: Optional[Dict[int, float]] = None,
) -> BaseEstimator:
    """Create a model instance from parameters."""

    if model_type == 'logistic_regression':
        model_params = {
            'C': params.get('C', 1.0),
            'penalty': params.get('penalty', 'l2'),
            'solver': 'saga',
            'max_iter': 1000,
            'class_weight': class_weights,
            'random_state': 42,
        }
        return LogisticRegression(**model_params)

    elif model_type == 'lightgbm':
        model_params = {
            'n_estimators': params.get('n_estimators', 300),
            'max_depth': params.get('max_depth', 7),
            'learning_rate': params.get('learning_rate', 0.05),
            'num_leaves': params.get('num_leaves', 31),
            'min_child_samples': params.get('min_child_samples', 20),
            'reg_alpha': params.get('reg_alpha', 0.0),
            'reg_lambda': params.get('reg_lambda', 0.0),
            'subsample': params.get('subsample', 1.0),
            'colsample_bytree': params.get('colsample_bytree', 1.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1,
        }

        if task == 'classification':
            model_params['class_weight'] = class_weights
            return lgb.LGBMClassifier(**model_params)
        else:
            return lgb.LGBMRegressor(**model_params)

    elif model_type == 'mlp':
        # Reconstruct hidden layer sizes
        n_layers = params.get('n_layers', 2)
        layers = [params.get(f'n_units_l{i}', 64) for i in range(n_layers)]

        model_params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': params.get('activation', 'relu'),
            'alpha': params.get('alpha', 0.001),
            'learning_rate': params.get('learning_rate', 'constant'),
            'learning_rate_init': params.get('learning_rate_init', 0.001),
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42,
        }

        if task == 'classification':
            return MLPClassifier(**model_params)
        else:
            return MLPRegressor(**model_params)

    elif model_type == 'ridge':
        return Ridge(alpha=params.get('alpha', 1.0), random_state=42)

    elif model_type == 'lasso':
        return Lasso(alpha=params.get('alpha', 0.01), max_iter=1000, random_state=42)

    elif model_type == 'elasticnet':
        return ElasticNet(
            alpha=params.get('alpha', 0.01),
            l1_ratio=params.get('l1_ratio', 0.5),
            max_iter=1000,
            random_state=42,
        )

    raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Model Saving/Loading
# =============================================================================

def save_model(
    model: BaseEstimator,
    model_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : BaseEstimator
        Trained model
    model_path : str or Path
        Path to save model
    metadata : dict, optional
        Additional metadata to save
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path)

    # Save metadata if provided
    if metadata:
        meta_path = model_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def load_model(model_path: Union[str, Path]) -> Tuple[BaseEstimator, Optional[Dict[str, Any]]]:
    """
    Load trained model from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to saved model

    Returns
    -------
    model : BaseEstimator
        Loaded model
    metadata : dict or None
        Associated metadata if available
    """
    model_path = Path(model_path)
    model = joblib.load(model_path)

    # Try to load metadata
    meta_path = model_path.with_suffix('.json')
    metadata = None
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    return model, metadata
