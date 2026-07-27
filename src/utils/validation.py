"""
Cross-validation strategy implementations for the CAA Sprint pipeline.
Implements three strategies matching the WA paper methodology:
1. Random 5-fold CV with bootstrap repeats
2. Province-disjoint CV (GroupKFold by province)
3. Era-disjoint CV (temporal train/test splits)
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, clone
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def evaluate_model(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute R², RMSE, MAE."""
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def random_cv(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = 500,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> dict[str, float]:
    """
    Random k-fold CV with bootstrap repeats.
    Returns mean and 95% CI for R², RMSE, MAE.

    Args:
        model: Scikit-learn estimator (will be cloned for each fold).
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        n_folds: Number of folds per repeat.
        n_repeats: Number of bootstrap repeats.
        random_seed: Base random seed.
        n_jobs: Parallel jobs (-1 = all cores).

    Returns:
        Dictionary with mean and CI bounds for each metric.
    """

    def _single_repeat(seed: int) -> list[dict]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_results = []
        for train_idx, test_idx in kf.split(X):
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            preds = m.predict(X[test_idx])
            fold_results.append(evaluate_model(y[test_idx], preds))
        return fold_results

    seeds = [random_seed + i for i in range(n_repeats)]
    all_fold_results = Parallel(n_jobs=n_jobs)(
        delayed(_single_repeat)(s) for s in seeds
    )

    # Flatten all fold results
    flat = [r for repeat in all_fold_results for r in repeat]
    df = pd.DataFrame(flat)

    result = {}
    for metric in ["R2", "RMSE", "MAE"]:
        vals = df[metric].values
        result[f"{metric}_mean"] = np.mean(vals)
        result[f"{metric}_ci_lo"] = np.percentile(vals, 2.5)
        result[f"{metric}_ci_hi"] = np.percentile(vals, 97.5)
        result[f"{metric}_std"] = np.std(vals)

    result["n_samples"] = len(y)
    result["n_folds"] = n_folds
    result["n_repeats"] = n_repeats
    return result


def province_disjoint_cv(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
) -> dict[str, float]:
    """
    Province-disjoint cross-validation using GroupKFold.
    All athletes from a given province appear exclusively in train OR test.

    Args:
        model: Scikit-learn estimator.
        X: Feature matrix.
        y: Target vector.
        groups: Province labels for each sample.
        n_folds: Number of folds (limited by number of unique groups).

    Returns:
        Dictionary with mean metrics and number of groups.
    """
    unique_groups = np.unique(groups)
    actual_folds = min(n_folds, len(unique_groups))

    if actual_folds < 2:
        logger.warning(f"Only {len(unique_groups)} groups, skipping province-disjoint CV")
        return {}

    gkf = GroupKFold(n_splits=actual_folds)
    fold_results = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[test_idx])
        fold_results.append(evaluate_model(y[test_idx], preds))

    df = pd.DataFrame(fold_results)
    result = {}
    for metric in ["R2", "RMSE", "MAE"]:
        result[f"{metric}_mean"] = df[metric].mean()
        result[f"{metric}_std"] = df[metric].std()

    result["n_samples"] = len(y)
    result["n_groups"] = len(unique_groups)
    result["n_folds"] = actual_folds
    return result


def era_disjoint_cv(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    athlete_last_year: np.ndarray,
    era_splits: list[dict],
) -> list[dict[str, float]]:
    """
    Era-disjoint validation: train on earlier athletes, test on later.

    Args:
        model: Scikit-learn estimator.
        X: Feature matrix.
        y: Target vector.
        athlete_last_year: Last competition year for each athlete.
        era_splits: List of {train_end, test_start, test_end, label}.

    Returns:
        List of result dictionaries, one per split.
    """
    results = []
    for split in era_splits:
        train_mask = athlete_last_year <= split["train_end"]
        test_mask = (athlete_last_year >= split["test_start"]) & (
            athlete_last_year <= split["test_end"]
        )

        n_train = train_mask.sum()
        n_test = test_mask.sum()

        if n_train < 50 or n_test < 50:
            logger.warning(
                f"Split {split['label']}: train={n_train}, test={n_test} — skipping"
            )
            continue

        m = clone(model)
        m.fit(X[train_mask], y[train_mask])
        preds = m.predict(X[test_mask])
        metrics = evaluate_model(y[test_mask], preds)
        metrics["split_label"] = split["label"]
        metrics["n_train"] = int(n_train)
        metrics["n_test"] = int(n_test)
        results.append(metrics)

    return results
