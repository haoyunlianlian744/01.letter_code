"""
feature_selection.py — Pre-filtering utilities for aluminum alloy ML pipeline.

Used by N03:
  - filter_low_variance()   — remove near-zero-variance features
  - filter_high_correlation() — remove highly correlated duplicates

N03 applies these filters before running its own ExtraTrees importance ranking
and mlxtend SFFS (Sequential Floating Forward Selection) for final selection.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_selection import VarianceThreshold


# ============================================================================
# Low variance filter
# ============================================================================

def filter_low_variance(
    X: pd.DataFrame,
    threshold: float = 1e-4,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove near-zero variance features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Variance threshold (features with variance < threshold are removed).

    Returns
    -------
    X_filtered : pd.DataFrame
    kept_cols : list of str
    """
    sel = VarianceThreshold(threshold=threshold)
    X_arr = sel.fit_transform(X.fillna(0))
    kept = X.columns[sel.get_support(indices=True)].tolist()
    removed = [c for c in X.columns if c not in kept]
    if removed:
        print(f'  [VarFilter] Removed {len(removed)} near-zero variance features')
    return pd.DataFrame(X_arr, columns=kept, index=X.index), kept


# ============================================================================
# High correlation filter
# ============================================================================

def filter_high_correlation(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features (keep the one with higher variance).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Correlation coefficient above which one feature is removed.

    Returns
    -------
    X_filtered : pd.DataFrame
    kept_cols : list of str
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        correlated = upper[col][upper[col] > threshold].index.tolist()
        for c in correlated:
            # Keep the one with higher variance
            if X[col].var() >= X[c].var():
                to_drop.add(c)
            else:
                to_drop.add(col)
    kept = [c for c in X.columns if c not in to_drop]
    removed = [c for c in X.columns if c in to_drop]
    if removed:
        print(f'  [CorrFilter] Removed {len(removed)} highly correlated features (r > {threshold})')
    return X[kept], kept
