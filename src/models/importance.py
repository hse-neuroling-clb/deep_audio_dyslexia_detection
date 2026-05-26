import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.pipeline import Pipeline


# Linear / regularized models
def linear_importance(model: Pipeline, x_columns, *args, **kwargs):
    coefs = np.abs(model.named_steps["model"].coef_).ravel()
    return pd.Series(coefs, index=x_columns).sort_values(ascending=False)

# Tree-based models
def tree_importance(model: Pipeline, x_columns, *args, **kwargs):
    importances = model.named_steps["model"].feature_importances_
    return pd.Series(importances, index=x_columns).sort_values(ascending=False)

# Permutation-based importance (model-agnostic)
def permutation_importance(model: Pipeline, x_columns, X_val, y_val, random_state, n_jobs=2):
    if X_val is None or y_val is None:
        raise ValueError("X_test and y_test required for permutation importance")
    print("[Permutation Importance] Computing permutation importance on validation set...")
    result = sklearn_permutation_importance(model, X_val, y_val, n_repeats=10, random_state=random_state, n_jobs=1, max_samples=200)

    means = result.importances_mean  # type: ignore
    stds = result.importances_std  # type: ignore

    str_values = [f"{mean:.4f} ± {std:.4f}" for mean, std in zip(means, stds)]

    return pd.Series(str_values, index=x_columns).sort_values(ascending=False)
