# Grid functions for hyperparameter search
from optuna import Trial


def linreg_grid(trial: Trial):
    return {
        "model__fit_intercept": trial.suggest_categorical("model__fit_intercept", [True, False]),
        "model__copy_X": trial.suggest_categorical("model__copy_X", [True, False]),
        "model__positive": trial.suggest_categorical("model__positive", [False, True]),
    }


def gradient_boosting_grid(trial: Trial):
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 1200),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.025, 0.1),
        "model__max_depth": trial.suggest_int("model__max_depth", 3, 6),
        "model__min_samples_split": trial.suggest_int("model__min_samples_split", 2, 5),
        "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 2),
        "model__subsample": trial.suggest_float("model__subsample", 0.8, 1.0),
        "model__max_features": trial.suggest_categorical("model__max_features", ["sqrt", None]),
    }


def adaboost_grid(trial: Trial):
    return {
        "model__n_estimators": trial.suggest_int("model__n_estimators", 1, 900),
        "model__learning_rate": trial.suggest_float("model__learning_rate", 0.025, 0.2),
        "model__loss": trial.suggest_categorical("model__loss", ["linear", "square", "exponential"]),
    }


def knn_grid(trial: Trial):
    return {
        "model__n_neighbors": trial.suggest_int("model__n_neighbors", 3, 30),
        "model__weights": trial.suggest_categorical("model__weights", ["uniform", "distance"]),
        "model__p": trial.suggest_int("model__p", 1, 2),
        "model__leaf_size": trial.suggest_int("model__leaf_size", 2, 16),
    }


def mlp_grid(trial: Trial):
    return {
        "model__activation": trial.suggest_categorical("model__activation", ["relu", "tanh"]),
        "model__alpha": trial.suggest_float("model__alpha", 1e-4, 1e-1), # 5.1e-2 was a baseline
        "model__learning_rate_init": trial.suggest_float("model__learning_rate_init", 0.001, 0.01), # 0.00123 was a baseline
        "model__max_iter": trial.suggest_int("model__max_iter", 500, 1500),
        "model__early_stopping": trial.suggest_categorical("model__early_stopping", [True, False]),
    }
