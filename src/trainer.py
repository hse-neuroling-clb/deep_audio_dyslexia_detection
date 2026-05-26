import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, f1_score, roc_auc_score, precision_recall_fscore_support
from dataclasses import dataclass

from sklearn.pipeline import Pipeline, clone
import xgboost

from src.experiment import Experiment
from src.models.smoothing import SmoothingTransformer


@dataclass
class TrainerResult:
    y_val: np.ndarray | pd.Series
    y_val_pred: np.ndarray | pd.Series
    y_train: np.ndarray | pd.Series
    y_train_pred: np.ndarray | pd.Series


class Trainer:
    def __init__(self, experiment: Experiment, model: Pipeline):
        self.model = model
        self.experiment = experiment
        self.smoothing_transformer = SmoothingTransformer(
            experiment.smoothing_scale_type, experiment.calculate_thresholds, experiment.smoothing_clip_upper_bound_type
        )

    def train(self, X_train, X_val, base_y_train, base_y_val):
        self.smoothing_transformer.fit(base_y_train)
        y_train = self.smoothing_transformer.transform(base_y_train)
        y_val = self.smoothing_transformer.transform(base_y_val)

        self.model.fit(X_train, y_train)

        y_val_pred = self.model.predict(X_val)
        y_train_pred = self.model.predict(X_train)
        return TrainerResult(y_val=y_val, y_val_pred=y_val_pred, y_train=y_train, y_train_pred=y_train_pred) # pyright: ignore[reportArgumentType]
    
    def transform_y(self, base_y_data):
        return self.smoothing_transformer.transform(base_y_data)
    
    def predict(self, X):
        return self.model.predict(X)


def evaluate_classification(y_val, y_val_pred, y_val_score, y_train, y_train_pred, y_train_score, y_labels:list|None, y_names:list[str]|None):
    classification_metrics = [f1_score, roc_auc_score]
    results = dict()
    for metric in classification_metrics:
        results[metric.__name__] = metric(y_val, y_val_pred)
        results[f"{metric.__name__}_train"] = metric(y_train, y_train_pred)
        results[f"{metric.__name__}_overfit_gap"] = results[f"{metric.__name__}_train"] - results[metric.__name__]
    results["cls_val_size"] = len(y_val)
    results["cls_train_size"] = len(y_train)

    cm = confusion_matrix(y_val, y_val_pred, labels=y_labels)
    report = classification_report(y_val, y_val_pred, labels=y_labels, target_names=y_names, output_dict=True)
    assert type(report) == dict, "Classification report should be a dict when output_dict=True"
    precision_recall_fscore_support_value = precision_recall_fscore_support(y_val, y_val_pred, labels=y_labels)
    return results, cm, precision_recall_fscore_support_value, report


def evaluate_regression(y_val, y_val_pred, y_train, y_train_pred):
    regression_metrics = [r2_score, mean_absolute_error]
    results = dict()
    for metric in regression_metrics:
        results[metric.__name__] = metric(y_val, y_val_pred)
        results[f"{metric.__name__}_train"] = metric(y_train, y_train_pred)
        results[f"{metric.__name__}_overfit_gap"] = results[f"{metric.__name__}_train"] - results[metric.__name__]
    return results


CLASSIFICATION_FROM_REGRESSION_THRESHOLD = 0.45


def binarize_predictions(y_pred):
    return np.where(y_pred >= CLASSIFICATION_FROM_REGRESSION_THRESHOLD, 1, 0)


def evaluate(result: TrainerResult, y_labels:list|None = None, y_names:list[str]|None = None):

    results = dict()
    regression_metrics = evaluate_regression(result.y_val, result.y_val_pred, result.y_train, result.y_train_pred)
    classification_metrics, confusion_matrix, precision_recall_fscore_support_value, report = evaluate_classification(
        binarize_predictions(result.y_val),
        binarize_predictions(result.y_val_pred),
        result.y_val_pred,
        binarize_predictions(result.y_train),
        binarize_predictions(result.y_train_pred),
        result.y_train_pred,
        y_labels,
        y_names
    )
    results.update(regression_metrics)
    results.update(classification_metrics)
    return results, confusion_matrix, precision_recall_fscore_support_value, report
