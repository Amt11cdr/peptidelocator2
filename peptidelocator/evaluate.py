import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    log_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    matthews_corrcoef,
)
from sklearn.utils.multiclass import type_of_target
from scipy.stats import pearsonr, spearmanr


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> dict:
    """Compute evaluation metrics for classification or regression.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (1D array for binary classification)

    Returns:
        Dictionary of metric names to values
    """
    results = {}
    task_type = type_of_target(y_true)
    is_classification = task_type in ["binary", "multiclass"]

    if is_classification:
        results["task"] = "classification"
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        try:
            results["mcc"] = matthews_corrcoef(y_true, y_pred)
        except Exception:
            results["mcc"] = np.nan

        results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        results["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if y_pred_proba is not None:
            try:
                results["log_loss"] = log_loss(y_true, y_pred_proba)
            except Exception:
                results["log_loss"] = np.nan
            try:
                results["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
            except Exception:
                results["roc_auc"] = np.nan
    else:
        results["task"] = "regression"
        results["mse"] = mean_squared_error(y_true, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["mae"] = mean_absolute_error(y_true, y_pred)
        results["r2"] = r2_score(y_true, y_pred)
        try:
            results["pcc"] = pearsonr(y_true, y_pred)[0]
        except Exception:
            results["pcc"] = np.nan
        try:
            results["spcc"] = spearmanr(y_true, y_pred)[0]
        except Exception:
            results["spcc"] = np.nan

    return results
