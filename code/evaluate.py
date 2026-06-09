import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    log_loss, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    matthews_corrcoef
)
from sklearn.utils.multiclass import type_of_target
from scipy.stats import pearsonr, spearmanr


def evaluate_model(y_true, y_pred, y_pred_proba=None) -> pd.DataFrame:
    results = {}

    # Detect task type
    task_type = type_of_target(y_true)
    is_classification = task_type in ["binary", "multiclass"]

    if is_classification:
        results["task"] = "classification"

        # Basic metrics
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

        # MCC
        try:
            results["mcc"] = matthews_corrcoef(y_true, y_pred)
        except:
            results["mcc"] = np.nan

        # Averaged metrics
        results["precision_macro"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results["recall_macro"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        results["precision_weighted"] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results["recall_weighted"] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Probabilistic metrics
        if y_pred_proba is not None:
            try:
                results["log_loss"] = log_loss(y_true, y_pred_proba)
            except:
                results["log_loss"] = np.nan

            try:
                if task_type == "binary":
                    results["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    results["roc_auc"] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                results["roc_auc"] = np.nan

    else:
        results["task"] = "regression"

        results["mse"] = mean_squared_error(y_true, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["mae"] = mean_absolute_error(y_true, y_pred)
        results["r2"] = r2_score(y_true, y_pred)

        # Correlation metrics (VERY important for regression)
        try:
            results["pcc"] = pearsonr(y_true, y_pred)[0]
        except:
            results["pcc"] = np.nan

        try:
            results["spcc"] = spearmanr(y_true, y_pred)[0]
        except:
            results["spcc"] = np.nan

    return results
