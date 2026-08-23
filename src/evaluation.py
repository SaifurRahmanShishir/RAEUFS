"""Evaluation utilities for RAEUFS."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def _hungarian_accuracy(true_labels, predicted_labels):
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    D = max(predicted_labels.max(), true_labels.max()) + 1
    contingency = np.zeros((D, D), dtype=np.int64)

    for pred, true in zip(predicted_labels, true_labels):
        contingency[pred, true] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    return (
        sum(contingency[row, col] for row, col in zip(row_ind, col_ind))
        / len(true_labels)
    )


class EVMetrics:
    def __init__(self, model, features, labels, selected_features):
        self.model = model
        self.features = features
        self.labels = labels
        self.selected_features = selected_features

    def kmeans_cluster_accuracy_nmi(self, repeats: int = 100):
        """Evaluate clustering after applying the learned W matrix.

        ``np.einsum`` is used instead of NumPy ``@`` for the feature transform
        because some Apple-Silicon/NumPy combinations emit spurious floating-
        point matmul warnings even for numerically valid matrices.
        """
        W = self.model.FS.W.detach().cpu().numpy()
        features = np.asarray(self.features, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)

        if not np.isfinite(W).all():
            raise ValueError("W contains NaN or Inf before evaluation.")
        if not np.isfinite(features).all():
            raise ValueError("Features contain NaN or Inf before evaluation.")

        X_test_selected = np.einsum("nd,dp->np", features, W)

        if not np.isfinite(X_test_selected).all():
            raise ValueError("Selected feature matrix contains NaN or Inf.")

        k = len(np.unique(self.labels))
        acc = []
        nmi = []

        # KMeans itself may call NumPy matmul internally. Scope the warning
        # suppression to this evaluation only; values are checked explicitly.
        old_settings = np.seterr(divide="ignore", over="ignore", invalid="ignore")
        try:
            for _ in range(repeats):
                kmeans = KMeans(n_clusters=k).fit(X_test_selected)
                predicted_labels = kmeans.labels_

                acc.append(_hungarian_accuracy(self.labels, predicted_labels))
                nmi.append(
                    normalized_mutual_info_score(self.labels, predicted_labels)
                )
        finally:
            np.seterr(**old_settings)

        return np.mean(acc), np.mean(nmi)


# Backward-compatible alias for the original class name.
EV_metrics = EVMetrics


def baseline_performance(X_selected, labels, X_eval, eval_labels, repeats=100):
    if hasattr(X_selected, "detach"):
        X_selected = X_selected.detach().cpu().numpy()
    if hasattr(X_eval, "detach"):
        X_eval = X_eval.detach().cpu().numpy()

    X_selected = np.asarray(X_selected, dtype=np.float64)
    X_eval = np.asarray(X_eval, dtype=np.float64)

    k = len(np.unique(labels))
    acc = []
    nmi = []

    old_settings = np.seterr(divide="ignore", over="ignore", invalid="ignore")
    try:
        for _ in range(repeats):
            kmeans = KMeans(n_clusters=k).fit(X_selected)
            predicted_labels = kmeans.predict(X_eval)
            acc.append(_hungarian_accuracy(eval_labels, predicted_labels))
            nmi.append(normalized_mutual_info_score(eval_labels, predicted_labels))
    finally:
        np.seterr(**old_settings)

    return [
        np.mean(acc) * 100,
        np.std(acc) * 100,
        np.mean(nmi) * 100,
        np.std(nmi) * 100,
    ]
