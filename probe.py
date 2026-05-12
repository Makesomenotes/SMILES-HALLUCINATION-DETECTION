"""
probe.py — Multi-classifier probe: LogReg, SVM, RF etc.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._net = None
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._pca_components: int = 100
        self._threshold: float = 0.5
        self._clf = None  # best classifier

    def _build_network(self, input_dim: int) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            X_scaled = self._scaler.fit_transform(X)
            n_components = min(self._pca_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
            self._pca = PCA(n_components=n_components, random_state=42)
            X_scaled = self._pca.fit_transform(X_scaled)
        else:
            X_scaled = self._scaler.transform(X)
            if self._pca is not None:
                X_scaled = self._pca.transform(X_scaled)
        return X_scaled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_scaled = self._preprocess(X, fit=True)
        candidates = {
            "logreg_c01": LogisticRegression(
                C=0.1, max_iter=10000, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            "logreg_c1": LogisticRegression(
                C=1.0, max_iter=10000, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            "logreg_c10": LogisticRegression(
                C=10.0, max_iter=10000, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            "logreg_c001": LogisticRegression(
                C=0.01, max_iter=10000, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            "svm_rbf": SVC(
                C=1.0, kernel='rbf', probability=True,
                class_weight='balanced', random_state=42
            ),
            "svm_rbf_c10": SVC(
                C=10.0, kernel='rbf', probability=True,
                class_weight='balanced', random_state=42
            ),
            "gb_50": GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            "gb_100": GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
        }

        best_score = -1.0
        best_name = None
        best_clf = None

        for name, clf in candidates.items():
            try:
                scores = cross_val_score(
                    clf, X_scaled, y, cv=5, scoring='roc_auc'
                )
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    best_clf = clf
            except Exception:
                continue
        best_clf.fit(X_scaled, y)
        self._clf = best_clf
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0
        for t in np.linspace(0.1, 0.9, 161):
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._preprocess(X, fit=False)
        return self._clf.predict_proba(X_scaled)