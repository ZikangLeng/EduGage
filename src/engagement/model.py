"""Fusion modules and classifier definitions (Phase E)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


WindowEmbeddingIndex = dict[str, dict[str, list[tuple[str, np.ndarray]]]]


def _safe_softmax(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(centered)
    denom = exp.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return exp / denom


@dataclass
class SourceAwareModalityFusion:
    """Fuse multiple sources per modality.

    This implementation is intentionally stateless and uses mean fusion across
    available sources per modality.
    """

    selected_modalities: tuple[str, ...]
    modality_dims: dict[str, int]

    def fit(
        self,
        embedding_index: WindowEmbeddingIndex | None = None,
        train_window_ids: set[str] | None = None,
    ) -> None:
        # Stateless mean fusion has no fit-time state.
        _ = embedding_index
        _ = train_window_ids

    def fuse_window(
        self,
        window_modalities: dict[str, list[tuple[str, np.ndarray]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        embeddings: list[np.ndarray] = []
        mask: list[float] = []

        for modality in self.selected_modalities:
            target_dim = int(self.modality_dims.get(modality, 0))
            rows = window_modalities.get(modality, [])
            fused = np.zeros((target_dim,), dtype=float)

            if rows and target_dim > 0:
                vectors: list[np.ndarray] = []
                for _source_id, vector in rows:
                    vec = np.asarray(vector, dtype=float).reshape(-1)
                    if vec.size < target_dim:
                        padded = np.zeros((target_dim,), dtype=float)
                        padded[: vec.size] = vec
                        vec = padded
                    elif vec.size > target_dim:
                        vec = vec[:target_dim]
                    vectors.append(vec)

                stacked = np.stack(vectors, axis=0)
                fused = np.mean(stacked, axis=0)

                mask.append(1.0)
            else:
                mask.append(0.0)

            embeddings.append(fused)

        if embeddings:
            return np.concatenate(embeddings, axis=0), np.asarray(mask, dtype=float)
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fusion_impl": "mean",
            "selected_modalities": list(self.selected_modalities),
            "modality_dims": {k: int(v) for k, v in self.modality_dims.items()},
        }


@dataclass
class LinearClassifierHead:
    """Mask-aware linear classifier head using sklearn logistic regression."""

    input_dim: int
    n_classes: int
    seed: int = 42
    max_iter: int = 400
    artifact_schema_version: int = 2
    model_impl: str = "sklearn_logreg_v1"
    _estimator: LogisticRegression | None = None
    _coef: np.ndarray | None = None
    _intercept: np.ndarray | None = None
    _class_ids: np.ndarray | None = None
    _constant_class: int | None = None

    def _build_features(self, x_embed: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
        x_embed = np.asarray(x_embed, dtype=float)
        x_mask = np.asarray(x_mask, dtype=float)
        if x_embed.ndim != 2:
            raise ValueError("x_embed must be 2D.")
        if x_mask.ndim != 2:
            raise ValueError("x_mask must be 2D.")
        if x_embed.shape[0] != x_mask.shape[0]:
            raise ValueError("x_embed and x_mask must have the same number of rows.")
        return np.concatenate([x_embed, x_mask], axis=1)

    def fit(
        self,
        x_embed: np.ndarray,
        x_mask: np.ndarray,
        y: np.ndarray,
        class_weights: dict[int, float],
    ) -> dict[str, Any]:
        x = self._build_features(x_embed, x_mask)
        y = np.asarray(y, dtype=int).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y row counts must match.")
        if x.shape[0] == 0:
            raise ValueError("Cannot train with zero samples.")

        unique_labels = sorted({int(v) for v in y.tolist()})
        if len(unique_labels) < 2:
            self._estimator = None
            self._coef = None
            self._intercept = None
            self._constant_class = int(unique_labels[0])
            self._class_ids = np.asarray([self._constant_class], dtype=int)
            return {
                "model_impl": "constant_class_fallback",
                "constant_class": int(self._constant_class),
                "trained_epochs": 0,
                "best_loss": np.nan,
            }

        valid_class_weights = {
            int(k): float(v)
            for k, v in class_weights.items()
            if float(v) > 0.0 and int(k) in unique_labels
        }
        if not valid_class_weights:
            valid_class_weights = None

        estimator = LogisticRegression(
            solver="lbfgs",
            class_weight=valid_class_weights,
            random_state=self.seed,
            max_iter=self.max_iter,
        )
        estimator.fit(x, y)

        self._estimator = estimator
        self._coef = np.asarray(estimator.coef_, dtype=float)
        self._intercept = np.asarray(estimator.intercept_, dtype=float)
        self._constant_class = None
        self._class_ids = np.asarray(estimator.classes_, dtype=int)

        n_iter = int(np.asarray(estimator.n_iter_).max()) if hasattr(estimator, "n_iter_") else self.max_iter
        return {
            "model_impl": self.model_impl,
            "trained_epochs": n_iter,
            "best_loss": np.nan,
        }

    def predict_proba(self, x_embed: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
        x = self._build_features(x_embed, x_mask)

        if self._constant_class is not None:
            out = np.zeros((x.shape[0], self.n_classes), dtype=float)
            out[:, int(self._constant_class)] = 1.0
            return out

        if self._estimator is not None:
            proba = self._estimator.predict_proba(x)
            classes = np.asarray(self._estimator.classes_, dtype=int)
            if proba.shape[1] == self.n_classes and np.array_equal(classes, np.arange(self.n_classes, dtype=int)):
                return proba

            aligned = np.zeros((x.shape[0], self.n_classes), dtype=float)
            for idx, class_id in enumerate(classes.tolist()):
                if 0 <= int(class_id) < self.n_classes:
                    aligned[:, int(class_id)] = proba[:, idx]
            return aligned

        if self._coef is None or self._intercept is None:
            raise RuntimeError("Model is not fitted.")

        logits = x @ self._coef.T + self._intercept
        class_ids = self._class_ids if self._class_ids is not None else np.arange(logits.shape[1], dtype=int)

        if logits.shape[1] == 1 and class_ids.size == 2:
            pos_logits = logits.reshape(-1)
            pos_proba = 1.0 / (1.0 + np.exp(-pos_logits))
            proba = np.stack([1.0 - pos_proba, pos_proba], axis=1)
        else:
            proba = _safe_softmax(logits)
            if class_ids.size != proba.shape[1]:
                class_ids = np.arange(proba.shape[1], dtype=int)

        aligned = np.zeros((x.shape[0], self.n_classes), dtype=float)
        for idx, class_id in enumerate(class_ids.tolist()):
            if 0 <= int(class_id) < self.n_classes and idx < proba.shape[1]:
                aligned[:, int(class_id)] = proba[:, idx]
        return aligned

    def predict(self, x_embed: np.ndarray, x_mask: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x_embed, x_mask)
        return np.argmax(proba, axis=1).astype(int)

    def save(self, path: Path) -> None:
        if self._constant_class is None and self._coef is None and self._estimator is None:
            raise RuntimeError("Cannot save an unfitted model.")

        if self._coef is None and self._estimator is not None:
            self._coef = np.asarray(self._estimator.coef_, dtype=float)
            self._intercept = np.asarray(self._estimator.intercept_, dtype=float)
        if self._class_ids is None and self._estimator is not None:
            self._class_ids = np.asarray(self._estimator.classes_, dtype=int)

        coef = np.asarray(self._coef, dtype=float) if self._coef is not None else np.empty((0, 0), dtype=float)
        intercept = (
            np.asarray(self._intercept, dtype=float)
            if self._intercept is not None
            else np.empty((0,), dtype=float)
        )
        class_ids = (
            np.asarray(self._class_ids, dtype=int)
            if self._class_ids is not None
            else np.empty((0,), dtype=int)
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            artifact_schema_version=np.asarray([self.artifact_schema_version], dtype=int),
            model_impl=np.asarray([self.model_impl]),
            input_dim=np.asarray([self.input_dim], dtype=int),
            n_classes=np.asarray([self.n_classes], dtype=int),
            seed=np.asarray([self.seed], dtype=int),
            max_iter=np.asarray([self.max_iter], dtype=int),
            coef=coef,
            intercept=intercept,
            class_ids=class_ids,
            constant_class=np.asarray(
                [self._constant_class if self._constant_class is not None else -1],
                dtype=int,
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearClassifierHead":
        with np.load(path, allow_pickle=False) as data:
            input_dim = int(data.get("input_dim", np.asarray([0], dtype=int))[0])
            n_classes = int(data.get("n_classes", np.asarray([0], dtype=int))[0])
            seed = int(data.get("seed", np.asarray([42], dtype=int))[0])
            max_iter = int(data.get("max_iter", np.asarray([400], dtype=int))[0])

            model_impl_raw = data.get("model_impl")
            model_impl = str(model_impl_raw[0]) if model_impl_raw is not None else "unknown"

            model = cls(
                input_dim=input_dim,
                n_classes=n_classes,
                seed=seed,
                max_iter=max_iter,
            )
            model.model_impl = model_impl
            class_ids = np.asarray(data.get("class_ids", np.empty((0,), dtype=int)), dtype=int).reshape(-1)
            model._class_ids = class_ids if class_ids.size else None

            if "coef" in data and "intercept" in data:
                coef = np.asarray(data["coef"], dtype=float)
                intercept = np.asarray(data["intercept"], dtype=float)
                model._coef = coef if coef.size else None
                model._intercept = intercept if intercept.size else None
                constant_class = int(data.get("constant_class", np.asarray([-1], dtype=int))[0])
                model._constant_class = None if constant_class < 0 else constant_class
                if model._class_ids is None and model._coef is not None:
                    model._class_ids = np.arange(model._coef.shape[0], dtype=int)
                return model

            raise ValueError(f"Unsupported checkpoint schema at {path}.")


def save_fusion_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)






