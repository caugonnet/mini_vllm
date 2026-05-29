from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class LinearModel:
    coef: float
    intercept: float
    r2: float


@dataclass
class AffineModel:
    coef: dict[str, float]
    intercept: float
    r2: float


@dataclass
class MaxAffineModel:
    components: list[AffineModel]
    r2: float


FEATURE_NAMES = (
    "total_query_len",
    "past_len",
    "sum_total_len_times_past_len",
    "num_reqs",
)


class ProfileModeler:
    def __init__(
        self,
        model_type: str = "linear_1d",
        num_components: int = 1,
        max_iter: int = 50,
        seed: int = 0,
    ) -> None:
        self.model_type = model_type
        self.num_components = num_components
        self.max_iter = max_iter
        self.seed = seed

    def load(self, path: str) -> list[dict[str, object]]:
        if path.endswith(".csv"):
            return self._load_csv(path)
        return self._load_jsonl(path)

    def fit_models(
        self, records: Iterable[dict[str, object]]
    ) -> dict[str, dict[str, object]]:
        by_type: dict[str, list[dict[str, object]]] = {}
        for row in records:
            if row.get("record_type") != "batch":
                continue
            batch_type = str(row.get("batch_type", "unknown"))
            by_type.setdefault(batch_type, []).append(row)

        if self.model_type == "max_affine":
            models: dict[str, dict[str, MaxAffineModel]] = {}
            for batch_type, rows in by_type.items():
                x, kept_rows = self._build_feature_matrix(rows)
                latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
                energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
                power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])
                models[batch_type] = {
                    "latency_ms": self._fit_max_affine(x, latency),
                    "energy_j": self._fit_max_affine(x, energy),
                    "avg_power_w": self._fit_max_affine(x, power),
                }
            return models

        models = {}
        for batch_type, rows in by_type.items():
            x = np.array([float(r.get("total_tokens", 0)) for r in rows])
            latency = np.array([float(r.get("latency_ms", 0.0)) for r in rows])
            energy = np.array([float(r.get("energy_j", 0.0)) for r in rows])
            models[batch_type] = {
                "latency_ms": self._fit_linear(x, latency),
                "energy_j": self._fit_linear(x, energy),
            }
        return models

    def save_models(self, path: str, models: dict[str, dict[str, object]]) -> None:
        if self.model_type == "max_affine":
            serializable: dict[str, object] = {
                "model_type": "max_affine",
                "features": list(FEATURE_NAMES),
                "combine": "max",
                "models": {},
            }
            for batch_type, model_dict in models.items():
                serializable["models"][batch_type] = {}
                for name, model in model_dict.items():
                    max_model = model
                    serializable["models"][batch_type][name] = {
                        "r2": max_model.r2,
                        "components": [
                            {
                                "coef": comp.coef,
                                "intercept": comp.intercept,
                                "r2": comp.r2,
                            }
                            for comp in max_model.components
                        ],
                    }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            return

        serializable = {}
        for batch_type, model_dict in models.items():
            serializable[batch_type] = {
                name: {
                    "coef": model.coef,
                    "intercept": model.intercept,
                    "r2": model.r2,
                }
                for name, model in model_dict.items()
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    def plot(self, records: Iterable[dict[str, object]], output_prefix: str) -> None:
        import matplotlib.pyplot as plt

        by_type: dict[str, list[dict[str, object]]] = {}
        for row in records:
            if row.get("record_type") != "batch":
                continue
            batch_type = str(row.get("batch_type", "unknown"))
            by_type.setdefault(batch_type, []).append(row)

        if self.model_type == "max_affine":
            for batch_type, rows in by_type.items():
                x, kept_rows = self._build_feature_matrix(rows)
                latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
                energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
                power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])

                models = self.fit_models(rows)
                latency_pred = self._predict_max_affine(
                    x, models[batch_type]["latency_ms"].components
                )
                energy_pred = self._predict_max_affine(
                    x, models[batch_type]["energy_j"].components
                )
                power_pred = self._predict_max_affine(
                    x, models[batch_type]["avg_power_w"].components
                )

                fig, ax = plt.subplots(1, 3, figsize=(14, 4))
                self._plot_pred_vs_actual(ax[0], latency, latency_pred, "latency (ms)")
                self._plot_pred_vs_actual(ax[1], energy, energy_pred, "energy (J)")
                self._plot_pred_vs_actual(ax[2], power, power_pred, "avg power (W)")

                fig.tight_layout()
                fig.savefig(f"{output_prefix}_{batch_type}_max_affine.png", dpi=200)
                plt.close(fig)
            return

        for batch_type, rows in by_type.items():
            x = np.array([float(r.get("total_tokens", 0)) for r in rows])
            latency = np.array([float(r.get("latency_ms", 0.0)) for r in rows])
            energy = np.array([float(r.get("energy_j", 0.0)) for r in rows])

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(x, latency, s=16)
            ax[0].set_title(f"{batch_type} latency")
            ax[0].set_xlabel("total tokens")
            ax[0].set_ylabel("latency (ms)")

            ax[1].scatter(x, energy, s=16)
            ax[1].set_title(f"{batch_type} energy")
            ax[1].set_xlabel("total tokens")
            ax[1].set_ylabel("energy (J)")

            fig.tight_layout()
            fig.savefig(f"{output_prefix}_{batch_type}.png", dpi=200)
            plt.close(fig)

    def _fit_linear(self, x: np.ndarray, y: np.ndarray) -> LinearModel:
        if len(x) < 2:
            return LinearModel(coef=0.0, intercept=float(y.mean()) if len(y) else 0.0, r2=0.0)
        coef, intercept = np.polyfit(x, y, 1)
        y_pred = coef * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return LinearModel(coef=float(coef), intercept=float(intercept), r2=float(r2))

    def _feature_vector(self, row: dict[str, object]) -> list[float]:
        total_query_len = row.get("total_query_len")
        past_len = row.get("past_len")
        sum_total_len_times_past_len = row.get("sum_total_len_times_past_len")
        num_reqs = row.get("num_reqs")

        if total_query_len is None:
            total_query_len = row.get("total_query_tokens", 0.0)
        if past_len is None:
            past_len = row.get("total_context_tokens", 0.0)
        if sum_total_len_times_past_len is None:
            total_tokens = row.get("total_tokens")
            if total_tokens is None:
                total_tokens = float(total_query_len) + float(past_len)
            sum_total_len_times_past_len = float(total_tokens) * float(past_len)
        if num_reqs is None:
            num_reqs = row.get("num_requests", 0.0)

        return [
            float(total_query_len or 0.0),
            float(past_len or 0.0),
            float(sum_total_len_times_past_len or 0.0),
            float(num_reqs or 0.0),
        ]

    def _build_feature_matrix(
        self, rows: Iterable[dict[str, object]]
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        kept_rows: list[dict[str, object]] = []
        feats: list[list[float]] = []
        for row in rows:
            feats.append(self._feature_vector(row))
            kept_rows.append(row)
        if not feats:
            return np.zeros((0, len(FEATURE_NAMES))), []
        return np.asarray(feats, dtype=float), kept_rows

    def _fit_affine(self, x: np.ndarray, y: np.ndarray) -> AffineModel:
        if len(x) < 2:
            coef = {name: 0.0 for name in FEATURE_NAMES}
            intercept = float(y.mean()) if len(y) else 0.0
            return AffineModel(coef=coef, intercept=intercept, r2=0.0)

        x_aug = np.column_stack([x, np.ones(len(x))])
        weights, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
        coef = {name: float(w) for name, w in zip(FEATURE_NAMES, weights[:-1])}
        intercept = float(weights[-1])
        y_pred = x_aug @ weights
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return AffineModel(coef=coef, intercept=intercept, r2=float(r2))

    def _predict_max_affine(self, x: np.ndarray, components: list[AffineModel]) -> np.ndarray:
        if not components:
            return np.zeros(len(x))
        preds = []
        for comp in components:
            weights = np.array([comp.coef[name] for name in FEATURE_NAMES], dtype=float)
            preds.append(x @ weights + comp.intercept)
        return np.max(np.vstack(preds), axis=0)

    def _fit_max_affine(self, x: np.ndarray, y: np.ndarray) -> MaxAffineModel:
        if len(x) < 2 or self.num_components <= 1:
            component = self._fit_affine(x, y)
            return MaxAffineModel(components=[component], r2=component.r2)

        rng = np.random.default_rng(self.seed)
        num_components = min(self.num_components, len(y))
        quantiles = np.linspace(0.0, 1.0, num_components + 1)[1:-1]
        if quantiles.size:
            split_points = np.quantile(y, quantiles)
            assignments = np.digitize(y, split_points, right=True)
        else:
            assignments = rng.integers(0, num_components, size=len(y))

        components: list[AffineModel] = []
        for _ in range(self.max_iter):
            components = []
            for k in range(num_components):
                idx = assignments == k
                if idx.sum() < 2:
                    components.append(self._fit_affine(x, y))
                else:
                    components.append(self._fit_affine(x[idx], y[idx]))

            preds = np.vstack(
                [
                    x @ np.array([c.coef[name] for name in FEATURE_NAMES], dtype=float)
                    + c.intercept
                    for c in components
                ]
            )
            new_assignments = np.argmax(preds, axis=0)
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

        y_pred = self._predict_max_affine(x, components)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return MaxAffineModel(components=components, r2=float(r2))

    def _plot_pred_vs_actual(
        self, ax, y_true: np.ndarray, y_pred: np.ndarray, label: str
    ) -> None:
        ax.scatter(y_true, y_pred, s=16)
        if len(y_true) >= 2:
            min_v = min(y_true.min(), y_pred.min())
            max_v = max(y_true.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], color="tab:red")
        ax.set_xlabel(f"actual {label}")
        ax.set_ylabel(f"pred {label}")

    def _load_jsonl(self, path: str) -> list[dict[str, object]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_csv(self, path: str) -> list[dict[str, object]]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
