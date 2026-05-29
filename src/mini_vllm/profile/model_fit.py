from __future__ import annotations

import argparse
import json
import os
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


DEFAULT_FEATURE_NAMES = (
    "total_query_len",
    "past_len",
    "sum_total_len_times_past_len",
    "num_reqs",
)


def _load_jsonl(path: str) -> list[dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fit_linear(x: np.ndarray, y: np.ndarray) -> LinearModel:
    if len(x) < 2:
        return LinearModel(
            coef=0.0, intercept=float(y.mean()) if len(y) else 0.0, r2=0.0
        )
    coef, intercept = np.polyfit(x, y, 1)
    y_pred = coef * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return LinearModel(coef=float(coef), intercept=float(intercept), r2=float(r2))


def _get_feature_value(row: dict[str, object], feature_name: str) -> float:
    value = row.get(feature_name)
    if value is not None:
        return float(value or 0.0)

    if feature_name == "total_query_len":
        return float(row.get("total_query_tokens", 0.0) or 0.0)
    if feature_name == "past_len":
        return float(row.get("total_context_tokens", 0.0) or 0.0)
    if feature_name == "sum_total_len_times_past_len":
        total_query_len = _get_feature_value(row, "total_query_len")
        past_len = _get_feature_value(row, "past_len")
        total_tokens = row.get("total_tokens")
        if total_tokens is None:
            total_tokens = total_query_len + past_len
        return float(total_tokens) * past_len
    if feature_name == "num_reqs":
        return float(row.get("num_requests", 0.0) or 0.0)
    return 0.0


def _feature_vector(row: dict[str, object], feature_names: Iterable[str]) -> list[float]:
    return [_get_feature_value(row, name) for name in feature_names]


def _build_feature_matrix(
    rows: Iterable[dict[str, object]],
    feature_names: list[str],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    kept_rows: list[dict[str, object]] = []
    feats: list[list[float]] = []
    for row in rows:
        feats.append(_feature_vector(row, feature_names))
        kept_rows.append(row)
    if not feats:
        return np.zeros((0, len(feature_names))), []
    return np.asarray(feats, dtype=float), kept_rows


def _fit_affine(x: np.ndarray, y: np.ndarray, feature_names: list[str]) -> AffineModel:
    if len(x) < 2:
        coef = {name: 0.0 for name in feature_names}
        intercept = float(y.mean()) if len(y) else 0.0
        return AffineModel(coef=coef, intercept=intercept, r2=0.0)

    x_aug = np.column_stack([x, np.ones(len(x))])
    weights, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    coef = {name: float(w) for name, w in zip(feature_names, weights[:-1])}
    intercept = float(weights[-1])
    y_pred = x_aug @ weights
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return AffineModel(coef=coef, intercept=intercept, r2=float(r2))


def _predict_affine(x: np.ndarray, model: AffineModel, feature_names: list[str]) -> np.ndarray:
    weights = np.array([model.coef[name] for name in feature_names], dtype=float)
    return x @ weights + model.intercept


def _predict_max_affine(
    x: np.ndarray, components: list[AffineModel], feature_names: list[str]
) -> np.ndarray:
    if not components:
        return np.zeros(len(x))
    preds = []
    for comp in components:
        weights = np.array([comp.coef[name] for name in feature_names], dtype=float)
        preds.append(x @ weights + comp.intercept)
    return np.max(np.vstack(preds), axis=0)


def _fit_max_affine(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    num_components: int,
    max_iter: int,
    seed: int,
) -> MaxAffineModel:
    if len(x) < 2 or num_components <= 1:
        component = _fit_affine(x, y, feature_names)
        return MaxAffineModel(components=[component], r2=component.r2)

    rng = np.random.default_rng(seed)
    if len(y) < num_components:
        num_components = max(1, len(y))

    quantiles = np.linspace(0.0, 1.0, num_components + 1)[1:-1]
    if quantiles.size:
        split_points = np.quantile(y, quantiles)
        assignments = np.digitize(y, split_points, right=True)
    else:
        assignments = rng.integers(0, num_components, size=len(y))

    components: list[AffineModel] = []
    for _ in range(max_iter):
        components = []
        for k in range(num_components):
            idx = assignments == k
            if idx.sum() < 2:
                components.append(_fit_affine(x, y, feature_names))
            else:
                components.append(_fit_affine(x[idx], y[idx], feature_names))

        preds = np.vstack(
            [
                x @ np.array([c.coef[name] for name in feature_names], dtype=float)
                + c.intercept
                for c in components
            ]
        )
        new_assignments = np.argmax(preds, axis=0)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

    y_pred = _predict_max_affine(x, components, feature_names)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return MaxAffineModel(components=components, r2=float(r2))


def _group_by_type(
    records: Iterable[dict[str, object]]
) -> dict[str, list[dict[str, object]]]:
    by_type: dict[str, list[dict[str, object]]] = {}
    for row in records:
        if row.get("record_type") != "batch":
            continue
        batch_type = str(row.get("batch_type", "unknown"))
        by_type.setdefault(batch_type, []).append(row)
    return by_type


def fit_models(
    records: Iterable[dict[str, object]],
    x_key: str,
) -> dict[str, dict[str, LinearModel]]:
    by_type = _group_by_type(records)
    models: dict[str, dict[str, LinearModel]] = {}
    for batch_type, rows in by_type.items():
        x = np.array([float(r.get(x_key, 0.0)) for r in rows])
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in rows])
        models[batch_type] = {
            "latency_ms": _fit_linear(x, latency),
            "energy_j": _fit_linear(x, energy),
            "avg_power_w": _fit_linear(x, power),
        }
    return models


def fit_max_affine_models(
    records: Iterable[dict[str, object]],
    feature_names: list[str],
    num_components: int,
    max_iter: int,
    seed: int,
) -> dict[str, dict[str, MaxAffineModel]]:
    by_type = _group_by_type(records)
    models: dict[str, dict[str, MaxAffineModel]] = {}
    for batch_type, rows in by_type.items():
        x, kept_rows = _build_feature_matrix(rows, feature_names)
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])
        models[batch_type] = {
            "latency_ms": _fit_max_affine(
                x, latency, feature_names, num_components, max_iter, seed
            ),
            "energy_j": _fit_max_affine(
                x, energy, feature_names, num_components, max_iter, seed
            ),
            "avg_power_w": _fit_max_affine(
                x, power, feature_names, num_components, max_iter, seed
            ),
        }
    return models


def fit_affine_models(
    records: Iterable[dict[str, object]],
    feature_names: list[str],
) -> dict[str, dict[str, AffineModel]]:
    by_type = _group_by_type(records)
    models: dict[str, dict[str, AffineModel]] = {}
    for batch_type, rows in by_type.items():
        x, kept_rows = _build_feature_matrix(rows, feature_names)
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])
        models[batch_type] = {
            "latency_ms": _fit_affine(x, latency, feature_names),
            "energy_j": _fit_affine(x, energy, feature_names),
            "avg_power_w": _fit_affine(x, power, feature_names),
        }
    return models


def save_linear_models(
    path: str,
    models: dict[str, dict[str, dict[str, LinearModel]]],
    model_params: Optional[dict[str, object]] = None,
) -> None:
    serializable = {}
    for x_key, per_type in models.items():
        serializable[x_key] = {}
        for batch_type, model_dict in per_type.items():
            serializable[x_key][batch_type] = {
                name: {
                    "coef": model.coef,
                    "intercept": model.intercept,
                    "r2": model.r2,
                }
                for name, model in model_dict.items()
            }
    if model_params is not None:
        serializable["__model_params__"] = model_params
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_max_affine_models(
    path: str,
    models: dict[str, dict[str, MaxAffineModel]],
    feature_names: list[str],
    model_params: Optional[dict[str, object]] = None,
) -> None:
    serializable: dict[str, object] = {
        "model_type": "max_affine",
        "features": feature_names,
        "combine": "max",
        "models": {},
    }
    if model_params is not None:
        serializable["model_params"] = model_params
    for batch_type, model_dict in models.items():
        serializable["models"][batch_type] = {}
        for name, model in model_dict.items():
            serializable["models"][batch_type][name] = {
                "r2": model.r2,
                "components": [
                    {
                        "coef": comp.coef,
                        "intercept": comp.intercept,
                        "r2": comp.r2,
                    }
                    for comp in model.components
                ],
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_affine_models(
    path: str,
    models: dict[str, dict[str, AffineModel]],
    feature_names: list[str],
    model_params: Optional[dict[str, object]] = None,
) -> None:
    serializable: dict[str, object] = {
        "model_type": "affine",
        "features": feature_names,
        "models": {},
    }
    if model_params is not None:
        serializable["model_params"] = model_params
    for batch_type, model_dict in models.items():
        serializable["models"][batch_type] = {}
        for name, model in model_dict.items():
            serializable["models"][batch_type][name] = {
                "coef": model.coef,
                "intercept": model.intercept,
                "r2": model.r2,
            }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def plot(
    records: Iterable[dict[str, object]],
    models: dict[str, dict[str, dict[str, LinearModel]]],
    x_key: str,
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    by_type = _group_by_type(records)

    for batch_type, rows in by_type.items():
        x = np.array([float(r.get(x_key, 0.0)) for r in rows])
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in rows])

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        _plot_series(ax[0], x, latency, models[x_key][batch_type]["latency_ms"], x_key, "latency (ms)")
        _plot_series(ax[1], x, energy, models[x_key][batch_type]["energy_j"], x_key, "energy (J)")
        _plot_series(ax[2], x, power, models[x_key][batch_type]["avg_power_w"], x_key, "avg power (W)")

        fig.suptitle(f"{batch_type} regression")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{batch_type}_regression.png"), dpi=200)
        plt.close(fig)


def _plot_series(ax, x, y, model: LinearModel, x_label: str, y_label: str) -> None:
    ax.scatter(x, y, s=16)
    if len(x) >= 2:
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = model.coef * x_line + model.intercept
        ax.plot(x_line, y_line, color="tab:red")
        ax.set_title(f"r2={model.r2:.3f}, k={model.coef:.4f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_max_affine(
    records: Iterable[dict[str, object]],
    models: dict[str, dict[str, MaxAffineModel]],
    feature_names: list[str],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    by_type = _group_by_type(records)

    for batch_type, rows in by_type.items():
        x, kept_rows = _build_feature_matrix(rows, feature_names)
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])

        latency_pred = _predict_max_affine(
            x, models[batch_type]["latency_ms"].components, feature_names
        )
        energy_pred = _predict_max_affine(
            x, models[batch_type]["energy_j"].components, feature_names
        )
        power_pred = _predict_max_affine(
            x, models[batch_type]["avg_power_w"].components, feature_names
        )

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        _plot_pred_vs_actual(ax[0], latency, latency_pred, "latency (ms)")
        _plot_pred_vs_actual(ax[1], energy, energy_pred, "energy (J)")
        _plot_pred_vs_actual(ax[2], power, power_pred, "avg power (W)")

        fig.suptitle(f"{batch_type} max-affine regression")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{batch_type}_max_affine.png"), dpi=200)
        plt.close(fig)


def plot_affine(
    records: Iterable[dict[str, object]],
    models: dict[str, dict[str, AffineModel]],
    feature_names: list[str],
    output_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    by_type = _group_by_type(records)

    for batch_type, rows in by_type.items():
        x, kept_rows = _build_feature_matrix(rows, feature_names)
        latency = np.array([float(r.get("latency_ms", 0.0)) for r in kept_rows])
        energy = np.array([float(r.get("energy_j", 0.0)) for r in kept_rows])
        power = np.array([float(r.get("avg_power_w", 0.0)) for r in kept_rows])

        latency_pred = _predict_affine(x, models[batch_type]["latency_ms"], feature_names)
        energy_pred = _predict_affine(x, models[batch_type]["energy_j"], feature_names)
        power_pred = _predict_affine(x, models[batch_type]["avg_power_w"], feature_names)

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        _plot_pred_vs_actual(ax[0], latency, latency_pred, "latency (ms)")
        _plot_pred_vs_actual(ax[1], energy, energy_pred, "energy (J)")
        _plot_pred_vs_actual(ax[2], power, power_pred, "avg power (W)")
        fig.suptitle(f"{batch_type} affine regression")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{batch_type}_affine.png"), dpi=200)
        plt.close(fig)


def _plot_pred_vs_actual(ax, y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    ax.scatter(y_true, y_pred, s=16)
    if len(y_true) >= 2:
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], color="tab:red")
    ax.set_xlabel(f"actual {label}")
    ax.set_ylabel(f"pred {label}")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit power/latency/energy models.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--x_keys",
        type=str,
        default="total_tokens",
        help="Comma-separated list, e.g. total_tokens,total_query_tokens",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=("linear_1d", "affine", "max_affine"),
        default="linear_1d",
    )
    parser.add_argument(
        "--feature_keys",
        type=str,
        default=",".join(DEFAULT_FEATURE_NAMES),
        help=(
            "Comma-separated multi-input features for affine/max_affine, "
            "e.g. total_query_len,past_len,num_reqs"
        ),
    )
    parser.add_argument("--num_components", type=int, default=1)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_json", type=str, default="energy_latency_power_model.json")
    parser.add_argument("--plot_dir", type=str, default="profile_plots")
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    records = _load_jsonl(args.input)
    if args.model_type == "linear_1d":
        x_keys = [k.strip() for k in args.x_keys.split(",") if k.strip()]
        if not x_keys:
            raise ValueError("no x_keys provided")
        models: dict[str, dict[str, dict[str, LinearModel]]] = {}
        for x_key in x_keys:
            models[x_key] = fit_models(records, x_key)
        save_linear_models(
            args.out_json,
            models,
            model_params={
                "model_type": "linear_1d",
                "x_keys": x_keys,
            },
        )
        if not args.no_plots:
            for x_key in x_keys:
                plot_dir = os.path.join(args.plot_dir, x_key)
                plot(records, models, x_key, plot_dir)
        return 0

    feature_names = [k.strip() for k in args.feature_keys.split(",") if k.strip()]
    if not feature_names:
        raise ValueError("no feature_keys provided")

    if args.model_type == "affine":
        models = fit_affine_models(records, feature_names)
        save_affine_models(
            args.out_json,
            models,
            feature_names,
            model_params={
                "model_type": "affine",
                "feature_keys": feature_names,
            },
        )
        if not args.no_plots:
            plot_affine(records, models, feature_names, args.plot_dir)
        return 0

    models = fit_max_affine_models(
        records,
        feature_names=feature_names,
        num_components=args.num_components,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    save_max_affine_models(
        args.out_json,
        models,
        feature_names,
        model_params={
            "model_type": "max_affine",
            "feature_keys": feature_names,
            "num_components": args.num_components,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
    )
    if not args.no_plots:
        plot_max_affine(records, models, feature_names, args.plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
