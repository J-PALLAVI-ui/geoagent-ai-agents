"""Model training and evaluation pipeline for GeoAgent AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from geoagent_ai.config import ARTIFACTS_DIR, FEATURE_COLUMNS
from geoagent_ai.ml.dataset_generator import generate_training_dataframe

CLASSIFIER_PATH = ARTIFACTS_DIR / "disaster_classifier.joblib"
REGRESSOR_PATH = ARTIFACTS_DIR / "risk_regressor.joblib"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"


def _build_classifier() -> tuple[Any, str]:
    """Construct ensemble classifier (XGBoost + RF + SVM when xgboost is available)."""
    rf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_iter=120, random_state=42)
    svm_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", CalibratedClassifierCV(SVC(kernel="rbf", C=3.0, random_state=42), cv=3)),
        ]
    )

    estimators: list[tuple[str, Any]] = [("rf", rf), ("hgb", hgb), ("svm", svm_pipe)]
    model_desc = "Ensemble(RandomForest + HistGradientBoosting + RBF-SVM)"

    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.08,
            eval_metric="mlogloss",
            random_state=42,
        )
        estimators.append(("xgb", xgb))
        model_desc = "Ensemble(RandomForest + HistGradientBoosting + RBF-SVM + XGBoost)"
    except ImportError:
        pass

    return VotingClassifier(estimators=estimators, voting="soft"), model_desc


def _build_regressor() -> tuple[Any, str]:
    rf_reg = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=1)
    hgb_reg = HistGradientBoostingRegressor(max_iter=120, random_state=42)
    estimators: list[tuple[str, Any]] = [("rf", rf_reg), ("hgb", hgb_reg)]
    model_desc = "EnsembleRegressor(RandomForest + HistGradientBoosting)"

    try:
        from xgboost import XGBRegressor  # type: ignore

        xgb_reg = XGBRegressor(n_estimators=140, max_depth=5, learning_rate=0.08, random_state=42)
        estimators.append(("xgb", xgb_reg))
        model_desc = "EnsembleRegressor(RandomForest + HistGradientBoosting + XGBoost)"
    except ImportError:
        pass

    return VotingRegressor(estimators=estimators), model_desc


def train_and_save_models(samples_per_class: int = 320) -> dict[str, Any]:
    """Train disaster classifier and risk severity regressor and persist artifacts."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_training_dataframe(samples_per_class=samples_per_class)

    X = df[FEATURE_COLUMNS]
    y_cls = df["disaster_type"]
    y_reg = df["severity_score"]

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf, clf_desc = _build_classifier()
    clf.fit(X_train, y_cls_train)
    cls_preds = clf.predict(X_test)
    cls_acc = float(accuracy_score(y_cls_test, cls_preds))
    cls_f1 = float(f1_score(y_cls_test, cls_preds, average="weighted"))

    reg, reg_desc = _build_regressor()
    reg.fit(X_train, y_reg_train)
    reg_preds = reg.predict(X_test)
    reg_mae = float(mean_absolute_error(y_reg_test, reg_preds))
    reg_r2 = float(r2_score(y_reg_test, reg_preds))

    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(reg, REGRESSOR_PATH)

    metrics = {
        "classifier_model": clf_desc,
        "regressor_model": reg_desc,
        "features": FEATURE_COLUMNS,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classification_accuracy": round(cls_acc, 4),
        "classification_weighted_f1": round(cls_f1, 4),
        "severity_mae": round(reg_mae, 3),
        "severity_r2": round(reg_r2, 4),
        "classes": list(clf.classes_),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def ensure_models_trained() -> dict[str, Any]:
    """Load metrics if artifacts exist, otherwise train models immediately."""
    if CLASSIFIER_PATH.exists() and REGRESSOR_PATH.exists() and METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train_and_save_models()


if __name__ == "__main__":
    summary = train_and_save_models()
    print(json.dumps(summary, indent=2))
