"""Can a left touchdown be detected from proprioception?

The reference phase is, by definition, the fraction of the cycle elapsed since
the last left touchdown. Every estimator tried so far regresses that phase
directly from the 30-frame context, and all of them stall near 0.78 rad. That
may not be an information limit at all: absolute phase is a multivalued,
time-sensitive quantity, whereas a touchdown is a sparse event with a sharp
boundary.

If touchdowns are detectable, the clock can be built as "detect the event, then
integrate since the last one" -- which sidesteps the continuous regression
entirely. This probe measures only the premise: how separable touchdown frames
are from everything else, train -> validation, using a linear model so the
number is a floor rather than a ceiling.

A high AUC does not prove the event route will pass the gate; it would only
justify building it. A low AUC closes the route cheaply.

Reads train and validation only. The test split is never loaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from scripts.train_g1_phase_encoder_v2_1b import build_context_windows, split_arrays

DATA_PATH = ROOT / "results" / "g1_phase_dataset_20261019" / "phase_dataset.npz"
OUT_PATH = ROOT / "results" / "g1_phase_gate_v4_2" / "touchdown_detectability.json"

CONTEXT_FRAMES = 30
PCA_COMPONENTS = 32
RANDOM_STATE = 20261020


def main() -> int:
    archive = np.load(DATA_PATH, allow_pickle=False)
    train = split_arrays(archive, "train")
    validation = split_arrays(archive, "validation")

    train_windows, _ = build_context_windows(train, CONTEXT_FRAMES)
    validation_windows, validation_valid = build_context_windows(validation, CONTEXT_FRAMES)

    train_flat = train_windows.reshape(len(train_windows), -1).astype(np.float32)
    validation_flat = validation_windows.reshape(len(validation_windows), -1).astype(np.float32)
    mean = train_flat.mean(axis=0, keepdims=True)
    scale = train_flat.std(axis=0, keepdims=True)
    train_flat = (train_flat - mean) / np.where(scale < 1e-8, 1.0, scale)
    validation_flat = (validation_flat - mean) / np.where(scale < 1e-8, 1.0, scale)

    components = min(PCA_COMPONENTS, train_flat.shape[0], train_flat.shape[1])
    pca = PCA(n_components=components, svd_solver="randomized", random_state=RANDOM_STATE).fit(train_flat)
    train_reduced = pca.transform(train_flat)
    validation_reduced = pca.transform(validation_flat)

    train_label = train["left_touchdown"].astype(bool)
    validation_label = validation["left_touchdown"].astype(bool)
    if train_label.sum() < 10 or validation_label.sum() < 10:
        raise ValueError("too few touchdown frames to evaluate")

    model = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE)
    model.fit(train_reduced, train_label)
    probability = model.predict_proba(validation_reduced)[:, 1]

    mask = validation_valid
    label = validation_label[mask]
    score = probability[mask]
    prediction = score >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(label, prediction, average="binary", zero_division=0)

    payload = {
        "schema": "g1.touchdown_detectability.v1",
        "split": "train -> validation",
        "test_split_read": False,
        "settings": {
            "context_frames": CONTEXT_FRAMES,
            "pca_components": int(components),
            "model": "logistic regression, class_weight=balanced, C=1.0",
            "evaluated_frames": int(mask.sum()),
        },
        "class_balance": {
            "train_touchdown_frames": int(train_label.sum()),
            "train_frames": int(len(train_label)),
            "validation_touchdown_frames": int(label.sum()),
            "validation_frames": int(len(label)),
        },
        "result": {
            "roc_auc": float(roc_auc_score(label, score)),
            "average_precision": float(average_precision_score(label, score)),
            "precision_at_half": float(precision),
            "recall_at_half": float(recall),
            "f1_at_half": float(f1),
            "baseline_precision": float(label.mean()),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    print(f"touchdown frames: train {payload['class_balance']['train_touchdown_frames']}/"
          f"{payload['class_balance']['train_frames']}, validation {label.sum()}/{len(label)}")
    print(f"ROC AUC            : {payload['result']['roc_auc']:.4f}")
    print(f"average precision  : {payload['result']['average_precision']:.4f}  "
          f"(baseline {payload['result']['baseline_precision']:.4f})")
    print(f"precision / recall at 0.5 : {precision:.4f} / {recall:.4f}  (F1 {f1:.4f})")
    print(f"-> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
