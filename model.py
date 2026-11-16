"""Shared model code: synthetic data (mirrors the PHP simulator), training, loading, prediction."""
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

FEATURES = ["mean_iat", "iat_cv", "burst_freq", "burst_period_cv", "pkt_size_std", "duration"]
MODEL_PATH = Path(__file__).parent / "model.joblib"
BURST_GAP, MIN_BURST, NO_PATTERN, WINDOW = 0.05, 3, 1.5, 60


# ---------- feature extraction (same definitions as FeatureExtractor.php) ----------
def extract_features(packets):
    packets = sorted(packets)
    t = np.array([p[0] for p in packets]); sizes = np.array([p[1] for p in packets], dtype=float)
    iat = np.diff(t); duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    mean_iat = float(iat.mean()) if len(iat) else 0.0
    iat_cv = float(iat.std() / mean_iat) if mean_iat > 0 else 0.0
    starts, run, run_start = [], 1, t[0]
    for i, g in enumerate(iat):
        if g < BURST_GAP:
            run += 1
        else:
            if run >= MIN_BURST: starts.append(run_start)
            run, run_start = 1, t[i + 1]
    if run >= MIN_BURST: starts.append(run_start)
    period_cv = NO_PATTERN
    if len(starts) >= 3:
        gaps = np.diff(starts); period_cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else NO_PATTERN
    return {"mean_iat": mean_iat, "iat_cv": iat_cv, "burst_freq": len(starts) / duration if duration > 0 else 0.0,
            "burst_period_cv": period_cv, "pkt_size_std": float(sizes.std()), "duration": duration}


# ---------- simulator (port of TrafficCapture::simulate) ----------
def _size(rng, m, sd): return int(max(40, min(1500, rng.gauss(m, sd))))


def simulate(kind, rng):
    p, t = [], rng.random()
    if kind == "lowrate":
        period = 0.8 + rng.random() * 1.2
        jitter = 0.30 if rng.random() < 0.25 else 0.08
        varied = rng.random() < 0.20
        while len(p) < WINDOW:
            for _ in range(rng.randint(6, 14)):
                if len(p) >= WINDOW: break
                t += 0.001 + rng.random() * 0.009
                p.append((t, rng.randint(60, 600) if varied else rng.randint(60, 150)))
            t += period * (1 + (rng.random() * 2 - 1) * jitter)
        return p, 1
    if kind == "flash":
        polling = rng.random() < 0.30
        period = 0.8 + rng.random() * 1.5
        while len(p) < WINDOW:
            burst = rng.randint(4, 9) if polling else rng.randint(3, 20)
            for _ in range(burst):
                if len(p) >= WINDOW: break
                t += (0.002 + rng.random() * 0.02) if polling else (0.002 + rng.random() * 0.03)
                p.append((t, _size(rng, 400, 200) if polling else _size(rng, 700, 350)))
            t += period * (1 + (rng.random() * 2 - 1) * 0.25) if polling else rng.expovariate(1 / 1.5)
        return p, 0
    mean = 0.08 + rng.random() * 0.5
    while len(p) < WINDOW:
        t += rng.expovariate(1 / mean); p.append((t, _size(rng, 600, 250)))
    return p, 0


def synthetic_dataset(n=2000, seed=7):
    rng = random.Random(seed); rows = []
    for i in range(n):
        r = i % 10
        kind = "normal" if r < 4 else ("flash" if r < 6 else "lowrate")
        pk, y = simulate(kind, rng)
        rows.append({**extract_features(pk), "label": y})
    return pd.DataFrame(rows)


# ---------- training / loading ----------
def train(df, n_estimators=200, max_depth=None, test_size=0.2, seed=42):
    X, y = df[FEATURES], df["label"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    metrics = {"accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred, zero_division=0),
               "recall": recall_score(yte, pred, zero_division=0), "f1": f1_score(yte, pred, zero_division=0),
               "fpr": fp / max(1, fp + tn), "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
               "trained_on": len(Xtr), "tested_on": len(Xte),
               "importance": dict(zip(FEATURES, map(float, clf.feature_importances_)))}
    joblib.dump({"model": clf, "metrics": metrics}, MODEL_PATH)
    return clf, metrics


def load():
    if not MODEL_PATH.exists():
        train(synthetic_dataset())
    return joblib.load(MODEL_PATH)


def predict_proba(bundle, features):
    row = pd.DataFrame([[float(features[k]) for k in FEATURES]], columns=FEATURES)
    return float(bundle["model"].predict_proba(row)[0][1])
