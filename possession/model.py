"""Frame classifier: P(H), P(A), P(D) per frame from possession/features.py.

LightGBM, CPU. Training is not required to be bit-reproducible (organiser ruling, issue #11), but
inference from the committed weights is deterministic on CPU.
"""
import lightgbm as lgb
import numpy as np

PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, seed=0, deterministic=True, force_row_wise=True, num_threads=8,
              verbose=-1)
N_ROUNDS = 300


def train(X, y, params=None, rounds=N_ROUNDS):
    p = dict(PARAMS, **(params or {}))
    return lgb.train(p, lgb.Dataset(X, y.astype(np.int32), free_raw_data=True), rounds)


def predict(booster, X, X_swapped):
    """Average of the normal view and the team-swapped view (with H/A mapped back)."""
    p = booster.predict(X, num_threads=8)
    q = booster.predict(X_swapped, num_threads=8)
    out = np.empty_like(p)
    out[:, 0] = (p[:, 0] + q[:, 1]) / 2
    out[:, 1] = (p[:, 1] + q[:, 0]) / 2
    out[:, 2] = (p[:, 2] + q[:, 2]) / 2
    return out


def save(booster, path):
    booster.save_model(path)


def load(path):
    return lgb.Booster(model_file=path)
