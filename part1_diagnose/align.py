"""Euclidean alignment (He & Wu, 2019).

For each subject s: R_s = mean trial covariance, X <- R_s^{-1/2} X, so every subject's mean covariance
becomes the identity. Uses no labels, but it does use the TEST subject's unlabelled trials, so any
result that uses it is TRANSDUCTIVE and must be labelled as such.
"""
import numpy as np


def inv_sqrt(R, eps=1e-10):
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, eps * w.max())
    return (V / np.sqrt(w)) @ V.T


def euclidean_align(X, groups, limit=None):
    """X (n, ch, t), groups (n,). limit: optional {subject: K} -> estimate R_s from that subject's
    first K trials only (stricter "calibration" variant); other subjects use all their trials."""
    Xa = np.empty_like(X, dtype=np.float32)
    for s in np.unique(groups):
        idx = np.flatnonzero(groups == s)
        est = idx[:limit[s]] if limit and s in limit else idx
        Z = X[est].astype(np.float64)
        R = np.einsum("nct,ndt->cd", Z, Z) / (Z.shape[0] * Z.shape[2])
        Xa[idx] = np.einsum("cd,ndt->nct", inv_sqrt(R), X[idx].astype(np.float64))
    return Xa
