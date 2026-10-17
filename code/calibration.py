"""Finite-sample calibration certificates. Requires only NumPy.

See report.pdf for assumptions/proofs. The confidence construction uses raw
counts; pseudocounts occur only in the separately reported point estimate.
"""
from __future__ import annotations
import math
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x.sum(axis=-1) <= 0):
        raise ValueError("Expected finite nonnegative weights with positive sum")
    return x / x.sum(axis=-1, keepdims=True)


def counts_array(x):
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2) or x.shape[-1] < 1:
        raise ValueError("Counts must be a nonempty vector or row matrix")
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError("Counts must be finite nonnegative integers")
    if np.any(x.sum(axis=-1) <= 0):
        raise ValueError("Every count row must have positive depth")
    return x


def bernoulli_kl(x, p):
    """d(x || p), with the conventional boundary limits."""
    x, p = np.broadcast_arrays(np.asarray(x, float), np.asarray(p, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(x == 0, 0, x * (np.log(x) - np.log(p)))
        b = np.where(x == 1, 0, (1-x) * (np.log1p(-x) - np.log1p(-p)))
    return np.maximum(a + b, 0)


def binomial_kl_interval(k, n, tail):
    """Two-sided Chernoff interval, each tail has failure probability <= tail.

    Not Clopper-Pearson: this is a conservative exponential-tail inversion.
    Bisection returns OUTER endpoints. Floating-point arithmetic is not a
    formally verified interval arithmetic implementation.
    """
    if not 0 < tail < 0.5:
        raise ValueError("tail must be between 0 and 0.5")
    k, n = np.broadcast_arrays(np.asarray(k, float), np.asarray(n, float))
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("Expected 0 <= k <= n and n > 0")
    x = k / n
    threshold = -math.log(tail) / n
    low, high = np.zeros_like(x), x.copy()
    for _ in range(60):
        mid = (low + high) / 2
        outside = bernoulli_kl(x, mid) > threshold
        low = np.where(outside, mid, low)
        high = np.where(outside, high, mid)
    lower = np.where(k == 0, 0, low)
    low, high = x.copy(), np.ones_like(x)
    for _ in range(60):
        mid = (low + high) / 2
        outside = bernoulli_kl(x, mid) > threshold
        high = np.where(outside, mid, high)
        low = np.where(outside, low, mid)
    upper = np.where(k == n, 1, high)
    return lower, upper


def normalized_box(lower_weights, upper_weights):
    """Outer bounds after normalizing positive weights; permits +inf uppers."""
    a, b = np.broadcast_arrays(np.asarray(lower_weights, float),
                               np.asarray(upper_weights, float))
    if np.any(a < 0) or np.any(b < a) or np.any(~np.isfinite(a)) or np.any(np.isnan(b)):
        raise ValueError("Invalid weight box")
    if a.shape[-1] == 1:
        return np.ones_like(a), np.ones_like(b)
    lower, upper = np.zeros_like(a), np.ones_like(b)
    for i in range(a.shape[-1]):
        others = [j for j in range(a.shape[-1]) if j != i]
        sa, sb = a[..., others].sum(axis=-1), b[..., others].sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            lower[..., i] = np.where(a[..., i] == 0, 0, a[..., i] / (a[..., i] + sb))
            upper[..., i] = np.where(np.isinf(b[..., i]), 1,
                                     b[..., i] / (b[..., i] + sa))
    return np.clip(lower, 0, 1), np.clip(upper, 0, 1)


def calibration_intervals(y, c, known_composition, alpha=0.05):
    """Simultaneous 1-alpha intervals for all relative abundances of ONE sample.

    c: one full-support calibration composition's counts; rows may be pooled
    only when their true composition AND bias vector are the same.
    y/c may have broadcast-compatible batch dimensions.
    """
    y, c = counts_array(y), counts_array(c)
    r = np.asarray(known_composition, dtype=float)
    if r.ndim != 1 or len(r) != y.shape[-1] or len(r) != c.shape[-1]:
        raise ValueError("Taxon dimensions must match")
    if np.any(~np.isfinite(r)) or np.any(r <= 0) or not np.isclose(r.sum(), 1, atol=1e-12):
        raise ValueError("Calibration composition must be strictly positive and sum to one")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    q = len(r)
    tail = alpha / (4*q)  # 2 tails x q taxa x 2 count tables
    yl, yu = binomial_kl_interval(y, y.sum(axis=-1, keepdims=True), tail)
    cl, cu = binomial_kl_interval(c, c.sum(axis=-1, keepdims=True), tail)
    a = r * yl / cu
    with np.errstate(divide="ignore"):
        b = r * yu / cl
    return normalized_box(a, b)


def fixed_bias_intervals(y, bias, alpha=0.05):
    """Reference method: valid only if bias is truly known (not estimated)."""
    y = counts_array(y)
    bias = np.asarray(bias, float)
    if np.any(bias <= 0) or np.any(~np.isfinite(bias)) or bias.shape[-1] != y.shape[-1]:
        raise ValueError("Invalid bias vector")
    q = y.shape[-1]
    lo, hi = binomial_kl_interval(y, y.sum(axis=-1, keepdims=True), alpha/(2*q))
    return normalized_box(lo/bias, hi/bias)


def point_estimate(y, c, known_composition, pseudocount=0.5):
    """Smoothed descriptive point estimate; no unbiasedness is claimed."""
    y, c = counts_array(y), counts_array(c)
    if pseudocount <= 0:
        raise ValueError("Use a strictly positive pseudocount")
    return normalize((y+pseudocount) * np.asarray(known_composition) / (c+pseudocount))


def expand_for_drift(lower, upper, log_range):
    """If osc(log(b_target/b_cal)) <= log_range, widen composition intervals."""
    if not math.isfinite(log_range) or log_range < 0 or log_range > 700:
        raise ValueError("Expected a finite nonnegative log range <= 700")
    e = math.exp(log_range)
    lower, upper = np.asarray(lower), np.asarray(upper)
    return lower / (lower + e*(1-lower)), e*upper / (1-upper+e*upper)


def drift_l1_bound(log_range):
    if not math.isfinite(log_range) or log_range < 0:
        raise ValueError("Expected finite nonnegative log range")
    return 2*math.tanh(log_range/4)


def l1_radius(point, lower, upper):
    """Max L1 distance over the interval box intersected with the simplex.

    Exact finite subset formula (in real arithmetic). Exponential in taxa;
    deliberately limited to q <= 12. Not the sharp radius of the original
    multinomial confidence set, which is smaller than this outer box.
    """
    point, lower, upper = np.broadcast_arrays(point, lower, upper)
    q = point.shape[-1]
    if q > 12:
        raise ValueError("Subset certificate limited to 12 taxa")
    if np.any(lower > upper+1e-12) or np.any(lower.sum(-1)>1+1e-10) or np.any(upper.sum(-1)<1-1e-10):
        raise ValueError("Infeasible simplex box")
    out = np.zeros(point.shape[:-1])
    for mask in range(1 << q):
        chosen = [i for i in range(q) if mask & (1 << i)]
        rest = [i for i in range(q) if not mask & (1 << i)]
        max_mass = np.minimum(upper[..., chosen].sum(-1), 1-lower[..., rest].sum(-1))
        out = np.maximum(out, 2*(max_mass-point[..., chosen].sum(-1)))
    return np.clip(out+1e-12, 0, 2)
