"""Canonical Gram (discrete orthogonal polynomial) fitting, ported
verbatim (minus unused decompositions) from gp-paper
src/lib/encoders/gram.py - the published instrument's parameterization.
Coefficients are computed by direct projection: orthogonality at uniform
points means no linear system. g0 is exactly the mean; g1 exactly the
linear component.
"""
import math

import numpy as np
from scipy.special import comb


def gram_polynomial_value(k, i, n):
    result = 0.0
    for s in range(k + 1):
        if i < s or n - 1 < s:
            continue
        result += ((-1) ** s * comb(k, s, exact=True)
                   * comb(k + s, s, exact=True) * comb(i, s, exact=True)
                   / comb(n - 1, s, exact=True))
    return result


def gram_basis_matrix_fast(n, degree):
    actual_degree = min(degree, n - 1)
    G = np.zeros((n, actual_degree + 1))
    if n == 1:
        G[0, 0] = 1.0
        return G
    for k in range(actual_degree + 1):
        for i in range(n):
            G[i, k] = gram_polynomial_value(k, i, n)
    return G


def gram_fit(embeddings, degree=5):
    """(n_tokens, dim) -> (degree+1, dim) coefficients."""
    n_tokens, emb_dim = embeddings.shape
    if n_tokens == 0:
        return np.zeros((degree + 1, emb_dim))
    if n_tokens == 1:
        coeffs = np.zeros((degree + 1, emb_dim))
        coeffs[0] = embeddings[0]
        return coeffs
    actual_degree = min(degree, n_tokens - 1)
    G = gram_basis_matrix_fast(n_tokens, actual_degree)
    norms_sq = np.sum(G ** 2, axis=0)
    coeffs = np.zeros((degree + 1, emb_dim))
    for k in range(actual_degree + 1):
        if norms_sq[k] > 1e-10:
            coeffs[k] = (G[:, k] @ embeddings) / norms_sq[k]
    return coeffs


def gram_curve(coeffs, n, tau_grid):
    """Evaluate the fitted polynomial (from gram_fit on an n-point
    sequence) on tau_grid in [0, 1], mapping tau to the continuous index
    i = tau * (n - 1) in each basis polynomial."""
    degree = coeffs.shape[0] - 1
    out = np.zeros((len(tau_grid), coeffs.shape[1]))
    for k in range(degree + 1):
        vals = np.array([_gram_value_continuous(k, t * (n - 1), n)
                         for t in tau_grid])
        out += vals[:, None] * coeffs[k][None, :]
    return out


def _gram_value_continuous(k, x, n):
    """gram_polynomial_value with a continuous point argument (the
    binomial in i generalizes via the falling factorial)."""
    result = 0.0
    for s in range(k + 1):
        if n - 1 < s:
            continue
        num = 1.0
        for j in range(s):
            num *= (x - j)
        num /= math.factorial(s) if s else 1.0
        result += ((-1) ** s * comb(k, s, exact=True)
                   * comb(k + s, s, exact=True) * num
                   / comb(n - 1, s, exact=True))
    return result
