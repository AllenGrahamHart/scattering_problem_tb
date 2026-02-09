"""
Spherical quadrature for powder averaging.

Uses Gauss-Legendre (polar) × uniform (azimuthal) quadrature.
"""

import numpy as np
from typing import Tuple


def spherical_quadrature(n_theta: int, n_phi: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spherical quadrature points and weights.

    Uses Gauss-Legendre quadrature for polar angle and uniform
    quadrature for azimuthal angle.

    Parameters
    ----------
    n_theta : int
        Number of polar angle points
    n_phi : int
        Number of azimuthal angle points

    Returns
    -------
    directions : np.ndarray
        Unit vectors, shape (n_theta * n_phi, 3)
    weights : np.ndarray
        Quadrature weights, shape (n_theta * n_phi,), sum to 1
    """
    # Gauss-Legendre for cos(theta) ∈ [-1, 1]
    # Note: we only need hemisphere since powder is symmetric,
    # but using full sphere for clarity
    cos_theta, w_theta = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(cos_theta)

    # Uniform quadrature for phi ∈ [0, 2π)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = np.ones(n_phi) / n_phi

    # Create grid
    n_total = n_theta * n_phi
    directions = np.zeros((n_total, 3))
    weights = np.zeros(n_total)

    idx = 0
    for i, (th, wt) in enumerate(zip(theta, w_theta)):
        sin_th = np.sin(th)
        cos_th = cos_theta[i]
        for j, (ph, wp) in enumerate(zip(phi, w_phi)):
            # Unit vector in spherical coordinates
            directions[idx, 0] = sin_th * np.cos(ph)
            directions[idx, 1] = sin_th * np.sin(ph)
            directions[idx, 2] = cos_th

            # Weight: (1/4π) from spherical integration, absorbed into normalization
            # w_theta already accounts for sin(theta) via Gauss-Legendre on [-1,1]
            weights[idx] = wt * wp

            idx += 1

    # Normalize weights to sum to 1 (for easy averaging)
    weights = weights / weights.sum()

    return directions, weights


def recommended_quadrature_points(N_a: int, N_b: int, N_c: int) -> Tuple[int, int]:
    """
    Recommend quadrature points based on crystallite size.

    Rule: n_theta = n_phi = 2 × max(N_a, N_b, N_c)

    Parameters
    ----------
    N_a, N_b, N_c : int
        Crystallite dimensions

    Returns
    -------
    n_theta, n_phi : int
        Recommended number of quadrature points
    """
    n = 2 * max(N_a, N_b, N_c)
    return n, n


if __name__ == '__main__':
    # Test quadrature
    n_theta, n_phi = 20, 40
    directions, weights = spherical_quadrature(n_theta, n_phi)

    print(f"Number of points: {len(directions)}")
    print(f"Sum of weights: {weights.sum():.6f}")

    # Verify unit vectors
    norms = np.linalg.norm(directions, axis=1)
    print(f"Vector norms: min={norms.min():.6f}, max={norms.max():.6f}")

    # Verify coverage: average of x, y, z should be near 0
    print(f"Mean x: {np.average(directions[:, 0], weights=weights):.6f}")
    print(f"Mean y: {np.average(directions[:, 1], weights=weights):.6f}")
    print(f"Mean z: {np.average(directions[:, 2], weights=weights):.6f}")

    # Recommended points for (10, 10, 20) crystallite
    n_t, n_p = recommended_quadrature_points(10, 10, 20)
    print(f"Recommended for (10,10,20): {n_t} × {n_p} = {n_t * n_p} points")
