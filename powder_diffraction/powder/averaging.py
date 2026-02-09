"""
Powder averaging via spherical quadrature.

I(Q) = (1/4π) ∫_{|Q'|=Q} E[|ψ(Q')|²] dΩ
     ≈ Σ_i w_i · E[|ψ(Q·û_i)|²]
"""

import numpy as np
from typing import Callable, Tuple
from .quadrature import spherical_quadrature, recommended_quadrature_points


def powder_average(Q_mags: np.ndarray, intensity_func: Callable[[np.ndarray], float],
                  n_theta: int = None, n_phi: int = None,
                  N_a: int = 10, N_b: int = 10, N_c: int = 20) -> np.ndarray:
    """
    Compute powder-averaged intensity I(Q).

    Parameters
    ----------
    Q_mags : np.ndarray
        Array of Q magnitudes
    intensity_func : callable
        Function that computes intensity for a 3D Q vector
    n_theta : int, optional
        Number of polar angle points
    n_phi : int, optional
        Number of azimuthal angle points
    N_a, N_b, N_c : int
        Crystallite dimensions (used for default quadrature if not specified)

    Returns
    -------
    np.ndarray
        Powder-averaged intensity at each Q magnitude
    """
    # Get quadrature points
    if n_theta is None or n_phi is None:
        n_theta, n_phi = recommended_quadrature_points(N_a, N_b, N_c)

    directions, weights = spherical_quadrature(n_theta, n_phi)

    # Compute powder average for each Q magnitude
    Q_mags = np.asarray(Q_mags)
    I_powder = np.zeros(Q_mags.shape)

    for i, Q_mag in enumerate(Q_mags):
        # Integrate over sphere
        total = 0.0
        for j, (direction, weight) in enumerate(zip(directions, weights)):
            Q_vec = Q_mag * direction
            total += weight * intensity_func(Q_vec)
        I_powder[i] = total

    return I_powder


def powder_average_vectorized(Q_mags: np.ndarray,
                             intensity_func_batch: Callable[[np.ndarray], np.ndarray],
                             n_theta: int = None, n_phi: int = None,
                             N_a: int = 10, N_b: int = 10, N_c: int = 20) -> np.ndarray:
    """
    Compute powder-averaged intensity I(Q) with batch evaluation.

    Parameters
    ----------
    Q_mags : np.ndarray
        Array of Q magnitudes
    intensity_func_batch : callable
        Function that computes intensity for multiple Q vectors at once
        Input: (N, 3) array, Output: (N,) array
    n_theta, n_phi : int, optional
        Number of quadrature points
    N_a, N_b, N_c : int
        Crystallite dimensions

    Returns
    -------
    np.ndarray
        Powder-averaged intensity at each Q magnitude
    """
    if n_theta is None or n_phi is None:
        n_theta, n_phi = recommended_quadrature_points(N_a, N_b, N_c)

    directions, weights = spherical_quadrature(n_theta, n_phi)

    Q_mags = np.asarray(Q_mags)
    I_powder = np.zeros(Q_mags.shape)

    for i, Q_mag in enumerate(Q_mags):
        Q_vecs = Q_mag * directions  # (n_points, 3)
        intensities = intensity_func_batch(Q_vecs)  # (n_points,)
        I_powder[i] = np.dot(weights, intensities)

    return I_powder


def plot_powder_pattern(Q_grid: np.ndarray, param_sets: dict,
                       N_a: int = 10, N_b: int = 10, N_c: int = 20,
                       save_path: str = None):
    """
    Create validation plot of powder patterns for different disorder parameters.

    Parameters
    ----------
    Q_grid : np.ndarray
        Q values to compute
    param_sets : dict
        Dictionary mapping names to (α, β, γ, δ) tuples
    N_a, N_b, N_c : int
        Crystallite dimensions
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    from ..markov.transition import build_transition_matrix
    from ..correlation.cdelta import CorrelationComputer
    from ..geometry.crystallite import compute_intensity
    from ..geometry.lattice import HexagonalLattice

    fig, ax = plt.subplots(figsize=(12, 6))

    lattice = HexagonalLattice()

    for name, params in param_sets.items():
        alpha, beta, gamma, delta = params
        T = build_transition_matrix(alpha, beta, gamma, delta)
        corr_computer = CorrelationComputer(T, lattice)

        def intensity_func(Q):
            return compute_intensity(Q, N_a, N_b, N_c, corr_computer, lattice)

        I_powder = powder_average(Q_grid, intensity_func,
                                 N_a=N_a, N_b=N_b, N_c=N_c)

        ax.plot(Q_grid, I_powder, label=name, linewidth=1.5)

    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('I(Q) (arb. units)', fontsize=12)
    ax.set_title(f'Powder Diffraction Pattern\nCrystallite: {N_a}×{N_b}×{N_c}',
                fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark expected peak positions
    # Hexagonal ice: (100), (002), (101), (102), (110), etc.
    # These are approximate for ice I
    d_values = [3.9, 3.66, 3.45, 2.67, 2.25]  # Å (approximate)
    Q_peaks = [2 * np.pi / d for d in d_values]
    for qp in Q_peaks:
        if qp < Q_grid.max():
            ax.axvline(qp, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    # Quick test
    from ..markov.transition import build_transition_matrix
    from ..correlation.cdelta import CorrelationComputer
    from ..geometry.crystallite import compute_intensity
    from ..geometry.lattice import HexagonalLattice

    lattice = HexagonalLattice()
    T = build_transition_matrix(0.5, 0.5, 0.5, 0.5)
    corr_computer = CorrelationComputer(T, lattice)

    def intensity_func(Q):
        return compute_intensity(Q, 10, 10, 20, corr_computer, lattice)

    # Test on a few Q values
    Q_test = np.array([1.0, 1.5, 2.0])
    I_test = powder_average(Q_test, intensity_func, n_theta=20, n_phi=20)
    print(f"Test powder average: {I_test}")
