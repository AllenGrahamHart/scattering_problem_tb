"""
In-plane structure factor for finite crystallite.

Uses Dirichlet kernel for efficient computation of geometric series.
"""

import numpy as np
from .lattice import HexagonalLattice


def dirichlet_kernel(x: np.ndarray, N: int) -> np.ndarray:
    """
    Compute Dirichlet kernel D_N(x) = Σ_{m=0}^{N-1} exp(i·m·x).

    This is a geometric series that sums to:
    D_N(x) = (1 - exp(i·N·x)) / (1 - exp(i·x))  for x ≠ 2πk
           = N  for x = 2πk (Bragg condition)

    Uses stable sinc form: D_N(x) = exp(i·(N-1)·x/2) · sin(N·x/2) / sin(x/2)

    Parameters
    ----------
    x : np.ndarray
        Phase argument (Q · a)
    N : int
        Number of terms (unit cells)

    Returns
    -------
    np.ndarray
        Complex Dirichlet kernel values
    """
    x = np.asarray(x)
    result = np.zeros(x.shape, dtype=complex)

    # Handle near-Bragg points (x ≈ 2πk) separately
    half_x = x / 2
    sin_half = np.sin(half_x)
    near_bragg = np.abs(sin_half) < 1e-10

    # Regular points: use sinc form
    if np.any(~near_bragg):
        sin_n_half = np.sin(N * half_x[~near_bragg])
        phase = np.exp(1j * (N - 1) * half_x[~near_bragg])
        result[~near_bragg] = phase * sin_n_half / sin_half[~near_bragg]

    # Near-Bragg points: D_N → N
    if np.any(near_bragg):
        result[near_bragg] = N

    return result


def in_plane_structure_factor(Q: np.ndarray, N_a: int, N_b: int,
                              lattice: HexagonalLattice = None) -> np.ndarray:
    """
    Compute in-plane structure factor S_ab(Q).

    S_ab(Q) = Σ_{m1,m2} exp(-i Q·(m1·a1 + m2·a2))
            = D_{N_a}(Q·a1) · D_{N_b}(Q·a2)

    Parameters
    ----------
    Q : np.ndarray
        Scattering vector(s), shape (..., 3) or (3,)
    N_a : int
        Number of unit cells along a1
    N_b : int
        Number of unit cells along a2
    lattice : HexagonalLattice, optional
        Lattice geometry. Uses default if None.

    Returns
    -------
    np.ndarray
        Complex structure factor values
    """
    if lattice is None:
        lattice = HexagonalLattice()

    Q = np.asarray(Q)
    if Q.ndim == 1:
        Q = Q.reshape(1, 3)
        squeeze = True
    else:
        squeeze = False

    # Compute Q·a1 and Q·a2 using only in-plane components
    Q_dot_a1 = Q[:, 0] * lattice.a1[0] + Q[:, 1] * lattice.a1[1]
    Q_dot_a2 = Q[:, 0] * lattice.a2[0] + Q[:, 1] * lattice.a2[1]

    # Structure factor is product of Dirichlet kernels
    # Note: we want exp(-i Q·r), so negate the phase
    S_ab = dirichlet_kernel(-Q_dot_a1, N_a) * dirichlet_kernel(-Q_dot_a2, N_b)

    if squeeze:
        return S_ab[0]
    return S_ab


def compute_intensity(Q: np.ndarray, N_a: int, N_b: int, N_c: int,
                     correlation_computer, lattice: HexagonalLattice = None) -> np.ndarray:
    """
    Compute single-crystal intensity E[|ψ(Q)|²].

    E[|ψ(Q)|²] = |S_ab(Q)|² · Σ_Δ (N_c - |Δ|) · C_Δ(Q) · exp(-i Q_c Δ d)

    Parameters
    ----------
    Q : np.ndarray
        Scattering vector (3D)
    N_a : int
        Number of unit cells along a1
    N_b : int
        Number of unit cells along a2
    N_c : int
        Number of layers along c
    correlation_computer : CorrelationComputer
        Precomputed correlation computer
    lattice : HexagonalLattice, optional
        Lattice geometry

    Returns
    -------
    float
        Intensity (real, non-negative)
    """
    if lattice is None:
        lattice = HexagonalLattice()

    Q = np.asarray(Q)

    # In-plane structure factor
    S_ab = in_plane_structure_factor(Q, N_a, N_b, lattice)

    # Get Q_c (z-component)
    Q_c = Q[2]

    # Compute correlations
    correlations = correlation_computer.C_delta_array(Q, N_c - 1)

    # Sum over Δ: Σ_Δ (N_c - |Δ|) · C_Δ · exp(-i Q_c Δ d)
    # Note: C_{-Δ} = C_Δ* (Hermitian symmetry)
    layer_sum = 0.0

    # Δ = 0 term
    layer_sum += N_c * correlations[0]

    # Δ > 0 terms (count both +Δ and -Δ)
    for delta in range(1, N_c):
        weight = N_c - delta
        phase = np.exp(-1j * Q_c * delta * lattice.d)
        # C_Δ exp(-i Q_c Δ d) + C_{-Δ} exp(+i Q_c Δ d)
        # = C_Δ exp(-i Q_c Δ d) + C_Δ* exp(+i Q_c Δ d)
        # = 2 Re[C_Δ exp(-i Q_c Δ d)]
        layer_sum += 2 * weight * np.real(correlations[delta] * phase)

    intensity = np.abs(S_ab) ** 2 * layer_sum

    return np.real(intensity)


def compute_intensity_vectorized(Q_vecs: np.ndarray, N_a: int, N_b: int, N_c: int,
                                  correlation_computer, lattice: HexagonalLattice) -> np.ndarray:
    """
    Vectorized intensity computation for multiple Q vectors.

    Computes E[|ψ(Q)|²] for many Q vectors at once using array operations,
    giving ~8x speedup over looping.

    Parameters
    ----------
    Q_vecs : np.ndarray
        Array of scattering vectors, shape (n_directions, 3)
    N_a : int
        Number of unit cells along a1
    N_b : int
        Number of unit cells along a2
    N_c : int
        Number of layers along c
    correlation_computer : CorrelationComputer
        Precomputed correlation computer
    lattice : HexagonalLattice
        Lattice geometry

    Returns
    -------
    np.ndarray
        Intensities for each Q vector, shape (n_directions,)
    """
    from ..markov.states import STATES
    from ..correlation.cdelta import F_O

    n_dirs = len(Q_vecs)
    n_states = len(STATES)

    # In-plane structure factor (vectorized)
    Q_dot_a1 = Q_vecs[:, 0] * lattice.a1[0] + Q_vecs[:, 1] * lattice.a1[1]
    Q_dot_a2 = Q_vecs[:, 0] * lattice.a2[0] + Q_vecs[:, 1] * lattice.a2[1]
    S_ab = dirichlet_kernel(-Q_dot_a1, N_a) * dirichlet_kernel(-Q_dot_a2, N_b)
    S_ab_sq = np.abs(S_ab) ** 2

    # z-component of Q vectors
    Q_c = Q_vecs[:, 2]

    # State amplitudes for all Q vectors at once
    # Phi[i, s] = amplitude for direction i, state s
    Phi = np.zeros((n_dirs, n_states), dtype=complex)
    for s, state in enumerate(STATES):
        last_layer = state[3]  # X_n
        s_offset = lattice.layer_offset_3d(last_layer)
        phase = -1j * (Q_vecs @ s_offset)
        Phi[:, s] = F_O * np.exp(phase)

    # Get stationary distribution and transition matrix
    pi = correlation_computer.pi
    T = correlation_computer.T

    # Compute weighted Phi: weighted[i, s] = Φ*[i,s] · π[s]
    weighted = np.conj(Phi) * pi  # (n_dirs, n_states)

    # Initialize layer sum
    layer_sum = np.zeros(n_dirs)

    # Δ = 0 term: C_0 = Σ_{s,t} weighted[i,s] · I[s,t] · Φ[i,t]
    T_d = np.eye(n_states)
    C_0 = np.einsum('is,st,it->i', weighted, T_d, Phi)
    layer_sum += N_c * np.real(C_0)

    # Δ > 0 terms
    for delta in range(1, N_c):
        weight = N_c - delta
        T_d = T_d @ T
        # C_Δ for all directions at once
        C_d = np.einsum('is,st,it->i', weighted, T_d, Phi)
        # Phase factors for this delta
        phase = np.exp(-1j * Q_c * delta * lattice.d)
        # Add contribution: 2 * weight * Re[C_Δ * exp(-i Q_c Δ d)]
        layer_sum += 2 * weight * np.real(C_d * phase)

    intensity = S_ab_sq * layer_sum
    return np.real(intensity)


def plot_single_crystal_intensity(correlation_computer, N_a: int = 10,
                                  N_b: int = 10, N_c: int = 20,
                                  Q_max: float = 3.0, n_points: int = 100,
                                  save_path: str = None):
    """
    Create 2D cross-section plots of single-crystal intensity.

    Parameters
    ----------
    correlation_computer : CorrelationComputer
        Precomputed correlation computer
    N_a, N_b, N_c : int
        Crystallite dimensions
    Q_max : float
        Maximum Q value
    n_points : int
        Number of points per axis
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    lattice = correlation_computer.lattice

    # Create grids
    Q_vals = np.linspace(-Q_max, Q_max, n_points)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Q_x - Q_y plane at Q_z = 0
    ax1 = axes[0]
    intensity_xy = np.zeros((n_points, n_points))
    for i, qx in enumerate(Q_vals):
        for j, qy in enumerate(Q_vals):
            Q = np.array([qx, qy, 0.0])
            intensity_xy[j, i] = compute_intensity(Q, N_a, N_b, N_c,
                                                   correlation_computer, lattice)

    im1 = ax1.imshow(np.log10(intensity_xy + 1), extent=[-Q_max, Q_max, -Q_max, Q_max],
                     origin='lower', cmap='viridis', aspect='equal')
    ax1.set_xlabel('Q_x (Å⁻¹)', fontsize=12)
    ax1.set_ylabel('Q_y (Å⁻¹)', fontsize=12)
    ax1.set_title('Q_x - Q_y plane (Q_z = 0)', fontsize=14)
    plt.colorbar(im1, ax=ax1, label='log₁₀(I + 1)')

    # Plot 2: Q_x - Q_z plane at Q_y = 0
    ax2 = axes[1]
    intensity_xz = np.zeros((n_points, n_points))
    for i, qx in enumerate(Q_vals):
        for j, qz in enumerate(Q_vals):
            Q = np.array([qx, 0.0, qz])
            intensity_xz[j, i] = compute_intensity(Q, N_a, N_b, N_c,
                                                   correlation_computer, lattice)

    im2 = ax2.imshow(np.log10(intensity_xz + 1), extent=[-Q_max, Q_max, -Q_max, Q_max],
                     origin='lower', cmap='viridis', aspect='equal')
    ax2.set_xlabel('Q_x (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('Q_z (Å⁻¹)', fontsize=12)
    ax2.set_title('Q_x - Q_z plane (Q_y = 0)', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='log₁₀(I + 1)')

    plt.suptitle(f'Single-Crystal Intensity\nCrystallite: {N_a}×{N_b}×{N_c}',
                fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, axes


if __name__ == '__main__':
    # Test Dirichlet kernel
    x = np.linspace(-0.1, 0.1, 100)
    D = dirichlet_kernel(x, 10)
    print(f"Dirichlet kernel at x=0: {D[50]:.1f} (expected 10)")

    # Test structure factor
    lattice = HexagonalLattice()
    Q = np.array([0.0, 0.0, 0.0])
    S = in_plane_structure_factor(Q, 10, 10, lattice)
    print(f"S_ab at Q=0: {S:.1f} (expected 100)")
