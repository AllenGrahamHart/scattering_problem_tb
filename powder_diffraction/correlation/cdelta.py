"""
Correlation function C_Δ(Q) via eigendecomposition.

C_Δ(Q) = Tr(F* · Diag(π) · T^Δ)

where F_ij = Φ_i · Φ_j* and Φ_i = g_{X(i)}(Q)
"""

import numpy as np
from typing import Tuple
from ..markov.states import STATES, index_to_state
from ..markov.transition import stationary_distribution
from ..geometry.lattice import ABC_OFFSETS, HexagonalLattice

# Oxygen coherent scattering length (fm)
F_O = 5.803


class CorrelationComputer:
    """
    Computes correlation function C_Δ(Q) efficiently via eigendecomposition.

    The transition matrix is eigendecomposed once, then T^Δ can be
    computed efficiently for any Δ.
    """

    def __init__(self, T: np.ndarray, lattice: HexagonalLattice = None):
        """
        Initialize with transition matrix.

        Parameters
        ----------
        T : np.ndarray
            24×24 transition matrix
        lattice : HexagonalLattice, optional
            Lattice geometry. Uses default if None.
        """
        self.T = T
        self.lattice = lattice if lattice is not None else HexagonalLattice()
        self.n_states = len(STATES)

        # Compute stationary distribution
        self.pi = stationary_distribution(T)

        # Eigendecompose T = Q Λ Q^(-1)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(T)
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

    def layer_amplitude(self, Q: np.ndarray, layer: str) -> complex:
        """
        Compute layer scattering amplitude g_X(Q).

        g_X(Q) = f_O · exp(-i Q · s_X)

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector (3D)
        layer : str
            Layer type: 'A', 'B', or 'C'

        Returns
        -------
        complex
            Scattering amplitude
        """
        # Get 3D offset
        s = self.lattice.layer_offset_3d(layer)
        phase = -1j * np.dot(Q, s)
        return F_O * np.exp(phase)

    def state_amplitudes(self, Q: np.ndarray) -> np.ndarray:
        """
        Compute amplitude Φ_i for each state.

        Φ_i = g_{X_n(i)}(Q) where X_n is the last layer in state i.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector (3D)

        Returns
        -------
        np.ndarray
            Complex amplitudes for each state
        """
        Phi = np.zeros(self.n_states, dtype=complex)
        for i, state in enumerate(STATES):
            last_layer = state[3]  # X_n
            Phi[i] = self.layer_amplitude(Q, last_layer)
        return Phi

    def T_power(self, delta: int) -> np.ndarray:
        """
        Compute T^Δ efficiently using eigendecomposition.

        T^Δ = Q · Λ^Δ · Q^(-1)

        Parameters
        ----------
        delta : int
            Power to raise T to (non-negative)

        Returns
        -------
        np.ndarray
            T^Δ matrix
        """
        if delta == 0:
            return np.eye(self.n_states)
        elif delta == 1:
            return self.T

        # Λ^Δ
        lambda_delta = np.diag(self.eigenvalues ** delta)

        # T^Δ = Q · Λ^Δ · Q^(-1)
        return self.eigenvectors @ lambda_delta @ self.eigenvectors_inv

    def C_delta(self, Q: np.ndarray, delta: int) -> complex:
        """
        Compute correlation function C_Δ(Q).

        C_Δ(Q) = Tr(F* · Diag(π) · T^Δ)
              = Σ_{i,j} Φ_i* · π_i · (T^Δ)_{ij} · Φ_j

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector (3D)
        delta : int
            Layer separation (non-negative)

        Returns
        -------
        complex
            Correlation value
        """
        Phi = self.state_amplitudes(Q)
        T_d = self.T_power(delta)

        # C_Δ = Σ_{i,j} Φ_i* · π_i · T^Δ_{ij} · Φ_j
        # = (Φ* ⊙ π)^T · T^Δ · Φ
        weighted = np.conj(Phi) * self.pi
        result = weighted @ T_d @ Phi

        return result

    def C_delta_array(self, Q: np.ndarray, delta_max: int) -> np.ndarray:
        """
        Compute C_Δ(Q) for Δ = 0, 1, ..., delta_max.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector (3D)
        delta_max : int
            Maximum layer separation

        Returns
        -------
        np.ndarray
            Array of correlation values for each Δ
        """
        Phi = self.state_amplitudes(Q)
        correlations = np.zeros(delta_max + 1, dtype=complex)

        # Efficient iteration: T^Δ = T^(Δ-1) · T
        T_d = np.eye(self.n_states)
        weighted = np.conj(Phi) * self.pi

        for delta in range(delta_max + 1):
            correlations[delta] = weighted @ T_d @ Phi
            T_d = T_d @ self.T

        return correlations

    def C_infinity(self, Q: np.ndarray) -> complex:
        """
        Compute limiting correlation C_∞(Q) = |E[g]|².

        As Δ → ∞, correlation approaches the uncorrelated limit.

        Parameters
        ----------
        Q : np.ndarray
            Scattering vector (3D)

        Returns
        -------
        complex
            Limiting correlation value
        """
        Phi = self.state_amplitudes(Q)
        mean_g = np.sum(self.pi * Phi)
        return np.abs(mean_g) ** 2


def plot_correlation_decay(param_sets: dict, Q: np.ndarray = None,
                          delta_max: int = 30, save_path: str = None):
    """
    Create validation plot showing |C_Δ| decay.

    Parameters
    ----------
    param_sets : dict
        Dictionary mapping names to (α, β, γ, δ) tuples
    Q : np.ndarray, optional
        Scattering vector. Uses (1, 0.5, 0.3) if None.
    delta_max : int
        Maximum Δ to plot
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    from ..markov.transition import build_transition_matrix

    if Q is None:
        Q = np.array([1.0, 0.5, 0.3])

    fig, ax = plt.subplots(figsize=(10, 6))

    deltas = np.arange(delta_max + 1)

    for name, params in param_sets.items():
        alpha, beta, gamma, delta = params
        T = build_transition_matrix(alpha, beta, gamma, delta)
        computer = CorrelationComputer(T)

        correlations = computer.C_delta_array(Q, delta_max)
        c_inf = computer.C_infinity(Q)

        ax.plot(deltas, np.abs(correlations), 'o-', label=name, markersize=4)
        ax.axhline(np.abs(c_inf), linestyle='--', alpha=0.5)

    ax.set_xlabel('Δ (layer separation)', fontsize=12)
    ax.set_ylabel('|C_Δ(Q)|', fontsize=12)
    ax.set_title(f'Correlation Decay\nQ = {Q}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Verify C_0 annotation
    ax.annotate('C_0 = E[|g|²]', xy=(0, np.abs(correlations[0])),
               xytext=(3, np.abs(correlations[0]) + 5),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    from ..markov.transition import build_transition_matrix

    # Test correlation computation
    T = build_transition_matrix(0.5, 0.5, 0.5, 0.5)
    computer = CorrelationComputer(T)

    Q = np.array([1.0, 0.5, 0.3])

    print("C_0 =", computer.C_delta(Q, 0))
    print("C_1 =", computer.C_delta(Q, 1))
    print("C_10 =", computer.C_delta(Q, 10))
    print("C_∞ =", computer.C_infinity(Q))

    # Verify C_0 = Σ π_i |Φ_i|²
    Phi = computer.state_amplitudes(Q)
    expected_c0 = np.sum(computer.pi * np.abs(Phi)**2)
    print(f"\nVerification: C_0 = {computer.C_delta(Q, 0):.6f}")
    print(f"Expected (Σ π|Φ|²) = {expected_c0:.6f}")

    # Create plot
    param_sets = {
        'Nearly Cubic': (0.9, 0.9, 0.9, 0.9),
        'Nearly Hexagonal': (0.1, 0.1, 0.1, 0.1),
        'Random': (0.5, 0.5, 0.5, 0.5),
    }
    plot_correlation_decay(param_sets, save_path='correlation_decay.png')
