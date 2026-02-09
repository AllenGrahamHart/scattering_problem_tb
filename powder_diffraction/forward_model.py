"""
Main forward model for powder neutron diffraction from stacking-disordered ice.

Combines all components: geometry, Markov chain, correlation, powder averaging,
and Lorentzian convolution.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field

from .geometry.lattice import HexagonalLattice, DEFAULT_A, DEFAULT_D
from .geometry.crystallite import compute_intensity, compute_intensity_vectorized, in_plane_structure_factor
from .markov.transition import build_transition_matrix, stationary_distribution
from .correlation.cdelta import CorrelationComputer
from .powder.quadrature import spherical_quadrature, recommended_quadrature_points
from .convolution.lorentzian import lorentzian_convolve


@dataclass
class PowderDiffractionModel:
    """
    Forward model for powder neutron diffraction from stacking-disordered ice.

    Parameters
    ----------
    N_a : int
        Number of unit cells along a1 direction
    N_b : int
        Number of unit cells along a2 direction
    N_c : int
        Number of layers along c direction
    a : float
        In-plane lattice constant (Å)
    d : float
        Interlayer spacing (Å)
    n_theta : int, optional
        Number of polar quadrature points (default: auto)
    n_phi : int, optional
        Number of azimuthal quadrature points (default: auto)
    """
    N_a: int = 10
    N_b: int = 10
    N_c: int = 20
    a: float = DEFAULT_A
    d: float = DEFAULT_D
    n_theta: Optional[int] = None
    n_phi: Optional[int] = None

    # Internal state (not exposed as constructor params)
    _lattice: HexagonalLattice = field(init=False, repr=False)
    _T: np.ndarray = field(init=False, repr=False, default=None)
    _corr_computer: CorrelationComputer = field(init=False, repr=False, default=None)
    _directions: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize lattice and quadrature."""
        self._lattice = HexagonalLattice(self.a, self.d)

        # Set up quadrature
        if self.n_theta is None or self.n_phi is None:
            self.n_theta, self.n_phi = recommended_quadrature_points(
                self.N_a, self.N_b, self.N_c)

        self._directions, self._weights = spherical_quadrature(
            self.n_theta, self.n_phi)

    def set_disorder_params(self, alpha: float, beta: float,
                           gamma: float, delta: float) -> None:
        """
        Set the Markov chain disorder parameters.

        Parameters
        ----------
        alpha : float
            P(K | HH), in [0, 1]
        beta : float
            P(K | HK), in [0, 1]
        gamma : float
            P(K | KH), in [0, 1]
        delta : float
            P(K | KK), in [0, 1]
        """
        # Validate
        for name, val in [('alpha', alpha), ('beta', beta),
                         ('gamma', gamma), ('delta', delta)]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

        self._T = build_transition_matrix(alpha, beta, gamma, delta)
        self._corr_computer = CorrelationComputer(self._T, self._lattice)

    def _single_crystal_intensity(self, Q: np.ndarray) -> float:
        """Compute single-crystal intensity for a given Q vector."""
        if self._corr_computer is None:
            raise RuntimeError("Must call set_disorder_params() first")

        return compute_intensity(Q, self.N_a, self.N_b, self.N_c,
                                self._corr_computer, self._lattice)

    def _powder_average_single_Q(self, Q_mag: float) -> float:
        """Compute powder-averaged intensity for a single Q magnitude."""
        # Vectorized computation for all directions at once
        Q_vecs = Q_mag * self._directions
        intensities = compute_intensity_vectorized(
            Q_vecs, self.N_a, self.N_b, self.N_c,
            self._corr_computer, self._lattice
        )
        return np.sum(self._weights * intensities)

    def compute_raw_pattern(self, Q_grid: np.ndarray) -> np.ndarray:
        """
        Compute powder pattern without instrument broadening.

        Parameters
        ----------
        Q_grid : np.ndarray
            Array of Q magnitudes (Å⁻¹)

        Returns
        -------
        np.ndarray
            Powder-averaged intensity at each Q
        """
        Q_grid = np.asarray(Q_grid)
        I_raw = np.zeros(Q_grid.shape)

        for i, Q_mag in enumerate(Q_grid):
            I_raw[i] = self._powder_average_single_Q(Q_mag)

        return I_raw

    def compute_pattern(self, Q_grid: np.ndarray, Gamma: float) -> np.ndarray:
        """
        Compute full powder pattern with Lorentzian broadening.

        Parameters
        ----------
        Q_grid : np.ndarray
            Array of Q magnitudes (Å⁻¹)
        Gamma : float
            Lorentzian half-width (Å⁻¹)

        Returns
        -------
        np.ndarray
            Convolved powder-averaged intensity at each Q
        """
        I_raw = self.compute_raw_pattern(Q_grid)
        I_obs = lorentzian_convolve(Q_grid, I_raw, Gamma)
        return I_obs

    def compute_pattern_fast(self, Q_grid: np.ndarray, Gamma: float,
                            alpha: float, beta: float,
                            gamma: float, delta: float) -> np.ndarray:
        """
        One-shot computation with all parameters.

        Parameters
        ----------
        Q_grid : np.ndarray
            Array of Q magnitudes (Å⁻¹)
        Gamma : float
            Lorentzian half-width (Å⁻¹)
        alpha, beta, gamma, delta : float
            Markov chain parameters

        Returns
        -------
        np.ndarray
            Convolved powder-averaged intensity
        """
        self.set_disorder_params(alpha, beta, gamma, delta)
        return self.compute_pattern(Q_grid, Gamma)


# Default test parameters
DEFAULT_PARAMS = {
    # Crystallite
    'N_a': 10,
    'N_b': 10,
    'N_c': 20,
    'a': 4.5,
    'd': 3.66,

    # Q range
    'Q_min': 0.5,
    'Q_max': 5.0,
    'n_Q': 200,

    # Broadening
    'Gamma': 0.02,
}

# Test parameter sets
PARAM_SETS = {
    'cubic': (0.9, 0.9, 0.9, 0.9),      # Mostly K → cubic
    'hexagonal': (0.1, 0.1, 0.1, 0.1),  # Mostly H → hexagonal
    'random': (0.5, 0.5, 0.5, 0.5),     # Random
    'isd': (0.7, 0.4, 0.6, 0.5),        # Realistic ice I_sd
}


def generate_synthetic_data(model: PowderDiffractionModel,
                           Q_grid: np.ndarray,
                           alpha: float, beta: float,
                           gamma: float, delta: float,
                           Gamma: float,
                           noise_level: float = 0.0) -> np.ndarray:
    """
    Generate synthetic powder diffraction data.

    Parameters
    ----------
    model : PowderDiffractionModel
        Forward model
    Q_grid : np.ndarray
        Q values
    alpha, beta, gamma, delta : float
        True disorder parameters
    Gamma : float
        True instrument broadening
    noise_level : float
        Standard deviation of Gaussian noise (relative to max intensity)

    Returns
    -------
    np.ndarray
        Synthetic intensity data
    """
    I_true = model.compute_pattern_fast(Q_grid, Gamma, alpha, beta, gamma, delta)

    if noise_level > 0:
        noise = np.random.randn(len(Q_grid)) * noise_level * I_true.max()
        I_true = I_true + noise

    return I_true


if __name__ == '__main__':
    import time

    # Create model
    model = PowderDiffractionModel(N_a=10, N_b=10, N_c=20)
    print(f"Created model: {model.N_a}×{model.N_b}×{model.N_c} crystallite")
    print(f"Quadrature: {model.n_theta}×{model.n_phi} = {model.n_theta * model.n_phi} points")

    # Test computation
    Q_grid = np.linspace(0.5, 5.0, 50)  # Fewer points for quick test

    for name, params in PARAM_SETS.items():
        alpha, beta, gamma, delta = params
        print(f"\nComputing {name} pattern...")

        t0 = time.time()
        I = model.compute_pattern_fast(Q_grid, Gamma=0.02,
                                       alpha=alpha, beta=beta,
                                       gamma=gamma, delta=delta)
        elapsed = time.time() - t0

        print(f"  Time: {elapsed:.2f} s")
        print(f"  I range: [{I.min():.2e}, {I.max():.2e}]")
