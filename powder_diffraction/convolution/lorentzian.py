"""
Lorentzian convolution for instrument broadening.

I_obs(Q) = ∫ I(Q') · R_Γ(Q - Q') dQ'
R_Γ(x) = Γ / (π(Γ² + x²))
"""

import numpy as np
from typing import Union


def lorentzian(x: np.ndarray, Gamma: float) -> np.ndarray:
    """
    Lorentzian line shape.

    R_Γ(x) = Γ / (π(Γ² + x²))

    Normalized so that ∫ R_Γ(x) dx = 1.

    Parameters
    ----------
    x : np.ndarray
        Position values
    Gamma : float
        Half-width at half-maximum

    Returns
    -------
    np.ndarray
        Lorentzian values
    """
    return Gamma / (np.pi * (Gamma**2 + x**2))


def lorentzian_convolve(Q_grid: np.ndarray, I_raw: np.ndarray, Gamma: float,
                       n_sigma: float = 50.0) -> np.ndarray:
    """
    Convolve intensity with Lorentzian broadening.

    Uses direct quadrature with truncation at ±n_sigma×Gamma.

    Parameters
    ----------
    Q_grid : np.ndarray
        Q values (assumed uniformly spaced)
    I_raw : np.ndarray
        Raw intensity values
    Gamma : float
        Lorentzian half-width
    n_sigma : float
        Truncation range in units of Gamma

    Returns
    -------
    np.ndarray
        Convolved intensity
    """
    if Gamma <= 0:
        return I_raw.copy()

    Q_grid = np.asarray(Q_grid)
    I_raw = np.asarray(I_raw)
    n_Q = len(Q_grid)

    # Grid spacing
    dQ = Q_grid[1] - Q_grid[0] if n_Q > 1 else 1.0

    # Truncation distance
    truncation = n_sigma * Gamma

    # Build convolution kernel
    # We need kernel at integer multiples of dQ
    n_kernel = int(np.ceil(truncation / dQ))
    kernel_x = np.arange(-n_kernel, n_kernel + 1) * dQ
    kernel = lorentzian(kernel_x, Gamma) * dQ  # Multiply by dQ for discrete sum

    # Convolve
    I_conv = np.zeros(n_Q)

    for i in range(n_Q):
        total = 0.0
        for j, kx in enumerate(kernel_x):
            # Find index in I_raw
            Q_source = Q_grid[i] - kx
            # Linear interpolation
            idx_float = (Q_source - Q_grid[0]) / dQ
            idx_low = int(np.floor(idx_float))
            idx_high = idx_low + 1

            if 0 <= idx_low < n_Q and 0 <= idx_high < n_Q:
                frac = idx_float - idx_low
                I_interp = (1 - frac) * I_raw[idx_low] + frac * I_raw[idx_high]
                total += kernel[j] * I_interp
            elif 0 <= idx_low < n_Q:
                total += kernel[j] * I_raw[idx_low]

        I_conv[i] = total

    return I_conv


def lorentzian_convolve_fft(Q_grid: np.ndarray, I_raw: np.ndarray,
                           Gamma: float) -> np.ndarray:
    """
    Convolve intensity with Lorentzian using FFT.

    Faster for large arrays but may have edge effects.

    Parameters
    ----------
    Q_grid : np.ndarray
        Q values (assumed uniformly spaced)
    I_raw : np.ndarray
        Raw intensity values
    Gamma : float
        Lorentzian half-width

    Returns
    -------
    np.ndarray
        Convolved intensity
    """
    if Gamma <= 0:
        return I_raw.copy()

    from scipy.signal import fftconvolve

    Q_grid = np.asarray(Q_grid)
    I_raw = np.asarray(I_raw)
    n_Q = len(Q_grid)

    dQ = Q_grid[1] - Q_grid[0] if n_Q > 1 else 1.0

    # Create kernel on same grid
    Q_range = Q_grid[-1] - Q_grid[0]
    kernel_x = np.arange(-n_Q//2, n_Q//2 + 1) * dQ
    kernel = lorentzian(kernel_x, Gamma) * dQ

    # FFT convolution
    I_conv = fftconvolve(I_raw, kernel, mode='same')

    return I_conv


def plot_convolution_effect(Q_grid: np.ndarray, I_raw: np.ndarray,
                           Gamma_values: list, save_path: str = None):
    """
    Create plot showing effect of Lorentzian convolution.

    Parameters
    ----------
    Q_grid : np.ndarray
        Q values
    I_raw : np.ndarray
        Raw intensity values
    Gamma_values : list
        List of Gamma values to compare
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(Q_grid, I_raw, 'k-', linewidth=2, label='Raw', alpha=0.5)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(Gamma_values)))

    for Gamma, color in zip(Gamma_values, colors):
        I_conv = lorentzian_convolve(Q_grid, I_raw, Gamma)
        ax.plot(Q_grid, I_conv, color=color, linewidth=1.5,
               label=f'Γ = {Gamma:.3f} Å⁻¹')

    ax.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax.set_ylabel('I(Q) (arb. units)', fontsize=12)
    ax.set_title('Effect of Lorentzian Convolution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    # Test Lorentzian normalization
    x = np.linspace(-10, 10, 1000)
    Gamma = 1.0
    L = lorentzian(x, Gamma)
    print(f"Lorentzian integral: {np.trapz(L, x):.6f} (expected 1.0)")

    # Test convolution
    Q = np.linspace(0, 5, 200)
    # Create test signal with peaks
    I_test = np.exp(-((Q - 1.5)**2) / 0.01) + np.exp(-((Q - 3.0)**2) / 0.01)
    I_conv = lorentzian_convolve(Q, I_test, Gamma=0.05)

    print(f"Raw peak height: {I_test.max():.3f}")
    print(f"Convolved peak height: {I_conv.max():.3f}")
