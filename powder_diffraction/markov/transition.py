"""
Transition matrix construction for Reichweite-4 Markov chain.

The transition probabilities depend on the HK pair:
P(S_n = K | S_{n-2}, S_{n-1}) =
    α if HH
    β if HK
    γ if KH
    δ if KK
"""

import numpy as np
from typing import Tuple
from .states import (STATES, STATE_TO_INDEX, get_hk_pair,
                    get_valid_next_layers, index_to_state)


def build_transition_matrix(alpha: float, beta: float,
                           gamma: float, delta: float) -> np.ndarray:
    """
    Build the 24×24 transition matrix T(α, β, γ, δ).

    Each row has exactly 2 nonzero entries (the two valid next layers).

    Parameters
    ----------
    alpha : float
        P(K | HH)
    beta : float
        P(K | HK)
    gamma : float
        P(K | KH)
    delta : float
        P(K | KK)

    Returns
    -------
    np.ndarray
        24×24 transition matrix
    """
    # Map HK pairs to their P(K) probability
    prob_k = {
        ('H', 'H'): alpha,
        ('H', 'K'): beta,
        ('K', 'H'): gamma,
        ('K', 'K'): delta
    }

    n_states = len(STATES)
    T = np.zeros((n_states, n_states))

    for i, state in enumerate(STATES):
        x0, x1, x2, x3 = state
        hk_pair = get_hk_pair(state)

        # Get probability of K for next transition
        p_k = prob_k[hk_pair]
        p_h = 1.0 - p_k

        # Valid next layers
        valid_next = get_valid_next_layers(x3)
        assert len(valid_next) == 2

        for next_layer in valid_next:
            # New state after transition
            new_state = (x1, x2, x3, next_layer)
            j = STATE_TO_INDEX[new_state]

            # Determine if this transition is H or K
            # S_n = H if X_{n-1} == X_{n+1} (i.e., x2 == next_layer)
            # S_n = K otherwise
            if x2 == next_layer:
                T[i, j] = p_h  # This transition is H
            else:
                T[i, j] = p_k  # This transition is K

    # Verify row sums
    row_sums = T.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums not 1: {row_sums}"

    return T


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution π such that π^T T = π^T.

    This is the left eigenvector of T with eigenvalue 1.

    Parameters
    ----------
    T : np.ndarray
        Transition matrix

    Returns
    -------
    np.ndarray
        Stationary distribution (normalized to sum to 1)
    """
    # Left eigenvector: solve v^T T = v^T, or T^T v = v
    eigenvalues, eigenvectors = np.linalg.eig(T.T)

    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))

    # Get corresponding eigenvector
    pi = np.real(eigenvectors[:, idx])

    # Normalize to sum to 1
    pi = pi / pi.sum()

    # Ensure non-negative (numerical issues may cause small negatives)
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()

    return pi


def sample_stacking_sequence(T: np.ndarray, n_layers: int,
                            initial_state: Tuple[str, str, str, str] = None,
                            rng: np.random.Generator = None) -> str:
    """
    Sample a stacking sequence from the Markov chain.

    Parameters
    ----------
    T : np.ndarray
        Transition matrix
    n_layers : int
        Number of layers to generate (minimum 4)
    initial_state : tuple, optional
        Starting 4-gram. If None, samples from stationary distribution.
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    str
        Stacking sequence as string (e.g., 'ABCBACBA...')
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_layers < 4:
        raise ValueError("Need at least 4 layers")

    # Initialize
    if initial_state is None:
        pi = stationary_distribution(T)
        state_idx = rng.choice(len(STATES), p=pi)
        state = index_to_state(state_idx)
    else:
        state = initial_state
        state_idx = STATE_TO_INDEX[state]

    # Start sequence with initial 4-gram
    sequence = list(state)

    # Generate remaining layers
    for _ in range(n_layers - 4):
        probs = T[state_idx, :]
        state_idx = rng.choice(len(STATES), p=probs)
        state = index_to_state(state_idx)
        sequence.append(state[3])  # Append the new layer

    return ''.join(sequence)


def plot_stacking_sequences(param_sets: dict, n_layers: int = 50,
                           save_path: str = None):
    """
    Create validation plot showing stacking sequences for different parameters.

    Parameters
    ----------
    param_sets : dict
        Dictionary mapping names to (α, β, γ, δ) tuples
    n_layers : int
        Number of layers to generate
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n_sets = len(param_sets)
    fig, axes = plt.subplots(n_sets, 1, figsize=(14, 2 * n_sets))

    if n_sets == 1:
        axes = [axes]

    colors = {'A': '#E41A1C', 'B': '#377EB8', 'C': '#4DAF4A'}
    rng = np.random.default_rng(42)

    for ax, (name, params) in zip(axes, param_sets.items()):
        alpha, beta, gamma, delta = params
        T = build_transition_matrix(alpha, beta, gamma, delta)
        seq = sample_stacking_sequence(T, n_layers, rng=rng)

        # Draw colored bars for each layer
        for i, layer in enumerate(seq):
            rect = Rectangle((i, 0), 1, 1, facecolor=colors[layer],
                            edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

        ax.set_xlim(0, n_layers)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Layer index')
        ax.set_title(f'{name}: (α={alpha}, β={beta}, γ={gamma}, δ={delta})')

        # Compute H/K ratio
        h_count = sum(1 for i in range(len(seq)-2)
                     if seq[i] == seq[i+2])
        k_count = len(seq) - 2 - h_count
        ax.text(n_layers + 1, 0.5,
               f'H:{h_count} K:{k_count}',
               va='center', fontsize=10)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='s', color='w',
                             markerfacecolor=colors[l], markersize=15,
                             label=f'Layer {l}')
                      for l in ['A', 'B', 'C']]
    fig.legend(handles=legend_elements, loc='upper right', ncol=3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, axes


if __name__ == '__main__':
    # Test transition matrix
    T = build_transition_matrix(0.5, 0.5, 0.5, 0.5)
    print("Transition matrix shape:", T.shape)
    print("Row sums:", T.sum(axis=1))

    pi = stationary_distribution(T)
    print("Stationary distribution (first 5):", pi[:5])
    print("Sum of π:", pi.sum())

    # Test sampling
    seq = sample_stacking_sequence(T, 20)
    print(f"Sample sequence: {seq}")

    # Create validation plot
    param_sets = {
        'Nearly Cubic (high K)': (0.9, 0.9, 0.9, 0.9),
        'Nearly Hexagonal (high H)': (0.1, 0.1, 0.1, 0.1),
        'Random': (0.5, 0.5, 0.5, 0.5),
        'Realistic Ice I_sd': (0.7, 0.4, 0.6, 0.5)
    }
    plot_stacking_sequences(param_sets, n_layers=50, save_path='stacking_sequences.png')
