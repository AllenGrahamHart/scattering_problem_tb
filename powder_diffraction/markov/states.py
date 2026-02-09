"""
4-gram state enumeration for Reichweite-4 stacking disorder.

States are 4-tuples (X_{n-3}, X_{n-2}, X_{n-1}, X_n) where X ∈ {A, B, C}
with close-packing constraint: consecutive letters must differ.
"""

import numpy as np
from typing import Tuple, List, Dict

# Layer labels
LAYERS = ['A', 'B', 'C']


def is_valid_sequence(seq: Tuple[str, ...]) -> bool:
    """Check if a sequence satisfies close-packing constraint."""
    return all(seq[i] != seq[i+1] for i in range(len(seq) - 1))


def enumerate_states() -> List[Tuple[str, str, str, str]]:
    """
    Enumerate all valid 4-gram states.

    Returns
    -------
    list
        List of 24 valid 4-grams as tuples
    """
    states = []
    for x0 in LAYERS:
        for x1 in LAYERS:
            for x2 in LAYERS:
                for x3 in LAYERS:
                    seq = (x0, x1, x2, x3)
                    if is_valid_sequence(seq):
                        states.append(seq)
    return states


# Pre-compute states and index mappings
STATES = enumerate_states()
STATE_TO_INDEX = {state: i for i, state in enumerate(STATES)}
INDEX_TO_STATE = {i: state for i, state in enumerate(STATES)}

assert len(STATES) == 24, f"Expected 24 states, got {len(STATES)}"


def state_to_index(state: Tuple[str, str, str, str]) -> int:
    """Convert a 4-gram state to its index."""
    return STATE_TO_INDEX[state]


def index_to_state(index: int) -> Tuple[str, str, str, str]:
    """Convert an index to its 4-gram state."""
    return INDEX_TO_STATE[index]


def get_hk_pair(state: Tuple[str, str, str, str]) -> Tuple[str, str]:
    """
    Compute the HK pair (S_{n-2}, S_{n-1}) from a 4-gram state.

    S_{n-2} = H if X_{n-3} == X_{n-1}, else K
    S_{n-1} = H if X_{n-2} == X_n, else K

    Parameters
    ----------
    state : tuple
        4-gram (X_{n-3}, X_{n-2}, X_{n-1}, X_n)

    Returns
    -------
    tuple
        (S_{n-2}, S_{n-1}) where each is 'H' or 'K'
    """
    x0, x1, x2, x3 = state
    s_nm2 = 'H' if x0 == x2 else 'K'
    s_nm1 = 'H' if x1 == x3 else 'K'
    return (s_nm2, s_nm1)


def get_valid_next_layers(current_layer: str) -> List[str]:
    """Get layers that can follow the current layer (close-packing)."""
    return [l for l in LAYERS if l != current_layer]


def print_state_table():
    """Print table of all states with their indices and HK pairs."""
    print(f"{'Index':>5} | {'State':>12} | {'HK Pair':>8}")
    print("-" * 30)
    for i, state in enumerate(STATES):
        hk = get_hk_pair(state)
        state_str = ''.join(state)
        hk_str = ''.join(hk)
        print(f"{i:>5} | {state_str:>12} | {hk_str:>8}")


if __name__ == '__main__':
    print("Valid 4-gram states for ice stacking:")
    print_state_table()
    print(f"\nTotal states: {len(STATES)}")
