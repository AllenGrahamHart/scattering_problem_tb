"""
Hexagonal lattice geometry for ice I structure.

Defines lattice vectors and ABC layer offsets for close-packed ice.
"""

import numpy as np
from dataclasses import dataclass

# Default lattice parameters for ice I (in Angstroms)
DEFAULT_A = 4.5   # In-plane lattice constant
DEFAULT_D = 3.66  # Interlayer spacing

# ABC layer offsets in fractional coordinates (2D, in-plane)
# These form the triangular close-packed arrangement
ABC_OFFSETS = {
    'A': np.array([0.0, 0.0]),
    'B': np.array([2.0/3.0, 1.0/3.0]),
    'C': np.array([1.0/3.0, 2.0/3.0])
}


@dataclass
class HexagonalLattice:
    """
    Hexagonal lattice for ice I structure.

    Parameters
    ----------
    a : float
        In-plane lattice constant (Angstroms)
    d : float
        Interlayer spacing (Angstroms)
    """
    a: float = DEFAULT_A
    d: float = DEFAULT_D

    def __post_init__(self):
        """Compute lattice vectors."""
        # In-plane lattice vectors
        self.a1 = np.array([self.a, 0.0, 0.0])
        self.a2 = np.array([self.a / 2, self.a * np.sqrt(3) / 2, 0.0])
        # Out-of-plane vector
        self.a3 = np.array([0.0, 0.0, self.d])

    def layer_offset_3d(self, layer: str) -> np.ndarray:
        """
        Get 3D offset for a layer type.

        Parameters
        ----------
        layer : str
            Layer type: 'A', 'B', or 'C'

        Returns
        -------
        np.ndarray
            3D offset vector in Cartesian coordinates
        """
        frac = ABC_OFFSETS[layer]
        # Convert fractional to Cartesian (in-plane only)
        return frac[0] * self.a1 + frac[1] * self.a2

    def get_layer_positions(self, n_cells_x: int = 3, n_cells_y: int = 3) -> dict:
        """
        Get atom positions for all layer types over a grid of unit cells.

        Parameters
        ----------
        n_cells_x : int
            Number of unit cells in x direction
        n_cells_y : int
            Number of unit cells in y direction

        Returns
        -------
        dict
            Dictionary with keys 'A', 'B', 'C' containing (N, 2) arrays of positions
        """
        positions = {'A': [], 'B': [], 'C': []}

        for m1 in range(n_cells_x):
            for m2 in range(n_cells_y):
                base = m1 * self.a1[:2] + m2 * self.a2[:2]
                for layer in ['A', 'B', 'C']:
                    offset = self.layer_offset_3d(layer)[:2]
                    positions[layer].append(base + offset)

        return {k: np.array(v) for k, v in positions.items()}


def plot_layer_positions(lattice: HexagonalLattice = None, n_cells: int = 3,
                        save_path: str = None):
    """
    Create validation plot showing ABC layer positions.

    Parameters
    ----------
    lattice : HexagonalLattice, optional
        Lattice to plot. Uses default if None.
    n_cells : int
        Number of unit cells in each direction
    save_path : str, optional
        Path to save figure. If None, displays interactively.
    """
    import matplotlib.pyplot as plt

    if lattice is None:
        lattice = HexagonalLattice()

    positions = lattice.get_layer_positions(n_cells, n_cells)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
    markers = {'A': 'o', 'B': 's', 'C': '^'}

    for layer in ['A', 'B', 'C']:
        pos = positions[layer]
        ax.scatter(pos[:, 0], pos[:, 1], c=colors[layer], marker=markers[layer],
                  s=200, label=f'Layer {layer}', edgecolors='black', linewidth=1)

    # Draw unit cell outline
    origin = np.array([0, 0])
    v1 = lattice.a1[:2]
    v2 = lattice.a2[:2]
    cell_corners = np.array([origin, v1, v1 + v2, v2, origin])
    ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'k--', linewidth=2,
            label='Unit cell')

    ax.set_xlabel('x (Å)', fontsize=12)
    ax.set_ylabel('y (Å)', fontsize=12)
    ax.set_title(f'ABC Layer Positions (a = {lattice.a} Å)\nTriangular Close-Packed Arrangement',
                fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    # Quick test
    lattice = HexagonalLattice()
    print(f"a1 = {lattice.a1}")
    print(f"a2 = {lattice.a2}")
    print(f"a3 = {lattice.a3}")

    for layer in ['A', 'B', 'C']:
        print(f"Layer {layer} offset: {lattice.layer_offset_3d(layer)}")

    plot_layer_positions(save_path='layer_positions.png')
