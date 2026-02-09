"""
Powder Neutron Diffraction from Stacking-Disordered Ice

Forward model and inverse solver for ice I_sd with Reichweite-4 disorder.
"""

from .forward_model import PowderDiffractionModel

__all__ = ['PowderDiffractionModel']
