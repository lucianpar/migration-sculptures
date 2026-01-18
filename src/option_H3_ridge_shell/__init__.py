"""
Module H3: SDF Ridge Shell — Curve → Volume → Mesh

Converts wrapped curves on a sphere into winding topographic ridge structures
using volumetric field construction and iso-surface extraction.

Key anti-blob techniques:
1. Constrain field to thin spherical shell around radius R
2. Use narrow tubular kernel for density near curves
3. Extract iso-surface at tuned threshold

The result is a ridge network hugging the sphere surface.
"""

from pathlib import Path

__version__ = "1.0.0"

MODULE_DIR = Path(__file__).parent

from .build import (
    H3Config,
    build_h3_ridge_shell,
    build_h3_from_tracks,
    resample_curve,
    resample_curves,
    curves_to_points,
    curves_from_track_data,
    save_mesh_ply,
)

__all__ = [
    "H3Config",
    "build_h3_ridge_shell",
    "build_h3_from_tracks",
    "resample_curve",
    "resample_curves",
    "curves_to_points",
    "curves_from_track_data",
    "save_mesh_ply",
]