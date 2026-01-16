"""
Common modules for all sculpture generation options.

Unit Model (NON-NEGOTIABLE):
- Raw GPS → UTM Zone 10N (meters) → geometry ops → normalize → export
- Mode A (default): max(bbox dimension) = 2.0 units (normalized)
- Mode B: real-world meters (debug only)
"""

from .config import Config, UnitMode
from .coords import CoordinateTransformer, project_to_utm
from .io import load_tracks, save_mesh, TrackData
from .normalize import normalize_mesh, NormalizationResult
from .voxel import VoxelGrid, rasterize_tracks
from .mesh_ops import smooth_mesh, ensure_manifold, compute_mesh_stats

__all__ = [
    'Config', 'UnitMode',
    'CoordinateTransformer', 'project_to_utm',
    'load_tracks', 'save_mesh', 'TrackData',
    'normalize_mesh', 'NormalizationResult',
    'VoxelGrid', 'rasterize_tracks',
    'smooth_mesh', 'ensure_manifold', 'compute_mesh_stats',
]
