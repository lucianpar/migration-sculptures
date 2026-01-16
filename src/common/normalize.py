"""
Mesh normalization utilities.

CRITICAL: Normalization happens AFTER geometry generation, NEVER before.

Unit Model:
- Mode A (NORMALIZED): max(bbox dimension) = 2.0 units
- Mode B (METERS): real-world scale (debug only)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@dataclass
class NormalizationResult:
    """Result of mesh normalization."""
    scale_factor: float
    translation: np.ndarray
    bbox_before: Dict[str, Tuple[float, float]]
    bbox_after: Dict[str, Tuple[float, float]]
    max_dim_before: float
    max_dim_after: float


def get_mesh_bounds(vertices: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Get bounding box of vertices."""
    return {
        'x': (float(vertices[:, 0].min()), float(vertices[:, 0].max())),
        'y': (float(vertices[:, 1].min()), float(vertices[:, 1].max())),
        'z': (float(vertices[:, 2].min()), float(vertices[:, 2].max()))
    }


def get_max_dimension(bounds: Dict[str, Tuple[float, float]]) -> float:
    """Get maximum dimension from bounds."""
    extents = [b[1] - b[0] for b in bounds.values()]
    return max(extents) if extents else 0.0


def normalize_mesh(
    mesh: "trimesh.Trimesh",
    target_max_dim: float = 2.0,
    center: bool = True
) -> Tuple["trimesh.Trimesh", NormalizationResult]:
    """
    Normalize mesh to target maximum dimension.
    
    CRITICAL: This function should be called AFTER all geometry operations.
    
    Rules:
    1. Compute mesh bounding box
    2. Scale uniformly so max(x_range, y_range, z_range) == target_max_dim
    3. Optionally center mesh at origin
    
    Args:
        mesh: Input trimesh mesh
        target_max_dim: Target maximum dimension (default 2.0)
        center: Whether to center mesh at origin
        
    Returns:
        Tuple of (normalized_mesh, NormalizationResult)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh normalization")
    
    # Get original bounds
    bbox_before = get_mesh_bounds(mesh.vertices)
    max_dim_before = get_max_dimension(bbox_before)
    
    if max_dim_before < 1e-10:
        logger.warning("Mesh has zero extent, cannot normalize")
        return mesh, NormalizationResult(
            scale_factor=1.0,
            translation=np.zeros(3),
            bbox_before=bbox_before,
            bbox_after=bbox_before,
            max_dim_before=max_dim_before,
            max_dim_after=max_dim_before
        )
    
    # Compute scale factor
    scale_factor = target_max_dim / max_dim_before
    
    # Create copy and scale
    normalized = mesh.copy()
    normalized.vertices = normalized.vertices * scale_factor
    
    # Center at origin if requested
    translation = np.zeros(3)
    if center:
        centroid = normalized.vertices.mean(axis=0)
        normalized.vertices = normalized.vertices - centroid
        translation = -centroid
    
    # Get new bounds
    bbox_after = get_mesh_bounds(normalized.vertices)
    max_dim_after = get_max_dimension(bbox_after)
    
    logger.info(f"Normalized mesh: {max_dim_before:.2f}m → {max_dim_after:.2f} units (scale={scale_factor:.6f})")
    
    return normalized, NormalizationResult(
        scale_factor=scale_factor,
        translation=translation,
        bbox_before=bbox_before,
        bbox_after=bbox_after,
        max_dim_before=max_dim_before,
        max_dim_after=max_dim_after
    )


def normalize_vertices(
    vertices: np.ndarray,
    target_max_dim: float = 2.0,
    center: bool = True
) -> Tuple[np.ndarray, NormalizationResult]:
    """
    Normalize raw vertices to target maximum dimension.
    
    Same as normalize_mesh but works on raw vertex arrays.
    
    Args:
        vertices: Nx3 array of vertices
        target_max_dim: Target maximum dimension
        center: Whether to center at origin
        
    Returns:
        Tuple of (normalized_vertices, NormalizationResult)
    """
    bbox_before = get_mesh_bounds(vertices)
    max_dim_before = get_max_dimension(bbox_before)
    
    if max_dim_before < 1e-10:
        logger.warning("Vertices have zero extent, cannot normalize")
        return vertices.copy(), NormalizationResult(
            scale_factor=1.0,
            translation=np.zeros(3),
            bbox_before=bbox_before,
            bbox_after=bbox_before,
            max_dim_before=max_dim_before,
            max_dim_after=max_dim_before
        )
    
    scale_factor = target_max_dim / max_dim_before
    normalized = vertices * scale_factor
    
    translation = np.zeros(3)
    if center:
        centroid = normalized.mean(axis=0)
        normalized = normalized - centroid
        translation = -centroid
    
    bbox_after = get_mesh_bounds(normalized)
    max_dim_after = get_max_dimension(bbox_after)
    
    return normalized, NormalizationResult(
        scale_factor=scale_factor,
        translation=translation,
        bbox_before=bbox_before,
        bbox_after=bbox_after,
        max_dim_before=max_dim_before,
        max_dim_after=max_dim_after
    )
