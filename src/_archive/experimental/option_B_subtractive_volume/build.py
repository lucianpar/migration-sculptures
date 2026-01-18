"""
Option B: Subtractive Volume (Void-as-Data)

Create a single mass and carve migration corridors out of it.
This option is subtractive-first, not blob-first.

Algorithm:
1. Create bounding solid block
2. Rasterize migration tracks as density field
3. Boolean subtract: block AND NOT migration
4. Marching cubes to extract mesh
5. Normalize to max dim = 2.0
6. Export

Acceptance criteria:
- ONE continuous sculptural body
- Migration reads as carved absence (void = data)
- No floating islands
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config, UnitMode, MeshMetadata
from common.io import TrackData, save_mesh
from common.normalize import normalize_mesh
from common.voxel import VoxelGrid, create_voxel_grid, rasterize_tracks, create_solid_block
from common.mesh_ops import smooth_mesh, ensure_manifold, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
    from scipy.ndimage import label as ndimage_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def keep_largest_component(voxels: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Ensures no floating islands in final sculpture.
    
    Args:
        voxels: Binary voxel array
        
    Returns:
        Voxel array with only largest component
    """
    if not SCIPY_AVAILABLE:
        logger.warning("scipy not available, skipping component filtering")
        return voxels
    
    labeled, n_components = ndimage_label(voxels > 0)
    
    if n_components <= 1:
        return voxels
    
    # Find largest component
    component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
    largest_idx = np.argmax(component_sizes) + 1
    
    result = (labeled == largest_idx).astype(voxels.dtype)
    
    removed = n_components - 1
    if removed > 0:
        logger.info(f"Removed {removed} floating components, kept largest ({component_sizes[largest_idx-1]} voxels)")
    
    return result


def build_subtractive_volume(
    track_data: TrackData,
    config: Optional[Config] = None,
    carve_radius_m: float = 2500.0,
    carve_threshold: float = 0.15,
    smooth_sigma: float = 1.0,
    block_padding: float = 0.2
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option B: Subtractive Volume sculpture.
    
    Creates a solid block and carves migration corridors through it.
    The voids represent where whales traveled.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        carve_radius_m: Radius of carved corridors in meters
        carve_threshold: Absolute threshold for carving (0-1)
        smooth_sigma: Gaussian smoothing for carved regions
        block_padding: Padding around bounding block (fraction)
        
    Returns:
        Tuple of (mesh, metadata)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    
    config = config or Config()
    
    logger.info("=" * 60)
    logger.info("Option B: Subtractive Volume (Void-as-Data)")
    logger.info("=" * 60)
    
    # Get track bounds in meters
    bounds = track_data.bounds_m
    min_corner = np.array([bounds['x'][0], bounds['y'][0], bounds['z'][0]])
    max_corner = np.array([bounds['x'][1], bounds['y'][1], bounds['z'][1]])
    
    # Add padding to create solid border around carved region
    extent = max_corner - min_corner
    padding_m = extent * block_padding
    min_corner -= padding_m
    max_corner += padding_m
    
    logger.info(f"Bounding block: {min_corner} to {max_corner} (meters)")
    
    # Step 1: Create solid block
    logger.info("\nStep 1: Creating solid block")
    block = create_solid_block(
        bounds_m=(min_corner, max_corner),
        resolution=config.voxel_resolution
    )
    logger.info(f"Block grid: {block.shape}, spacing: {block.spacing:.1f}m")
    
    # Step 2: Rasterize migration tracks as corridors to carve
    logger.info("\nStep 2: Rasterizing migration tracks")
    points = track_data.all_points_m
    
    # Use smaller radius for corridor carving
    corridor_radius = min(carve_radius_m, block.spacing * 3)
    
    migration_density = rasterize_tracks(
        points_m=points,
        grid=block,
        radius_m=corridor_radius,
        falloff="gaussian"
    )
    
    # Smooth the density field
    if SCIPY_AVAILABLE:
        migration_density.data = gaussian_filter(
            migration_density.data,
            sigma=smooth_sigma
        )
    
    max_density = migration_density.data.max()
    logger.info(f"Migration density: max={max_density:.3f}")
    
    # Step 3: Boolean subtraction - carve corridors
    logger.info("\nStep 3: Boolean subtraction (carving corridors)")
    
    # Use absolute threshold - only carve where density is significant
    carve_mask = migration_density.data > carve_threshold
    
    # Don't dilate too much - preserve the block structure
    if SCIPY_AVAILABLE and carve_mask.sum() > 0:
        carve_mask = binary_dilation(carve_mask, iterations=1)
    
    # Subtract: block AND NOT carve_mask  
    result_voxels = block.data.copy()
    result_voxels[carve_mask] = 0
    
    carved_fraction = carve_mask.sum() / carve_mask.size
    remaining_fraction = (result_voxels > 0).sum() / result_voxels.size
    logger.info(f"Carved {carved_fraction*100:.1f}% of volume, {remaining_fraction*100:.1f}% remaining")
    
    # Check if we carved too much
    if remaining_fraction < 0.1:
        logger.warning("Carving removed too much - adjusting threshold")
        # Fall back to higher threshold
        carve_mask = migration_density.data > 0.5
        result_voxels = block.data.copy()
        result_voxels[carve_mask] = 0
        remaining_fraction = (result_voxels > 0).sum() / result_voxels.size
        logger.info(f"Adjusted: {remaining_fraction*100:.1f}% remaining")
    
    # Step 4: Keep largest component (remove floating pieces)
    logger.info("\nStep 4: Removing floating components")
    result_voxels = keep_largest_component(result_voxels)
    
    # Step 5: Marching cubes
    logger.info("\nStep 5: Extracting mesh (marching cubes)")
    result_grid = VoxelGrid(
        data=result_voxels,
        origin=block.origin,
        spacing=block.spacing
    )
    
    verts_m, faces = result_grid.to_mesh(threshold=0.5)
    
    if len(verts_m) == 0:
        raise ValueError("Marching cubes produced empty mesh - carving too aggressive?")
    
    mesh = trimesh.Trimesh(vertices=verts_m, faces=faces)
    
    # Step 6: Clean up mesh
    logger.info("\nStep 6: Mesh cleanup")
    mesh = ensure_manifold(mesh)
    mesh = smooth_mesh(mesh, iterations=config.smoothing_iterations)
    
    # Record bounds before normalization
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
    # Step 7: Normalize
    logger.info("\nStep 7: Normalization")
    if config.unit_mode == UnitMode.NORMALIZED:
        mesh, norm_result = normalize_mesh(mesh, config.normalized_max_dim)
        scale_factor = norm_result.scale_factor
        max_dim_after = norm_result.max_dim_after
        normalization_applied = True
    else:
        scale_factor = 1.0
        max_dim_after = max_dim_before
        normalization_applied = False
    
    # Build metadata
    stats_after = compute_mesh_stats(mesh)
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="B",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "voxel_resolution": config.voxel_resolution,
            "carve_radius_m": carve_radius_m,
            "carve_threshold": carve_threshold,
            "smooth_sigma": smooth_sigma,
            "carved_fraction": carved_fraction
        }
    )
    
    logger.info(f"\nResult: {metadata.n_vertices} vertices, {metadata.n_triangles} triangles")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Option B: Subtractive Volume (Void-as-Data)")
    print("Run via: python -m src.run_all --modules B")
