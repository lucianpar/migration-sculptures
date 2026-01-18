"""
Option D: Organic Volume (Hybrid)

Combines the best of B and C:
- Solid continuous body like membrane (not spiky/jagged)
- Organic subtractive sculpting like void-as-data
- Smooth, flowing forms that suggest erosion/flow

Algorithm:
1. Create metaball-based blob from track density
2. Add organic noise displacement for natural variation  
3. Gentle carving influenced by migration paths (not harsh corridors)
4. Heavy smoothing for soft, organic result
5. Normalize to max dim = 2.0
6. Export

Acceptance criteria:
- ONE continuous sculptural body (no floating islands)
- Smooth, organic surface (no jagged spikes)
- Form suggests flow/erosion from migration
- Readable as both solid object AND shaped by movement
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
from common.voxel import VoxelGrid, create_voxel_grid, rasterize_tracks
from common.mesh_ops import smooth_mesh, smooth_mesh_laplacian, ensure_manifold, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
    from scipy.ndimage import label as ndimage_label
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def keep_largest_component(voxels: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
    if not SCIPY_AVAILABLE:
        return voxels
    
    labeled, n_components = ndimage_label(voxels > 0)
    
    if n_components <= 1:
        return voxels
    
    component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
    largest_idx = np.argmax(component_sizes) + 1
    
    result = (labeled == largest_idx).astype(voxels.dtype)
    removed = n_components - 1
    if removed > 0:
        logger.info(f"Removed {removed} floating components")
    
    return result


def create_density_blob(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
    influence_radius_m: float = 5000.0,
    density_power: float = 0.5
) -> np.ndarray:
    """
    Create organic blob based on point density.
    
    Uses a soft metaball-like approach - each point contributes 
    to a smooth density field.
    
    Args:
        points: Nx3 track points in meters
        grid_shape: Voxel grid shape
        origin: Grid origin
        spacing: Voxel spacing
        influence_radius_m: How far each point influences
        density_power: Power falloff (lower = softer blobs)
        
    Returns:
        Density field (0-1)
    """
    density = np.zeros(grid_shape, dtype=np.float32)
    
    # Build KD-tree for efficient queries
    tree = cKDTree(points)
    
    # For each voxel, compute distance to nearest points
    nz, ny, nx = grid_shape
    
    # Create coordinate arrays
    z_coords = origin[2] + np.arange(nz) * spacing
    y_coords = origin[1] + np.arange(ny) * spacing
    x_coords = origin[0] + np.arange(nx) * spacing
    
    # Grid of all voxel centers
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    voxel_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Query k nearest neighbors for soft influence
    k = min(10, len(points))
    distances, _ = tree.query(voxel_centers, k=k)
    
    if k == 1:
        distances = distances.reshape(-1, 1)
    
    # Soft falloff - closer points contribute more
    # Using inverse distance with soft cutoff
    normalized_dist = distances / influence_radius_m
    contributions = np.exp(-normalized_dist ** 2)  # Gaussian falloff
    
    # Sum contributions from k nearest points
    density_flat = contributions.mean(axis=1)
    
    # Apply power for organic feel
    density_flat = density_flat ** density_power
    
    density = density_flat.reshape(grid_shape)
    
    # Normalize to 0-1
    density = (density - density.min()) / (density.max() - density.min() + 1e-10)
    
    return density


def apply_organic_noise(
    density: np.ndarray,
    noise_scale: float = 0.15,
    smoothness: float = 3.0
) -> np.ndarray:
    """
    Add organic noise variation to density field.
    
    Creates natural-looking surface variation without harsh edges.
    """
    noise = np.random.randn(*density.shape).astype(np.float32)
    
    # Smooth noise for organic feel
    if SCIPY_AVAILABLE:
        noise = gaussian_filter(noise, sigma=smoothness)
    
    # Normalize noise to -1 to 1
    noise = noise / (np.abs(noise).max() + 1e-10)
    
    # Modulate density with noise
    modulated = density * (1.0 + noise_scale * noise)
    
    return np.clip(modulated, 0, 1)


def apply_flow_erosion(
    volume: np.ndarray,
    points: np.ndarray,
    grid: VoxelGrid,
    erosion_strength: float = 0.3,
    erosion_radius_m: float = 3000.0
) -> np.ndarray:
    """
    Apply gentle erosion along migration paths.
    
    Unlike harsh corridor carving, this creates gentle 
    depressions that suggest flow/movement.
    """
    nz, ny, nx = volume.shape
    
    # Create coordinate arrays
    z_coords = grid.origin[2] + np.arange(nz) * grid.spacing
    y_coords = grid.origin[1] + np.arange(ny) * grid.spacing
    x_coords = grid.origin[0] + np.arange(nx) * grid.spacing
    
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    voxel_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Distance to track paths
    tree = cKDTree(points)
    distances, _ = tree.query(voxel_centers, k=1)
    distances = distances.reshape(volume.shape)
    
    # Soft erosion mask - closer to paths = more erosion
    normalized_dist = distances / erosion_radius_m
    erosion_mask = np.exp(-normalized_dist ** 2)
    
    # Smooth the erosion for organic effect
    if SCIPY_AVAILABLE:
        erosion_mask = gaussian_filter(erosion_mask, sigma=2.0)
    
    # Apply erosion - reduce density near paths
    eroded = volume * (1.0 - erosion_strength * erosion_mask)
    
    return eroded


def build_organic_volume(
    track_data: TrackData,
    config: Optional[Config] = None,
    resolution: int = 80,
    influence_radius_m: float = 6000.0,
    density_threshold: float = 0.25,
    noise_scale: float = 0.12,
    erosion_strength: float = 0.25,
    erosion_radius_m: float = 4000.0,
    smoothing_iterations: int = 40,
    padding_factor: float = 0.3
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option D: Organic Volume sculpture.
    
    Creates a smooth, organic solid body influenced by migration patterns.
    Combines the connectivity of membrane with the organic carving approach.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        resolution: Voxel grid resolution (higher = more detail)
        influence_radius_m: Point influence radius for blob creation
        density_threshold: Threshold for surface extraction
        noise_scale: Amount of organic noise (0-1)
        erosion_strength: How much to erode along paths (0-1)
        erosion_radius_m: Erosion influence radius
        smoothing_iterations: Laplacian smoothing iterations
        padding_factor: Padding around data bounds
        
    Returns:
        Tuple of (mesh, metadata)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Option D")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for marching cubes")
    
    config = config or Config()
    
    logger.info("=" * 60)
    logger.info("Option D: Organic Volume (Hybrid)")
    logger.info("=" * 60)
    
    # Get track data
    points = track_data.all_points_m
    bounds = track_data.bounds_m
    
    logger.info(f"Track data: {len(points)} points from {track_data.n_tracks} tracks")
    
    # Compute padded bounds
    min_corner = np.array([bounds['x'][0], bounds['y'][0], bounds['z'][0]])
    max_corner = np.array([bounds['x'][1], bounds['y'][1], bounds['z'][1]])
    extent = max_corner - min_corner
    padding = extent * padding_factor
    min_corner -= padding
    max_corner += padding
    
    # Compute grid spacing
    max_extent = np.max(extent + 2 * padding)
    spacing = max_extent / resolution
    
    # Compute grid shape
    grid_extent = max_corner - min_corner
    grid_shape = tuple((grid_extent / spacing).astype(int) + 1)
    
    logger.info(f"\nStep 1: Creating voxel grid")
    logger.info(f"Grid shape: {grid_shape}, spacing: {spacing:.1f}m")
    
    # Step 2: Create density blob from track points
    logger.info("\nStep 2: Creating density blob from migration data")
    density = create_density_blob(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing,
        influence_radius_m=influence_radius_m,
        density_power=0.5
    )
    logger.info(f"Density range: {density.min():.3f} to {density.max():.3f}")
    
    # Step 3: Add organic noise
    logger.info(f"\nStep 3: Adding organic noise (scale={noise_scale})")
    density = apply_organic_noise(density, noise_scale=noise_scale, smoothness=3.0)
    
    # Step 4: Smooth density field
    logger.info("\nStep 4: Smoothing density field")
    density = gaussian_filter(density, sigma=2.0)
    
    # Step 5: Apply gentle flow erosion
    logger.info(f"\nStep 5: Applying flow erosion (strength={erosion_strength})")
    grid = VoxelGrid(
        data=density,
        origin=min_corner,
        spacing=spacing
    )
    density = apply_flow_erosion(
        volume=density,
        points=points,
        grid=grid,
        erosion_strength=erosion_strength,
        erosion_radius_m=erosion_radius_m
    )
    
    # Final smooth
    density = gaussian_filter(density, sigma=1.5)
    
    # Step 6: Extract mesh with marching cubes
    logger.info(f"\nStep 6: Extracting mesh (threshold={density_threshold})")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            density,
            level=density_threshold,
            spacing=(spacing, spacing, spacing)
        )
        # Offset vertices to world coordinates
        verts = verts + min_corner
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError(f"Could not extract surface at threshold {density_threshold}")
    
    if len(verts) == 0:
        raise ValueError("Marching cubes produced empty mesh")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Step 7: Keep largest component
    logger.info("\nStep 7: Keeping largest connected component")
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.vertices))
        logger.info(f"Kept largest of {len(components)} components: {len(mesh.vertices)} vertices")
    
    # Step 8: Heavy smoothing for organic feel
    logger.info(f"\nStep 8: Laplacian smoothing ({smoothing_iterations} iterations)")
    try:
        mesh = smooth_mesh_laplacian(mesh, iterations=smoothing_iterations, lamb=0.5)
    except Exception as e:
        logger.warning(f"Laplacian failed, using simple smooth: {e}")
        mesh = smooth_mesh(mesh, iterations=smoothing_iterations // 2)
    
    # Step 9: Mesh cleanup
    logger.info("\nStep 9: Mesh cleanup")
    mesh = ensure_manifold(mesh)
    
    # Record bounds before normalization
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
    # Step 10: Normalize
    logger.info("\nStep 10: Normalization")
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
        option="D",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "resolution": resolution,
            "influence_radius_m": influence_radius_m,
            "density_threshold": density_threshold,
            "noise_scale": noise_scale,
            "erosion_strength": erosion_strength,
            "erosion_radius_m": erosion_radius_m,
            "smoothing_iterations": smoothing_iterations,
            "watertight": mesh.is_watertight
        }
    )
    
    logger.info(f"\nResult: {metadata.n_vertices} vertices, {metadata.n_triangles} triangles")
    logger.info(f"Watertight: {mesh.is_watertight}")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Option D: Organic Volume (Hybrid)")
    print("Run via: python -m src.run_all --modules D")
