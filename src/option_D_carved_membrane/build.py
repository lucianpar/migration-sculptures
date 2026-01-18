"""
Option D: Carved Membrane (Hybrid of B + C, non-spiky)

Generate a single continuous "envelope" volume around the migration corridor,
then carve an internal void that represents the corridor itself.

Key idea: Do both "membrane" and "subtraction" in voxels/SDF, then marching cubes once.
No normal-based displacement.

Algorithm:
1. Build base density field (rasterize tracks, gaussian smooth)
2. Create two masks: OUTER envelope and INNER cavity using dual thresholds
3. "De-spike" in voxel domain using SDF distance transform smoothing
4. Carved membrane boolean: M_final = M_outer AND (NOT M_inner)
5. Marching cubes once, then mesh polish
6. Optional mild flow field deformation
7. Normalize to max dimension = 2.0

Acceptance criteria:
- ONE continuous sculptural body
- Smooth organic surface (no jagged spikes)
- Internal void represents migration corridor
- Readable as membrane with carved interior
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config, UnitMode, MeshMetadata
from common.io import TrackData, save_mesh
from common.normalize import normalize_mesh
from common.mesh_ops import smooth_mesh, smooth_mesh_laplacian, ensure_manifold, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.ndimage import (
        gaussian_filter, 
        binary_dilation, 
        binary_erosion,
        binary_closing,
        binary_opening,
        distance_transform_edt,
        label as ndimage_label
    )
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
        logger.info(f"Removed {removed} floating components, kept largest ({component_sizes[largest_idx-1]} voxels)")
    
    return result


def rasterize_tracks_to_density(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
    radius_vox: float = 2.0
) -> np.ndarray:
    """
    Rasterize track points into a density field.
    
    Each point contributes a gaussian blob to the density field.
    
    Args:
        points: Nx3 track points in meters
        grid_shape: Voxel grid shape (nz, ny, nx)
        origin: Grid origin in meters
        spacing: Voxel spacing in meters
        radius_vox: Influence radius in voxels
        
    Returns:
        Density field (float, 0 to ~1)
    """
    density = np.zeros(grid_shape, dtype=np.float32)
    
    nz, ny, nx = grid_shape
    
    for pt in points:
        # Convert point to voxel coordinates
        vox_coords = (pt - origin) / spacing
        ix, iy, iz = int(vox_coords[0]), int(vox_coords[1]), int(vox_coords[2])
        
        # Bounds check with radius
        r = int(np.ceil(radius_vox * 2))
        x_min, x_max = max(0, ix - r), min(nx, ix + r + 1)
        y_min, y_max = max(0, iy - r), min(ny, iy + r + 1)
        z_min, z_max = max(0, iz - r), min(nz, iz + r + 1)
        
        # Add gaussian contribution
        for z in range(z_min, z_max):
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    dist_sq = (x - vox_coords[0])**2 + (y - vox_coords[1])**2 + (z - vox_coords[2])**2
                    density[z, y, x] += np.exp(-dist_sq / (2 * radius_vox**2))
    
    return density


def rasterize_tracks_fast(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float
) -> np.ndarray:
    """
    Fast track rasterization using KD-tree for distance queries.
    
    More efficient for larger point clouds.
    """
    density = np.zeros(grid_shape, dtype=np.float32)
    
    nz, ny, nx = grid_shape
    
    # Build coordinate arrays
    z_coords = origin[2] + (np.arange(nz) + 0.5) * spacing
    y_coords = origin[1] + (np.arange(ny) + 0.5) * spacing
    x_coords = origin[0] + (np.arange(nx) + 0.5) * spacing
    
    # Grid of all voxel centers
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    voxel_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Query distances to track points
    tree = cKDTree(points)
    
    # Get distance to nearest k points
    k = min(5, len(points))
    distances, _ = tree.query(voxel_centers, k=k)
    
    if k == 1:
        distances = distances.reshape(-1, 1)
    
    # Convert to density using gaussian falloff
    # Use spacing as sigma for scale-invariance
    sigma = spacing * 2.5
    contributions = np.exp(-(distances**2) / (2 * sigma**2))
    density_flat = contributions.sum(axis=1)
    
    density = density_flat.reshape(grid_shape)
    
    return density


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance field from binary mask.
    
    Negative inside, positive outside.
    """
    # Distance transform gives distance to nearest zero
    dist_outside = distance_transform_edt(~mask)
    dist_inside = distance_transform_edt(mask)
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    return sdf.astype(np.float32)


def smooth_mask_via_sdf(
    mask: np.ndarray,
    blur_sigma: float = 0.8
) -> np.ndarray:
    """
    Smooth a binary mask using SDF distance transform smoothing.
    
    This is the key "de-spiking" technique:
    1. Compute SDF of mask
    2. Blur the SDF slightly
    3. Re-threshold at 0 to recover smooth mask
    
    Args:
        mask: Binary mask
        blur_sigma: Gaussian blur sigma for SDF (in voxels)
        
    Returns:
        Smoothed binary mask
    """
    # Compute SDF
    sdf = compute_sdf(mask)
    
    # Blur the SDF
    sdf_smooth = gaussian_filter(sdf, sigma=blur_sigma)
    
    # Re-threshold at 0 (inside = negative)
    smooth_mask = sdf_smooth < 0
    
    return smooth_mask


def build_carved_membrane(
    track_data: TrackData,
    config: Optional[Config] = None,
    voxel_res: int = 160,
    blur_sigma_density: float = 1.4,
    t_outer_factor: float = 0.22,
    t_inner_factor: float = 0.55,
    sdf_blur_sigma: float = 0.8,
    closing_radius_vox: int = 1,
    use_morphology_backup: bool = False,
    target_tris: int = 100000,
    smoothing_iterations: int = 15,
    padding_factor: float = 0.15
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option D: Carved Membrane sculpture.
    
    Creates a continuous envelope volume with an internal void representing
    the migration corridor. Uses SDF smoothing for organic surfaces.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        voxel_res: Voxel grid resolution (128-160 recommended)
        blur_sigma_density: Gaussian blur sigma for density field
        t_outer_factor: Outer envelope threshold (fraction of max density)
        t_inner_factor: Inner cavity threshold (fraction of max density)
        sdf_blur_sigma: SDF blur sigma for de-spiking
        closing_radius_vox: Morphological closing radius (backup smoothing)
        use_morphology_backup: Use morphological closing in addition to SDF
        target_tris: Target triangle count for decimation
        smoothing_iterations: Mesh smoothing iterations
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
    logger.info("Option D: Carved Membrane (Hybrid B+C)")
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
    
    # Compute grid spacing to achieve target resolution
    max_extent = np.max(extent + 2 * padding)
    spacing = max_extent / voxel_res
    
    # Compute grid shape
    grid_extent = max_corner - min_corner
    grid_shape = tuple((grid_extent / spacing).astype(int) + 1)
    # Ensure ZYX order for proper indexing
    grid_shape = (grid_shape[2], grid_shape[1], grid_shape[0])
    
    logger.info(f"\n=== Step D1: Build base density field ===")
    logger.info(f"Grid shape: {grid_shape}, spacing: {spacing:.1f}m, resolution: {voxel_res}")
    
    # Rasterize tracks to density field
    density = rasterize_tracks_fast(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing
    )
    
    # Gaussian smooth density field
    logger.info(f"Gaussian smoothing density (sigma={blur_sigma_density})")
    density = gaussian_filter(density, sigma=blur_sigma_density)
    
    # Keep as float, don't binarize yet
    max_density = density.max()
    logger.info(f"Density range: 0 to {max_density:.4f}")
    
    # === Step D2: Create OUTER envelope and INNER cavity masks ===
    logger.info(f"\n=== Step D2: Create dual threshold masks ===")
    
    t_outer = t_outer_factor * max_density
    t_inner = t_inner_factor * max_density
    
    logger.info(f"Outer envelope threshold: {t_outer:.4f} ({t_outer_factor*100:.0f}% of max)")
    logger.info(f"Inner cavity threshold: {t_inner:.4f} ({t_inner_factor*100:.0f}% of max)")
    
    M_outer = density > t_outer
    M_inner = density > t_inner
    
    outer_voxels = M_outer.sum()
    inner_voxels = M_inner.sum()
    logger.info(f"Outer mask: {outer_voxels} voxels ({100*outer_voxels/M_outer.size:.1f}%)")
    logger.info(f"Inner mask: {inner_voxels} voxels ({100*inner_voxels/M_inner.size:.1f}%)")
    
    if outer_voxels == 0:
        raise ValueError(f"Outer mask is empty - try lowering t_outer_factor (currently {t_outer_factor})")
    
    # === Step D3: De-spike using SDF smoothing ===
    logger.info(f"\n=== Step D3: De-spike via SDF smoothing (sigma={sdf_blur_sigma}) ===")
    
    # SDF smoothing for outer envelope
    logger.info("Computing SDF for outer envelope...")
    M_outer_smooth = smooth_mask_via_sdf(M_outer, blur_sigma=sdf_blur_sigma)
    
    # SDF smoothing for inner cavity (only if it exists)
    if inner_voxels > 0:
        logger.info("Computing SDF for inner cavity...")
        M_inner_smooth = smooth_mask_via_sdf(M_inner, blur_sigma=sdf_blur_sigma)
    else:
        logger.info("Inner cavity is empty - creating solid body")
        M_inner_smooth = np.zeros_like(M_inner)
    
    # Optional morphological closing as backup
    if use_morphology_backup and closing_radius_vox > 0:
        logger.info(f"Applying morphological closing (radius={closing_radius_vox})")
        from scipy.ndimage import generate_binary_structure, iterate_structure
        struct = generate_binary_structure(3, 1)
        struct = iterate_structure(struct, closing_radius_vox)
        M_outer_smooth = binary_closing(M_outer_smooth, structure=struct)
        if inner_voxels > 0:
            M_inner_smooth = binary_closing(M_inner_smooth, structure=struct)
    
    outer_smooth_voxels = M_outer_smooth.sum()
    inner_smooth_voxels = M_inner_smooth.sum()
    logger.info(f"Smoothed outer: {outer_smooth_voxels} voxels")
    logger.info(f"Smoothed inner: {inner_smooth_voxels} voxels")
    
    # === Step D4: Carved membrane boolean ===
    logger.info(f"\n=== Step D4: Boolean subtraction (carved membrane) ===")
    
    # M_final = M_outer AND (NOT M_inner)
    M_final = M_outer_smooth & (~M_inner_smooth)
    
    final_voxels = M_final.sum()
    logger.info(f"Final carved membrane: {final_voxels} voxels ({100*final_voxels/M_final.size:.1f}%)")
    
    if final_voxels == 0:
        raise ValueError("Carved membrane is empty - inner cavity consumed entire outer envelope")
    
    # Keep largest connected component
    M_final = keep_largest_component(M_final.astype(np.float32))
    
    # === Step D5: Marching cubes ===
    logger.info(f"\n=== Step D5: Marching cubes + mesh polish ===")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            M_final.astype(np.float32),
            level=0.5,
            spacing=(spacing, spacing, spacing)
        )
        # Offset vertices to world coordinates (note: marching cubes uses ZYX)
        verts = verts[:, ::-1]  # Reverse to XYZ
        verts = verts + min_corner
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError(f"Could not extract surface from carved membrane")
    
    if len(verts) == 0:
        raise ValueError("Marching cubes produced empty mesh")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Mesh smoothing (Taubin or Laplacian)
    logger.info(f"Mesh smoothing ({smoothing_iterations} iterations)")
    try:
        # Try Taubin smoothing if available
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)
    except Exception:
        try:
            mesh = smooth_mesh_laplacian(mesh, iterations=smoothing_iterations, lamb=0.5)
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}")
            mesh = smooth_mesh(mesh, iterations=smoothing_iterations // 2)
    
    # Decimation to target triangle count
    if len(mesh.faces) > target_tris:
        logger.info(f"Decimating {len(mesh.faces)} -> {target_tris} triangles")
        try:
            mesh = mesh.simplify_quadric_decimation(target_tris)
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
    
    logger.info(f"After polish: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # === Step D6: Optional flow field deformation (skipped for now) ===
    # Can be added later if desired
    
    # === Step D7: Mesh cleanup and normalization ===
    logger.info(f"\n=== Step D6: Cleanup and normalization ===")
    mesh = ensure_manifold(mesh)
    
    # Record bounds before normalization
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
    # Normalize
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
    
    # Compute envelope radius in meters
    envelope_radius_m = spacing * (outer_voxels / (4/3 * np.pi)) ** (1/3) if outer_voxels > 0 else 0
    
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
            "algorithm": "carved_membrane",
            "voxel_res": voxel_res,
            "voxel_spacing_m": spacing,
            "blur_sigma_density": blur_sigma_density,
            "t_outer_factor": t_outer_factor,
            "t_outer_absolute": float(t_outer),
            "t_inner_factor": t_inner_factor,
            "t_inner_absolute": float(t_inner),
            "sdf_blur_sigma": sdf_blur_sigma,
            "envelope_radius_vox": float((outer_voxels / (4/3 * np.pi)) ** (1/3)) if outer_voxels > 0 else 0,
            "envelope_radius_m": float(envelope_radius_m),
            "smoothing_iterations": smoothing_iterations,
            "target_tris": target_tris,
            "watertight": mesh.is_watertight,
            "outer_voxels": int(outer_voxels),
            "inner_voxels": int(inner_voxels),
            "final_voxels": int(final_voxels)
        }
    )
    
    logger.info(f"\n=== Result ===")
    logger.info(f"Vertices: {metadata.n_vertices}, Triangles: {metadata.n_triangles}")
    logger.info(f"Watertight: {mesh.is_watertight}")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Option D: Carved Membrane (Hybrid B+C)")
    print("Run via: python -m src.run_all --modules D")
