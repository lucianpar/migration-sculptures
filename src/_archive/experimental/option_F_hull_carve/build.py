"""
Option F: Hull-Constrained Organic Carving

Generate a clean E-style hull, then subtract a D-style organic corridor volume
using SDF/voxel difference so the exterior stays sculptural while the interior
stays rich.

Algorithm:
F1. Build clean hull SDF (PCA capsule - E-like silhouette)
F2. Create corridor density field (two-pass: core + envelope)
F3. Convert density to carve solid mask (organic but controlled)
F4. Final field composition: max(S_hull, -S_carve)
F5. Mesh polish and export

Core idea: A minus B = max(S_A, -S_B) in SDF space
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import logging
import json
import argparse

logger = logging.getLogger(__name__)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config, UnitMode, MeshMetadata
from common.io import TrackData, save_mesh
from common.normalize import normalize_mesh
from common.mesh_ops import smooth_mesh, ensure_manifold, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.ndimage import (
        gaussian_filter, 
        distance_transform_edt, 
        label as ndimage_label,
        binary_closing,
        binary_dilation
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


@dataclass
class OptionFParams:
    """Parameters for Option F generation."""
    # Voxel grid
    vox_res: int = 192
    padding_factor: float = 0.10  # padding = 0.10 * hull_bbox_diag
    
    # Hull (F1)
    hull_margin: float = 0.12  # inflate capsule 12%
    hull_radius_percentile: float = 85.0  # percentile for lateral distance
    
    # Density rasterization (F2)
    paint_radius_core_factor: float = 0.02  # * hull_bbox_diag
    paint_radius_core_min_m: float = 2000.0
    paint_radius_core_max_m: float = 8000.0
    
    paint_radius_env_factor: float = 0.05  # * hull_bbox_diag
    paint_radius_env_min_m: float = 6000.0
    paint_radius_env_max_m: float = 16000.0
    
    blur_core_sigma_vox: float = 1.0
    blur_env_sigma_vox: float = 2.0
    
    # Thresholds (F3)
    t_core: float = 0.55
    t_env: float = 0.25
    band_vox: int = 5  # distance band for envelope detail
    
    # Binary closing
    closing_radius_vox: int = 1
    
    # SDF smoothing (F3/F4)
    sdf_blur_sigma_vox: float = 0.8
    sigma_field_vox: float = 0.6  # final field blur
    
    # Mesh polish (F5)
    taubin_iters: int = 6
    target_tris: int = 120_000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vox_res": self.vox_res,
            "padding_factor": self.padding_factor,
            "hull_margin": self.hull_margin,
            "hull_radius_percentile": self.hull_radius_percentile,
            "paint_radius_core_factor": self.paint_radius_core_factor,
            "paint_radius_env_factor": self.paint_radius_env_factor,
            "blur_core_sigma_vox": self.blur_core_sigma_vox,
            "blur_env_sigma_vox": self.blur_env_sigma_vox,
            "t_core": self.t_core,
            "t_env": self.t_env,
            "band_vox": self.band_vox,
            "closing_radius_vox": self.closing_radius_vox,
            "sdf_blur_sigma_vox": self.sdf_blur_sigma_vox,
            "sigma_field_vox": self.sigma_field_vox,
            "taubin_iters": self.taubin_iters,
            "target_tris": self.target_tris,
        }


def downsample_points(points: np.ndarray, max_points: int = 10000) -> np.ndarray:
    """Downsample points if needed."""
    if len(points) <= max_points:
        return points
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices]


def compute_pca_capsule(
    points: np.ndarray,
    margin: float = 0.12,
    radius_percentile: float = 85.0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute PCA-aligned capsule parameters.
    
    Returns:
        centroid: Center of capsule
        axis: Principal axis (unit vector)
        half_length: Half length along axis (with margin)
        radius: Lateral radius (with margin)
    """
    # Compute PCA
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Principal axis (longest direction)
    axis = eigenvectors[:, 0]
    
    # Project points onto principal axis
    projections = centered @ axis
    
    # Capsule half-length
    half_length = (projections.max() - projections.min()) / 2
    
    # Lateral distance (perpendicular to axis)
    # Project out the axis component
    lateral_vecs = centered - (projections[:, np.newaxis] * axis)
    lateral_dists = np.linalg.norm(lateral_vecs, axis=1)
    
    # Use percentile for radius (more robust than max)
    radius = np.percentile(lateral_dists, radius_percentile)
    
    # Apply margin
    half_length *= (1 + margin)
    radius *= (1 + margin)
    
    return centroid, axis, half_length, radius


def capsule_sdf(
    grid_coords: np.ndarray,
    centroid: np.ndarray,
    axis: np.ndarray,
    half_length: float,
    radius: float
) -> np.ndarray:
    """
    Compute analytic SDF for a capsule.
    
    S < 0: inside
    S > 0: outside
    S = 0: surface
    
    Args:
        grid_coords: (N, 3) array of query points
        centroid: Center of capsule
        axis: Principal axis (unit vector)
        half_length: Half length along axis
        radius: Lateral radius
        
    Returns:
        SDF values (negative inside, positive outside)
    """
    # Vector from centroid to each point
    local = grid_coords - centroid
    
    # Project onto axis
    t = local @ axis
    
    # Clamp to capsule segment
    t_clamped = np.clip(t, -half_length, half_length)
    
    # Nearest point on capsule axis
    nearest_on_axis = t_clamped[:, np.newaxis] * axis
    
    # Distance to axis
    dist_to_axis = np.linalg.norm(local - nearest_on_axis, axis=1)
    
    # SDF: distance to surface (negative inside)
    sdf = dist_to_axis - radius
    
    return sdf


def rasterize_density(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
    paint_radius_m: float
) -> np.ndarray:
    """
    Rasterize track points into density field using Gaussian splats.
    
    Args:
        points: (N, 3) track points in meters
        grid_shape: (nz, ny, nx) voxel grid shape
        origin: (3,) world origin of grid
        spacing: Voxel spacing in meters
        paint_radius_m: Paint radius in meters
        
    Returns:
        Density field (nz, ny, nx)
    """
    density = np.zeros(grid_shape, dtype=np.float32)
    nz, ny, nx = grid_shape
    
    # Create voxel coordinate grid
    z_coords = origin[2] + (np.arange(nz) + 0.5) * spacing
    y_coords = origin[1] + (np.arange(ny) + 0.5) * spacing
    x_coords = origin[0] + (np.arange(nx) + 0.5) * spacing
    
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    voxel_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Build KD-tree for efficient queries
    tree = cKDTree(points)
    
    # Query k nearest points
    k = min(8, len(points))
    distances, _ = tree.query(voxel_centers, k=k)
    
    if k == 1:
        distances = distances.reshape(-1, 1)
    
    # Gaussian splat contribution from each nearby point
    sigma = paint_radius_m
    contributions = np.exp(-(distances**2) / (2 * sigma**2))
    density_flat = contributions.sum(axis=1)
    
    density = density_flat.reshape(grid_shape)
    
    return density


def mask_to_sdf(mask: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """
    Convert binary mask to SDF using distance transform.
    
    SDF < 0: inside
    SDF > 0: outside
    """
    # Distance from outside to inside
    dist_inside = distance_transform_edt(mask) * spacing
    # Distance from inside to outside
    dist_outside = distance_transform_edt(~mask) * spacing
    
    # SDF: negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    return sdf.astype(np.float32)


def keep_largest_component(voxels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Keep only the largest connected component. Returns (result, n_components)."""
    labeled, n_components = ndimage_label(voxels > 0)
    
    if n_components <= 1:
        return voxels, n_components
    
    component_sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
    largest_idx = np.argmax(component_sizes) + 1
    
    result = (labeled == largest_idx).astype(voxels.dtype)
    
    return result, n_components


def build_hull_carve(
    track_data: TrackData,
    config: Optional[Config] = None,
    params: Optional[OptionFParams] = None
) -> Tuple["trimesh.Trimesh", MeshMetadata, Dict[str, Any]]:
    """
    Build Option F: Hull-Constrained Organic Carving.
    
    E-style hull outside, D-style organic interior.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        params: Option F parameters (uses defaults if None)
        
    Returns:
        Tuple of (mesh, metadata, stats_dict)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Option F")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for marching cubes")
    
    config = config or Config()
    params = params or OptionFParams()
    
    logger.info("=" * 60)
    logger.info("Option F: Hull-Constrained Organic Carving")
    logger.info("=" * 60)
    
    # Get track data
    points = track_data.all_points_m.astype(np.float64)
    
    logger.info(f"Track data: {len(points)} points from {track_data.n_tracks} tracks")
    
    # Downsample for PCA if needed
    points_pca = downsample_points(points, max_points=10000)
    
    # ========== F1: Build clean hull SDF (PCA capsule) ==========
    logger.info(f"\n=== F1: Build PCA capsule hull ===")
    
    centroid, axis, half_length, radius = compute_pca_capsule(
        points_pca,
        margin=params.hull_margin,
        radius_percentile=params.hull_radius_percentile
    )
    
    logger.info(f"Capsule centroid: {centroid}")
    logger.info(f"Capsule axis: {axis}")
    logger.info(f"Capsule half_length: {half_length:.1f}m, radius: {radius:.1f}m")
    
    # Compute hull bounding box
    # The capsule extends half_length along axis in both directions
    # and radius perpendicular
    hull_extent = 2 * (half_length + radius)
    hull_bbox_diag = np.sqrt(3) * hull_extent  # Conservative estimate
    
    logger.info(f"Hull bbox diagonal (est): {hull_bbox_diag:.1f}m")
    
    # Compute grid bounds from capsule
    padding = params.padding_factor * hull_bbox_diag
    
    # Capsule endpoints
    p1 = centroid - half_length * axis
    p2 = centroid + half_length * axis
    
    # Bound the capsule with radius + padding
    min_corner = np.minimum(p1, p2) - radius - padding
    max_corner = np.maximum(p1, p2) + radius + padding
    
    # Also include actual point bounds
    point_min = points.min(axis=0)
    point_max = points.max(axis=0)
    min_corner = np.minimum(min_corner, point_min - padding)
    max_corner = np.maximum(max_corner, point_max + padding)
    
    # Compute grid
    extent = max_corner - min_corner
    max_extent = extent.max()
    spacing = max_extent / params.vox_res
    
    grid_shape = tuple(np.ceil(extent / spacing).astype(int) + 1)
    grid_shape = (grid_shape[2], grid_shape[1], grid_shape[0])  # ZYX order
    
    logger.info(f"Grid shape: {grid_shape}, spacing: {spacing:.1f}m")
    
    # Create grid coordinates
    nz, ny, nx = grid_shape
    z_coords = min_corner[2] + (np.arange(nz) + 0.5) * spacing
    y_coords = min_corner[1] + (np.arange(ny) + 0.5) * spacing
    x_coords = min_corner[0] + (np.arange(nx) + 0.5) * spacing
    
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    grid_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Compute hull SDF
    S_hull = capsule_sdf(grid_coords, centroid, axis, half_length, radius)
    S_hull = S_hull.reshape(grid_shape).astype(np.float32)
    
    hull_volume = (S_hull < 0).sum()
    logger.info(f"Hull volume: {hull_volume} voxels ({100*hull_volume/S_hull.size:.1f}%)")
    
    # ========== F2: Create corridor density field (two-pass) ==========
    logger.info(f"\n=== F2: Create corridor density field ===")
    
    # Compute paint radii
    paint_radius_core = params.paint_radius_core_factor * hull_bbox_diag
    paint_radius_core = np.clip(paint_radius_core, params.paint_radius_core_min_m, params.paint_radius_core_max_m)
    
    paint_radius_env = params.paint_radius_env_factor * hull_bbox_diag
    paint_radius_env = np.clip(paint_radius_env, params.paint_radius_env_min_m, params.paint_radius_env_max_m)
    
    logger.info(f"Paint radius core: {paint_radius_core:.1f}m")
    logger.info(f"Paint radius env: {paint_radius_env:.1f}m")
    
    # Pass A: Core corridor (tight routes, strong cavities)
    logger.info("Rasterizing core density...")
    D_core = rasterize_density(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing,
        paint_radius_m=paint_radius_core
    )
    D_core = gaussian_filter(D_core, sigma=params.blur_core_sigma_vox)
    
    # Pass B: Envelope corridor (broader use, organic scalloping)
    logger.info("Rasterizing envelope density...")
    D_env = rasterize_density(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing,
        paint_radius_m=paint_radius_env
    )
    D_env = gaussian_filter(D_env, sigma=params.blur_env_sigma_vox)
    
    logger.info(f"D_core range: {D_core.min():.4f} to {D_core.max():.4f}")
    logger.info(f"D_env range: {D_env.min():.4f} to {D_env.max():.4f}")
    
    # ========== F3: Convert density to carve solid mask ==========
    logger.info(f"\n=== F3: Convert density to carve mask ===")
    
    # Normalize densities
    D_core_norm = D_core / (D_core.max() + 1e-10)
    D_env_norm = D_env / (D_env.max() + 1e-10)
    
    # Threshold
    M_core = D_core_norm > params.t_core
    M_env = D_env_norm > params.t_env
    
    logger.info(f"M_core volume: {M_core.sum()} voxels")
    logger.info(f"M_env volume: {M_env.sum()} voxels")
    
    # Compute distance from M_core
    if M_core.any():
        dist_to_core = distance_transform_edt(~M_core)
    else:
        dist_to_core = np.ones_like(M_core, dtype=np.float32) * 1000
    
    # near(M_core) = within band_vox distance
    near_core = dist_to_core < params.band_vox
    
    # Combine: M_carve = M_core OR (M_env AND near(M_core))
    M_carve = M_core | (M_env & near_core)
    
    logger.info(f"M_carve volume (before closing): {M_carve.sum()} voxels")
    
    # Binary closing to remove jagged perforations
    if params.closing_radius_vox > 0:
        from scipy.ndimage import generate_binary_structure, iterate_structure
        struct = generate_binary_structure(3, 1)
        struct = iterate_structure(struct, params.closing_radius_vox)
        M_carve = binary_closing(M_carve, structure=struct)
        logger.info(f"M_carve volume (after closing): {M_carve.sum()} voxels")
    
    # Convert carve mask to SDF and smooth
    logger.info("Converting carve mask to SDF...")
    S_carve = mask_to_sdf(M_carve, spacing=1.0)  # In voxel units
    
    # Blur carve SDF slightly for smoother edges
    S_carve = gaussian_filter(S_carve, sigma=params.sdf_blur_sigma_vox)
    
    # ========== F4: Final field composition ==========
    logger.info(f"\n=== F4: Final field composition ===")
    
    # Convert hull SDF to voxel units for consistent operations
    S_hull_vox = S_hull / spacing
    
    # SDF difference: A \ B = max(S_A, -S_B)
    # S_hull: negative inside hull
    # S_carve: negative inside carve volume
    # We want: hull minus carve
    S_final = np.maximum(S_hull_vox, -S_carve)
    
    # Light final blur
    S_final = gaussian_filter(S_final, sigma=params.sigma_field_vox)
    
    solid_volume = (S_final < 0).sum()
    logger.info(f"Final solid volume: {solid_volume} voxels")
    
    # ========== F5: Extract mesh and polish ==========
    logger.info(f"\n=== F5: Extract mesh and polish ===")
    
    try:
        # Marching cubes at level 0
        # Negate so solid (S < 0) becomes positive for marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            -S_final,
            level=0,
            spacing=(spacing, spacing, spacing)
        )
        
        # Convert from ZYX to XYZ and offset to world coordinates
        verts = verts[:, ::-1]
        verts = verts + min_corner
        
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError("Could not extract surface")
    
    if len(verts) == 0:
        raise ValueError("Marching cubes produced empty mesh")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Keep largest connected component
    components = mesh.split(only_watertight=False)
    n_components_before = len(components)
    if n_components_before > 1:
        mesh = max(components, key=lambda m: len(m.vertices))
        logger.info(f"Kept largest of {n_components_before} components: {len(mesh.vertices)} vertices")
    
    # Taubin smoothing
    logger.info(f"Taubin smoothing ({params.taubin_iters} iterations)")
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=params.taubin_iters)
    except Exception as e:
        logger.warning(f"Taubin smoothing failed: {e}")
    
    # Decimate if needed
    if len(mesh.faces) > params.target_tris:
        logger.info(f"Decimating {len(mesh.faces)} -> {params.target_tris} triangles")
        try:
            mesh = mesh.simplify_quadric_decimation(params.target_tris)
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
    
    # Ensure manifold
    mesh = ensure_manifold(mesh)
    
    # Get stats before normalization
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
    
    # Final stats
    stats_after = compute_mesh_stats(mesh)
    
    # Build metadata
    quality_stats = {
        "n_components_before": int(n_components_before),
        "n_components_after": 1,
        "watertight": bool(mesh.is_watertight),
        "hull_volume_voxels": int(hull_volume),
        "carve_volume_voxels": int(M_carve.sum()),
        "final_solid_voxels": int(solid_volume),
    }
    
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="F",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "algorithm": "hull_constrained_organic_carving",
            **params.to_dict(),
            "computed": {
                "hull_bbox_diag_m": float(hull_bbox_diag),
                "capsule_centroid": [float(x) for x in centroid],
                "capsule_axis": [float(x) for x in axis],
                "capsule_half_length_m": float(half_length),
                "capsule_radius_m": float(radius),
                "paint_radius_core_m": float(paint_radius_core),
                "paint_radius_env_m": float(paint_radius_env),
                "spacing_m": float(spacing),
                "grid_shape": [int(x) for x in grid_shape],
                "voxel_size_m": float(spacing),
            },
            "quality": quality_stats
        }
    )
    
    logger.info(f"\n=== Result ===")
    logger.info(f"Vertices: {metadata.n_vertices}, Triangles: {metadata.n_triangles}")
    logger.info(f"Watertight: {mesh.is_watertight}")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata, quality_stats


def select_test_specimens(
    data_dir: Path,
    n_specimens: int = 3,
    species: Optional[str] = None,
    season: Optional[str] = None
) -> List[Path]:
    """
    Select test specimens based on coherence or temporal spread.
    
    Args:
        data_dir: Directory containing track data files
        n_specimens: Number of specimens to select
        species: Optional species filter
        season: Optional season filter
        
    Returns:
        List of paths to selected data files
    """
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        # Try parquet
        csv_files = list(data_dir.glob("*.parquet"))
    
    if not csv_files:
        raise ValueError(f"No data files found in {data_dir}")
    
    # For now, just pick evenly spaced files
    if len(csv_files) <= n_specimens:
        return csv_files
    
    # Pick evenly spaced
    indices = np.linspace(0, len(csv_files) - 1, n_specimens, dtype=int)
    return [csv_files[i] for i in indices]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Option F: Hull-Constrained Organic Carving"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        help="Path to track data CSV/parquet file or directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/option_F_hull_carve"),
        help="Output directory"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Generate N preview specimens (auto-select)"
    )
    parser.add_argument(
        "--species",
        type=str,
        help="Filter by species"
    )
    parser.add_argument(
        "--season",
        type=str,
        help="Filter by season"
    )
    parser.add_argument(
        "--unit-mode",
        type=str,
        choices=["normalized", "meters"],
        default="normalized",
        help="Output unit mode"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export GLB files"
    )
    parser.add_argument(
        "--vox-res",
        type=int,
        default=192,
        help="Voxel resolution"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Determine input files
    if args.preview > 0:
        # Auto-select specimens
        if args.data and args.data.is_dir():
            data_files = select_test_specimens(
                args.data,
                n_specimens=args.preview,
                species=args.species,
                season=args.season
            )
        else:
            # Use default subsets directory
            subsets_dir = Path("data/subsets")
            if subsets_dir.exists():
                data_files = list(subsets_dir.glob("*.csv"))[:args.preview]
            else:
                raise ValueError("No data directory specified and no subsets found")
    elif args.data:
        if args.data.is_file():
            data_files = [args.data]
        elif args.data.is_dir():
            data_files = list(args.data.glob("*.csv"))
            if not data_files:
                data_files = list(args.data.glob("*.parquet"))
        else:
            raise ValueError(f"Data path not found: {args.data}")
    else:
        raise ValueError("Must specify --data or --preview")
    
    logger.info(f"Processing {len(data_files)} specimens")
    
    # Setup config
    config = Config(
        unit_mode=UnitMode.NORMALIZED if args.unit_mode == "normalized" else UnitMode.METERS
    )
    
    params = OptionFParams(vox_res=args.vox_res)
    
    # Create output directories
    meshes_dir = args.output / "meshes"
    meta_dir = args.output / "meta"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results = []
    for data_file in data_files:
        specimen_id = data_file.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {specimen_id}")
        logger.info(f"{'='*60}")
        
        try:
            # Load track data
            from common.io import load_tracks
            track_data = load_tracks(data_file)
            
            # Build mesh
            mesh, metadata, stats = build_hull_carve(
                track_data=track_data,
                config=config,
                params=params
            )
            
            # Export
            if args.export or True:  # Always export for now
                glb_path = meshes_dir / f"{specimen_id}.glb"
                json_path = meta_dir / f"{specimen_id}.json"
                
                save_mesh(mesh, glb_path, metadata)
                
                # Save detailed metadata
                with open(json_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                logger.info(f"Exported: {glb_path}")
            
            results.append({
                "specimen_id": specimen_id,
                "success": True,
                "vertices": metadata.n_vertices,
                "triangles": metadata.n_triangles,
                "watertight": stats["watertight"]
            })
            
        except Exception as e:
            logger.error(f"Failed to process {specimen_id}: {e}")
            results.append({
                "specimen_id": specimen_id,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    for r in successful:
        logger.info(f"  {r['specimen_id']}: {r['vertices']}v, {r['triangles']}t, watertight={r['watertight']}")
    
    if failed:
        logger.info(f"Failed: {len(failed)}")
        for r in failed:
            logger.info(f"  {r['specimen_id']}: {r['error']}")


if __name__ == "__main__":
    main()
