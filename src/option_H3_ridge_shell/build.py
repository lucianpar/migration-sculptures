"""
Module H3: SDF Ridge Shell — Curve → Volume → Mesh

Converts wrapped curves on a sphere into winding topographic ridge structures
using volumetric field construction and iso-surface extraction.

Algorithm:
H3.1 - Prepare curves: resample for uniform spacing
H3.2 - Build 3D voxel grid centered on sphere
H3.3 - Compute tubular density field from curve points (narrow kernel)
H3.4 - Apply spherical shell mask to constrain to surface
H3.5 - Extract iso-surface via marching cubes
H3.6 - Smooth and clean mesh
H3.7 - Export

Key anti-blob techniques:
- Narrow tube radius (ridge_radius << sphere_radius)
- Shell mask constrains field to thin band around sphere surface
- Higher iso-threshold for thinner ridges
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - H3 requires scipy")

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available - H3 requires skimage")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@dataclass
class H3Config:
    """Configuration for H3 Ridge Shell generation."""
    
    # Sphere parameters
    sphere_radius: float = 1.0
    
    # Ridge geometry
    ridge_radius: float = 0.03      # Tube radius around curve (narrow = ridges)
    ridge_height: float = 0.05      # How far ridges protrude from sphere
    
    # Shell constraint
    shell_inner: float = 0.92       # Inner shell boundary (fraction of radius)
    shell_outer: float = 1.15       # Outer shell boundary (fraction of radius)
    
    # Curve resampling
    resample_spacing: float = 0.02  # Distance between resampled points
    
    # Volume grid
    voxel_resolution: int = 128     # Grid resolution
    voxel_size: Optional[float] = None  # Auto-computed if None
    
    # Field parameters
    density_falloff: float = 2.0    # Gaussian falloff steepness
    
    # Iso-surface
    iso_threshold: float = 0.15     # Higher = thinner ridges
    
    # Mesh smoothing
    smooth_iterations: int = 10
    smooth_factor: float = 0.5
    
    # Optional noise for topographic texture
    noise_scale: float = 0.0        # 0 = disabled
    noise_strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sphere_radius": self.sphere_radius,
            "ridge_radius": self.ridge_radius,
            "ridge_height": self.ridge_height,
            "shell_inner": self.shell_inner,
            "shell_outer": self.shell_outer,
            "resample_spacing": self.resample_spacing,
            "voxel_resolution": self.voxel_resolution,
            "density_falloff": self.density_falloff,
            "iso_threshold": self.iso_threshold,
            "smooth_iterations": self.smooth_iterations,
            "smooth_factor": self.smooth_factor,
        }


def resample_curve(points: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline curve to uniform spacing.
    
    Args:
        points: (N, 3) array of curve points
        spacing: Target distance between points
        
    Returns:
        Resampled (M, 3) array
    """
    if len(points) < 2:
        return points
    
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]
    
    if total_length < spacing:
        return points
    
    # Generate uniform samples along arc length
    n_samples = max(2, int(total_length / spacing) + 1)
    uniform_arc = np.linspace(0, total_length, n_samples)
    
    # Interpolate each coordinate
    resampled = np.zeros((n_samples, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(uniform_arc, cumulative, points[:, dim])
    
    return resampled


def resample_curves(curves: List[np.ndarray], spacing: float) -> List[np.ndarray]:
    """Resample multiple curves."""
    return [resample_curve(c, spacing) for c in curves]


def curves_to_points(curves: List[np.ndarray]) -> np.ndarray:
    """Concatenate all curve points into single array."""
    if not curves:
        return np.zeros((0, 3))
    return np.vstack(curves)


def build_tubular_density_field(
    curve_points: np.ndarray,
    grid_coords: np.ndarray,
    tube_radius: float,
    falloff: float = 2.0
) -> np.ndarray:
    """
    Build density field with tubular falloff around curve points.
    
    Uses KD-tree for efficient nearest-neighbor queries.
    Density falls off with squared distance from curve.
    
    Args:
        curve_points: (N, 3) points along curves
        grid_coords: (M, 3) voxel center coordinates
        tube_radius: Characteristic radius of tubes
        falloff: Steepness of Gaussian falloff
        
    Returns:
        (M,) density values
    """
    if len(curve_points) == 0:
        return np.zeros(len(grid_coords))
    
    # Build KD-tree for curve points
    tree = cKDTree(curve_points)
    
    # Query distance to nearest curve point for each voxel
    distances, _ = tree.query(grid_coords, k=1)
    
    # Gaussian-like density falloff
    # density = exp(-(d/r)^falloff)
    normalized_dist = distances / tube_radius
    density = np.exp(-(normalized_dist ** falloff))
    
    return density


def apply_shell_mask(
    density: np.ndarray,
    grid_coords: np.ndarray,
    sphere_radius: float,
    shell_inner: float,
    shell_outer: float
) -> np.ndarray:
    """
    Mask density to only exist within a spherical shell.
    
    This constrains ridges to the sphere surface region.
    
    Args:
        density: (M,) density values
        grid_coords: (M, 3) voxel coordinates
        sphere_radius: Radius of sphere
        shell_inner: Inner boundary as fraction of radius
        shell_outer: Outer boundary as fraction of radius
        
    Returns:
        Masked density (M,)
    """
    # Compute radial distance from origin
    r = np.linalg.norm(grid_coords, axis=1)
    
    # Shell boundaries
    r_inner = sphere_radius * shell_inner
    r_outer = sphere_radius * shell_outer
    
    # Smooth mask (avoid hard edges)
    # Use smoothstep-like transition
    shell_width = (r_outer - r_inner) * 0.1
    
    # Inner falloff
    inner_mask = np.clip((r - r_inner) / shell_width, 0, 1)
    inner_mask = inner_mask * inner_mask * (3 - 2 * inner_mask)  # smoothstep
    
    # Outer falloff
    outer_mask = np.clip((r_outer - r) / shell_width, 0, 1)
    outer_mask = outer_mask * outer_mask * (3 - 2 * outer_mask)
    
    # Combined shell mask
    shell_mask = inner_mask * outer_mask
    
    return density * shell_mask


def add_sphere_base(
    density: np.ndarray,
    grid_coords: np.ndarray,
    sphere_radius: float,
    base_strength: float = 0.0
) -> np.ndarray:
    """
    Optionally add a base sphere contribution to the density.
    
    Set base_strength > 0 to have ridges on a solid sphere base.
    Set base_strength = 0 for ridges only (floating shell).
    """
    if base_strength <= 0:
        return density
    
    r = np.linalg.norm(grid_coords, axis=1)
    sphere_sdf = r - sphere_radius
    sphere_density = np.clip(-sphere_sdf / (sphere_radius * 0.05), 0, 1) * base_strength
    
    return np.maximum(density, sphere_density)


def build_h3_ridge_shell(
    curves: List[np.ndarray],
    sphere_radius: float = 1.0,
    cfg: Optional[Union[H3Config, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build H3 Ridge Shell mesh from curves on a sphere.
    
    Main entry point for Module H3.
    
    Args:
        curves: List of (N, 3) polyline arrays, points on/near sphere surface
        sphere_radius: Radius of the base sphere
        cfg: H3Config or dict of config overrides
        
    Returns:
        (vertices, faces) tuple where:
            vertices: (V, 3) float array
            faces: (F, 3) int array (triangle indices)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for H3 module")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for H3 module")
    
    # Parse config
    if cfg is None:
        config = H3Config(sphere_radius=sphere_radius)
    elif isinstance(cfg, dict):
        config = H3Config(sphere_radius=sphere_radius, **cfg)
    else:
        config = cfg
        config.sphere_radius = sphere_radius
    
    logger.info("=" * 60)
    logger.info("Module H3: SDF Ridge Shell")
    logger.info("=" * 60)
    
    # ========== H3.1: Resample curves ==========
    logger.info(f"\n=== H3.1: Resample curves ===")
    logger.info(f"Input: {len(curves)} curves")
    
    total_points_before = sum(len(c) for c in curves)
    
    resampled = resample_curves(curves, config.resample_spacing)
    curve_points = curves_to_points(resampled)
    
    logger.info(f"Resampled: {total_points_before} -> {len(curve_points)} points")
    logger.info(f"Spacing: {config.resample_spacing}")
    
    if len(curve_points) == 0:
        raise ValueError("No curve points after resampling")
    
    # ========== H3.2: Build voxel grid ==========
    logger.info(f"\n=== H3.2: Build voxel grid ===")
    
    # Grid extent based on shell outer boundary
    extent = config.sphere_radius * config.shell_outer * 1.1
    n = config.voxel_resolution
    
    if config.voxel_size is None:
        voxel_size = 2 * extent / n
    else:
        voxel_size = config.voxel_size
    
    logger.info(f"Grid: {n}x{n}x{n}, extent: ±{extent:.3f}, voxel: {voxel_size:.4f}")
    
    # Create grid coordinates
    coords_1d = np.linspace(-extent, extent, n)
    xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    grid_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # ========== H3.3: Build tubular density field ==========
    logger.info(f"\n=== H3.3: Build tubular density field ===")
    logger.info(f"Ridge radius: {config.ridge_radius}")
    logger.info(f"Falloff: {config.density_falloff}")
    
    density = build_tubular_density_field(
        curve_points=curve_points,
        grid_coords=grid_coords,
        tube_radius=config.ridge_radius,
        falloff=config.density_falloff
    )
    
    logger.info(f"Raw density range: {density.min():.4f} to {density.max():.4f}")
    
    # ========== H3.4: Apply shell mask ==========
    logger.info(f"\n=== H3.4: Apply shell mask ===")
    logger.info(f"Shell: {config.shell_inner:.2f}R to {config.shell_outer:.2f}R")
    
    density = apply_shell_mask(
        density=density,
        grid_coords=grid_coords,
        sphere_radius=config.sphere_radius,
        shell_inner=config.shell_inner,
        shell_outer=config.shell_outer
    )
    
    logger.info(f"Masked density range: {density.min():.4f} to {density.max():.4f}")
    
    # Reshape to 3D grid
    density_3d = density.reshape((n, n, n))
    
    # Light smoothing of the field
    density_3d = gaussian_filter(density_3d, sigma=0.5)
    
    # ========== H3.5: Extract iso-surface ==========
    logger.info(f"\n=== H3.5: Extract iso-surface ===")
    logger.info(f"Threshold: {config.iso_threshold}")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            density_3d,
            level=config.iso_threshold,
            spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        # Center the mesh
        verts = verts - extent
        
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError(f"Could not extract surface: {e}")
    
    logger.info(f"Initial mesh: {len(verts)} vertices, {len(faces)} faces")
    
    if len(verts) == 0:
        raise ValueError("Empty mesh - try lowering iso_threshold or increasing ridge_radius")
    
    # ========== H3.6: Mesh cleanup and smoothing ==========
    logger.info(f"\n=== H3.6: Mesh cleanup ===")
    
    if TRIMESH_AVAILABLE:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Keep largest component
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda m: len(m.vertices))
            logger.info(f"Kept largest of {len(components)} components")
        
        # Smoothing
        if config.smooth_iterations > 0:
            logger.info(f"Smoothing: {config.smooth_iterations} iterations, factor {config.smooth_factor}")
            try:
                # Taubin smoothing preserves volume better
                mesh = trimesh.smoothing.filter_taubin(
                    mesh, 
                    iterations=config.smooth_iterations
                )
            except Exception as e:
                logger.warning(f"Smoothing failed: {e}")
        
        verts = mesh.vertices
        faces = mesh.faces
        
        logger.info(f"Final mesh: {len(verts)} vertices, {len(faces)} faces")
        logger.info(f"Watertight: {mesh.is_watertight}")
    
    return verts.astype(np.float32), faces.astype(np.int32)


def curves_from_track_data(
    track_data,
    sphere_radius: float = 1.0
) -> List[np.ndarray]:
    """
    Convert track data to curves on a sphere.
    
    Maps track coordinates to spherical surface.
    
    Args:
        track_data: TrackData object with all_points_m and bounds_m
        sphere_radius: Target sphere radius
        
    Returns:
        List of (N, 3) curve arrays on sphere surface
    """
    points = track_data.all_points_m
    bounds = track_data.bounds_m
    
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    
    # Map to lon/lat range
    lon = ((points[:, 0] - x_min) / (x_max - x_min + 1e-10) - 0.5) * 120  # degrees
    lat = ((points[:, 1] - y_min) / (y_max - y_min + 1e-10) - 0.5) * 60
    
    # Convert to spherical coordinates
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    # Spherical to Cartesian
    x = sphere_radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = sphere_radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = sphere_radius * np.sin(lat_rad)
    
    curve_points = np.column_stack([x, y, z])
    
    # Return as single curve for now
    # TODO: split by track_id for multi-curve
    return [curve_points]


def build_h3_from_tracks(
    track_data,
    config = None,
    h3_config: Optional[H3Config] = None
) -> Tuple:
    """
    Build H3 Ridge Shell from track data.
    
    Convenience wrapper that handles coordinate mapping.
    
    Args:
        track_data: TrackData object
        config: Pipeline Config for unit mode
        h3_config: H3-specific configuration
        
    Returns:
        (mesh, metadata, stats) tuple
    """
    from common.config import Config, UnitMode, MeshMetadata
    from common.normalize import normalize_mesh
    from common.mesh_ops import ensure_manifold, compute_mesh_stats
    
    config = config or Config()
    h3_config = h3_config or H3Config()
    
    logger.info(f"Track data: {len(track_data.all_points_m)} points from {track_data.n_tracks} tracks")
    
    # Convert tracks to curves on sphere
    sphere_radius = 1.0
    curves = curves_from_track_data(track_data, sphere_radius)
    
    logger.info(f"Converted to {len(curves)} curves on sphere")
    
    # Build ridge shell
    verts, faces = build_h3_ridge_shell(
        curves=curves,
        sphere_radius=sphere_radius,
        cfg=h3_config
    )
    
    # Create trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Ensure manifold
    mesh = ensure_manifold(mesh)
    
    # Normalize
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
    if config.unit_mode == UnitMode.NORMALIZED:
        from common.normalize import normalize_mesh
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
    
    quality_stats = {
        "watertight": bool(mesh.is_watertight),
        "n_curves": len(curves),
        "total_curve_points": sum(len(c) for c in curves),
    }
    
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="H3",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "algorithm": "sdf_ridge_shell",
            **h3_config.to_dict(),
            "quality": quality_stats
        }
    )
    
    logger.info(f"\n=== Result ===")
    logger.info(f"Vertices: {metadata.n_vertices}, Triangles: {metadata.n_triangles}")
    logger.info(f"Watertight: {mesh.is_watertight}")
    
    return mesh, metadata, quality_stats


def save_mesh_ply(verts: np.ndarray, faces: np.ndarray, path: Path):
    """Save mesh to PLY format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    logger.info(f"Saved PLY: {path}")


def save_debug_volume(density_3d: np.ndarray, path: Path, config: H3Config):
    """Save volume field for debugging."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(path, 
             density=density_3d,
             config=config.to_dict())
    
    logger.info(f"Saved debug volume: {path}")


# ============== CLI ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Module H3: SDF Ridge Shell"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        required=True,
        help="Track data CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs/option_H3_ridge_shell"),
        help="Output directory"
    )
    parser.add_argument(
        "--ridge-radius",
        type=float,
        default=0.03,
        help="Ridge tube radius (smaller = thinner ridges)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Iso-surface threshold (higher = thinner)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Voxel grid resolution"
    )
    parser.add_argument(
        "--format",
        choices=["ply", "glb", "obj"],
        default="glb",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load track data
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from common.io import load_tracks
    from common.config import Config, UnitMode
    
    track_data = load_tracks(args.data)
    
    # Configure
    config = Config(unit_mode=UnitMode.NORMALIZED)
    h3_config = H3Config(
        ridge_radius=args.ridge_radius,
        iso_threshold=args.threshold,
        voxel_resolution=args.resolution
    )
    
    # Build
    mesh, metadata, _ = build_h3_from_tracks(track_data, config, h3_config)
    
    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.format == "ply":
        output_path = args.output / "meshes" / f"{track_data.specimen_id}.ply"
        save_mesh_ply(mesh.vertices, mesh.faces, output_path)
    else:
        from common.io import save_mesh
        output_path = args.output / "meshes" / f"{track_data.specimen_id}.glb"
        save_mesh(mesh, output_path, metadata)
    
    logger.info(f"Output: {output_path}")
