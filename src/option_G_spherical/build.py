"""
Option G: Spherical Migration Sculpture

Wrap migration tracks around a sphere, creating a globe-like sculpture
where track density creates surface relief.

Algorithm:
G1. Map geographic coordinates (lon/lat) to spherical coordinates (theta/phi)
G2. Create spherical density field from track points
G3. Build base sphere SDF
G4. Modulate sphere radius by density (outward bulges for high activity)
G5. Optional: Carve interior corridors using density
G6. Extract mesh via marching cubes
G7. Polish and normalize

The result is a sphere where migration routes create raised ridges
and dense activity areas create prominent bulges.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

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
    from scipy.ndimage import gaussian_filter, distance_transform_edt, label as ndimage_label
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


@dataclass
class OptionGParams:
    """Parameters for Option G spherical sculpture."""
    # Sphere parameters
    base_radius: float = 1.0  # Base sphere radius (before normalization)
    
    # Density field
    vox_res: int = 128  # Resolution of 3D voxel grid
    angular_res: int = 256  # Resolution of spherical density map (theta x phi)
    
    # Relief parameters
    relief_scale: float = 0.3  # Max relief as fraction of radius (0.3 = 30% bulge)
    relief_power: float = 0.7  # Power for density-to-relief mapping (< 1 = more subtle)
    
    # Smoothing
    density_blur_sigma: float = 3.0  # Blur on spherical density map
    
    # Carving (optional interior detail)
    enable_carving: bool = True
    carve_threshold: float = 0.6  # Density threshold for carving
    carve_depth: float = 0.15  # How deep to carve as fraction of radius
    
    # Mesh polish
    taubin_iters: int = 8
    target_tris: int = 100_000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_radius": self.base_radius,
            "vox_res": self.vox_res,
            "angular_res": self.angular_res,
            "relief_scale": self.relief_scale,
            "relief_power": self.relief_power,
            "density_blur_sigma": self.density_blur_sigma,
            "enable_carving": self.enable_carving,
            "carve_threshold": self.carve_threshold,
            "carve_depth": self.carve_depth,
            "taubin_iters": self.taubin_iters,
            "target_tris": self.target_tris,
        }


def lonlat_to_spherical(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert longitude/latitude to spherical coordinates (theta, phi).
    
    theta: azimuthal angle (0 to 2*pi), from longitude
    phi: polar angle (0 to pi), from latitude
    
    Args:
        lon: Longitude in degrees (-180 to 180)
        lat: Latitude in degrees (-90 to 90)
        
    Returns:
        theta: Azimuthal angle in radians
        phi: Polar angle in radians
    """
    # Convert to radians
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    # theta = longitude (shifted to 0-2pi range)
    theta = lon_rad + np.pi  # Shift from [-pi, pi] to [0, 2*pi]
    
    # phi = colatitude (90 - latitude), from north pole
    phi = np.pi/2 - lat_rad  # [0, pi] where 0 = north pole
    
    return theta, phi


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian.
    
    Args:
        theta: Azimuthal angle (0 to 2*pi)
        phi: Polar angle (0 to pi)
        r: Radius
        
    Returns:
        (N, 3) array of XYZ coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return np.column_stack([x, y, z])


def create_spherical_density_map(
    lon: np.ndarray,
    lat: np.ndarray,
    angular_res: int = 256,
    sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D density map on the sphere surface.
    
    Args:
        lon: Longitude values
        lat: Latitude values
        angular_res: Resolution of the density map
        sigma: Gaussian blur sigma
        
    Returns:
        density_map: (angular_res, angular_res*2) density values
        theta_grid: Theta values for each column
        phi_grid: Phi values for each row
    """
    # Convert to spherical
    theta, phi = lonlat_to_spherical(lon, lat)
    
    # Create grid
    theta_bins = np.linspace(0, 2*np.pi, angular_res * 2 + 1)
    phi_bins = np.linspace(0, np.pi, angular_res + 1)
    
    # Histogram
    density_map, _, _ = np.histogram2d(
        phi, theta,
        bins=[phi_bins, theta_bins]
    )
    
    # Gaussian blur (with wrapping in theta direction)
    # Pad for wrapping
    pad_width = int(sigma * 3)
    padded = np.pad(density_map, ((0, 0), (pad_width, pad_width)), mode='wrap')
    blurred = gaussian_filter(padded, sigma=sigma)
    density_map = blurred[:, pad_width:-pad_width]
    
    # Also blur in phi direction (no wrapping)
    density_map = gaussian_filter(density_map, sigma=(sigma, 0))
    
    # Normalize
    if density_map.max() > 0:
        density_map = density_map / density_map.max()
    
    # Grid coordinates (cell centers)
    theta_grid = (theta_bins[:-1] + theta_bins[1:]) / 2
    phi_grid = (phi_bins[:-1] + phi_bins[1:]) / 2
    
    return density_map, theta_grid, phi_grid


def sample_density_at_angles(
    density_map: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    theta_query: np.ndarray,
    phi_query: np.ndarray
) -> np.ndarray:
    """
    Sample the density map at given spherical coordinates.
    
    Uses bilinear interpolation.
    """
    # Normalize query angles to grid indices
    theta_idx = (theta_query / (2 * np.pi)) * len(theta_grid)
    phi_idx = (phi_query / np.pi) * len(phi_grid)
    
    # Wrap theta
    theta_idx = theta_idx % len(theta_grid)
    
    # Clamp phi
    phi_idx = np.clip(phi_idx, 0, len(phi_grid) - 1)
    
    # Bilinear interpolation
    t0 = np.floor(theta_idx).astype(int) % len(theta_grid)
    t1 = (t0 + 1) % len(theta_grid)
    p0 = np.floor(phi_idx).astype(int)
    p1 = np.minimum(p0 + 1, len(phi_grid) - 1)
    
    tt = theta_idx - np.floor(theta_idx)
    tp = phi_idx - np.floor(phi_idx)
    
    # Sample four corners
    v00 = density_map[p0, t0]
    v01 = density_map[p0, t1]
    v10 = density_map[p1, t0]
    v11 = density_map[p1, t1]
    
    # Interpolate
    v0 = v00 * (1 - tt) + v01 * tt
    v1 = v10 * (1 - tt) + v11 * tt
    density = v0 * (1 - tp) + v1 * tp
    
    return density


def build_spherical_sculpture(
    track_data: TrackData,
    config: Optional[Config] = None,
    params: Optional[OptionGParams] = None
) -> Tuple["trimesh.Trimesh", MeshMetadata, Dict[str, Any]]:
    """
    Build Option G: Spherical Migration Sculpture.
    
    Args:
        track_data: Input track data
        config: Configuration (uses defaults if None)
        params: Option G parameters (uses defaults if None)
        
    Returns:
        Tuple of (mesh, metadata, stats_dict)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Option G")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for marching cubes")
    
    config = config or Config()
    params = params or OptionGParams()
    
    logger.info("=" * 60)
    logger.info("Option G: Spherical Migration Sculpture")
    logger.info("=" * 60)
    
    # Get original lon/lat coordinates (before UTM conversion)
    # We need to reload from the track data's raw coordinates
    points_m = track_data.all_points_m
    
    logger.info(f"Track data: {len(points_m)} points from {track_data.n_tracks} tracks")
    
    # We need lon/lat - check if available in track_data or extract from bounds
    # For now, we'll reverse-project from UTM or use the bounds to estimate
    # Actually, let's work with the UTM coordinates mapped to a sphere
    # treating X as longitude-like and Y as latitude-like
    
    bounds = track_data.bounds_m
    x_coords = points_m[:, 0]
    y_coords = points_m[:, 1]
    
    # Normalize coordinates to lon/lat-like range
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    
    # Map to longitude (-180 to 180) and latitude (-60 to 60) range
    # We'll use a portion of the sphere to avoid pole distortion
    lon = ((x_coords - x_min) / (x_max - x_min + 1e-10) - 0.5) * 120  # -60 to 60 degrees
    lat = ((y_coords - y_min) / (y_max - y_min + 1e-10) - 0.5) * 60   # -30 to 30 degrees
    
    logger.info(f"Mapped coordinates: lon=[{lon.min():.1f}, {lon.max():.1f}], lat=[{lat.min():.1f}, {lat.max():.1f}]")
    
    # ========== G1: Create spherical density map ==========
    logger.info(f"\n=== G1: Create spherical density map ===")
    
    density_map, theta_grid, phi_grid = create_spherical_density_map(
        lon, lat,
        angular_res=params.angular_res,
        sigma=params.density_blur_sigma
    )
    
    logger.info(f"Density map shape: {density_map.shape}")
    logger.info(f"Density range: {density_map.min():.3f} to {density_map.max():.3f}")
    
    # ========== G2: Build 3D voxel grid for marching cubes ==========
    logger.info(f"\n=== G2: Build 3D SDF grid ===")
    
    # Create 3D grid
    n = params.vox_res
    extent = params.base_radius * (1 + params.relief_scale) * 1.2  # Extra margin
    
    coords = np.linspace(-extent, extent, n)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    
    # Convert to spherical
    r_grid = np.sqrt(xx**2 + yy**2 + zz**2)
    theta_grid_3d = np.arctan2(yy, xx) + np.pi  # 0 to 2*pi
    phi_grid_3d = np.arccos(np.clip(zz / (r_grid + 1e-10), -1, 1))  # 0 to pi
    
    # Sample density at each grid point's angular position
    logger.info("Sampling density field...")
    density_3d = sample_density_at_angles(
        density_map, theta_grid, phi_grid,
        theta_grid_3d.ravel(), phi_grid_3d.ravel()
    ).reshape((n, n, n))
    
    # ========== G3: Compute modulated sphere SDF ==========
    logger.info(f"\n=== G3: Compute relief-modulated sphere ===")
    
    # Relief: high density = larger radius (outward bulge)
    relief = params.relief_scale * params.base_radius * (density_3d ** params.relief_power)
    modulated_radius = params.base_radius + relief
    
    # SDF: negative inside sphere
    sdf_sphere = r_grid - modulated_radius
    
    logger.info(f"Relief range: {relief.min():.3f} to {relief.max():.3f}")
    logger.info(f"Modulated radius range: {modulated_radius.min():.3f} to {modulated_radius.max():.3f}")
    
    # ========== G4: Optional carving for interior detail ==========
    if params.enable_carving:
        logger.info(f"\n=== G4: Interior carving ===")
        
        # Create carve field based on high-density areas
        carve_mask = density_3d > params.carve_threshold
        carve_depth_actual = params.carve_depth * params.base_radius
        
        # Inner sphere for carving
        inner_radius = params.base_radius - carve_depth_actual
        sdf_inner = r_grid - inner_radius
        
        # Carve where density is high AND we're in the outer shell
        # This creates tunnels/cavities in dense route areas
        carve_region = carve_mask & (sdf_sphere < 0) & (sdf_inner > 0)
        
        # Apply carving by making those regions positive (outside)
        sdf_final = sdf_sphere.copy()
        # Smooth transition for carving
        carve_sdf = distance_transform_edt(~carve_region) - distance_transform_edt(carve_region)
        carve_sdf = gaussian_filter(carve_sdf.astype(np.float32), sigma=1.0)
        
        # Blend: where carve_region, push SDF positive (hollow out)
        blend_factor = 0.5
        sdf_final = np.where(
            carve_region,
            np.maximum(sdf_sphere, -carve_sdf * 0.1),
            sdf_sphere
        )
        
        carve_volume = carve_region.sum()
        logger.info(f"Carve region: {carve_volume} voxels")
    else:
        sdf_final = sdf_sphere
    
    # Light blur for smoothness
    sdf_final = gaussian_filter(sdf_final.astype(np.float32), sigma=0.5)
    
    solid_volume = (sdf_final < 0).sum()
    logger.info(f"Final solid volume: {solid_volume} voxels")
    
    # ========== G5: Extract mesh ==========
    logger.info(f"\n=== G5: Extract mesh ===")
    
    spacing = 2 * extent / n
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            -sdf_final,  # Negate so solid is positive
            level=0,
            spacing=(spacing, spacing, spacing)
        )
        
        # Center the mesh
        verts = verts - extent
        
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError("Could not extract surface")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Keep largest component
    components = mesh.split(only_watertight=False)
    n_components = len(components)
    if n_components > 1:
        mesh = max(components, key=lambda m: len(m.vertices))
        logger.info(f"Kept largest of {n_components} components")
    
    # ========== G6: Mesh polish ==========
    logger.info(f"\n=== G6: Mesh polish ===")
    
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
    
    # ========== G7: Normalize ==========
    logger.info(f"\n=== G7: Normalize ===")
    
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
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
    
    quality_stats = {
        "n_components": int(n_components),
        "watertight": bool(mesh.is_watertight),
        "solid_volume_voxels": int(solid_volume),
        "density_map_shape": list(density_map.shape),
    }
    
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="G",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "algorithm": "spherical_migration_sculpture",
            **params.to_dict(),
            "computed": {
                "lon_range": [float(lon.min()), float(lon.max())],
                "lat_range": [float(lat.min()), float(lat.max())],
                "voxel_extent": float(extent),
                "voxel_spacing": float(spacing),
            },
            "quality": quality_stats
        }
    )
    
    logger.info(f"\n=== Result ===")
    logger.info(f"Vertices: {metadata.n_vertices}, Triangles: {metadata.n_triangles}")
    logger.info(f"Watertight: {mesh.is_watertight}")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata, quality_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Option G: Spherical Migration Sculpture")
    parser.add_argument("--data", "-d", type=Path, required=True, help="Track data CSV file")
    parser.add_argument("--output", "-o", type=Path, default=Path("outputs/option_G_spherical"))
    parser.add_argument("--relief", type=float, default=0.3, help="Relief scale (0-1)")
    parser.add_argument("--carving", action="store_true", help="Enable interior carving")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    from common.io import load_tracks
    
    track_data = load_tracks(args.data)
    config = Config(unit_mode=UnitMode.NORMALIZED)
    params = OptionGParams(
        relief_scale=args.relief,
        enable_carving=args.carving
    )
    
    mesh, metadata, _ = build_spherical_sculpture(track_data, config, params)
    
    output_path = args.output / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    logger.info(f"Saved: {output_path}")
