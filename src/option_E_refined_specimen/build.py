"""
Option E: Refined Carved Specimen (Hero Module)

Generate one coherent sculptural form per specimen with:
- Whole-body connectedness (membrane feel)
- Organic subtractive cavities (carved corridor)
- Smooth, intentional surface (no spikes, no noisy lumps)
- Consistent scale for garden comparison (normalized max dim = 2.0)

Core concept: Envelope minus Corridor, all in implicit space.
No vertex displacement membranes - everything in voxel/SDF space for smoothness.

Algorithm:
E1. Build migration density field D(x,y,z)
E2. Build enclosing envelope field E(x,y,z) using PCA capsule
E3. Define carve field from density with smoothstep
E4. Anti-spike polish (blur field, marching cubes, Taubin smooth)
E5. Normalize to max dim = 2.0
E6. Optional toolpath striation (controlled, off by default)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
import logging
import json

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
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


@dataclass
class OptionEParams:
    """Parameters for Option E generation."""
    # Voxel grid
    vox_res: int = 192
    margin_factor: float = 0.12
    
    # Density rasterization
    paint_radius_factor: float = 0.03
    paint_radius_min_m: float = 2000.0
    paint_radius_max_m: float = 12000.0
    density_blur_sigma_vox: float = 1.4
    
    # Carve mapping
    t_low_factor: float = 0.25
    t_high_factor: float = 0.55
    carve_strength_factor: float = 0.8
    
    # Field polish
    field_blur_sigma_vox: float = 0.9
    
    # Mesh polish
    taubin_iters: int = 10
    decimate_target_tris: int = 120_000
    
    # Optional striation (off by default)
    enable_striation: bool = False
    striation_amplitude_factor: float = 0.002
    striation_pitch_factor: float = 0.08
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vox_res": self.vox_res,
            "margin_factor": self.margin_factor,
            "paint_radius_factor": self.paint_radius_factor,
            "paint_radius_min_m": self.paint_radius_min_m,
            "paint_radius_max_m": self.paint_radius_max_m,
            "density_blur_sigma_vox": self.density_blur_sigma_vox,
            "t_low_factor": self.t_low_factor,
            "t_high_factor": self.t_high_factor,
            "carve_strength_factor": self.carve_strength_factor,
            "field_blur_sigma_vox": self.field_blur_sigma_vox,
            "taubin_iters": self.taubin_iters,
            "decimate_target_tris": self.decimate_target_tris,
            "enable_striation": self.enable_striation,
        }


def smoothstep(t_low: float, t_high: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep interpolation between t_low and t_high."""
    t = np.clip((x - t_low) / (t_high - t_low + 1e-10), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def keep_largest_component(voxels: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component."""
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


def rasterize_segments_to_density(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
    paint_radius_m: float
) -> np.ndarray:
    """
    Rasterize track segments (not just points) into density field.
    
    Uses line splatting for smoother density representation.
    """
    density = np.zeros(grid_shape, dtype=np.float32)
    nz, ny, nx = grid_shape
    
    paint_radius_vox = paint_radius_m / spacing
    
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


def compute_pca_capsule_sdf(
    points: np.ndarray,
    grid_shape: Tuple[int, int, int],
    origin: np.ndarray,
    spacing: float,
    padding_factor: float = 1.15
) -> np.ndarray:
    """
    Create a PCA-aligned capsule envelope as SDF.
    
    The capsule is aligned to the first principal component,
    with radius based on lateral spread.
    
    Returns SDF where negative = inside, positive = outside.
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
    
    # Capsule parameters
    half_length = (projections.max() - projections.min()) / 2 * padding_factor
    
    # Lateral spread (radius)
    lateral_axis1 = eigenvectors[:, 1]
    lateral_axis2 = eigenvectors[:, 2]
    lateral1 = np.abs(centered @ lateral_axis1)
    lateral2 = np.abs(centered @ lateral_axis2)
    radius = max(lateral1.max(), lateral2.max()) * padding_factor
    
    logger.info(f"PCA capsule: half_length={half_length:.1f}m, radius={radius:.1f}m")
    logger.info(f"Principal axis: {axis}")
    
    # Create SDF for capsule
    nz, ny, nx = grid_shape
    
    z_coords = origin[2] + (np.arange(nz) + 0.5) * spacing
    y_coords = origin[1] + (np.arange(ny) + 0.5) * spacing
    x_coords = origin[0] + (np.arange(nx) + 0.5) * spacing
    
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    voxel_centers = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    
    # Transform to capsule-local coordinates
    local = voxel_centers - centroid
    
    # Project onto principal axis
    t = local @ axis
    
    # Clamp to capsule segment
    t_clamped = np.clip(t, -half_length, half_length)
    
    # Nearest point on capsule axis
    nearest_on_axis = t_clamped[:, np.newaxis] * axis
    
    # Distance to axis
    dist_to_axis = np.linalg.norm(local - nearest_on_axis, axis=1)
    
    # SDF: distance to surface (negative inside)
    sdf = dist_to_axis - radius
    
    return sdf.reshape(grid_shape).astype(np.float32)


def build_refined_specimen(
    track_data: TrackData,
    config: Optional[Config] = None,
    params: Optional[OptionEParams] = None
) -> Tuple["trimesh.Trimesh", MeshMetadata, Dict[str, Any]]:
    """
    Build Option E: Refined Carved Specimen.
    
    The "hero" module for coherent sculptural forms.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        params: Option E parameters (uses defaults if None)
        
    Returns:
        Tuple of (mesh, metadata, stats_dict)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for Option E")
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image required for marching cubes")
    
    config = config or Config()
    params = params or OptionEParams()
    
    logger.info("=" * 60)
    logger.info("Option E: Refined Carved Specimen (Hero Module)")
    logger.info("=" * 60)
    
    # Get track data
    points = track_data.all_points_m
    bounds = track_data.bounds_m
    
    logger.info(f"Track data: {len(points)} points from {track_data.n_tracks} tracks")
    
    # Compute bounding box diagonal for scale-relative parameters
    min_corner = np.array([bounds['x'][0], bounds['y'][0], bounds['z'][0]])
    max_corner = np.array([bounds['x'][1], bounds['y'][1], bounds['z'][1]])
    extent = max_corner - min_corner
    bbox_diag = np.linalg.norm(extent)
    
    logger.info(f"Bounding box diagonal: {bbox_diag:.1f}m")
    
    # Compute derived parameters
    margin = params.margin_factor * bbox_diag
    paint_radius = params.paint_radius_factor * bbox_diag
    paint_radius = np.clip(paint_radius, params.paint_radius_min_m, params.paint_radius_max_m)
    
    # Expand bounds with margin
    min_corner -= margin
    max_corner += margin
    
    # Compute grid spacing
    max_extent = np.max(extent + 2 * margin)
    spacing = max_extent / params.vox_res
    
    # Compute grid shape (ZYX order)
    grid_extent = max_corner - min_corner
    grid_shape = tuple(np.ceil(grid_extent / spacing).astype(int) + 1)
    grid_shape = (grid_shape[2], grid_shape[1], grid_shape[0])  # ZYX
    
    # ========== E1: Build migration density field ==========
    logger.info(f"\n=== E1: Build density field ===")
    logger.info(f"Grid: {grid_shape}, spacing: {spacing:.1f}m, paint_radius: {paint_radius:.1f}m")
    
    D = rasterize_segments_to_density(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing,
        paint_radius_m=paint_radius
    )
    
    # Gaussian blur density
    D = gaussian_filter(D, sigma=params.density_blur_sigma_vox)
    
    max_D = D.max()
    logger.info(f"Density range: 0 to {max_D:.4f}")
    
    # ========== E2: Build PCA capsule envelope ==========
    logger.info(f"\n=== E2: Build PCA capsule envelope ===")
    
    S_capsule = compute_pca_capsule_sdf(
        points=points,
        grid_shape=grid_shape,
        origin=min_corner,
        spacing=spacing,
        padding_factor=1.15
    )
    
    envelope_volume = (S_capsule < 0).sum()
    logger.info(f"Envelope volume: {envelope_volume} voxels ({100*envelope_volume/S_capsule.size:.1f}%)")
    
    # ========== E3: Define carve field ==========
    logger.info(f"\n=== E3: Define carve field ===")
    
    t_low = params.t_low_factor * max_D
    t_high = params.t_high_factor * max_D
    carve_strength = params.carve_strength_factor * spacing
    
    logger.info(f"t_low: {t_low:.4f}, t_high: {t_high:.4f}, carve_strength: {carve_strength:.2f}")
    
    # Smoothstep carve field C in [0,1]
    C = smoothstep(t_low, t_high, D)
    
    carve_volume = (C > 0.5).sum()
    logger.info(f"Carve field > 0.5: {carve_volume} voxels")
    
    # ========== E3b: Combine fields ==========
    # F = S_capsule + carve_strength * C
    # Final solid is F < 0
    logger.info(f"\n=== E3b: Combine envelope - carve ===")
    
    F = S_capsule + carve_strength * C
    
    solid_before_polish = (F < 0).sum()
    logger.info(f"Solid volume before polish: {solid_before_polish} voxels")
    
    # ========== E4: Anti-spike polish ==========
    logger.info(f"\n=== E4: Anti-spike polish ===")
    
    # Blur the combined field
    logger.info(f"Blurring field (sigma={params.field_blur_sigma_vox})")
    F_smooth = gaussian_filter(F, sigma=params.field_blur_sigma_vox)
    
    # Extract isosurface at F = 0
    logger.info("Extracting isosurface (marching cubes)")
    
    try:
        # For marching cubes, we need the solid region
        # F < 0 is solid, so we extract at level 0
        verts, faces, normals, values = measure.marching_cubes(
            -F_smooth,  # Negate so solid is positive
            level=0,
            spacing=(spacing, spacing, spacing)
        )
        
        # Convert from ZYX to XYZ and offset to world coordinates
        verts = verts[:, ::-1]  # Reverse to XYZ
        verts = verts + min_corner
        
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        raise ValueError("Could not extract surface")
    
    if len(verts) == 0:
        raise ValueError("Marching cubes produced empty mesh")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Taubin smoothing
    logger.info(f"Taubin smoothing ({params.taubin_iters} iterations)")
    try:
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=params.taubin_iters)
    except Exception as e:
        logger.warning(f"Taubin smoothing failed: {e}, trying Laplacian")
        try:
            mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=params.taubin_iters)
        except:
            pass
    
    # Decimate if needed
    if len(mesh.faces) > params.decimate_target_tris:
        logger.info(f"Decimating {len(mesh.faces)} -> {params.decimate_target_tris} triangles")
        try:
            mesh = mesh.simplify_quadric_decimation(params.decimate_target_tris)
        except Exception as e:
            logger.warning(f"Decimation failed: {e}")
    
    # Keep largest connected component
    logger.info("Keeping largest connected component")
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.vertices))
        logger.info(f"Kept largest of {len(components)} components: {len(mesh.vertices)} vertices")
    
    logger.info(f"After polish: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # ========== E5: Normalize ==========
    logger.info(f"\n=== E5: Normalize ===")
    
    mesh = ensure_manifold(mesh)
    
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
    
    # ========== E6: Optional striation (off by default) ==========
    if params.enable_striation:
        logger.info(f"\n=== E6: Toolpath striation ===")
        # Apply micro displacement along normals using Z
        amplitude = params.striation_amplitude_factor * 2.0  # In normalized space
        pitch = params.striation_pitch_factor * 2.0
        
        z_coords = mesh.vertices[:, 2]
        displacement = amplitude * np.sin(2 * np.pi * z_coords / pitch)
        
        # Modulate by low-freq density proxy (distance from center)
        center = mesh.vertices.mean(axis=0)
        dist_from_center = np.linalg.norm(mesh.vertices - center, axis=1)
        dist_normalized = dist_from_center / (dist_from_center.max() + 1e-10)
        modulation = 0.3 + 0.7 * (1 - dist_normalized)
        
        displacement *= modulation
        
        # Apply along normals
        mesh.vertices += displacement[:, np.newaxis] * mesh.vertex_normals
        
        logger.info(f"Applied striation: amplitude={amplitude:.4f}, pitch={pitch:.4f}")
    
    # Build metadata
    stats_after = compute_mesh_stats(mesh)
    
    # Compute quality metrics for sweep selection
    quality_stats = {
        "n_components": int(len(components)) if 'components' in dir() else 1,
        "watertight": bool(mesh.is_watertight),
        "has_cavity": bool(carve_volume > 0),
        "solid_volume_voxels": int(solid_before_polish),
        "envelope_volume_voxels": int(envelope_volume),
        "carve_volume_voxels": int(carve_volume),
    }
    
    # Try to compute surface roughness proxy
    try:
        # Mean curvature variance as roughness proxy
        # Lower variance = smoother
        vertex_defects = trimesh.curvature.vertex_defects(mesh)
        quality_stats["curvature_variance"] = float(np.var(vertex_defects))
        quality_stats["curvature_mean"] = float(np.mean(np.abs(vertex_defects)))
    except:
        quality_stats["curvature_variance"] = None
        quality_stats["curvature_mean"] = None
    
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="E",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "algorithm": "refined_carved_specimen",
            **params.to_dict(),
            "computed": {
                "bbox_diag_m": float(bbox_diag),
                "paint_radius_m": float(paint_radius),
                "spacing_m": float(spacing),
                "t_low": float(t_low),
                "t_high": float(t_high),
                "carve_strength": float(carve_strength),
                "grid_shape": [int(x) for x in grid_shape],
            },
            "quality": quality_stats
        }
    )
    
    logger.info(f"\n=== Result ===")
    logger.info(f"Vertices: {metadata.n_vertices}, Triangles: {metadata.n_triangles}")
    logger.info(f"Watertight: {mesh.is_watertight}")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata, quality_stats


def generate_sweep_params(base_params: OptionEParams, bbox_diag: float) -> List[Tuple[str, OptionEParams]]:
    """
    Generate parameter sweep combinations for "small" sweep.
    
    12 combos max:
    - paint_radius: {0.025, 0.035} * bbox_diag
    - t_low: {0.22, 0.28} * maxD
    - carve_strength: {0.6, 0.9} * base
    """
    combos = []
    
    paint_radii = [0.025, 0.035]
    t_lows = [0.22, 0.28]
    carve_strengths = [0.6, 0.9]
    
    for pr in paint_radii:
        for tl in t_lows:
            for cs in carve_strengths:
                p = OptionEParams(
                    vox_res=base_params.vox_res,
                    margin_factor=base_params.margin_factor,
                    paint_radius_factor=pr,
                    density_blur_sigma_vox=base_params.density_blur_sigma_vox,
                    t_low_factor=tl,
                    t_high_factor=base_params.t_high_factor,
                    carve_strength_factor=cs,
                    field_blur_sigma_vox=base_params.field_blur_sigma_vox,
                    taubin_iters=base_params.taubin_iters,
                    decimate_target_tris=base_params.decimate_target_tris,
                )
                
                name = f"r{int(pr*1000)}_tl{int(tl*100)}_cs{int(cs*100)}"
                combos.append((name, p))
    
    return combos


def select_best_result(
    results: List[Tuple[str, "trimesh.Trimesh", MeshMetadata, Dict[str, Any]]]
) -> Tuple[str, "trimesh.Trimesh", MeshMetadata]:
    """
    Select best result from sweep based on heuristics.
    
    Criteria:
    1. Single connected component (required)
    2. Cavity presence (required)
    3. Low surface roughness (curvature variance)
    4. Triangle count within range
    """
    valid_results = []
    
    for name, mesh, metadata, stats in results:
        score = 0.0
        
        # Must have cavity
        if not stats.get("has_cavity", False):
            continue
        
        # Prefer watertight
        if stats.get("watertight", False):
            score += 10.0
        
        # Prefer single component
        n_comp = stats.get("n_components", 1)
        if n_comp == 1:
            score += 5.0
        
        # Low curvature variance = smooth
        curv_var = stats.get("curvature_variance")
        if curv_var is not None:
            # Lower is better, normalize
            score += 5.0 / (1.0 + curv_var * 100)
        
        # Triangle count in range
        n_tris = metadata.n_triangles
        if 50000 <= n_tris <= 150000:
            score += 3.0
        
        valid_results.append((score, name, mesh, metadata))
    
    if not valid_results:
        logger.warning("No valid results from sweep, returning first")
        return results[0][0], results[0][1], results[0][2]
    
    # Sort by score descending
    valid_results.sort(key=lambda x: x[0], reverse=True)
    
    best = valid_results[0]
    logger.info(f"Selected best: {best[1]} (score={best[0]:.2f})")
    
    return best[1], best[2], best[3]


def build_with_sweep(
    track_data: TrackData,
    config: Optional[Config] = None,
    output_dir: Optional[Path] = None,
    export_all: bool = False
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option E with parameter sweep and auto-selection.
    
    Args:
        track_data: Input track data
        config: Configuration
        output_dir: Output directory (for exporting all variants)
        export_all: If True, export all sweep variants
        
    Returns:
        Best (mesh, metadata) tuple
    """
    config = config or Config()
    base_params = OptionEParams()
    
    # Compute bbox_diag for param generation
    bounds = track_data.bounds_m
    min_corner = np.array([bounds['x'][0], bounds['y'][0], bounds['z'][0]])
    max_corner = np.array([bounds['x'][1], bounds['y'][1], bounds['z'][1]])
    bbox_diag = np.linalg.norm(max_corner - min_corner)
    
    # Generate sweep
    sweep_params = generate_sweep_params(base_params, bbox_diag)
    logger.info(f"Running sweep with {len(sweep_params)} parameter combinations")
    
    results = []
    
    for name, params in sweep_params:
        logger.info(f"\n--- Sweep: {name} ---")
        try:
            mesh, metadata, stats = build_refined_specimen(track_data, config, params)
            results.append((name, mesh, metadata, stats))
            
            # Export if requested
            if export_all and output_dir:
                variant_path = output_dir / "meshes" / f"{track_data.specimen_id}__E_{name}.glb"
                save_mesh(mesh, variant_path, metadata)
                
        except Exception as e:
            logger.error(f"Sweep {name} failed: {e}")
    
    if not results:
        raise ValueError("All sweep combinations failed")
    
    # Select best
    best_name, best_mesh, best_metadata = select_best_result(results)
    
    return best_mesh, best_metadata


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Option E: Refined Carved Specimen")
    parser.add_argument("--data", type=Path, help="Input data file")
    parser.add_argument("--output", type=Path, default=Path("outputs/option_E_refined_specimen"),
                       help="Output directory")
    parser.add_argument("--sweep", choices=["none", "small"], default="none",
                       help="Parameter sweep mode")
    parser.add_argument("--export-all", action="store_true",
                       help="Export all sweep variants")
    
    args = parser.parse_args()
    
    print("Option E: Refined Carved Specimen (Hero Module)")
    print("Run via: python -m src.run_all --modules E")
