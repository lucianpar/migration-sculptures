"""
Option C: Constrained Membrane (Pressure Skin)

Generate a single enclosing skin deformed by migration density.

Algorithm:
1. Create base membrane (convex hull or sphere)
2. Compute distance field from membrane to tracks
3. Displace vertices along normals based on proximity
4. Laplacian smoothing
5. Normalize to max dim = 2.0
6. Export

Acceptance criteria:
- One continuous membrane
- Bulges correlate with migration density
- No noisy micro-lumps
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
from common.mesh_ops import smooth_mesh_laplacian, smooth_mesh, ensure_manifold, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.spatial import ConvexHull, cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def create_convex_hull_membrane(points: np.ndarray) -> "trimesh.Trimesh":
    """
    Create convex hull membrane around points.
    
    Args:
        points: Nx3 array of points in meters
        
    Returns:
        Convex hull mesh
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for convex hull")
    
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh creation")
    
    # Compute convex hull
    hull = ConvexHull(points)
    
    # Extract vertices and faces
    vertices = points[hull.vertices]
    
    # Remap face indices to new vertex array
    vertex_map = {old: new for new, old in enumerate(hull.vertices)}
    faces = np.array([[vertex_map[v] for v in face] for face in hull.simplices])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    
    logger.info(f"Created convex hull: {len(vertices)} vertices, {len(faces)} faces")
    return mesh


def create_sphere_membrane(
    center: np.ndarray,
    radius: float,
    subdivisions: int = 3
) -> "trimesh.Trimesh":
    """
    Create sphere membrane.
    
    Args:
        center: Sphere center
        radius: Sphere radius
        subdivisions: Icosphere subdivisions
        
    Returns:
        Sphere mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh creation")
    
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.vertices += center
    
    logger.info(f"Created sphere: {len(sphere.vertices)} vertices, radius={radius:.1f}m")
    return sphere


def create_ellipsoid_membrane(
    center: np.ndarray,
    radii: np.ndarray,
    subdivisions: int = 3
) -> "trimesh.Trimesh":
    """
    Create ellipsoid membrane matching data extent.
    
    Args:
        center: Ellipsoid center
        radii: (rx, ry, rz) radii for each axis
        subdivisions: Icosphere subdivisions
        
    Returns:
        Ellipsoid mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh creation")
    
    # Start with unit sphere
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    
    # Scale to ellipsoid
    sphere.vertices *= radii
    sphere.vertices += center
    
    logger.info(f"Created ellipsoid: {len(sphere.vertices)} vertices, radii={radii}")
    return sphere


def compute_displacement_field(
    membrane_vertices: np.ndarray,
    membrane_normals: np.ndarray,
    track_points: np.ndarray,
    sigma_m: float = 3000.0,
    amplitude_m: float = 2000.0,
    direction: str = "inward"
) -> np.ndarray:
    """
    Compute displacement for each membrane vertex based on proximity to tracks.
    
    Displacement formula: disp = exp(-(d^2)/(2*sigma^2)) * amplitude
    
    Args:
        membrane_vertices: Membrane vertex positions
        membrane_normals: Vertex normals
        track_points: Track point positions
        sigma_m: Gaussian sigma in meters
        amplitude_m: Maximum displacement amplitude in meters
        direction: "inward" or "outward"
        
    Returns:
        Nx3 array of displacement vectors
    """
    if SCIPY_AVAILABLE:
        tree = cKDTree(track_points)
        distances, _ = tree.query(membrane_vertices, k=1)
    else:
        # Fallback: brute force
        distances = np.array([
            np.min(np.linalg.norm(track_points - v, axis=1))
            for v in membrane_vertices
        ])
    
    # Gaussian falloff
    weights = np.exp(-(distances**2) / (2 * sigma_m**2))
    
    # Direction multiplier
    sign = -1.0 if direction == "inward" else 1.0
    
    # Displacement along normals
    displacements = sign * amplitude_m * weights[:, np.newaxis] * membrane_normals
    
    logger.info(f"Displacement: max={np.abs(displacements).max():.1f}m, "
                f"mean={np.abs(displacements).mean():.1f}m")
    
    return displacements


def build_constrained_membrane(
    track_data: TrackData,
    config: Optional[Config] = None,
    membrane_type: str = "ellipsoid",
    subdivisions: int = 4,
    sigma_m: float = 3000.0,
    amplitude_m: float = 2500.0,
    smoothing_iterations: int = 25,
    padding_factor: float = 1.3
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option C: Constrained Membrane sculpture.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        membrane_type: "convex_hull", "sphere", or "ellipsoid"
        subdivisions: Mesh subdivisions (for sphere/ellipsoid)
        sigma_m: Gaussian sigma for proximity influence (meters)
        amplitude_m: Maximum displacement amplitude (meters)
        smoothing_iterations: Laplacian smoothing iterations
        padding_factor: How much larger membrane is than data bounds
        
    Returns:
        Tuple of (mesh, metadata)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    
    config = config or Config()
    
    logger.info("=" * 60)
    logger.info("Option C: Constrained Membrane (Pressure Skin)")
    logger.info("=" * 60)
    
    # Get track data
    points = track_data.all_points_m
    bounds = track_data.bounds_m
    
    # Compute center and extent
    center = np.array([
        (bounds['x'][0] + bounds['x'][1]) / 2,
        (bounds['y'][0] + bounds['y'][1]) / 2,
        (bounds['z'][0] + bounds['z'][1]) / 2
    ])
    
    extents = np.array([
        bounds['x'][1] - bounds['x'][0],
        bounds['y'][1] - bounds['y'][0],
        bounds['z'][1] - bounds['z'][0]
    ])
    
    # Step 1: Create base membrane
    logger.info(f"\nStep 1: Creating base membrane ({membrane_type})")
    
    if membrane_type == "convex_hull":
        # Downsample points for hull
        if len(points) > 500:
            indices = np.random.choice(len(points), 500, replace=False)
            hull_points = points[indices]
        else:
            hull_points = points
        membrane = create_convex_hull_membrane(hull_points)
    
    elif membrane_type == "sphere":
        radius = np.max(extents) / 2 * padding_factor
        membrane = create_sphere_membrane(center, radius, subdivisions)
    
    else:  # ellipsoid
        radii = extents / 2 * padding_factor
        membrane = create_ellipsoid_membrane(center, radii, subdivisions)
    
    # Step 2: Compute vertex normals
    logger.info("\nStep 2: Computing vertex normals")
    membrane.fix_normals()
    vertex_normals = membrane.vertex_normals
    
    # Step 3: Compute displacement field
    logger.info("\nStep 3: Computing displacement field")
    displacements = compute_displacement_field(
        membrane_vertices=membrane.vertices,
        membrane_normals=vertex_normals,
        track_points=points,
        sigma_m=sigma_m,
        amplitude_m=amplitude_m,
        direction="inward"
    )
    
    # Step 4: Apply displacement
    logger.info("\nStep 4: Applying displacement")
    membrane.vertices = membrane.vertices + displacements
    
    # Step 5: Laplacian smoothing
    logger.info(f"\nStep 5: Laplacian smoothing ({smoothing_iterations} iterations)")
    try:
        membrane = smooth_mesh_laplacian(membrane, iterations=smoothing_iterations)
    except Exception as e:
        logger.warning(f"Laplacian smoothing failed, using simple smooth: {e}")
        membrane = smooth_mesh(membrane, iterations=smoothing_iterations // 2)
    
    # Step 6: Cleanup
    logger.info("\nStep 6: Mesh cleanup")
    membrane = ensure_manifold(membrane)
    
    # Record bounds before normalization
    stats_before = compute_mesh_stats(membrane)
    max_dim_before = stats_before["max_extent"]
    
    # Step 7: Normalize
    logger.info("\nStep 7: Normalization")
    if config.unit_mode == UnitMode.NORMALIZED:
        membrane, norm_result = normalize_mesh(membrane, config.normalized_max_dim)
        scale_factor = norm_result.scale_factor
        max_dim_after = norm_result.max_dim_after
        normalization_applied = True
    else:
        scale_factor = 1.0
        max_dim_after = max_dim_before
        normalization_applied = False
    
    # Build metadata
    stats_after = compute_mesh_stats(membrane)
    metadata = MeshMetadata(
        unit_mode=config.unit_mode.value,
        bbox_max_dimension=max_dim_after,
        normalization_applied=normalization_applied,
        specimen_id=track_data.specimen_id,
        option="C",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "membrane_type": membrane_type,
            "subdivisions": subdivisions,
            "sigma_m": sigma_m,
            "amplitude_m": amplitude_m,
            "smoothing_iterations": smoothing_iterations,
            "padding_factor": padding_factor
        }
    )
    
    logger.info(f"\nResult: {metadata.n_vertices} vertices, {metadata.n_triangles} triangles")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return membrane, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Option C: Constrained Membrane (Pressure Skin)")
    print("Run via: python -m src.run_all --modules C")
