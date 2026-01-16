"""
Mesh operation utilities.

Common mesh operations: smoothing, manifold repair, statistics.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from scipy.ndimage import uniform_filter1d
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_mesh_stats(mesh: "trimesh.Trimesh") -> Dict[str, Any]:
    """
    Compute comprehensive mesh statistics.
    
    Args:
        mesh: Trimesh mesh object
        
    Returns:
        Dictionary of mesh statistics
    """
    bounds = mesh.bounds
    extents = mesh.extents
    
    return {
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "bounds": {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist()
        },
        "extents": extents.tolist(),
        "max_extent": float(max(extents)),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area": float(mesh.area),
        "is_watertight": mesh.is_watertight,
        "is_winding_consistent": mesh.is_winding_consistent,
        "euler_number": mesh.euler_number
    }


def smooth_mesh(
    mesh: "trimesh.Trimesh",
    iterations: int = 3,
    lamb: float = 0.5
) -> "trimesh.Trimesh":
    """
    Apply Laplacian smoothing to mesh.
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        lamb: Smoothing factor (0-1)
        
    Returns:
        Smoothed mesh copy
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh smoothing")
    
    smoothed = mesh.copy()
    vertices = smoothed.vertices.copy()
    
    # Build adjacency
    edges = smoothed.edges_unique
    n_verts = len(vertices)
    
    for iteration in range(iterations):
        # Compute vertex neighbors mean
        new_vertices = vertices.copy()
        
        # For each vertex, average with neighbors
        for edge in edges:
            v0, v1 = edge
            # Move v0 toward v1
            new_vertices[v0] += lamb * (vertices[v1] - vertices[v0]) / 6
            # Move v1 toward v0
            new_vertices[v1] += lamb * (vertices[v0] - vertices[v1]) / 6
        
        vertices = new_vertices
    
    smoothed.vertices = vertices
    return smoothed


def smooth_mesh_laplacian(
    mesh: "trimesh.Trimesh",
    iterations: int = 20,
    lamb: float = 0.5
) -> "trimesh.Trimesh":
    """
    Apply proper Laplacian smoothing using sparse matrices.
    
    More accurate but slower than simple smoothing.
    
    Args:
        mesh: Input mesh
        iterations: Number of smoothing iterations
        lamb: Smoothing factor
        
    Returns:
        Smoothed mesh copy
    """
    if not TRIMESH_AVAILABLE or not SCIPY_AVAILABLE:
        raise ImportError("trimesh and scipy required for Laplacian smoothing")
    
    smoothed = mesh.copy()
    vertices = smoothed.vertices.copy()
    n_verts = len(vertices)
    
    # Build Laplacian matrix
    L = lil_matrix((n_verts, n_verts))
    
    for face in smoothed.faces:
        for i in range(3):
            v0 = face[i]
            v1 = face[(i + 1) % 3]
            L[v0, v1] = 1
            L[v1, v0] = 1
    
    # Normalize rows
    row_sums = np.array(L.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    
    for i in range(n_verts):
        if row_sums[i] > 0:
            L[i, :] /= row_sums[i]
            L[i, i] = -1
    
    L = L.tocsr()
    
    # Apply smoothing iterations
    for _ in range(iterations):
        vertices = vertices - lamb * (L @ vertices)
    
    smoothed.vertices = vertices
    return smoothed


def ensure_manifold(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Attempt to repair mesh to be manifold.
    
    Args:
        mesh: Input mesh
        
    Returns:
        Repaired mesh (best effort)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for manifold repair")
    
    repaired = mesh.copy()
    
    # Remove degenerate faces (trimesh 4.x API)
    if hasattr(repaired, 'remove_degenerate_faces'):
        repaired.remove_degenerate_faces()
    elif hasattr(repaired, 'update_faces'):
        # Filter degenerate faces manually
        mask = repaired.nondegenerate_faces()
        if mask is not None and not mask.all():
            repaired.update_faces(mask)
    
    # Remove duplicate faces
    if hasattr(repaired, 'remove_duplicate_faces'):
        repaired.remove_duplicate_faces()
    
    # Remove unreferenced vertices
    if hasattr(repaired, 'remove_unreferenced_vertices'):
        repaired.remove_unreferenced_vertices()
    
    # Fill holes if possible
    if hasattr(repaired, 'fill_holes'):
        try:
            repaired.fill_holes()
        except Exception:
            pass
    
    # Fix normals
    if hasattr(repaired, 'fix_normals'):
        repaired.fix_normals()
    
    logger.info(f"Manifold repair: {len(mesh.vertices)}→{len(repaired.vertices)} verts, "
                f"watertight={repaired.is_watertight}")
    
    return repaired


def simplify_mesh(
    mesh: "trimesh.Trimesh",
    target_faces: int
) -> "trimesh.Trimesh":
    """
    Simplify mesh to target face count.
    
    Args:
        mesh: Input mesh
        target_faces: Target number of faces
        
    Returns:
        Simplified mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for simplification")
    
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    
    # Use trimesh's built-in simplification
    simplified = mesh.simplify_quadric_decimation(target_faces)
    
    logger.info(f"Simplified: {len(mesh.faces)}→{len(simplified.faces)} faces")
    
    return simplified


def merge_meshes(meshes: list) -> "trimesh.Trimesh":
    """
    Merge multiple meshes into one.
    
    Args:
        meshes: List of trimesh meshes
        
    Returns:
        Combined mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh merging")
    
    if not meshes:
        return trimesh.Trimesh()
    
    if len(meshes) == 1:
        return meshes[0].copy()
    
    combined = trimesh.util.concatenate(meshes)
    logger.info(f"Merged {len(meshes)} meshes: {len(combined.vertices)} verts, {len(combined.faces)} faces")
    
    return combined


def create_tube_mesh(
    centerline: np.ndarray,
    radius: float,
    segments: int = 8
) -> "trimesh.Trimesh":
    """
    Create a tube mesh along a centerline.
    
    Args:
        centerline: Nx3 array of points
        radius: Tube radius
        segments: Number of segments around tube
        
    Returns:
        Tube mesh
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for tube mesh creation")
    
    if len(centerline) < 2:
        return trimesh.Trimesh()
    
    # Use trimesh's sweep function if available
    try:
        # Create circle profile
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        circle = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
        
        # Create path
        from trimesh.creation import sweep_polygon
        from shapely.geometry import Polygon
        
        poly = Polygon(circle)
        tube = sweep_polygon(poly, centerline)
        return tube
    except Exception as e:
        logger.warning(f"Sweep failed, using fallback: {e}")
        
        # Fallback: create simple tube manually
        return _create_tube_manual(centerline, radius, segments)


def _create_tube_manual(
    centerline: np.ndarray,
    radius: float,
    segments: int = 8
) -> "trimesh.Trimesh":
    """Manual tube creation fallback."""
    n_points = len(centerline)
    
    # Generate vertices
    vertices = []
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    
    for i, point in enumerate(centerline):
        # Compute tangent
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i == n_points - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
        
        # Find perpendicular vectors
        if abs(tangent[2]) < 0.9:
            perp1 = np.cross(tangent, [0, 0, 1])
        else:
            perp1 = np.cross(tangent, [1, 0, 0])
        perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
        perp2 = np.cross(tangent, perp1)
        
        # Generate ring vertices
        for angle in angles:
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(point + offset)
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    for i in range(n_points - 1):
        for j in range(segments):
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + j
            v3 = (i + 1) * segments + (j + 1) % segments
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    faces = np.array(faces)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)
