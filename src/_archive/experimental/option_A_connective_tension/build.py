"""
Option A: Connective Tension Structure

Turn migration data into a single connected filament system where:
- nodes = spatial anchors from tracks
- edges = tension-bearing connectors

Algorithm:
1. Node extraction (resample + cluster)
2. Graph construction (kNN)
3. Curve generation (splines with sag)
4. Tube sweep
5. Normalize to max dim = 2.0
6. Export

Acceptance criteria:
- ONE connected object
- No floating pieces
- Reads as infrastructure / connective tissue
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.config import Config, UnitMode, MeshMetadata
from common.io import TrackData, save_mesh
from common.normalize import normalize_mesh
from common.mesh_ops import create_tube_mesh, merge_meshes, smooth_mesh, compute_mesh_stats

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using fallback clustering")

try:
    from scipy.interpolate import splprep, splev
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def extract_nodes(
    track_data: TrackData,
    resample_distance_m: float = 3000.0,
    n_clusters: int = 100,
    method: str = "kmeans"
) -> np.ndarray:
    """
    Extract spatial anchor nodes from tracks.
    
    Args:
        track_data: Input track data in meters
        resample_distance_m: Resample tracks to this spacing
        n_clusters: Target number of nodes
        method: "kmeans" or "dbscan"
        
    Returns:
        Nx3 array of node positions in meters
    """
    # Resample tracks
    resampled = track_data.resample_all(resample_distance_m)
    points = resampled.all_points_m
    
    logger.info(f"Resampled to {len(points)} points for node extraction")
    
    if len(points) < n_clusters:
        logger.warning(f"Fewer points ({len(points)}) than target clusters ({n_clusters})")
        return points
    
    if SKLEARN_AVAILABLE and method == "kmeans":
        # K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(points)), random_state=42)
        kmeans.fit(points)
        nodes = kmeans.cluster_centers_
    elif SKLEARN_AVAILABLE and method == "dbscan":
        # DBSCAN clustering
        eps = resample_distance_m * 1.5
        dbscan = DBSCAN(eps=eps, min_samples=2)
        labels = dbscan.fit_predict(points)
        
        # Get cluster centers
        unique_labels = set(labels) - {-1}
        nodes = []
        for label in unique_labels:
            cluster_points = points[labels == label]
            nodes.append(cluster_points.mean(axis=0))
        nodes = np.array(nodes) if nodes else points[:n_clusters]
    else:
        # Fallback: uniform sampling
        indices = np.linspace(0, len(points) - 1, n_clusters, dtype=int)
        nodes = points[indices]
    
    logger.info(f"Extracted {len(nodes)} nodes")
    return nodes


def build_graph(
    nodes: np.ndarray,
    k: int = 3,
    max_edge_percentile: float = 90
) -> List[Tuple[int, int]]:
    """
    Build kNN graph between nodes.
    
    Args:
        nodes: Nx3 array of node positions
        k: Number of nearest neighbors
        max_edge_percentile: Remove edges above this percentile length
        
    Returns:
        List of (node_i, node_j) edges
    """
    if SKLEARN_AVAILABLE:
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(nodes)))
        nn.fit(nodes)
        distances, indices = nn.kneighbors(nodes)
        
        # Build edges (skip self-connection at index 0)
        edges = set()
        edge_lengths = []
        for i, neighbors in enumerate(indices):
            for j, neighbor in enumerate(neighbors[1:]):  # Skip self
                edge = tuple(sorted([i, neighbor]))
                if edge not in edges:
                    edges.add(edge)
                    edge_lengths.append(distances[i, j + 1])
        
        edges = list(edges)
        edge_lengths = np.array(edge_lengths)
    else:
        # Fallback: connect nearest neighbors manually
        edges = []
        edge_lengths = []
        for i in range(len(nodes)):
            dists = np.linalg.norm(nodes - nodes[i], axis=1)
            dists[i] = np.inf  # Exclude self
            nearest = np.argsort(dists)[:k]
            for j in nearest:
                edge = tuple(sorted([i, j]))
                if edge not in edges:
                    edges.append(edge)
                    edge_lengths.append(dists[j])
        edge_lengths = np.array(edge_lengths)
    
    # Filter by length percentile
    if len(edge_lengths) > 0:
        threshold = np.percentile(edge_lengths, max_edge_percentile)
        edges = [e for e, l in zip(edges, edge_lengths) if l <= threshold]
    
    logger.info(f"Built graph: {len(nodes)} nodes, {len(edges)} edges")
    return edges


def ensure_connected(
    nodes: np.ndarray,
    edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Ensure graph is connected by adding minimum spanning tree edges.
    
    Args:
        nodes: Node positions
        edges: Current edges
        
    Returns:
        Modified edge list ensuring connectivity
    """
    n = len(nodes)
    if n <= 1:
        return edges
    
    # Build adjacency
    adjacency = {i: set() for i in range(n)}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    # Find connected components
    visited = set()
    components = []
    
    for start in range(n):
        if start in visited:
            continue
        component = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        components.append(component)
    
    if len(components) == 1:
        logger.info("Graph is already connected")
        return edges
    
    logger.info(f"Graph has {len(components)} components, connecting...")
    
    # Connect components with shortest edges
    new_edges = list(edges)
    component_list = [list(c) for c in components]
    
    for i in range(len(component_list) - 1):
        comp_a = component_list[i]
        comp_b = component_list[i + 1]
        
        # Find shortest edge between components
        best_dist = np.inf
        best_edge = None
        
        for a in comp_a:
            for b in comp_b:
                dist = np.linalg.norm(nodes[a] - nodes[b])
                if dist < best_dist:
                    best_dist = dist
                    best_edge = (a, b)
        
        if best_edge:
            new_edges.append(best_edge)
            # Merge components
            component_list[i + 1] = comp_a + comp_b
    
    logger.info(f"Added {len(new_edges) - len(edges)} edges to ensure connectivity")
    return new_edges


def create_curved_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    sag: float = 0.1,
    n_points: int = 20
) -> np.ndarray:
    """
    Create curved edge between two points with sag.
    
    Args:
        p0: Start point
        p1: End point
        sag: Sag factor (0 = straight, higher = more curve)
        n_points: Number of points along curve
        
    Returns:
        Nx3 array of curve points
    """
    # Parameterize along edge
    t = np.linspace(0, 1, n_points)
    
    # Linear interpolation
    points = p0[np.newaxis, :] * (1 - t)[:, np.newaxis] + p1[np.newaxis, :] * t[:, np.newaxis]
    
    # Add sag (catenary-like curve, downward in z)
    edge_length = np.linalg.norm(p1 - p0)
    sag_amount = edge_length * sag
    
    # Parabolic sag profile
    sag_profile = 4 * t * (1 - t)  # Peaks at t=0.5
    
    # Apply sag in negative z direction (like gravity)
    points[:, 2] -= sag_profile * sag_amount
    
    return points


def compute_density_at_nodes(
    nodes: np.ndarray,
    track_data: TrackData,
    radius_m: float = 5000.0
) -> np.ndarray:
    """
    Compute track density around each node.
    
    Args:
        nodes: Node positions in meters
        track_data: Original track data
        radius_m: Influence radius
        
    Returns:
        Array of density values per node
    """
    points = track_data.all_points_m
    
    if SCIPY_AVAILABLE:
        tree = cKDTree(points)
        counts = tree.query_ball_point(nodes, r=radius_m, return_length=True)
        densities = np.array(counts, dtype=float)
    else:
        densities = np.zeros(len(nodes))
        for i, node in enumerate(nodes):
            dists = np.linalg.norm(points - node, axis=1)
            densities[i] = np.sum(dists < radius_m)
    
    # Normalize to 0-1
    if densities.max() > 0:
        densities = densities / densities.max()
    
    return densities


def build_connective_tension(
    track_data: TrackData,
    config: Optional[Config] = None,
    n_nodes: int = 100,
    k_neighbors: int = 3,
    tube_radius_min_m: float = 200.0,
    tube_radius_max_m: float = 800.0,
    edge_sag: float = 0.08
) -> Tuple["trimesh.Trimesh", MeshMetadata]:
    """
    Build Option A: Connective Tension sculpture.
    
    Args:
        track_data: Input track data in meters
        config: Configuration (uses defaults if None)
        n_nodes: Target number of nodes
        k_neighbors: k for kNN graph
        tube_radius_min_m: Minimum tube radius in meters
        tube_radius_max_m: Maximum tube radius in meters
        edge_sag: Sag factor for curved edges
        
    Returns:
        Tuple of (mesh, metadata)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for mesh generation")
    
    config = config or Config()
    
    logger.info("=" * 60)
    logger.info("Option A: Connective Tension Structure")
    logger.info("=" * 60)
    
    # Step 1: Extract nodes
    logger.info("\nStep 1: Node extraction")
    nodes = extract_nodes(
        track_data,
        resample_distance_m=config.track_resample_distance_m,
        n_clusters=n_nodes
    )
    
    # Step 2: Build graph
    logger.info("\nStep 2: Graph construction")
    edges = build_graph(nodes, k=k_neighbors)
    edges = ensure_connected(nodes, edges)
    
    # Step 3: Compute densities for tube radius
    logger.info("\nStep 3: Computing densities")
    densities = compute_density_at_nodes(nodes, track_data)
    
    # Step 4: Create tube meshes
    logger.info("\nStep 4: Creating tube meshes")
    tube_meshes = []
    
    for i, j in edges:
        p0, p1 = nodes[i], nodes[j]
        
        # Curved centerline
        centerline = create_curved_edge(p0, p1, sag=edge_sag)
        
        # Radius based on average density of endpoints
        avg_density = (densities[i] + densities[j]) / 2
        radius = tube_radius_min_m + avg_density * (tube_radius_max_m - tube_radius_min_m)
        
        # Create tube
        tube = create_tube_mesh(centerline, radius, segments=8)
        if len(tube.vertices) > 0:
            tube_meshes.append(tube)
    
    logger.info(f"Created {len(tube_meshes)} tube segments")
    
    # Step 5: Merge meshes
    logger.info("\nStep 5: Merging meshes")
    if not tube_meshes:
        raise ValueError("No tube meshes created - cannot build sculpture")
    
    mesh = merge_meshes(tube_meshes)
    
    # Step 6: Smooth
    logger.info("\nStep 6: Smoothing")
    mesh = smooth_mesh(mesh, iterations=config.smoothing_iterations)
    
    # Record bounds before normalization
    stats_before = compute_mesh_stats(mesh)
    max_dim_before = stats_before["max_extent"]
    
    # Step 7: Normalize (if configured)
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
        option="A",
        n_triangles=stats_after["n_faces"],
        n_vertices=stats_after["n_vertices"],
        scale_factor=scale_factor,
        bbox_before_normalization={
            "max_dimension": max_dim_before,
            "bounds": stats_before["bounds"]
        },
        generation_params={
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "k_neighbors": k_neighbors,
            "tube_radius_range_m": [tube_radius_min_m, tube_radius_max_m],
            "edge_sag": edge_sag
        }
    )
    
    logger.info(f"\nResult: {metadata.n_vertices} vertices, {metadata.n_triangles} triangles")
    logger.info(f"Unit mode: {metadata.unit_mode}, max dimension: {metadata.bbox_max_dimension:.2f}")
    
    return mesh, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Option A: Connective Tension Structure")
    print("Run via: python -m src.run_all --modules A")
