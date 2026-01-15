"""
Mesh Generator Module

High-level API for generating 3D sculpture meshes from processed tracks.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .isosurface import IsosurfaceExtractor, Mesh

logger = logging.getLogger(__name__)

# Try to import trimesh for mesh operations
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available. Install with: pip install trimesh")


@dataclass
class SculptureMetadata:
    """Metadata to be embedded with a sculpture mesh."""
    species: str
    season: str
    year: int
    n_tracks: int
    coherence_score: float = 0.0
    density_entropy: float = 0.0
    centroid_drift_km: float = 0.0
    temporal_variability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "season": self.season,
            "year": self.year,
            "n_tracks": self.n_tracks,
            "metrics": {
                "coherence": self.coherence_score,
                "entropy": self.density_entropy,
                "centroid_drift_km": self.centroid_drift_km,
                "temporal_variability": self.temporal_variability
            }
        }


class MeshGenerator:
    """
    Generates 3D sculpture meshes from bundled whale tracks.
    
    Supports two visualization modes:
    1. Bundle Mode: Trajectory volumes (tubular sculptures)
    2. Terrain Mode: Density heatmap as topographic surface
    """
    
    def __init__(
        self,
        resolution: int = 128,
        smoothing: float = 2.0,
        threshold: float = 0.3,
        max_triangles: int = 100000,
        normalize_size: bool = True,
        normalized_max_dim: float = 2.0
    ):
        """
        Initialize mesh generator.
        
        Args:
            resolution: Voxel grid resolution for isosurface
            smoothing: Gaussian smoothing sigma
            threshold: Isosurface density threshold (0-1)
            max_triangles: Target for mesh simplification
            normalize_size: Whether to normalize mesh to standard size
            normalized_max_dim: Target max dimension when normalizing
        """
        self.resolution = resolution
        self.smoothing = smoothing
        self.threshold = threshold
        self.max_triangles = max_triangles
        self.normalize_size = normalize_size
        self.normalized_max_dim = normalized_max_dim
        
        self.extractor = IsosurfaceExtractor(
            resolution=resolution,
            smoothing_sigma=smoothing
        )
    
    def generate_bundle_sculpture(
        self,
        bundled_tracks: List,
        metadata: Optional[SculptureMetadata] = None,
        z_mode: str = "spread"
    ) -> Tuple[Mesh, Dict[str, Any]]:
        """
        Generate a bundle-mode sculpture from bundled tracks.
        
        Args:
            bundled_tracks: List of BundledTrack objects
            metadata: Optional sculpture metadata
            z_mode: How to handle z-coordinate (flat, time, spread)
            
        Returns:
            Tuple of (Mesh, metadata_dict)
        """
        logger.info(f"Generating bundle sculpture from {len(bundled_tracks)} tracks")
        
        # Extract isosurface
        mesh = self.extractor.process_bundled_tracks(
            bundled_tracks,
            z_mode=z_mode,
            threshold=self.threshold
        )
        
        # Simplify if needed
        if TRIMESH_AVAILABLE and mesh.n_faces > self.max_triangles:
            mesh = self._simplify_mesh(mesh)
        
        # Normalize size
        if self.normalize_size:
            mesh = self._normalize_mesh_size(mesh)
        
        # Ensure normals are computed
        if mesh.normals is None:
            mesh.compute_normals()
        
        # Prepare metadata
        meta_dict = metadata.to_dict() if metadata else {}
        meta_dict["mode"] = "bundle"
        meta_dict["generation_params"] = {
            "resolution": self.resolution,
            "threshold": self.threshold,
            "z_mode": z_mode
        }
        
        return mesh, meta_dict
    
    def generate_terrain_sculpture(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]],
        grid_size: int = 100,
        height_scale: float = 0.2,
        metadata: Optional[SculptureMetadata] = None
    ) -> Tuple[Mesh, Dict[str, Any]]:
        """
        Generate a terrain-mode sculpture (density as height map).
        
        Args:
            tracks: List of (x, y) coordinate arrays
            grid_size: Resolution of the terrain grid
            height_scale: Height scale relative to horizontal extent
            metadata: Optional sculpture metadata
            
        Returns:
            Tuple of (Mesh, metadata_dict)
        """
        logger.info(f"Generating terrain sculpture from {len(tracks)} tracks")
        
        # Collect all points
        all_x = np.concatenate([t[0] for t in tracks])
        all_y = np.concatenate([t[1] for t in tracks])
        
        # Determine bounds
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        # Create 2D density grid
        density = np.zeros((grid_size, grid_size))
        
        for tx, ty in tracks:
            # Convert to grid indices
            ix = ((tx - x_min) / (x_max - x_min) * (grid_size - 1)).astype(int)
            iy = ((ty - y_min) / (y_max - y_min) * (grid_size - 1)).astype(int)
            
            ix = np.clip(ix, 0, grid_size - 1)
            iy = np.clip(iy, 0, grid_size - 1)
            
            for i in range(len(ix)):
                density[ix[i], iy[i]] += 1
        
        # Smooth the density
        try:
            from scipy.ndimage import gaussian_filter
            density = gaussian_filter(density, sigma=2)
        except ImportError:
            pass
        
        # Normalize
        if np.max(density) > 0:
            density = density / np.max(density)
        
        # Create mesh grid
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range)
        height_max = max_range * height_scale
        
        # Generate vertices
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        
        vertices = []
        for i in range(grid_size):
            for j in range(grid_size):
                z = density[i, j] * height_max
                vertices.append([xx[i], yy[j], z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Generate faces (two triangles per grid cell)
        faces = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Vertex indices
                v00 = i * grid_size + j
                v10 = (i + 1) * grid_size + j
                v01 = i * grid_size + (j + 1)
                v11 = (i + 1) * grid_size + (j + 1)
                
                # Two triangles
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
        
        faces = np.array(faces, dtype=np.uint32)
        
        mesh = Mesh(vertices=vertices, faces=faces)
        mesh.compute_normals()
        
        # Normalize if needed
        if self.normalize_size:
            mesh = self._normalize_mesh_size(mesh)
        
        # Metadata
        meta_dict = metadata.to_dict() if metadata else {}
        meta_dict["mode"] = "terrain"
        meta_dict["generation_params"] = {
            "grid_size": grid_size,
            "height_scale": height_scale
        }
        
        return mesh, meta_dict
    
    def _simplify_mesh(self, mesh: Mesh) -> Mesh:
        """Simplify mesh using trimesh if available."""
        if not TRIMESH_AVAILABLE:
            return mesh
        
        # Convert to trimesh
        tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        
        # Target face count
        target = self.max_triangles
        
        if len(tm.faces) > target:
            # Simplify
            tm = tm.simplify_quadric_decimation(target)
            logger.info(f"Simplified mesh from {mesh.n_faces} to {len(tm.faces)} faces")
        
        return Mesh(
            vertices=tm.vertices.astype(np.float32),
            faces=tm.faces.astype(np.uint32)
        )
    
    def _normalize_mesh_size(self, mesh: Mesh) -> Mesh:
        """Normalize mesh to fit within standard bounding box."""
        # Compute current bounds
        min_coords = np.min(mesh.vertices, axis=0)
        max_coords = np.max(mesh.vertices, axis=0)
        
        # Center at origin
        center = (min_coords + max_coords) / 2
        centered = mesh.vertices - center
        
        # Scale to target size
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        
        if max_extent > 0:
            scale = self.normalized_max_dim / max_extent
            normalized = centered * scale
        else:
            normalized = centered
        
        return Mesh(
            vertices=normalized.astype(np.float32),
            faces=mesh.faces,
            normals=mesh.normals,
            vertex_colors=mesh.vertex_colors
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("MeshGenerator module loaded")
    print(f"Trimesh available: {TRIMESH_AVAILABLE}")
