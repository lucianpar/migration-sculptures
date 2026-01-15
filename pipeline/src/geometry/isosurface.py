"""
Isosurface Extraction Module

Uses marching cubes algorithm to extract 3D mesh surfaces from
voxel density grids representing bundled whale tracks.
"""

import logging
from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import scikit-image for marching cubes
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")


@dataclass
class Mesh:
    """A 3D triangular mesh."""
    vertices: np.ndarray  # (N, 3) array of vertex positions
    faces: np.ndarray     # (M, 3) array of triangle indices
    normals: Optional[np.ndarray] = None  # (N, 3) vertex normals
    vertex_colors: Optional[np.ndarray] = None  # (N, 3) or (N, 4) vertex colors
    
    @property
    def n_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def n_faces(self) -> int:
        return len(self.faces)
    
    def compute_normals(self) -> None:
        """Compute vertex normals from face normals."""
        # Initialize vertex normals to zero
        self.normals = np.zeros_like(self.vertices)
        
        # Compute face normals and accumulate to vertices
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            
            # Face normal (cross product of two edges)
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Accumulate to vertices
            for idx in face:
                self.normals[idx] += face_normal
        
        # Normalize
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        self.normals = self.normals / norms


class IsosurfaceExtractor:
    """
    Extracts isosurfaces from 3D density volumes using marching cubes.
    
    The workflow:
    1. Create a 3D voxel grid
    2. "Paint" bundled trajectories into the grid (accumulate density)
    3. Apply Gaussian smoothing for organic shapes
    4. Extract isosurface at a threshold density
    """
    
    def __init__(
        self,
        resolution: int = 128,
        padding: float = 0.1,
        line_radius: float = 3.0,  # Voxels
        smoothing_sigma: float = 2.0
    ):
        """
        Initialize the extractor.
        
        Args:
            resolution: Grid resolution (e.g., 128 means 128³ voxels)
            padding: Padding around data bounds as fraction
            line_radius: Radius of trajectory lines in voxels
            smoothing_sigma: Gaussian smoothing sigma in voxels
        """
        self.resolution = resolution
        self.padding = padding
        self.line_radius = line_radius
        self.smoothing_sigma = smoothing_sigma
    
    def create_density_grid(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        bounds: Optional[Tuple[float, float, float, float, float, float]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Create a 3D density grid from trajectory data.
        
        Args:
            tracks: List of (x, y, z) coordinate arrays for each track.
                    z can be None for 2D tracks (will use 0).
            bounds: Optional (min_x, max_x, min_y, max_y, min_z, max_z)
            
        Returns:
            Tuple of (density_grid, grid_info) where grid_info contains
            transformation parameters
        """
        # Collect all points to determine bounds
        all_x = np.concatenate([t[0] for t in tracks])
        all_y = np.concatenate([t[1] for t in tracks])
        all_z = np.concatenate([
            t[2] if t[2] is not None else np.zeros_like(t[0])
            for t in tracks
        ])
        
        if bounds:
            min_x, max_x, min_y, max_y, min_z, max_z = bounds
        else:
            min_x, max_x = np.min(all_x), np.max(all_x)
            min_y, max_y = np.min(all_y), np.max(all_y)
            min_z, max_z = np.min(all_z), np.max(all_z)
        
        # Add padding
        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max(max_z - min_z, x_range * 0.1)  # Ensure some z range
        
        min_x -= x_range * self.padding
        max_x += x_range * self.padding
        min_y -= y_range * self.padding
        max_y += y_range * self.padding
        min_z -= z_range * self.padding
        max_z += z_range * self.padding
        
        # Create grid
        grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=np.float32)
        
        # Grid transformation info
        grid_info = {
            "min_x": min_x, "max_x": max_x,
            "min_y": min_y, "max_y": max_y,
            "min_z": min_z, "max_z": max_z,
            "scale_x": (max_x - min_x) / self.resolution,
            "scale_y": (max_y - min_y) / self.resolution,
            "scale_z": (max_z - min_z) / self.resolution,
            "resolution": self.resolution
        }
        
        logger.info(f"Creating {self.resolution}³ density grid")
        logger.debug(f"Grid bounds: x=[{min_x:.1f}, {max_x:.1f}], "
                    f"y=[{min_y:.1f}, {max_y:.1f}], z=[{min_z:.1f}, {max_z:.1f}]")
        
        # Paint each track into the grid
        for track_idx, (tx, ty, tz) in enumerate(tracks):
            if tz is None:
                tz = np.zeros_like(tx)
            
            self._paint_trajectory(grid, tx, ty, tz, grid_info)
        
        logger.info(f"Painted {len(tracks)} tracks into grid")
        
        return grid, grid_info
    
    def _paint_trajectory(
        self,
        grid: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        grid_info: dict
    ) -> None:
        """Paint a single trajectory as a thick line in the grid."""
        res = self.resolution
        
        # Convert coordinates to grid indices
        ix = ((x - grid_info["min_x"]) / (grid_info["max_x"] - grid_info["min_x"]) * res).astype(int)
        iy = ((y - grid_info["min_y"]) / (grid_info["max_y"] - grid_info["min_y"]) * res).astype(int)
        iz = ((z - grid_info["min_z"]) / (grid_info["max_z"] - grid_info["min_z"]) * res).astype(int)
        
        # Clamp to grid bounds
        ix = np.clip(ix, 0, res - 1)
        iy = np.clip(iy, 0, res - 1)
        iz = np.clip(iz, 0, res - 1)
        
        # Paint points with a spherical brush
        radius = int(self.line_radius)
        
        for i in range(len(ix)):
            # Get brush region bounds
            x0 = max(0, ix[i] - radius)
            x1 = min(res, ix[i] + radius + 1)
            y0 = max(0, iy[i] - radius)
            y1 = min(res, iy[i] + radius + 1)
            z0 = max(0, iz[i] - radius)
            z1 = min(res, iz[i] + radius + 1)
            
            # Add density in spherical pattern
            for gx in range(x0, x1):
                for gy in range(y0, y1):
                    for gz in range(z0, z1):
                        dist = np.sqrt(
                            (gx - ix[i])**2 +
                            (gy - iy[i])**2 +
                            (gz - iz[i])**2
                        )
                        if dist <= radius:
                            # Falloff from center
                            value = 1.0 - (dist / radius) ** 2
                            grid[gx, gy, gz] += value
    
    def smooth_grid(self, grid: np.ndarray) -> np.ndarray:
        """Apply Gaussian smoothing to the density grid."""
        try:
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(grid, sigma=self.smoothing_sigma)
            logger.info(f"Applied Gaussian smoothing (sigma={self.smoothing_sigma})")
            return smoothed
        except ImportError:
            logger.warning("scipy not available, skipping smoothing")
            return grid
    
    def extract_surface(
        self,
        grid: np.ndarray,
        grid_info: dict,
        threshold: Optional[float] = None
    ) -> Mesh:
        """
        Extract isosurface mesh from density grid.
        
        Args:
            grid: 3D density array
            grid_info: Grid transformation info
            threshold: Density threshold (0-1 relative to max). If None, auto-compute.
            
        Returns:
            Mesh object with vertices and faces
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for isosurface extraction")
        
        # Normalize grid to 0-1
        grid_max = np.max(grid)
        if grid_max > 0:
            grid_normalized = grid / grid_max
        else:
            raise ValueError("Empty density grid - no data to extract")
        
        # Auto-compute threshold if not provided
        if threshold is None:
            # Use Otsu-like approach: find threshold that separates foreground/background
            threshold = 0.3  # Default
            logger.info(f"Using density threshold: {threshold}")
        
        # Extract isosurface using marching cubes
        verts, faces, normals, values = marching_cubes(
            grid_normalized,
            level=threshold,
            spacing=(1, 1, 1),
            allow_degenerate=False
        )
        
        logger.info(f"Extracted surface with {len(verts)} vertices, {len(faces)} faces")
        
        # Transform vertices from grid coordinates to world coordinates
        verts_world = np.zeros_like(verts)
        verts_world[:, 0] = (verts[:, 0] / self.resolution) * (grid_info["max_x"] - grid_info["min_x"]) + grid_info["min_x"]
        verts_world[:, 1] = (verts[:, 1] / self.resolution) * (grid_info["max_y"] - grid_info["min_y"]) + grid_info["min_y"]
        verts_world[:, 2] = (verts[:, 2] / self.resolution) * (grid_info["max_z"] - grid_info["min_z"]) + grid_info["min_z"]
        
        mesh = Mesh(
            vertices=verts_world.astype(np.float32),
            faces=faces.astype(np.uint32),
            normals=normals.astype(np.float32)
        )
        
        return mesh
    
    def process_bundled_tracks(
        self,
        bundled_tracks: List,
        z_mode: str = "flat",
        threshold: float = 0.3
    ) -> Mesh:
        """
        Full pipeline: bundled tracks -> density grid -> mesh.
        
        Args:
            bundled_tracks: List of BundledTrack objects
            z_mode: How to handle z-coordinate:
                    - "flat": All tracks at z=0 (simple 2D -> 3D)
                    - "time": Use time/progress as z (shows temporal aspect)
                    - "spread": Spread tracks vertically for visibility
            threshold: Isosurface density threshold
            
        Returns:
            Mesh object
        """
        # Convert bundled tracks to coordinate lists
        tracks = []
        
        for i, bt in enumerate(bundled_tracks):
            x = bt.x
            y = bt.y
            
            n_points = len(x)
            
            if z_mode == "flat":
                z = np.zeros(n_points)
            elif z_mode == "time":
                # Use progress along track as z
                z = np.linspace(0, 1, n_points) * (np.max(x) - np.min(x)) * 0.2
            elif z_mode == "spread":
                # Spread tracks vertically
                z = np.ones(n_points) * (i / len(bundled_tracks) - 0.5) * (np.max(x) - np.min(x)) * 0.1
            else:
                z = np.zeros(n_points)
            
            tracks.append((x, y, z))
        
        # Create density grid
        grid, grid_info = self.create_density_grid(tracks)
        
        # Smooth
        grid = self.smooth_grid(grid)
        
        # Extract surface
        mesh = self.extract_surface(grid, grid_info, threshold)
        
        return mesh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    np.random.seed(42)
    
    # Create some synthetic tracks
    tracks = []
    for i in range(5):
        t = np.linspace(0, 1, 30)
        x = t * 10000 + np.random.randn(30) * 500
        y = np.sin(t * np.pi) * 3000 + np.random.randn(30) * 500 + i * 200
        z = np.zeros(30)
        tracks.append((x, y, z))
    
    extractor = IsosurfaceExtractor(resolution=64, line_radius=2)
    grid, info = extractor.create_density_grid(tracks)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Grid max density: {np.max(grid):.2f}")
    
    if SKIMAGE_AVAILABLE:
        grid = extractor.smooth_grid(grid)
        mesh = extractor.extract_surface(grid, info, threshold=0.3)
        print(f"Mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
