"""
Voxel grid utilities for sculpture generation.

All operations work in meters until final normalization.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
    from skimage.measure import marching_cubes
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy/skimage not available")


@dataclass
class VoxelGrid:
    """
    3D voxel grid for density/occupancy operations.
    
    Coordinates are in meters until final normalization.
    """
    data: np.ndarray  # 3D array of values
    origin: np.ndarray  # (x, y, z) of grid origin in meters
    spacing: float  # Grid spacing in meters
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def bounds_m(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) in meters."""
        min_corner = self.origin
        max_corner = self.origin + np.array(self.shape) * self.spacing
        return min_corner, max_corner
    
    def world_to_grid(self, points_m: np.ndarray) -> np.ndarray:
        """Convert world coordinates (meters) to grid indices."""
        return ((points_m - self.origin) / self.spacing).astype(int)
    
    def grid_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Convert grid indices to world coordinates (meters)."""
        return indices.astype(float) * self.spacing + self.origin
    
    def smooth(self, sigma: float = 1.0) -> "VoxelGrid":
        """Apply Gaussian smoothing."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for smoothing")
        smoothed = gaussian_filter(self.data.astype(float), sigma=sigma)
        return VoxelGrid(data=smoothed, origin=self.origin.copy(), spacing=self.spacing)
    
    def dilate(self, iterations: int = 1) -> "VoxelGrid":
        """Binary dilation."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for dilation")
        dilated = binary_dilation(self.data > 0, iterations=iterations)
        return VoxelGrid(data=dilated.astype(float), origin=self.origin.copy(), spacing=self.spacing)
    
    def erode(self, iterations: int = 1) -> "VoxelGrid":
        """Binary erosion."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for erosion")
        eroded = binary_erosion(self.data > 0, iterations=iterations)
        return VoxelGrid(data=eroded.astype(float), origin=self.origin.copy(), spacing=self.spacing)
    
    def boolean_subtract(self, other: "VoxelGrid") -> "VoxelGrid":
        """
        Boolean subtraction: self AND NOT other.
        
        Useful for carving migration corridors from solid blocks.
        """
        # Resample other to match self if needed
        if self.shape != other.shape:
            logger.warning("Voxel grids have different shapes, results may be unexpected")
        
        result = self.data.copy()
        result[other.data > 0] = 0
        return VoxelGrid(data=result, origin=self.origin.copy(), spacing=self.spacing)
    
    def to_mesh(
        self,
        threshold: float = 0.5,
        step_size: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh using marching cubes.
        
        Returns vertices in METERS (world coordinates).
        
        Args:
            threshold: Isosurface threshold
            step_size: Step size for marching cubes
            
        Returns:
            Tuple of (vertices_m, faces)
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("skimage required for marching cubes")
        
        try:
            verts, faces, _, _ = marching_cubes(
                self.data,
                level=threshold,
                step_size=step_size,
                allow_degenerate=False
            )
        except ValueError as e:
            logger.error(f"Marching cubes failed: {e}")
            return np.empty((0, 3)), np.empty((0, 3), dtype=int)
        
        # Convert to world coordinates (meters)
        verts_m = verts * self.spacing + self.origin
        
        logger.info(f"Extracted mesh: {len(verts_m)} vertices, {len(faces)} faces")
        return verts_m, faces


def create_voxel_grid(
    bounds_m: Tuple[np.ndarray, np.ndarray],
    resolution: int = 128,
    padding: float = 0.1
) -> VoxelGrid:
    """
    Create empty voxel grid covering given bounds.
    
    Args:
        bounds_m: (min_corner, max_corner) in meters
        resolution: Number of voxels along longest axis
        padding: Fractional padding to add (0.1 = 10%)
        
    Returns:
        Empty VoxelGrid
    """
    min_corner, max_corner = bounds_m
    extent = max_corner - min_corner
    max_extent = np.max(extent)
    
    # Add padding
    pad_amount = max_extent * padding
    min_corner = min_corner - pad_amount
    max_corner = max_corner + pad_amount
    extent = max_corner - min_corner
    
    # Compute spacing
    spacing = np.max(extent) / resolution
    
    # Compute grid shape
    shape = np.ceil(extent / spacing).astype(int)
    shape = np.maximum(shape, 1)
    
    logger.info(f"Creating voxel grid: {tuple(shape)}, spacing={spacing:.2f}m")
    
    return VoxelGrid(
        data=np.zeros(tuple(shape), dtype=np.float32),
        origin=min_corner.copy(),
        spacing=spacing
    )


def rasterize_tracks(
    points_m: np.ndarray,
    grid: VoxelGrid,
    radius_m: float = 500.0,
    falloff: str = "gaussian"
) -> VoxelGrid:
    """
    Rasterize track points into voxel grid as density field.
    
    Args:
        points_m: Nx3 array of points in meters
        grid: VoxelGrid to rasterize into
        radius_m: Influence radius in meters
        falloff: "gaussian" or "linear"
        
    Returns:
        VoxelGrid with density values
    """
    result = grid.data.copy()
    
    # Convert radius to grid units
    radius_voxels = int(np.ceil(radius_m / grid.spacing))
    sigma_voxels = radius_voxels / 3.0  # For Gaussian falloff
    
    # Convert points to grid coordinates
    grid_coords = grid.world_to_grid(points_m)
    
    # Clamp to grid bounds
    shape = np.array(grid.shape)
    valid_mask = np.all((grid_coords >= 0) & (grid_coords < shape), axis=1)
    grid_coords = grid_coords[valid_mask]
    
    logger.info(f"Rasterizing {len(grid_coords)} points (radius={radius_m}m = {radius_voxels} voxels)")
    
    # Paint each point
    for coord in grid_coords:
        # Define affected region
        x_min = max(0, coord[0] - radius_voxels)
        x_max = min(shape[0], coord[0] + radius_voxels + 1)
        y_min = max(0, coord[1] - radius_voxels)
        y_max = min(shape[1], coord[1] + radius_voxels + 1)
        z_min = max(0, coord[2] - radius_voxels)
        z_max = min(shape[2], coord[2] + radius_voxels + 1)
        
        # Create coordinate grids
        xx, yy, zz = np.mgrid[x_min:x_max, y_min:y_max, z_min:z_max]
        
        # Compute distance
        dist = np.sqrt(
            (xx - coord[0])**2 +
            (yy - coord[1])**2 +
            (zz - coord[2])**2
        )
        
        # Compute falloff
        if falloff == "gaussian":
            contribution = np.exp(-dist**2 / (2 * sigma_voxels**2))
        else:  # linear
            contribution = np.maximum(0, 1 - dist / radius_voxels)
        
        # Accumulate
        result[x_min:x_max, y_min:y_max, z_min:z_max] += contribution
    
    # Normalize to 0-1
    if result.max() > 0:
        result = result / result.max()
    
    return VoxelGrid(data=result, origin=grid.origin.copy(), spacing=grid.spacing)


def create_solid_block(bounds_m: Tuple[np.ndarray, np.ndarray], resolution: int = 128) -> VoxelGrid:
    """
    Create solid voxel block covering given bounds.
    
    Used as starting point for subtractive sculpting.
    
    Args:
        bounds_m: (min_corner, max_corner) in meters
        resolution: Voxel resolution
        
    Returns:
        VoxelGrid filled with 1.0
    """
    grid = create_voxel_grid(bounds_m, resolution, padding=0.1)
    grid.data[:] = 1.0
    return grid
