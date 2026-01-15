"""
Flow Field Module

Implements vector field functions for deforming sculpture geometry.
Creates coherent "current-like" deformations across multiple sculptures.

Two modes:
- Mode A: Analytic field (drift + vortices) - procedural
- Mode B: Sampled grid field (from preprocessed data)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Vortex:
    """A 2D vortex in the XY plane."""
    center_x: float
    center_y: float
    strength: float  # Positive = CCW, Negative = CW
    radius: float    # Falloff radius
    
    def velocity_at(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tangential velocity from this vortex.
        
        Returns (vx, vy) velocity components.
        """
        # Distance from vortex center
        dx = x - self.center_x
        dy = y - self.center_y
        r = np.sqrt(dx**2 + dy**2) + 1e-10
        
        # Tangential direction (perpendicular to radial)
        tx = -dy / r
        ty = dx / r
        
        # Magnitude with smooth falloff (Gaussian-like)
        # Peak at r=radius/2, falls off beyond
        magnitude = self.strength * np.exp(-0.5 * (r / self.radius)**2)
        
        return tx * magnitude, ty * magnitude


@dataclass
class FlowFieldConfig:
    """Configuration for an analytic flow field."""
    # Base drift (constant flow direction)
    drift: Tuple[float, float, float] = (0.6, 0.2, 0.0)
    
    # Vortices
    vortices: List[Vortex] = field(default_factory=list)
    
    # Z-axis influence
    z_drift_factor: float = 0.1  # How much Z affects drift
    z_spiral: float = 0.0        # Spiral around Z axis
    
    # Overall scale
    field_scale: float = 1.0


class FlowField:
    """
    Vector field for sculpture deformation.
    
    Computes velocity-like vectors at any point in 3D space.
    Used to deform geometry vertices to create coherent flow effects.
    """
    
    def __init__(self, mode: str = "analytic"):
        """
        Initialize flow field.
        
        Args:
            mode: "analytic" for procedural field, "grid" for sampled data
        """
        self.mode = mode
        self.config: Optional[FlowFieldConfig] = None
        self.grid_data: Optional[Dict[str, Any]] = None
        
    @classmethod
    def create_analytic(
        cls,
        drift: Tuple[float, float, float] = (0.6, 0.2, 0.0),
        vortices: Optional[List[Dict]] = None,
        z_spiral: float = 0.0
    ) -> "FlowField":
        """
        Create an analytic flow field with drift and vortices.
        
        Args:
            drift: Base drift vector (constant flow)
            vortices: List of vortex configs [{"center": (x,y), "strength": s, "radius": r}, ...]
            z_spiral: Amount of spiral around Z axis
            
        Returns:
            Configured FlowField instance
        """
        field = cls(mode="analytic")
        
        # Convert vortex dicts to objects
        vortex_objects = []
        if vortices:
            for v in vortices:
                center = v.get("center", (0, 0))
                vortex_objects.append(Vortex(
                    center_x=center[0],
                    center_y=center[1],
                    strength=v.get("strength", 0.5),
                    radius=v.get("radius", 0.5)
                ))
        
        field.config = FlowFieldConfig(
            drift=drift,
            vortices=vortex_objects,
            z_spiral=z_spiral
        )
        
        logger.info(f"Created analytic flow field: drift={drift}, {len(vortex_objects)} vortices")
        return field
    
    @classmethod
    def create_default(cls) -> "FlowField":
        """Create a default analytic field with standard parameters."""
        return cls.create_analytic(
            drift=(0.6, 0.2, 0.0),
            vortices=[
                {"center": (-0.4, 0.1), "strength": 0.8, "radius": 0.6},
                {"center": (0.3, -0.2), "strength": -0.8, "radius": 0.6},
            ],
            z_spiral=0.1
        )
    
    @classmethod
    def from_grid_file(cls, json_path: str) -> "FlowField":
        """
        Load a sampled flow field from a JSON grid file.
        
        Expected JSON format:
        {
            "resolution": [nx, ny, nz],
            "bounds": {"min": [x,y,z], "max": [x,y,z]},
            "vectors": [[[vx,vy,vz], ...], ...]  # 3D array
        }
        """
        field = cls(mode="grid")
        
        with open(json_path, 'r') as f:
            field.grid_data = json.load(f)
        
        res = field.grid_data.get("resolution", [64, 64, 64])
        logger.info(f"Loaded grid flow field: resolution={res}")
        return field
    
    def sample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample the flow field at given points.
        
        Args:
            x, y, z: Coordinate arrays (can be scalars or arrays)
            
        Returns:
            Tuple of (vx, vy, vz) velocity components
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        
        if self.mode == "analytic":
            return self._sample_analytic(x, y, z)
        else:
            return self._sample_grid(x, y, z)
    
    def _sample_analytic(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the analytic field."""
        if self.config is None:
            raise ValueError("Analytic field not configured")
        
        # Start with base drift
        vx = np.full_like(x, self.config.drift[0], dtype=float)
        vy = np.full_like(y, self.config.drift[1], dtype=float)
        vz = np.full_like(z, self.config.drift[2], dtype=float)
        
        # Add vortex contributions
        for vortex in self.config.vortices:
            dvx, dvy = vortex.velocity_at(x, y)
            vx += dvx
            vy += dvy
        
        # Add Z-spiral effect (rotation around vertical axis based on height)
        if self.config.z_spiral != 0:
            # Distance from Z axis
            r_xy = np.sqrt(x**2 + y**2) + 1e-10
            # Tangential direction
            tx = -y / r_xy
            ty = x / r_xy
            # Spiral strength increases with height
            spiral_strength = self.config.z_spiral * z
            vx += tx * spiral_strength
            vy += ty * spiral_strength
        
        # Scale
        vx *= self.config.field_scale
        vy *= self.config.field_scale
        vz *= self.config.field_scale
        
        return vx, vy, vz
    
    def _sample_grid(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the grid-based field with trilinear interpolation."""
        if self.grid_data is None:
            raise ValueError("Grid data not loaded")
        
        bounds = self.grid_data["bounds"]
        res = self.grid_data["resolution"]
        vectors = np.array(self.grid_data["vectors"])
        
        # Normalize coordinates to grid indices
        min_bound = np.array(bounds["min"])
        max_bound = np.array(bounds["max"])
        extent = max_bound - min_bound + 1e-10
        
        # Convert to grid coordinates
        gx = (x - min_bound[0]) / extent[0] * (res[0] - 1)
        gy = (y - min_bound[1]) / extent[1] * (res[1] - 1)
        gz = (z - min_bound[2]) / extent[2] * (res[2] - 1)
        
        # Clamp to valid range
        gx = np.clip(gx, 0, res[0] - 1.001)
        gy = np.clip(gy, 0, res[1] - 1.001)
        gz = np.clip(gz, 0, res[2] - 1.001)
        
        # Trilinear interpolation
        ix = gx.astype(int)
        iy = gy.astype(int)
        iz = gz.astype(int)
        
        fx = gx - ix
        fy = gy - iy
        fz = gz - iz
        
        # Sample 8 corners and interpolate
        vx = np.zeros_like(x, dtype=float)
        vy = np.zeros_like(y, dtype=float)
        vz = np.zeros_like(z, dtype=float)
        
        for di in [0, 1]:
            for dj in [0, 1]:
                for dk in [0, 1]:
                    # Weight for this corner
                    wx = (1 - fx) if di == 0 else fx
                    wy = (1 - fy) if dj == 0 else fy
                    wz = (1 - fz) if dk == 0 else fz
                    w = wx * wy * wz
                    
                    # Sample indices (clamped)
                    si = np.clip(ix + di, 0, res[0] - 1)
                    sj = np.clip(iy + dj, 0, res[1] - 1)
                    sk = np.clip(iz + dk, 0, res[2] - 1)
                    
                    # Accumulate weighted velocity
                    for idx in range(len(x)):
                        v = vectors[si[idx], sj[idx], sk[idx]]
                        vx[idx] += w[idx] * v[0]
                        vy[idx] += w[idx] * v[1]
                        vz[idx] += w[idx] * v[2]
        
        return vx, vy, vz
    
    def to_json(self, path: str, resolution: int = 64, bounds: Tuple = (-1, 1)):
        """
        Export the analytic field as a sampled grid JSON for JavaScript.
        
        Args:
            path: Output JSON file path
            resolution: Grid resolution per axis
            bounds: (min, max) for each axis
        """
        if self.mode != "analytic":
            raise ValueError("Can only export analytic fields to grid")
        
        min_b, max_b = bounds
        
        # Create sample grid
        x = np.linspace(min_b, max_b, resolution)
        y = np.linspace(min_b, max_b, resolution)
        z = np.linspace(min_b, max_b, resolution)
        
        vectors = np.zeros((resolution, resolution, resolution, 3))
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    vx, vy, vz = self.sample(
                        np.array([xi]),
                        np.array([yj]),
                        np.array([zk])
                    )
                    vectors[i, j, k] = [vx[0], vy[0], vz[0]]
        
        grid_data = {
            "resolution": [resolution, resolution, resolution],
            "bounds": {
                "min": [min_b, min_b, min_b],
                "max": [max_b, max_b, max_b]
            },
            "vectors": vectors.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(grid_data, f)
        
        logger.info(f"Exported flow field grid to {path}")


def create_ocean_current_field() -> FlowField:
    """
    Create a flow field inspired by ocean currents.
    
    Features:
    - Dominant northward drift (California Current-like)
    - Two vortices representing eddies
    - Slight upwelling effect (Z component)
    """
    return FlowField.create_analytic(
        drift=(0.4, 0.6, 0.05),  # North-northeast with slight upward
        vortices=[
            # Warm-core eddy (CW in Northern Hemisphere)
            {"center": (-0.3, 0.2), "strength": -0.7, "radius": 0.5},
            # Cold-core eddy (CCW)
            {"center": (0.4, -0.1), "strength": 0.6, "radius": 0.4},
            # Small nearshore eddy
            {"center": (-0.5, -0.3), "strength": 0.3, "radius": 0.25},
        ],
        z_spiral=0.08
    )
