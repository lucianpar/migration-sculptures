#!/usr/bin/env python3
"""
Aggressive Flow Field Deformation

Creates VISIBLE sculptural deformation using flow fields.
Based on the specification:
- Field function: drift + vortices
- Deform strength: 0.3-0.5 (30-50% of mesh extent)  
- Falloff: Gaussian from center
- Multiple iterations for cumulative effect
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import trimesh
import logging
from dataclasses import dataclass
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Vortex:
    """2D vortex in XY plane."""
    cx: float  # center x
    cy: float  # center y
    strength: float  # positive=CCW, negative=CW
    radius: float  # influence radius
    
    def velocity(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tangential velocity."""
        dx = x - self.cx
        dy = y - self.cy
        r = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Tangent direction (perpendicular to radial)
        tx, ty = -dy / r, dx / r
        
        # Rankine vortex: linear inside core, 1/r outside
        core_radius = self.radius * 0.3
        
        # Magnitude: peaks at core_radius, decays outside
        mag = np.where(
            r < core_radius,
            self.strength * r / core_radius,  # Linear inside
            self.strength * core_radius / r * np.exp(-(r - core_radius) / self.radius)  # Decay outside
        )
        
        return tx * mag, ty * mag


def flow_field(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytic flow field: base drift + vortices.
    
    Per spec:
    - Base drift: D = (0.6, 0.2, 0.0)
    - Vortex 1: center=(-0.4, 0.1), strength=+0.8
    - Vortex 2: center=(0.3, -0.2), strength=-0.8
    """
    # Base drift (constant flow)
    vx = np.full_like(x, 0.6)
    vy = np.full_like(y, 0.2)
    vz = np.full_like(z, 0.0)
    
    # Vortices
    vortices = [
        Vortex(cx=-0.4, cy=0.1, strength=0.8, radius=0.6),
        Vortex(cx=0.3, cy=-0.2, strength=-0.8, radius=0.6),
    ]
    
    for vortex in vortices:
        dvx, dvy = vortex.velocity(x, y)
        vx += dvx
        vy += dvy
    
    # Add slight z-dependent twist (spiral effect)
    # Rotation around z-axis increases with height
    r_xy = np.sqrt(x**2 + y**2) + 1e-6
    twist_strength = 0.15 * z  # More twist higher up
    vx += -y / r_xy * twist_strength
    vy += x / r_xy * twist_strength
    
    return vx, vy, vz


def falloff(vertices: np.ndarray, center: np.ndarray = None, sigma: float = 0.6) -> np.ndarray:
    """
    Gaussian falloff from scene center.
    
    Per spec:
    - falloff = exp(-(d²)/(2σ²))
    - Also damp by z: falloff *= exp(-k*|z|)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    
    # Distance to center
    d = np.linalg.norm(vertices - center, axis=1)
    
    # Gaussian falloff
    f = np.exp(-d**2 / (2 * sigma**2))
    
    # Z damping: reduce deformation at top/bottom
    z_damp = np.exp(-0.5 * np.abs(vertices[:, 2]))
    f *= z_damp
    
    return f


def normalize_mesh(mesh: trimesh.Trimesh, target_size: float = 1.5) -> trimesh.Trimesh:
    """
    Per spec: each mesh scaled so bounding box max dimension = S
    """
    mesh.vertices -= mesh.centroid
    scale = target_size / mesh.bounding_box.extents.max()
    mesh.vertices *= scale
    return mesh


def deform_mesh(
    mesh: trimesh.Trimesh,
    deform_strength: float = 0.3,
    iterations: int = 3,
    sigma: float = 0.8
) -> trimesh.Trimesh:
    """
    Apply flow field deformation.
    
    Per spec:
    - For each vertex: v = F(p), apply p += v * deformStrength * falloff(p)
    - Recompute normals after
    
    Args:
        mesh: Input mesh (will be modified)
        deform_strength: How far to push vertices (0.3 = 30% of unit)
        iterations: Number of deformation passes
        sigma: Falloff sigma (larger = more uniform deformation)
    """
    vertices = mesh.vertices.copy()
    
    logger.info(f"Deforming {len(vertices)} vertices")
    logger.info(f"  Strength: {deform_strength}, Iterations: {iterations}, Sigma: {sigma}")
    
    for i in range(iterations):
        # Sample flow field
        vx, vy, vz = flow_field(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        velocity = np.column_stack([vx, vy, vz])
        
        # Normalize velocity vectors (we control magnitude via deform_strength)
        vel_mag = np.linalg.norm(velocity, axis=1, keepdims=True)
        vel_mag = np.maximum(vel_mag, 1e-6)
        velocity_normalized = velocity / vel_mag
        
        # Compute falloff weights
        f = falloff(vertices, sigma=sigma)
        
        # Apply displacement
        # Decrease strength each iteration for stability
        iter_strength = deform_strength * (0.7 ** i)
        displacement = velocity_normalized * f[:, np.newaxis] * iter_strength
        
        vertices += displacement
        
        disp_mag = np.linalg.norm(displacement, axis=1)
        logger.info(f"  Iter {i+1}: max displacement = {disp_mag.max():.4f}, mean = {disp_mag.mean():.4f}")
    
    mesh.vertices = vertices
    
    # Recompute normals (critical per spec)
    mesh.fix_normals()
    
    return mesh


def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "output" / "models"
    
    logger.info("=" * 60)
    logger.info("AGGRESSIVE Flow Field Deformation")
    logger.info("=" * 60)
    
    # Load base sculpture
    input_path = output_dir / "whale_sculpture_v2.glb"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return 1
    
    mesh = trimesh.load(str(input_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    logger.info(f"Loaded: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
    logger.info(f"Original extents: {mesh.bounding_box.extents}")
    
    # Normalize
    mesh = normalize_mesh(mesh, target_size=1.5)
    logger.info(f"Normalized extents: {mesh.bounding_box.extents}")
    
    # Test multiple strength levels - MUCH stronger this time
    test_configs = [
        {"strength": 0.15, "iterations": 3, "sigma": 0.8, "name": "mild"},
        {"strength": 0.25, "iterations": 4, "sigma": 0.7, "name": "moderate"},
        {"strength": 0.40, "iterations": 5, "sigma": 0.6, "name": "strong"},
        {"strength": 0.60, "iterations": 6, "sigma": 0.5, "name": "extreme"},
    ]
    
    deformed_dir = output_dir / "deformed_v2"
    deformed_dir.mkdir(parents=True, exist_ok=True)
    
    for config in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Config: {config['name']}")
        logger.info(f"{'='*50}")
        
        # Work on a copy
        test_mesh = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy()
        )
        
        deformed = deform_mesh(
            test_mesh,
            deform_strength=config["strength"],
            iterations=config["iterations"],
            sigma=config["sigma"]
        )
        
        # Report change
        original_verts = mesh.vertices
        new_verts = deformed.vertices
        total_displacement = np.linalg.norm(new_verts - original_verts, axis=1)
        
        logger.info(f"Total displacement: min={total_displacement.min():.4f}, "
                   f"max={total_displacement.max():.4f}, mean={total_displacement.mean():.4f}")
        logger.info(f"As % of mesh size: {total_displacement.max() / 1.5 * 100:.1f}%")
        
        # Save
        out_path = deformed_dir / f"whale_{config['name']}.glb"
        deformed.export(str(out_path))
        logger.info(f"Saved: {out_path}")
        
        # Also OBJ
        obj_path = deformed_dir / f"whale_{config['name']}.obj"
        deformed.export(str(obj_path))
    
    # Also do the ribbon mesh
    logger.info(f"\n{'='*50}")
    logger.info("Processing ribbon mesh")
    logger.info(f"{'='*50}")
    
    ribbon_path = output_dir / "whale_ribbons_v2.glb"
    if ribbon_path.exists():
        ribbon = trimesh.load(str(ribbon_path))
        if isinstance(ribbon, trimesh.Scene):
            ribbon = trimesh.util.concatenate(list(ribbon.geometry.values()))
        
        ribbon = normalize_mesh(ribbon, target_size=1.5)
        ribbon_deformed = deform_mesh(ribbon, deform_strength=0.35, iterations=4, sigma=0.6)
        
        ribbon_out = deformed_dir / "whale_ribbons_strong.glb"
        ribbon_deformed.export(str(ribbon_out))
        logger.info(f"Saved: {ribbon_out}")
    
    logger.info(f"\n{'='*60}")
    logger.info("DONE - Check models in Blender!")
    logger.info(f"{'='*60}")
    logger.info(f"\nFiles in: {deformed_dir}")
    logger.info("  whale_mild.glb      - subtle lean")
    logger.info("  whale_moderate.glb  - noticeable curve")
    logger.info("  whale_strong.glb    - dramatic deformation")
    logger.info("  whale_extreme.glb   - very aggressive")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
