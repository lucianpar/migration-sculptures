"""
Mesh Deformation Module

Applies flow field deformation to sculpture meshes.
Creates coherent "current-shaped" aesthetics across specimens.
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@dataclass
class DeformationConfig:
    """Configuration for mesh deformation."""
    # Deformation strength (how far vertices move)
    strength: float = 0.05
    
    # Falloff parameters
    falloff_sigma: float = 0.8      # Gaussian falloff sigma
    falloff_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Z-based damping
    z_damping: float = 0.5          # Damp deformation at high Z
    z_damping_center: float = 0.0   # Z level with no damping
    
    # Boundary protection
    clamp_displacement: float = 0.2  # Max displacement per vertex
    
    # Normalization
    target_scale: float = 1.5        # Target bounding box size


class MeshDeformer:
    """
    Deforms mesh geometry using a flow field.
    
    The deformation process:
    1. Normalize mesh to consistent scale
    2. For each vertex, sample flow field
    3. Apply displacement with falloff
    4. Recompute normals
    """
    
    def __init__(self, flow_field, config: Optional[DeformationConfig] = None):
        """
        Initialize mesh deformer.
        
        Args:
            flow_field: A FlowField instance
            config: Deformation configuration
        """
        self.flow_field = flow_field
        self.config = config or DeformationConfig()
    
    def normalize_mesh(
        self,
        mesh: "trimesh.Trimesh",
        target_scale: Optional[float] = None
    ) -> Tuple["trimesh.Trimesh", float]:
        """
        Normalize mesh to target scale.
        
        Args:
            mesh: Input trimesh
            target_scale: Target size for longest dimension
            
        Returns:
            Tuple of (normalized_mesh, scale_factor)
        """
        target = target_scale or self.config.target_scale
        
        # Center the mesh
        centroid = mesh.centroid
        mesh.vertices -= centroid
        
        # Scale to target size
        current_scale = mesh.bounding_box.extents.max()
        scale_factor = target / current_scale
        mesh.vertices *= scale_factor
        
        logger.debug(f"Normalized mesh: scale={scale_factor:.4f}, target={target}")
        return mesh, scale_factor
    
    def compute_falloff(
        self,
        vertices: np.ndarray
    ) -> np.ndarray:
        """
        Compute falloff weights for each vertex.
        
        Vertices near the falloff center get full deformation,
        those far away get less.
        
        Args:
            vertices: (N, 3) array of vertex positions
            
        Returns:
            (N,) array of falloff weights in [0, 1]
        """
        center = np.array(self.config.falloff_center)
        
        # Distance from center (XY plane primarily)
        dx = vertices[:, 0] - center[0]
        dy = vertices[:, 1] - center[1]
        dist_xy = np.sqrt(dx**2 + dy**2)
        
        # Gaussian falloff in XY
        sigma = self.config.falloff_sigma
        falloff = np.exp(-0.5 * (dist_xy / sigma)**2)
        
        # Z-based damping (reduce deformation at extremes)
        if self.config.z_damping > 0:
            z_offset = vertices[:, 2] - self.config.z_damping_center
            z_falloff = np.exp(-self.config.z_damping * np.abs(z_offset))
            falloff *= z_falloff
        
        return falloff
    
    def deform(
        self,
        mesh: "trimesh.Trimesh",
        normalize: bool = True,
        iterations: int = 1
    ) -> "trimesh.Trimesh":
        """
        Apply flow field deformation to mesh.
        
        Args:
            mesh: Input trimesh
            normalize: Whether to normalize scale first
            iterations: Number of deformation passes
            
        Returns:
            Deformed mesh (modified in place)
        """
        if not TRIMESH_AVAILABLE:
            raise ImportError("trimesh is required for mesh deformation")
        
        # Step 1: Normalize scale
        if normalize:
            mesh, scale_factor = self.normalize_mesh(mesh)
            logger.info(f"Normalized mesh to scale {self.config.target_scale}")
        
        # Step 2: Get vertex positions
        vertices = mesh.vertices.copy()
        n_verts = len(vertices)
        logger.info(f"Deforming {n_verts} vertices, strength={self.config.strength}")
        
        # Step 3: Iterative deformation
        for iteration in range(iterations):
            # Sample flow field at each vertex
            vx, vy, vz = self.flow_field.sample(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2]
            )
            
            # Stack into displacement vectors
            displacement = np.column_stack([vx, vy, vz])
            
            # Compute falloff weights
            falloff = self.compute_falloff(vertices)
            
            # Apply weighted displacement
            weighted_disp = displacement * falloff[:, np.newaxis] * self.config.strength
            
            # Clamp displacement magnitude
            disp_mag = np.linalg.norm(weighted_disp, axis=1)
            clamp_mask = disp_mag > self.config.clamp_displacement
            if np.any(clamp_mask):
                scale = self.config.clamp_displacement / (disp_mag[clamp_mask] + 1e-10)
                weighted_disp[clamp_mask] *= scale[:, np.newaxis]
            
            # Apply displacement
            vertices += weighted_disp
            
            if iterations > 1:
                logger.debug(f"Iteration {iteration + 1}: max displacement = {disp_mag.max():.4f}")
        
        # Step 4: Write back to mesh
        mesh.vertices = vertices
        
        # Step 5: Recompute normals
        mesh.fix_normals()
        
        # Log statistics
        total_disp = np.linalg.norm(mesh.vertices - mesh.vertices, axis=1)
        logger.info(f"Deformation complete. Max displacement: {disp_mag.max():.4f}")
        
        return mesh
    
    def deform_batch(
        self,
        meshes: list,
        normalize: bool = True
    ) -> list:
        """
        Deform multiple meshes with consistent parameters.
        
        Args:
            meshes: List of trimesh objects
            normalize: Whether to normalize each mesh
            
        Returns:
            List of deformed meshes
        """
        logger.info(f"Batch deforming {len(meshes)} meshes")
        
        deformed = []
        for i, mesh in enumerate(meshes):
            logger.info(f"Processing mesh {i + 1}/{len(meshes)}")
            deformed_mesh = self.deform(mesh.copy(), normalize=normalize)
            deformed.append(deformed_mesh)
        
        return deformed


def deform_sculpture(
    mesh: "trimesh.Trimesh",
    strength: float = 0.05,
    field_type: str = "ocean_current",
    normalize: bool = True
) -> "trimesh.Trimesh":
    """
    High-level function to deform a sculpture mesh.
    
    Args:
        mesh: Input trimesh
        strength: Deformation strength (0.02-0.1 typical)
        field_type: "ocean_current" or "default"
        normalize: Whether to normalize scale
        
    Returns:
        Deformed mesh
    """
    from geometry.flow_field import FlowField, create_ocean_current_field
    
    # Create flow field
    if field_type == "ocean_current":
        flow_field = create_ocean_current_field()
    else:
        flow_field = FlowField.create_default()
    
    # Configure deformation
    config = DeformationConfig(
        strength=strength,
        falloff_sigma=1.0,
        z_damping=0.3,
        target_scale=1.5,
        clamp_displacement=0.15
    )
    
    # Deform
    deformer = MeshDeformer(flow_field, config)
    return deformer.deform(mesh, normalize=normalize)


def load_and_deform_glb(
    input_path: str,
    output_path: str,
    strength: float = 0.05,
    field_type: str = "ocean_current"
) -> None:
    """
    Load a GLB file, deform it, and save.
    
    Args:
        input_path: Path to input GLB
        output_path: Path to output GLB
        strength: Deformation strength
        field_type: Flow field type
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")
    
    logger.info(f"Loading {input_path}")
    mesh = trimesh.load(input_path)
    
    # Handle scene vs single mesh
    if isinstance(mesh, trimesh.Scene):
        # Combine all meshes
        meshes = list(mesh.geometry.values())
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)
    
    # Deform
    deformed = deform_sculpture(mesh, strength=strength, field_type=field_type)
    
    # Save
    deformed.export(output_path)
    logger.info(f"Saved deformed mesh to {output_path}")
