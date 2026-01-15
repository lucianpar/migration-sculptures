"""
Sculptural Transformation Module

Transforms 2D whale tracks into 3D sculptural forms using:
- Temporal z-axis mapping (time becomes height)
- Ribbon extrusion perpendicular to track direction
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RibbonTrack:
    """A track transformed into a 3D ribbon."""
    track_id: str
    # Center line of ribbon
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray  # Time-based height
    # Ribbon edges (for mesh generation)
    left_x: np.ndarray
    left_y: np.ndarray
    left_z: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    right_z: np.ndarray
    # Ribbon width at each point
    widths: np.ndarray


class SculpturalTransform:
    """
    Transforms 2D whale migration tracks into 3D sculptural forms.
    
    The transformation:
    1. Maps time to z-axis (temporal flow)
    2. Extrudes tracks into ribbons perpendicular to movement direction
    3. Varies ribbon width based on speed/density
    """
    
    def __init__(
        self,
        z_scale: float = 1.0,
        base_ribbon_width: float = 500.0,  # meters
        width_variation: float = 0.3,  # fraction
        smooth_normals: bool = True
    ):
        """
        Initialize sculptural transform.
        
        Args:
            z_scale: Scale factor for z-axis (time dimension)
            base_ribbon_width: Base width of ribbon extrusion in meters
            width_variation: How much width varies (0-1)
            smooth_normals: Whether to smooth the perpendicular directions
        """
        self.z_scale = z_scale
        self.base_ribbon_width = base_ribbon_width
        self.width_variation = width_variation
        self.smooth_normals = smooth_normals
    
    def transform_tracks(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]],
        timestamps: Optional[List[np.ndarray]] = None,
        track_ids: Optional[List[str]] = None
    ) -> List[RibbonTrack]:
        """
        Transform 2D tracks into 3D ribbon sculptures.
        
        Args:
            tracks: List of (x, y) coordinate arrays
            timestamps: Optional list of timestamp arrays for each track
            track_ids: Optional list of track identifiers
            
        Returns:
            List of RibbonTrack objects
        """
        ribbons = []
        
        # Compute global time range for consistent z-mapping
        if timestamps:
            all_times = np.concatenate(timestamps)
            t_min, t_max = np.min(all_times), np.max(all_times)
        else:
            t_min, t_max = 0, 1
        
        # Compute global spatial extent for z-scaling
        all_x = np.concatenate([t[0] for t in tracks])
        all_y = np.concatenate([t[1] for t in tracks])
        spatial_extent = max(np.ptp(all_x), np.ptp(all_y))
        
        # Z range should be proportional to spatial extent for sculptural form
        z_range = spatial_extent * self.z_scale
        
        logger.info(f"Spatial extent: {spatial_extent:.1f}m, Z range: {z_range:.1f}m")
        
        for i, (x, y) in enumerate(tracks):
            track_id = track_ids[i] if track_ids else f"track_{i}"
            
            # Get timestamps or generate sequential
            if timestamps and i < len(timestamps):
                t = timestamps[i].astype(float)
                # Normalize to 0-1
                t_norm = (t - t_min) / (t_max - t_min + 1e-10)
            else:
                # Use position along track as proxy for time
                t_norm = np.linspace(0, 1, len(x))
            
            # Map time to z-axis
            z = t_norm * z_range
            
            # Compute ribbon extrusion
            ribbon = self._extrude_ribbon(x, y, z, track_id)
            ribbons.append(ribbon)
        
        logger.info(f"Transformed {len(ribbons)} tracks into ribbons")
        return ribbons
    
    def _extrude_ribbon(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        track_id: str
    ) -> RibbonTrack:
        """
        Extrude a single track into a ribbon perpendicular to movement direction.
        """
        n = len(x)
        
        # Compute tangent vectors (direction of movement)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        
        # Normalize tangents
        tangent_length = np.sqrt(dx**2 + dy**2 + dz**2)
        tangent_length = np.maximum(tangent_length, 1e-10)
        
        tx = dx / tangent_length
        ty = dy / tangent_length
        tz = dz / tangent_length
        
        # Compute perpendicular vectors in the XY plane
        # Perpendicular to tangent, staying mostly horizontal
        # Use cross product with up vector (0, 0, 1)
        perp_x = -ty  # Cross product with (0,0,1)
        perp_y = tx
        perp_z = np.zeros_like(x)
        
        # Normalize perpendicular vectors
        perp_length = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
        perp_length = np.maximum(perp_length, 1e-10)
        
        perp_x = perp_x / perp_length
        perp_y = perp_y / perp_length
        
        # Optionally smooth the perpendicular directions
        if self.smooth_normals and n > 5:
            from scipy.ndimage import gaussian_filter1d
            perp_x = gaussian_filter1d(perp_x, sigma=2)
            perp_y = gaussian_filter1d(perp_y, sigma=2)
            # Re-normalize after smoothing
            perp_length = np.sqrt(perp_x**2 + perp_y**2)
            perp_length = np.maximum(perp_length, 1e-10)
            perp_x = perp_x / perp_length
            perp_y = perp_y / perp_length
        
        # Compute varying ribbon width based on speed
        # Slower movement = wider ribbon (more time spent there)
        speed = tangent_length
        speed_norm = speed / (np.mean(speed) + 1e-10)
        # Invert: slow = wide, fast = narrow
        width_factor = 1.0 / (speed_norm + 0.5)
        width_factor = width_factor / np.mean(width_factor)  # Normalize
        
        # Apply width variation
        widths = self.base_ribbon_width * (
            1.0 + self.width_variation * (width_factor - 1.0)
        )
        
        # Compute left and right edges
        half_width = widths / 2
        
        left_x = x + perp_x * half_width
        left_y = y + perp_y * half_width
        left_z = z.copy()
        
        right_x = x - perp_x * half_width
        right_y = y - perp_y * half_width
        right_z = z.copy()
        
        return RibbonTrack(
            track_id=track_id,
            x=x, y=y, z=z,
            left_x=left_x, left_y=left_y, left_z=left_z,
            right_x=right_x, right_y=right_y, right_z=right_z,
            widths=widths
        )
    
    def ribbons_to_mesh_points(
        self,
        ribbons: List[RibbonTrack],
        add_thickness: bool = True,
        thickness: float = 100.0
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Convert ribbons to point clouds for voxelization.
        
        Args:
            ribbons: List of RibbonTrack objects
            add_thickness: Whether to add vertical thickness to ribbons
            thickness: Vertical thickness in meters
            
        Returns:
            List of (x, y, z) point arrays for each ribbon
        """
        tracks_xyz = []
        
        for ribbon in ribbons:
            # Collect points from center and edges
            points_x = []
            points_y = []
            points_z = []
            
            n = len(ribbon.x)
            
            # Sample across ribbon width
            for t in np.linspace(0, 1, 5):  # 5 samples across width
                px = ribbon.left_x * (1 - t) + ribbon.right_x * t
                py = ribbon.left_y * (1 - t) + ribbon.right_y * t
                pz = ribbon.z.copy()
                
                points_x.append(px)
                points_y.append(py)
                points_z.append(pz)
                
                # Add thickness layers
                if add_thickness:
                    for z_off in np.linspace(-thickness/2, thickness/2, 3):
                        points_x.append(px)
                        points_y.append(py)
                        points_z.append(pz + z_off)
            
            # Combine all points
            all_x = np.concatenate(points_x)
            all_y = np.concatenate(points_y)
            all_z = np.concatenate(points_z)
            
            tracks_xyz.append((all_x, all_y, all_z))
        
        return tracks_xyz
    
    def ribbons_to_direct_mesh(
        self,
        ribbons: List[RibbonTrack]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ribbons directly to a triangle mesh (faster than voxelization).
        
        Returns:
            Tuple of (vertices, faces) arrays
        """
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for ribbon in ribbons:
            n = len(ribbon.x)
            
            # Create vertices for left and right edges
            left_verts = np.column_stack([ribbon.left_x, ribbon.left_y, ribbon.left_z])
            right_verts = np.column_stack([ribbon.right_x, ribbon.right_y, ribbon.right_z])
            
            # Interleave: left0, right0, left1, right1, ...
            vertices = np.empty((n * 2, 3))
            vertices[0::2] = left_verts
            vertices[1::2] = right_verts
            
            # Create triangle strip faces
            faces = []
            for i in range(n - 1):
                # Two triangles per quad
                v0 = vertex_offset + i * 2
                v1 = vertex_offset + i * 2 + 1
                v2 = vertex_offset + (i + 1) * 2
                v3 = vertex_offset + (i + 1) * 2 + 1
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
            
            all_vertices.append(vertices)
            all_faces.extend(faces)
            vertex_offset += len(vertices)
        
        vertices = np.vstack(all_vertices)
        faces = np.array(all_faces)
        
        logger.info(f"Created ribbon mesh: {len(vertices)} vertices, {len(faces)} faces")
        return vertices, faces


def create_sculptural_mesh(
    tracks: List[Tuple[np.ndarray, np.ndarray]],
    timestamps: Optional[List[np.ndarray]] = None,
    track_ids: Optional[List[str]] = None,
    z_scale: float = 0.8,
    ribbon_width: float = 800.0,
    use_voxels: bool = True,
    voxel_resolution: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-level function to create a sculptural mesh from whale tracks.
    
    Args:
        tracks: List of (x, y) coordinate arrays
        timestamps: Optional timestamp arrays
        track_ids: Optional track identifiers
        z_scale: Scale factor for temporal z-axis
        ribbon_width: Width of ribbon extrusion
        use_voxels: Whether to use voxelization (smoother) or direct mesh (faster)
        voxel_resolution: Resolution if using voxels
        
    Returns:
        Tuple of (vertices, faces) arrays
    """
    # Create sculptural transform
    transform = SculpturalTransform(
        z_scale=z_scale,
        base_ribbon_width=ribbon_width,
        width_variation=0.4
    )
    
    # Transform tracks to ribbons
    ribbons = transform.transform_tracks(tracks, timestamps, track_ids)
    
    if use_voxels:
        # Use voxelization for smoother organic forms
        from geometry.isosurface import IsosurfaceExtractor
        
        # Convert ribbons to dense point clouds
        tracks_xyz = transform.ribbons_to_mesh_points(
            ribbons, 
            add_thickness=True,
            thickness=ribbon_width * 0.3
        )
        
        # Create isosurface
        extractor = IsosurfaceExtractor(
            resolution=voxel_resolution,
            line_radius=4.0,
            smoothing_sigma=1.5
        )
        
        density_grid, grid_info = extractor.create_density_grid(tracks_xyz)
        
        # Apply smoothing
        density_grid = extractor.smooth_grid(density_grid)
        
        # Extract surface
        threshold = 0.15
        mesh = extractor.extract_surface(density_grid, grid_info, threshold=threshold)
        
        if mesh is None or mesh.n_vertices == 0:
            # Fallback to direct mesh
            logger.warning("Voxelization produced no mesh, using direct mesh")
            return transform.ribbons_to_direct_mesh(ribbons)
        
        return mesh.vertices, mesh.faces
    else:
        # Direct mesh from ribbons
        return transform.ribbons_to_direct_mesh(ribbons)
