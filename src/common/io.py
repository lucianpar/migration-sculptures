"""
Data I/O utilities.

Handles loading raw tracks and saving meshes with proper metadata.
All track data is in canonical format: track_id, t_seconds, x_m, y_m, z_m
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from .coords import CoordinateTransformer
from .config import UnitMode, MeshMetadata

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not available")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    logger.warning("trimesh not available")


@dataclass
class Track:
    """
    Single whale track in canonical format (UTM meters).
    
    All coordinates are in meters after UTM projection.
    z_m is time-mapped height (also in meters, scaled to match x/y extent).
    """
    track_id: str
    x_m: np.ndarray  # Easting in meters
    y_m: np.ndarray  # Northing in meters
    z_m: np.ndarray  # Time-mapped height in meters
    t_seconds: np.ndarray  # Time in seconds from epoch
    
    @property
    def n_points(self) -> int:
        return len(self.x_m)
    
    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds for each axis."""
        return {
            'x': (float(np.min(self.x_m)), float(np.max(self.x_m))),
            'y': (float(np.min(self.y_m)), float(np.max(self.y_m))),
            'z': (float(np.min(self.z_m)), float(np.max(self.z_m)))
        }
    
    @property
    def points_m(self) -> np.ndarray:
        """Return Nx3 array of points in meters."""
        return np.column_stack([self.x_m, self.y_m, self.z_m])
    
    def resample(self, distance_m: float) -> "Track":
        """
        Resample track to approximately uniform spacing.
        
        Args:
            distance_m: Target distance between points in meters
            
        Returns:
            New Track with resampled points
        """
        if self.n_points < 2:
            return self
        
        # Compute cumulative distance
        dx = np.diff(self.x_m)
        dy = np.diff(self.y_m)
        dz = np.diff(self.z_m)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_dist = cumulative_dist[-1]
        
        if total_dist < distance_m:
            return self
        
        # New sample points
        n_samples = max(2, int(total_dist / distance_m))
        new_dists = np.linspace(0, total_dist, n_samples)
        
        # Interpolate
        new_x = np.interp(new_dists, cumulative_dist, self.x_m)
        new_y = np.interp(new_dists, cumulative_dist, self.y_m)
        new_z = np.interp(new_dists, cumulative_dist, self.z_m)
        new_t = np.interp(new_dists, cumulative_dist, self.t_seconds)
        
        return Track(
            track_id=self.track_id,
            x_m=new_x,
            y_m=new_y,
            z_m=new_z,
            t_seconds=new_t
        )


@dataclass
class TrackData:
    """
    Collection of tracks for a specimen.
    
    All coordinates are in meters (UTM projection).
    """
    specimen_id: str
    tracks: List[Track]
    utm_zone: int = 10
    utm_hemisphere: str = "N"
    
    @property
    def n_tracks(self) -> int:
        return len(self.tracks)
    
    @property
    def n_points(self) -> int:
        return sum(t.n_points for t in self.tracks)
    
    @property
    def all_points_m(self) -> np.ndarray:
        """Return all points as Nx3 array in meters."""
        if not self.tracks:
            return np.empty((0, 3))
        return np.vstack([t.points_m for t in self.tracks])
    
    @property
    def bounds_m(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds in meters for each axis."""
        pts = self.all_points_m
        if len(pts) == 0:
            return {'x': (0, 0), 'y': (0, 0), 'z': (0, 0)}
        return {
            'x': (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
            'y': (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
            'z': (float(np.min(pts[:, 2])), float(np.max(pts[:, 2])))
        }
    
    @property
    def max_extent_m(self) -> float:
        """Return maximum extent in meters."""
        bounds = self.bounds_m
        extents = [b[1] - b[0] for b in bounds.values()]
        return max(extents) if extents else 0.0
    
    def resample_all(self, distance_m: float) -> "TrackData":
        """Resample all tracks to approximately uniform spacing."""
        return TrackData(
            specimen_id=self.specimen_id,
            tracks=[t.resample(distance_m) for t in self.tracks],
            utm_zone=self.utm_zone,
            utm_hemisphere=self.utm_hemisphere
        )


def load_tracks(
    path: Path,
    specimen_id: Optional[str] = None,
    z_scale: float = 1.0,
    utm_zone: int = 10,
    utm_hemisphere: str = "N"
) -> TrackData:
    """
    Load tracks from CSV or parquet file.
    
    Expected columns:
    - timestamp or t_seconds
    - location-long or longitude or lon
    - location-lat or latitude or lat
    - individual-local-identifier or track_id
    
    Args:
        path: Path to data file
        specimen_id: ID for this specimen (defaults to filename)
        z_scale: Scale factor for time→z mapping
        utm_zone: UTM zone for projection
        utm_hemisphere: UTM hemisphere
        
    Returns:
        TrackData with all tracks in meters
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for loading tracks")
    
    path = Path(path)
    specimen_id = specimen_id or path.stem
    
    # Load data
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    logger.info(f"Loaded {len(df)} points from {path}")
    
    # Find columns
    def find_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None
    
    lon_col = find_col(['location-long', 'longitude', 'lon'])
    lat_col = find_col(['location-lat', 'latitude', 'lat'])
    time_col = find_col(['timestamp', 't_seconds', 'time'])
    id_col = find_col(['individual-local-identifier', 'track_id', 'id', 'whale_id'])
    
    if not lon_col or not lat_col:
        raise ValueError(f"Could not find lat/lon columns in {df.columns.tolist()}")
    
    # Project to UTM
    transformer = CoordinateTransformer(utm_zone, utm_hemisphere)
    
    # Parse timestamps
    if time_col:
        if df[time_col].dtype == 'object':
            df['_t'] = pd.to_datetime(df[time_col])
            t_seconds = (df['_t'] - df['_t'].min()).dt.total_seconds().values
        else:
            t_seconds = df[time_col].values.astype(float)
            t_seconds = t_seconds - t_seconds.min()
    else:
        t_seconds = np.arange(len(df), dtype=float)
    
    # Compute global time range for z-mapping
    t_range = t_seconds.max() - t_seconds.min() if len(t_seconds) > 1 else 1.0
    
    # Project all points
    lons = df[lon_col].values
    lats = df[lat_col].values
    utm = transformer.to_utm(lons, lats)
    
    # Compute spatial extent for z scaling
    spatial_extent = max(np.ptp(utm.x_m), np.ptp(utm.y_m))
    z_range = spatial_extent * z_scale
    
    # Group by track ID
    tracks = []
    if id_col and id_col in df.columns:
        for track_id, indices in df.groupby(id_col).groups.items():
            idx = indices.values if hasattr(indices, 'values') else list(indices)
            idx = sorted(idx)  # Sort by original order
            
            t_sec = t_seconds[idx]
            t_norm = (t_sec - t_sec.min()) / (t_range + 1e-10)
            z_m = t_norm * z_range
            
            tracks.append(Track(
                track_id=str(track_id),
                x_m=utm.x_m[idx],
                y_m=utm.y_m[idx],
                z_m=z_m,
                t_seconds=t_sec
            ))
    else:
        # Single track
        t_norm = t_seconds / (t_range + 1e-10)
        z_m = t_norm * z_range
        
        tracks.append(Track(
            track_id="track_0",
            x_m=utm.x_m,
            y_m=utm.y_m,
            z_m=z_m,
            t_seconds=t_seconds
        ))
    
    logger.info(f"Parsed {len(tracks)} tracks, {sum(t.n_points for t in tracks)} total points")
    logger.info(f"Spatial extent: {spatial_extent:.0f}m, Z range: {z_range:.0f}m")
    
    return TrackData(
        specimen_id=specimen_id,
        tracks=tracks,
        utm_zone=utm_zone,
        utm_hemisphere=utm_hemisphere
    )


def save_mesh(
    mesh: "trimesh.Trimesh",
    path: Path,
    metadata: MeshMetadata
) -> None:
    """
    Save mesh to GLB file with metadata sidecar.
    
    Args:
        mesh: Trimesh mesh object
        path: Output path (should end in .glb)
        metadata: MeshMetadata object (will be saved as .json sidecar)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for saving meshes")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save mesh
    mesh.export(str(path))
    logger.info(f"Saved mesh: {path} ({metadata.n_vertices} verts, {metadata.n_triangles} tris)")
    
    # Save metadata sidecar
    meta_path = path.with_suffix('.json')
    metadata.save(meta_path)
    logger.info(f"Saved metadata: {meta_path}")


def load_mesh(path: Path) -> Tuple["trimesh.Trimesh", Optional[MeshMetadata]]:
    """
    Load mesh and its metadata sidecar.
    
    Args:
        path: Path to mesh file
        
    Returns:
        Tuple of (mesh, metadata) - metadata may be None if not found
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh required for loading meshes")
    
    path = Path(path)
    mesh = trimesh.load(str(path))
    
    meta_path = path.with_suffix('.json')
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = MeshMetadata.from_dict(json.load(f))
    
    return mesh, metadata
