"""
Coordinate Transformation Module

Handles conversion between geographic coordinates (lat/lon) and 
projected coordinates (UTM) for 3D modeling.
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Import pyproj conditionally to handle missing dependency gracefully
try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logger.warning("pyproj not installed. Install with: pip install pyproj")


@dataclass
class BoundingBox:
    """Geographic or projected bounding box."""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )


class CoordinateTransformer:
    """
    Transforms coordinates between geographic (WGS84) and projected (UTM) systems.
    
    The UTM projection is used because:
    - Distances are in meters, suitable for 3D modeling
    - It preserves local shapes and angles
    - Zone 10N covers Southern California (Santa Barbara Channel)
    """
    
    # Santa Barbara Channel is in UTM Zone 10N
    DEFAULT_UTM_ZONE = 10
    DEFAULT_HEMISPHERE = "N"
    
    # EPSG codes
    WGS84_EPSG = 4326
    UTM_10N_EPSG = 32610
    
    def __init__(
        self,
        utm_zone: int = DEFAULT_UTM_ZONE,
        hemisphere: str = DEFAULT_HEMISPHERE,
        origin: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize the coordinate transformer.
        
        Args:
            utm_zone: UTM zone number (1-60)
            hemisphere: 'N' for northern, 'S' for southern
            origin: Optional origin point (lon, lat) for local coordinates.
                    If provided, all coordinates will be relative to this point.
        """
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere
        self.origin = origin
        self._origin_utm: Optional[Tuple[float, float]] = None
        
        if not PYPROJ_AVAILABLE:
            raise ImportError("pyproj is required for coordinate transformation")
        
        # Compute EPSG code for UTM zone
        if hemisphere.upper() == "N":
            self.utm_epsg = 32600 + utm_zone
        else:
            self.utm_epsg = 32700 + utm_zone
        
        # Create transformers
        self._to_utm = Transformer.from_crs(
            CRS.from_epsg(self.WGS84_EPSG),
            CRS.from_epsg(self.utm_epsg),
            always_xy=True  # lon, lat order
        )
        
        self._to_wgs84 = Transformer.from_crs(
            CRS.from_epsg(self.utm_epsg),
            CRS.from_epsg(self.WGS84_EPSG),
            always_xy=True
        )
        
        # If origin specified, compute its UTM coordinates
        if origin:
            self._origin_utm = self._to_utm.transform(origin[0], origin[1])
            logger.info(f"Origin set to: lon={origin[0]}, lat={origin[1]} -> "
                       f"x={self._origin_utm[0]:.1f}m, y={self._origin_utm[1]:.1f}m")
    
    def to_utm(
        self,
        lon: Union[float, np.ndarray],
        lat: Union[float, np.ndarray],
        relative: bool = True
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert geographic coordinates to UTM.
        
        Args:
            lon: Longitude(s) in degrees
            lat: Latitude(s) in degrees
            relative: If True and origin is set, return coordinates relative to origin
            
        Returns:
            Tuple of (easting, northing) in meters
        """
        x, y = self._to_utm.transform(lon, lat)
        
        if relative and self._origin_utm:
            x = x - self._origin_utm[0]
            y = y - self._origin_utm[1]
        
        return x, y
    
    def to_wgs84(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        relative: bool = True
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert UTM coordinates to geographic.
        
        Args:
            x: Easting(s) in meters
            y: Northing(s) in meters
            relative: If True and origin is set, treat input as relative to origin
            
        Returns:
            Tuple of (longitude, latitude) in degrees
        """
        if relative and self._origin_utm:
            x = x + self._origin_utm[0]
            y = y + self._origin_utm[1]
        
        lon, lat = self._to_wgs84.transform(x, y)
        return lon, lat
    
    def transform_tracks(
        self,
        tracks: list,
        lon_field: str = "longitude",
        lat_field: str = "latitude"
    ) -> list:
        """
        Transform a list of tracks from geographic to UTM coordinates.
        
        Args:
            tracks: List of track dictionaries, each containing coordinate arrays
            lon_field: Name of longitude field
            lat_field: Name of latitude field
            
        Returns:
            List of tracks with added 'x' and 'y' fields (UTM meters)
        """
        transformed = []
        
        for track in tracks:
            track_copy = track.copy()
            lon = np.array(track[lon_field])
            lat = np.array(track[lat_field])
            
            x, y = self.to_utm(lon, lat)
            track_copy['x'] = x
            track_copy['y'] = y
            
            transformed.append(track_copy)
        
        return transformed
    
    def get_bounds(
        self,
        lon: np.ndarray,
        lat: np.ndarray
    ) -> Tuple[BoundingBox, BoundingBox]:
        """
        Compute bounding boxes in both coordinate systems.
        
        Args:
            lon: Array of longitudes
            lat: Array of latitudes
            
        Returns:
            Tuple of (geographic_bounds, utm_bounds)
        """
        geo_bounds = BoundingBox(
            min_x=float(np.min(lon)),
            max_x=float(np.max(lon)),
            min_y=float(np.min(lat)),
            max_y=float(np.max(lat))
        )
        
        x, y = self.to_utm(lon, lat, relative=False)
        utm_bounds = BoundingBox(
            min_x=float(np.min(x)),
            max_x=float(np.max(x)),
            min_y=float(np.min(y)),
            max_y=float(np.max(y))
        )
        
        return geo_bounds, utm_bounds
    
    @classmethod
    def for_santa_barbara_channel(cls) -> "CoordinateTransformer":
        """
        Create a transformer configured for the Santa Barbara Channel.
        
        The origin is set to the approximate center of the channel.
        
        Returns:
            Configured CoordinateTransformer instance
        """
        # Approximate center of Santa Barbara Channel
        channel_center = (-120.0, 34.25)
        
        return cls(
            utm_zone=10,
            hemisphere="N",
            origin=channel_center
        )


def compute_distance(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute Euclidean distance between points in projected coordinates.
    
    Args:
        x1, y1: First point(s) coordinates in meters
        x2, y2: Second point(s) coordinates in meters
        
    Returns:
        Distance(s) in meters
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_heading(
    x1: Union[float, np.ndarray],
    y1: Union[float, np.ndarray],
    x2: Union[float, np.ndarray],
    y2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Compute heading (bearing) from point 1 to point 2.
    
    Args:
        x1, y1: First point(s) coordinates
        x2, y2: Second point(s) coordinates
        
    Returns:
        Heading in degrees (0=North, 90=East, etc.)
    """
    dx = x2 - x1
    dy = y2 - y1
    
    # atan2 gives angle from positive x-axis (East)
    # Convert to compass bearing (from North)
    heading = np.degrees(np.arctan2(dx, dy))
    
    # Normalize to 0-360
    heading = np.mod(heading, 360)
    
    return heading


if __name__ == "__main__":
    # Test the transformer
    logging.basicConfig(level=logging.INFO)
    
    transformer = CoordinateTransformer.for_santa_barbara_channel()
    
    # Test point: Santa Barbara Harbor
    test_lon, test_lat = -119.6885, 34.4008
    
    x, y = transformer.to_utm(test_lon, test_lat)
    print(f"Santa Barbara Harbor: ({test_lon}, {test_lat}) -> ({x:.1f}m, {y:.1f}m)")
    
    # Round trip
    lon_back, lat_back = transformer.to_wgs84(x, y)
    print(f"Round trip: ({lon_back:.6f}, {lat_back:.6f})")
