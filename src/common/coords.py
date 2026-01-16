"""
Coordinate transformation utilities.

Unit Flow (NON-NEGOTIABLE):
Raw GPS (lat/lon, degrees) → UTM Zone 10N → meters (x_m, y_m)

This is the ONLY place where coordinate transformation should happen.
All downstream processing works in meters.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from pyproj import Transformer, CRS
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    logger.warning("pyproj not available. Using simplified projection.")


@dataclass
class UTMCoordinates:
    """UTM coordinates in meters."""
    x_m: np.ndarray  # Easting in meters
    y_m: np.ndarray  # Northing in meters
    zone: int
    hemisphere: str
    
    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in meters."""
        return (
            float(np.min(self.x_m)),
            float(np.max(self.x_m)),
            float(np.min(self.y_m)),
            float(np.max(self.y_m))
        )
    
    @property
    def extent_m(self) -> Tuple[float, float]:
        """Return (x_extent, y_extent) in meters."""
        x_min, x_max, y_min, y_max = self.bounds
        return (x_max - x_min, y_max - y_min)
    
    @property
    def max_extent_m(self) -> float:
        """Return maximum extent in meters."""
        return max(self.extent_m)


class CoordinateTransformer:
    """
    Transform GPS coordinates to UTM (meters).
    
    This is the ONLY place where coordinate transformation should happen.
    All downstream processing works in meters.
    """
    
    def __init__(self, utm_zone: int = 10, hemisphere: str = "N"):
        """
        Initialize transformer.
        
        Args:
            utm_zone: UTM zone number (default 10 for California)
            hemisphere: 'N' or 'S'
        """
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere
        
        if PYPROJ_AVAILABLE:
            # WGS84 to UTM
            self.crs_wgs84 = CRS.from_epsg(4326)
            epsg_utm = 32600 + utm_zone if hemisphere == "N" else 32700 + utm_zone
            self.crs_utm = CRS.from_epsg(epsg_utm)
            self.transformer = Transformer.from_crs(
                self.crs_wgs84, self.crs_utm, always_xy=True
            )
            logger.info(f"Using pyproj for UTM Zone {utm_zone}{hemisphere} (EPSG:{epsg_utm})")
        else:
            self.transformer = None
            logger.warning("Using simplified equirectangular projection")
    
    def to_utm(
        self,
        lon: np.ndarray,
        lat: np.ndarray
    ) -> UTMCoordinates:
        """
        Transform longitude/latitude to UTM coordinates in meters.
        
        Args:
            lon: Longitude array (degrees)
            lat: Latitude array (degrees)
            
        Returns:
            UTMCoordinates with x_m, y_m in meters
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        
        if self.transformer is not None:
            x_m, y_m = self.transformer.transform(lon, lat)
        else:
            # Simplified projection (less accurate but no dependencies)
            lat_rad = np.radians(lat)
            lon_rad = np.radians(lon)
            
            # Reference point (center of data)
            lat0 = np.mean(lat_rad)
            lon0 = np.mean(lon_rad)
            
            # Earth radius in meters
            R = 6371000.0
            
            # Simple equirectangular projection
            x_m = R * (lon_rad - lon0) * np.cos(lat0)
            y_m = R * (lat_rad - lat0)
        
        return UTMCoordinates(
            x_m=x_m.astype(np.float64),
            y_m=y_m.astype(np.float64),
            zone=self.utm_zone,
            hemisphere=self.hemisphere
        )
    
    @classmethod
    def for_santa_barbara_channel(cls) -> "CoordinateTransformer":
        """Create transformer for Santa Barbara Channel (UTM Zone 10N)."""
        return cls(utm_zone=10, hemisphere="N")


def project_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    zone: int = 10,
    hemisphere: str = "N"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to project GPS to UTM meters.
    
    Args:
        lon: Longitude array
        lat: Latitude array
        zone: UTM zone
        hemisphere: 'N' or 'S'
        
    Returns:
        Tuple of (x_meters, y_meters)
    """
    transformer = CoordinateTransformer(zone, hemisphere)
    utm = transformer.to_utm(lon, lat)
    return utm.x_m, utm.y_m
