"""
Track Processor Module

Handles track segmentation, resampling, smoothing, and outlier removal
for whale migration data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class Track:
    """Represents a single whale track (series of positions)."""
    track_id: str
    individual_id: str
    species: str
    timestamps: np.ndarray  # datetime64 array
    lons: np.ndarray
    lats: np.ndarray
    x: Optional[np.ndarray] = None  # UTM easting
    y: Optional[np.ndarray] = None  # UTM northing
    z: Optional[np.ndarray] = None  # Depth (if available)
    speeds: Optional[np.ndarray] = None  # km/h
    headings: Optional[np.ndarray] = None  # degrees
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_points(self) -> int:
        return len(self.timestamps)
    
    @property
    def duration_days(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]) / np.timedelta64(1, 'D')
    
    @property
    def start_date(self) -> datetime:
        return self.timestamps[0].astype('datetime64[s]').astype(datetime)
    
    @property
    def end_date(self) -> datetime:
        return self.timestamps[-1].astype('datetime64[s]').astype(datetime)


@dataclass  
class Specimen:
    """
    A specimen represents a collection of tracks for one sculpture.
    
    Specimens are grouped by species, season, and year.
    """
    species: str
    season: str
    year: int
    tracks: List[Track] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_tracks(self) -> int:
        return len(self.tracks)
    
    @property
    def n_points(self) -> int:
        return sum(t.n_points for t in self.tracks)


class TrackProcessor:
    """
    Processes raw whale tracking data into clean, consistent tracks.
    
    Pipeline stages:
    1. Parse raw data into Track objects
    2. Filter by geographic bounds and date range
    3. Compute derived fields (speed, heading)
    4. Remove outlier points (unrealistic speeds)
    5. Resample to uniform intervals
    6. Smooth using spline interpolation
    7. Group into Specimens by species/season/year
    """
    
    # Default processing parameters
    DEFAULT_MAX_SPEED_KMH = 30.0  # Maximum realistic whale speed
    DEFAULT_RESAMPLE_HOURS = 1.0
    DEFAULT_MIN_TRACK_POINTS = 10
    
    def __init__(
        self,
        max_speed_kmh: float = DEFAULT_MAX_SPEED_KMH,
        resample_interval_hours: float = DEFAULT_RESAMPLE_HOURS,
        min_track_points: int = DEFAULT_MIN_TRACK_POINTS
    ):
        """
        Initialize track processor.
        
        Args:
            max_speed_kmh: Maximum realistic whale speed for outlier detection
            resample_interval_hours: Target interval for resampling
            min_track_points: Minimum points required for a valid track
        """
        self.max_speed_kmh = max_speed_kmh
        self.resample_interval_hours = resample_interval_hours
        self.min_track_points = min_track_points
    
    def parse_movebank_data(self, df: "pd.DataFrame") -> List[Track]:
        """
        Parse Movebank tracking data into Track objects.
        
        Expected columns:
        - individual-local-identifier or tag-local-identifier
        - timestamp
        - location-long
        - location-lat
        - taxon-canonical-name or individual-taxon-canonical-name
        
        Args:
            df: Pandas DataFrame with Movebank data
            
        Returns:
            List of Track objects
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data parsing")
        
        tracks = []
        
        # Identify column names (Movebank uses various naming conventions)
        id_col = self._find_column(df, [
            "individual-local-identifier",
            "tag-local-identifier",
            "individual_id"
        ])
        
        time_col = self._find_column(df, ["timestamp", "study-local-timestamp"])
        lon_col = self._find_column(df, ["location-long", "longitude"])
        lat_col = self._find_column(df, ["location-lat", "latitude"])
        species_col = self._find_column(df, [
            "taxon-canonical-name",
            "individual-taxon-canonical-name",
            "species"
        ])
        
        # Parse timestamps
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Group by individual
        for ind_id, group in df.groupby(id_col):
            group = group.sort_values(time_col)
            
            # Get species (should be same for all points)
            species = group[species_col].iloc[0] if species_col else "unknown"
            
            track = Track(
                track_id=f"track_{ind_id}",
                individual_id=str(ind_id),
                species=self._normalize_species_name(species),
                timestamps=group[time_col].values.astype('datetime64[ns]'),
                lons=group[lon_col].values.astype(float),
                lats=group[lat_col].values.astype(float),
                metadata={
                    "source": "movebank",
                    "raw_species": species
                }
            )
            
            tracks.append(track)
        
        logger.info(f"Parsed {len(tracks)} tracks from Movebank data")
        return tracks
    
    def _find_column(self, df: "pd.DataFrame", candidates: List[str]) -> Optional[str]:
        """Find first matching column name from candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def _normalize_species_name(self, name: str) -> str:
        """Normalize scientific name to common identifier."""
        name = name.lower().strip()
        
        species_map = {
            "balaenoptera musculus": "blue_whale",
            "balaenoptera physalus": "fin_whale",
            "eschrichtius robustus": "gray_whale",
            "megaptera novaeangliae": "humpback_whale",
        }
        
        for sci_name, common in species_map.items():
            if sci_name in name:
                return common
        
        return name.replace(" ", "_")
    
    def filter_by_bounds(
        self,
        tracks: List[Track],
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        require_majority: bool = True
    ) -> List[Track]:
        """
        Filter tracks to those within geographic bounds.
        
        Args:
            tracks: List of Track objects
            min_lon, max_lon, min_lat, max_lat: Bounding box
            require_majority: If True, require >50% of points within bounds
            
        Returns:
            Filtered list of tracks
        """
        filtered = []
        
        for track in tracks:
            in_bounds = (
                (track.lons >= min_lon) & (track.lons <= max_lon) &
                (track.lats >= min_lat) & (track.lats <= max_lat)
            )
            
            fraction_in_bounds = np.mean(in_bounds)
            
            if require_majority and fraction_in_bounds > 0.5:
                filtered.append(track)
            elif not require_majority and np.any(in_bounds):
                filtered.append(track)
        
        logger.info(f"Filtered to {len(filtered)}/{len(tracks)} tracks within bounds")
        return filtered
    
    def filter_by_date_range(
        self,
        tracks: List[Track],
        start_date: datetime,
        end_date: datetime,
        require_majority: bool = True
    ) -> List[Track]:
        """
        Filter tracks to those within date range.
        
        Args:
            tracks: List of Track objects
            start_date: Start of date range
            end_date: End of date range
            require_majority: If True, require >50% of points within range
            
        Returns:
            Filtered list of tracks
        """
        filtered = []
        
        start_np = np.datetime64(start_date)
        end_np = np.datetime64(end_date)
        
        for track in tracks:
            in_range = (track.timestamps >= start_np) & (track.timestamps <= end_np)
            fraction_in_range = np.mean(in_range)
            
            if require_majority and fraction_in_range > 0.5:
                filtered.append(track)
            elif not require_majority and np.any(in_range):
                filtered.append(track)
        
        logger.info(f"Filtered to {len(filtered)}/{len(tracks)} tracks in date range")
        return filtered
    
    def compute_derived_fields(
        self,
        track: Track,
        transformer: Optional[Any] = None
    ) -> Track:
        """
        Compute speed, heading, and UTM coordinates for a track.
        
        Args:
            track: Track object
            transformer: CoordinateTransformer instance
            
        Returns:
            Track with computed fields
        """
        n = track.n_points
        
        # Convert to UTM if transformer provided
        if transformer:
            track.x, track.y = transformer.to_utm(track.lons, track.lats)
        
        # Compute speeds and headings
        if track.x is not None and track.y is not None:
            # Use UTM coordinates for accurate distances
            dx = np.diff(track.x)
            dy = np.diff(track.y)
        else:
            # Approximate using lon/lat (less accurate)
            dx = np.diff(track.lons) * 111000 * np.cos(np.radians(track.lats[:-1]))
            dy = np.diff(track.lats) * 111000
        
        # Distance in meters
        distances = np.sqrt(dx**2 + dy**2)
        
        # Time differences in hours
        dt_hours = np.diff(track.timestamps) / np.timedelta64(1, 'h')
        dt_hours = np.maximum(dt_hours, 1e-6)  # Avoid division by zero
        
        # Speed in km/h
        speeds = (distances / 1000) / dt_hours
        track.speeds = np.concatenate([[0], speeds])  # First point has no speed
        
        # Heading in degrees (0=North)
        headings = np.degrees(np.arctan2(dx, dy))
        headings = np.mod(headings, 360)
        track.headings = np.concatenate([[headings[0] if len(headings) > 0 else 0], headings])
        
        return track
    
    def remove_outlier_points(self, track: Track) -> Track:
        """
        Remove points with unrealistic speeds (GPS/Argos errors).
        
        Args:
            track: Track with computed speeds
            
        Returns:
            Track with outlier points removed
        """
        if track.speeds is None:
            return track
        
        # Keep points with reasonable speeds
        valid = track.speeds <= self.max_speed_kmh
        
        # Always keep first point
        valid[0] = True
        
        n_removed = np.sum(~valid)
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} outlier points from track {track.track_id}")
        
        # Create new track with filtered points
        return Track(
            track_id=track.track_id,
            individual_id=track.individual_id,
            species=track.species,
            timestamps=track.timestamps[valid],
            lons=track.lons[valid],
            lats=track.lats[valid],
            x=track.x[valid] if track.x is not None else None,
            y=track.y[valid] if track.y is not None else None,
            z=track.z[valid] if track.z is not None else None,
            speeds=track.speeds[valid],
            headings=track.headings[valid] if track.headings is not None else None,
            metadata=track.metadata
        )
    
    def resample_track(
        self,
        track: Track,
        interval_hours: Optional[float] = None
    ) -> Track:
        """
        Resample track to uniform time intervals using interpolation.
        
        Args:
            track: Track to resample
            interval_hours: Time interval in hours (default: self.resample_interval_hours)
            
        Returns:
            Resampled track
        """
        if track.n_points < 2:
            return track
        
        interval = interval_hours or self.resample_interval_hours
        
        # Create new timestamp array
        start = track.timestamps[0]
        end = track.timestamps[-1]
        
        interval_ns = np.timedelta64(int(interval * 3600), 's')
        new_timestamps = np.arange(start, end, interval_ns)
        
        if len(new_timestamps) < 2:
            return track
        
        # Convert timestamps to numeric for interpolation
        t_orig = (track.timestamps - start) / np.timedelta64(1, 's')
        t_new = (new_timestamps - start) / np.timedelta64(1, 's')
        
        # Interpolate each coordinate
        new_lons = np.interp(t_new, t_orig, track.lons)
        new_lats = np.interp(t_new, t_orig, track.lats)
        
        new_x = np.interp(t_new, t_orig, track.x) if track.x is not None else None
        new_y = np.interp(t_new, t_orig, track.y) if track.y is not None else None
        
        return Track(
            track_id=track.track_id,
            individual_id=track.individual_id,
            species=track.species,
            timestamps=new_timestamps,
            lons=new_lons,
            lats=new_lats,
            x=new_x,
            y=new_y,
            metadata={**track.metadata, "resampled": True, "interval_hours": interval}
        )
    
    def smooth_track(
        self,
        track: Track,
        window_size: int = 5,
        method: str = "moving_average"
    ) -> Track:
        """
        Smooth track coordinates to reduce noise.
        
        Args:
            track: Track to smooth
            window_size: Size of smoothing window
            method: Smoothing method ("moving_average" or "savgol")
            
        Returns:
            Smoothed track
        """
        if track.n_points < window_size:
            return track
        
        if method == "moving_average":
            # Simple moving average
            kernel = np.ones(window_size) / window_size
            
            # Pad to handle edges
            pad = window_size // 2
            
            lons_padded = np.pad(track.lons, pad, mode='edge')
            lats_padded = np.pad(track.lats, pad, mode='edge')
            
            smooth_lons = np.convolve(lons_padded, kernel, mode='valid')
            smooth_lats = np.convolve(lats_padded, kernel, mode='valid')
            
            # Handle x, y if present
            smooth_x = None
            smooth_y = None
            if track.x is not None and track.y is not None:
                x_padded = np.pad(track.x, pad, mode='edge')
                y_padded = np.pad(track.y, pad, mode='edge')
                smooth_x = np.convolve(x_padded, kernel, mode='valid')
                smooth_y = np.convolve(y_padded, kernel, mode='valid')
        
        else:
            # Fallback to no smoothing
            smooth_lons = track.lons
            smooth_lats = track.lats
            smooth_x = track.x
            smooth_y = track.y
        
        return Track(
            track_id=track.track_id,
            individual_id=track.individual_id,
            species=track.species,
            timestamps=track.timestamps,
            lons=smooth_lons,
            lats=smooth_lats,
            x=smooth_x,
            y=smooth_y,
            metadata={**track.metadata, "smoothed": True, "smooth_method": method}
        )
    
    def group_into_specimens(
        self,
        tracks: List[Track],
        season_config: Dict[str, Dict[str, Any]]
    ) -> List[Specimen]:
        """
        Group tracks into specimens by species, season, and year.
        
        Args:
            tracks: List of processed tracks
            season_config: Season definitions from config
            
        Returns:
            List of Specimen objects
        """
        specimens = {}
        
        for track in tracks:
            species = track.species
            
            if species not in season_config:
                continue
            
            # Determine year from track midpoint
            mid_idx = len(track.timestamps) // 2
            mid_date = track.timestamps[mid_idx]
            year = mid_date.astype('datetime64[Y]').astype(int) + 1970
            
            # Determine season
            month = (mid_date.astype('datetime64[M]').astype(int) % 12) + 1
            
            season = None
            for season_name, season_def in season_config[species]["seasons"].items():
                start_month = season_def["start_month"]
                end_month = season_def["end_month"]
                
                # Handle year-crossing seasons
                if start_month <= end_month:
                    if start_month <= month <= end_month:
                        season = season_name
                        break
                else:
                    if month >= start_month or month <= end_month:
                        season = season_name
                        break
            
            if season is None:
                continue
            
            # Create specimen key
            key = (species, season, year)
            
            if key not in specimens:
                specimens[key] = Specimen(
                    species=species,
                    season=season,
                    year=year
                )
            
            specimens[key].tracks.append(track)
        
        result = list(specimens.values())
        logger.info(f"Grouped tracks into {len(result)} specimens")
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy data
    processor = TrackProcessor()
    
    # Create a simple test track
    track = Track(
        track_id="test_001",
        individual_id="whale_001",
        species="blue_whale",
        timestamps=np.array([
            np.datetime64('2015-04-01T00:00'),
            np.datetime64('2015-04-01T01:00'),
            np.datetime64('2015-04-01T02:00'),
            np.datetime64('2015-04-01T03:00'),
        ]),
        lons=np.array([-120.0, -120.1, -120.2, -120.3]),
        lats=np.array([34.0, 34.1, 34.2, 34.3])
    )
    
    print(f"Test track: {track.n_points} points, {track.duration_days:.2f} days")
