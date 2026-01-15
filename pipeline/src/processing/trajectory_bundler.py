"""
Trajectory Bundling Module

Implements force-directed edge bundling to pull whale tracks together
into cohesive migration "corridors" while preserving overall route patterns.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BundledTrack:
    """A track after bundling transformation."""
    track_id: str
    x: np.ndarray  # Bundled x coordinates
    y: np.ndarray  # Bundled y coordinates
    z: Optional[np.ndarray] = None  # Optional depth/time as z
    original_x: Optional[np.ndarray] = None  # Original coordinates
    original_y: Optional[np.ndarray] = None


class TrajectoryBundler:
    """
    Bundles multiple trajectories using force-directed edge bundling.
    
    The algorithm iteratively attracts each point toward nearby points
    on other tracks, creating a tighter "bundle" where routes overlap
    while preserving divergent areas.
    
    Based on: Holten & Van Wijk (2009) "Force-Directed Edge Bundling for Graph Visualization"
    """
    
    def __init__(
        self,
        k_neighbors: int = 5,
        attraction_strength: float = 0.3,
        iterations: int = 10,
        decay_rate: float = 0.9,
        compatibility_threshold: float = 0.5
    ):
        """
        Initialize the trajectory bundler.
        
        Args:
            k_neighbors: Number of nearest tracks to consider for attraction
            attraction_strength: How strongly points are pulled together (0-1)
            iterations: Number of bundling iterations
            decay_rate: Reduce attraction strength each iteration
            compatibility_threshold: Minimum compatibility for bundling (0-1)
        """
        self.k_neighbors = k_neighbors
        self.attraction_strength = attraction_strength
        self.iterations = iterations
        self.decay_rate = decay_rate
        self.compatibility_threshold = compatibility_threshold
    
    def bundle(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]],
        track_ids: Optional[List[str]] = None
    ) -> List[BundledTrack]:
        """
        Apply bundling to a set of tracks.
        
        Args:
            tracks: List of (x, y) coordinate arrays for each track
            track_ids: Optional list of track identifiers
            
        Returns:
            List of BundledTrack objects with adjusted coordinates
        """
        if len(tracks) < 2:
            # Nothing to bundle
            return [
                BundledTrack(
                    track_id=track_ids[i] if track_ids else f"track_{i}",
                    x=tracks[i][0].copy(),
                    y=tracks[i][1].copy(),
                    original_x=tracks[i][0].copy(),
                    original_y=tracks[i][1].copy()
                )
                for i in range(len(tracks))
            ]
        
        logger.info(f"Bundling {len(tracks)} tracks with {self.iterations} iterations")
        
        # Initialize working copies
        bundled_x = [t[0].copy() for t in tracks]
        bundled_y = [t[1].copy() for t in tracks]
        
        # Store original for reference
        original_x = [t[0].copy() for t in tracks]
        original_y = [t[1].copy() for t in tracks]
        
        # Iterative bundling
        strength = self.attraction_strength
        
        for iteration in range(self.iterations):
            logger.debug(f"Bundling iteration {iteration + 1}/{self.iterations}, strength={strength:.3f}")
            
            # Compute displacement for each track
            displacements_x = [np.zeros_like(x) for x in bundled_x]
            displacements_y = [np.zeros_like(y) for y in bundled_y]
            
            # For each track, compute attraction to nearby tracks
            for i in range(len(tracks)):
                self._compute_attraction(
                    i, bundled_x, bundled_y,
                    displacements_x[i], displacements_y[i],
                    strength
                )
            
            # Apply displacements
            for i in range(len(tracks)):
                bundled_x[i] += displacements_x[i]
                bundled_y[i] += displacements_y[i]
            
            # Decay strength for next iteration
            strength *= self.decay_rate
        
        # Create result objects
        results = []
        for i in range(len(tracks)):
            tid = track_ids[i] if track_ids else f"track_{i}"
            results.append(BundledTrack(
                track_id=tid,
                x=bundled_x[i],
                y=bundled_y[i],
                original_x=original_x[i],
                original_y=original_y[i]
            ))
        
        return results
    
    def _compute_attraction(
        self,
        track_idx: int,
        all_x: List[np.ndarray],
        all_y: List[np.ndarray],
        disp_x: np.ndarray,
        disp_y: np.ndarray,
        strength: float
    ) -> None:
        """
        Compute attraction forces for points on a single track.
        
        For each point on the track, finds the nearest points on
        k neighboring tracks and computes an attraction vector
        toward their centroid.
        """
        my_x = all_x[track_idx]
        my_y = all_y[track_idx]
        
        n_points = len(my_x)
        n_tracks = len(all_x)
        
        # Number of neighbors to consider (capped by available tracks)
        k = min(self.k_neighbors, n_tracks - 1)
        
        if k < 1:
            return
        
        for p in range(n_points):
            px, py = my_x[p], my_y[p]
            
            # Find nearest points on other tracks
            neighbor_points = []
            
            for j in range(n_tracks):
                if j == track_idx:
                    continue
                
                other_x = all_x[j]
                other_y = all_y[j]
                
                # Find closest point on track j
                distances = np.sqrt((other_x - px)**2 + (other_y - py)**2)
                nearest_idx = np.argmin(distances)
                nearest_dist = distances[nearest_idx]
                
                neighbor_points.append((
                    nearest_dist,
                    other_x[nearest_idx],
                    other_y[nearest_idx]
                ))
            
            # Sort by distance and take k nearest
            neighbor_points.sort(key=lambda x: x[0])
            k_nearest = neighbor_points[:k]
            
            if not k_nearest:
                continue
            
            # Compute centroid of nearest neighbors
            centroid_x = np.mean([p[1] for p in k_nearest])
            centroid_y = np.mean([p[2] for p in k_nearest])
            
            # Compute displacement toward centroid
            dx = centroid_x - px
            dy = centroid_y - py
            
            # Apply attraction (scaled by strength)
            disp_x[p] = dx * strength
            disp_y[p] = dy * strength
    
    def compute_centerline(
        self,
        bundled_tracks: List[BundledTrack],
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the average centerline path from bundled tracks.
        
        Args:
            bundled_tracks: List of bundled track objects
            n_points: Number of points for the centerline
            
        Returns:
            Tuple of (x, y) arrays for the centerline
        """
        if not bundled_tracks:
            return np.array([]), np.array([])
        
        # Normalize all tracks to same number of points
        normalized = []
        for track in bundled_tracks:
            if len(track.x) < 2:
                continue
            
            # Interpolate to n_points
            t_orig = np.linspace(0, 1, len(track.x))
            t_new = np.linspace(0, 1, n_points)
            
            x_interp = np.interp(t_new, t_orig, track.x)
            y_interp = np.interp(t_new, t_orig, track.y)
            
            normalized.append((x_interp, y_interp))
        
        if not normalized:
            return np.array([]), np.array([])
        
        # Compute mean
        all_x = np.array([t[0] for t in normalized])
        all_y = np.array([t[1] for t in normalized])
        
        centerline_x = np.mean(all_x, axis=0)
        centerline_y = np.mean(all_y, axis=0)
        
        return centerline_x, centerline_y
    
    def compute_track_deviation(
        self,
        track: BundledTrack,
        centerline_x: np.ndarray,
        centerline_y: np.ndarray
    ) -> float:
        """
        Compute average deviation of a track from the centerline.
        
        Args:
            track: Bundled track to measure
            centerline_x, centerline_y: Centerline coordinates
            
        Returns:
            Average deviation in coordinate units
        """
        if len(track.x) < 2 or len(centerline_x) < 2:
            return float('inf')
        
        # For each point on track, find distance to nearest centerline point
        deviations = []
        for i in range(len(track.x)):
            px, py = track.x[i], track.y[i]
            distances = np.sqrt((centerline_x - px)**2 + (centerline_y - py)**2)
            deviations.append(np.min(distances))
        
        return np.mean(deviations)
    
    def filter_outlier_tracks(
        self,
        bundled_tracks: List[BundledTrack],
        std_threshold: float = 2.0
    ) -> List[BundledTrack]:
        """
        Remove tracks that deviate significantly from the bundle.
        
        Args:
            bundled_tracks: List of bundled tracks
            std_threshold: Remove tracks with deviation > threshold * std
            
        Returns:
            Filtered list of tracks
        """
        if len(bundled_tracks) < 3:
            return bundled_tracks
        
        # Compute centerline
        centerline_x, centerline_y = self.compute_centerline(bundled_tracks)
        
        if len(centerline_x) == 0:
            return bundled_tracks
        
        # Compute deviations
        deviations = [
            self.compute_track_deviation(t, centerline_x, centerline_y)
            for t in bundled_tracks
        ]
        
        mean_dev = np.mean(deviations)
        std_dev = np.std(deviations)
        
        threshold = mean_dev + std_threshold * std_dev
        
        filtered = [
            t for t, d in zip(bundled_tracks, deviations)
            if d <= threshold
        ]
        
        n_removed = len(bundled_tracks) - len(filtered)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} outlier tracks (threshold={threshold:.1f})")
        
        return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test bundling with synthetic tracks
    np.random.seed(42)
    
    # Generate some synthetic tracks
    n_tracks = 10
    tracks = []
    
    for i in range(n_tracks):
        # Base path with some variation
        t = np.linspace(0, 1, 50)
        base_x = t * 100000  # 100 km
        base_y = np.sin(t * np.pi) * 20000 + np.random.randn() * 5000  # Sinusoidal with offset
        
        # Add noise
        noise_x = np.random.randn(len(t)) * 2000
        noise_y = np.random.randn(len(t)) * 2000
        
        tracks.append((base_x + noise_x, base_y + noise_y))
    
    # Bundle them
    bundler = TrajectoryBundler(
        k_neighbors=5,
        attraction_strength=0.3,
        iterations=10
    )
    
    bundled = bundler.bundle(tracks)
    
    print(f"Bundled {len(bundled)} tracks")
    
    # Compute centerline
    cx, cy = bundler.compute_centerline(bundled)
    print(f"Centerline has {len(cx)} points")
