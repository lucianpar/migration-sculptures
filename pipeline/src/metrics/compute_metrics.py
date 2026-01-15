"""
Metrics Calculation Module

Computes quantitative metrics for whale migration sculptures:
- Route Coherence: How tightly bundled the migration paths are
- Density Entropy: Spatial concentration of travel paths (Shannon entropy)
- Year-to-Year Drift: Centroid shift compared to baseline
- Temporal Variability: Route consistency across time
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SculptureMetrics:
    """Computed metrics for a migration sculpture."""
    
    # Route Coherence (0-1, higher = tighter bundle)
    coherence: float
    coherence_rating: str  # "Low", "Moderate", "High"
    
    # Density Entropy (0-1 normalized, lower = more concentrated)
    entropy: float
    entropy_rating: str
    
    # Year-to-Year Drift (in km from baseline)
    centroid_drift_km: float
    drift_direction: str  # "North", "South", "East", "West", etc.
    
    # Temporal Variability (0-1, lower = more consistent)
    temporal_variability: float
    variability_rating: str
    
    def to_dict(self) -> Dict:
        return {
            "coherence": {
                "value": round(self.coherence, 3),
                "rating": self.coherence_rating
            },
            "entropy": {
                "value": round(self.entropy, 3),
                "rating": self.entropy_rating
            },
            "centroid_drift": {
                "value_km": round(self.centroid_drift_km, 2),
                "direction": self.drift_direction
            },
            "temporal_variability": {
                "value": round(self.temporal_variability, 3),
                "rating": self.variability_rating
            }
        }


class MetricsCalculator:
    """
    Calculates quantitative metrics for sculpture specimens.
    
    These metrics provide analytical context for the artistic visualizations,
    allowing viewers to understand what the sculptural form represents
    in ecological terms.
    """
    
    def __init__(
        self,
        grid_size: int = 50,
        baseline_year: Optional[int] = None
    ):
        """
        Initialize calculator.
        
        Args:
            grid_size: Resolution for spatial calculations
            baseline_year: Reference year for drift calculations
        """
        self.grid_size = grid_size
        self.baseline_year = baseline_year
    
    def calculate_all(
        self,
        bundled_tracks: List,
        original_tracks: Optional[List] = None,
        historical_centroid: Optional[Tuple[float, float]] = None
    ) -> SculptureMetrics:
        """
        Calculate all metrics for a specimen.
        
        Args:
            bundled_tracks: List of BundledTrack objects
            original_tracks: Optional original (unbundled) tracks for comparison
            historical_centroid: Optional (x, y) centroid from baseline year
            
        Returns:
            SculptureMetrics object
        """
        # Extract coordinates
        coords = [(t.x, t.y) for t in bundled_tracks]
        
        # Calculate individual metrics
        coherence = self.calculate_coherence(coords)
        entropy = self.calculate_density_entropy(coords)
        drift_km, drift_dir = self.calculate_centroid_drift(coords, historical_centroid)
        variability = self.calculate_temporal_variability(coords, original_tracks)
        
        return SculptureMetrics(
            coherence=coherence,
            coherence_rating=self._rate_coherence(coherence),
            entropy=entropy,
            entropy_rating=self._rate_entropy(entropy),
            centroid_drift_km=drift_km,
            drift_direction=drift_dir,
            temporal_variability=variability,
            variability_rating=self._rate_variability(variability)
        )
    
    def calculate_coherence(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Calculate route coherence (bundle tightness).
        
        Coherence is the inverse of average deviation from the median path.
        High coherence = all whales following similar routes.
        
        Args:
            tracks: List of (x, y) coordinate arrays
            
        Returns:
            Coherence score (0-1)
        """
        if len(tracks) < 2:
            return 1.0
        
        # Normalize all tracks to same number of points
        n_points = 100
        normalized = []
        
        for x, y in tracks:
            if len(x) < 2:
                continue
            t_orig = np.linspace(0, 1, len(x))
            t_new = np.linspace(0, 1, n_points)
            x_interp = np.interp(t_new, t_orig, x)
            y_interp = np.interp(t_new, t_orig, y)
            normalized.append(np.column_stack([x_interp, y_interp]))
        
        if len(normalized) < 2:
            return 1.0
        
        normalized = np.array(normalized)  # (n_tracks, n_points, 2)
        
        # Compute median path
        median_path = np.median(normalized, axis=0)  # (n_points, 2)
        
        # Compute average deviation from median
        deviations = []
        for track in normalized:
            distances = np.sqrt(np.sum((track - median_path)**2, axis=1))
            deviations.append(np.mean(distances))
        
        mean_deviation = np.mean(deviations)
        
        # Compute scale (typical distance between start and end)
        scale = np.sqrt(
            (median_path[-1, 0] - median_path[0, 0])**2 +
            (median_path[-1, 1] - median_path[0, 1])**2
        )
        
        if scale == 0:
            return 1.0
        
        # Normalize deviation by scale and convert to coherence
        normalized_deviation = mean_deviation / scale
        coherence = np.exp(-normalized_deviation * 5)  # Exponential mapping
        
        return float(np.clip(coherence, 0, 1))
    
    def calculate_density_entropy(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """
        Calculate spatial density entropy.
        
        Lower entropy = more concentrated migration corridor.
        Higher entropy = more dispersed movement patterns.
        
        Uses Shannon entropy on a 2D density grid.
        
        Args:
            tracks: List of (x, y) coordinate arrays
            
        Returns:
            Normalized entropy (0-1)
        """
        if not tracks:
            return 0.0
        
        # Collect all points
        all_x = np.concatenate([t[0] for t in tracks])
        all_y = np.concatenate([t[1] for t in tracks])
        
        # Create density grid
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        # Avoid division by zero
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        
        # Bin points into grid
        density = np.zeros((self.grid_size, self.grid_size))
        
        ix = ((all_x - x_min) / x_range * (self.grid_size - 1)).astype(int)
        iy = ((all_y - y_min) / y_range * (self.grid_size - 1)).astype(int)
        
        ix = np.clip(ix, 0, self.grid_size - 1)
        iy = np.clip(iy, 0, self.grid_size - 1)
        
        for i in range(len(ix)):
            density[ix[i], iy[i]] += 1
        
        # Normalize to probability distribution
        total = np.sum(density)
        if total == 0:
            return 0.0
        
        p = density / total
        
        # Compute Shannon entropy
        # H = -sum(p * log(p)) for p > 0
        p_nonzero = p[p > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(self.grid_size * self.grid_size)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def calculate_centroid_drift(
        self,
        tracks: List[Tuple[np.ndarray, np.ndarray]],
        historical_centroid: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, str]:
        """
        Calculate drift of current centroid from historical baseline.
        
        Args:
            tracks: List of (x, y) coordinate arrays
            historical_centroid: Optional (x, y) of baseline centroid
            
        Returns:
            Tuple of (drift_km, direction_string)
        """
        if not tracks:
            return 0.0, "Unknown"
        
        # Compute current centroid
        all_x = np.concatenate([t[0] for t in tracks])
        all_y = np.concatenate([t[1] for t in tracks])
        
        current_centroid = (np.mean(all_x), np.mean(all_y))
        
        if historical_centroid is None:
            return 0.0, "N/A (no baseline)"
        
        # Compute drift
        dx = current_centroid[0] - historical_centroid[0]
        dy = current_centroid[1] - historical_centroid[1]
        
        drift_m = np.sqrt(dx**2 + dy**2)
        drift_km = drift_m / 1000.0
        
        # Determine direction
        if drift_m < 100:  # Less than 100m = essentially no drift
            direction = "Stable"
        else:
            angle = np.degrees(np.arctan2(dx, dy))  # 0 = North, 90 = East
            
            if -22.5 <= angle < 22.5:
                direction = "North"
            elif 22.5 <= angle < 67.5:
                direction = "Northeast"
            elif 67.5 <= angle < 112.5:
                direction = "East"
            elif 112.5 <= angle < 157.5:
                direction = "Southeast"
            elif angle >= 157.5 or angle < -157.5:
                direction = "South"
            elif -157.5 <= angle < -112.5:
                direction = "Southwest"
            elif -112.5 <= angle < -67.5:
                direction = "West"
            else:
                direction = "Northwest"
        
        return float(drift_km), direction
    
    def calculate_temporal_variability(
        self,
        current_tracks: List[Tuple[np.ndarray, np.ndarray]],
        reference_tracks: Optional[List] = None
    ) -> float:
        """
        Calculate temporal variability (route consistency over time).
        
        If reference tracks from other years are provided, compares
        the current routes to historical patterns.
        
        Args:
            current_tracks: Current specimen tracks
            reference_tracks: Optional historical tracks for comparison
            
        Returns:
            Variability score (0-1, lower = more consistent)
        """
        if not current_tracks:
            return 0.0
        
        if reference_tracks is None:
            # Without reference, estimate from internal variance
            coherence = self.calculate_coherence(current_tracks)
            # Inverse of coherence represents variability
            return 1.0 - coherence
        
        # With reference, compare median paths
        def get_median_path(tracks, n_points=100):
            normalized = []
            for x, y in tracks:
                if len(x) < 2:
                    continue
                t = np.linspace(0, 1, len(x))
                t_new = np.linspace(0, 1, n_points)
                normalized.append(np.column_stack([
                    np.interp(t_new, t, x),
                    np.interp(t_new, t, y)
                ]))
            if not normalized:
                return None
            return np.median(normalized, axis=0)
        
        current_median = get_median_path(current_tracks)
        ref_median = get_median_path(reference_tracks)
        
        if current_median is None or ref_median is None:
            return 0.5  # Unknown
        
        # Compute path difference
        distances = np.sqrt(np.sum((current_median - ref_median)**2, axis=1))
        mean_distance = np.mean(distances)
        
        # Normalize by path length
        path_length = np.sum(np.sqrt(np.sum(np.diff(ref_median, axis=0)**2, axis=1)))
        
        if path_length == 0:
            return 0.5
        
        normalized_diff = mean_distance / path_length
        variability = np.tanh(normalized_diff * 5)  # Map to 0-1
        
        return float(variability)
    
    def _rate_coherence(self, value: float) -> str:
        """Convert coherence value to qualitative rating."""
        if value >= 0.7:
            return "High"
        elif value >= 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def _rate_entropy(self, value: float) -> str:
        """Convert entropy value to qualitative rating."""
        if value <= 0.3:
            return "Low (concentrated)"
        elif value <= 0.6:
            return "Moderate"
        else:
            return "High (dispersed)"
    
    def _rate_variability(self, value: float) -> str:
        """Convert variability value to qualitative rating."""
        if value <= 0.3:
            return "Low (consistent)"
        elif value <= 0.6:
            return "Moderate"
        else:
            return "High (variable)"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    np.random.seed(42)
    
    # Create tight bundle (high coherence)
    tight_tracks = []
    for i in range(10):
        t = np.linspace(0, 1, 50)
        x = t * 100000 + np.random.randn(50) * 1000
        y = np.sin(t * np.pi) * 20000 + np.random.randn(50) * 1000
        tight_tracks.append((x, y))
    
    # Create loose bundle (low coherence)
    loose_tracks = []
    for i in range(10):
        t = np.linspace(0, 1, 50)
        x = t * 100000 + np.random.randn(50) * 10000
        y = np.sin(t * np.pi + np.random.randn() * 0.5) * 20000 + np.random.randn(50) * 10000
        loose_tracks.append((x, y))
    
    calc = MetricsCalculator()
    
    print("Tight bundle metrics:")
    print(f"  Coherence: {calc.calculate_coherence(tight_tracks):.3f}")
    print(f"  Entropy: {calc.calculate_density_entropy(tight_tracks):.3f}")
    
    print("\nLoose bundle metrics:")
    print(f"  Coherence: {calc.calculate_coherence(loose_tracks):.3f}")
    print(f"  Entropy: {calc.calculate_density_entropy(loose_tracks):.3f}")
