"""
Configuration and constants for sculpture generation.

Unit Model (NON-NEGOTIABLE):
- Raw GPS → UTM Zone 10N (meters) → geometry ops → normalize → export
- Mode A (default): max(bbox dimension) = 2.0 units (normalized)
- Mode B: real-world meters (debug only)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from pathlib import Path


class UnitMode(Enum):
    """
    Output unit modes - MUST be one of these.
    
    NORMALIZED (default): max dimension = 2.0 units
        - Sculptures are comparable in size
        - Ideal for specimen garden / comparison
        - In Three.js / Blender: 1 unit ≈ 1 meter (virtual)
    
    METERS: real-world scale
        - Used ONLY for debugging, sanity checks, scale screenshots
        - A 50 km corridor = 50,000 units
        - NOT for final garden layouts
    """
    NORMALIZED = "normalized"
    METERS = "meters"


@dataclass
class MeshMetadata:
    """
    Required metadata for every exported mesh.
    
    Every exported mesh MUST include:
    - unit_mode: "normalized" | "meters"
    - bbox_max_dimension: 2.0 | <meters>
    - normalization_applied: true | false
    """
    unit_mode: str
    bbox_max_dimension: float
    normalization_applied: bool
    specimen_id: str
    option: str  # A, B, or C
    n_triangles: int
    n_vertices: int
    scale_factor: Optional[float] = None  # For normalized mode
    bbox_before_normalization: Optional[Dict[str, float]] = None
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_mode": self.unit_mode,
            "bbox_max_dimension": self.bbox_max_dimension,
            "normalization_applied": self.normalization_applied,
            "specimen_id": self.specimen_id,
            "option": self.option,
            "n_triangles": self.n_triangles,
            "n_vertices": self.n_vertices,
            "scale_factor": self.scale_factor,
            "bbox_before_normalization": self.bbox_before_normalization,
            "generation_params": self.generation_params
        }
    
    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshMetadata":
        return cls(**data)


@dataclass
class Config:
    """
    Global configuration for sculpture generation.
    
    Unit Model (NON-NEGOTIABLE):
    - Raw GPS → UTM Zone 10N (meters) → geometry ops → normalize → export
    - Normalized mode: max(bbox dimension) = 2.0 units
    - Meters mode: real-world scale (debug only)
    """
    
    # Unit mode (default: normalized)
    unit_mode: UnitMode = UnitMode.NORMALIZED
    
    # Normalization target (only used if unit_mode == NORMALIZED)
    normalized_max_dim: float = 2.0
    
    # UTM zone for projection (Zone 10N for California/Santa Barbara)
    utm_zone: int = 10
    utm_hemisphere: str = "N"
    
    # Voxel grid settings
    voxel_resolution: int = 128
    
    # Mesh settings
    max_triangles: int = 100000
    smoothing_iterations: int = 3
    
    # Track processing
    track_resample_distance_m: float = 2000.0  # 2 km default
    
    # Paths (relative to project root)
    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    def get_output_path(self, option: str) -> Path:
        """Get output path for a specific option (A, B, or C)."""
        return self.output_dir / f"option_{option}_{'connective_tension' if option == 'A' else 'subtractive_volume' if option == 'B' else 'constrained_membrane'}" / "meshes"
    
    def get_meta_path(self, option: str) -> Path:
        """Get metadata path for a specific option."""
        return self.output_dir / f"option_{option}_{'connective_tension' if option == 'A' else 'subtractive_volume' if option == 'B' else 'constrained_membrane'}" / "meta"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_mode": self.unit_mode.value,
            "normalized_max_dim": self.normalized_max_dim,
            "utm_zone": self.utm_zone,
            "utm_hemisphere": self.utm_hemisphere,
            "voxel_resolution": self.voxel_resolution,
            "max_triangles": self.max_triangles,
            "smoothing_iterations": self.smoothing_iterations,
            "track_resample_distance_m": self.track_resample_distance_m
        }
    
    @classmethod
    def from_json(cls, path: Path) -> "Config":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        data["unit_mode"] = UnitMode(data.get("unit_mode", "normalized"))
        data["data_dir"] = Path(data.get("data_dir", "data"))
        data["output_dir"] = Path(data.get("output_dir", "outputs"))
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global default config
DEFAULT_CONFIG = Config()
