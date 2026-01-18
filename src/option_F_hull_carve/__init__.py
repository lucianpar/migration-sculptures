"""
Option F: Hull-Constrained Organic Carving

Generate a clean E-style hull, then subtract a D-style organic corridor volume
using SDF/voxel difference so the exterior stays sculptural while the interior
stays rich.

Core concept: A minus B = max(S_A, -S_B) in SDF space
- S_A: Clean PCA capsule hull (E-like silhouette)
- S_B: Organic corridor volume (D-like interior)
"""

from pathlib import Path

__version__ = "1.0.0"

MODULE_DIR = Path(__file__).parent
