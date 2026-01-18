"""
Module H3: SDF Ridge Shell for Blender

This module provides tools to create winding topographic ridge structures
in Blender using Geometry Nodes. It transforms curve data (from migration
tracks) into thin, branching ridge geometry on a sphere.

Contents:
- h3_setup_blender.py: Blender Python script to create the node group
- export_curves.py: Export track data as Blender-importable curves
"""

from pathlib import Path

__version__ = "1.0.0"

MODULE_DIR = Path(__file__).parent
