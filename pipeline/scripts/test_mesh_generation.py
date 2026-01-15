#!/usr/bin/env python3
"""
Test script for mesh generation from whale track data.

This script tests the full pipeline from raw data to 3D mesh,
using the synthetic or downloaded data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Migration Sculptures - Test Mesh Generation")
    logger.info("=" * 60)
    
    # Load the tracking data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "raw" / "movebank" / "blue_fin_whale_tracks.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return 1
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} points")
    
    # Import pipeline modules
    from processing.coordinate_transform import CoordinateTransformer
    from processing.trajectory_bundler import TrajectoryBundler
    from geometry.isosurface import IsosurfaceExtractor
    from geometry.mesh_generator import MeshGenerator
    from geometry.gltf_exporter import GLTFExporter
    
    # Step 1: Transform coordinates to UTM
    logger.info("\nStep 1: Coordinate transformation")
    transformer = CoordinateTransformer.for_santa_barbara_channel()
    
    # Parse tracks by individual
    tracks_by_individual = {}
    for ind_id, group in df.groupby('individual-local-identifier'):
        group = group.sort_values('timestamp')
        lons = group['location-long'].values
        lats = group['location-lat'].values
        
        # Transform to UTM
        x, y = transformer.to_utm(lons, lats)
        tracks_by_individual[ind_id] = (x, y)
    
    logger.info(f"Transformed {len(tracks_by_individual)} whale tracks to UTM")
    
    # Step 2: Prepare tracks for bundling
    logger.info("\nStep 2: Trajectory bundling")
    track_coords = []
    track_ids = []
    
    for track_id, (x, y) in tracks_by_individual.items():
        if len(x) >= 10:  # Minimum points
            track_coords.append((x, y))
            track_ids.append(track_id)
    
    logger.info(f"Prepared {len(track_coords)} tracks for bundling")
    
    # Apply bundling
    bundler = TrajectoryBundler(
        compatibility_threshold=0.6,
        attraction_strength=0.4,
        iterations=5
    )
    bundled = bundler.bundle(track_coords, track_ids)
    logger.info(f"Bundled {len(bundled)} tracks")
    
    # Step 3: Generate voxel grid from bundled tracks
    logger.info("\nStep 3: Generating voxel density grid")
    
    # Prepare tracks for isosurface extraction (x, y, z)
    tracks_xyz = []
    for bt in bundled:
        # Add z-coordinate based on sequence for 3D sculpture effect
        z = np.linspace(0, 100, len(bt.x))  # Height represents time progression
        tracks_xyz.append((bt.x, bt.y, z))
    
    logger.info(f"Prepared {len(tracks_xyz)} tracks for voxelization")
    
    # Create isosurface extractor and generate density grid
    resolution = 64  # Lower resolution for faster testing
    extractor = IsosurfaceExtractor(
        resolution=resolution,
        line_radius=3.0,
        smoothing_sigma=2.0
    )
    
    density_grid, grid_info = extractor.create_density_grid(tracks_xyz)
    
    logger.info(f"Voxel grid shape: {density_grid.shape}")
    logger.info(f"Max density: {density_grid.max():.3f}")
    
    # Step 4: Extract isosurface
    logger.info("\nStep 4: Extracting isosurface mesh")
    
    # Find good iso value (percentile of non-zero values)
    non_zero = density_grid[density_grid > 0]
    if len(non_zero) > 0:
        iso_value = np.percentile(non_zero, 30)
    else:
        iso_value = 0.01
    
    logger.info(f"Using iso value: {iso_value:.4f}")
    
    # Normalize iso_value to 0-1 range for the threshold
    threshold = iso_value / density_grid.max() if density_grid.max() > 0 else 0.3
    
    mesh_result = extractor.extract_surface(density_grid, grid_info, threshold=threshold)
    
    if mesh_result is None or mesh_result.n_vertices == 0:
        logger.warning("No isosurface generated. Trying lower threshold...")
        threshold = threshold / 2
        mesh_result = extractor.extract_surface(density_grid, grid_info, threshold=threshold)
    
    if mesh_result is None:
        logger.error("Failed to generate mesh")
        return 1
        
    vertices = mesh_result.vertices
    faces = mesh_result.faces
    
    logger.info(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Step 5: Create trimesh and export
    logger.info("\nStep 5: Exporting to GLB")
    
    import trimesh
    
    if len(vertices) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Center and normalize
        mesh.vertices -= mesh.centroid
        scale = 1.0 / mesh.bounding_box.extents.max()
        mesh.vertices *= scale
        
        logger.info(f"Final mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
        logger.info(f"Bounds: {mesh.bounds}")
        
        # Export
        output_dir = project_root / "output" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "test_whale_sculpture.glb"
        mesh.export(str(output_path))
        logger.info(f"\nExported to: {output_path}")
        
        # Also export OBJ for compatibility
        obj_path = output_dir / "test_whale_sculpture.obj"
        mesh.export(str(obj_path))
        logger.info(f"Also exported OBJ: {obj_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS! 3D model generated.")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("Failed to generate mesh - no geometry produced")
        return 1


if __name__ == "__main__":
    sys.exit(main())
