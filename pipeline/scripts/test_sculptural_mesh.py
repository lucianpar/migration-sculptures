#!/usr/bin/env python3
"""
Test script for sculptural mesh generation from whale track data.

Creates 3D sculptural forms using:
- Time-based z-axis (temporal flow)
- Ribbon extrusion perpendicular to movement
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
    logger.info("Migration Sculptures - Sculptural Mesh Generation")
    logger.info("=" * 60)
    
    # Load the tracking data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / "data" / "raw" / "movebank" / "blue_fin_whale_tracks.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return 1
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} points")
    
    # Import pipeline modules
    from processing.coordinate_transform import CoordinateTransformer
    from processing.trajectory_bundler import TrajectoryBundler
    from geometry.sculptural_transform import SculpturalTransform, create_sculptural_mesh
    from geometry.isosurface import IsosurfaceExtractor
    
    # Step 1: Transform coordinates to UTM
    logger.info("\n" + "=" * 40)
    logger.info("Step 1: Coordinate transformation")
    logger.info("=" * 40)
    transformer = CoordinateTransformer.for_santa_barbara_channel()
    
    # Parse tracks by individual, keeping timestamps
    tracks = []
    timestamps = []
    track_ids = []
    
    for ind_id, group in df.groupby('individual-local-identifier'):
        group = group.sort_values('timestamp')
        lons = group['location-long'].values
        lats = group['location-lat'].values
        times = group['timestamp'].values
        
        # Transform to UTM
        x, y = transformer.to_utm(lons, lats)
        
        tracks.append((x, y))
        timestamps.append(times.astype('datetime64[s]').astype(float))
        track_ids.append(str(ind_id))
    
    logger.info(f"Transformed {len(tracks)} whale tracks to UTM")
    
    # Log spatial extent
    all_x = np.concatenate([t[0] for t in tracks])
    all_y = np.concatenate([t[1] for t in tracks])
    logger.info(f"X range: {all_x.min():.0f} to {all_x.max():.0f} ({np.ptp(all_x):.0f}m)")
    logger.info(f"Y range: {all_y.min():.0f} to {all_y.max():.0f} ({np.ptp(all_y):.0f}m)")
    
    # Step 2: Trajectory bundling
    logger.info("\n" + "=" * 40)
    logger.info("Step 2: Trajectory bundling")
    logger.info("=" * 40)
    
    bundler = TrajectoryBundler(
        compatibility_threshold=0.5,
        attraction_strength=0.3,
        iterations=8
    )
    
    bundled = bundler.bundle(tracks, track_ids)
    logger.info(f"Bundled {len(bundled)} tracks")
    
    # Convert bundled tracks back to list format
    bundled_tracks = [(bt.x, bt.y) for bt in bundled]
    bundled_ids = [bt.track_id for bt in bundled]
    
    # Step 3: Sculptural transformation (time as Z + ribbon extrusion)
    logger.info("\n" + "=" * 40)
    logger.info("Step 3: Sculptural transformation")
    logger.info("=" * 40)
    
    sculpture_transform = SculpturalTransform(
        z_scale=0.7,           # Z height relative to XY extent
        base_ribbon_width=1500.0,  # Ribbon width in meters
        width_variation=0.4,    # Width varies with speed
        smooth_normals=True
    )
    
    ribbons = sculpture_transform.transform_tracks(
        bundled_tracks,
        timestamps=timestamps,
        track_ids=bundled_ids
    )
    
    # Log ribbon info
    for ribbon in ribbons[:3]:  # First 3
        logger.info(f"  {ribbon.track_id}: {len(ribbon.x)} points, "
                   f"z range: {ribbon.z.min():.0f} - {ribbon.z.max():.0f}m")
    
    # Step 4: Convert to voxel grid and extract isosurface
    logger.info("\n" + "=" * 40)
    logger.info("Step 4: Voxelization and isosurface extraction")
    logger.info("=" * 40)
    
    # Convert ribbons to dense point clouds
    tracks_xyz = sculpture_transform.ribbons_to_mesh_points(
        ribbons,
        add_thickness=True,
        thickness=800.0  # Vertical thickness
    )
    
    total_points = sum(len(t[0]) for t in tracks_xyz)
    logger.info(f"Generated {total_points} points from ribbons")
    
    # Create isosurface extractor
    resolution = 80  # Higher resolution for better detail
    extractor = IsosurfaceExtractor(
        resolution=resolution,
        line_radius=3.0,
        smoothing_sigma=1.8
    )
    
    density_grid, grid_info = extractor.create_density_grid(tracks_xyz)
    logger.info(f"Voxel grid: {density_grid.shape}, max density: {density_grid.max():.2f}")
    
    # Apply Gaussian smoothing for organic form
    density_grid = extractor.smooth_grid(density_grid)
    logger.info(f"After smoothing: max density: {density_grid.max():.2f}")
    
    # Extract isosurface
    threshold = 0.12  # Lower threshold for more volume
    mesh = extractor.extract_surface(density_grid, grid_info, threshold=threshold)
    
    if mesh is None or mesh.n_vertices == 0:
        logger.warning("First extraction failed, trying lower threshold...")
        threshold = 0.06
        mesh = extractor.extract_surface(density_grid, grid_info, threshold=threshold)
    
    if mesh is None or mesh.n_vertices == 0:
        logger.error("Failed to extract isosurface")
        return 1
    
    logger.info(f"Extracted mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
    
    # Step 5: Post-process and export
    logger.info("\n" + "=" * 40)
    logger.info("Step 5: Export")
    logger.info("=" * 40)
    
    import trimesh
    
    # Create trimesh object
    sculpture = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
    
    # Center the mesh
    sculpture.vertices -= sculpture.centroid
    
    # Normalize to unit scale (longest axis = 1)
    scale = 1.0 / sculpture.bounding_box.extents.max()
    sculpture.vertices *= scale
    
    # Log final dimensions
    extents = sculpture.bounding_box.extents
    logger.info(f"Final dimensions: {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f}")
    logger.info(f"Aspect ratio: 1 : {extents[1]/extents[0]:.2f} : {extents[2]/extents[0]:.2f}")
    
    # Export
    output_dir = project_root / "output" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GLB format (for web/Three.js)
    glb_path = output_dir / "whale_sculpture_v2.glb"
    sculpture.export(str(glb_path))
    logger.info(f"Exported GLB: {glb_path}")
    
    # OBJ format (for Blender/other)
    obj_path = output_dir / "whale_sculpture_v2.obj"
    sculpture.export(str(obj_path))
    logger.info(f"Exported OBJ: {obj_path}")
    
    # Also export the direct ribbon mesh for comparison
    logger.info("\n" + "=" * 40)
    logger.info("Bonus: Direct ribbon mesh (no voxelization)")
    logger.info("=" * 40)
    
    ribbon_verts, ribbon_faces = sculpture_transform.ribbons_to_direct_mesh(ribbons)
    ribbon_mesh = trimesh.Trimesh(vertices=ribbon_verts, faces=ribbon_faces)
    ribbon_mesh.vertices -= ribbon_mesh.centroid
    ribbon_mesh.vertices *= 1.0 / ribbon_mesh.bounding_box.extents.max()
    
    ribbon_path = output_dir / "whale_ribbons_v2.glb"
    ribbon_mesh.export(str(ribbon_path))
    logger.info(f"Exported ribbon mesh: {ribbon_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS! Sculptural models generated.")
    logger.info("=" * 60)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {glb_path} (voxelized, organic)")
    logger.info(f"  - {obj_path}")  
    logger.info(f"  - {ribbon_path} (direct ribbons)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
