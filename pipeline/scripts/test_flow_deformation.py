#!/usr/bin/env python3
"""
Test script for flow field deformation of whale sculptures.

Applies ocean current-like deformations to create coherent
sculptural aesthetics across specimens.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_flow_field():
    """Generate a 2D visualization of the flow field (optional debug)."""
    from geometry.flow_field import create_ocean_current_field
    
    try:
        import matplotlib.pyplot as plt
        
        field = create_ocean_current_field()
        
        # Sample on 2D grid
        n = 20
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        vx, vy, vz = field.sample(X.flatten(), Y.flatten(), Z.flatten())
        VX = vx.reshape(X.shape)
        VY = vy.reshape(Y.shape)
        
        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, VX, VY, scale=15)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ocean Current Flow Field (Z=0)')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        output_path = Path(__file__).parent.parent.parent / "output" / "flow_field_viz.png"
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved flow field visualization: {output_path}")
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")


def test_deformation_strengths():
    """Test different deformation strengths to find optimal value."""
    import trimesh
    from geometry.flow_field import create_ocean_current_field
    from geometry.mesh_deform import MeshDeformer, DeformationConfig
    
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "output" / "models" / "whale_sculpture_v2.glb"
    output_dir = project_root / "output" / "models" / "deformed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input model not found: {input_path}")
        logger.error("Run test_sculptural_mesh.py first to generate the base sculpture")
        return False
    
    logger.info(f"Loading base sculpture: {input_path}")
    mesh = trimesh.load(input_path)
    
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    
    logger.info(f"Loaded mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
    
    # Create flow field
    flow_field = create_ocean_current_field()
    
    # Test different strengths
    strengths = [0.02, 0.05, 0.08, 0.12, 0.15]
    
    for strength in strengths:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing strength = {strength}")
        logger.info(f"{'='*50}")
        
        # Configure deformation
        config = DeformationConfig(
            strength=strength,
            falloff_sigma=1.0,
            falloff_center=(0.0, 0.0, 0.0),
            z_damping=0.3,
            z_damping_center=0.0,
            clamp_displacement=0.2,
            target_scale=1.5
        )
        
        # Deform a copy
        deformer = MeshDeformer(flow_field, config)
        deformed = deformer.deform(mesh.copy(), normalize=True)
        
        # Analyze deformation
        original_verts = mesh.vertices.copy()
        # Re-normalize original for comparison
        original_verts -= original_verts.mean(axis=0)
        scale = 1.5 / np.ptp(original_verts, axis=0).max()
        original_verts *= scale
        
        # Compute displacement statistics
        if len(original_verts) == len(deformed.vertices):
            displacements = np.linalg.norm(deformed.vertices - original_verts, axis=1)
            logger.info(f"  Displacement - min: {displacements.min():.4f}, "
                       f"max: {displacements.max():.4f}, mean: {displacements.mean():.4f}")
        
        # Export
        output_path = output_dir / f"whale_deformed_s{int(strength*100):02d}.glb"
        deformed.export(str(output_path))
        logger.info(f"  Exported: {output_path}")
        
        # Also export OBJ
        obj_path = output_dir / f"whale_deformed_s{int(strength*100):02d}.obj"
        deformed.export(str(obj_path))
    
    return True


def create_final_deformed_sculpture():
    """Create the final deformed sculpture with optimal parameters."""
    import trimesh
    from geometry.flow_field import create_ocean_current_field
    from geometry.mesh_deform import MeshDeformer, DeformationConfig
    
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "output" / "models"
    
    # Deform both the voxelized and ribbon versions
    models = [
        ("whale_sculpture_v2.glb", "whale_sculpture_v2_deformed.glb"),
        ("whale_ribbons_v2.glb", "whale_ribbons_v2_deformed.glb"),
    ]
    
    flow_field = create_ocean_current_field()
    
    # Optimal configuration (tune these based on test results)
    config = DeformationConfig(
        strength=0.08,           # Moderate deformation
        falloff_sigma=1.2,       # Broad falloff
        falloff_center=(0.0, 0.0, 0.0),
        z_damping=0.2,           # Light Z damping
        z_damping_center=0.0,
        clamp_displacement=0.15,
        target_scale=1.5
    )
    
    deformer = MeshDeformer(flow_field, config)
    
    for input_name, output_name in models:
        input_path = output_dir / input_name
        output_path = output_dir / output_name
        
        if not input_path.exists():
            logger.warning(f"Skipping {input_name} - not found")
            continue
        
        logger.info(f"\nProcessing {input_name}")
        mesh = trimesh.load(str(input_path))
        
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        
        deformed = deformer.deform(mesh.copy(), normalize=True, iterations=2)
        
        deformed.export(str(output_path))
        logger.info(f"Saved: {output_path}")
        
        # Also OBJ
        obj_path = output_path.with_suffix('.obj')
        deformed.export(str(obj_path))
        logger.info(f"Saved: {obj_path}")


def export_flow_field_json():
    """Export the flow field as JSON for potential JavaScript use."""
    from geometry.flow_field import create_ocean_current_field
    
    project_root = Path(__file__).parent.parent.parent
    output_path = project_root / "output" / "flow_field_grid.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    field = create_ocean_current_field()
    field.to_json(str(output_path), resolution=32, bounds=(-1.5, 1.5))
    logger.info(f"Exported flow field JSON: {output_path}")


def main():
    logger.info("=" * 60)
    logger.info("Flow Field Deformation Test")
    logger.info("=" * 60)
    
    # Step 1: Visualize the flow field
    logger.info("\n>>> Step 1: Visualizing flow field")
    visualize_flow_field()
    
    # Step 2: Test different deformation strengths
    logger.info("\n>>> Step 2: Testing deformation strengths")
    if not test_deformation_strengths():
        return 1
    
    # Step 3: Create final deformed sculptures
    logger.info("\n>>> Step 3: Creating final deformed sculptures")
    create_final_deformed_sculpture()
    
    # Step 4: Export flow field for JavaScript
    logger.info("\n>>> Step 4: Exporting flow field JSON")
    export_flow_field_json()
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nGenerated files:")
    logger.info("  output/models/deformed/whale_deformed_s*.glb  (strength tests)")
    logger.info("  output/models/whale_sculpture_v2_deformed.glb (final)")
    logger.info("  output/models/whale_ribbons_v2_deformed.glb   (final)")
    logger.info("  output/flow_field_grid.json                   (for JS)")
    logger.info("  output/flow_field_viz.png                     (visualization)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
