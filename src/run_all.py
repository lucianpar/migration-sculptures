#!/usr/bin/env python3
"""
Migration Sculptures - Orchestrator

Run sculpture generation modules on migration track data.

CURRENT DEFAULT: H3 (SDF Ridge Shell)
- Produces thin winding ridges on a spherical shell
- Anti-blob design for topographic sculptural forms

DEPRECATED MODULES (still available but not recommended):
- A: Connective Tension (tube connections)
- B: Subtractive Volume (point cloud carving)
- C: Constrained Membrane (convex hull)
- D: Carved Membrane (hybrid B+C)
- E: Refined Carved Specimen (PCA hull + density carving)
- F: Hull-Constrained Carving (E hull minus D corridors)
- G: Spherical Migration (density-modulated sphere)

LEGACY RENDER MODULES (removed):
- module_F_specimen_render (Three.js render pipeline) - superseded by H3

Usage:
    # Default: H3 module
    python src/run_all.py --data data/subsets/subset_full.csv
    
    # Explicit module selection
    python src/run_all.py --data data/subsets/subset_full.csv --modules H3
    
    # High resolution
    python src/run_all.py --data data/subsets/subset_full.csv --resolution 192
    
    # Legacy modules (deprecated)
    python src/run_all.py --data data/subsets/subset_full.csv --modules E F G
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from common.config import Config, UnitMode
from common.io import load_tracks, save_mesh, TrackData

logger = logging.getLogger(__name__)


def find_specimen_files(data_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """
    Find specimen data files.
    
    Looks for:
    - data/specimens/*/tracks.parquet
    - data/raw/**/*.csv
    - data/raw/**/*.parquet
    """
    files = []
    
    # Check specimens directory
    specimens_dir = data_dir / "specimens"
    if specimens_dir.exists():
        for specimen_dir in specimens_dir.iterdir():
            if specimen_dir.is_dir():
                tracks_file = specimen_dir / "tracks.parquet"
                if tracks_file.exists():
                    files.append(tracks_file)
    
    # Check raw directory
    raw_dir = data_dir / "raw"
    if raw_dir.exists():
        files.extend(raw_dir.glob("**/*.csv"))
        files.extend(raw_dir.glob("**/*.parquet"))
    
    # Filter out non-track files
    files = [f for f in files if 'track' in f.name.lower() or 'whale' in f.name.lower()]
    
    if limit:
        files = files[:limit]
    
    logger.info(f"Found {len(files)} specimen files")
    return files


def run_module_a(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option A: Connective Tension. [DEPRECATED]"""
    from _archive.experimental.option_A_connective_tension.build import build_connective_tension
    
    mesh, metadata = build_connective_tension(track_data, config)
    
    # Save
    output_path = output_dir / "option_A_connective_tension" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_b(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option B: Subtractive Volume. [DEPRECATED]"""
    from _archive.experimental.option_B_subtractive_volume.build import build_subtractive_volume
    
    mesh, metadata = build_subtractive_volume(track_data, config)
    
    # Save
    output_path = output_dir / "option_B_subtractive_volume" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_c(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option C: Constrained Membrane. [DEPRECATED]"""
    from _archive.experimental.option_C_constrained_membrane.build import build_constrained_membrane
    
    mesh, metadata = build_constrained_membrane(track_data, config)
    
    # Save
    output_path = output_dir / "option_C_constrained_membrane" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_d(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option D: Carved Membrane (Hybrid B+C). [DEPRECATED]"""
    from _archive.experimental.option_D_carved_membrane.build import build_carved_membrane
    
    mesh, metadata = build_carved_membrane(track_data, config)
    
    # Save
    output_path = output_dir / "option_D_carved_membrane" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_e(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option E: Refined Carved Specimen (Hero Module). [DEPRECATED]"""
    from _archive.experimental.option_E_refined_specimen.build import build_refined_specimen
    
    mesh, metadata, _ = build_refined_specimen(track_data, config)
    
    # Save
    output_path = output_dir / "option_E_refined_specimen" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_f(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option F: Hull-Constrained Organic Carving. [DEPRECATED]"""
    from _archive.experimental.option_F_hull_carve.build import build_hull_carve
    
    mesh, metadata, _ = build_hull_carve(track_data, config)
    
    # Save
    output_path = output_dir / "option_F_hull_carve" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_g(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option G: Spherical Migration Sculpture. [DEPRECATED]"""
    from _archive.experimental.option_G_spherical.build import build_spherical_sculpture
    
    mesh, metadata, _ = build_spherical_sculpture(track_data, config)
    
    # Save
    output_path = output_dir / "option_G_spherical" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_h3(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option H3: SDF Ridge Shell. [ACTIVE]"""
    from functional.H3.option_H3_ridge_shell.build import build_h3_from_tracks
    
    mesh, metadata, _ = build_h3_from_tracks(track_data, config)
    
    # Save to functional output location
    output_path = output_dir / "functional" / "H3" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_all(
    data_files: List[Path],
    modules: List[str],
    config: Config,
    output_dir: Path
) -> dict:
    """
    Run specified modules on all data files.
    
    Args:
        data_files: List of data file paths
        modules: List of module letters (A, B, C)
        config: Configuration
        output_dir: Output directory
        
    Returns:
        Summary dictionary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "modules": modules,
        "specimens": [],
        "errors": []
    }
    
    module_runners = {
        'A': run_module_a,
        'B': run_module_b,
        'C': run_module_c,
        'D': run_module_d,
        'E': run_module_e,
        'F': run_module_f,
        'G': run_module_g,
        'H3': run_module_h3
    }
    
    for data_file in data_files:
        specimen_id = data_file.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {specimen_id}")
        logger.info(f"{'='*60}")
        
        specimen_results = {
            "specimen_id": specimen_id,
            "data_file": str(data_file),
            "modules": {}
        }
        
        try:
            # Load track data
            track_data = load_tracks(
                data_file,
                specimen_id=specimen_id,
                utm_zone=config.utm_zone,
                utm_hemisphere=config.utm_hemisphere
            )
            
            # Run each module
            for module in modules:
                module = module.upper()
                if module not in module_runners:
                    logger.warning(f"Unknown module: {module}")
                    continue
                
                logger.info(f"\n--- Module {module} ---")
                try:
                    result = module_runners[module](track_data, config, output_dir)
                    specimen_results["modules"][module] = {
                        "status": "success",
                        "metadata": result
                    }
                except Exception as e:
                    logger.error(f"Module {module} failed: {e}")
                    specimen_results["modules"][module] = {
                        "status": "error",
                        "error": str(e)
                    }
                    summary["errors"].append({
                        "specimen": specimen_id,
                        "module": module,
                        "error": str(e)
                    })
        
        except Exception as e:
            logger.error(f"Failed to load {specimen_id}: {e}")
            specimen_results["error"] = str(e)
            summary["errors"].append({
                "specimen": specimen_id,
                "module": "load",
                "error": str(e)
            })
        
        summary["specimens"].append(specimen_results)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Migration Sculptures - Run sculpture generation modules"
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        help="Specific data file to process"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory to search for specimens"
    )
    parser.add_argument(
        "--specimens", "-n",
        type=int,
        default=None,
        help="Maximum number of specimens to process"
    )
    parser.add_argument(
        "--modules", "-m",
        nargs="+",
        default=["H3"],
        help="Modules to run. Default: H3. Available: H3 (recommended), A, B, C, D, E, F, G (deprecated)"
    )
    parser.add_argument(
        "--unit-mode", "-u",
        choices=["normalized", "meters"],
        default="normalized",
        help="Output unit mode"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("outputs"),
        help="Output directory"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=128,
        help="Voxel grid resolution"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Build config
    config = Config(
        unit_mode=UnitMode(args.unit_mode),
        voxel_resolution=args.resolution,
        output_dir=args.output
    )
    
    # Find data files
    if args.data:
        data_files = [args.data]
    else:
        data_files = find_specimen_files(args.data_dir, args.specimens)
    
    if not data_files:
        logger.error("No data files found!")
        sys.exit(1)
    
    # Run
    logger.info(f"Processing {len(data_files)} specimens with modules {args.modules}")
    logger.info(f"Unit mode: {config.unit_mode.value}")
    logger.info(f"Output: {args.output}")
    
    summary = run_all(
        data_files=data_files,
        modules=args.modules,
        config=config,
        output_dir=args.output
    )
    
    # Save summary
    summary_path = args.output / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Print summary
    n_success = sum(
        1 for s in summary["specimens"]
        for m in s.get("modules", {}).values()
        if m.get("status") == "success"
    )
    n_errors = len(summary["errors"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {n_success} successful, {n_errors} errors")
    logger.info(f"{'='*60}")
    
    if n_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
