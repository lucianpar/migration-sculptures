#!/usr/bin/env python3
"""
Migration Sculptures - Orchestrator

Run all sculpture generation modules on specimen data.

Usage:
    python src/run_all.py --specimens 10 --unit_mode normalized --modules A B C
    python src/run_all.py --data data/raw/movebank/blue_fin_whale_tracks.csv --modules A
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
    """Run Option A: Connective Tension."""
    from option_A_connective_tension.build import build_connective_tension
    
    mesh, metadata = build_connective_tension(track_data, config)
    
    # Save
    output_path = output_dir / "option_A_connective_tension" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_b(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option B: Subtractive Volume."""
    from option_B_subtractive_volume.build import build_subtractive_volume
    
    mesh, metadata = build_subtractive_volume(track_data, config)
    
    # Save
    output_path = output_dir / "option_B_subtractive_volume" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_c(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option C: Constrained Membrane."""
    from option_C_constrained_membrane.build import build_constrained_membrane
    
    mesh, metadata = build_constrained_membrane(track_data, config)
    
    # Save
    output_path = output_dir / "option_C_constrained_membrane" / "meshes" / f"{track_data.specimen_id}.glb"
    save_mesh(mesh, output_path, metadata)
    
    return metadata.to_dict()


def run_module_d(track_data: TrackData, config: Config, output_dir: Path) -> dict:
    """Run Option D: Carved Membrane (Hybrid B+C)."""
    from option_D_carved_membrane.build import build_carved_membrane
    
    mesh, metadata = build_carved_membrane(track_data, config)
    
    # Save
    output_path = output_dir / "option_D_carved_membrane" / "meshes" / f"{track_data.specimen_id}.glb"
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
        'D': run_module_d
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
        default=["A", "B", "C"],
        help="Modules to run (A, B, C)"
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
