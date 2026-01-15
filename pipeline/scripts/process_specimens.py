#!/usr/bin/env python3
"""
Main specimen processing script.

Processes raw whale tracking data into 3D sculpture meshes:
1. Load and parse raw tracking data
2. Filter by species, season, and year
3. Transform coordinates to UTM
4. Apply trajectory bundling
5. Generate 3D isosurface mesh
6. Compute metrics
7. Export as glTF

Usage:
    python process_specimens.py --species blue_whale --season spring --years 2010,2012,2014,2016,2018
    python process_specimens.py --all-initial  # Process all 10 initial specimens
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process whale tracking data into 3D sculpture meshes"
    )
    
    parser.add_argument(
        "--species",
        choices=["blue_whale", "fin_whale", "gray_whale", "humpback_whale"],
        help="Species to process"
    )
    
    parser.add_argument(
        "--season",
        choices=["spring", "fall"],
        help="Season to process"
    )
    
    parser.add_argument(
        "--years",
        type=str,
        help="Comma-separated list of years (e.g., 2010,2012,2014)"
    )
    
    parser.add_argument(
        "--all-initial",
        action="store_true",
        help="Process all 10 initial target specimens"
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input data directory (default: data/raw)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for models (default: output/models)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Voxel grid resolution (default: 128)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    return parser.parse_args()


def get_initial_specimens():
    """Get the list of 10 initial target specimens."""
    return [
        {"species": "blue_whale", "season": "spring", "year": 2010},
        {"species": "blue_whale", "season": "spring", "year": 2012},
        {"species": "blue_whale", "season": "spring", "year": 2014},
        {"species": "blue_whale", "season": "spring", "year": 2016},
        {"species": "blue_whale", "season": "spring", "year": 2018},
        {"species": "blue_whale", "season": "fall", "year": 2010},
        {"species": "blue_whale", "season": "fall", "year": 2012},
        {"species": "blue_whale", "season": "fall", "year": 2014},
        {"species": "blue_whale", "season": "fall", "year": 2016},
        {"species": "blue_whale", "season": "fall", "year": 2018},
    ]


def process_specimen(
    species: str,
    season: str,
    year: int,
    input_dir: Path,
    output_dir: Path,
    resolution: int = 128,
    dry_run: bool = False
):
    """
    Process a single specimen.
    
    Args:
        species: Species name
        season: Season name
        year: Year
        input_dir: Directory containing raw data
        output_dir: Output directory for meshes
        resolution: Voxel grid resolution
        dry_run: If True, don't actually process
    """
    logger.info(f"Processing: {species} / {season} / {year}")
    
    if dry_run:
        logger.info("  [DRY RUN] Would process this specimen")
        return
    
    try:
        # Import modules (deferred to allow --dry-run without deps)
        import pandas as pd
        import numpy as np
        import yaml
        
        from processing.coordinate_transform import CoordinateTransformer
        from processing.track_processor import TrackProcessor, Specimen
        from processing.trajectory_bundler import TrajectoryBundler
        from geometry.mesh_generator import MeshGenerator, SculptureMetadata
        from geometry.gltf_exporter import GLTFExporter
        from metrics.compute_metrics import MetricsCalculator
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install dependencies: pip install -r requirements.txt")
        return
    
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load raw data
    data_file = input_dir / "movebank" / "blue_fin_whale_tracks.csv"
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.error("Please run download_movebank_data.py first")
        return
    
    logger.info(f"  Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Initialize processors
    transformer = CoordinateTransformer.for_santa_barbara_channel()
    track_processor = TrackProcessor()
    bundler = TrajectoryBundler()
    mesh_gen = MeshGenerator(resolution=resolution)
    metrics_calc = MetricsCalculator()
    exporter = GLTFExporter()
    
    # Parse tracks
    tracks = track_processor.parse_movebank_data(df)
    logger.info(f"  Parsed {len(tracks)} tracks")
    
    # Filter by species
    tracks = [t for t in tracks if t.species == species]
    logger.info(f"  {len(tracks)} tracks for {species}")
    
    # Filter by region (Santa Barbara Channel)
    bounds = config["region"]["bounds"]
    tracks = track_processor.filter_by_bounds(
        tracks,
        bounds["min_lon"], bounds["max_lon"],
        bounds["min_lat"], bounds["max_lat"]
    )
    
    # Filter by season and year
    season_config = config["species"][species]["seasons"][season]
    start_date = datetime(year, season_config["start_month"], season_config["start_day"])
    end_date = datetime(year, season_config["end_month"], season_config["end_day"])
    
    tracks = track_processor.filter_by_date_range(tracks, start_date, end_date)
    logger.info(f"  {len(tracks)} tracks in {season} {year}")
    
    if len(tracks) < config["species"][species].get("min_tracks_per_specimen", 5):
        logger.warning(f"  Insufficient tracks ({len(tracks)}). Skipping.")
        return
    
    # Transform coordinates
    for track in tracks:
        track_processor.compute_derived_fields(track, transformer)
    
    # Remove outliers
    tracks = [track_processor.remove_outlier_points(t) for t in tracks]
    
    # Resample
    tracks = [track_processor.resample_track(t) for t in tracks]
    
    # Bundle trajectories
    track_coords = [(t.x, t.y) for t in tracks if t.x is not None]
    track_ids = [t.track_id for t in tracks if t.x is not None]
    
    bundled = bundler.bundle(track_coords, track_ids)
    bundled = bundler.filter_outlier_tracks(bundled)
    
    logger.info(f"  Bundled {len(bundled)} tracks")
    
    # Generate mesh
    metadata = SculptureMetadata(
        species=species,
        season=season,
        year=year,
        n_tracks=len(bundled)
    )
    
    mesh, mesh_meta = mesh_gen.generate_bundle_sculpture(bundled, metadata)
    logger.info(f"  Generated mesh: {mesh.n_vertices} vertices, {mesh.n_faces} faces")
    
    # Compute metrics
    metrics = metrics_calc.calculate_all(bundled)
    mesh_meta["metrics"] = metrics.to_dict()
    
    logger.info(f"  Metrics: coherence={metrics.coherence:.3f}, entropy={metrics.entropy:.3f}")
    
    # Export
    output_path = exporter.export_specimen(
        mesh,
        species, season, year,
        output_dir,
        mesh_meta
    )
    
    logger.info(f"  Exported to: {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine directories
    project_root = Path(__file__).parent.parent.parent
    input_dir = args.input_dir or (project_root / "data" / "raw")
    output_dir = args.output_dir or (project_root / "output" / "models")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Migration Sculptures - Specimen Processing")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Resolution: {args.resolution}")
    
    if args.dry_run:
        logger.info("MODE: DRY RUN")
    logger.info("")
    
    # Get specimens to process
    if args.all_initial:
        specimens = get_initial_specimens()
    elif args.species and args.season and args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
        specimens = [
            {"species": args.species, "season": args.season, "year": y}
            for y in years
        ]
    else:
        logger.error("Please specify --all-initial OR --species/--season/--years")
        return 1
    
    logger.info(f"Processing {len(specimens)} specimens:")
    for spec in specimens:
        logger.info(f"  - {spec['species']} / {spec['season']} / {spec['year']}")
    logger.info("")
    
    # Process each specimen
    success_count = 0
    for spec in specimens:
        try:
            process_specimen(
                spec["species"],
                spec["season"],
                spec["year"],
                input_dir,
                output_dir,
                args.resolution,
                args.dry_run
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {spec}: {e}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Completed: {success_count}/{len(specimens)} specimens")
    logger.info("=" * 60)
    
    return 0 if success_count == len(specimens) else 1


if __name__ == "__main__":
    sys.exit(main())
