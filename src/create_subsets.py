#!/usr/bin/env python3
"""
Data Subset Utility

Creates subsets of tracking data for testing different data volumes.
Splits by individual whale/specimen.
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_subsets(input_file: Path, output_dir: Path) -> dict:
    """
    Create data subsets from tracking data.
    
    Creates three subsets:
    - single: 1 whale with most data points
    - small: 3 whales
    - full: all whales
    
    Args:
        input_file: Path to input CSV
        output_dir: Directory for output files
        
    Returns:
        Dict mapping subset name to file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Get individual column (try common names)
    id_col = None
    for col in ['individual-local-identifier', 'individual_id', 'whale_id', 'id']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"Could not find individual ID column in: {df.columns.tolist()}")
    
    # Count points per individual
    counts = df[id_col].value_counts()
    logger.info(f"Found {len(counts)} individuals:")
    for ind, count in counts.items():
        logger.info(f"  {ind}: {count} points")
    
    subsets = {}
    
    # 1. Single whale (most data points)
    single_whale = counts.index[0]
    df_single = df[df[id_col] == single_whale]
    single_path = output_dir / "subset_single_whale.csv"
    df_single.to_csv(single_path, index=False)
    subsets['single'] = single_path
    logger.info(f"Created single whale subset: {len(df_single)} points ({single_whale})")
    
    # 2. Three whales (top 3 by data points)
    three_whales = counts.index[:3].tolist()
    df_three = df[df[id_col].isin(three_whales)]
    three_path = output_dir / "subset_three_whales.csv"
    df_three.to_csv(three_path, index=False)
    subsets['three'] = three_path
    logger.info(f"Created three whale subset: {len(df_three)} points ({three_whales})")
    
    # 3. Full dataset (just copy/link)
    full_path = output_dir / "subset_full.csv"
    df.to_csv(full_path, index=False)
    subsets['full'] = full_path
    logger.info(f"Created full subset: {len(df)} points (all {len(counts)} whales)")
    
    return subsets


def main():
    parser = argparse.ArgumentParser(description="Create data subsets for testing")
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="Input tracking data CSV")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/subsets"),
                       help="Output directory for subsets")
    
    args = parser.parse_args()
    
    subsets = create_subsets(args.input, args.output)
    
    print("\n" + "=" * 50)
    print("Created subsets:")
    for name, path in subsets.items():
        print(f"  {name}: {path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
