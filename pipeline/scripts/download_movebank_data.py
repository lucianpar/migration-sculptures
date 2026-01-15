#!/usr/bin/env python3
"""
Download whale tracking data from Movebank Data Repository.

Primary dataset: Blue and Fin Whale Satellite Tracks off California
Source: Irvine et al. (2019) via Movebank
DOI: 10.5441/001/1.47h576f2

This script attempts multiple download methods:
1. Direct API access (requires Movebank account for some datasets)
2. Published dataset download via DOI
3. Provides manual download instructions as fallback
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acquisition.movebank_client import MovebankClient, download_blue_fin_whale_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download whale tracking data."""
    # Determine output directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw" / "movebank"
    
    logger.info("=" * 60)
    logger.info("Migration Sculptures - Movebank Data Download")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Target Dataset:")
    logger.info("  Blue and Fin Whale Satellite Tracks off California")
    logger.info("  DOI: 10.5441/001/1.47h576f2")
    logger.info("  Source: Irvine et al. (2019)")
    logger.info("")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    try:
        output_path = download_blue_fin_whale_data(output_dir)
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"Data saved to: {output_path}")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("AUTOMATIC DOWNLOAD FAILED")
        logger.error("=" * 60)
        logger.error("")
        logger.error(str(e))
        logger.error("")
        logger.error("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.error("-" * 40)
        logger.error("1. Visit: https://www.movebank.org/cms/webapp")
        logger.error("2. Search for study: 'Blue whale  Argos  California'")
        logger.error("   Or use DOI: 10.5441/001/1.47h576f2")
        logger.error("3. Create a free Movebank account if needed")
        logger.error("4. Download the tracking data as CSV")
        logger.error(f"5. Save to: {output_dir / 'blue_fin_whale_tracks.csv'}")
        logger.error("")
        logger.error("Alternative: Visit the data repository directly:")
        logger.error("  https://www.datarepository.movebank.org/")
        logger.error("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
