#!/usr/bin/env python3
"""
Download whale occurrence data from OBIS-SEAMAP.

This script downloads supplementary whale sighting/tracking data
from OBIS-SEAMAP for the Santa Barbara Channel region.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acquisition.obis_seamap_client import OBISSeamapClient, download_santa_barbara_whale_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Download OBIS-SEAMAP whale data."""
    # Determine output directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw" / "obis_seamap"
    
    logger.info("=" * 60)
    logger.info("Migration Sculptures - OBIS-SEAMAP Data Download")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Downloading whale occurrence data for Santa Barbara Channel")
    logger.info("Species: Blue Whale, Fin Whale, Gray Whale, Humpback Whale")
    logger.info("")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    try:
        results = download_santa_barbara_whale_data(output_dir)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        
        for species, path in results.items():
            logger.info(f"  {species}: {path}")
        
        if not results:
            logger.warning("No data was downloaded. OBIS-SEAMAP API may be unavailable.")
            logger.info("")
            logger.info("Manual alternative:")
            logger.info("  Visit: https://seamap.env.duke.edu/")
            logger.info("  Search for whale species in Santa Barbara Channel region")
            
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("")
        logger.error("Manual download instructions:")
        logger.error("  1. Visit: https://seamap.env.duke.edu/")
        logger.error("  2. Use 'Explore Data' to search for whale species")
        logger.error("  3. Filter to Santa Barbara Channel region")
        logger.error("  4. Download as CSV or JSON")
        return 1


if __name__ == "__main__":
    sys.exit(main())
