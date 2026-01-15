"""
OBIS-SEAMAP Data Acquisition Module

Downloads whale sighting and tracking data from OBIS-SEAMAP
(Ocean Biogeographic Information System - Spatial Ecological Analysis of Megavertebrate Populations)
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import json

logger = logging.getLogger(__name__)


class OBISSeamapClient:
    """Client for accessing OBIS-SEAMAP marine animal tracking database."""
    
    BASE_URL = "https://seamap.env.duke.edu"
    API_URL = f"{BASE_URL}/api"
    
    # Known datasets relevant to Santa Barbara Channel whales
    DATASETS = {
        "gray_whales_count": {
            "id": "1899b433-f6e2-40e4-827c-94d6f59785d3",
            "name": "Gray Whales Count annual survey 2011-2012",
            "description": "Northbound gray whale migration in Santa Barbara Channel",
            "url": "https://obis.org/dataset/1899b433-f6e2-40e4-827c-94d6f59785d3"
        }
    }
    
    # Species taxonomy IDs
    SPECIES_CODES = {
        "blue_whale": 137090,       # Balaenoptera musculus
        "fin_whale": 137091,        # Balaenoptera physalus
        "gray_whale": 137095,       # Eschrichtius robustus
        "humpback_whale": 137092,   # Megaptera novaeangliae
    }
    
    def __init__(self):
        """Initialize OBIS-SEAMAP client."""
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MigrationSculptures/0.1"
        })
    
    def search_datasets(
        self,
        species: Optional[str] = None,
        bbox: Optional[tuple] = None,
        keyword: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for available datasets.
        
        Args:
            species: Species common name (e.g., "blue_whale")
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            keyword: Search keyword
            
        Returns:
            List of matching dataset metadata
        """
        params = {}
        
        if species and species in self.SPECIES_CODES:
            params["sp"] = self.SPECIES_CODES[species]
        
        if bbox:
            params["xmin"], params["ymin"], params["xmax"], params["ymax"] = bbox
        
        if keyword:
            params["q"] = keyword
        
        url = f"{self.API_URL}/dataset/list"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a dataset.
        
        Args:
            dataset_id: OBIS-SEAMAP dataset identifier
            
        Returns:
            Dataset metadata dictionary
        """
        url = f"{self.API_URL}/dataset/{dataset_id}"
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def download_dataset(
        self,
        dataset_id: str,
        output_dir: Path,
        format: str = "csv"
    ) -> Path:
        """
        Download a dataset.
        
        Args:
            dataset_id: OBIS-SEAMAP dataset identifier
            output_dir: Directory to save the downloaded file
            format: Output format (csv, shapefile, kml)
            
        Returns:
            Path to downloaded file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # OBIS-SEAMAP download endpoint
        url = f"{self.BASE_URL}/dataset/{dataset_id}/download"
        params = {"format": format}
        
        logger.info(f"Downloading dataset {dataset_id}...")
        response = self.session.get(url, params=params, stream=True)
        response.raise_for_status()
        
        ext = {"csv": "csv", "shapefile": "zip", "kml": "kml"}.get(format, "dat")
        output_file = output_dir / f"obis_seamap_{dataset_id}.{ext}"
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded to {output_file}")
        return output_file
    
    def query_occurrences(
        self,
        species: str,
        bbox: tuple,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Query occurrence records (sightings/tracking points).
        
        Args:
            species: Species common name
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum records to return
            
        Returns:
            List of occurrence records
        """
        if species not in self.SPECIES_CODES:
            raise ValueError(f"Unknown species: {species}")
        
        params = {
            "sp": self.SPECIES_CODES[species],
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
            "limit": limit
        }
        
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        url = f"{self.API_URL}/occurrence/search"
        
        logger.info(f"Querying occurrences for {species} in bbox {bbox}...")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        logger.info(f"Found {len(data)} occurrence records")
        
        return data


def download_santa_barbara_whale_data(
    output_dir: Path,
    species: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Download whale data for Santa Barbara Channel region from OBIS-SEAMAP.
    
    Args:
        output_dir: Directory to save data
        species: List of species to download (default: all available)
        
    Returns:
        Dictionary mapping species to downloaded file paths
    """
    client = OBISSeamapClient()
    
    # Santa Barbara Channel bounding box
    bbox = (-121.0, 33.5, -119.0, 35.0)
    
    species_list = species or ["blue_whale", "fin_whale", "gray_whale", "humpback_whale"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for sp in species_list:
        try:
            logger.info(f"Fetching {sp} data...")
            occurrences = client.query_occurrences(sp, bbox)
            
            if occurrences:
                output_file = output_dir / f"obis_seamap_{sp}.json"
                with open(output_file, 'w') as f:
                    json.dump(occurrences, f, indent=2)
                results[sp] = output_file
                logger.info(f"Saved {len(occurrences)} records to {output_file}")
            else:
                logger.warning(f"No occurrences found for {sp}")
                
        except Exception as e:
            logger.error(f"Error fetching {sp} data: {e}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "raw"
    download_santa_barbara_whale_data(output_dir)
