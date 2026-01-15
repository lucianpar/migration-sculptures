"""
Movebank Data Acquisition Module

Downloads whale tracking data from Movebank Data Repository.
Primary dataset: Blue and Fin whale satellite tracks (Irvine et al. 2019)
DOI: 10.5441/001/1.47h576f2
"""

import os
import logging
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MovebankClient:
    """Client for accessing Movebank Data Repository."""
    
    # Movebank API endpoints
    BASE_URL = "https://www.movebank.org/movebank/service/direct-read"
    DOWNLOAD_URL = "https://www.datarepository.movebank.org/handle"
    
    # Known study IDs for whale datasets
    WHALE_STUDIES = {
        "blue_fin_whales_california": {
            "study_id": 1241071646,
            "doi": "10.5441/001/1.47h576f2",
            "description": "Blue and fin whale satellite tracks off California 1994-2018",
            "citation": "Irvine LM, Palacios DM, Urbán J, Mate BR (2019)"
        }
    }
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize Movebank client.
        
        Args:
            username: Movebank account username (optional for public datasets)
            password: Movebank account password
        """
        self.username = username or os.getenv("MOVEBANK_USERNAME")
        self.password = password or os.getenv("MOVEBANK_PASSWORD")
        self.session = requests.Session()
        
        if self.username and self.password:
            self.session.auth = (self.username, self.password)
    
    def get_study_info(self, study_id: int) -> dict:
        """
        Get metadata about a Movebank study.
        
        Args:
            study_id: Movebank study identifier
            
        Returns:
            Dictionary with study metadata
        """
        params = {
            "entity_type": "study",
            "study_id": study_id
        }
        
        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}
    
    def get_individuals(self, study_id: int) -> pd.DataFrame:
        """
        Get list of tagged individuals in a study.
        
        Args:
            study_id: Movebank study identifier
            
        Returns:
            DataFrame with individual animal metadata
        """
        params = {
            "entity_type": "individual",
            "study_id": study_id
        }
        
        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        from io import StringIO
        return pd.read_csv(StringIO(response.text))
    
    def get_track_data(
        self,
        study_id: int,
        individual_id: Optional[int] = None,
        sensor_type: str = "gps"
    ) -> pd.DataFrame:
        """
        Download tracking data for a study.
        
        Args:
            study_id: Movebank study identifier
            individual_id: Optional specific individual to download
            sensor_type: Type of sensor data (gps, argos, etc.)
            
        Returns:
            DataFrame with tracking locations
        """
        params = {
            "entity_type": "event",
            "study_id": study_id,
            "sensor_type_id": self._get_sensor_type_id(sensor_type),
            "attributes": "all"
        }
        
        if individual_id:
            params["individual_id"] = individual_id
        
        logger.info(f"Downloading track data for study {study_id}...")
        response = self.session.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        logger.info(f"Downloaded {len(df)} tracking points")
        return df
    
    def _get_sensor_type_id(self, sensor_type: str) -> int:
        """Map sensor type name to Movebank ID."""
        sensor_types = {
            "gps": 653,
            "argos": 82798,
            "bird-ring": 397,
            "acceleration": 2365683,
        }
        return sensor_types.get(sensor_type.lower(), 653)
    
    def download_published_dataset(
        self,
        doi: str,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Path:
        """
        Download a published dataset from Movebank Data Repository.
        
        Many whale tracking datasets are available as published datasets
        with DOIs that can be downloaded directly without API authentication.
        
        Args:
            doi: Dataset DOI (e.g., "10.5441/001/1.47h576f2")
            output_dir: Directory to save downloaded files
            filename: Optional output filename
            
        Returns:
            Path to downloaded file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Construct download URL from DOI
        # Movebank published datasets are typically available as CSV
        doi_suffix = doi.replace("10.5441/001/", "")
        
        # Try direct download URL patterns
        urls_to_try = [
            f"https://www.datarepository.movebank.org/bitstream/handle/10255/{doi_suffix}/data.csv",
            f"https://datarepository.movebank.org/entities/datapackage/{doi_suffix}",
        ]
        
        logger.info(f"Attempting to download dataset DOI: {doi}")
        
        for url in urls_to_try:
            try:
                response = self.session.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    output_file = output_dir / (filename or f"movebank_{doi_suffix}.csv")
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_file, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                    
                    logger.info(f"Downloaded to {output_file}")
                    return output_file
            except Exception as e:
                logger.debug(f"URL {url} failed: {e}")
                continue
        
        raise RuntimeError(f"Could not download dataset {doi}. Manual download may be required.")


def download_blue_fin_whale_data(output_dir: Path) -> Path:
    """
    Download the blue and fin whale satellite tracking dataset.
    
    This is the primary dataset for the migration sculptures project,
    containing tracks from 176 blue whales and 95 fin whales (1994-2018).
    
    Args:
        output_dir: Directory to save the data
        
    Returns:
        Path to downloaded CSV file
    """
    client = MovebankClient()
    
    study_info = client.WHALE_STUDIES["blue_fin_whales_california"]
    
    logger.info(f"Downloading: {study_info['description']}")
    logger.info(f"Citation: {study_info['citation']}")
    
    # First try to download via API
    try:
        df = client.get_track_data(study_info["study_id"])
        output_file = Path(output_dir) / "blue_fin_whale_tracks.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        return output_file
    except Exception as e:
        logger.warning(f"API download failed: {e}")
    
    # Fallback: try published dataset download
    try:
        return client.download_published_dataset(
            study_info["doi"],
            output_dir,
            "blue_fin_whale_tracks.csv"
        )
    except Exception as e:
        logger.error(f"Published dataset download failed: {e}")
    
    # Provide manual download instructions
    raise RuntimeError(
        f"Automatic download failed. Please manually download the dataset:\n"
        f"1. Visit: https://www.movebank.org/cms/webapp?gwt_fragment=page=studies,path=study{study_info['study_id']}\n"
        f"2. Or search DOI: {study_info['doi']}\n"
        f"3. Download the tracking data CSV\n"
        f"4. Save to: {output_dir}/blue_fin_whale_tracks.csv"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test download
    output_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "raw"
    download_blue_fin_whale_data(output_dir)
