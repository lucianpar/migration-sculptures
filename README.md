# Migration Sculptures

**Transforming whale migration data into interactive 3D sculptures**

This project creates artistic 3D visualizations ("sculptures") from real whale tracking data in the Santa Barbara Channel. Each sculpture represents the combined migration paths of a species during a specific season and year, bundled into organic tubular forms.

## Project Overview

The system consists of two main components:

1. **Data Processing Pipeline (Python)** - Transforms raw tracking data into 3D geometry
2. **Interactive Visualization (Three.js)** - Renders sculptures in an explorable web-based "specimen garden"

## Target Species

- **Blue Whales** (Tier 1) - Primary focus using Movebank dataset (1994-2018)
- **Fin Whales** (Tier 1) - Secondary species from same dataset
- **Gray Whales** (Tier 2) - Future addition when data becomes available
- **Humpback Whales** (Tier 2) - Future addition pending data access

## Data Sources

### Primary Source

- **Movebank Data Repository**: Irvine et al. (2019) blue and fin whale satellite tracks
  - DOI: 10.5441/001/1.47h576f2
  - 271 tracks from 176 blue whales and 95 fin whales (1994-2018)
  - Coverage includes Santa Barbara Channel feeding grounds

### Supplementary Sources

- **OBIS-SEAMAP**: Gray Whales Count project, opportunistic sightings
- **NOAA Animal Telemetry Network (ATN)**: Additional whale telemetry data

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Python Environment Setup

```bash
cd pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Web Visualization Setup

```bash
cd visualization
npm install
```

## Usage

### 1. Data Acquisition

```bash
cd pipeline
python scripts/download_movebank_data.py
python scripts/download_obis_data.py
```

### 2. Process Data into 3D Models

```bash
python scripts/process_specimens.py --species blue_whale --years 2010-2018
```

### 3. Run Visualization

```bash
cd visualization
npm run dev
```

Open `http://localhost:5173` in your browser.

## Project Structure

```
migration-sculptures/
├── pipeline/                 # Python data processing
│   ├── src/                  # Core processing modules
│   │   ├── acquisition/      # Data download utilities
│   │   ├── processing/       # Track processing & bundling
│   │   ├── geometry/         # 3D mesh generation
│   │   └── metrics/          # Sculpture metrics computation
│   ├── scripts/              # Executable scripts
│   ├── tests/                # Unit tests
│   └── requirements.txt
├── visualization/            # Three.js web frontend
│   ├── src/
│   │   ├── js/               # JavaScript modules
│   │   ├── css/              # Styles
│   │   └── assets/           # Static assets
│   ├── public/
│   │   └── models/           # Generated .glb files
│   └── package.json
├── data/
│   ├── raw/                  # Original downloaded data
│   ├── processed/            # Cleaned/filtered tracks
│   └── metadata/             # Species/dataset metadata
├── output/
│   ├── models/               # Generated 3D models (.glb)
│   └── metrics/              # Computed metrics JSON
├── docs/                     # Documentation
└── config/                   # Configuration files
```

## Sculpture Metrics

Each sculpture includes computed metrics:

- **Route Coherence**: How tightly bundled the migration paths are (0-1)
- **Density Entropy**: Spatial concentration of travel paths (Shannon entropy)
- **Year-to-Year Drift**: Centroid shift compared to baseline year
- **Temporal Variability**: Route consistency across the time period

## Initial Target: 10 Sculptures

| Species    | Season | Years                        |
| ---------- | ------ | ---------------------------- |
| Blue Whale | Spring | 2010, 2012, 2014, 2016, 2018 |
| Blue Whale | Fall   | 2010, 2012, 2014, 2016, 2018 |

## License

Data usage subject to original data provider terms. See `docs/DATA_SOURCES.md` for attribution requirements.

## References

- Irvine, L.M., et al. (2019). Scales of Blue and Fin Whale Feeding Behavior off California, USA
- Abrahms, B., et al. (2019). Memory and resource tracking drive blue whale migrations
- OBIS-SEAMAP: https://seamap.env.duke.edu/
