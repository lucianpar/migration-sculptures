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
├── src/                          # Python sculpting modules
│   ├── common/                   # Shared utilities
│   │   ├── io.py                 # Data loading, mesh export
│   │   ├── config.py             # Configuration & metadata
│   │   ├── normalize.py          # Mesh normalization
│   │   └── mesh_ops.py           # Mesh operations
│   ├── option_H3_ridge_shell/    # ★ DEFAULT MODULE - SDF Ridge Shell
│   │   ├── __init__.py
│   │   └── build.py              # Curve → Volume → Mesh
│   ├── option_A_connective_tension/  # [deprecated]
│   ├── option_B_subtractive_volume/  # [deprecated]
│   ├── option_C_constrained_membrane/# [deprecated]
│   ├── option_D_carved_membrane/     # [deprecated]
│   ├── option_E_refined_specimen/    # [deprecated]
│   ├── option_F_hull_carve/          # [deprecated]
│   ├── option_G_spherical/           # [deprecated]
│   └── run_all.py                # Orchestrator CLI
├── tests/                        # Unit tests
├── data/
│   ├── raw/                      # Original downloaded data
│   ├── subsets/                  # Pre-filtered test subsets
│   └── processed/                # Cleaned/filtered tracks
├── outputs/                      # Generated sculptures
│   └── H3_*/                     # H3 module outputs
├── pipeline/
│   └── venv/                     # Python virtual environment
└── docs/                         # Documentation
```

## Architecture

### Current Pipeline (H3 Ridge Shell)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Module H3: SDF Ridge Shell                    │
│                    ★ DEFAULT SCULPTURE MODULE ★                  │
└─────────────────────────────────────────────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    ▼                           ▼                           ▼
┌─────────┐               ┌─────────┐               ┌─────────┐
│  H3.1   │               │  H3.2   │               │  H3.3   │
│ Resample│──────────────▶│  Voxel  │──────────────▶│ Tubular │
│ Curves  │               │  Grid   │               │ Density │
└─────────┘               └─────────┘               └─────────┘
                                                        │
                                                        ▼
                          ┌─────────┐               ┌─────────┐
                          │  H3.5   │               │  H3.4   │
                          │ Marching│◀──────────────│  Shell  │
                          │  Cubes  │               │  Mask   │
                          └─────────┘               └─────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    ▼                           ▼                           ▼
┌─────────┐               ┌─────────┐               ┌─────────┐
│  H3.6   │               │  H3.7   │               │ Output  │
│ Taubin  │──────────────▶│ Export  │──────────────▶│  GLB/   │
│ Smooth  │               │  Mesh   │               │  PLY    │
└─────────┘               └─────────┘               └─────────┘
```

### Anti-Blob Techniques

H3 produces **thin winding ridges** instead of blobby masses:

| Technique          | Parameter             | Effect                       |
| ------------------ | --------------------- | ---------------------------- |
| Narrow tube kernel | `ridge_radius=0.03`   | Thin ridge cross-section     |
| Shell constraint   | `shell_inner/outer`   | Constrains to sphere surface |
| High iso-threshold | `iso_threshold=0.15`  | Thinner ridge extraction     |
| Gaussian falloff   | `density_falloff=2.0` | Sharp ridge definition       |

### Deprecated Modules

The following modules are still available but **not recommended** for production:

| Module | Description                                    | Status     |
| ------ | ---------------------------------------------- | ---------- |
| A      | Connective Tension (tube connections)          | Deprecated |
| B      | Subtractive Volume (point cloud carving)       | Deprecated |
| C      | Constrained Membrane (convex hull)             | Deprecated |
| D      | Carved Membrane (hybrid B+C)                   | Deprecated |
| E      | Refined Carved Specimen (PCA hull + density)   | Deprecated |
| F      | Hull-Constrained Carving (E minus D corridors) | Deprecated |
| G      | Spherical Migration (density-modulated sphere) | Deprecated |

### Removed Components

| Component                   | Description              | Notes                        |
| --------------------------- | ------------------------ | ---------------------------- |
| `module_F_specimen_render/` | Three.js render pipeline | Superseded by H3 mesh output |

## Usage

### Quick Start (Default: H3 Module)

```bash
# Activate environment
source pipeline/venv/bin/activate

# Run with default settings
python src/run_all.py --data data/subsets/subset_full.csv

# High resolution output
python src/run_all.py --data data/subsets/subset_full.csv --resolution 192
```

### H3 Configuration

```python
from option_H3_ridge_shell import H3Config, build_h3_ridge_shell

config = H3Config(
    ridge_radius=0.03,      # Tube radius (smaller = thinner)
    iso_threshold=0.15,     # Iso-surface level (higher = thinner)
    voxel_resolution=128,   # Grid resolution
    shell_inner=0.92,       # Inner shell boundary
    shell_outer=1.15,       # Outer shell boundary
)
```

### Legacy Module Usage (Deprecated)

```bash
# Run deprecated modules (not recommended)
python src/run_all.py --data data/subsets/subset_full.csv --modules E F G
```

## Sculpture Metrics

Each sculpture includes computed metrics:

- **Route Coherence**: How tightly bundled the migration paths are (0-1)
- **Density Entropy**: Spatial concentration of travel paths (Shannon entropy)
- **Year-to-Year Drift**: Centroid shift compared to baseline year
- **Temporal Variability**: Route consistency across the time period

## Current Data Status

### Active Dataset

- **Blue/Fin Whale Tracks**: 742 points, 10 whales, March-April 2018
- **Location**: Santa Barbara Channel
- **Subsets**: `subset_single_whale.csv` (97 pts), `subset_three_whales.csv` (274 pts), `subset_full.csv` (742 pts)

### Data Limitations

- Current dataset covers only 6 weeks (March 15 - April 30, 2018)
- Full Movebank dataset (1994-2018) has broader coverage
- Additional migration data sources being evaluated

## Additional Data Sources (Identified)

### OBIS-SEAMAP (Duke University)

Primary repository for marine mammal satellite telemetry. 1,616+ datasets available.

| Dataset                             | Species        | Records | Type          |
| ----------------------------------- | -------------- | ------- | ------------- |
| Happywhale - Humpback (N. Pacific)  | Humpback whale | 198,537 | Photo-ID      |
| Happywhale - Humpback (S. Atlantic) | Humpback whale | 16,048  | Photo-ID      |
| Happywhale - Killer (N. Pacific)    | Killer whale   | 6,934   | Photo-ID      |
| Happywhale - Sperm (N. Atlantic)    | Sperm whale    | 7,680   | Photo-ID      |
| Happywhale - Gray (N. Pacific)      | Gray whale     | 4,602   | Photo-ID      |
| Happywhale - Blue (N. Pacific)      | Blue whale     | 1,596   | Photo-ID      |
| Happywhale - Fin (N. Pacific)       | Fin whale      | 1,337   | Photo-ID      |
| Gulf of Maine Humpback Tagging 2012 | Humpback whale | 4,017   | Satellite tag |
| Gulf of Maine Humpback Tagging 2013 | Humpback whale | 1,075   | Satellite tag |

### Movebank Data Repository

396 curated datasets with 466 million locations from ~18,000 animals.

- **Blue/Fin Whale Study** (Irvine et al.): 271 tracks, 176 blue + 95 fin whales, 1994-2018
- DOI: 10.5441/001/1.47h576f2

### NOAA Sources

- **SWFSC Marine Mammal Surveys**: Multiple cruises (CCES, ORCAWALE, etc.)
- **SEFSC GoMMAPPS**: Gulf of Mexico seasonal surveys
- **NEFSC Aerial Surveys**: North Atlantic right whale, harbor porpoise

### Download Instructions

```bash
# OBIS-SEAMAP datasets (requires registration)
# Visit: https://seamap.env.duke.edu/dataset/list
# Select dataset → Download → CSV format

# Movebank Data Repository
# Visit: https://datarepository.movebank.org/
# Search for whale studies → Download with DOI citation
```

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
