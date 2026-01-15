# Development Guide

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Git

### Setup

```bash
# Clone the repository
cd migration-sculptures

# Set up Python environment
cd pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up web frontend
cd ../visualization
npm install
```

### Download Data

```bash
cd pipeline
python scripts/download_movebank_data.py
python scripts/download_obis_data.py
```

> **Note:** If automatic download fails, follow the manual download instructions provided by the scripts.

### Process Sculptures

```bash
# Process all 10 initial specimens
python scripts/process_specimens.py --all-initial

# Or process specific specimens
python scripts/process_specimens.py --species blue_whale --season spring --years 2010,2012
```

### Run Visualization

```bash
cd visualization
npm run dev
```

Open http://localhost:5173 in your browser.

---

## Project Structure

```
migration-sculptures/
├── pipeline/                     # Python data processing
│   ├── src/
│   │   ├── acquisition/          # Data download modules
│   │   │   ├── movebank_client.py
│   │   │   └── obis_seamap_client.py
│   │   ├── processing/           # Track processing
│   │   │   ├── coordinate_transform.py
│   │   │   ├── track_processor.py
│   │   │   └── trajectory_bundler.py
│   │   ├── geometry/             # 3D mesh generation
│   │   │   ├── isosurface.py
│   │   │   ├── mesh_generator.py
│   │   │   └── gltf_exporter.py
│   │   └── metrics/              # Metrics computation
│   │       └── compute_metrics.py
│   └── scripts/                  # Executable scripts
│       ├── download_movebank_data.py
│       ├── download_obis_data.py
│       └── process_specimens.py
├── visualization/                # Three.js web frontend
│   ├── src/
│   │   ├── js/
│   │   │   ├── main.js
│   │   │   ├── scene.js
│   │   │   ├── sculptures.js
│   │   │   └── ui.js
│   │   └── css/
│   │       └── styles.css
│   ├── public/models/            # Generated .glb files
│   └── index.html
├── config/
│   └── pipeline_config.yaml      # Processing configuration
├── data/
│   ├── raw/                      # Downloaded data
│   └── processed/                # Intermediate files
└── output/
    └── models/                   # Generated sculptures
```

---

## Pipeline Configuration

Edit `config/pipeline_config.yaml` to adjust:

- **Region bounds:** Geographic filtering area
- **Season definitions:** Date ranges for spring/fall
- **Processing parameters:** Resampling, bundling, smoothing
- **Geometry settings:** Resolution, thresholds, mesh simplification

---

## Processing Pipeline

### 1. Data Acquisition
- Downloads whale tracking data from Movebank and OBIS-SEAMAP
- Outputs: CSV/JSON files in `data/raw/`

### 2. Track Processing
- Parses raw data into Track objects
- Filters by species, region, and date range
- Transforms coordinates to UTM projection
- Removes outlier points (unrealistic speeds)
- Resamples to uniform intervals
- Groups into Specimens (species + season + year)

### 3. Trajectory Bundling
- Force-directed bundling pulls tracks together
- Creates cohesive "migration corridors"
- Preserves divergent areas

### 4. Mesh Generation
- Creates 3D voxel density grid
- Extracts isosurface using marching cubes
- Simplifies mesh to target triangle count
- Normalizes size for consistent display

### 5. Metrics Computation
- Route Coherence: Bundle tightness
- Density Entropy: Spatial concentration
- Centroid Drift: Year-to-year movement
- Temporal Variability: Route consistency

### 6. Export
- Saves mesh as glTF binary (.glb)
- Embeds metadata and metrics in file

---

## Visualization Architecture

### Scene Setup
- Three.js WebGL rendering
- OrbitControls for navigation
- Custom lighting for sculptural emphasis

### Sculpture Display
- Grid layout of specimens
- Color-coded by species
- Interactive selection
- Info panel with metrics

### Demo Mode
- When models aren't available
- Procedural placeholder sculptures
- Demonstrates UI functionality

---

## Testing

```bash
cd pipeline
python -m pytest tests/
```

---

## Troubleshooting

### "Import could not be resolved" errors
These appear before dependencies are installed. Run:
```bash
pip install -r requirements.txt
```

### Data download fails
- Check internet connection
- Some datasets require Movebank account
- Follow manual download instructions

### No sculptures visible
- Check browser console for errors
- Ensure models are in `visualization/public/models/`
- Demo mode activates if models not found

### Performance issues
- Reduce mesh resolution in config
- Limit number of visible sculptures with filters
