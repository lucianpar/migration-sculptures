# Migration Sculptures — Data to 3D Model Pipeline (v2.0)# Migration Sculptures — Data to 3D Model Pipeline



This document explains how whale satellite tracking data is transformed into 3D sculptural forms.This document explains how whale satellite tracking data is transformed into 3D sculptural forms.



------



## Overview## Overview



Raw GPS tracking data from tagged whales is transformed through a multi-stage pipeline into organic 3D sculptures that visualize migration patterns through space and time.Raw GPS tracking data from tagged whales is transformed through a multi-stage pipeline into organic 3D sculptures that visualize migration patterns through space and time.



The system supports **three distinct sculpture generation approaches**, each testing a different spatial encoding hypothesis:```

CSV Data → Coordinate Transform → Sculptural Transform → Mesh Generation → Flow Deformation → GLB Export

| Option | Name | Hypothesis |```

|--------|------|------------|

| **A** | Connective Tension | Structure through connectivity |---

| **B** | Subtractive Volume | Structure through absence |

| **C** | Constrained Membrane | Structure through constraint |## 1. Raw Data (Input)



---**Source**: Synthetic data based on Movebank whale tracking format  

**File**: `data/raw/movebank/blue_fin_whale_tracks.csv`

## Unit Model (NON-NEGOTIABLE)

| Field | Description | Example |

### Coordinate & Unit Flow|-------|-------------|---------|

```| `timestamp` | When the GPS fix was recorded | `2018-03-21 01:00:00` |

Raw GPS (lat/lon, degrees)| `location-long` | Longitude in degrees | `-120.32` |

  → UTM Zone 10N projection| `location-lat` | Latitude in degrees | `34.37` |

  → meters (x_m, y_m)| `individual-local-identifier` | Whale ID | `whale_001` |

  → z_m (time mapped to height, in meters)

  → bundling / density / geometry ops (still meters)**Dataset**: 743 GPS points from 10 whales over ~10 days in the Santa Barbara Channel.

  → NORMALIZATION (after geometry generation)

  → export mesh---

```

## 2. Coordinate Transformation

### Two Valid Output Modes

**Module**: `src/processing/coordinate_transform.py`

**Mode A — Normalized (DEFAULT)**

- Units: dimensionlessGPS coordinates (WGS84) are projected to **UTM Zone 10N** (EPSG:32610) — a flat Cartesian coordinate system in meters.

- Rule: `max(bbox dimension) = 2.0 units`

- Ideal for specimen garden / comparison```

- In Three.js / Blender: 1 unit ≈ 1 meter (virtual)(longitude, latitude) → (x_meters, y_meters)

   -120.32, 34.37    →   (712450, 3805200)

**Mode B — Real-World Scale (Debug Only)**```

- Units: meters

- A 50 km corridor = 50,000 units**Why?** 

- Used ONLY for debugging, sanity checks, scale screenshots- Meters allow accurate distance calculations

- NOT for final garden layouts- Flat projection works well for the ~50km study area

- Mesh generation needs Cartesian coordinates

### Required Metadata

---

Every exported mesh includes:

```json## 3. Sculptural Transformation (Time → Z Axis)

{

  "unit_mode": "normalized" | "meters",**Module**: `src/geometry/sculptural_transform.py`

  "bbox_max_dimension": 2.0,

  "normalization_applied": true,### The Key Insight: Time Becomes Height

  "scale_factor": 0.000045,

  "bbox_before_normalization": {```python

    "max_dimension": 44500.0,z = (timestamp - t_min) / (t_max - t_min) * z_range

    "bounds": {...}```

  }

}- **Earlier** GPS points → **bottom** of sculpture

```- **Later** GPS points → **top** of sculpture

- Each whale's path becomes a **3D trajectory through space-time**

---

### Ribbon Extrusion

## Repository Structure

Each track is extruded into a ribbon perpendicular to movement direction:

```

migration-sculptures/```

├── data/         ←  width  →

│   ├── specimens/           # Processed specimen data    left ●─────────● right

│   │   └── SB-BLUE-2015-FALL/         │  center │

│   │       ├── tracks.parquet         │ (track) │

│   │       └── meta.json    left ●─────────● right

│   └── raw/                 # Raw tracking data         │         │

│       └── movebank/         ▼         ▼

│           └── blue_fin_whale_tracks.csv```

│

├── outputs/**Width varies with speed**:

│   ├── option_A_connective_tension/- Slower movement = wider ribbon (more time spent there)

│   │   ├── meshes/*.glb- Faster movement = narrower ribbon

│   │   └── meta/*.json

│   ├── option_B_subtractive_volume/---

│   │   ├── meshes/*.glb

│   │   └── meta/*.json## 4. Mesh Generation

│   └── option_C_constrained_membrane/

│       ├── meshes/*.glb**Module**: `src/geometry/isosurface.py`

│       └── meta/*.json

│### Process:

└── src/1. **Voxel Grid**: Ribbons are rasterized into a 3D density grid (128³)

    ├── common/              # Shared utilities2. **Gaussian Smoothing**: Density field is smoothed for organic shapes

    │   ├── config.py        # UnitMode, Config, MeshMetadata3. **Marching Cubes**: Isosurface extraction creates triangulated mesh

    │   ├── coords.py        # CoordinateTransformer4. **Normalization**: Mesh is scaled to fit within a unit bounding box

    │   ├── io.py            # Track loading, mesh saving

    │   ├── normalize.py     # Mesh normalization---

    │   ├── voxel.py         # VoxelGrid, rasterization

    │   └── mesh_ops.py      # Smoothing, manifold repair## 5. Flow Field Deformation

    │

    ├── option_A_connective_tension/**Modules**: `src/geometry/flow_field.py`, `src/geometry/mesh_deform.py`

    │   └── build.py         # Filament/tube network

    │A vector field displaces mesh vertices to add dynamic "lean" and curvature.

    ├── option_B_subtractive_volume/

    │   └── build.py         # Carved void sculpture### Flow Field Composition:

    │```python

    ├── option_C_constrained_membrane/field(x, y, z) = drift + Σ vortices

    │   └── build.py         # Pressure-deformed skin```

    │

    └── run_all.py           # Orchestrator- **Drift**: Constant directional pull (e.g., simulating current)

```- **Vortices**: Rotational patterns at specified centers



---### Deformation Application:

```python

## 1. Raw Data (Input)vertex_new = vertex_old + flow_field(vertex) * strength * falloff

```

**Source**: Movebank whale tracking data (or synthetic equivalent)  

**File**: `data/raw/movebank/blue_fin_whale_tracks.csv`- **Strength**: 0.15 (mild) to 0.6 (extreme)

- **Falloff**: Gaussian decay from mesh center

| Field | Description | Example |- **Iterations**: Multiple passes for cumulative effect

|-------|-------------|---------|

| `timestamp` | When the GPS fix was recorded | `2018-03-21 01:00:00` |### Generated Models:

| `location-long` | Longitude in degrees | `-120.32` |

| `location-lat` | Latitude in degrees | `34.37` || Model | Strength | Max Displacement | Visual Effect |

| `individual-local-identifier` | Whale ID | `whale_001` ||-------|----------|------------------|---------------|

| `whale_mild.glb` | 0.15 | 21% | Subtle lean |

---| `whale_moderate.glb` | 0.25 | 39% | Noticeable curve |

| `whale_strong.glb` | 0.40 | 59% | Dramatic deformation |

## 2. Common Preprocessing| `whale_extreme.glb` | 0.60 | 72% | Very aggressive |



### Coordinate Transformation---

**Module**: `src/common/coords.py`

## 6. Output

GPS coordinates (WGS84) are projected to **UTM Zone 10N** (EPSG:32610):

```python**Format**: glTF Binary (`.glb`)  

transformer = CoordinateTransformer.for_santa_barbara_channel()**Location**: `output/models/deformed_v2/`

utm = transformer.to_utm(lon, lat)  # Returns UTMCoordinates with x_m, y_m

```The GLB files can be:

- Viewed in any glTF viewer or Blender

### Track Data Structure- 3D printed directly

**Module**: `src/common/io.py`- Used in web-based Three.js visualizations



Canonical track format:---

```python

@dataclass## Visual Summary

class Track:

    track_id: str```

    x_m: np.ndarray      # Easting in metersRAW DATA                         3D SCULPTURE

    y_m: np.ndarray      # Northing in meters─────────                        ─────────────

    z_m: np.ndarray      # Time-mapped height in meters                                        ┌─── t = later (top)

    t_seconds: np.ndarraywhale_001: lat/lon/time    →           /    \

```whale_002: lat/lon/time    →          │ ╲  ╱ │  ← ribbon surfaces

whale_003: lat/lon/time    →          │  ╲╱  │

### Normalization        ...                            \    /

**Module**: `src/common/normalize.py`                                        └─── t = earlier (bottom)



⚠️ **CRITICAL**: Normalization happens AFTER geometry generation, NEVER before.X axis = East-West position (meters)

Y axis = North-South position (meters)

```pythonZ axis = TIME (when the whale was there)

mesh, norm_result = normalize_mesh(mesh, target_max_dim=2.0)Width  = SPEED (slower = wider ribbon)

# Returns: normalized mesh + NormalizationResult with scale_factorLean   = FLOW FIELD deformation

``````



------



## 3. Option A: Connective Tension## Interpretation



**Goal**: Turn migration data into a single connected filament system.The sculpture encodes multiple data dimensions:



### Algorithm| Visual Property | Data Meaning |

1. **Node extraction**: Resample tracks every 2-5 km, cluster to 80-150 nodes|-----------------|--------------|

2. **Graph construction**: kNN graph (k=3), remove long edges| Horizontal position (X,Y) | Geographic location |

3. **Curve generation**: Splines with tension/sag| Vertical position (Z) | Time in migration |

4. **Tube sweep**: Radius proportional to local track density| Ribbon width | Swimming speed (inverse) |

5. **Merge & normalize**: Single mesh, max dim = 2.0| Overall lean/curve | Artistic flow field |

| Surface continuity | Spatial clustering of tracks |

### Visual Character

- Infrastructure / connective tissueYou can "read" the sculpture from bottom to top to follow the whales' journey through time.

- Tension-bearing filaments

- ONE connected object---



```*Last updated: January 15, 2026*

    ●━━━━━━●━━━━━━●
     \    / \    /
      ●━━●   ●━━●
       \ /   /
        ●━━━●
```

---

## 4. Option B: Subtractive Volume

**Goal**: Create a solid mass and carve migration corridors out of it.

### Algorithm
1. **Bounding solid**: Axis-aligned box, inflated 10%
2. **Migration density field**: Rasterize tracks into voxel grid
3. **Boolean subtract**: `final = block AND NOT migration`
4. **Marching cubes**: Extract isosurface
5. **Normalize**: Max dim = 2.0

### Visual Character
- Migration as carved absence (void = data)
- Solid sculptural mass
- ONE continuous body

```
    ┌─────────────┐
    │  ░░░░░░░░░  │
    │ ░░░╔═══╗░░░ │  ← carved corridor
    │ ░░░║   ║░░░ │
    │  ░░╚═══╝░░  │
    └─────────────┘
```

---

## 5. Option C: Constrained Membrane

**Goal**: Generate a single enclosing skin deformed by migration density.

### Algorithm
1. **Base membrane**: Convex hull or ellipsoid around tracks
2. **Distance field**: For each vertex, distance to nearest track
3. **Displacement**: `disp = exp(-d²/2σ²) * amplitude` along normals
4. **Laplacian smoothing**: 20-30 iterations
5. **Normalize**: Max dim = 2.0

### Visual Character
- Pressure skin / organic membrane
- Bulges correlate with migration density
- ONE continuous surface

```
        ╭─────────╮
       ╱    ◉◉     ╲
      │    ◉◉◉◉     │  ← bulge where tracks cluster
      │     ◉◉      │
       ╲           ╱
        ╰─────────╯
```

---

## 6. Running the Pipeline

### Single Specimen, All Options
```bash
python src/run_all.py \
  --data data/raw/movebank/blue_fin_whale_tracks.csv \
  --modules A B C \
  --unit-mode normalized
```

### Multiple Specimens
```bash
python src/run_all.py \
  --specimens 10 \
  --modules A B C \
  --unit-mode normalized \
  --output outputs
```

### Output
```
outputs/
├── option_A_connective_tension/meshes/specimen.glb
├── option_B_subtractive_volume/meshes/specimen.glb
├── option_C_constrained_membrane/meshes/specimen.glb
└── run_summary.json
```

---

## 7. Viewing in Blender / Three.js

### Normalized Meshes (Default)
- Max size = 2.0 units
- Treat 1 unit ≈ 1 meter (virtual)
- Camera distance: ~5-10 units
- Lighting scale assumes human-scale sculpture

### Real-World Scale (Debug)
- Divide by ~1000 to view comfortably
- Only for checking geographic accuracy

---

## 8. Design Intent

These are not independent art objects. Each module tests a different spatial encoding hypothesis:

| Module | Hypothesis | If it fails... |
|--------|------------|----------------|
| **A** | Structure through connectivity | Multiple disconnected pieces |
| **B** | Structure through absence | Floating islands, inconsistent carving |
| **C** | Structure through constraint | Noisy micro-lumps, no clear bulges |

A successful sculpture should:
- Be ONE continuous object
- Have consistent scale (normalized to 2.0 units)
- Clearly encode the migration data in its form

---

## 9. Data Encoding Summary

| Visual Property | Data Meaning |
|-----------------|--------------|
| Horizontal position (X,Y) | Geographic location |
| Vertical position (Z) | Time in migration |
| **Option A**: Filament thickness | Track density |
| **Option A**: Sag | Edge length / tension |
| **Option B**: Void depth | Track density |
| **Option C**: Bulge amplitude | Track proximity |

---

*Last updated: January 15, 2026 — v2.0 (Module-based architecture)*
