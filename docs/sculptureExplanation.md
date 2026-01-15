# Migration Sculptures — Data to 3D Model Pipeline

This document explains how whale satellite tracking data is transformed into 3D sculptural forms.

---

## Overview

Raw GPS tracking data from tagged whales is transformed through a multi-stage pipeline into organic 3D sculptures that visualize migration patterns through space and time.

```
CSV Data → Coordinate Transform → Sculptural Transform → Mesh Generation → Flow Deformation → GLB Export
```

---

## 1. Raw Data (Input)

**Source**: Synthetic data based on Movebank whale tracking format  
**File**: `data/raw/movebank/blue_fin_whale_tracks.csv`

| Field | Description | Example |
|-------|-------------|---------|
| `timestamp` | When the GPS fix was recorded | `2018-03-21 01:00:00` |
| `location-long` | Longitude in degrees | `-120.32` |
| `location-lat` | Latitude in degrees | `34.37` |
| `individual-local-identifier` | Whale ID | `whale_001` |

**Dataset**: 743 GPS points from 10 whales over ~10 days in the Santa Barbara Channel.

---

## 2. Coordinate Transformation

**Module**: `src/processing/coordinate_transform.py`

GPS coordinates (WGS84) are projected to **UTM Zone 10N** (EPSG:32610) — a flat Cartesian coordinate system in meters.

```
(longitude, latitude) → (x_meters, y_meters)
   -120.32, 34.37    →   (712450, 3805200)
```

**Why?** 
- Meters allow accurate distance calculations
- Flat projection works well for the ~50km study area
- Mesh generation needs Cartesian coordinates

---

## 3. Sculptural Transformation (Time → Z Axis)

**Module**: `src/geometry/sculptural_transform.py`

### The Key Insight: Time Becomes Height

```python
z = (timestamp - t_min) / (t_max - t_min) * z_range
```

- **Earlier** GPS points → **bottom** of sculpture
- **Later** GPS points → **top** of sculpture
- Each whale's path becomes a **3D trajectory through space-time**

### Ribbon Extrusion

Each track is extruded into a ribbon perpendicular to movement direction:

```
         ←  width  →
    left ●─────────● right
         │  center │
         │ (track) │
    left ●─────────● right
         │         │
         ▼         ▼
```

**Width varies with speed**:
- Slower movement = wider ribbon (more time spent there)
- Faster movement = narrower ribbon

---

## 4. Mesh Generation

**Module**: `src/geometry/isosurface.py`

### Process:
1. **Voxel Grid**: Ribbons are rasterized into a 3D density grid (128³)
2. **Gaussian Smoothing**: Density field is smoothed for organic shapes
3. **Marching Cubes**: Isosurface extraction creates triangulated mesh
4. **Normalization**: Mesh is scaled to fit within a unit bounding box

---

## 5. Flow Field Deformation

**Modules**: `src/geometry/flow_field.py`, `src/geometry/mesh_deform.py`

A vector field displaces mesh vertices to add dynamic "lean" and curvature.

### Flow Field Composition:
```python
field(x, y, z) = drift + Σ vortices
```

- **Drift**: Constant directional pull (e.g., simulating current)
- **Vortices**: Rotational patterns at specified centers

### Deformation Application:
```python
vertex_new = vertex_old + flow_field(vertex) * strength * falloff
```

- **Strength**: 0.15 (mild) to 0.6 (extreme)
- **Falloff**: Gaussian decay from mesh center
- **Iterations**: Multiple passes for cumulative effect

### Generated Models:

| Model | Strength | Max Displacement | Visual Effect |
|-------|----------|------------------|---------------|
| `whale_mild.glb` | 0.15 | 21% | Subtle lean |
| `whale_moderate.glb` | 0.25 | 39% | Noticeable curve |
| `whale_strong.glb` | 0.40 | 59% | Dramatic deformation |
| `whale_extreme.glb` | 0.60 | 72% | Very aggressive |

---

## 6. Output

**Format**: glTF Binary (`.glb`)  
**Location**: `output/models/deformed_v2/`

The GLB files can be:
- Viewed in any glTF viewer or Blender
- 3D printed directly
- Used in web-based Three.js visualizations

---

## Visual Summary

```
RAW DATA                         3D SCULPTURE
─────────                        ─────────────
                                        ┌─── t = later (top)
whale_001: lat/lon/time    →           /    \
whale_002: lat/lon/time    →          │ ╲  ╱ │  ← ribbon surfaces
whale_003: lat/lon/time    →          │  ╲╱  │
        ...                            \    /
                                        └─── t = earlier (bottom)

X axis = East-West position (meters)
Y axis = North-South position (meters)
Z axis = TIME (when the whale was there)
Width  = SPEED (slower = wider ribbon)
Lean   = FLOW FIELD deformation
```

---

## Interpretation

The sculpture encodes multiple data dimensions:

| Visual Property | Data Meaning |
|-----------------|--------------|
| Horizontal position (X,Y) | Geographic location |
| Vertical position (Z) | Time in migration |
| Ribbon width | Swimming speed (inverse) |
| Overall lean/curve | Artistic flow field |
| Surface continuity | Spatial clustering of tracks |

You can "read" the sculpture from bottom to top to follow the whales' journey through time.

---

*Last updated: January 15, 2026*
