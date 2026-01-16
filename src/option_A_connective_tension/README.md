# Option A: Connective Tension Structure

## Goal
Turn migration data into a single connected filament system where:
- **Nodes** = spatial anchors from tracks
- **Edges** = tension-bearing connectors
- Fragments (if any) are secondary

## Algorithm (meters → normalize → export)

### 1. Node Extraction
- Resample tracks every 2–5 km
- Cluster points (DBSCAN or k-means)
- Target: 80–150 nodes

### 2. Graph Construction
- kNN graph (k=3)
- Remove edges above 90th percentile length
- Optional: directional filtering using local velocity

### 3. Curve Generation
- Convert edges to splines
- Add slight sag/tension curvature

### 4. Tube Sweep
- Tube radius proportional to local track density
- Merge into single mesh

### 5. Normalization
- Apply `normalize_mesh(mesh, 2.0)`

### 6. Export
- `outputs/option_A_connective_tension/meshes/<SPECIMEN>.glb`

## Acceptance Criteria
- ✅ One connected object
- ✅ No floating, disconnected pieces
- ✅ Reads as infrastructure / connective tissue

## Design Intent
Tests hypothesis: **structure through connectivity**
