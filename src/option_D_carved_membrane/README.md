# Option D: Carved Membrane (Hybrid B+C)

Generate a single continuous "envelope" volume around the migration corridor,
then carve an internal void that represents the corridor itself.

## Key Idea

Do both "membrane" and "subtraction" in voxels/SDF, then marching cubes **once**.
No normal-based displacement (which causes spikiness in Option C).

## Algorithm

### Step D1 — Build base density field
- Rasterize track points into voxel grid (128³–160³)
- Gaussian smooth density field (sigma 1–2 voxels)
- Keep density as float volume D(x,y,z)

### Step D2 — Create two masks: OUTER envelope and INNER cavity
Use two thresholds of the **same** density field:
```
M_outer = D > t_outer   (default: 0.18–0.28 × max(D))
M_inner = D > t_inner   (default: 0.45–0.65 × max(D))
```
This is the core hybrid trick: outer = membrane, inner = corridor.

### Step D3 — De-spike via SDF smoothing
Distance-transform smoothing (preferred):
```
S_outer = sdf(M_outer)
S_inner = sdf(M_inner)
```
Blur the SDF volumes slightly (sigma ~ 0.5–1.2 vox), then re-threshold at 0:
```
M_outer_smooth = S_outer < 0
M_inner_smooth = S_inner < 0
```

### Step D4 — Carved membrane boolean
```
M_final = M_outer_smooth AND (NOT M_inner_smooth)
```
This yields a single continuous body with an organic void.

### Step D5 — Marching cubes + mesh polish
- Marching cubes on M_final
- Taubin/Laplacian smoothing (low iterations)
- Decimate to target triangle count

### Step D6 — Normalize
Normalize to max dimension = 2.0 units after mesh generation.

## Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_res` | 160 | Grid resolution (128 if performance tight) |
| `blur_sigma_density` | 1.4 | Density field smoothing |
| `t_outer_factor` | 0.22 | Outer envelope threshold (× max density) |
| `t_inner_factor` | 0.55 | Inner cavity threshold (× max density) |
| `sdf_blur_sigma` | 0.8 | SDF smoothing sigma |
| `smoothing_iterations` | 15 | Mesh smoothing passes |
| `target_tris` | 100k | Target triangle count |

## Acceptance Criteria

- ✅ ONE continuous sculptural body
- ✅ Smooth organic surface (no jagged spikes)
- ✅ Internal void represents migration corridor
- ✅ Readable as membrane with carved interior

## Usage

```bash
python -m src.run_all --modules D --data data/raw/movebank/blue_fin_whale_tracks.csv
```

## Output

```
output/models/option_D_carved_membrane/<SPECIMEN_ID>.glb
output/models/option_D_carved_membrane/<SPECIMEN_ID>.json  # metadata
```

Metadata includes:
- Thresholds used (outer/inner)
- Envelope radius (voxels + meters)
- Voxel resolution and spacing
- unit_mode (normalized/meters)
