# Option E: Refined Carved Specimen (Hero Module)

Generate one coherent sculptural form per specimen with:

- **Whole-body connectedness** (membrane feel)
- **Organic subtractive cavities** (carved corridor)
- **Smooth, intentional surface** (no spikes, no noisy lumps)
- **Consistent scale** for garden comparison (normalized max dim = 2.0)

## Core Concept

**Envelope minus Corridor, all in implicit space.**

No vertex displacement membranes. Everything happens in voxel/SDF space for smoothness.

## Algorithm

### E1 — Build migration density field D(x,y,z)

- Rasterize track segments into voxels (segment splatting)
- Gaussian blur density (σ = 1.4 voxels)
- Keep D as float — don't threshold early

### E2 — Build PCA capsule envelope

- Compute PCA on track points
- Define capsule aligned to first principal axis
- Create signed distance function S_capsule(p)
- Gives "whole-body" silhouette without jaggedness

### E3 — Define carve field from density

- Smoothstep mapping: `C = smoothstep(t_low, t_high, D)`
- Combine: `F = S_capsule + carve_strength × C`
- Final solid is `F < 0`
- Where density high → C pushes F positive → creates cavities

### E4 — Anti-spike polish (mandatory)

- Gaussian blur F (σ = 0.9 voxels)
- Marching cubes at F = 0
- Taubin smoothing (10 iterations)
- Decimate to target (120k triangles)
- Keep largest connected component

### E5 — Normalize

- Scale to max dimension = 2.0
- Center at origin

### E6 — Optional toolpath striation

- Micro displacement along normals using Z axis
- Off by default — only if form needs subtle texture

## Default Parameters

| Parameter                | Default | Description                         |
| ------------------------ | ------- | ----------------------------------- |
| `vox_res`                | 192     | Grid resolution                     |
| `margin_factor`          | 0.12    | Margin as fraction of bbox diagonal |
| `paint_radius_factor`    | 0.03    | Paint radius (× bbox_diag)          |
| `density_blur_sigma_vox` | 1.4     | Density field blur                  |
| `t_low_factor`           | 0.25    | Carve start threshold (× max D)     |
| `t_high_factor`          | 0.55    | Carve end threshold (× max D)       |
| `carve_strength_factor`  | 0.8     | Carve strength (× voxel size)       |
| `field_blur_sigma_vox`   | 0.9     | Final field blur                    |
| `taubin_iters`           | 10      | Mesh smoothing iterations           |
| `decimate_target_tris`   | 120,000 | Target triangle count               |

## Parameter Sweep (3-Selection Test)

Built-in refinement for fast convergence:

```bash
python -m src.option_E_refined_specimen.build \
  --sweep small \
  --export-all
```

**Sweep "small" (12 combos):**

- `paint_radius`: {0.025, 0.035} × bbox_diag
- `t_low`: {0.22, 0.28} × max(D)
- `carve_strength`: {0.6, 0.9} × base

**Best selection heuristic:**

1. Single connected component (required)
2. Cavity presence (required)
3. Low surface roughness (curvature variance)
4. Triangle count in range (50k–150k)

## Acceptance Criteria

✅ One coherent body  
✅ Readable carved corridors  
✅ Smooth macro silhouette  
✅ No "spike hedgehog" artifacts  
✅ Fits inside 2.0-unit cube after normalization

## Usage

```bash
# Single run with default params
python -m src.run_all --modules E --data data/raw/movebank/blue_fin_whale_tracks.csv

# With parameter sweep
python -m src.option_E_refined_specimen.build --sweep small --export-all
```

## Output

```
outputs/option_E_refined_specimen/meshes/<SPECIMEN_ID>.glb
outputs/option_E_refined_specimen/meta/<SPECIMEN_ID>.json
outputs/option_E_refined_specimen/previews/<SPECIMEN_ID>.png  (optional)
```

## Implementation Notes

- Do subtraction in field/voxel space, not mesh boolean
- Always blur the field before marching cubes
- Always keep only the largest connected component post-mesh
- Do not add multiple style layers — if the form reads well, stop
