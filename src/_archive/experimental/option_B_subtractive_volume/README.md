# Option B: Subtractive Volume (Void-as-Data)

## Goal

Create a single mass and carve migration corridors out of it.

This option is **subtractive-first**, not blob-first.

## Algorithm (voxel-first, robust)

### 1. Bounding Solid (meters)

- Axis-aligned or PCA-aligned box
- Inflate by 10% bbox

### 2. Migration Density Voxel Field

- Rasterize tracks into voxel grid
- Paint radius in meters
- Gaussian blur

### 3. Voxel Boolean

```
final_voxels = block_voxels AND NOT migration_voxels
```

### 4. Marching Cubes

- Extract mesh from final_voxels

### 5. Normalization

- Normalize to max dimension = 2.0

### 6. Export

- `outputs/option_B_subtractive_volume/meshes/<SPECIMEN>.glb`

## Acceptance Criteria

- ✅ ONE continuous sculptural body
- ✅ Migration reads as carved absence
- ✅ No floating islands

## Design Intent

Tests hypothesis: **structure through absence**
