# Option C: Constrained Membrane (Pressure Skin)

## Goal

Generate a single enclosing skin deformed by migration density.

## Algorithm

### 1. Base Membrane

- Convex hull of downsampled track points
- OR sphere scaled to bbox

### 2. Distance Field

- For each membrane vertex: distance to nearest track point

### 3. Displacement

```
disp = exp(-(d^2)/(2*sigma^2)) * amp
```

- Push along vertex normal

### 4. Smoothing

- Laplacian smoothing (20–30 iterations)

### 5. Normalization

- Normalize to max dimension = 2.0

### 6. Export

- `outputs/option_C_constrained_membrane/meshes/<SPECIMEN>.glb`

## Acceptance Criteria

- ✅ One continuous membrane
- ✅ Bulges clearly correlate with migration density
- ✅ No noisy micro-lumps

## Design Intent

Tests hypothesis: **structure through constraint**
