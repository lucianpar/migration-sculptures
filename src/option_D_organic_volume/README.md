# Option D: Organic Volume (Hybrid)

Combines the best of Options B and C:
- **Solid continuous body** like the constrained membrane (not spiky/jagged)
- **Organic sculpting** influenced by migration paths like subtractive volume
- **Smooth, flowing forms** that suggest erosion and natural flow

## Concept

Instead of harsh corridor carving (Option B) or membrane displacement (Option C), 
Option D creates an organic blob from migration density, then gently erodes it 
along migration paths. Heavy smoothing creates a soft, sculptural result.

## Algorithm

1. **Density Blob**: Create soft metaball-like field from track point density
2. **Organic Noise**: Add natural surface variation without harsh edges
3. **Flow Erosion**: Gentle depression along migration paths (not harsh cuts)
4. **Smoothing**: Heavy Laplacian smoothing for organic feel
5. **Normalize**: Scale to max dimension = 2.0 units
6. **Export**: GLB with metadata

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 80 | Voxel grid resolution |
| `influence_radius_m` | 6000 | Point influence for blob creation |
| `density_threshold` | 0.25 | Surface extraction level |
| `noise_scale` | 0.12 | Organic noise amount (0-1) |
| `erosion_strength` | 0.25 | Path erosion intensity (0-1) |
| `smoothing_iterations` | 40 | Laplacian smoothing passes |

## Acceptance Criteria

- ✅ ONE continuous sculptural body (no floating islands)
- ✅ Smooth, organic surface (no jagged spikes)
- ✅ Form suggests flow/erosion from migration
- ✅ Readable as both solid object AND shaped by movement

## Usage

```bash
python -m src.run_all --modules D --data data/raw/movebank/blue_fin_whale_tracks.csv
```
