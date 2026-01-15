# Generated 3D sculpture models (.glb files) go here

This directory is where the processed sculpture models are placed
for the web visualization to load.

After running the pipeline:
```bash
python pipeline/scripts/process_specimens.py --all-initial
```

The following files will be generated:
- `blue_whale_2010_spring.glb`
- `blue_whale_2012_spring.glb`
- `blue_whale_2014_spring.glb`
- `blue_whale_2016_spring.glb`
- `blue_whale_2018_spring.glb`
- `blue_whale_2010_fall.glb`
- `blue_whale_2012_fall.glb`
- `blue_whale_2014_fall.glb`
- `blue_whale_2016_fall.glb`
- `blue_whale_2018_fall.glb`

Copy these to `visualization/public/models/` for the web app:
```bash
cp output/models/*.glb visualization/public/models/
```
