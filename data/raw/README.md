# Raw data goes here

This directory stores downloaded tracking data before processing.

## Expected Structure

```
raw/
├── movebank/
│   └── blue_fin_whale_tracks.csv    # Main tracking dataset
├── obis_seamap/
│   ├── blue_whale.json              # Blue whale occurrences
│   ├── fin_whale.json               # Fin whale occurrences
│   ├── gray_whale.json              # Gray whale occurrences
│   └── humpback_whale.json          # Humpback whale occurrences
└── README.md
```

## Data Download

Run the download scripts:

```bash
cd pipeline
python scripts/download_movebank_data.py
python scripts/download_obis_data.py
```

If automatic download fails, follow manual instructions in the script output.
