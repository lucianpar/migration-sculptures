#!/usr/bin/env python3
"""
Generate realistic whale migration track data for sculpture generation.

Creates CSV files matching the format expected by the pipeline:
- timestamp, location-long, location-lat, individual-local-identifier, tag-local-identifier, study-name

Migration patterns based on known routes:
1. Gray Whale: Baja California → Alaska (coastal)
2. Humpback Whale: Hawaii → Alaska 
3. Blue Whale: California coast feeding grounds
4. Fin Whale: Channel Islands area
5. North Atlantic Right Whale: Florida → Cape Cod
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

# Seed for reproducibility
np.random.seed(42)

def add_noise(coords, noise_scale=0.05):
    """Add realistic GPS noise and whale movement variation."""
    return coords + np.random.normal(0, noise_scale, coords.shape)


def generate_track_segment(start_lon, start_lat, end_lon, end_lat, 
                          n_points, noise_scale=0.1, meander=0.3):
    """Generate a meandering track between two points."""
    t = np.linspace(0, 1, n_points)
    
    # Base path
    lon = start_lon + (end_lon - start_lon) * t
    lat = start_lat + (end_lat - start_lat) * t
    
    # Add meandering (sinusoidal deviation)
    meander_freq = np.random.uniform(2, 5)
    meander_phase = np.random.uniform(0, 2*np.pi)
    
    # Perpendicular to travel direction
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    dist = np.sqrt(dx**2 + dy**2)
    perp_x = -dy / dist if dist > 0 else 0
    perp_y = dx / dist if dist > 0 else 1
    
    meander_amount = meander * np.sin(meander_freq * np.pi * t + meander_phase)
    lon += perp_x * meander_amount
    lat += perp_y * meander_amount
    
    # Add noise
    lon = add_noise(lon, noise_scale)
    lat = add_noise(lat, noise_scale)
    
    return lon, lat


def generate_gray_whale_track(whale_id, start_date):
    """
    Gray Whale: Baja California → Alaska coastal migration
    Route follows coastline, ~6000 miles
    """
    # Key waypoints along migration
    waypoints = [
        (-114.0, 27.0),   # Baja California breeding lagoons
        (-117.5, 32.5),   # San Diego
        (-120.5, 34.5),   # Santa Barbara
        (-122.5, 37.5),   # San Francisco
        (-124.5, 42.0),   # Oregon coast
        (-125.0, 48.0),   # Washington coast
        (-135.0, 57.0),   # Southeast Alaska
        (-145.0, 60.0),   # Gulf of Alaska feeding grounds
    ]
    
    all_lon, all_lat, all_times = [], [], []
    current_time = start_date
    
    for i in range(len(waypoints) - 1):
        n_points = np.random.randint(60, 100)
        lon, lat = generate_track_segment(
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1],
            n_points, noise_scale=0.08, meander=0.4
        )
        
        # Time progression (4-6 hours between points)
        times = [current_time + timedelta(hours=np.random.uniform(4, 6) * j) for j in range(n_points)]
        current_time = times[-1]
        
        all_lon.extend(lon)
        all_lat.extend(lat)
        all_times.extend(times)
    
    return create_dataframe(all_lon, all_lat, all_times, whale_id, 
                           "Gray Whale Pacific Migration")


def generate_humpback_hawaii_track(whale_id, start_date):
    """
    Humpback Whale: Hawaii → Alaska migration
    Open ocean crossing, ~3000 miles
    """
    waypoints = [
        (-156.5, 20.8),   # Maui breeding grounds
        (-157.0, 25.0),   # North of Hawaii
        (-155.0, 35.0),   # Mid-Pacific
        (-150.0, 45.0),   # North Pacific
        (-145.0, 55.0),   # Gulf of Alaska approach
        (-148.0, 58.5),   # Prince William Sound feeding
    ]
    
    all_lon, all_lat, all_times = [], [], []
    current_time = start_date
    
    for i in range(len(waypoints) - 1):
        n_points = np.random.randint(80, 120)
        lon, lat = generate_track_segment(
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1],
            n_points, noise_scale=0.15, meander=0.6
        )
        
        times = [current_time + timedelta(hours=np.random.uniform(3, 5) * j) for j in range(n_points)]
        current_time = times[-1]
        
        all_lon.extend(lon)
        all_lat.extend(lat)
        all_times.extend(times)
    
    return create_dataframe(all_lon, all_lat, all_times, whale_id,
                           "Humpback Whale Hawaii-Alaska Migration")


def generate_blue_whale_california_track(whale_id, start_date):
    """
    Blue Whale: California coast feeding - complex foraging patterns
    Concentrated in Channel Islands and Monterey Bay areas
    """
    # Foraging area with multiple loops
    centers = [
        (-120.5, 34.0),   # Channel Islands
        (-121.5, 34.5),   # Santa Barbara Channel
        (-122.0, 36.5),   # Monterey Bay
        (-121.0, 35.5),   # Central coast
        (-120.0, 33.5),   # Southern feeding
    ]
    
    all_lon, all_lat, all_times = [], [], []
    current_time = start_date
    
    # Create looping foraging pattern
    visit_order = [0, 1, 2, 1, 3, 4, 3, 1, 0, 2]
    
    for i in range(len(visit_order) - 1):
        n_points = np.random.randint(40, 70)
        c1, c2 = centers[visit_order[i]], centers[visit_order[i+1]]
        
        lon, lat = generate_track_segment(
            c1[0], c1[1], c2[0], c2[1],
            n_points, noise_scale=0.1, meander=0.5
        )
        
        times = [current_time + timedelta(hours=np.random.uniform(2, 4) * j) for j in range(n_points)]
        current_time = times[-1]
        
        all_lon.extend(lon)
        all_lat.extend(lat)
        all_times.extend(times)
    
    return create_dataframe(all_lon, all_lat, all_times, whale_id,
                           "Blue Whale California Feeding Grounds")


def generate_fin_whale_channel_track(whale_id, start_date):
    """
    Fin Whale: Channel Islands area - fast swimmer, long dives
    More linear paths with occasional turns
    """
    waypoints = [
        (-119.5, 33.5),   # San Nicolas Island
        (-120.0, 34.0),   # Santa Rosa Island
        (-119.0, 33.0),   # San Clemente Island
        (-118.5, 33.5),   # Catalina area
        (-120.5, 34.5),   # Santa Barbara
        (-119.0, 34.0),   # Anacapa area
    ]
    
    all_lon, all_lat, all_times = [], [], []
    current_time = start_date
    
    for i in range(len(waypoints) - 1):
        n_points = np.random.randint(50, 80)
        lon, lat = generate_track_segment(
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1],
            n_points, noise_scale=0.06, meander=0.25
        )
        
        times = [current_time + timedelta(hours=np.random.uniform(2, 3) * j) for j in range(n_points)]
        current_time = times[-1]
        
        all_lon.extend(lon)
        all_lat.extend(lat)
        all_times.extend(times)
    
    return create_dataframe(all_lon, all_lat, all_times, whale_id,
                           "Fin Whale Channel Islands Study")


def generate_right_whale_atlantic_track(whale_id, start_date):
    """
    North Atlantic Right Whale: Florida → Cape Cod migration
    Critically endangered, coastal route
    """
    waypoints = [
        (-80.5, 30.5),    # Florida calving grounds
        (-79.5, 32.0),    # Georgia coast
        (-78.5, 34.0),    # North Carolina
        (-75.5, 37.0),    # Virginia
        (-74.0, 39.5),    # New Jersey
        (-70.0, 41.5),    # Cape Cod Bay feeding
        (-69.0, 42.5),    # Gulf of Maine
    ]
    
    all_lon, all_lat, all_times = [], [], []
    current_time = start_date
    
    for i in range(len(waypoints) - 1):
        n_points = np.random.randint(70, 110)
        lon, lat = generate_track_segment(
            waypoints[i][0], waypoints[i][1],
            waypoints[i+1][0], waypoints[i+1][1],
            n_points, noise_scale=0.1, meander=0.35
        )
        
        times = [current_time + timedelta(hours=np.random.uniform(3, 5) * j) for j in range(n_points)]
        current_time = times[-1]
        
        all_lon.extend(lon)
        all_lat.extend(lat)
        all_times.extend(times)
    
    return create_dataframe(all_lon, all_lat, all_times, whale_id,
                           "North Atlantic Right Whale Migration")


def create_dataframe(lon, lat, times, whale_id, study_name):
    """Create DataFrame in pipeline format."""
    n_points = len(lon)
    
    # Create multiple individuals within the track
    n_individuals = np.random.randint(3, 8)
    individuals = []
    for i in range(n_points):
        ind_num = (i // (n_points // n_individuals)) % n_individuals + 1
        individuals.append(f"whale_{ind_num:03d}")
    
    df = pd.DataFrame({
        'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') for t in times],
        'location-long': lon,
        'location-lat': lat,
        'individual-local-identifier': individuals,
        'tag-local-identifier': [f"tag_{i:03d}" for i in range(1, n_individuals + 1) for _ in range(n_points // n_individuals + 1)][:n_points],
        'study-name': study_name
    })
    
    return df


def main():
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "new_tracks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = [
        ("gray_whale_pacific", generate_gray_whale_track, datetime(2024, 3, 1)),
        ("humpback_hawaii_alaska", generate_humpback_hawaii_track, datetime(2024, 4, 15)),
        ("blue_whale_california", generate_blue_whale_california_track, datetime(2024, 7, 1)),
        ("fin_whale_channel_islands", generate_fin_whale_channel_track, datetime(2024, 8, 1)),
        ("right_whale_atlantic", generate_right_whale_atlantic_track, datetime(2024, 1, 15)),
    ]
    
    print("Generating whale migration track datasets...")
    print("=" * 60)
    
    for name, generator, start_date in datasets:
        df = generator(f"{name}_001", start_date)
        output_path = output_dir / f"{name}.csv"
        df.to_csv(output_path, index=False)
        
        n_individuals = df['individual-local-identifier'].nunique()
        print(f"✓ {name}")
        print(f"  Points: {len(df)}, Individuals: {n_individuals}")
        print(f"  Saved: {output_path}")
        print()
    
    print("=" * 60)
    print(f"Generated {len(datasets)} datasets in {output_dir}")


if __name__ == "__main__":
    main()
