# Train-Tunnel Collision Detection System

A Python system that detects potential collisions between trains and tunnel walls using 3D point cloud processing and real-time simulation.

## Overview

This system processes tunnel data from 3D laser scans and simulates train movement to detect safety violations before they occur. Developed for Slovenske železnice (Slovenian Railways).

## Demo

[demo](https://github.com/user-attachments/assets/aec65d15-5602-41ec-bd91-1eff6ea842b2)

## Features

- 🚂 **Train Simulation**: Realistic wagon movement along curved paths
- 🏔️ **Tunnel Processing**: Convert 2D tunnel cross-sections to 3D geometry
- ⚠️ **Collision Detection**: Real-time safety violation detection
- 📊 **Visualization**: Interactive 3D simulation with PyVista
- 🎯 **Safety Analysis**: Configurable safety margins and violation types

## Quick Start

### Install Dependencies

```bash
pip install numpy pandas scipy pyvista openpyxl pyarrow
```

### Run Simulation

```python
python main.py
```

### Configure Tunnel

Edit `main.py` to select tunnel:

```python
tunnel = "globoko"  # or "ringo"
```

## How It Works

1. **Load tunnel data** from Excel files (2D cross-sections)
2. **Transform to 3D** using control points and curve fitting
3. **Generate B-splines** for smooth tunnel wall representation
4. **Simulate train movement** with accurate wagon geometry
5. **Check collisions** using 6 critical points per wagon height
6. **Visualize results** with real-time 3D graphics

## Project Structure

```
masters_thesis/
├── main.py                 # Run this to start
├── simulation.py           # Main simulation controller
├── collision_detector.py   # Collision detection logic
├── train_generator.py      # Train/wagon modeling
├── tunnel_slicer.py        # Tunnel geometry processing
├── data/                   # Tunnel data files
└── videos/                 # Output animations
```

## Output

- **Interactive 3D view** of tunnel, train, and collision points
- **Collision reports** with distance measurements
- **Safety violations** highlighted in orange/yellow
- **Optional MP4 export** for documentation

## Collision Types

- 🔴 **Outside Tunnel**: Train extends beyond tunnel boundaries
- 🟡 **Too Close**: Distance less than safety margin

## Author

**Matic Stare** - University of Ljubljana
📧 ms79450@student.uni-lj.si
👨‍🏫 Supervisor: doc. dr. Uroš Čibej

## License

Master's thesis project - University of Ljubljana, Faculty of Computer and Information Science
