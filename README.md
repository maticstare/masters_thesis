# Train-Tunnel Collision Detection System

A Python system that predicts collisions between rail wagons and tunnel walls using 3D point cloud processing and parametric simulation — including in curved tunnels where static cross-section checks fail. Developed for Slovenske železnice (Slovenian Railways).

## Demo

[demo](https://github.com/user-attachments/assets/aec65d15-5602-41ec-bd91-1eff6ea842b2)

## Features

- 🚂 **Dynamic collision detection** — six critical points per wagon layer, curved-tunnel aware
- 📐 **Tunnel reconstruction** — 2D cross-sections lifted into 3D via Rodrigues rotation
- 🪚 **Largest safe wagon** — iterative shaving of an oversized wagon to fit a given tunnel
- 📦 **Cargo fitting** — Euler-angle search to fit arbitrary cargo into the shaved wagon
- 🎥 **3D visualization** — interactive PyVista scene, optional MP4 export

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Edit [main.py](main.py) to pick the tunnel and mode:

```python
tunnel = "ringo"   # or "globoko"
mode = 0           # see table below
```

| Mode | Purpose |
|---|---|
| `normal` | Run wagon through tunnel with safety margin; flag violations |
| `calculating_collision_margins` | Record per-layer wall penetrations to `collision_margins.json` |
| `shaved_off_model` | Run with the previously generated `shaved_off_wagon_model.vtk` |
| `train_model` | Run with an arbitrary STL train mesh |

Other entry points:

```bash
python collision_margins_to_mesh.py   # margins JSON → shaved_off_wagon_model.vtk
python fit_cargo.py                   # try fitting example cargo into shaved models
```

## Author

**Matic Stare**  
University of Ljubljana, Faculty of Computer and Information Science  
📧 ms79450@student.uni-lj.si  
👨‍🏫 Supervisor: doc. dr. Uroš Čibej  
Master's thesis, 2026
