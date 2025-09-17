import numpy as np
from typing import Tuple, Optional
import pyvista as pv

def mesh_to_points(mesh: pv.PolyData, resolution: float = 50.0) -> np.ndarray:
    """Convert mesh to point cloud for distance calculations"""
    bounds = mesh.bounds
    
    x_size = int((bounds[1] - bounds[0]) / resolution) + 1
    y_size = int((bounds[3] - bounds[2]) / resolution) + 1  
    z_size = int((bounds[5] - bounds[4]) / resolution) + 1
    
    x_size = min(x_size, 50)
    y_size = min(y_size, 50)
    z_size = min(z_size, 50)
    
    x = np.linspace(bounds[0], bounds[1], x_size)
    y = np.linspace(bounds[2], bounds[3], y_size)
    z = np.linspace(bounds[4], bounds[5], z_size)
    
    points = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    
    point_cloud = pv.PolyData(points)
    inside_mask = point_cloud.select_enclosed_points(mesh, check_surface=False)['SelectedPoints']
    
    return points[inside_mask]

def can_fit(cargo: pv.PolyData, container: pv.PolyData, yaw: float, pitch: float, roll: float) -> bool:
    """Check if cargo can fit inside container at given orientation"""
    
    rotated_cargo = cargo.copy()
    rotated_cargo = rotated_cargo.rotate_z(yaw)
    rotated_cargo = rotated_cargo.rotate_y(pitch)
    rotated_cargo = rotated_cargo.rotate_x(roll)
    
    cargo_points = mesh_to_points(rotated_cargo, resolution=50.0)
    
    cargo_cloud = pv.PolyData(cargo_points)
    inside_flags = cargo_cloud.select_enclosed_points(container, check_surface=False)['SelectedPoints']
    
    containment_ratio = np.sum(inside_flags) / len(inside_flags) if len(inside_flags) > 0 else 0
    
    return containment_ratio > 0.99

def find_any_fit(cargo: pv.PolyData, container: pv.PolyData, step_degrees: int = 90) -> Optional[Tuple[float, float, float]]:
    """Try to find any orientation where cargo fits inside container"""
    for yaw in range(0, 360, step_degrees):
        for pitch in range(0, 360, step_degrees):
            for roll in range(0, 360, step_degrees):
                if can_fit(cargo, container, yaw, pitch, roll):
                    return (yaw, pitch, roll)
    return None

def visualize_fit(cargo: pv.PolyData, container: pv.PolyData, yaw: float, pitch: float, roll: float):
    """Visualize cargo and container with given rotation"""
    plotter = pv.Plotter()
    
    plotter.add_mesh(container, style="wireframe", color="black", line_width=2, opacity=1, label="Container")
    
    rotated_cargo = cargo.copy()
    rotated_cargo = rotated_cargo.rotate_z(yaw)
    rotated_cargo = rotated_cargo.rotate_y(pitch) 
    rotated_cargo = rotated_cargo.rotate_x(roll)
    
    plotter.add_mesh(rotated_cargo, color="blue", opacity=1, label="Cargo")
    
    plotter.add_title(f'Cargo Fit Visualization\nRotation: yaw={yaw}°, pitch={pitch}°, roll={roll}°')
    
    plotter.view_xy()
    plotter.camera.position = (0, 0, -abs(plotter.camera.position[2]))
    
    
    plotter.show_axes()
    plotter.add_legend()
    plotter.show()

def generate_cargo_object() -> pv.PolyData:
    """Create a more complex cargo object"""
    cube1 = pv.Cube(center=(0, 0, 0), x_length=2000, y_length=2000, z_length=6000).triangulate()
    cube2 = pv.Cube(center=(0, 2000, -2000), x_length=1000, y_length=2000, z_length=1000).triangulate()
    
    cargo = cube1.boolean_union(cube2)
    
    cargo.translate(-np.array(cargo.center), inplace=True)
    return cargo


def read_container_model() -> pv.PolyData:
    """Read and prepare the container model"""
    container = pv.read("data/ringo/shaved_off_wagon_model.vtk")
    container.translate(-np.array(container.center), inplace=True)
    return container

if __name__ == "__main__":
    container = read_container_model()
    cargo = generate_cargo_object()
    
    result = find_any_fit(cargo, container, step_degrees=45)
    
    if result:
        yaw, pitch, roll = result
        print(f"Cargo can fit at rotation: yaw={yaw}°, pitch={pitch}°, roll={roll}°")
        visualize_fit(cargo, container, yaw, pitch, roll)
    else:
        print("Cargo cannot fit in any tested orientation")
        visualize_fit(cargo, container, 0, 0, 0)