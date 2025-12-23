import numpy as np
from typing import Tuple, Optional
import pyvista as pv
import trimesh

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


def visualize_fit(cargo: Tuple[str, pv.PolyData], container: Tuple[str, pv.PolyData], yaw: float, pitch: float, roll: float, success: bool = True) -> None:
    """Visualize cargo and container with given rotation"""
    plotter = pv.Plotter()
    
    plotter.add_mesh(container[1], style="wireframe", color="black", line_width=2, opacity=1, label="Container")
    
    rotated_cargo = cargo[1].copy()
    rotated_cargo = rotated_cargo.rotate_z(yaw)
    rotated_cargo = rotated_cargo.rotate_y(pitch) 
    rotated_cargo = rotated_cargo.rotate_x(roll)
    rotated_cargo.translate(-np.array(rotated_cargo.center), inplace=True)
    
    plotter.add_mesh(rotated_cargo, color="blue", opacity=1, label="Cargo")
    
    status_text = "Fit Successful" if success else "Fit Unsuccessful"
    plotter.add_text(f'{cargo[0]} Cargo in Tunnel {container[0]} - {status_text}\nRotation: yaw={yaw}°, pitch={pitch}°, roll={roll}°')
    
    plotter.view_xy()
    plotter.camera.position = (0, 0, -abs(plotter.camera.position[2]))
    
    plotter.show()


def combine_meshes(meshes: list[pv.PolyData]) -> pv.PolyData:
    """Combine multiple meshes into one"""
    def _pv_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        return trimesh.Trimesh(vertices=mesh.points, faces=faces)

    def _trimesh_to_pv(t: trimesh.Trimesh) -> pv.PolyData:
        verts = t.vertices
        faces = t.faces
        faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        return pv.PolyData(verts, faces_flat)

    cargo = _trimesh_to_pv(trimesh.boolean.union([_pv_to_trimesh(mesh) for mesh in meshes]))
    cargo.translate(-np.array(cargo.center), inplace=True)
    return cargo


def generate_simple_cargo_object() -> pv.PolyData:
    """Create a simple cargo object composed of two cubes"""
    cube1 = pv.Cube(center=(0, 0, 0), x_length=2000, y_length=2000, z_length=6300).triangulate()
    cube2 = pv.Cube(center=(0, 2000, -2000), x_length=1000, y_length=2000, z_length=1000).triangulate()

    return combine_meshes([cube1, cube2])


def generate_sail_cargo_object() -> pv.PolyData:
    """Create a sail cargo object composed of cylinders"""
    cylinder1 = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=200, height=6200).triangulate()
    cylinder2 = pv.Cylinder(center=(1500, 0, -500), direction=(1, 0, 0), radius=200, height=3000).triangulate()
    cylinder3 = pv.Cylinder(center=(-1400, 0, 1350), direction=(1, 0, 1), radius=50, height=4300).triangulate()
    cylinder3.rotate_y(95, inplace=True)

    return combine_meshes([cylinder1, cylinder2, cylinder3])


def read_container_models() -> pv.PolyData:
    """Read and prepare the container models"""
    container1 = pv.read("data/ringo/shaved_off_wagon_model.vtk")
    container1.translate(-np.array(container1.center), inplace=True)
    
    container2 = pv.read("data/globoko/shaved_off_wagon_model.vtk")
    container2.translate(-np.array(container2.center), inplace=True)
    return {"Ringo": container1, "Globoko": container2}


if __name__ == "__main__":
    containers = read_container_models()
    cargos = {"Simple": generate_simple_cargo_object(), "Sail": generate_sail_cargo_object()}
    
    for container_name, container in containers.items():
        for cargo_name, cargo in cargos.items():
            print(f"\nTesting {cargo_name} Cargo in Container {container_name}:")
            result = find_any_fit(cargo, container, step_degrees=45)
            
            if result:
                yaw, pitch, roll = result
                print(f"Cargo can fit at rotation: yaw={yaw}°, pitch={pitch}°, roll={roll}°")
                visualize_fit((cargo_name, cargo), (container_name, container), yaw, pitch, roll, success=True)
            else:
                print("Cargo cannot fit in any tested orientation")
                visualize_fit((cargo_name, cargo), (container_name, container), 0, 0, 0, success=False)