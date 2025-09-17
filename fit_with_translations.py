import numpy as np
from typing import Tuple, Optional
import pyvista as pv

def mesh_to_points(
    mesh: pv.PolyData,
    resolution: float = 50.0
) -> np.ndarray:
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


def find_any_fit(
    cargo: pv.PolyData,
    container: pv.PolyData,
    step_degrees: int = 90,
    step_translation: float = 500.0
) -> Optional[Tuple[float, float, float, float, float, float]]:
    
    container_bounds = np.array(container.bounds).reshape(3, 2)
    container_center = np.array(container.center)
    container_size = container_bounds[:, 1] - container_bounds[:, 0]
    
    for yaw in range(0, 360, step_degrees):
        for pitch in range(0, 360, step_degrees):
            for roll in range(0, 360, step_degrees):
                rotated_cargo = cargo.copy()
                rotated_cargo.rotate_z(yaw, inplace=True)
                rotated_cargo.rotate_y(pitch, inplace=True)
                rotated_cargo.rotate_x(roll, inplace=True)

                cargo_bounds = np.array(rotated_cargo.bounds).reshape(3, 2)
                cargo_size = cargo_bounds[:, 1] - cargo_bounds[:, 0]

                # Quick rejection if cargo is larger than container in any dimension
                if np.any(cargo_size > container_size):
                    continue

                max_offset = (container_size - cargo_size) / 2
                
                tx_min = container_center[0] - max_offset[0]
                tx_max = container_center[0] + max_offset[0]
                ty_min = container_center[1] - max_offset[1]
                ty_max = container_center[1] + max_offset[1]
                tz_min = container_center[2] - max_offset[2]
                tz_max = container_center[2] + max_offset[2]

                for tx in np.arange(tx_min, tx_max, step_translation):
                    for ty in np.arange(ty_min, ty_max, step_translation):
                        for tz in np.arange(tz_min, tz_max, step_translation):
                            current_center = rotated_cargo.center
                            translation = [tx - current_center[0], 
                                         ty - current_center[1], 
                                         tz - current_center[2]]
                            
                            translated_cargo = rotated_cargo.copy()
                            translated_cargo.translate(translation, inplace=True)

                            cargo_points = mesh_to_points(translated_cargo, resolution=50.0)
                            if len(cargo_points) == 0:
                                continue
                            
                            cargo_cloud = pv.PolyData(cargo_points)
                            inside_flags = cargo_cloud.select_enclosed_points(container, check_surface=False)['SelectedPoints']
                            
                            containment_ratio = np.sum(inside_flags) / len(inside_flags)
                            if containment_ratio > 0.99:
                                return (yaw, pitch, roll, translation[0], translation[1], translation[2])

    return None


def visualize_fit(
    cargo: pv.PolyData,
    container: pv.PolyData,
    yaw: float,
    pitch: float,
    roll: float,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0
):
    """Visualize cargo and container with given rotation + translation"""
    plotter = pv.Plotter()

    plotter.add_mesh(container, style="wireframe", color="black",
                     line_width=2, opacity=1, label="Container")

    transformed_cargo = cargo.copy()
    transformed_cargo.rotate_z(yaw, inplace=True)
    transformed_cargo.rotate_y(pitch, inplace=True)
    transformed_cargo.rotate_x(roll, inplace=True)
    transformed_cargo.translate([tx, ty, tz], inplace=True)

    plotter.add_mesh(transformed_cargo, color="blue", opacity=1, label="Cargo")

    plotter.add_title(
        f"Cargo Fit Visualization\n"
        f"Rotation: yaw={yaw}°, pitch={pitch}°, roll={roll}°\n"
        f"Translation: x={tx}, y={ty}, z={tz}"
    )
    
    plotter.view_xy()
    plotter.camera.position = (0, 0, -abs(plotter.camera.position[2]))
    
    plotter.show_axes()
    plotter.add_legend()
    plotter.show()


def generate_cargo_object() -> pv.PolyData:
    """Create a more complex cargo object"""
    cube1 = pv.Cube(center=(0, 0, 0), x_length=2000, y_length=2000, z_length=6000).triangulate()
    cube2 = pv.Cube(center=(0, 2000, -2000), x_length=1000, y_length=3000, z_length=1000).triangulate()
    cargo = cube1.boolean_union(cube2)
    return cargo.translate(-np.array(cargo.center), inplace=True)

def read_container_model(file_path: str) -> pv.PolyData:
    """Read and prepare the container model"""
    container = pv.read(file_path)
    return container.translate(-np.array(container.center), inplace=True)

if __name__ == "__main__":
    
    #tunnel = "globoko"
    tunnel = "ringo"

    container = read_container_model(f"data/{tunnel}/shaved_off_wagon_model.vtk")
    cargo = generate_cargo_object()
    
    result = find_any_fit(cargo, container, step_degrees=45, step_translation=1000.0)

    if result:
        yaw, pitch, roll, tx, ty, tz = result
        print(f"Cargo fits at yaw={yaw}, pitch={pitch}, roll={roll}, translation=({tx}, {ty}, {tz})")
        visualize_fit(cargo, container, yaw, pitch, roll, tx, ty, tz)
    else:
        print("Cargo cannot fit in any tested orientation/translation")

