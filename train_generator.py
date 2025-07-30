import pyvista as pv
import numpy as np
import time
from scipy.interpolate import splprep, splev
from scipy.optimize import root_scalar
from tunnel_slicer import TunnelSlicer
from data_preprocessing.excel_parser import *
#from collision_detector import CollisionDetector

class TrainWagon:
    def __init__(self, width: int, height: int, depth: int, center: tuple, color: str):
        """
        Initialize the TrainWagon object.
        
        :param width: Width of the wagon.
        :param height: Height of the wagon.
        :param depth: Depth of the wagon.
        :param center: Center position of the wagon (x, y, z).
        :param color: Color of the wagon.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.center = center
        self.color = color
    
    def create_mesh(self) -> pv.PolyData:
        """Creates and returns a PyVista train wagon mesh."""
        wagon = pv.Cube(
            center=self.center, 
            x_length=self.width, 
            y_length=self.height, 
            z_length=self.depth
        )
        wagon.color = self.color
        return wagon


def calculate_orthogonal_coordinate_system(control_points: np.ndarray, i: int, depth: float) -> tuple:
    """
    Calculate orthogonal coordinate system (forward, up, right) at i-th control point.
    
    :param control_points: Array of control points defining the path
    :param i: Current control point index
    :param depth: Distance to calculate the forward direction
    :return: Tuple of (p0, p1, forward, up, right) vectors
    """
    p0 = control_points[i]
    p1 = get_p1(control_points[i], control_points, depth)
    
    # Handle the cases at the end of the tunnel
    if p1[2] < p0[2]:
        direction = control_points[i+1] - p0
        direction = direction / np.linalg.norm(direction)
        p1 = p0 + depth * direction
        
    # Calculate orthogonal coordinate system
    forward = (p1 - p0) / np.linalg.norm(p1 - p0)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    
    return p0, p1, forward, up, right


def get_new_positions(control_points: np.ndarray, i: int, wagon: TrainWagon) -> np.ndarray:
    """
    Calculate new vertex positions for the wagon mesh based on control points.
    
    :param control_points: Array of control points defining the path
    :param i: Current control point index
    :param wagon: TrainWagon object
    :return: Array of new vertex positions for the wagon mesh
    """
    width, height, depth = wagon.width, wagon.height, wagon.depth
    
    # Get orthogonal coordinate system
    p0, p1, _, up, right = calculate_orthogonal_coordinate_system(control_points, i, depth)
    
    half_width = width / 2
    
    vertices = [
        p0 - right * half_width,
        p1 - right * half_width,
        p1 - right * half_width + up * height,
        p0 - right * half_width + up * height,
        p0 + right * half_width,
        p0 + right * half_width + up * height,
        p1 + right * half_width + up * height,
        p1 + right * half_width
    ]
    
    return np.array(vertices)


def get_p1(point: np.ndarray, control_points: np.ndarray, radius: float) -> np.ndarray:
    """
    Calculate the next point along a B-spline curve.
    
    :param point: Current 3D point [x, y, z] from which to measure distance
    :param control_points: Array of control points defining the path, shape (n, 3)
    :param radius: Distance (radius) from current point to find the next point
    :return: Next point as [x, 0, z]
    """
    x_c, z_c = point[0], point[2]
    x_points, z_points = control_points[:, 0], control_points[:, 2]

    # Fit a parametric B-spline (tck contains the knots, coefficients, and degree)
    tck, _ = splprep([x_points, z_points], s=0)

    # Function to compute the intersection equation
    def intersection_function(t: float) -> float:
        """Calculate distance squared minus radius squared for parameter t."""
        x, z = splev(t, tck)
        return (x - x_c) ** 2 + (z - z_c) ** 2 - radius ** 2

    # Find intersection points by checking sign changes in the function
    t_vals = np.linspace(0, 1, 1000)
    intersection_points = []

    for i in range(len(t_vals) - 1):
        t1, t2 = t_vals[i], t_vals[i + 1]
        if intersection_function(t1) * intersection_function(t2) < 0:  # Sign change means root exists
            root = root_scalar(intersection_function, bracket=[t1, t2]).root
            intersection_points.append(splev(root, tck))  # Store intersection coordinates

    # Take the point with highest z value
    intersection_points = np.array(intersection_points)
    x, z = intersection_points[np.argmax(intersection_points[:, 1])]
    return np.array([x, 0, z])


def update_camera(plotter: pv.Plotter, control_points: np.ndarray, i: int, wagon: TrainWagon, wagon_center: np.ndarray, distance_back: float = 2.0, height_above: float = 1.5):
    """
    Update camera position to follow the wagon using orthogonal coordinate system.
    
    :param plotter: PyVista plotter object
    :param control_points: Array of control points defining the path
    :param i: Current control point index
    :param wagon: TrainWagon object
    :param wagon_center: Current center position of the wagon
    :param distance_back: Distance behind the wagon (multiplied by wagon depth)
    :param height_above: Height above the wagon (multiplied by wagon height)
    """
    # Get orthogonal coordinate system
    _, _, forward, up, _ = calculate_orthogonal_coordinate_system(control_points, i, wagon.depth)
    
    # Calculate camera position using the orthogonal coordinate system
    camera_position = wagon_center - forward * (wagon.depth * distance_back) + up * (wagon.height * height_above)
    
    # Set camera position and orientation
    plotter.camera.position = camera_position
    plotter.camera.focal_point = wagon_center
    plotter.camera.up = up


def simulate_wagon_movement(plotter: pv.Plotter, control_points: np.ndarray, wagon: TrainWagon, tunnel_slicer: TunnelSlicer, control_points_offset: float = 0, speed: float = 0.01, export_mp4: bool = False, stop_on_safety_violation: bool = True, safety_margin: float = 300.0):
    """
    Simulate the movement of a train wagon along a tunnel with collision detection.
    
    :param plotter: PyVista plotter object
    :param control_points: Array of control points defining the path
    :param wagon: TrainWagon object
    :param tunnel_slicer: TunnelSlicer instance for collision detection
    :param control_points_offset: Offset to apply to control points
    :param speed: Animation speed (seconds between frames)
    :param export_mp4: Whether to export animation to MP4
    :param stop_on_safety_violation: If True, stop and visualize first safety violation OR collision
    :param safety_margin: Safety distance threshold in mm (default: 300mm)
    """
    # Apply offset to control points
    control_points[:, 0] += control_points_offset
    
    #collision_detector = CollisionDetector(tunnel_slicer, safety_margin=safety_margin)
    
    # Setup visualization
    wagon_mesh = wagon.create_mesh()
    plotter.add_mesh(wagon_mesh, color=wagon.color, show_edges=True, label="Train Wagon")
    plotter.add_points(control_points, color="red", point_size=5, label="Control Points")
    plotter.show(interactive_update=True)
    plotter.show_axes()
    plotter.disable()
    
    if export_mp4: plotter.open_movie("videos/tunnel.mp4")    

    # Main simulation loop
    for i in range(len(control_points) - 1):
        new_positions = get_new_positions(control_points, i, wagon)
        if new_positions is None:
            break
        
        # Update wagon position
        wagon_mesh.points = new_positions
        
        # Update camera to follow wagon
        update_camera(plotter, control_points, i, wagon, wagon_mesh.center, distance_back=5.0, height_above=0.7)
        
        # Collision Detection
        """ collision_result = collision_detector.check_collision(
            new_positions,
            frame_number=i,
            safety_margin=safety_margin,
            stop_on_safety_violation=stop_on_safety_violation
        ) """
        
        #if len(collision_detector.collision_history) > 0:
        #   break
        
        
        
        
        # Render frame
        plotter.update()
        if export_mp4: plotter.write_frame()
        time.sleep(speed)
    
    
    if export_mp4:
        plotter.close()