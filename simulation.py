import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import root_scalar
import time
from collision_detector import CollisionDetector

class Simulation:
    def __init__(self, plotter, control_points, wagon, tunnel_slicer, control_points_offset=0, speed=0.01, export_mp4=False, stop_on_safety_violation=True, safety_margin=300.0):
        self.plotter = plotter
        self.control_points = control_points.copy()
        self.wagon = wagon
        self.tunnel_slicer = tunnel_slicer
        self.control_points_offset = control_points_offset
        self.speed = speed
        self.export_mp4 = export_mp4
        self.stop_on_safety_violation = stop_on_safety_violation
        self.safety_margin = safety_margin
        
        # Initialize components
        self.collision_detector = CollisionDetector(tunnel_slicer, safety_margin=safety_margin)
        self.wagon_mesh = None
        self._setup_simulation()
    
    def _setup_simulation(self):
        """Initialize the simulation environment."""
        # Apply offset to control points
        self.control_points[:, 0] += self.control_points_offset
        
        # Setup visualization
        self.wagon_mesh = self.wagon.create_mesh()
        self.plotter.add_mesh(self.wagon_mesh, color=self.wagon.color, show_edges=True, label="Train Wagon")
        self.plotter.add_points(self.control_points, color="red", point_size=5, label="Control Points")
        self.plotter.show(interactive_update=True)
        self.plotter.show_axes()
        self.plotter.disable()
        
        if self.export_mp4:
            self.plotter.open_movie("videos/tunnel.mp4")
    
    def get_new_positions(self, i: int) -> np.ndarray:
        """
        Calculate new vertex positions for the wagon mesh based on control points.
        
        :param i: Current control point index
        :return: Array of new vertex positions for the wagon mesh
        """
        width, height, depth = self.wagon.width, self.wagon.height, self.wagon.depth
        
        # Get orthogonal coordinate system
        p0, p1, _, up, right = self._calculate_orthogonal_coordinate_system(i, depth)
        
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
    
    def _get_p1(self, point: np.ndarray, radius: float) -> np.ndarray:
        """
        Calculate the next point along a B-spline curve.
        
        :param point: Current 3D point [x, y, z] from which to measure distance
        :param radius: Distance (radius) from current point to find the next point
        :return: Next point as [x, 0, z]
        """
        x_c, z_c = point[0], point[2]
        x_points, z_points = self.control_points[:, 0], self.control_points[:, 2]

        # Fit a parametric B-spline
        tck, _ = splprep([x_points, z_points], s=0)

        # Function to compute the intersection equation
        def intersection_function(t: float) -> float:
            """Calculate distance squared minus radius squared for parameter t."""
            x, z = splev(t, tck)
            return (x - x_c) ** 2 + (z - z_c) ** 2 - radius ** 2

        # Find intersection points by checking sign changes
        t_vals = np.linspace(0, 1, 1000)
        intersection_points = []

        for i in range(len(t_vals) - 1):
            t1, t2 = t_vals[i], t_vals[i + 1]
            if intersection_function(t1) * intersection_function(t2) < 0:
                root = root_scalar(intersection_function, bracket=[t1, t2]).root
                intersection_points.append(splev(root, tck))

        # Take the point with highest z value
        intersection_points = np.array(intersection_points)
        x, z = intersection_points[np.argmax(intersection_points[:, 1])]
        return np.array([x, 0, z])
    
    def _calculate_orthogonal_coordinate_system(self, i: int, depth: float) -> tuple:
        """
        Calculate orthogonal coordinate system (forward, up, right) at i-th control point.
        
        :param i: Current control point index
        :param depth: Distance to calculate the forward direction
        :return: Tuple of (p0, p1, forward, up, right) vectors
        """
        p0 = self.control_points[i]
        p1 = self._get_p1(self.control_points[i], depth)
        
        # Handle the cases at the end of the tunnel
        if p1[2] < p0[2]:
            direction = self.control_points[i+1] - p0
            direction = direction / np.linalg.norm(direction)
            p1 = p0 + depth * direction
            
        # Calculate orthogonal coordinate system
        forward = (p1 - p0) / np.linalg.norm(p1 - p0)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        return p0, p1, forward, up, right
    
    def _update_camera(self, i: int, wagon_center: np.ndarray, distance_back: float = 2.0, height_above: float = 1.5):
        """
        Update camera position to follow the wagon using orthogonal coordinate system.
        
        :param i: Current control point index
        :param wagon_center: Current center position of the wagon
        :param distance_back: Distance behind the wagon (multiplied by wagon depth)
        :param height_above: Height above the wagon (multiplied by wagon height)
        """
        # Get orthogonal coordinate system
        _, _, forward, up, _ = self._calculate_orthogonal_coordinate_system(i, self.wagon.depth)
        
        # Calculate camera position using the orthogonal coordinate system
        camera_position = wagon_center - forward * (self.wagon.depth * distance_back) + up * (self.wagon.height * height_above)
        
        # Set camera position and orientation
        self.plotter.camera.position = camera_position
        self.plotter.camera.focal_point = wagon_center
        self.plotter.camera.up = up
    
    def run(self):
        """Run the simulation."""
        # Main simulation loop
        for i in range(len(self.control_points) - 1):
            new_positions = self.get_new_positions(i)
            if new_positions is None:
                break
            
            # Update wagon position
            self.wagon_mesh.points = new_positions
            self.wagon.center = self.wagon_mesh.center
            
            # Update camera to follow wagon
            self._update_camera(i, self.wagon.center, distance_back=5.0, height_above=0.7)

            # Collision Detection
            collision_result = self.collision_detector.check_collision(
                new_positions,
                frame_number=i,
                safety_margin=self.safety_margin
            )

            if len(self.collision_detector.collision_history) > 0 and self.stop_on_safety_violation:
                print("Stopping simulation due to safety violation.")
                break

            # Render frame
            self.plotter.update()
            if self.export_mp4:
                self.plotter.write_frame()
            time.sleep(self.speed)
        
        if self.export_mp4:
            self.plotter.close()