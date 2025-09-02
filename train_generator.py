from typing import Tuple, Optional
import pyvista as pv
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import root_scalar, minimize_scalar

class Wagon:
    """
    Represents a single train wagon with 3D geometry and positioning capabilities.
    
    The wagon is modeled as a rectangular box that can be positioned and oriented
    along a curved path defined by control points.
    """

    def __init__(self, width: int, height: int, depth: int, center: Tuple[float, float, float] = (0, 0, 0), wheel_offset: float = 0.25, color: str = "blue", train_model: Optional[pv.PolyData] = None):
        """
        Initialize a train wagon.
        
        Args:
            width: Wagon width in millimeters
            height: Wagon height in millimeters  
            depth: Wagon depth (length) in millimeters
            center: Initial center position (x, y, z)
            color: Color for visualization
            wheel_offset: Offset for wheel position from front or back [0, 0.5)
        """
        self.width = width
        self.height = height
        self.depth = depth  
        self.center = center
        self.color = color
        self.wheel_offset = np.clip(wheel_offset, 0, np.nextafter(0.5, 0))
        self.bounding_box = self.create_bounding_box()
        self.control_points_2d_tck: Optional[tuple] = None
        self.p0 = None
        self.p1 = None

        if train_model:
            self.train_model = train_model.copy()
            self.train_model.points = self.train_model.points - self.train_model.center
            self.original_train_points = self.train_model.points.copy()
        else:
            self.train_model = None
            self.original_train_points = None
    
    def create_bounding_box(self) -> pv.PolyData:
        """
        Create PyVista mesh representation of the wagon.
        
        Returns:
            PyVista cube mesh for visualization
        """
        return pv.Cube(
            center=self.center, 
            x_length=self.width, 
            y_length=self.height, 
            z_length=self.depth
        )
        
    def _create_vertices(self, p0: np.ndarray, p1: np.ndarray, up: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Create the 8 vertices of the wagon box from position and orientation vectors.
        
        Args:
            p0: Rear center position of wagon
            p1: Front center position of wagon
            up: Up direction vector
            right: Right direction vector
            
        Returns:
            Array of 8 vertices in specific order for PyVista cube
        """
        
        # Calculate actual wagon rear and front positions based on p0 and p1
        forward = (p1 - p0) / np.linalg.norm(p1 - p0)
        
        # Calculate actual rear and front positions of the wagon
        wagon_rear = p0 - forward * (self.depth * self.wheel_offset)
        wagon_front = p0 + forward * (self.depth * (1.0 - self.wheel_offset))
        
        half_width = self.width / 2
        return np.array([
            wagon_rear - right * half_width,                    # rear left bottom
            wagon_front - right * half_width,                   # front left bottom
            wagon_front - right * half_width + up * self.height, # front left top
            wagon_rear - right * half_width + up * self.height,  # rear left top
            wagon_rear + right * half_width,                    # rear right bottom
            wagon_rear + right * half_width + up * self.height, # rear right top
            wagon_front + right * half_width + up * self.height, # front right top
            wagon_front + right * half_width                    # front right bottom
        ])
    
    def _calculate_orthogonal_coordinate_system(self, control_points: np.ndarray, i: int, wheelbase: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate orthogonal coordinate system at a control point for wagon orientation.
        
        Args:
            control_points: Array of control points
            i: Index of current control point
            wheelbase: Wagon wheelbase for calculating front position

        Returns:
            Tuple of (p0, p1, forward, up, right) vectors
        """
        self.p0 = control_points[i]
        self.p1 = self._find_point_at_distance(control_points, self.p0, wheelbase)

        # Calculate orthogonal coordinate system
        forward = (self.p1 - self.p0) / np.linalg.norm(self.p1 - self.p0)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward) / np.linalg.norm(np.cross(up, forward))

        return self.p0, self.p1, forward, up, right

    def _find_point_at_distance(self, control_points: np.ndarray, start_point: np.ndarray, distance: float) -> np.ndarray:
        #TODO: POTENTIAL PERFORMANCE FIX
        """
        Find point at given distance along spline using circle-curve intersection method.
        
        This method finds the intersection of a circle (centered at start_point with 
        radius equal to distance) and the spline curve defined by control points.
        
        Args:
            control_points: Array of control points defining the path
            start_point: Starting position [x, y, z] 
            distance: Distance to travel along the curve in millimeters
            
        Returns:
            Point [x, 0, z] at specified distance along the curve
        """
        x_c, z_c = start_point[0], start_point[2]
        
        # Fit parametric B-spline to control points if not already done
        if self.control_points_2d_tck is None:
            self.control_points_2d_tck, _ = splprep([control_points[:, 0], control_points[:, 2]], s=0)

        def intersection_function(t: float) -> float:
            """Calculate distance squared minus target distance squared for parameter t."""
            x, z = splev(t, self.control_points_2d_tck)
            return (x - x_c) ** 2 + (z - z_c) ** 2 - distance ** 2

        # Find intersection points by checking sign changes
        t_vals = np.linspace(0, 1, 1000)
        intersection_points = []

        for i in range(len(t_vals) - 1):
            t1, t2 = t_vals[i], t_vals[i + 1]
            if intersection_function(t1) * intersection_function(t2) < 0:
                root = root_scalar(intersection_function, bracket=[t1, t2]).root
                intersection_points.append(splev(root, self.control_points_2d_tck))
        
        if len(intersection_points) == 0 or (len(intersection_points) == 1 and intersection_points[0][1] < start_point[2]):
            # More efficient fallback
            def distance_squared_to_start(t):
                x, z = splev(t, self.control_points_2d_tck)
                return (x - x_c) ** 2 + (z - z_c) ** 2
            
            closest_t = minimize_scalar(distance_squared_to_start, bounds=(0, 1), method='bounded').x
            tangent = splev(closest_t, self.control_points_2d_tck, der=1)
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 0:
                tangent3d = np.array([tangent[0], 0, tangent[1]]) / tangent_norm
                return start_point + tangent3d * distance
            else:
                # Fallback if tangent is zero
                return start_point + np.array([0, 0, distance])

        # Select point with highest z value (forward direction)
        intersection_points = np.array(intersection_points)
        x, z = intersection_points[np.argmax(intersection_points[:, 1])]
        return np.array([x, 0, z])

    def update_position(self, control_points: np.ndarray, control_point_index: int) -> bool:
        """
        Update position of the wagon.
        
        Args:
            control_points: Array of control points defining the track path
            control_point_index: Current position index along the control points
            
        Returns:
            bool: True if wagon has reached end of tunnel, False otherwise
        """
        if control_point_index >= len(control_points) - 1:
            return True

        # Initialize spline once per update
        if self.control_points_2d_tck is None:
            self.control_points_2d_tck, _ = splprep([control_points[:, 0], control_points[:, 2]], s=0)

        # Update bounding box position
        wheelbase = self.depth * (1 - 2 * self.wheel_offset)
        p0, p1, forward, up, right = self._calculate_orthogonal_coordinate_system(
            control_points, control_point_index, wheelbase
        )

        self.bounding_box.points = self._create_vertices(p0, p1, up, right)
        self.center = self.bounding_box.center
    
        if self.train_model:
            # Update train model position
            self.train_model.points = self.original_train_points.copy()
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, 0] = right
            transform_matrix[:3, 1] = up
            transform_matrix[:3, 2] = forward
            transform_matrix[:3, 3] = self.center
            
            # Apply transformation
            self.train_model.transform(transform_matrix, inplace=True)

        # Check if wagon has passed near the end of the tunnel
        _, current_p1, _, _, _ = self._calculate_orthogonal_coordinate_system(
            control_points, control_point_index, wheelbase
        )
        return current_p1[2] >= control_points[-2][2]