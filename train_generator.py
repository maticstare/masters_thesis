from typing import Tuple, List, Optional
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
    
    def __init__(self, width: int, height: int, depth: int, center: Tuple[float, float, float] = (0, 0, 0), color: str = "blue"):
        """
        Initialize a train wagon.
        
        Args:
            width: Wagon width in millimeters
            height: Wagon height in millimeters  
            depth: Wagon depth (length) in millimeters
            center: Initial center position (x, y, z)
            color: Color for visualization
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.center = center
        self.color = color
    
    def create_mesh(self) -> pv.PolyData:
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
    
    def get_vertices_at_position(self, control_points: np.ndarray, position_index: int, train_instance: 'Train') -> np.ndarray:
        """
        Calculate wagon vertices at a specific control point position.
        
        Args:
            control_points: Array of control points defining the path
            position_index: Index of the control point to position wagon at
            train_instance: Train instance for accessing spline methods
        
        Returns:
            Array of 8 vertices defining the wagon box corners
        """
        p0, p1, _, up, right = self._calculate_orthogonal_coordinate_system(
            control_points, position_index, self.depth, train_instance
        )
        return self._create_vertices(p0, p1, up, right)

    def get_vertices_from_p0(self, control_points: np.ndarray, p0: np.ndarray, train_instance: 'Train') -> np.ndarray:
        """
        Calculate wagon vertices given a specific starting position.
        
        Args:
            control_points: Array of control points defining the path
            p0: Starting position [x, y, z] for the wagon
            train_instance: Train instance for accessing spline methods
        
        Returns:
            Array of 8 vertices defining the wagon box corners
        """
        p1 = train_instance._find_point_at_distance(control_points, p0, self.depth)
        forward = (p1 - p0) / np.linalg.norm(p1 - p0)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward) / np.linalg.norm(np.cross(up, forward))
        return self._create_vertices(p0, p1, up, right)
    
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
        half_width = self.width / 2
        return np.array([
            p0 - right * half_width,                    # rear left bottom
            p1 - right * half_width,                    # front left bottom
            p1 - right * half_width + up * self.height, # front left top
            p0 - right * half_width + up * self.height, # rear left top
            p0 + right * half_width,                    # rear right bottom
            p0 + right * half_width + up * self.height, # rear right top
            p1 + right * half_width + up * self.height, # front right top
            p1 + right * half_width                     # front right bottom
        ])
    
    def _calculate_orthogonal_coordinate_system(self, control_points: np.ndarray, i: int, depth: float, train_instance: 'Train') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate orthogonal coordinate system at a control point for wagon orientation.
        
        Args:
            control_points: Array of control points
            i: Index of current control point
            depth: Wagon depth for calculating front position
            train_instance: Train instance for accessing spline methods
        
        Returns:
            Tuple of (p0, p1, forward, up, right) vectors
        """
        p0 = control_points[i]
        p1 = train_instance._find_point_at_distance(control_points, p0, depth)
        
        # Handle end of track case
        if p1[2] < p0[2]:
            direction = control_points[i+1] - p0
            p1 = p0 + depth * (direction / np.linalg.norm(direction))
        
        # Calculate orthogonal coordinate system
        forward = (p1 - p0) / np.linalg.norm(p1 - p0)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward) / np.linalg.norm(np.cross(up, forward))
        
        return p0, p1, forward, up, right

class Train:
    """
    Represents a multi-wagon train that can move along a curved path.
    
    The first wagon (index 0) acts as the control wagon that follows the control points,
    while subsequent wagons follow ahead maintaining proper spacing.
    """
    
    def __init__(self, wagons: List[Wagon], wagon_spacing: float):
        """
        Initialize a train with multiple wagons.
        
        Args:
            wagons: List of Wagon objects
            wagon_spacing: Distance between wagon centers in millimeters
        """
        self.wagons = wagons
        self.wagon_spacing = wagon_spacing
        self.wagon_meshes = [wagon.create_mesh() for wagon in wagons]
        self.control_points_2d_tck: Optional[tuple] = None
    
    def _find_point_at_distance(self, control_points: np.ndarray, start_point: np.ndarray, distance: float) -> np.ndarray:
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

    def update_wagon_positions(self, control_points: np.ndarray, lead_index: int) -> bool:
        """
        Update positions of all wagons in the train.
        
        The control wagon (index 0) follows the control points directly,
        while subsequent wagons follow behind maintaining proper spacing.
        
        Args:
            control_points: Array of control points defining the track path
            lead_index: Current position index along the control points
            
        Returns:
            bool: True if front wagon has reached end of tunnel, False otherwise
        """
        if lead_index >= len(control_points) - 1:
            return True

        # Initialize spline once per update
        if self.control_points_2d_tck is None:
            self.control_points_2d_tck, _ = splprep([control_points[:, 0], control_points[:, 2]], s=0)

        # Update control wagon
        control_wagon = self.wagons[0]
        self.wagon_meshes[0].points = control_wagon.get_vertices_at_position(control_points, lead_index, self)
        control_wagon.center = self.wagon_meshes[0].center
        
        # Get control wagon's p1 as starting point
        _, current_p1, _, _, _ = control_wagon._calculate_orthogonal_coordinate_system(
            control_points, lead_index, control_wagon.depth, self
        )
        
        # Update following wagons
        for i in range(1, len(self.wagons)):
            wagon = self.wagons[i]
            wagon_p0 = self._find_point_at_distance(control_points, current_p1, self.wagon_spacing)
            
            self.wagon_meshes[i].points = wagon.get_vertices_from_p0(control_points, wagon_p0, self)
            wagon.center = self.wagon_meshes[i].center
            
            current_p1 = self._find_point_at_distance(control_points, wagon_p0, wagon.depth)
        
        # Check if front wagon has passed near the end of the tunnel
        return current_p1[2] >= control_points[-2][2]