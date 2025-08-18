import numpy as np
import pyvista as pv
import pandas as pd
import os
from typing import Dict, Tuple, Optional
from data_preprocessing.excel_parser import *
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    """
    Processes and visualizes tunnel geometry with spline-based wall detection.
    
    Transforms tunnel point clouds along a curved path defined by control points,
    generates horizontal slices for collision detection, and provides visualization
    capabilities for tunnel walls and splines.
    """
    
    def __init__(self, points_dict: Dict[str, pd.DataFrame], control_points: np.ndarray, plotter: pv.Plotter, 
                 n_horizontal_slices: int, train_height: int, folder_path: str, 
                 control_points_offset: int, tunnel_center_offset: int, tunnel_slicer_min_height: int) -> None:
        """
        Initialize the TunnelSlicer object.
        
        Args:
            points_dict: Dictionary containing tunnel point data organized by keys
            control_points: Array of control points defining the track path
            plotter: PyVista plotter for visualization
            n_horizontal_slices: Number of horizontal slices for wall detection
            train_height: Maximum height for slicing in millimeters
            folder_path: Path to folder for caching processed data
            control_points_offset: Offset to align control points with track in millimeters
            tunnel_center_offset: Offset to align with tunnel center line in millimeters
            tunnel_slicer_min_height: Minimum height for slicing in millimeters
        """
        self.points_dict = points_dict
        self.tunnel_points: Optional[np.ndarray] = None
        self.plotter = plotter
        self.folder_path = folder_path
        self.wall_points: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
                
        # Shift the control points to align with the railway track
        self.control_points = control_points.copy()
        self.control_points[:, 0] += control_points_offset
        
        # Shift the control points to align with the tunnel center line
        self.control_points_tunnel_center = control_points.copy()
        self.control_points_tunnel_center[:, 0] += tunnel_center_offset
                
        self.control_points_b_spline: Optional[np.ndarray] = None
        self.n_horizontal_slices = n_horizontal_slices
        self.y_values = np.linspace(tunnel_slicer_min_height, train_height, n_horizontal_slices)

    def _find_b_spline_of_control_points(self) -> None:
        """Fit a B-spline to the control points."""
        tck, u = splprep(self.control_points.T, s=0)
        self.control_points_b_spline = np.column_stack((splev(u, tck)))
    
    def _visualize_b_spline_line(self, points: np.ndarray, color: str, width: int = 3) -> None:
        """
        Visualize the B-spline curve as a line.
        
        Args:
            points: Array of points defining the spline curve
            color: Color for the line visualization
            width: Line width for visualization
        """
        n_points = points.shape[0]
        lines = np.hstack([[n_points] + list(range(n_points))]).astype(np.int32)
        polyline = pv.PolyData(points)
        polyline.lines = lines
        self.plotter.add_mesh(polyline, color=color, line_width=width)
    
    def _fit_b_spline_to_walls(self, left_wall: np.ndarray, right_wall: np.ndarray, spline_degree: int, visualize: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a B-spline to the left and right walls of the tunnel.
        
        Args:
            left_wall: Array of left wall points
            right_wall: Array of right wall points
            spline_degree: Degree of the B-spline
            visualize: Whether to visualize the fitted splines
            
        Returns:
            Tuple of (left_wall_spline, right_wall_spline) arrays
        """
        tck_left, u_left = splprep(left_wall.T, s=0, k=spline_degree)
        left_wall_spline = np.column_stack((splev(u_left, tck_left)))

        tck_right, u_right = splprep(right_wall.T, s=0, k=spline_degree)
        right_wall_spline = np.column_stack((splev(u_right, tck_right)))

        if visualize:
            self._visualize_b_spline_line(left_wall_spline, color="blue", width=3)
            self._visualize_b_spline_line(right_wall_spline, color="red", width=3)
    
        return left_wall_spline, right_wall_spline

    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """
        Classify the point as left or right wall based on the B-spline curve.
        
        Args:
            point: 3D point to classify [x, y, z]
            
        Returns:
            True if point is on left wall, False if on right wall
        """
        px, _, pz = point
        curve_xz = self.control_points_tunnel_center[:, [0, 2]]
        
        # Find the closest point on the curve
        distances = np.linalg.norm(curve_xz - np.array([px, pz]), axis=1)
        closest_idx = np.argmin(distances)
        closest_point = curve_xz[closest_idx]

        # Compute the tangent vector
        if closest_idx == 0:
            tangent = curve_xz[1] - curve_xz[0]
        elif closest_idx == len(curve_xz) - 1:
            tangent = curve_xz[-1] - curve_xz[-2]
        else:
            tangent = curve_xz[closest_idx + 1] - curve_xz[closest_idx - 1]

        # Vector from curve to point
        vec_to_point = np.array([px, pz]) - closest_point
        
        # 2D cross product
        cross_product = np.cross(np.append(tangent, 0), np.append(vec_to_point, 0))
        return cross_product[2] <= 0

    def _generate_splines_at_y_values(self, epsilon: float, wall_spline_degree: int, visualize: bool) -> None:
        """
        Generate splines on walls at different y values.
        
        Args:
            epsilon: Tolerance for Y-value filtering in millimeters
            wall_spline_degree: Degree of spline for wall fitting
            visualize: Whether to visualize the generated splines
        """
        for y_value in self.y_values:
            left_wall, right_wall = [], []
            
            for key in self.points_dict.keys():
                X = self.points_dict[key]["X"].values
                Y = self.points_dict[key]["Y"].values
                Z = self.points_dict[key]["Z"].values
                
                y_mask = (Y >= y_value - epsilon) & (Y <= y_value + epsilon)
                if np.any(y_mask):
                    filtered_X = X[y_mask]
                    filtered_Y = Y[y_mask]
                    filtered_Z = Z[y_mask]
                    points = np.column_stack([filtered_X, filtered_Y, filtered_Z])
                    
                    # Apply curve transformation to all points
                    curved_points = []
                    for point in points:
                        curved_point = self._curve_control_point(point.copy(), int(key*2))
                        curved_points.append(curved_point)
                    curved_points = np.array(curved_points)
                    
                    # Separate into left/right wall points
                    left_wall_local = []
                    right_wall_local = []
                    
                    for point in curved_points:
                        if self._classify_based_on_b_spline(point):
                            left_wall_local.append(point)
                        else:
                            right_wall_local.append(point)
                    
                    # Find representative point (closest to centroid) for each wall
                    if left_wall_local:
                        left_wall_local = np.array(left_wall_local)
                        centroid = np.mean(left_wall_local, axis=0)
                        distances = np.linalg.norm(left_wall_local - centroid, axis=1)
                        closest_idx = np.argmin(distances)
                        representative_point = left_wall_local[closest_idx].copy()
                        representative_point[1] = y_value
                        left_wall.append(representative_point)
                        
                    if right_wall_local:
                        right_wall_local = np.array(right_wall_local)
                        centroid = np.mean(right_wall_local, axis=0)
                        distances = np.linalg.norm(right_wall_local - centroid, axis=1)
                        closest_idx = np.argmin(distances)
                        representative_point = right_wall_local[closest_idx].copy()
                        representative_point[1] = y_value
                        right_wall.append(representative_point)
            
            left_wall, right_wall = np.array(left_wall), np.array(right_wall)
        
            left_wall_spline, right_wall_spline = self._fit_b_spline_to_walls(
                left_wall, right_wall, wall_spline_degree, visualize
            )
            
            # Store the spline points
            self.wall_points[y_value] = (left_wall_spline, right_wall_spline)

    def _prepare_points_for_rendering(self, data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Prepare the data for rendering.
        
        Args:
            data: Dictionary containing point data organized by keys
            
        Returns:
            Array of all points prepared for visualization
        """
        # Count total points
        total_points = sum(len(data[key]["X"]) for key in data.keys())
        points = np.zeros((total_points, 3))
        
        index = 0
        for key in data.keys():
            X = data[key]["X"]
            Y = data[key]["Y"]
            Z = data[key]["Z"]
            n_points = len(X)
            
            points[index:index+n_points] = np.column_stack([X.values, Y.values, Z.values])
            index += n_points
            
        return points

    def _rotate_points(self, points: np.ndarray, n1: np.ndarray, n2: np.ndarray, line_point: np.ndarray) -> np.ndarray:
        """
        Rotate points from plane n1 to plane n2 around their intersection line.
        
        Args:
            points: Array of points to rotate
            n1: Normal vector of source plane
            n2: Normal vector of target plane
            line_point: Point on the rotation axis
            
        Returns:
            Array of rotated points
        """
        points = np.array(points)
        
        # Normalize normal vectors
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        # Rotation axis
        rotation_axis = np.cross(n1, n2)
        norm = np.linalg.norm(rotation_axis)
        
        if norm < 1e-10:  # Parallel normals
            if np.dot(n1, n2) > 0:
                return points
            else:
                # Anti-parallel case
                if abs(n1[0]) < abs(n1[1]):
                    rotation_axis = np.cross(n1, np.array([1.0, 0.0, 0.0]))
                else:
                    rotation_axis = np.cross(n1, np.array([0.0, 1.0, 0.0]))
                rotation_axis /= np.linalg.norm(rotation_axis)
                theta = np.pi
        else:
            rotation_axis /= norm
            cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
            theta = np.arccos(cos_theta)

        # Rodrigues' rotation formula
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
        
        points_centered = points - line_point
        rotated_points = points_centered @ R.T + line_point
        
        return rotated_points

    def _get_tangent_from_index(self, index: int) -> np.ndarray:
        """
        Get the tangent vector at the control point index.
        
        Args:
            index: Index of the control point
            
        Returns:
            Tangent vector at the specified control point
        """
        if index == 0:
            return self.control_points[index + 1] - self.control_points[index]
        elif index == len(self.control_points) - 1:
            return self.control_points[index] - self.control_points[index - 1]
        else:
            return self.control_points[index + 1] - self.control_points[index - 1]

    def _curve_control_point(self, point: np.ndarray, control_point_index: int) -> np.ndarray:
        """
        Transform a point according to the curvature at a specific control point.
        
        Args:
            point: 3D point to transform
            control_point_index: Index of the control point to use for transformation
            
        Returns:
            Transformed point following the curve
        """
        control_point = self.control_points[control_point_index]
        point[0] += control_point[0]
        tangent = self._get_tangent_from_index(control_point_index)
        rotated_point = self._rotate_points(point.reshape(1, -1), np.array([0, 0, 1]), tangent, control_point)
        return rotated_point[0]

    def _curve_points(self) -> None:
        """Curve the tunnel points around the center line."""            
        for i, key in enumerate(self.points_dict.keys()):
            index = min(int(key*2), len(self.control_points)-1)
            control_point = self.control_points[index]
            tangent = self._get_tangent_from_index(index)
            
            # Transform points
            points = np.zeros((len(self.points_dict[key]["X"]), 3))
            for j in range(len(self.points_dict[key]["X"])):
                points[j] = np.array([
                    self.points_dict[key]["X"].iloc[j] + control_point[0],
                    self.points_dict[key]["Y"].iloc[j],
                    self.points_dict[key]["Z"].iloc[j]
                ])
            
            rotated_points = self._rotate_points(points, np.array([0, 0, 1]), tangent, control_point)
            
            self.points_dict[key]["X"] = pd.Series(rotated_points[:, 0])
            self.points_dict[key]["Y"] = pd.Series(rotated_points[:, 1])
            self.points_dict[key]["Z"] = pd.Series(rotated_points[:, 2])

    def _transform_points(self) -> None:
        """Transform the tunnel points."""
        self._curve_points()
        self.tunnel_points = self._prepare_points_for_rendering(self.points_dict)

    def save_to_parquet(self, filename: str = "tunnel_pointcloud.parquet") -> None:
        """
        Save tunnel points to a Parquet file.
        
        Args:
            filename: Name of the output Parquet file
        """
        assert self.tunnel_points is not None, "No tunnel points to save."
        df = pd.DataFrame(self.tunnel_points, columns=["X", "Y", "Z"])
        df.to_parquet(filename, index=False)

    def load_from_parquet(self, filename: str) -> None:
        """
        Load tunnel points from a Parquet file.
        
        Args:
            filename: Path to the Parquet file to load
        """
        df = pd.read_parquet(filename)
        self.tunnel_points = df.to_numpy()
            
    def visualize_the_tunnel(self, wall_spline_degree: int = 3) -> None:
        """
        Visualize the tunnel, loading from cache if possible.
        
        Args:
            wall_spline_degree: Degree of spline for wall fitting
        """
        # Fit a B-spline to the control points
        self._find_b_spline_of_control_points()

        # Generate splines on walls at different y values 
        self._generate_splines_at_y_values(epsilon=5, wall_spline_degree=wall_spline_degree, visualize=True)

        # Load or create tunnel points
        parquet_path = f"{self.folder_path}/tunnel_pointcloud.parquet"
        if os.path.exists(parquet_path):
            self.load_from_parquet(parquet_path)
        else:
            self._transform_points()
            self.save_to_parquet(parquet_path)
        
        # Add to plot
        self.plotter.add_points(self.tunnel_points, color="lightblue", point_size=2, label="Tunnel")
        self._visualize_b_spline_line(self.control_points_b_spline, color="green", width=3)