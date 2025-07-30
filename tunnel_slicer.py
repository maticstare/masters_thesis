import numpy as np
import pyvista as pv
import pandas as pd
import os
from data_preprocessing.excel_parser import *
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    def __init__(self, points_dict: dict, control_points: np.ndarray, plotter: pv.Plotter, 
                 n_horizontal_slices: int, train_height: int, folder_path: str, 
                 control_points_offset: int, tunnel_center_offset: int, tunnel_slicer_min_height: int):
        """Initialize the TunnelSlicer object."""
        self.points_dict = points_dict
        self.tunnel_points = None
        self.plotter = plotter
        self.folder_path = folder_path
        self.wall_points = {}
                
        # Shift the control points to align with the railway track
        self.control_points = control_points.copy()
        self.control_points[:, 0] += control_points_offset
        
        # Shift the control points to align with the tunnel center line
        self.control_points_tunnel_center = control_points.copy()
        self.control_points_tunnel_center[:, 0] += tunnel_center_offset
                
        self.control_points_b_spline = None
        self.n_horizontal_slices = n_horizontal_slices
        self.y_values = np.linspace(tunnel_slicer_min_height, train_height, n_horizontal_slices)

    def _find_b_spline_of_control_points(self):
        """Fit a B-spline to the control points."""
        tck, u = splprep(self.control_points.T, s=0)
        self.control_points_b_spline = np.column_stack((splev(u, tck)))
    
    def _visualize_b_spline_line(self, points, color, width=3):
        """Visualize the B-spline curve as a line."""
        n_points = points.shape[0]
        lines = np.hstack([[n_points] + list(range(n_points))]).astype(np.int32)
        polyline = pv.PolyData(points)
        polyline.lines = lines
        self.plotter.add_mesh(polyline, color=color, line_width=width)
    
    def _fit_b_spline_to_walls(self, left_wall, right_wall, spline_degree, visualize):
        """Fit a B-spline to the left and right walls of the tunnel."""
        tck_left, u_left = splprep(left_wall.T, s=0, k=spline_degree)
        left_wall_spline = np.column_stack((splev(u_left, tck_left)))

        tck_right, u_right = splprep(right_wall.T, s=0, k=spline_degree)
        right_wall_spline = np.column_stack((splev(u_right, tck_right)))

        if visualize:
            self._visualize_b_spline_line(left_wall_spline, color="blue", width=3)
            self._visualize_b_spline_line(right_wall_spline, color="red", width=3)

    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """Classify the point as left or right wall based on the B-spline curve."""
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

    def _generate_splines_at_y_values(self, epsilon, wall_spline_degree, visualize):
        """Generate splines on walls at different y values."""
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
            self._fit_b_spline_to_walls(left_wall, right_wall, wall_spline_degree, visualize)
            self.wall_points[y_value] = (left_wall, right_wall)

    def _prepare_points_for_rendering(self, data: dict) -> np.ndarray:
        """Prepare the data for rendering."""
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

    def _rotate_points(self, points, n1, n2, line_point):
        """Rotate points from plane n1 to plane n2 around their intersection line."""
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
        """Get the tangent vector at the control point index."""
        if index == 0:
            return self.control_points[index + 1] - self.control_points[index]
        elif index == len(self.control_points) - 1:
            return self.control_points[index] - self.control_points[index - 1]
        else:
            return self.control_points[index + 1] - self.control_points[index - 1]

    def _curve_control_point(self, point: np.ndarray, control_point_index: int) -> np.ndarray:
        control_point = self.control_points[control_point_index]
        point[0] += control_point[0]
        tangent = self._get_tangent_from_index(control_point_index)
        rotated_point = self._rotate_points(point.reshape(1, -1), np.array([0, 0, 1]), tangent, control_point)
        return rotated_point[0]

    def _curve_points(self):
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

    def _transform_points(self):
        """Transform the tunnel points."""
        self._curve_points()
        self.tunnel_points = self._prepare_points_for_rendering(self.points_dict)

    def save_to_parquet(self, filename="tunnel_pointcloud.parquet"):
        """Save tunnel points to a Parquet file."""
        assert self.tunnel_points is not None, "No tunnel points to save."
        df = pd.DataFrame(self.tunnel_points, columns=["X", "Y", "Z"])
        df.to_parquet(filename, index=False)

    def load_from_parquet(self, filename):
        """Load tunnel points from a Parquet file."""
        df = pd.read_parquet(filename)
        self.tunnel_points = df.to_numpy()
            
    def visualize_the_tunnel(self, wall_spline_degree=3):
        """Visualize the tunnel, loading from cache if possible."""
        # Fit a B-spline to the control points
        self._find_b_spline_of_control_points()

        # Generate splines on walls at different y values 
        self._generate_splines_at_y_values(epsilon=5, wall_spline_degree=3, visualize=True)

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