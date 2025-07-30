import numpy as np
import pyvista as pv
import pandas as pd
import os
from data_preprocessing.excel_parser import *
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    def __init__(self, points_dict: dict, control_points: np.ndarray, plotter: pv.Plotter, n_horizontal_slices: int, train_height: int, folder_path: str, control_points_offset: int, tunnel_center_offset: int):
        """Initialize the TunnelSlicer object."""
        self.points_dict = points_dict
        self.tunnel_points = None
        self.plotter = plotter

        # Shift the control points to align with the railway track
        self.control_points = control_points.copy()
        self.control_points[:, 0] += control_points_offset
        
        # Shift the control points to align with the tunnel center line
        self.control_points_tunnel_center = control_points.copy()
        self.control_points_tunnel_center[:, 0] += tunnel_center_offset

        self.control_points_b_spline = None

        self.n_horizontal_slices = n_horizontal_slices

        self.y_values = np.linspace(500, train_height, n_horizontal_slices)
        
        self.folder_path = folder_path

        self.wall_points = {} # wall_points[y_value] = (left_wall, right_wall)

        self.sampled_left = None
        self.sampled_right = None

    def _find_b_spline_of_control_points(self):
        """Fit a B-spline to the control points."""

        tck, _ = splprep(self.control_points.T, s=0)
        self.control_points_b_spline = np.column_stack((splev(np.linspace(0, 1, 100), tck)))
    
    def _visualize_b_spline_line(self, points, color, width=3):
        """Visualize the B-spline curve as a line."""

        n_points = points.shape[0]
        
        lines = np.hstack([[n_points] + list(range(n_points))]).astype(np.int32)
        
        polyline = pv.PolyData(points)
        polyline.lines = lines
        
        self.plotter.add_mesh(polyline, color=color, line_width=width)

    """ def _fit_b_spline_to_walls(self, left_wall, right_wall, spline_degree, visualize):
        Fit a B-spline to the left and right walls of the tunnel.
        # Fit a b-spline to the left wall
        tck_left, _ = splprep(left_wall.T, s=0, k=spline_degree)
        left_wall_spline = np.column_stack((splev(np.linspace(0, 1, 1000), tck_left)))

        # Fit a b-spline to the right wall
        tck_right, _ = splprep(right_wall.T, s=0, k=spline_degree)
        right_wall_spline = np.column_stack((splev(np.linspace(0, 1, 1000), tck_right)))

        if visualize:
            # Plot the fitted B-spline curve as a line
            self._visualize_b_spline_line(left_wall_spline, color="blue", width=3)
            self._visualize_b_spline_line(right_wall_spline, color="red", width=3)
        
        # Sample splines at control point Z coordinates
        #self.sampled_left = self._sample_spline_at_control_points(tck_left, left_wall_spline)
        #self.sampled_right = self._sample_spline_at_control_points(tck_right, right_wall_spline)
        #self.wall_points """
    
    def _fit_b_spline_to_walls(self, left_wall, right_wall, spline_degree, visualize):
        """Fit a B-spline to the left and right walls of the tunnel."""
        # Fit a b-spline to the left wall
        tck_left, u_left = splprep(left_wall.T, s=0, k=spline_degree)
        left_wall_spline = np.column_stack((splev(u_left, tck_left)))

        # Fit a b-spline to the right wall
        tck_right, u_right = splprep(right_wall.T, s=0, k=spline_degree)
        right_wall_spline = np.column_stack((splev(u_right, tck_right)))

        if visualize:
            # Plot the fitted B-spline curve as a line
            self._visualize_b_spline_line(left_wall_spline, color="blue", width=3)
            self._visualize_b_spline_line(right_wall_spline, color="red", width=3)


    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """Classify the point as left or right wall based on the B-spline curve."""
        
        # Extract the x and z coordinates of the point
        px, _, pz = point

        # Project the curve points to the xz-plane
        curve_xz = self.control_points_tunnel_center[:, [0, 2]]  # Only keep x and z coordinates

        # Find the closest point on the curve in the xz-plane
        distances = np.linalg.norm(curve_xz - np.array([px, pz]), axis=1)
        closest_idx = np.argmin(distances)
        closest_point = curve_xz[closest_idx]

        # Compute the tangent vector at the closest point
        if closest_idx == 0:
            tangent = curve_xz[1] - curve_xz[0]  # Forward difference
        elif closest_idx == len(curve_xz) - 1:
            tangent = curve_xz[-1] - curve_xz[-2]  # Backward difference
        else:
            tangent = curve_xz[closest_idx + 1] - curve_xz[closest_idx - 1]  # Central difference

        # Vector from the closest point on the curve to the given point
        vec_to_point = np.array([px, pz]) - closest_point

        # Compute the 2D cross product (determinant) to find the relative position
        cross_product = np.cross(np.append(tangent, 0), np.append(vec_to_point, 0))  # Add 0 for 2D vectors
        
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
                    
                    # Separate into left/right wall points for this key
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
    

    def _get_total_number_of_points(self, data: dict) -> int:
        """Get the total number of points in the dataset."""
        count = 0
        for key in data.keys():
            X = data[key]["X"]
            count += len(X)
        return count

    def _prepare_points_for_rendering(self, data: dict) -> np.ndarray:
        """Prepare the data for rendering."""
        #curve_function = lambda z: 0
        N = self._get_total_number_of_points(data)
        points = np.zeros((N, 3))
        index = 0
        for key in data.keys():
            X = data[key]["X"]
            Y = data[key]["Y"]
            Z = data[key]["Z"]
            for i in range(len(X)):
                points[i+index] = np.array([
                    X.iloc[i],
                    Y.iloc[i],
                    Z.iloc[i]
                ])
            index += len(X)
        return points

    def _rotate_points(self, points, n1, n2, line_point):
        """
        Rotates a point cloud from plane 1 to plane 2 around their intersection line.

        :param points: (N,3) array of point cloud coordinates
        :param n1: (3,) normal of the original plane
        :param n2: (3,) normal of the target plane
        :param line_point: (3,) a point on the intersection line of the two planes
        :return: (N,3) rotated points
        """

        points = np.array(points)

        # Normalize normal vectors
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        # Rotation axis (intersection line direction)
        rotation_axis = np.cross(n1, n2)
        norm = np.linalg.norm(rotation_axis)
        
        # Check if the rotation axis is valid (not zero)
        if norm < 1e-10:  # If normals are parallel or anti-parallel
            # No rotation needed if normals are parallel
            if np.dot(n1, n2) > 0:
                return points
            # If normals are anti-parallel, we need a different rotation axis
            else:
                # Find a vector perpendicular to n1
                if abs(n1[0]) < abs(n1[1]):
                    rotation_axis = np.cross(n1, np.array([1.0, 0.0, 0.0]))
                else:
                    rotation_axis = np.cross(n1, np.array([0.0, 1.0, 0.0]))
                rotation_axis /= np.linalg.norm(rotation_axis)
                # Use 180 degree rotation
                theta = np.pi
        else:
            # Normal case - valid rotation axis
            rotation_axis /= norm  # Normalize
            
            # Rotation angle
            cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)  # Avoid precision errors
            theta = np.arccos(cos_theta)

        # Rodrigues' rotation formula components
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Skew-symmetric cross-product matrix of the rotation axis
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        # Rotation matrix using Rodrigues' formula: R = I + sinθ * K + (1 - cosθ) * K^2
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)

        # Translate points to the rotation center (line point)
        points_centered = points - line_point

        # Rotate the points
        rotated_points = points_centered @ R.T  # Apply rotation

        # Translate back
        rotated_points += line_point

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
        # Curve a control point around the center line.
        point[0] += control_point[0]

        tangent = self._get_tangent_from_index(control_point_index)

        # Rotate the point to align with the tangent vector
        rotated_point = self._rotate_points(point.reshape(1, -1), np.array([0, 0, 1]), tangent, control_point)
        
        return rotated_point[0]
        
        

    def _curve_points(self):
        """Curve the tunnel points around the center line."""            
        for i, key in enumerate(self.points_dict.keys()):
            # Find the corresponding control point index
            index = int(key*2)
            index = min(index, len(self.control_points)-1)
            
            # Get the control point position to curve around
            control_point = self.control_points[index]
            
            # Get the tangent vector at the control point
            tangent = self._get_tangent_from_index(index)
            
            # Now transform the points at the key so they would be perpendicular to the tangent vector
            points = np.zeros((len(self.points_dict[key]["X"]), 3))
            for j in range(len(self.points_dict[key]["X"])):
                # Use the original point positions, but offset X by the control point's X value
                # This applies the curve from the control points to the tunnel points
                points[j] = np.array([
                    self.points_dict[key]["X"].iloc[j] + control_point[0],
                    self.points_dict[key]["Y"].iloc[j],
                    self.points_dict[key]["Z"].iloc[j]
                ])
            
            # Rotate these points to align with the tangent vector
            rotated_points = self._rotate_points(points, np.array([0, 0, 1]), tangent, control_point)
            
            self.points_dict[key]["X"] = pd.Series(rotated_points[:, 0])
            self.points_dict[key]["Y"] = pd.Series(rotated_points[:, 1])
            self.points_dict[key]["Z"] = pd.Series(rotated_points[:, 2])

    def _transform_points(self):
        """Transform the tunnel points."""
    	
        # Curve the tunnel points around the center line
        self._curve_points()

        # prepare the points for rendering
        self.tunnel_points = self._prepare_points_for_rendering(self.points_dict)

    def save_to_parquet(self, filename="tunnel_pointcloud.parquet"):
        """Save tunnel points to a Parquet file."""
        assert self.tunnel_points is not None, "No tunnel points to save."
        
        # Create a DataFrame
        df = pd.DataFrame(self.tunnel_points, columns=["X", "Y", "Z"])
        
        # Save to Parquet
        df.to_parquet(filename, index=False)

    def load_from_parquet(self, filename):
        """Load tunnel points from a Parquet file."""
        df = pd.read_parquet(filename)
        self.tunnel_points = df.to_numpy()
        
            
    def visualize_the_tunnel(self, wall_spline_degree=3):
        """Visualize the tunnel, loading from cache if possible."""

        # Fit a B-spline to the control points
        self._find_b_spline_of_control_points()

        # Generate splines (on walls) at different y values 
        self._generate_splines_at_y_values(epsilon=5, wall_spline_degree=3, visualize=True)

        # Load if Parquet file exists
        if os.path.exists(f"{self.folder_path}/tunnel_pointcloud.parquet"):
            self.load_from_parquet(f"{self.folder_path}/tunnel_pointcloud.parquet")
        else:
            self._transform_points()
            self.save_to_parquet(f"{self.folder_path}/tunnel_pointcloud.parquet")
        
        # Add points to plot
        self.plotter.add_points(self.tunnel_points, color="lightblue", point_size=2, label="Tunnel")
        
        # Plot the fitted B-spline curve as a line
        self._visualize_b_spline_line(self.control_points_b_spline, color="green", width=3)