import numpy as np
import pyvista as pv
import pandas as pd
import os
from data_preprocessing.excel_parser import *
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    def __init__(self, points_dict: dict, control_points: np.ndarray, plotter: pv.Plotter, n_horizontal_slices: int, folder_path: str, control_points_offset: int):
        """Initialize the TunnelSlicer object."""
        self.points_dict = points_dict
        self.control_points = control_points.copy()
        self.tunnel_points = None
        self.plotter = plotter

        # Shift the control points to the center of the tunnel
        self.control_points_adjusted = control_points
        self.control_points_adjusted[:, 0] += control_points_offset

        self.tunnel_center_line_b_spline = None
        self._tck = None

        self.n_horizontal_slices = n_horizontal_slices

        # Bounds hardcoded so far ...
        self.hslice_planes = np.linspace(500, 5100.0, n_horizontal_slices)
        
        self.folder_path = folder_path

    
    def _find_b_spline_of_center_tunnel_line(self):
        """Fit a B-spline to the center tunnel line."""

        tck, _ = splprep(self.control_points_adjusted.T, s=0)
        self._tck = tck
        self.tunnel_center_line_b_spline = np.column_stack((splev(np.linspace(0, 1, 100), tck)))
    
    def _visualize_b_spline_line(self, points, color, width=3):
        """Visualize the B-spline curve as a line."""

        n_points = points.shape[0]
        
        lines = np.hstack([[n_points] + list(range(n_points))]).astype(np.int32)
        
        polyline = pv.PolyData(points)
        polyline.lines = lines
        
        self.plotter.add_mesh(polyline, color=color, line_width=width)

    def _fit_b_spline_to_walls(self, left_wall, right_wall, visualize):
        """Fit a B-spline to the left and right walls of the tunnel."""
        # Fit a b-spline to the left wall
        tck_left, _ = splprep(left_wall.T, s=0)
        left_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_left)))

        # Fit a b-spline to the right wall
        tck_right, _ = splprep(right_wall.T, s=0)
        right_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_right)))

        if visualize:
            # Plot the fitted B-spline curve as a line
            self._visualize_b_spline_line(left_wall, color="blue", width=3)
            self._visualize_b_spline_line(right_wall, color="red", width=3)


    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """Classify the point as left or right wall based on the B-spline curve."""
        
        # Extract the x and z coordinates of the point
        px, _, pz = point

        # Project the curve points to the xz-plane
        curve_xz = self.control_points_adjusted[:, [0, 2]]  # Only keep x and z coordinates

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

    def _generate_splines_at_y_values(self, epsilon, visualize):
        """Generate splines on walls at different y values."""
        # Slice the tunnel horizontally at different y values
        for y_value in self.hslice_planes:
            # filter the points near the y value within epsilon distance
            points_near_y_value = self.tunnel_points[
                np.where(
                (self.tunnel_points[:, 1] >= y_value - epsilon) & 
                (self.tunnel_points[:, 1] <= y_value + epsilon)
                )]

            left_wall, right_wall = [], []
            for point in points_near_y_value:
                if self._classify_based_on_b_spline(point):
                    left_wall.append(point)
                else:
                    right_wall.append(point)
            
            left_wall = np.array(left_wall)
            right_wall = np.array(right_wall)

            # Fit a B-spline, and visualize the left and right walls at each y value
            self._fit_b_spline_to_walls(left_wall, right_wall, visualize)

    def _get_total_number_of_points(self, data: dict) -> int:
        """Get the total number of points in the dataset."""
        count = 0
        for key in data.keys():
            X = data[key]["X"]
            count += len(X)
        return count

    def _prepare_points_for_rendering(self, data: dict) -> np.ndarray:
        """Prepare the data for rendering."""
        #curve_function = lambda z: z**2
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


    def _curve_points(self):
        """Curve the tunnel points around the center line."""            
        for i, key in enumerate(self.points_dict.keys()):
            # Find the corresponding control point index
            index = int(key*2)
            index = min(index, len(self.control_points_adjusted)-1)
            
            # Get the control point position to curve around
            control_point = self.control_points_adjusted[index]
            
            if index == 0:
                tangent = self.control_points_adjusted[index+1] - self.control_points_adjusted[index]
            elif index == len(self.control_points_adjusted) - 1:
                tangent = self.control_points_adjusted[index] - self.control_points_adjusted[index-1]
            else:
                tangent = self.control_points_adjusted[index+1] - self.control_points_adjusted[index-1]
            
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
        
            
    def visualize_the_tunnel(self):
        """Visualize the tunnel, loading from cache if possible."""

        # Fit a B-spline to the center tunnel line
        self._find_b_spline_of_center_tunnel_line()

        # Load if Parquet file exists
        if os.path.exists(f"{self.folder_path}/tunnel_pointcloud.parquet"):
            self.load_from_parquet(f"{self.folder_path}/tunnel_pointcloud.parquet")
        else:
            self._transform_points()
            self.save_to_parquet(f"{self.folder_path}/tunnel_pointcloud.parquet")
        
        # Add points to plot
        self.plotter.add_points(self.tunnel_points, color="lightblue", point_size=2, label="Tunnel")
        
        # Plot the fitted B-spline curve as a line
        self._visualize_b_spline_line(self.tunnel_center_line_b_spline, color="green", width=3)
        
        # Generate splines (on walls) at different y values 
        self._generate_splines_at_y_values(epsilon=5, visualize=True)
        
        #self._show_curve()


if __name__ == "__main__":
    """curve functions:
    lambda z: z
    lambda z: z**2
    lambda z: 1000 * np.cos(z / 5)
    lambda z: -np.log(z+1)*300
    """

    """ curve_function = lambda z: z
    space_out_factor = 1000

    points_dict = efficient_data_loading("data/Predor Ringo 511869.90-511746.75.xlsx", 0, curve_function, space_out_factor)
    control_points = prepare_control_points(points_dict, space_out_factor, curve_function)
    tunnel_slicer = TunnelSlicer(points_dict, control_points, pv.Plotter(), n_horizontal_slices=10)
    tunnel_slicer.visualize_the_tunnel()

    tunnel_slicer.plotter.add_legend()
    tunnel_slicer.plotter.show_axes()
    tunnel_slicer.plotter.camera.up = (0, 1, 0)
    tunnel_slicer.plotter.camera.zoom(1.5)
    tunnel_slicer.plotter.show() """