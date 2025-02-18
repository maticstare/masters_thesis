import numpy as np
import pyvista as pv
from data_preprocessing.excel_parser import *
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    def __init__(self, tunnel_points: np.ndarray, control_points: np.ndarray, n_horizontal_slices: int):
        """Initialize the slicer with tunnel points and number of horizontal and vertical slices."""
        self.tunnel_points = tunnel_points
        self.control_points = control_points
        self.plotter = pv.Plotter()

        # Shift the control points to the center of the tunnel
        self.tunnel_center_line_points = control_points
        self.tunnel_center_line_points[:, 0] += 1800

        self.tunnel_center_line_b_spline = None
        self.tck = None

        self.n_horizontal_slices = n_horizontal_slices

        # Bounds hardcoded so far ...
        self.hslice_planes = np.linspace(500, 5100.0, n_horizontal_slices)
       
       
    def _find_b_spline_of_center_tunnel_line(self):
        """Fit a B-spline to the center tunnel line."""

        tck, _ = splprep(self.tunnel_center_line_points.T, s=0)
        self.tck = tck
        self.tunnel_center_line_b_spline = np.column_stack((splev(np.linspace(0, 1, 100), tck)))
  

    def _fit_b_spline_to_walls(self, left_wall, right_wall, visualize):
        # Fit a b-spline to the left wall
        tck_left, _ = splprep(left_wall.T, s=0)
        left_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_left)))

        # Fit a b-spline to the right wall
        tck_right, _ = splprep(right_wall.T, s=0)
        right_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_right)))

        if visualize:
            # Plot the fitted B-spline curve as a line
            self.plotter.add_lines(left_wall, color="blue", width=3)
            self.plotter.add_lines(right_wall, color="red", width=3)

    def _generate_splines_at_y_values(self, epsilon, visualize):
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


    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """Classify the point as left or right wall based on the B-spline curve."""
        
        # Extract the x and z coordinates of the point
        px, _, pz = point

        # Project the curve points to the xz-plane
        curve_xz = self.tunnel_center_line_points[:, [0, 2]]  # Only keep x and z coordinates

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

    def _curve_points(self) -> np.ndarray:
        """Curve the tunnel points around the center line."""            
        curved_points = []
        for point in self.tunnel_points:
            x, y, z = point
            dx, _, dz = splev(z, self.tck, der=1)
            theta = np.arctan2(dz, dx)
            new_x = x * np.cos(theta) + z * np.sin(theta)
            new_z = -x * np.sin(theta) + z * np.cos(theta)
            curved_points.append([new_x, y, new_z])
        self.tunnel_points = np.array(curved_points)
        #return np.array(curved_points)


    def visualize(self):
        """Visualize the generated point cloud using PyVista."""
                
        # Fit a B-spline to the center tunnel line
        self._find_b_spline_of_center_tunnel_line()
        
        # Curve the tunnel points around the center line (TODO: not working)
        #self._curve_points()

        # Add the tunnel points to the plot
        self.plotter.add_points(self.tunnel_points, color="lightblue", point_size=2, label="Tunnel")

        # Add control points to the plot
        self.plotter.add_points(self.tunnel_center_line_points, color="red", point_size=5, label="Control Points")

        # Plot the fitted B-spline curve as a line
        self.plotter.add_lines(self.tunnel_center_line_b_spline, color="green", width=3, label="Fitted Curve")

        # Generate splines (on walls) at different y values 
        self._generate_splines_at_y_values(epsilon=5, visualize=False)

        self.plotter.add_legend()
        self.plotter.show_axes()
        self.plotter.show()

if __name__ == "__main__":

    data = parse_excel("data/Predor Ringo 511869.90-511746.75.xlsx", 0)


    points, control_points = prepare_data(data, space_out_factor=1000)


    tunnel_slicer = TunnelSlicer(points, control_points, n_horizontal_slices=20)

    tunnel_slicer.visualize()