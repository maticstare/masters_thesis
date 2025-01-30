import numpy as np
import pyvista as pv
from tunnel_generator import TunnelPointCloudGenerator
from scipy.interpolate import splprep, splev
pv.global_theme.allow_empty_mesh = True

class TunnelSlicer:
    def __init__(self, tunnel_points: np.ndarray, n_horizontal_slices: int, n_vertical_slices: int):
        """Initialize the slicer with tunnel points and number of horizontal and vertical slices."""
        self.tunnel_points = tunnel_points
        
        self.n_horizontal_slices = n_horizontal_slices
        self.n_vertical_slices = n_vertical_slices

        self.tunnel_width = np.ptp(self.tunnel_points[:, 0])
        self.tunnel_height = np.ptp(self.tunnel_points[:, 1])
        self.tunnel_length = np.ptp(self.tunnel_points[:, 2])

        self.width_offset = min(self.tunnel_points[:, 0])
        self.height_offset = max(self.tunnel_points[:, 1])
        self.length_offset = min(self.tunnel_points[:, 2])

        self.hslice_planes = np.linspace(1, self.tunnel_height-1, n_horizontal_slices)
        # Since there is a curvature in the tunnel, I will skip a few starting and ending points
        self.vslice_planes = np.linspace(0+3, self.tunnel_length-2, n_vertical_slices)

        self.tunnel_vertical_hyperplane_points = np.zeros((n_vertical_slices, 3))
        self.tunnel_vertical_hyperplane = None

        self.plotter = pv.Plotter()
        

    def _find_points_near_plane(self, point_cloud: np.ndarray, plane: pv.Plane, epsilon=0.1) -> np.ndarray:
        """
        Finds the points from the point cloud that are within epsilon distance from the plane.

        Parameters:
        - point_cloud: np.ndarray, a point cloud containing the points to check.
        - plane: pyvista.Plane, the plane object to compare the points to.
        - epsilon: float, the threshold distance from the plane to consider a point close (default is 0.1).

        Returns:
        - np.ndarray, the points from the point cloud that are within epsilon distance from the plane.
        """

        # Get the normalized normal vector of the plane
        normal = plane.active_normals[0]
        normal = normal / np.linalg.norm(normal)

        # Plane equation Ax + By + Cz + D = 0
        D = -np.dot(normal, plane.center)

        # Calculate the perpendicular distance from each point to the plane
        distances = np.abs(np.dot(point_cloud, normal) + D) / np.linalg.norm(normal)

        # Filter the points within epsilon distance from the plane
        close_points = point_cloud[distances <= epsilon]
        return close_points


    def _slice_the_tunnel_vertically(self, z_value: float) -> pv.Plane:
        """Slice the tunnel vertically at a given z value."""
        return pv.Plane(
                center=(self.tunnel_width/2 + self.width_offset,
                        -self.tunnel_height/2 + self.height_offset,
                        z_value + self.length_offset),
                direction=(0, 0, 1),
                i_size=self.tunnel_width,
                j_size=self.tunnel_height
            )
            

    def _slice_the_tunnel_horizontally(self, y_value: float) -> pv.Plane:
        """Slice the tunnel horizontally at a given y value."""
        return pv.Plane(
                center=(self.tunnel_points[:, 0].mean(),
                        -y_value,
                        self.tunnel_points[:, 2].mean()),
                direction=(0, 1, 0),
                i_size=self.tunnel_length,
                j_size=self.tunnel_width
            )


    def _find_control_points_of_center_tunnel_line(self, visualize=False):
        """Find the control points of the center tunnel line for B-spline fitting."""
        for i, z_value in enumerate(self.vslice_planes):
            plane = self._slice_the_tunnel_vertically(z_value)

            # Visualize the plane if requested
            if visualize:
                self.plotter.add_mesh(plane, color=pv.Color("red"), opacity=0.3)

            # Find points near the plane
            sliced_points = self._find_points_near_plane(self.tunnel_points, plane)

            # Append control point of a slice for the curve fitting
            self.tunnel_vertical_hyperplane_points[i] = [
                np.mean(sliced_points[:, 0]),
                -self.tunnel_height/2 + self.height_offset,
                z_value + self.length_offset]


    def _find_b_spline_of_center_tunnel_line(self):
        """Fit a B-spline to the center tunnel line."""

        tck, _ = splprep(self.tunnel_vertical_hyperplane_points.T, s=0)
        self.tunnel_vertical_hyperplane = np.column_stack((splev(np.linspace(0, 1, 100), tck)))
  

    def _classify_based_on_b_spline(self, point: np.ndarray) -> bool:
        """Classify the point as left or right wall based on the B-spline curve."""
        
        # Extract the x and z coordinates of the point
        px, _, pz = point

        # Project the curve points to the xz-plane
        curve_xz = self.tunnel_vertical_hyperplane_points[:, [0, 2]]  # Only keep x and z coordinates

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


    def _filter_fit_visualize(self, left_wall, right_wall, plane, points_near_plane):
        """ # Fit a b-spline to the left wall
            tck_left, _ = splprep(left_wall.T, s=0)
            left_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_left)))

            # Fit a b-spline to the right wall
            tck_right, _ = splprep(right_wall.T, s=0)
            right_wall = np.column_stack((splev(np.linspace(0, 1, 100), tck_right)))

            #Plot the fitted B-spline curve as a line
            self.plotter.add_lines(left_wall, color="blue", width=3, label="Left Wall")
            self.plotter.add_lines(right_wall, color="red", width=3, label="Right Wall") """
        pass


    def visualize(self):
        """Visualize the generated point cloud using PyVista."""
        self.plotter.add_points(self.tunnel_points, color="lightblue", point_size=2, label="Tunnel")
        
        # Find the control points of the center tunnel line for B-spline fitting
        self._find_control_points_of_center_tunnel_line(visualize=False)
        
        # Fit a B-spline to the center tunnel line
        self._find_b_spline_of_center_tunnel_line()

        # Add control points to the plot
        self.plotter.add_points(self.tunnel_vertical_hyperplane_points, color="red", point_size=5, label="Control Points")

        # Plot the fitted B-spline curve as a line
        self.plotter.add_lines(self.tunnel_vertical_hyperplane, color="green", width=3, label="Fitted Curve")


        # Slice the tunnel horizontally at different y values
        for y_value in self.hslice_planes:
            # Add a plane visualization at each y value
            plane = self._slice_the_tunnel_horizontally(y_value)

            # Find points near the plane
            points_near_plane = self._find_points_near_plane(self.tunnel_points, plane)

            left_wall, right_wall = [], []
            for point in points_near_plane:
                if self._classify_based_on_b_spline(point):
                    left_wall.append(point)
                else:
                    right_wall.append(point)
            
            left_wall = np.array(left_wall)
            right_wall = np.array(right_wall)

            # TODO: Filter the points, fit a B-spline, and visualize the left and right walls at each horizontal slice
            self._filter_fit_visualize(left_wall, right_wall, plane, points_near_plane)
            
            # Visualize the left and right walls
            self.plotter.add_points(left_wall, color="blue", point_size=5)
            self.plotter.add_points(right_wall, color="red", point_size=5)

            # Visualize the plane
            self.plotter.add_mesh(plane, color=pv.Color("red"), opacity=0.3)

        self.plotter.add_legend()
        self.plotter.show_axes()
        self.plotter.show()

if __name__ == "__main__":
    curve_function = lambda z: 0.35 * np.cos(z / 5)
    generator = TunnelPointCloudGenerator(curve_function=curve_function, tunnel_length=20)
    point_cloud, _, _, _, _ = generator.generate_pointcloud()

    slicer = TunnelSlicer(point_cloud, n_vertical_slices=10, n_horizontal_slices=10)
    slicer.visualize()