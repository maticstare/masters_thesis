import numpy as np
import pyvista as pv

class TunnelPointCloudGenerator:
    def __init__(
        self,
        tunnel_height=10,
        tunnel_width=8,
        track_width=0.3,
        track_gap=1.5,
        track_height=0.3,
        tunnel_length=20,
        density=500,
        offset_scale=0.25,
        curve_radius=30,
        curve_function=None
    ):
        self.tunnel_height = tunnel_height
        self.tunnel_width = tunnel_width
        self.track_width = track_width
        self.track_gap = track_gap
        self.track_height = track_height
        self.tunnel_length = tunnel_length
        self.density = density
        self.offset_scale = offset_scale
        self.curve_radius = curve_radius
        self.curve_function = curve_function if curve_function is not None else lambda z: z / self.curve_radius

    def _add_random_offset(self, points: np.ndarray) -> np.ndarray:
        """Add random offsets to points for realism."""
        offsets = np.random.uniform(-self.offset_scale, self.offset_scale, points.shape)
        return points + offsets

    def _curve_points(self, points: np.ndarray) -> np.ndarray:
        """Curve the tunnel points using a specified function (default: circular arc)."""            
        curved_points = []
        for point in points:
            x, y, z = point
            angle = self.curve_function(z)
            new_z = z * np.cos(angle) - x * np.sin(angle)
            new_x = z * np.sin(angle) + x * np.cos(angle)
            curved_points.append([new_x, y, new_z])
        return np.array(curved_points)

    def _generate_tunnel_points(self) -> np.ndarray:
        """Generate the main tunnel point cloud."""
        theta = np.linspace(np.pi, 2 * np.pi, self.density)
        semi_circle_x = (self.tunnel_width / 2) * np.cos(theta)
        semi_circle_y = self.tunnel_height * np.sin(theta)

        # Combine with straight walls
        x = np.concatenate([semi_circle_x, [-self.tunnel_width / 2, self.tunnel_width / 2]])
        y = np.concatenate([semi_circle_y, [0, 0]])
        z = np.linspace(0, self.tunnel_length, self.density)

        tunnel_points = np.array([[x_, y_, z_] for z_ in z for x_, y_ in zip(x, y)])
        tunnel_points = self._add_random_offset(tunnel_points)
        tunnel_points = self._curve_points(tunnel_points)
        return tunnel_points

    def _generate_floor_points(self) -> np.ndarray:
        """Generate the floor point cloud."""
        floor_x = np.linspace(-self.tunnel_width / 2, self.tunnel_width / 2, self.density)
        floor_z = np.linspace(0, self.tunnel_length, self.density)
        floor_points = np.array([[x_, 0, z_] for z_ in floor_z for x_ in floor_x])
        floor_points = self._add_random_offset(floor_points)
        floor_points = self._curve_points(floor_points)
        return floor_points

    def _generate_rail_points(self, x_offset: float) -> np.ndarray:
        """Generate the rail tracks as point clouds."""
        rail_x = np.linspace(-self.track_width / 2, self.track_width / 2, self.density // 10)
        rail_y = np.linspace(-self.track_height * 0.5, -self.track_height, self.density // 20)
        z = np.linspace(0, self.tunnel_length, self.density)
        rail_points = np.array([[x_ + x_offset, y_, z_] for z_ in z for x_ in rail_x for y_ in rail_y])
        rail_points = self._add_random_offset(rail_points)
        rail_points = self._curve_points(rail_points)
        return rail_points

    def generate_pointcloud(self):
        """Generate the combined point cloud for the tunnel, floor, and rails."""
        tunnel_points = self._generate_tunnel_points()
        floor_points = self._generate_floor_points()
        left_rail_points = self._generate_rail_points(-self.track_gap / 2)
        right_rail_points = self._generate_rail_points(self.track_gap / 2)

        all_points = np.vstack([tunnel_points, floor_points, left_rail_points, right_rail_points])
        return all_points, tunnel_points, floor_points, left_rail_points, right_rail_points

    def visualize(self):
        """Visualize the generated point cloud using PyVista."""
        _, tunnel_points, floor_points, left_rail_points, right_rail_points = self.generate_pointcloud()

        plotter = pv.Plotter()
        plotter.add_points(tunnel_points, color="lightblue", point_size=2, label="Tunnel")
        plotter.add_points(floor_points, color="gray", point_size=2, label="Floor")
        plotter.add_points(left_rail_points, color="brown", point_size=5, label="Left Rail")
        plotter.add_points(right_rail_points, color="brown", point_size=5, label="Right Rail")
        plotter.add_legend()
        plotter.show()

if __name__ == "__main__":
    curve_function = lambda z: 0.35 * np.cos(z / 5)
    generator = TunnelPointCloudGenerator(curve_function=curve_function)
    generator.visualize()
