import pyvista as pv
import numpy as np
import time
from scipy.interpolate import splprep, splev
from scipy.optimize import root_scalar
from tunnel_slicer import *
from data_preprocessing.excel_parser import *

class TrainWagon:
    def __init__(self, width=1.0, height=1.0, depth=1.0, center=(0, 0, 0), color="white"):
        """
        Initialize the TrainWagon object.
        
        :param width: Width of the wagon.
        :param height: Height of the wagon.
        :param depth: Depth of the wagon.
        :param center: Center position of the wagon (x, y, z).
        :param color: Color of the wagon.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.center = center
        self.color = color
    
    def create_mesh(self):
        """Creates and returns a PyVista train wagon mesh."""
        wagon = pv.Cube(center=self.center, x_length=self.width, y_length=self.height, z_length=self.depth)
        wagon.color = self.color
        return wagon


def get_new_positions(control_points, i, wagon):
        width, height, depth = wagon.width, wagon.height, wagon.depth
        p0 = control_points[i]
        p1 = get_p1(control_points[i], control_points, depth)
        if p1[2] < p0[2]:
            return

        new_positions = [
            p0 + np.array([-width/2, 0, 0]),
            p1 + np.array([-width/2, 0, 0]),
            p1 + np.array([-width/2, height, 0]),
            p0 + np.array([-width/2, height, 0]),
            p0 + np.array([width/2, 0, 0]),
            p0 + np.array([width/2, height, 0]),
            p1 + np.array([width/2, height, 0]),
            p1 + np.array([width/2, 0, 0])
        ]
        return np.array(new_positions)


def get_p1(point, control_points, radius):
    x_c, z_c = point[0], point[2]
    x_points, z_points = control_points[:, 0], control_points[:, 2]

    # Fit a parametric B-spline (tck contains the knots, coefficients, and degree)
    tck, _ = splprep([x_points, z_points], s=0)


    # Function to compute the intersection equation
    def intersection_function(t):
        x, z = splev(t, tck)
        return (x - x_c) ** 2 + (z - z_c) ** 2 - radius ** 2

    # Find intersection points by checking sign changes in the function
    t_vals = np.linspace(0, 1, 1000)
    intersection_points = []

    for i in range(len(t_vals) - 1):
        t1, t2 = t_vals[i], t_vals[i + 1]
        if intersection_function(t1) * intersection_function(t2) < 0:  # Sign change means root exists
            root = root_scalar(intersection_function, bracket=[t1, t2]).root
            intersection_points.append(splev(root, tck))  # Store intersection coordinates

    #take the point with highest z value
    intersection_points = np.array(intersection_points)
    x, z = intersection_points[np.argmax(intersection_points[:, 1])]
    return np.array([x, 0, z])


def simulate_wagon_movement(plotter, control_points, wagon, control_points_offset=0, speed=0.1, export_mp4=False):
    """Simulate the movement of a train wagon along a tunnel."""
    control_points[:, 0] += control_points_offset
    wagon_mesh = wagon.create_mesh()
    plotter.add_mesh(wagon_mesh, color=wagon.color, show_edges=True)
    plotter.add_points(control_points, color="red", point_size=5)
    plotter.show(interactive_update=True)
    plotter.show_axes()
    plotter.disable()
    
    if export_mp4: plotter.open_movie("videos/tunnel.mp4")    

    for i in range(len(control_points)):
        new_positions = get_new_positions(control_points, i, wagon)
        if new_positions is None:
            break
        wagon_mesh.points = new_positions
        plotter.camera.focal_point = wagon_mesh.center + np.array([0, wagon.height/2, 0])
        plotter.camera.position = plotter.camera.focal_point + np.array([0, 1000, -10500])

        plotter.update()
        if export_mp4: plotter.write_frame()
        time.sleep(speed)
    
