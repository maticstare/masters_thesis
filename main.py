from tunnel_slicer import TunnelSlicer
from train_generator import *
import pyvista as pv

"""curve functions:
lambda z: 0
lambda z: z**2
lambda z: 1000 * np.cos(z / 5)
lambda z: -np.log(z+1)*300
"""
curve_function = lambda z: 0
space_out_factor = 1000 # assuming 1 unit equals 1 millimeter


tunnels = {
    "ringo": {
        "folder_path": "data/ringo",
        "control_points_offset": 1800,
        "tunnel_center_offset": 3700,
        "train_max_height": 5700,
        "points_dict": efficient_data_loading(f"data/ringo/Predor Ringo 511869.90-511746.75.xlsx", 0, space_out_factor)
    },
    "globoko": {
        "folder_path": "data/globoko",
        "control_points_offset": 0,
        "tunnel_center_offset": 0,
        "train_max_height": 4900,
        "points_dict": efficient_data_loading(f"data/globoko/Predor Globoko 471.0-235.5.xlsx", 0, space_out_factor)
    }
}

tunnel = "globoko"
#tunnel = "ringo"

folder_path = tunnels[tunnel]["folder_path"]
control_points_offset = tunnels[tunnel]["control_points_offset"] # align with railway track
tunnel_center_offset = tunnels[tunnel]["tunnel_center_offset"] # align with tunnel center line
points_dict = tunnels[tunnel]["points_dict"]

control_points = prepare_control_points(points_dict, space_out_factor, curve_function, folder_path)

plotter = pv.Plotter()

train_height = 4900
train_width = 3000
train_depth = 6000
n_horizontal_slices = 30
wall_spline_degree = 3


assert train_height <= tunnels[tunnel]["train_max_height"], f"Train height {train_height} exceeds maximum height {tunnels[tunnel]['train_max_height']} for tunnel {tunnel}."


tunnel_slicer = TunnelSlicer(points_dict, control_points.copy(), plotter, n_horizontal_slices, train_height, folder_path, control_points_offset, tunnel_center_offset)
tunnel_slicer.visualize_the_tunnel(wall_spline_degree=wall_spline_degree)


wagon = TrainWagon(width=train_width, height=train_height, depth=train_depth, center=(0, 0, 0), color="blue")

simulate_wagon_movement(plotter, control_points, wagon, tunnel_slicer, control_points_offset, speed=0, export_mp4=False, stop_on_safety_violation=True, safety_margin=0)

plotter.enable()
plotter.show()