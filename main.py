import pyvista as pv
import numpy as np
from tunnel_slicer import TunnelSlicer
from data_preprocessing.excel_parser import efficient_data_loading, prepare_control_points
from train_generator import Wagon, Train
from simulation import Simulation

"""curve functions:
lambda z: 0
lambda z: z**2
lambda z: 1000 * np.cos(z / 5)
lambda z: -np.log(z+1)*300
"""
curve_function = lambda z: 0
space_out_factor = 1000 # assuming 1 unit equals 1 millimeter

tunnel_configs = {
    "ringo": {
        "folder_path": "data/ringo",
        "control_points_offset": 1800,
        "tunnel_center_offset": 3700,
        "train_max_height": 5700,
        "tunnel_slicer_min_height": 300,
        "excel_file": "data/ringo/Predor Ringo 511869.90-511746.75.xlsx"
    },
    "globoko": {
        "folder_path": "data/globoko",
        "control_points_offset": 0,
        "tunnel_center_offset": 0,
        "train_max_height": 4900,
        "tunnel_slicer_min_height": 50,
        "excel_file": "data/globoko/Predor Globoko 471.0-235.5.xlsx"
    }
}

tunnel = "globoko"
#tunnel = "ringo"

selected_config = tunnel_configs[tunnel]
folder_path = selected_config["folder_path"]
control_points_offset = selected_config["control_points_offset"] # align with railway track
tunnel_center_offset = selected_config["tunnel_center_offset"] # align with tunnel center line
tunnel_slicer_min_height = selected_config["tunnel_slicer_min_height"]

points_dict = efficient_data_loading(selected_config["excel_file"], 0, space_out_factor)

control_points = prepare_control_points(points_dict, space_out_factor, curve_function, folder_path)

plotter = pv.Plotter()

train_height = 3900
train_width = 3200
train_depth = 6000
n_horizontal_slices = 30
wall_spline_degree = 3
safety_margin = 300
stop_on_safety_violation = True
export_mp4 = False

assert train_height <= selected_config["train_max_height"], f"Train height {train_height} exceeds maximum height {selected_config['train_max_height']} for tunnel {tunnel}."

tunnel_slicer = TunnelSlicer(points_dict, control_points.copy(), plotter, n_horizontal_slices, train_height, folder_path, control_points_offset, tunnel_center_offset, tunnel_slicer_min_height)
tunnel_slicer.visualize_the_tunnel(wall_spline_degree=wall_spline_degree)


wagon1 = Wagon(width=train_width, height=train_height, depth=train_depth, color="blue")
wagon2 = Wagon(width=train_width, height=train_height, depth=train_depth, color="red")
wagon3 = Wagon(width=train_width, height=train_height, depth=train_depth, color="green")


train = Train(wagons=[wagon1, wagon2], wagon_spacing=500)

simulation = Simulation(
    plotter=plotter,
    control_points=control_points,
    train=train,
    tunnel_slicer=tunnel_slicer,
    control_points_offset=control_points_offset,
    export_mp4=export_mp4,
    stop_on_safety_violation=stop_on_safety_violation,
    safety_margin=safety_margin
)
simulation.run()

plotter.enable()
plotter.show()