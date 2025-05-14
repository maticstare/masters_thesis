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
        "points_dict": efficient_data_loading(f"data/ringo/Predor Ringo 511869.90-511746.75.xlsx", 0, space_out_factor)
    },
    "globoko": {
        "folder_path": "data/globoko",
        "control_points_offset": 0,
        "points_dict": efficient_data_loading(f"data/globoko/Predor Globoko 471.0-235.5.xlsx", 0, space_out_factor)
    }
}

tunnel = "globoko"
#tunnel = "ringo"

folder_path = tunnels[tunnel]["folder_path"]
control_points_offset = tunnels[tunnel]["control_points_offset"]
points_dict = tunnels[tunnel]["points_dict"]



control_points = prepare_control_points(points_dict, space_out_factor, curve_function, folder_path)

plotter = pv.Plotter()


tunnel_slicer = TunnelSlicer(points_dict, control_points.copy(), plotter, 10, folder_path, control_points_offset)
tunnel_slicer.visualize_the_tunnel()


wagon = TrainWagon(width=2000, height=5000, depth=6000, color="blue")

simulate_wagon_movement(plotter, control_points, wagon, control_points_offset=control_points_offset, speed=0.001, export_mp4=False)

plotter.enable()
plotter.show()