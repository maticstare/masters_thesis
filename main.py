from tunnel_slicer import TunnelSlicer
from data_preprocessing.excel_parser import *
from train_generator import *
import pyvista as pv

"""curve functions:
lambda z: z
lambda z: z**2
lambda z: 1000 * np.cos(z / 5)
lambda z: -np.log(z+1)*300
"""

curve_function = lambda z: z
space_out_factor = 1000 # assuming 1 unit equals 1 millimeter

points_dict = efficient_data_loading("data/Predor Ringo 511869.90-511746.75.xlsx", 0, curve_function, space_out_factor)
control_points = prepare_control_points(points_dict, space_out_factor, curve_function)

plotter = pv.Plotter()

tunnel_slicer = TunnelSlicer(points_dict, control_points.copy(), plotter, n_horizontal_slices=10)
tunnel_slicer.visualize_the_tunnel()


wagon = TrainWagon(width=2000, height=5000, depth=6000, color="blue")
simulate_wagon_movement(plotter, control_points, wagon, speed=0.1, export_mp4=False)

plotter.enable()    
plotter.show()