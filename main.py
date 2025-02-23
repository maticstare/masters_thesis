from tunnel_slicer import TunnelSlicer
from data_preprocessing.excel_parser import *

"""curve functions:
lambda z: z**2
lambda z: 1000 * np.cos(z / 5)
lambda z: -np.log(z+1)*300
"""

curve_function = lambda z: 1000 * np.cos(z / 5)
space_out_factor = 1000

points_dict = parse_excel_to_points_dict("data/Predor Ringo 511869.90-511746.75.xlsx", 0, curve_function)

control_points = prepare_control_points(points_dict, space_out_factor, curve_function)

tunnel_slicer = TunnelSlicer(points_dict, control_points, n_horizontal_slices=10)

tunnel_slicer.visualize()