from lines_generator import TunnelSlicer
from data_preprocessing.excel_parser import *

data = parse_excel("data/Predor Ringo 511869.90-511746.75.xlsx", 0)
points, control_points = prepare_data(data)

tunnel_slicer = TunnelSlicer(points, control_points, n_horizontal_slices=10)
tunnel_slicer.visualize()