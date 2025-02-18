from tunnel_generator import TunnelPointCloudGenerator
import numpy as np
from lines_generator import TunnelSlicer
from data_preprocessing.excel_parser import *

""" curve_function = lambda z: 0.35 * np.cos(z / 5)
pointcloud = TunnelPointCloudGenerator(curve_function=curve_function)
pointcloud.visualize()
 """

data = parse_excel("C:/Users/matic/Downloads/masters_thesis/data/Predor Ringo 511869.90-511746.75.xlsx", 0)


points, control_points = prepare_data(data)


tunnel_slicer = TunnelSlicer(points, control_points, n_horizontal_slices=10)

tunnel_slicer.visualize()