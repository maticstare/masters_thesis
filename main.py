from tunnel_generator import TunnelPointCloudGenerator
import numpy as np

curve_function = lambda z: 0.35 * np.cos(z / 5)
pointcloud = TunnelPointCloudGenerator(curve_function=curve_function)
pointcloud.visualize()