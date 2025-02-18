import pandas as pd
import numpy as np

def parse_excel(file_path, sheet_name=0):
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    dict = {}

    for i in range(0, len(data.columns)-2, 2):
        column_name = data.columns[i]
        distance = data[column_name].iloc[0]
        X = data.iloc[2:, i].dropna().astype(int)
        Y = data.iloc[2:, i+1].dropna().astype(int)

        dict[distance] = {
            "X": X,
            "Y": Y
        }
    return dict


def _get_total_number_of_points(data):
    count = 0
    for key in data.keys():
        X = data[key]["X"]
        count += len(X)
    return count


def prepare_data(data):
    curve_function = lambda z: 1000 * np.cos(z / 5)#lambda z: -np.log(z+1)*300
    # Prepare the tunnel pointcloud
    N = _get_total_number_of_points(data)
    space_out_factor = 1000
    points = np.zeros((N, 3))
    min_key = min(data.keys())
    index = 0
    for j, key in enumerate(data.keys()):
        normalized_key = key - min_key
        X = data[key]["X"]
        Y = data[key]["Y"]
        for i in range(len(X)):
            points[i+index] = np.array([
                X.iloc[i],#+curve_function(j),
                Y.iloc[i],
                normalized_key*space_out_factor
            ])
        index += len(X)
    
    # Prepare the control points
    control_points = np.zeros((len(data.keys()), 3))
    for i, key in enumerate(data.keys()):
        normalized_key = key - min_key
        control_points[i, 2] = normalized_key*space_out_factor
        #control_points[i, 0] = curve_function(i) # curve the control points
    return points, control_points #np.fromiter(data.keys(), dtype=float)*space_out_factor