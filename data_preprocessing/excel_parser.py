import pandas as pd
import numpy as np

def parse_excel_to_points_dict(file_path, sheet_name=0):
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    dict = {}

    min_distance = 511746.5
    space_out_factor = 1000

    for i in range(0, len(data.columns)-2, 2):
        column_name = data.columns[i]
        distance = data[column_name].iloc[0] - min_distance
        X = data.iloc[2:, i].dropna().astype(int)
        Y = data.iloc[2:, i+1].dropna().astype(int)
        Z = pd.Series(np.ones(len(X))*distance*space_out_factor)

        dict[distance] = {
            "X": X,
            "Y": Y,
            "Z": Z
        }
    
    # Curve X
    curve_function = lambda z: 1000 * np.cos(z / 5)#lambda z: z**2
    for j,key in enumerate(dict.keys()):
        X = dict[key]["X"]
        for i in range(len(X)):
            X.iloc[i] += curve_function(j)    

    return dict


def prepare_control_points(data: dict, space_out_factor: int, curve_function: callable = None):
    control_points = np.zeros((len(data.keys()), 3))
    min_key = min(data.keys())
    for i, key in enumerate(data.keys()):
        normalized_key = key - min_key
        control_points[i, 2] = normalized_key*space_out_factor
        control_points[i, 0] = curve_function(i) if curve_function else 0
    return control_points


#curve_function = lambda z: 1000 * np.cos(z / 5)#lambda z: -np.log(z+1)*300