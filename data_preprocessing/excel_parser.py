import pandas as pd
import numpy as np
import os

def parse_excel_to_points_dict(file_path: str, sheet_name=0, curve_function=None, space_out_factor=1000):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    dict = {}
    min_distance = data[data.columns[len(data.columns)-4]].iloc[0]

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
    for j, key in enumerate(dict.keys()):
        X = dict[key]["X"]
        for i in range(len(X)):
            X.iloc[i] = X.iloc[i] + curve_function(j) if curve_function else X.iloc[i]

    return dict

def load_control_points_from_txt(file_path, space_out_factor=1000):
    try:
        control_points = np.loadtxt(file_path)
        print(f"Loaded {len(control_points)} control points from {file_path}")
        return control_points
    except Exception as e:
        print(f"Error loading control points: {e}")
        return None

def prepare_control_points(data: dict, space_out_factor: int, curve_function: callable = None, folder_path=None):
    # Load control points from a file if it exists
    if folder_path:
        control_points_file = f"{folder_path}/control_points.txt"
        if os.path.exists(control_points_file):
            loaded_points = load_control_points_from_txt(control_points_file)
            if loaded_points is not None:
                return loaded_points
    
    # Else generate control points from data
    control_points = np.zeros((len(data.keys()), 3))
    min_key = min(data.keys())
    for i, key in enumerate(data.keys()):
        normalized_key = key - min_key
        control_points[i, 2] = normalized_key*space_out_factor
        control_points[i, 0] = curve_function(i) if curve_function else 0
    return control_points

def parse_to_parquet(file_path: str, sheet_name=0, curve_function=None, space_out_factor=1000):
    data = parse_excel_to_points_dict(file_path, sheet_name, curve_function, space_out_factor)
    rows = []
    for distance, coords in data.items():
        for i in range(len(coords["X"])):
            rows.append((distance, coords["X"].iloc[i], coords["Y"].iloc[i], coords["Z"].iloc[i]))
    df = pd.DataFrame(rows, columns=["distance", "X", "Y", "Z"])
    
    df.to_parquet(file_path.replace(".xlsx", ".parquet"), engine="pyarrow", compression="snappy")


def read_parquet(file_path):
    df_loaded = pd.read_parquet(file_path)
    data = {}
    for distance in df_loaded["distance"].unique():
        data[distance] = {
            "X": df_loaded[df_loaded["distance"] == distance]["X"],
            "Y": df_loaded[df_loaded["distance"] == distance]["Y"],
            "Z": df_loaded[df_loaded["distance"] == distance]["Z"]
        }
    return data

def efficient_data_loading(file_path: str, sheet_name=0, curve_function=None, space_out_factor=1000):
    try:
        return read_parquet(file_path.replace(".xlsx", ".parquet"))
    except:
        parse_to_parquet(file_path, sheet_name, curve_function, space_out_factor)
        return read_parquet(file_path.replace(".xlsx", ".parquet"))