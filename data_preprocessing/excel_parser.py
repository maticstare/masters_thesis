import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, Union, Callable

def parse_excel_to_points_dict(file_path: str, sheet_name: Union[int, str] = 0, space_out_factor: int = 1000) -> Dict[float, Dict[str, pd.Series]]:
    """
    Parse Excel file to dictionary of point coordinates organized by distance.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet name or index to read from
        space_out_factor: Multiplier for spacing out Z coordinates
        
    Returns:
        Dictionary mapping distances to coordinate dictionaries with X, Y, Z series
    """
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    points_dict = {}
    min_distance = data[data.columns[len(data.columns)-4]].iloc[0]

    for i in range(0, len(data.columns)-2, 2):
        column_name = data.columns[i]
        distance = data[column_name].iloc[0] - min_distance
        X = data.iloc[2:, i].dropna().astype(int)
        Y = data.iloc[2:, i+1].dropna().astype(int)
        Z = pd.Series(np.ones(len(X))*distance*space_out_factor)

        points_dict[distance] = {
            "X": X,
            "Y": Y,
            "Z": Z
        }

    return points_dict

def load_control_points_from_txt(file_path: str) -> Optional[np.ndarray]:
    """
    Load control points from a text file.
    
    Args:
        file_path: Path to the text file containing control points
        
    Returns:
        Array of control points if successful, None if failed
    """
    try:
        control_points = np.loadtxt(file_path)
        print(f"Loaded {len(control_points)} control points from {file_path}")
        return control_points
    except Exception as e:
        print(f"Error loading control points: {e}")
        return None

def prepare_control_points(data: Dict[float, Dict[str, pd.Series]], space_out_factor: int, 
                          curve_function: Optional[Callable[[int], float]] = None, 
                          folder_path: Optional[str] = None) -> np.ndarray:
    """
    Prepare control points either by loading from file or generating from data.
    
    Args:
        data: Dictionary containing point data organized by distances
        space_out_factor: Multiplier for spacing out coordinates
        curve_function: Optional function to generate curved path coordinates
        folder_path: Optional path to folder containing cached control points
        
    Returns:
        Array of control points defining the track path
    """
    # Load control points from a file if it exists
    if folder_path:
        control_points_file = f"{folder_path}/control_points.txt"
        if os.path.exists(control_points_file):
            loaded_points = load_control_points_from_txt(control_points_file)
            if loaded_points is not None:
                return loaded_points
    
    # Else generate control points from data
    sorted_keys = sorted(list(data.keys()))
    control_points = np.zeros((len(sorted_keys), 3))
    
    min_key = min(sorted_keys)
    for i, key in enumerate(sorted_keys):
        normalized_key = key - min_key
        control_points[i, 2] = normalized_key*space_out_factor
        control_points[i, 0] = curve_function(i) if curve_function else 0
        
    print("Generated control points from data")
    return control_points

def parse_to_parquet(file_path: str, sheet_name: Union[int, str] = 0, space_out_factor: int = 1000) -> None:
    """
    Parse Excel file and save as Parquet for efficient loading.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet name or index to read from
        space_out_factor: Multiplier for spacing out Z coordinates
    """
    data = parse_excel_to_points_dict(file_path, sheet_name, space_out_factor)
    rows = []
    for distance, coords in data.items():
        for i in range(len(coords["X"])):
            rows.append((distance, coords["X"].iloc[i], coords["Y"].iloc[i], coords["Z"].iloc[i]))
    df = pd.DataFrame(rows, columns=["distance", "X", "Y", "Z"])
    
    df.to_parquet(file_path.replace(".xlsx", ".parquet"), engine="pyarrow", compression="snappy")

def read_parquet(file_path: str) -> Dict[float, Dict[str, pd.Series]]:
    """
    Read point data from Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        
    Returns:
        Dictionary mapping distances to coordinate dictionaries with X, Y, Z series
    """
    df_loaded = pd.read_parquet(file_path)
    data = {}
    for distance in df_loaded["distance"].unique():
        data[distance] = {
            "X": df_loaded[df_loaded["distance"] == distance]["X"],
            "Y": df_loaded[df_loaded["distance"] == distance]["Y"],
            "Z": df_loaded[df_loaded["distance"] == distance]["Z"]
        }
    return data

def efficient_data_loading(file_path: str, sheet_name: Union[int, str] = 0, space_out_factor: int = 1000) -> Dict[float, Dict[str, pd.Series]]:
    """
    Efficiently load data by using cached Parquet file or creating it from Excel.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Sheet name or index to read from
        space_out_factor: Multiplier for spacing out Z coordinates
        
    Returns:
        Dictionary mapping distances to coordinate dictionaries with X, Y, Z series
    """
    try:
        return read_parquet(file_path.replace(".xlsx", ".parquet"))
    except:
        parse_to_parquet(file_path, sheet_name, space_out_factor)
        return read_parquet(file_path.replace(".xlsx", ".parquet"))