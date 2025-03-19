#read all files in the folder Predor Globoko

import os

def parse_txt_to_points_dict(folder_path: str, curve_function=None, space_out_factor=1000):
    files = os.listdir(folder_path)
    dict = {}
    for filename in files:
        #n = int(filename.split("_")[1])
        km = int(filename.split(".")[1]) - 2355
        
        data = []
        with open(folder_path + "/" + filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                x, y = map(int, line.split("\t"))
                z = km * space_out_factor
                data.append((x, y, z))
        #print(data)
        
        dict[km] = data
        #break
        
    return dict 
        
        
        
        
parse_txt_to_points_dict("data/Predor Globoko")