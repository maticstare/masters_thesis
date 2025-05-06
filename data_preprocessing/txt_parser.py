import os

def parse_txt_to_points_dict(folder_path: str, space_out_factor=1000):
    files = os.listdir(folder_path)
    dict = {}
    
    files = sorted(files, key=lambda x: int(x.split("_")[1]), reverse=True)
    #print(files[0:10])
    
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
        
        
if __name__ == "__main__":
    parse_txt_to_points_dict("data/Predor Globoko")