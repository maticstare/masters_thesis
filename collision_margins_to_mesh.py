import json
import numpy as np
import pyvista as pv


visualize = True

#tunnel_name = "globoko"
tunnel_name = "ringo"

with open(f"data/{tunnel_name}/collision_margins.json", "r") as f:
    collision_margins = json.load(f)

train_width = collision_margins["train_width"]
train_depth = collision_margins["train_depth"]

del collision_margins["train_width"]
del collision_margins["train_depth"]

y_levels = [float(y) for y in collision_margins.keys()]


points = np.array([
    [
        [-train_width/2, y, 0],
        [-train_width/2, y, train_depth/2],
        [-train_width/2, y, train_depth],
        [train_width/2, y, train_depth],
        [train_width/2, y, train_depth/2],
        [train_width/2, y, 0]
    ] for y in y_levels
])

for i, y in enumerate(collision_margins.keys()):
    for j, point_name in enumerate(["right_back", "right_middle", "right_front", "left_front", "left_middle", "left_back"]):
        margin = collision_margins[y][point_name]
        if margin > 0:
            if "right" in point_name:
                points[i, j, 0] += margin
            elif "left" in point_name:
                points[i, j, 0] -= margin

n_layers, n_per_layer = points.shape[:2]
points_flat = points.reshape(-1, 3)

# Side faces
faces = []
for k in range(n_layers-1):
    base = k * n_per_layer
    next_base = (k+1) * n_per_layer
    for i in range(n_per_layer):
        next_i = (i+1) % n_per_layer
        faces.extend([4, base+i, base+next_i, next_base+next_i, next_base+i])

# Bottom and top faces
faces.extend([n_per_layer] + list(range(n_per_layer))[::-1])
faces.extend([n_per_layer] + list(range((n_layers-1)*n_per_layer, n_layers*n_per_layer)))

mesh = pv.PolyData(points_flat, faces)

#save mesh to file
mesh.save(f"data/{tunnel_name}/shaved_off_wagon_model.vtk")

if visualize:
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color="lightblue", opacity=0.7)
    #plotter.add_points(points_flat, color="red", point_size=12)
    #plotter.add_point_labels(points_flat, [str(i+1) for i in range(len(points_flat))], point_size=0, font_size=14, text_color="black")
    plotter.camera_position = [(0, 0, -train_depth*5), (0, 0, 0), (0, 1, 0)]
    plotter.show_axes()
    plotter.show()