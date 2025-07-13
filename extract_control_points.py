from svgpathtools import svg2paths
import numpy as np

"""Extract control points from an SVG file and print them in a specific format."""

paths, attributes = svg2paths("control_points.svg")

points_list = []

for path in paths:
    if not path:
        continue

    start = path[0].start
    x = start.real
    y = start.imag
    point = [x, y]
    if point not in points_list:
        points_list.append(point)

    for segment in path:
        end = segment.end
        x = end.real
        y = end.imag
        point = [x, y]
        if point not in points_list:
            points_list.append(point)

points = np.array(points_list)

last_x = points[-1, 0]
points[:, 0] -= last_x
points[:, 0] *= 600
points = points[::-1]

for point, z in zip(points, np.linspace(0, 235500, len(points))):
    print(f"{point[0]:.6f} 0.000000 {z:.6f}")

print(f"Total unique points: {len(points)}")