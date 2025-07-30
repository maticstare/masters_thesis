import numpy as np
from tunnel_slicer import TunnelSlicer
import pyvista as pv
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize_scalar

class CollisionDetector:
    def __init__(self, tunnel_slicer: TunnelSlicer, safety_margin: float = 200.0):
        """
        Initialize the collision detection system.
        
        :param tunnel_slicer: TunnelSlicer instance for tunnel geometry
        :param safety_margin: Additional safety distance from walls (mm)
        """
        self.tunnel_slicer = tunnel_slicer
        self.safety_margin = safety_margin
        self.collision_history = []

    def find_closest_point_on_curve(self, point, curve_points, y_value):
        """Find closest point on curve and determine which side the point is on."""
        x, z = curve_points[:, 0], curve_points[:, 2]
        point_2d = np.array([point[0], point[2]])
        
        # Create spline
        tck, _ = splprep([x, z], s=0)
        
        # Find closest point
        result = minimize_scalar(lambda t: np.linalg.norm(splev(t, tck) - point_2d), bounds=(0, 1))
        closest_2d = np.array(splev(result.x, tck))
        tangent_2d = np.array(splev(result.x, tck, der=1))
        
        # Determine side using cross product
        vector_to_point = point_2d - closest_2d
        side_value = tangent_2d[0] * vector_to_point[1] - tangent_2d[1] * vector_to_point[0]
        
        # Convert to 3D
        point_3d = np.array([point_2d[0], y_value, point_2d[1]])
        closest_3d = np.array([closest_2d[0], y_value, closest_2d[1]])
        
        return side_value, point_3d, closest_3d

    def get_wagon_points(self, wagon_vertices, y_value):
        """Get wagon points for collision checking."""
        points = {
            'right_back': wagon_vertices[0] + [0, y_value, 0],
            'right_front': wagon_vertices[1] + [0, y_value, 0],
            'left_back': wagon_vertices[4] + [0, y_value, 0],
            'left_front': wagon_vertices[7] + [0, y_value, 0]
        }
        
        return {
            'left': {
                'back': points['left_back'],
                'middle': (points['left_back'] + points['left_front']) / 2,
                'front': points['left_front']
            },
            'right': {
                'back': points['right_back'],
                'middle': (points['right_back'] + points['right_front']) / 2,
                'front': points['right_front']
            }
        }

    def check_point_violation(self, point, point_name, wall_points, side, y_value, safety_margin, frame_number):
        """Check if a single point violates safety constraints."""
        side_value, point_3d, closest_wall = self.find_closest_point_on_curve(point, wall_points, y_value)
        distance = np.linalg.norm(point_3d - closest_wall)
        
        # Check for violations
        is_outside = (side == 'left' and side_value >= 0) or (side == 'right' and side_value <= 0)
        too_close = distance < safety_margin
        
        if is_outside or too_close:
            return {
                'wagon_point': point,
                'wagon_point_name': f"{side}_{point_name}",
                'wall_point': closest_wall,
                'wall_type': side,
                'y_slice': y_value,
                'distance': distance if not is_outside else -distance,
                'violation_type': 'outside_tunnel' if is_outside else 'too_close',
                'frame_number': frame_number
            }
        return None

    def check_collision(self, wagon_vertices, frame_number, safety_margin):
        """Main collision detection method."""
        results = {
            'safety_violation_detected': False,
            'violations': [],
            'closest_distance': float('inf'),
            'frame_number': frame_number
        }

        for y_value, (left_wall, right_wall) in self.tunnel_slicer.wall_points.items():
            wagon_points = self.get_wagon_points(wagon_vertices, y_value)
            
            # Check all wagon points
            for side in ['left', 'right']:
                wall_points = left_wall if side == 'left' else right_wall
                
                for point_name, point in wagon_points[side].items():
                    violation = self.check_point_violation(
                        point, point_name, wall_points, side, y_value, safety_margin, frame_number
                    )
                    
                    if violation:
                        results['safety_violation_detected'] = True
                        results['violations'].append(violation)
                        results['closest_distance'] = min(results['closest_distance'], abs(violation['distance']))
                        self.visualize_violation(self.tunnel_slicer.plotter, violation)

        if results['safety_violation_detected']:
            self.collision_history.append(results)
            
        return results

    def visualize_violation(self, plotter, violation):
        """Draw violation visualization."""
        wagon_point = violation['wagon_point']
        wall_point = violation['wall_point']
        distance = violation['distance']
        violation_type = violation['violation_type']
        
        # Create line and set color
        line = pv.Line(wagon_point, wall_point)
        color = "orange" if violation_type == 'outside_tunnel' else "yellow"
        label = f"{violation_type.replace('_', ' ').upper()} ({distance:.1f}mm)"
        
        # Add to plot
        plotter.add_mesh(line, color=color, line_width=6, label=label)
        plotter.add_points(wagon_point.reshape(1, -1), color="red", point_size=20)
        plotter.add_points(wall_point.reshape(1, -1), color="purple", point_size=20)
        plotter.add_point_labels(wagon_point.reshape(1, -1), [label], point_size=0, font_size=14, text_color=color)

    def get_collision_summary(self):
        """Get summary of all collisions."""
        all_violations = [v for collision in self.collision_history for v in collision.get('violations', [])]
        
        return {
            'total_collisions': len(self.collision_history),
            'total_violations': len(all_violations),
            'collision_frames': [c['frame_number'] for c in self.collision_history],
            'min_distance_overall': min([c['closest_distance'] for c in self.collision_history], default=float('inf')),
            'violation_types': list(set(v['violation_type'] for v in all_violations)),
            'affected_wagon_points': list(set(v['wagon_point_name'] for v in all_violations))
        }