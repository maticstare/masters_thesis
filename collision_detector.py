import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tunnel_slicer import TunnelSlicer
import pyvista as pv

class CollisionDetector:
    """
    Collision detection system for train wagons in tunnel environment.
    
    Detects safety violations by checking if wagon points are too close to
    tunnel walls or outside the tunnel boundaries.
    """
    
    def __init__(self, tunnel_slicer: TunnelSlicer, safety_margin: float = 200.0) -> None:
        """
        Initialize the collision detection system.
        
        Args:
            tunnel_slicer: TunnelSlicer instance for tunnel geometry
            safety_margin: Additional safety distance from walls in millimeters
        """
        self.tunnel_slicer = tunnel_slicer
        self.safety_margin = safety_margin
        self.collision_history: List[Dict[str, Any]] = []

    def find_closest_point_on_curve(self, point: np.ndarray, curve_points: np.ndarray, y_value: float) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Find closest point on curve and determine which side the point is on.
        
        Args:
            point: 3D point to check [x, y, z]
            curve_points: Array of curve points defining the wall
            y_value: Y-coordinate for the slice
            
        Returns:
            Tuple of (side_value, point_3d, closest_3d) where side_value indicates
            which side of the curve the point is on (positive/negative)
        """
        # curve_points are already the spline points, so find closest directly
        point_2d = np.array([point[0], point[2]])
        curve_2d = curve_points[:, [0, 2]]  # Extract x,z coordinates
        
        # Find closest point by direct distance calculation
        distances = np.linalg.norm(curve_2d - point_2d, axis=1)
        closest_idx = np.argmin(distances)
        closest_2d = curve_2d[closest_idx]
        
        # Calculate tangent at closest point for side determination
        if closest_idx == 0:
            tangent_2d = curve_2d[1] - curve_2d[0]
        elif closest_idx == len(curve_2d) - 1:
            tangent_2d = curve_2d[-1] - curve_2d[-2]
        else:
            # Use central difference for better accuracy
            tangent_2d = curve_2d[closest_idx + 1] - curve_2d[closest_idx - 1]
        
        tangent_2d = tangent_2d / np.linalg.norm(tangent_2d)
        
        # Determine side using cross product
        vector_to_point = point_2d - closest_2d
        side_value = tangent_2d[0] * vector_to_point[1] - tangent_2d[1] * vector_to_point[0]
        
        # Convert to 3D
        point_3d = np.array([point_2d[0], y_value, point_2d[1]])
        closest_3d = np.array([closest_2d[0], y_value, closest_2d[1]])
        
        return side_value, point_3d, closest_3d

    def get_bounding_box_points(self, wagon_vertices: np.ndarray, y_value: float) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get wagon points for collision checking at specified Y-level.
        
        Args:
            wagon_vertices: Array of 8 wagon vertices
            y_value: Y-coordinate for the slice
            
        Returns:
            Dictionary containing left/right side points (back, middle, front)
        """
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

    def check_point_violation(self, point: np.ndarray, point_name: str, wall_points: np.ndarray, side: str, y_value: float, safety_margin: float, frame_number: int) -> Optional[Dict[str, Any]]:
        """
        Check if a single point violates safety constraints.
        
        Args:
            point: 3D point to check
            point_name: Name identifier for the point (e.g., "back", "middle", "front")
            wall_points: Array of wall points for comparison
            side: Wall side ("left" or "right")
            y_value: Y-coordinate for the slice
            safety_margin: Safety distance from walls in millimeters
            frame_number: Current simulation frame number
            
        Returns:
            Violation dictionary if violation detected, None otherwise
        """
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

    def check_collision(self, wagon_vertices: np.ndarray, frame_number: int, safety_margin: float) -> Dict[str, Any]:
        """
        Main collision detection method.
        
        Args:
            wagon_vertices: Array of 8 wagon vertices
            frame_number: Current simulation frame number
            safety_margin: Safety distance from walls in millimeters
            
        Returns:
            Dictionary containing collision detection results including violations,
            closest distances, and safety status
        """
        results = {
            'safety_violation_detected': False,
            'violations': [],
            'closest_distance': float('inf'),
            'frame_number': frame_number
        }

        for y_value, (left_wall, right_wall) in self.tunnel_slicer.wall_points.items():
            wagon_points = self.get_bounding_box_points(wagon_vertices, y_value)
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

    def visualize_violation(self, plotter: pv.Plotter, violation: Dict[str, Any]) -> None:
        """
        Draw violation visualization in the plotter.
        
        Args:
            plotter: PyVista plotter instance
            violation: Violation dictionary containing point and distance information
        """
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

    def get_collision_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collisions detected during simulation.
        
        Returns:
            Dictionary containing collision statistics including total counts,
            affected frames, minimum distances, and violation types
        """
        all_violations = [v for collision in self.collision_history for v in collision.get('violations', [])]
        
        return {
            'total_collisions': len(self.collision_history),
            'total_violations': len(all_violations),
            'collision_frames': [c['frame_number'] for c in self.collision_history],
            'min_distance_overall': min([c['closest_distance'] for c in self.collision_history], default=float('inf')),
            'violation_types': list(set(v['violation_type'] for v in all_violations)),
            'affected_wagon_points': list(set(v['wagon_point_name'] for v in all_violations))
        }