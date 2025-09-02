import numpy as np
import pyvista as pv
from collision_detector import CollisionDetector
from train_generator import Wagon
from tunnel_slicer import TunnelSlicer

class Simulation:
    """
    Main simulation controller for wagon movement through tunnel.
    
    Handles wagon positioning, camera control, collision detection,
    and visualization/recording of the simulation.
    """

    def __init__(self, plotter: pv.Plotter, control_points: np.ndarray, wagon: Wagon, 
                 tunnel_slicer: TunnelSlicer, control_points_offset: int = 0, 
                 export_mp4: bool = False, stop_on_safety_violation: bool = True, 
                 safety_margin: float = 300.0) -> None:
        """
        Initialize the simulation.
        
        Args:
            plotter: PyVista plotter for visualization
            control_points: Array of points defining the track path
            bounding_box: Wagon object for simulation
            tunnel_slicer: TunnelSlicer object for tunnel geometry
            control_points_offset: Offset to align control points with track in millimeters
            export_mp4: Whether to export video as MP4 file
            stop_on_safety_violation: Whether to stop simulation on collision detection
            safety_margin: Safety distance from tunnel walls in millimeters
        """
        self.plotter = plotter
        self.control_points = control_points.copy()
        self.wagon = wagon
        self.tunnel_slicer = tunnel_slicer
        self.control_points_offset = control_points_offset
        self.export_mp4 = export_mp4
        self.stop_on_safety_violation = stop_on_safety_violation
        self.safety_margin = safety_margin
        
        self.collision_detector = CollisionDetector(tunnel_slicer, safety_margin=safety_margin)
        self._setup_simulation()
    
    def _setup_simulation(self) -> None:
        """Initialize the simulation environment and visualization."""
        self.control_points[:, 0] += self.control_points_offset

        bounding_box_opacity = 1
        if self.wagon.train_model:
            self.plotter.add_mesh(self.wagon.train_model, color="gray", show_edges=True)
            bounding_box_opacity = 0.3

        self.plotter.add_mesh(self.wagon.bounding_box, color=self.wagon.color, show_edges=True, opacity=bounding_box_opacity)

        self.plotter.add_points(self.control_points, color="red", point_size=5, label="Control Points")
        
        self.plotter.show(interactive_update=True)
        self.plotter.disable()
        
        if self.export_mp4:
            self.plotter.open_movie("videos/tunnel.mp4")
    
    def _update_camera(self, i: int, distance_back: float = 2.0, height_above: float = 1.0) -> None:
        """
        Update camera position to follow the wagon.
        
        Args:
            i: Current control point index
            distance_back: Distance multiplier behind wagon (relative to wagon depth)
            height_above: Height multiplier above wagon (relative to wagon height)
        """
        bounding_box_center = self.wagon.center
        wheelbase = self.wagon.depth * (1 - 2 * self.wagon.wheel_offset)
        _, _, forward, up, _ = self.wagon._calculate_orthogonal_coordinate_system(
            self.control_points, i, wheelbase
        )
        camera_position = (bounding_box_center -
                          forward * (self.wagon.depth * distance_back) +
                          up * (self.wagon.height * height_above))
        
        self.plotter.camera.position = camera_position
        self.plotter.camera.focal_point = bounding_box_center
        self.plotter.camera.up = up
    
    def run(self) -> None:
        """
        Execute the main simulation loop.
        
        Moves wagon along control points, checks for collisions,
        updates visualization, and handles video recording. The simulation
        can terminate early due to safety violations or reaching track end.
        """
        for i in range(len(self.control_points) - 1):
            terminate_simulation = self.wagon.update_position(self.control_points, i)

            self._update_camera(i, distance_back=2.0, height_above=0.8)

            # Perform collision detection for bounding box
            collision_result = self.collision_detector.check_collision(
                self.wagon.bounding_box.points, i, self.safety_margin
            )
            
            safety_violation = False
            if collision_result['safety_violation_detected']:
                print(f"Safety violation detected for Wagon at frame {i}!")
                for violation in collision_result['violations']:
                    print(f"  - Violation: {violation}")
                safety_violation = True
            
            # Stop simulation if safety violation occurs
            if safety_violation and self.stop_on_safety_violation:
                print("Simulation stopped due to safety violation.")
                #add p0 and p1
                self.plotter.add_points(self.wagon.p0.reshape(1, -1), color="orange", point_size=10)
                self.plotter.add_points(self.wagon.p1.reshape(1, -1), color="orange", point_size=10)
                break
            
            # Stop simulation if wagon reaches the end of control points
            if terminate_simulation:
                print("Wagon has reached the end of the track.")
                break
            
            # Update visualization and record frame
            self.plotter.update()
            if self.export_mp4:
                self.plotter.write_frame()
        
        if self.export_mp4:
            self.plotter.close()