import numpy as np
import pyvista as pv
from typing import Optional
from collision_detector import CollisionDetector
from train_generator import Train
from tunnel_slicer import TunnelSlicer

class Simulation:
    """
    Main simulation controller for train movement through tunnel.
    
    Handles train positioning, camera control, collision detection,
    and visualization/recording of the simulation.
    """

    def __init__(self, plotter: pv.Plotter, control_points: np.ndarray, train: Train, 
                 tunnel_slicer: TunnelSlicer, control_points_offset: int = 0, 
                 export_mp4: bool = False, stop_on_safety_violation: bool = True, 
                 safety_margin: float = 300.0) -> None:
        """
        Initialize the simulation.
        
        Args:
            plotter: PyVista plotter for visualization
            control_points: Array of points defining the track path
            train: Train object containing wagons
            tunnel_slicer: TunnelSlicer object for tunnel geometry
            control_points_offset: Offset to align control points with track in millimeters
            export_mp4: Whether to export video as MP4 file
            stop_on_safety_violation: Whether to stop simulation on collision detection
            safety_margin: Safety distance from tunnel walls in millimeters
        """
        self.plotter = plotter
        self.control_points = control_points.copy()
        self.train = train
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
        
        for i, wagon in enumerate(self.train.wagons):
            wagon_mesh = self.train.wagon_meshes[i]
            self.plotter.add_mesh(wagon_mesh, color=wagon.color, show_edges=True, label=f"Wagon {i+1}")
        
        self.plotter.add_points(self.control_points, color="red", point_size=5, label="Control Points")
        
        self.plotter.show(interactive_update=True)
        self.plotter.disable()
        
        if self.export_mp4:
            self.plotter.open_movie("videos/tunnel.mp4")
    
    def _update_camera(self, i: int, distance_back: float = 2.0, height_above: float = 1.0) -> None:
        """
        Update camera position to follow the control wagon.
        
        Args:
            i: Current control point index
            distance_back: Distance multiplier behind wagon (relative to wagon depth)
            height_above: Height multiplier above wagon (relative to wagon height)
        """
        control_wagon = self.train.wagons[0]
        wagon_center = control_wagon.center
        _, _, forward, up, _ = control_wagon._calculate_orthogonal_coordinate_system(
            self.control_points, i, control_wagon.depth, self.train
        )
        
        camera_position = (wagon_center - 
                          forward * (control_wagon.depth * distance_back) + 
                          up * (control_wagon.height * height_above))
        
        self.plotter.camera.position = camera_position
        self.plotter.camera.focal_point = wagon_center
        self.plotter.camera.up = up
    
    def run(self) -> None:
        """
        Execute the main simulation loop.
        
        Moves train along control points, checks for collisions,
        updates visualization, and handles video recording. The simulation
        can terminate early due to safety violations or reaching track end.
        """
        for i in range(len(self.control_points) - 1):
            # Update all wagon positions
            terminate_simulation = self.train.update_wagon_positions(self.control_points, i)
            
            # Update camera to follow control wagon
            self._update_camera(i, distance_back=2.0, height_above=0.8)
            
            # Perform collision detection for all wagons
            safety_violation = False
            for wagon_idx, wagon_mesh in enumerate(self.train.wagon_meshes):
                collision_result = self.collision_detector.check_collision(
                    wagon_mesh.points, i, self.safety_margin
                )
                
                # Report any safety violations
                if collision_result['safety_violation_detected']:
                    print(f"Safety violation detected for Wagon {wagon_idx + 1} at frame {i}!")
                    for violation in collision_result['violations']:
                        print(f"  - Violation: {violation}")
                    safety_violation = True
            
            # Stop simulation if safety violation occurs
            if safety_violation and self.stop_on_safety_violation:
                print("Simulation stopped due to safety violation.")
                break
            
            # Stop simulation if train reaches the end of control points
            if terminate_simulation:
                print("Train has reached the end of the track.")
                break
            
            # Update visualization and record frame
            self.plotter.update()
            if self.export_mp4:
                self.plotter.write_frame()
        
        if self.export_mp4:
            self.plotter.close()