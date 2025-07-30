import pyvista as pv

class TrainWagon:
    def __init__(self, width: int, height: int, depth: int, center: tuple = (0, 0, 0), color: str = "blue"):
        """Initialize the TrainWagon object."""
        self.width = width
        self.height = height
        self.depth = depth
        self.center = center
        self.color = color
    
    def create_mesh(self) -> pv.PolyData:
        """Creates and returns a PyVista train wagon mesh."""
        return pv.Cube(
            center=self.center, 
            x_length=self.width, 
            y_length=self.height, 
            z_length=self.depth
        )