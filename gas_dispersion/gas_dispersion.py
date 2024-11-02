import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DiffusionSimulator:
    """
    A class to simulate gas diffusion in a 2D grid using a finite difference method.

    Attributes:
    -----------
    grid_size : int
        The size of the 2D grid (e.g., 50x50).
    time_steps : int
        The number of time steps for the simulation.
    D : float
        The diffusion coefficient (how quickly gas diffuses).
    source_strength : float
        The concentration of gas at the center of the grid (constant source).
    dx : float
        The spatial step size (distance between grid points).
    dt : float
        The time step size, computed based on the diffusion coefficient and spatial step size.
    grid : np.ndarray
        A 2D numpy array representing the gas concentration at each grid point.
    center : int
        The index of the center of the grid.
    fig, ax : Matplotlib figure and axis
        Used for creating the visual representation of the grid.
    im : AxesImage
        The Matplotlib image object that displays the grid.

    Methods:
    --------
    diffuse():
        Compute the diffusion for one time step using the finite difference method.
    update(frame):
        Update the grid by performing one diffusion step and replenishing the gas source at the center.
    run_simulation():
        Run the diffusion simulation and visualize the result with animation.
    """

    def __init__(self, grid_size=50, time_steps=200, D=0.1, source_strength=100, dx=0.1):
        """
        Initialize the simulation parameters and create the grid.

        Parameters:
        -----------
        grid_size : int, optional
            The size of the 2D grid, default is 50.
        time_steps : int, optional
            The number of time steps for the simulation, default is 200.
        D : float, optional
            The diffusion coefficient, default is 0.1.
        source_strength : float, optional
            The concentration of gas at the center (constant source), default is 100.
        dx : float, optional
            The spatial step size (distance between grid points), default is 0.1.
        """
        # Initialize grid size (e.g., 50x50)
        self.grid_size = grid_size
        # Set number of time steps for the simulation
        self.time_steps = time_steps
        # Set the diffusion coefficient, which controls the rate of diffusion
        self.D = D
        # Set the concentration value of the gas source at the center of the grid
        self.source_strength = source_strength
        # Set the spatial step size, representing the distance between adjacent grid points
        self.dx = dx
        # Calculate time step size for stability in finite difference simulations
        self.dt = (dx**2) / (4 * D)  # Ensures numerical stability in diffusion

        # Initialize the grid as a 2D array with all values set to zero
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # Calculate the center index of the grid where the gas source is located
        self.center = self.grid_size // 2

        # Set up a Matplotlib figure and axis for visualizing the grid
        self.fig, self.ax = plt.subplots()
        # Display the initial grid using the 'plasma' colormap, with the source strength as max color scale
        self.im = self.ax.imshow(self.grid, cmap='plasma', origin='lower', vmin=0, vmax=source_strength, interpolation="bicubic")
        # Add a color bar to the side of the plot to indicate concentration levels
        plt.colorbar(self.im)

    def diffuse(self):
        """
        Compute the diffusion for one time step using the finite difference method.

        This method applies a discrete approximation of the diffusion equation
        to update the grid values based on the neighboring grid points.
        """
        # Create a copy of the grid to store updated values after diffusion step
        new_grid = np.copy(self.grid)
        
        # Loop through each grid cell, excluding the edges to prevent out-of-bounds indexing
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # Update each cell based on the concentration of its neighboring cells
                new_grid[i, j] = self.grid[i, j] + self.D * self.dt / (self.dx ** 2) * (
                    self.grid[i + 1, j] + self.grid[i - 1, j] +
                    self.grid[i, j + 1] + self.grid[i, j - 1] -
                    4 * self.grid[i, j]
                )
        # Replace the old grid with the new grid values after the diffusion step
        self.grid = new_grid
    
    def update(self, frame):
        """
        Update the grid for each time step during the animation.

        This method performs one diffusion step and replenishes the source concentration
        at the center of the grid. It updates the plot as well.

        Parameters:
        -----------
        frame : int
            The current frame number of the animation (automatically passed by FuncAnimation).
        
        Returns:
        --------
        list
            A list containing the updated image (used for blitting in FuncAnimation).
        """
        # Perform diffusion for the current time step
        self.diffuse()
        # Replenish gas concentration at the center source to maintain a constant source
        self.grid[self.center, self.center] = self.source_strength
        # Update the image array with the new grid values to refresh the visualization
        self.im.set_array(self.grid)
        # Return the updated image object as a list for FuncAnimation
        return [self.im]

    def run_simulation(self):
        """
        Run the diffusion simulation and create an animated plot.

        This method uses FuncAnimation to animate the diffusion process over time.

        Returns:
        --------
        None
        """
        # Create the animation object using FuncAnimation, updating each frame with update function
        ani = FuncAnimation(self.fig, self.update, frames=self.time_steps, interval=50, blit=True)
        # Display the animated diffusion plot
        plt.show()

# Example usage:
if __name__ == '__main__':
    # Create and run the diffusion simulation
    simulation = DiffusionSimulator(50, 200, 1e5, 52655.1813, 0.1)
    simulation.run_simulation()