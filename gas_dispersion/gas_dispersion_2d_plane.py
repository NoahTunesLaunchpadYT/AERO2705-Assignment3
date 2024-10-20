import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DiffusionSimulation:
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
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.D = D
        self.source_strength = source_strength
        self.dx = dx
        self.dt = (dx**2) / (4 * D)  # Time step size calculated for stability in the finite difference method
        
        # Initialize the grid for gas concentration (all zeros initially)
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.center = self.grid_size // 2  # Center of the grid

        # Set up the plot for visualizing the grid
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid, cmap='plasma', origin='lower', vmin=0, vmax=100, interpolation="bicubic")
        plt.colorbar(self.im)

    def diffuse(self):
        """
        Compute the diffusion for one time step using the finite difference method.

        This method applies a discrete approximation of the diffusion equation
        to update the grid values based on the neighboring grid points.
        """
        new_grid = np.copy(self.grid)  # Create a copy to store updated values
        
        # Loop through the grid points (excluding the boundaries)
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # Discrete approximation of the diffusion equation
                new_grid[i, j] = self.grid[i, j] + self.D * self.dt / (self.dx ** 2) * (
                    self.grid[i + 1, j] + self.grid[i - 1, j] +
                    self.grid[i, j + 1] + self.grid[i, j - 1] -
                    4 * self.grid[i, j]
                )
        self.grid = new_grid  # Update the grid with the new values
    
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
        self.diffuse()  # Perform diffusion for the current time step
        self.grid[self.center, self.center] = self.source_strength  # Replenish gas at the center
        self.im.set_array(self.grid)  # Update the plot with the new grid values
        return [self.im]  # Return the updated image for animation blitting

    def run_simulation(self):
        """
        Run the diffusion simulation and create an animated plot.

        This method uses FuncAnimation to animate the diffusion process over time.

        Returns:
        --------
        None
        """
        # Create the animation object
        ani = FuncAnimation(self.fig, self.update, frames=self.time_steps, interval=50, blit=True)
        plt.show()  # Display the animation

def test_gas_dispersion():
    simulation = DiffusionSimulation(grid_size=50, time_steps=200, D=0.1, source_strength=100, dx=0.1)
    simulation.run_simulation()

def main():
    test_gas_dispersion()

# Example usage:
if __name__ == '__main__':
    # Create and run the diffusion simulation
    main()
