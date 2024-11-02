# orbit_plots.py
# Author: 530494928
# Date: 2024-08-23
# Description: Provides plotting utilities for visualizing satellite orbits and ground tracks.
# It includes functions to plot 3D orbits, maneuver segments, and 2D ground tracks with or without Earth's rotation.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from matplotlib import cm
from ground_station_simulations import definitions as d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation

# Constants
EQUATORIAL_RADIUS = d.EQUATORIAL_RADIUS

class Arrow3D:
    def __init__(self, start, end, color='blue', linewidth=1, mutation_scale=20):
        self.start = np.array(start)
        self.end = np.array(end)
        self.color = color
        self.linewidth = linewidth
        self.mutation_scale = mutation_scale
    
    def draw(self, ax):
        # Draw arrow line
        ax.plot([self.start[0], self.end[0]],
                [self.start[1], self.end[1]],
                [self.start[2], self.end[2]],
                color=self.color, linewidth=self.linewidth)

        # Draw arrowhead
        direction = self.end - self.start
        direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
        head_length = self.mutation_scale * 0.05  # Adjust arrowhead size
        head_width = head_length * 0.5  # Adjust arrowhead width

        # Arrowhead vertices
        arrowhead_vertices = [
            self.end,
            self.end - direction * head_length + np.cross(direction, [1, 0, 0]) * head_width,
            self.end - direction * head_length + np.cross(direction, [0, 1, 0]) * head_width,
        ]
        
        arrowhead_poly = Poly3DCollection([arrowhead_vertices], color=self.color)
        ax.add_collection3d(arrowhead_poly)

def generate_orbit_coordinates(orbit, num_points=1000, ta_start=0, ta_end=2 * np.pi):
    """
    Generate the 3D coordinates for the orbit in ECI (Earth-Centered Inertial) frame based on the true anomaly.

    Args:
        orbit (Orbit): The orbit object containing the satellite's orbital parameters.
        num_points (int): Number of points to generate along the orbit.
        ta_start (float): Starting true anomaly (in radians).
        ta_end (float): Ending true anomaly (in radians).

    Returns:
        np.ndarray: Arrays of X, Y, and Z coordinates representing the orbit in ECI frame.
    """
    ta_values = np.linspace(ta_start, ta_end, num_points)
    x_coords, y_coords, z_coords = [], [], []

    for ta in ta_values:
        r = orbit.get_radius(ta)
        r_pf = np.array([r * np.cos(ta), r * np.sin(ta), 0])  # Position in perifocal frame
        r_eci = np.dot(orbit.perifocal_to_geocentric_rotation(), r_pf)  # Convert to ECI frame
        x_coords.append(r_eci[0])
        y_coords.append(r_eci[1])
        z_coords.append(r_eci[2])

    return np.array(x_coords), np.array(y_coords), np.array(z_coords)

def plot_path(ax, path, color_offset=0, base_label="Orbit", Earth: bool = True, label_ends=True, linestyle='-'):
    """
    Plot the orbit in 3D using the ECI coordinates for multiple maneuver segments.
    Each segment is plotted in a different color. The start and end points of the first and last segments 
    are labeled, respectively.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to plot the orbits.
        maneuver_segments (list of np.ndarray): List of state_array segments, where each segment corresponds to a maneuver.
        color_offset (float): Offset for the color map to distinguish between different paths.
        base_label (str): Base label for the orbit segments.
        Earth (bool): Whether or not to plot the Earth in the background.
    """
    R = EQUATORIAL_RADIUS
    maneuver_segments = path.solution_array_segments
    num_segments = len(maneuver_segments)

    colors = cm.jet(np.linspace(color_offset, 1 + color_offset, num_segments))

    if Earth:
        # Plot the Earth as a sphere
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        earth_x = R * np.cos(u) * np.sin(v)
        earth_y = R * np.sin(u) * np.sin(v)
        earth_z = R * np.cos(v)
        ax.plot_surface(earth_x, earth_y, earth_z, color='b', alpha=0.3, label="Earth")

    # Plot each maneuver segment in a different color
    for i, segment in enumerate(maneuver_segments):
        x, y, z = segment[0, :], segment[1, :], segment[2, :]
        label = f"{base_label} {i + 1}"
        ax.plot(x, y, z, label=label, color=colors[i], linestyle=linestyle)

        if label_ends:
            # If it's the first segment, plot the start point
            if i == 0:
                start_point = (x[0], y[0], z[0])
                ax.scatter(*start_point, color=colors[i], s=100, marker='o', label=f"{base_label} start", linestyle=linestyle)

            # If it's the last segment, plot the end point
            if i == num_segments - 1:
                end_point = (x[-1], y[-1], z[-1])
                ax.scatter(*end_point, color=colors[i], s=100, marker='^', label=f"{base_label} end", linestyle=linestyle)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    max_radius = np.max([np.max(np.linalg.norm(segment[:3, :], axis=0)) for segment in maneuver_segments])
    ax.set_xlim([-max_radius * 2, max_radius * 2])
    ax.set_ylim([-max_radius * 2, max_radius * 2])
    ax.set_zlim([-max_radius * 2/1.2, max_radius * 2/1.2])

def animate_path(path, color_offset=0, base_label="Orbit", Earth=True, speed_factor=1, pause_duration=3):
    """
    Animate the orbit in 3D, plotting each point in each maneuver segment with a constant time step.
    Leaves a colored trail behind to indicate each segment, and clears all segments at each repeat.

    Args:
        path: An object containing 'solution_array_segments'.
        color_offset (float): Offset for the color map to distinguish between different paths.
        base_label (str): Base label for the orbit segments.
        Earth (bool): Whether or not to plot the Earth in the background.
        speed_factor (int): Factor by which to speed up the animation (e.g., 10 for 10x faster).
        pause_duration (float): Duration to pause at the end of the animation (in seconds).
    """
    maneuver_segments = path.solution_array_segments
    num_segments = len(maneuver_segments)
    colors = cm.jet(np.linspace(color_offset, 1 + color_offset, num_segments))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if Earth:
        # Plot the Earth as a sphere
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        earth_x = EQUATORIAL_RADIUS * np.cos(u) * np.sin(v)
        earth_y = EQUATORIAL_RADIUS * np.sin(u) * np.sin(v)
        earth_z = EQUATORIAL_RADIUS * np.cos(v)
        ax.plot_surface(earth_x, earth_y, earth_z, color='b', alpha=0.3, label="Earth")

    # Initialize lines for each segment and a scatter point for the satellite
    lines = [ax.plot([], [], [], color=colors[i], linestyle='-')[0] for i in range(num_segments)]
    satellite_point, = ax.plot([], [], [], 'o', color='red', markersize=5)

    # Flatten the maneuver segments for easy indexing
    all_points = np.hstack(maneuver_segments)
    x_data, y_data, z_data = all_points[0], all_points[1], all_points[2]

    # Set axis limits based on the maximum radius
    max_radius = np.max(np.linalg.norm(all_points[:3, :], axis=0))
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    # Set a constant frame interval based on the speed factor
    interval = 100 / speed_factor  # Adjust the interval in milliseconds

    # Calculate the number of pause frames based on the pause duration
    pause_frames = int((pause_duration * 1000) / interval)

    def init():
        # Clear all line segments and reset the satellite point
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        satellite_point.set_data([], [])
        satellite_point.set_3d_properties([])
        return lines + [satellite_point]

    def update(frame):
        # Determine if we are in the pause phase
        if frame >= len(x_data):
            # During the pause, just display the last frame
            return lines + [satellite_point]

        # Determine the current segment index
        segment_idx = np.searchsorted(np.cumsum([segment.shape[1] for segment in maneuver_segments]), frame, side='right')
        
        # Update the current segment line by adding up to the current frame's points
        segment_start = 0 if segment_idx == 0 else sum([segment.shape[1] for segment in maneuver_segments[:segment_idx]])
        segment_end = frame + 1  # Add one to include the current frame point
        lines[segment_idx].set_data(x_data[segment_start:segment_end], y_data[segment_start:segment_end])
        lines[segment_idx].set_3d_properties(z_data[segment_start:segment_end])

        # Update the position of the satellite point
        satellite_point.set_data(x_data[frame], y_data[frame])
        satellite_point.set_3d_properties(z_data[frame])
        
        return lines + [satellite_point]

    # Create the animation with the init function to clear segments on repeat
    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_data) + pause_frames,  # Add pause frames at the end
        init_func=init,    # Clears the lines at the beginning of each repeat
        interval=interval, # Constant interval for each frame
        repeat=True,
        blit=True
    )

    plt.legend()
    plt.show()
    return ani

def plot_current_position(ax, path, label, color, marker='o'):
    """
    Plot the current position of the satellite in 3D.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to plot the satellite's position.
        path (Satellite_path): The current path of the satellite.
        label (str): Label for the satellite's position in the plot legend.
        color (str): Color of the marker.
        marker (str): Marker style for the satellite's position.
    """
    current_position = path.state_array[:3, -1]  # Get the latest (x, y, z) position
    ax.scatter(current_position[0], current_position[1], current_position[2], 
               color=color, label=label, s=100, marker=marker)

def plot_current_position(ax, vector, color, label, marker='o'):
    x, y, z = (vector[0], vector[1], vector[2])

    ax.scatter(x, y, z, 
               color=color, label=label, s=100, marker=marker)

def plot_unit_vector(ax, vector, color, label):
    # Plot a vector from the origin (0, 0, 0) to the point (x, y, z)
    vector = vector / np.linalg.norm(vector) * EQUATORIAL_RADIUS
    arrow = Arrow3D((0,0,0), vector, 
              color=color, linewidth=2, mutation_scale=20000)
    arrow.draw(ax)

def plot_state(ax, state, color):
    r = state[0:3]
    v = state[3:6]
    vector = v / np.linalg.norm(v) * EQUATORIAL_RADIUS
    arrow = Arrow3D(r, r + vector, 
              color=color, linewidth=2, mutation_scale=20000)
    arrow.draw(ax)

def plot_ground_track(solution_y, solution_t, greenwich_sidereal_time, stationary_ground: bool = False):
    """
    Plot the ground track (latitude vs longitude) of the satellite on a 2D map, accounting for Earth's rotation.

    Args:
        solution_y (np.ndarray): Array of position vectors from the simulation, shape [6, n] where the first 3 rows are the ECI positions.
        solution_t (np.ndarray): Array of time values corresponding to the positions in solution_y.
        stationary_ground (bool): If True, plots the ground track assuming the ground is stationary.
                                  If False, plots the ground track accounting for Earth's rotation.
    """
    x, y, z = solution_y[0, :], solution_y[1, :], solution_y[2, :]
    r = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.degrees(np.arcsin(z / r))
    longitude = np.degrees(np.arctan2(y, x))

    # Earth's rotation rate (15 degrees per hour if not stationary)
    earth_rotation_rate = 0 if stationary_ground else 15
    time_hours = solution_t / 3600
    longitude_shift = earth_rotation_rate * time_hours
    longitude = (longitude - greenwich_sidereal_time + longitude_shift + 180) % 360 - 180

    # Insert NaN to handle longitude jumps at the dateline
    for i in range(1, len(longitude)):
        if abs(longitude[i] - longitude[i - 1]) > 180:
            longitude[i] = latitude[i] = np.nan

    # Plot the ground track
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(longitude, latitude, label="Ground Track", color="red")
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.grid(True)

    # Optionally add Earth's map
    if not stationary_ground:
        url = "https://upload.wikimedia.org/wikipedia/commons/8/83/Equirectangular_projection_SW.jpg"
        with urllib.request.urlopen(url) as url_response:
            img = Image.open(url_response)
        img_array = np.array(img)
        ax.imshow(img_array, extent=[-180, 180, -90, 90], aspect='auto', alpha=0.5)

    plt.show()

def plot_transfer_maneuver(starting_orbit, transfer_orbit, target_orbit):
        """
        Plots the three orbits in 3D: the starting orbit, transfer orbit, and target orbit.
        The transfer orbit is plotted for half a period starting at ta = 0.
        
        Args:
            starting_orbit (Orbit): The initial orbit (chaser's orbit).
            transfer_orbit (Orbit): The transfer orbit used to move from starting to target orbit.
            target_orbit (Orbit): The target orbit (the final orbit).
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Constants
        num_points = 1000  # Number of points to discretize each orbit

        # Helper function to generate 3D coordinates for an orbit
        def generate_orbit_coordinates(orbit, num_points, ta_start=0, ta_end=2*np.pi):
            """
            Generate the 3D coordinates for the orbit in ECI frame based on the true anomaly.
            Args:
                orbit (Orbit): The orbit object.
                num_points (int): Number of points for the plot.
                ta_start (float): Starting true anomaly in radians.
                ta_end (float): Ending true anomaly in radians.
            
            Returns:
                np.ndarray: 3D positions (x, y, z) in ECI frame.
            """
            # Generate true anomalies over the given range
            ta_values = np.linspace(ta_start, ta_end, num_points)

            # Initialize arrays to store the 3D coordinates
            x_coords = []
            y_coords = []
            z_coords = []

            # For each true anomaly, compute the position in the ECI frame
            for ta in ta_values:
                # Get the radius at this true anomaly
                r = orbit.get_radius(ta)
                
                # Calculate position in perifocal frame (2D in the plane)
                r_pf = np.array([
                    r * np.cos(ta),  # x in perifocal
                    r * np.sin(ta),  # y in perifocal
                    0               # z is 0 in the perifocal frame
                ])
                
                # Rotate to the ECI frame using the orbit's rotation matrix
                r_eci = np.dot(orbit.perifocal_to_geocentric_rotation(), r_pf)
                
                # Append to coordinates lists
                x_coords.append(r_eci[0])
                y_coords.append(r_eci[1])
                z_coords.append(r_eci[2])

            return np.array(x_coords), np.array(y_coords), np.array(z_coords)

        # Step 1: Plot the full period for the starting orbit
        x_start, y_start, z_start = generate_orbit_coordinates(starting_orbit, num_points, 0, 2*np.pi)
        ax.plot(x_start, y_start, z_start, label='Starting Orbit (1 period)', color='blue')

        # Step 2: Plot half a period for the transfer orbit
        x_transfer, y_transfer, z_transfer = generate_orbit_coordinates(transfer_orbit, num_points, 0, np.pi)
        ax.plot(x_transfer, y_transfer, z_transfer, label='Transfer Orbit (half period)', color='green')

        # Step 3: Plot the full period for the target orbit
        x_target, y_target, z_target = generate_orbit_coordinates(target_orbit, num_points, 0, 2*np.pi)
        ax.plot(x_target, y_target, z_target, label='Target Orbit (1 period)', color='red')

        # Step 4: Plot the Earth as a sphere
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]  # Create a meshgrid for the sphere
        x_earth = d.EQUATORIAL_RADIUS * np.cos(u) * np.sin(v)  # X coordinates of the Earth
        y_earth = d.EQUATORIAL_RADIUS * np.sin(u) * np.sin(v)  # Y coordinates of the Earth
        z_earth = d.EQUATORIAL_RADIUS * np.cos(v)  # Z coordinates of the Earth

        # Plot Earth's surface with transparency (alpha)
        ax.plot_surface(x_earth, y_earth, z_earth, color='b', alpha=0.3, label='Earth')

        # Ensure the x, y, and z limits are the same
        max_radius = max(np.max(np.abs(x_start)), np.max(np.abs(y_start)), np.max(np.abs(z_start)),
                        np.max(np.abs(x_transfer)), np.max(np.abs(y_transfer)), np.max(np.abs(z_transfer)),
                        np.max(np.abs(x_target)), np.max(np.abs(y_target)), np.max(np.abs(z_target)),
                        d.EQUATORIAL_RADIUS)

        ax.set_xlim([-max_radius * 5, max_radius * 5])
        ax.set_ylim([-max_radius * 5, max_radius * 5])
        ax.set_zlim([-max_radius * 5/1.2, max_radius * 5/1.2])

        # Formatting the plot
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        # ax.set_title('3D Orbit Plot: Starting, Transfer, and Target Orbits')
        ax.legend()
        
        # Add grid and show the plot
        ax.grid(True)
        plt.show()

def plot_orbit(ax, orbit, num_points=1000, label="Orbit"):
    """
    Plots an orbit on the provided 3D axis without showing it.

    Args:
        ax (matplotlib.axes._axes.Axes): The 3D axis on which to plot the orbit.
        orbit: The orbit object with methods `get_radius_vector` to obtain (x, y, z) coordinates.
        num_points (int): The number of points to use for the orbit plot.
        label (str): The label for the orbit in the plot.
    """
    # Generate true anomaly points over a full orbit (0 to 2π)
    true_anomalies = np.linspace(0, 2 * np.pi, num_points)
    
    # Calculate the position vectors for each true anomaly
    x_vals = []
    y_vals = []
    z_vals = []
    
    for ta in true_anomalies:
        # Get the position vector in geocentric coordinates
        position_vector = orbit.get_radius_vector(ta)
        
        # Extract x, y, z from the position vector
        x_vals.append(position_vector[0])
        y_vals.append(position_vector[1])
        z_vals.append(position_vector[2])
    
    # Plot the orbit in 3D
    ax.plot(x_vals, y_vals, z_vals, label=label)

def new_figure():
    """
    Create a new 3D figure for plotting ECI orbits.

    Returns:
        ax (matplotlib.axes._subplots.Axes3DSubplot): A new 3D axis for plotting orbits.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    return ax

def show(ax) -> None:
    """
    Display the final plot with legends and formatting.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The axis of the plot.
    """
    plt.show()