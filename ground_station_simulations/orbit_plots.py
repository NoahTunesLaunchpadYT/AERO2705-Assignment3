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
import ground_station_simulations.definitions as d

# Constants
EQUATORIAL_RADIUS = d.EQUATORIAL_RADIUS

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

def plot_eci_orbit_segments(ax, maneuver_segments, color_offset=0, base_label="Orbit", Earth: bool = True, label_ends=True):
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
        ax.plot(x, y, z, label=label, color=colors[i])

        if label_ends:
            # If it's the first segment, plot the start point
            if i == 0:
                start_point = (x[0], y[0], z[0])
                ax.scatter(*start_point, color=colors[i], s=100, marker='o', label=f"{base_label} start")

            # If it's the last segment, plot the end point
            if i == num_segments - 1:
                end_point = (x[-1], y[-1], z[-1])
                ax.scatter(*end_point, color=colors[i], s=100, marker='^', label=f"{base_label} end")

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    max_radius = np.max([np.max(np.linalg.norm(segment[:3, :], axis=0)) for segment in maneuver_segments])
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])
    ax.legend()

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

            # print(orbit.get_radius(2.3543809297189533))
            # print(orbit.get_radius(np.pi))
            
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

        ax.set_xlim([-max_radius, max_radius])
        ax.set_ylim([-max_radius, max_radius])
        ax.set_zlim([-max_radius/1.2, max_radius/1.2])

        # Formatting the plot
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        # ax.set_title('3D Orbit Plot: Starting, Transfer, and Target Orbits')
        ax.legend()
        
        # Add grid and show the plot
        ax.grid(True)
        plt.show()


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
    ax.legend()
    plt.show()
