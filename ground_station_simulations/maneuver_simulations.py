# maneuver_simulations.py
# Author: 530494928
# Date: 2024-08-23
# Description: Performs simulations involving orbital transfers and plane-change maneuvers,
# including Hohmann transfer and inclination adjustments between two orbits.

import ground_station_simulations.orbit as o 
import ground_station_simulations.orbit_simulations as os
import ground_station_simulations.definitions as d
import numpy as np
import ground_station_simulations.orbit_plots as op

def find_best_angle(orbit, starting_orbit, target_orbit, ta_starting, ta_target, low=0.0, high=2*np.pi, tolerance=1e-6):
    """
    Finds the best rotation angle for the transfer orbit to minimize delta-v requirements.
    
    Args:
        orbit (Orbit): The transfer orbit object.
        starting_orbit (Orbit): The starting orbit object.
        target_orbit (Orbit): The target orbit object.
        ta_starting (float): True anomaly at node line of starting orbit.
        ta_target (float): True anomaly at node line of target orbit.
        low (float): Lower bound of the search range in radians.
        high (float): Upper bound of the search range in radians.
        tolerance (float): Tolerance for the binary search termination.

    Returns:
        float: The optimal rotation angle in radians.
    """
    def delta_v_total(angle):
        # Apply the rotation
        orbit.rotate_velocity(angle)
        
        # Calculate delta-v at start and end of transfer
        delta_v_1 = np.linalg.norm(starting_orbit.get_velocity(ta_starting) - orbit.get_velocity(0))
        delta_v_2 = np.linalg.norm(orbit.get_velocity(np.pi) - target_orbit.get_velocity(ta_target))
        
        # Reset the rotation to try the next angle
        orbit.rotate_velocity(-angle)  # Undo the rotation for a clean slate
        
        return delta_v_1 + delta_v_2
    
    while high - low > tolerance:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        
        if delta_v_total(mid1) < delta_v_total(mid2):
            high = mid2
        else:
            low = mid1
    
    return (low + high) / 2

def calculate_node_line(n_1: np.ndarray, n_2: np.ndarray) -> np.ndarray:
    return np.cross(n_1, n_2)


def calculate_node_line_azimuth(node_line: np.ndarray) -> float:
    return np.arctan2(node_line[1], node_line[0])


def get_true_anomaly_at_node_line(node_line: np.ndarray, rotation_matrix: np.ndarray) -> float:
    node_line_in_perifocal = np.dot(np.linalg.inv(rotation_matrix), node_line)
    return np.arctan2(node_line_in_perifocal[1], node_line_in_perifocal[0])

def generate_hohmann_transfer(starting_orbit: o.Orbit, target_orbit: o.Orbit) -> tuple[dict[str, float], float, float]:
    """
    Generate the Hohmann transfer parameters between the starting and target orbits.

    Args:
        starting_orbit (Orbit): Starting orbit object.
        target_orbit (Orbit): Target orbit object.

    Returns:
        Tuple: A dictionary with Hohmann transfer parameters, true anomaly at starting orbit, true anomaly at target orbit.
    """
    # Calculate node line and azimuth
    n_1 = starting_orbit.normal
    n_2 = target_orbit.normal
    node_line = np.cross(n_1, n_2)
    node_line_azimuth = np.arctan2(node_line[1], node_line[0])

    # Calculate true anomalies at node line
    ta_starting_orbit_at_node_line = get_true_anomaly_at_node_line(node_line, starting_orbit.perifocal_to_geocentric_rotation())
    ta_target_orbit_at_node_line = get_true_anomaly_at_node_line(node_line, target_orbit.perifocal_to_geocentric_rotation()) + np.pi

    # Create Hohmann transfer orbit parameters
    hohmann_perigee = starting_orbit.get_radius(ta_starting_orbit_at_node_line)
    hohmann_apogee = target_orbit.get_radius(ta_target_orbit_at_node_line)
    hohmann_inclination = starting_orbit.inclination_angle
    hohmann_raan = starting_orbit.raan

    vernal_equinox = np.array([1, 0, 0])

    # Define the rotation matrix for a counterclockwise rotation around the z-axis
    rotation_matrix = np.array([[np.cos(hohmann_raan), -np.sin(hohmann_raan), 0],
                                [np.sin(hohmann_raan),  np.cos(hohmann_raan), 0],
                                [0,              0,             0]])

    ascending_node = np.dot(rotation_matrix, vernal_equinox)

    # Compute the cross product
    dot_product = np.dot(ascending_node, node_line)

    # Compute the norms (magnitudes) of the vectors
    norm_ascending_node = np.linalg.norm(ascending_node)
    norm_node_line = np.linalg.norm(node_line)

    # Compute the hohmann_argument_of_perigee
    hohmann_argument_of_perigee = - np.arccos(dot_product / (norm_ascending_node * norm_node_line)) 

    hohmann_params = {
        "altitude_of_perigee": hohmann_perigee - o.Orbit.EQUATORIAL_RADIUS,
        "altitude_of_apogee": hohmann_apogee - o.Orbit.EQUATORIAL_RADIUS,
        "inclination_angle": np.degrees(hohmann_inclination),
        "raan": np.degrees(hohmann_raan),
        "argument_of_perigee": np.degrees(hohmann_argument_of_perigee)
    }

    return hohmann_params, ta_starting_orbit_at_node_line, ta_target_orbit_at_node_line

def calculate_delta_v(starting_orbit: o.Orbit, transfer_orbit: o.Orbit, target_orbit: o.Orbit, ta_start: float, ta_target: float) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    # Calculate the velocity vectors
    v_start = starting_orbit.get_velocity(ta_start)
    v_transfer_start = transfer_orbit.get_velocity(0)
    v_transfer_end = transfer_orbit.get_velocity(np.pi)
    v_target = target_orbit.get_velocity(ta_target)

    # Calculate delta-v vectors
    delta_v_vector_1 = v_transfer_start - v_start
    delta_v_vector_2 = v_target - v_transfer_end

    print(v_start)
    print(v_transfer_end)
    print(v_target)

    # Calculate the magnitudes of the delta-v vectors
    delta_v_1 = np.linalg.norm(delta_v_vector_1)
    delta_v_2 = np.linalg.norm(delta_v_vector_2)

    # Return magnitudes and vectors
    return delta_v_1, delta_v_2, delta_v_1 + delta_v_2, delta_v_vector_1, delta_v_vector_2


def calculate_mass_consumed(delta_v: float, isp_values: list[float]) -> np.ndarray:
    return 1 - np.exp(-delta_v / np.array(isp_values))


def calculate_transfer_time(starting_orbit: o.Orbit, transfer_orbit: o.Orbit, target_orbit: o.Orbit, ta_target: float) -> tuple[float, float, float]:
    transfer_orbit_time = transfer_orbit.orbital_period / 2
    dt = target_orbit.calculate_time_between(ta_target, 0)
    time_in_starting_orbit = dt - transfer_orbit_time
    return time_in_starting_orbit, transfer_orbit_time, dt

def print_orbit_parameters(orbit, orbit_name):
    # Accessing the orbital parameters from the object
    semi_major_axis = orbit.semi_major_axis
    eccentricity = orbit.eccentricity
    # Convert angles from radians to degrees
    inclination_angle = np.degrees(orbit.inclination_angle)
    raan = np.degrees(orbit.raan)
    argument_of_perigee = np.degrees(orbit.argument_of_perigee)

    # Formatting and printing the parameters
    print(f"\n{orbit_name} Parameters:")
    print(f"  Semi-major axis       : {semi_major_axis:.2f} km")
    print(f"  Eccentricity          : {eccentricity:.4f}")
    print(f"  Inclination angle     : {inclination_angle:.2f} degrees")
    print(f"  RAAN                  : {raan:.2f} degrees")
    print(f"  Argument of perigee   : {argument_of_perigee:.2f} degrees")

def investigate_maneuver(starting_params: dict, target_params: dict, optimise: bool = True) -> None:
    """Investigate orbital maneuvers and calculate key parameters"""

    # Create instances of the OrbitalParameters class
    starting_orbit = o.Orbit()
    transfer_orbit = o.Orbit()
    target_orbit = o.Orbit()

    # Parse the TLE data for starting and target orbits
    starting_orbit.parse_dictionary(starting_params)
    target_orbit.parse_dictionary(target_params)

    starting_orbit.calculate_orbital_constants_from_dict()
    target_orbit.calculate_orbital_constants_from_dict()

    # Generate Hohmann transfer parameters
    hohmann_params, ta_starting_orbit_at_node_line, ta_target_orbit_at_node_line = generate_hohmann_transfer(
        starting_orbit, target_orbit
    )

    # Parse and calculate constants for transfer orbit
    transfer_orbit.parse_dictionary(hohmann_params)
    transfer_orbit.calculate_orbital_constants_from_dict()
    transfer_orbit.calculate_initial_state()

    # Rotate velocity to find the best angle for transfer orbit
    if optimise:
        transfer_orbit.rotate_velocity(find_best_angle(transfer_orbit, starting_orbit, target_orbit,
                                                    ta_starting_orbit_at_node_line, ta_target_orbit_at_node_line))

    # Assuming the true anomalies and delta-v vectors are available
    ta_1 = ta_starting_orbit_at_node_line  # True anomaly for the first maneuver
    ta_2 = ta_target_orbit_at_node_line    # True anomaly for the second maneuver

    # Calculate delta-v for the maneuvers
    delta_v_1, delta_v_2, delta_v_total, delta_v_vector_1, delta_v_vector_2 = calculate_delta_v(
        starting_orbit, transfer_orbit, target_orbit, ta_1, ta_2
    )

    # Print the information for the first maneuver
    print(f"First Impulse:")
    print(f"  Starting Orbit True Anomaly (ta): {ta_1:.2f} radians")
    print(f"  Delta-V Vector   : {delta_v_vector_1}")

    # Print the information for the second maneuver
    print(f"\nSecond Impulse:")
    print(f"  Target Orbit True Anomaly (ta): {ta_2:.2f} radians")
    print(f"  Delta-V Vector   : {delta_v_vector_2}")

    # print(f"Inclination from starting orbit: {np.degrees(transfer_orbit.current_rotation)} degrees")
    # print(f"\nDelta v_1:      {delta_v_1} km/s")
    # print(f"Delta v_2:      {delta_v_2} km/s")
    # print(f"Delta v_total:  {delta_v_total} km/s")

    # Calculate time in starting and transfer orbit
    time_in_starting_orbit, transfer_orbit_time, total_time = calculate_transfer_time(starting_orbit, transfer_orbit, target_orbit, 
                                                                                     ta_target_orbit_at_node_line)

    starting_ta = (ta_starting_orbit_at_node_line - 2 * np.pi * time_in_starting_orbit / starting_orbit.orbital_period)
    while starting_ta < np.pi / 2:
        starting_ta += 2 * np.pi

    # print(f"Total time elapsed: {total_time} s")
    # print(f"Time in starting orbit: {time_in_starting_orbit} s")
    # print(f"Time in transfer orbit: {transfer_orbit_time} s")
    # print(f"True Anomaly of Starting Orbit: {starting_ta} rad")

    # # Calculate mass consumption for different ISPs
    # portion_of_mass_consumed = calculate_mass_consumed(delta_v_total, d.Isp_values)
    # for isp, portion in zip(d.Isp_values, portion_of_mass_consumed):
    #     print(f"ISP = {isp} s: {portion} kg/kg")

    # for isp, portion in zip(d.Isp_values, portion_of_mass_consumed):
    #     print(f"Mass consumed per 1000 kg for ISP = {isp} s: {portion * 1000} kg")

    # Assuming the objects starting_orbit, transfer_orbit, and target_orbit are defined
    print_orbit_parameters(starting_orbit, "Starting Orbit")
    print_orbit_parameters(transfer_orbit, "Transfer Orbit")
    print_orbit_parameters(target_orbit, "Target Orbit")

    # Plot the transfer maneuver
    # op.plot_transfer_maneuver(starting_orbit, transfer_orbit, target_orbit)

    return delta_v_1 + delta_v_2

def investigate_maneuvers() -> None:
    investigate_maneuver(d.starting_params, d.target_params, False)
    investigate_maneuver(d.starting_params, d.target_params, True)


def simulate_manoeuvres(param_array):
    orbits = []
    spatial_solution_array = []
    temporal_solution_array = []

    starting_params = param_array[0]
    current_orbit = o.Orbit()
    current_orbit.parse_dictionary(starting_params)
    current_orbit.calculate_orbital_constants_from_dict()  # This calls calculate_initial_state

    current_orbit.initial_true_anomaly = 0
    current_orbit.calculate_initial_state()        

    for i in range(1, len(param_array)):
        target_params = param_array[i]

        # Create an instance for the target orbit
        target_orbit = o.Orbit()
        target_orbit.parse_dictionary(target_params)
        target_orbit.calculate_orbital_constants_from_dict()  # This calls calculate_initial_state

        # Generate the Hohmann transfer parameters between current orbit and the next target orbit
        hohmann_params, ta_starting_orbit_at_node_line, ta_target_orbit_at_node_line = generate_hohmann_transfer(
            current_orbit, target_orbit
        )

        current_orbit.final_true_anomaly = ta_starting_orbit_at_node_line
        target_orbit.initial_true_anomaly = ta_target_orbit_at_node_line
        target_orbit.calculate_initial_state()        

        orbits.append(current_orbit)

        # Create and set up the transfer orbit
        transfer_orbit = o.Orbit()
        transfer_orbit.parse_dictionary(hohmann_params)
        transfer_orbit.initial_true_anomaly = 0
        transfer_orbit.final_true_anomaly = np.pi
        transfer_orbit.calculate_orbital_constants_from_dict()  # This calls calculate_initial_state

        orbits.append(transfer_orbit)

        current_orbit = target_orbit

        # If it's the last iteration, append the final target orbit
        if i == len(param_array) - 1:
            orbits.append(current_orbit)

    for orbit in orbits:
        temporal_solution, spatial_solution = orbit.generate_orbit_path()

        temporal_solution_np = np.array(temporal_solution)
        spatial_solution_np = np.array(spatial_solution)

        temporal_solution_array.append(temporal_solution_np)
        spatial_solution_array.append(spatial_solution_np)

    return temporal_solution_array, spatial_solution_array


def main() -> None:
    dv, sequence = investigate_maneuvers()
    return dv, sequence

if __name__ == "__main__":
    main()

