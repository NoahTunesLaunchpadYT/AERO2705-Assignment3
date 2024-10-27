import orbit as o
import numpy as np
from scipy.integrate import solve_ivp
import linear_algebra as la

hello = "hello"

class SatellitePath:
    """
    The Satellite_path class simulates the motion and maneuvers of a satellite in orbit around Earth. 
    It tracks the satellite's state (position, velocity, and mass) over time, models both coasting 
    phases (no thrust) and active thruster burns (with thrust), and supports orbital maneuvers such as 
    in-plane impulses and plane changes. The class provides methods to initialize the satellite's state 
    from a given orbit, simulate both thrust and coast phases, and append maneuver segments for plotting 
    or analysis.

    Attributes:
        MU (float): The gravitational constant for Earth (398600 km^3/s^2).
        EQUATORIAL_RADIUS (float): The equatorial radius of Earth (6378 km).
        ANGULAR_TOLERANCE (float): A small tolerance value for comparing angular differences (radians).
        J2 (float): Earth's second zonal harmonic, affecting perturbations in orbit.
        SIMULATION_RESOLUTION (int): Number of steps per unit time for numerical simulation.

        state_array (np.ndarray): A 2D array where each column represents a time step. 
                                  The rows store the satellite's state variables: 
                                  [x, y, z, x_dot, y_dot, z_dot, mass].
        time_array (np.ndarray): A 1D array storing the simulation time steps corresponding to state_array.
        thrust (float): The thrust of the satellite's engine (in N, Newtons). Defaults to 0 for coasting.
        isp (float): The specific impulse of the satellite's thruster (in seconds).
        maneuver_segments (list): A list of segments representing orbital maneuvers, useful for visualization.
        last_index (int): An index tracking the position in state_array for segment slicing during maneuvers.
    """
    
    MU = 398600                 # km^3/s^2
    EQUATORIAL_RADIUS = 6378    # km
    ANGULAR_TOLERANCE = 1e-10  # radians
    J2 = 1.08263e-3
    SIMULATION_RESOLUTION = 50
    MAX_ORBIT_DURATION = 1e7

    def __init__(self) -> None:
        # self.solution_array = np.empty((6, 0)) 
        # self.time_array = np.empty((0,))  # Empty array to store time values
        self.solution_array_segments = []          # List to store maneuver segments
        self.time_array_segments = []
        self.num_segments = 0                  # The number of segments so far
        self.dv = [] # Total delta v for path
        self.dv_total = 0

    def initial_state_from_orbit_params(self, orbit_params) -> None:
        """
        Generate the initial state vector from the given orbit, true anomaly, and mass.

        Args:
            orbit (o.Orbit): The orbit object to generate the state from.
            true_anomaly (float): The true anomaly at which to calculate the state vector.
            mass (float): The mass of the spacecraft or object.
        """
        orbit = o.Orbit()
        orbit.calculate_parameters_from_dictionary(orbit_params)

        r = orbit.initial_displacement
        v = orbit.initial_velocity

        # Create the state vector [x, y, z, x_dot, y_dot, z_dot, m]
        state_vector = np.hstack((r, v))

        # Reshape to (7, 1) so that it matches the solve_ivp output format
        state_vector = state_vector.reshape(6, 1)

        # Initialize the state_array with the initial state
        self.solution_array_segments.append(state_vector)

        self.time_array_segments = [np.array([0])]

    def ode_func(self, t: float, y_n: np.ndarray) -> np.ndarray:
        """
        Differential equation for the orbital dynamics around  a spherical planet

        Args:
            t (float): Time value.
            y_n (np.ndarray): State vector.
            T (float): Thrust force (optional, default is 0 for coast).

        Returns:
            np.ndarray: Derivatives of the state vector [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot].
        """
        mu = self.MU
        
        r = np.sqrt(y_n[0]**2 + y_n[1]**2 + y_n[2]**2)
        v = np.sqrt(y_n[3]**2 + y_n[4]**2 + y_n[5]**2)

        x_dot = y_n[3]
        y_dot = y_n[4]
        z_dot = y_n[5]

        x_ddot = -mu / r**3 * y_n[0]
        y_ddot = -mu / r**3 * y_n[1]
        z_ddot = -mu / r**3 * y_n[2]

        # Return derivatives [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]
        return np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])

    def simulate_coast_until_azimuth(self, target_azimuth: float):
        """
        Simulate a coast phase that stops once the spacecraft reaches the target azimuth.
        """
        # Normalize target azimuth to the range [-π, π]
        target_azimuth = self.normalize_angle(target_azimuth)

        def stopping_condition(t, y):
            current_azimuth = np.arctan2(y[1], y[0])  # Calculate the current azimuth (angle in XY plane)
            current_azimuth = self.normalize_angle(current_azimuth)  # Normalize current azimuth to [-π, π]

            # Calculate the difference between the current and target azimuths
            azimuth_diff = current_azimuth - target_azimuth

            # Adjust the difference to be within the range [-π, π]
            azimuth_diff = self.normalize_angle(azimuth_diff)

            return azimuth_diff  # Stop when the azimuth difference is zero

        stopping_condition.terminal = True

        last_segment = self.solution_array_segments[-1]

        initial_state = last_segment[:, -1]
        starting_time = self.time_array_segments[-1][-1]
        
        t_span = (starting_time, starting_time+self.MAX_ORBIT_DURATION)  # Use a large time span, we'll stop at the event

        # Solve the ODE for the coast phase with the event to stop at the target azimuth
        solution = solve_ivp(self.ode_func, t_span, initial_state,
                            events=stopping_condition,
                            max_step=self.MAX_ORBIT_DURATION
                                /self.SIMULATION_RESOLUTION, rtol=1e-6)


        # Append the new results to the existing state array
        self.solution_array_segments[-1] = np.hstack(
            (self.solution_array_segments[-1], solution.y))  # Append along columns

        self.time_array_segments[-1] = np.hstack(
            (self.time_array_segments[-1], solution.t))  # Add time steps

    def simulate_coast(self, duration=3600):
        """
        Simulate a coast phase that stops once the spacecraft reaches the target azimuth.
        """
        # Normalize target azimuth to the range [-π, π]
        last_segment = self.solution_array_segments[-1]
        initial_state = last_segment[:, -1]
        starting_time = self.time_array_segments[-1][-1]
        
        t_span = (starting_time, starting_time+duration)  # Use a large time span, we'll stop at the event

        # Solve the ODE for the coast phase with the event to stop at the target azimuth
        solution = solve_ivp(self.ode_func, t_span, initial_state,
                            max_step=self.SIMULATION_RESOLUTION, rtol=1e-6)

        # Append the new results to the existing state array
        self.solution_array_segments[-1] = np.hstack(
            (self.solution_array_segments[-1], solution.y))  # Append along columns

        self.time_array_segments[-1] = np.hstack(
            (self.time_array_segments[-1], solution.t))  # Add time steps


    def normalize_angle(self, angle):
        """Normalize an angle to the range [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def simulate_impulse(self, target_velocity: np.ndarray):
        """
        Simulate an impulsive burn where velocity changes instantaneously, 
        adjusting the mass according to the delta-v required for the maneuver.

        Args:
            target_velocity (np.ndarray): The target velocity vector [vx_dot, vy_dot, vz_dot].
        """
        # Get the most recent state from state_array
        last_state = self.solution_array_segments[-1][:, -1]
        last_velocity = last_state[3:6]  # Extract the velocity components
        
        # Calculate delta-v for the maneuver (magnitude of the velocity change)
        dv = np.linalg.norm(target_velocity - last_velocity)
        self.dv.append(dv)
        self.dv_total += dv

        # Create a new state vector with the updated velocity and mass, then append to the state_array
        new_state = np.copy(last_state)
        new_state[3:6] = target_velocity  # Apply the new velocity

        # Create new solution array segment
        self.solution_array_segments.append(new_state.reshape(6, 1))

        self.num_segments += 1
        
        self.time_array_segments.append(np.array([self.time_array_segments[self.num_segments-1][-1]])) # Copy last time value 

    def generate_path(self, orbits_params, sequence_type="All", plotting = False):

        self.initial_state_from_orbit_params(orbits_params[0])

        # Try all transfer methods and pick the best one
        for i in range(1, len(orbits_params)):
            starting_orbit = o.Orbit()
            target_orbit = o.Orbit()

            most_recent_state_vector = self.solution_array_segments[-1][:, -1]

            starting_orbit.calculate_initial_parameters_from_state_vector(most_recent_state_vector)
            target_orbit.calculate_parameters_from_dictionary(orbits_params[i])

            if sequence_type == "All":
                # Try each orbit approach 
                # Save the current state vector
                prev_solution_array = self.solution_array_segments
                prev_time_array = self.time_array_segments
                prev_num_segments = self.num_segments
                prev_dv = self.dv
                
                # Try transfer manoeuvre
                dvs = []
                self.generate_hohman_like_transfer(starting_orbit, target_orbit)
                dvs[0] = self.dv

                # Reset the state vector
                self.solution_array_segments = prev_solution_array
                self.time_array_segments = prev_time_array
                self.num_segments = prev_num_segments
                self.dv = prev_dv
                
                self.generate_circularised_transfer(starting_orbit, target_orbit)
                dvs[1] = self.dv

                self.solution_array_segments = prev_solution_array
                self.time_array_segments = prev_time_array
                self.num_segments = prev_num_segments
                self.dv = prev_dv

                self.generate_lambert_transfer(starting_orbit, target_orbit)
                dvs[2] = self.dv

                self.solution_array_segments = prev_solution_array
                self.time_array_segments = prev_time_array
                self.num_segments = prev_num_segments
                self.dv = prev_dv

                best_orbit = dvs.index(min(dvs))

                if best_orbit == 0:
                    self.generate_hohman_like_transfer(starting_orbit, target_orbit)
                elif best_orbit == 1:
                    self.generate_circularised_transfer(starting_orbit, target_orbit)
                elif best_orbit == 2:
                    self.generate_lambert_transfer(starting_orbit, target_orbit)

            if sequence_type == "hohmann-like":
                self.generate_hohman_like_transfer(starting_orbit, target_orbit)
            elif sequence_type == "circularising":
                self.generate_circularising_transfer(starting_orbit, target_orbit)
            elif sequence_type == "lambert":
                self.generate_lambert_transfer(starting_orbit, target_orbit)
        
        print("Delta-V:")
        print(self.dv)

    def ta_at_intersection_line(self, node_line: np.ndarray, rotation_matrix: np.ndarray) -> float:
        node_line_in_perifocal = np.dot(np.linalg.inv(rotation_matrix), node_line)
        return np.arctan2(node_line_in_perifocal[1], node_line_in_perifocal[0])

    def generate_hohman_like_transfer(self, starting_orbit, target_orbit):        
        starting_orbit = starting_orbit
        transfer_orbit = o.Orbit()
        target_orbit = target_orbit

        n_1 = starting_orbit.normal
        n_2 = target_orbit.normal
        angle_diff = la.angle_between_vectors(n_1, n_2)
        intersection_line = np.cross(n_1, n_2)

        ta_starting_orbit_at_node_line = self.ta_at_intersection_line(
            intersection_line, 
            starting_orbit.perifocal_to_geocentric_rotation()
        )

        ta_target_orbit_at_node_line = self.ta_at_intersection_line(
            -intersection_line, 
            target_orbit.perifocal_to_geocentric_rotation()
        )

        hohmann_perigee = starting_orbit.get_radius(ta_starting_orbit_at_node_line)
        hohmann_apogee = target_orbit.get_radius(ta_target_orbit_at_node_line)

        if (hohmann_apogee < hohmann_perigee):
            temp = hohmann_apogee
            hohmann_apogee = hohmann_perigee
            hohmann_perigee = temp
            ta_transfer = np.pi
            hohmann_inclination = target_orbit.inclination_angle
            hohmann_raan = target_orbit.raan
        else:
            ta_transfer = 0
            hohmann_inclination = starting_orbit.inclination_angle
            hohmann_raan = starting_orbit.raan

        vernal_equinox = np.array([1, 0, 0])
        
        # Define the rotation matrix for a counterclockwise rotation around the z-axis
        rotation_matrix = np.array([[np.cos(hohmann_raan), -np.sin(hohmann_raan), 0],
                                    [np.sin(hohmann_raan),  np.cos(hohmann_raan), 0],
                                    [0,              0,             0]])

        ascending_node = np.dot(rotation_matrix, vernal_equinox)

        # Compute the norms (magnitudes) of the vectors
        norm_ascending_node = np.linalg.norm(ascending_node)
        norm_intersection_line = np.linalg.norm(intersection_line)

        # Calculate argument of perigee
        cos_omega = (np.dot(ascending_node, intersection_line) 
                    / (norm_ascending_node * norm_intersection_line)) # This is the cosine of ω
        omega = np.arccos(np.clip(cos_omega, -1, 1))  # Use arccos to get ω

        # Check Direction of arugment of perigee
        if intersection_line[2] < 0:
            omega = 2 * np.pi - omega

        hohmann_params = {
            "altitude_of_perigee": hohmann_perigee - o.Orbit.EQUATORIAL_RADIUS,
            "altitude_of_apogee": hohmann_apogee - o.Orbit.EQUATORIAL_RADIUS,
            "inclination_angle": np.degrees(hohmann_inclination),
            "raan": np.degrees(hohmann_raan),
            "argument_of_perigee": np.degrees(omega),
            "initial_true_anomaly": ta_transfer
        }

        # Create transfer orbit
        transfer_orbit.calculate_parameters_from_dictionary(hohmann_params)

        manoeuvre_1_radius_vector = starting_orbit.get_radius_vector(ta_starting_orbit_at_node_line)
        manoeuvre_2_radius_vector = target_orbit.get_radius_vector(ta_target_orbit_at_node_line)

        manoeuvre_1_azimuth = np.arctan2(manoeuvre_1_radius_vector[1], manoeuvre_1_radius_vector[0])
        manoeuvre_2_azimuth = np.arctan2(manoeuvre_2_radius_vector[1], manoeuvre_2_radius_vector[0])

        self.simulate_coast_until_azimuth(manoeuvre_1_azimuth)
        self.simulate_impulse(transfer_orbit.get_velocity(ta_transfer))
        self.simulate_coast_until_azimuth(manoeuvre_2_azimuth)
        self.simulate_impulse(target_orbit.get_velocity(ta_target_orbit_at_node_line))
        self.simulate_coast(3600)



def main() -> None:
    return

if __name__ == "__main__":
    main()