import numpy as np
from scipy.integrate import solve_ivp
from ground_station_simulations import orbit as o
from ground_station_simulations import linear_algebra as la
from ground_station_simulations import definitions as d

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
    
    MU = d.MU                 # km^3/s^2
    EQUATORIAL_RADIUS = d.EQUATORIAL_RADIUS    # km
    J2 = 1.08263e-3
    SIMULATION_RESOLUTION = 500
    MAX_ORBIT_DURATION = 1e7

    def __init__(self) -> None:
        # self.solution_array = np.empty((6, 0)) 
        # self.time_array = np.empty((0,))  # Empty array to store time values
        self.solution_array_segments = []          # List to store maneuver segments
        self.time_array_segments = []
        self.num_segments = 0                  # The number of segments so far
        self.dv = [] # Total delta v for path
        self.dv_total = 0
        self.ta_at_last_impulse = 0

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

    def simulate_coast(self, duration=3600):
        """
        Simulate a coast phase that stops once the spacecraft reaches the target azimuth.
        """
        if duration < 1e-6:
            print(f"ValueError: 'duration' is too small {duration}")

        # Normalize target azimuth to the range [-π, π]
        last_segment = self.solution_array_segments[-1]
        initial_state = last_segment[:, -1]
        starting_time = self.time_array_segments[-1][-1]
        
        t_span = (starting_time, starting_time+duration)  # Use a large time span, we'll stop at the event
        t_eval = np.linspace(t_span[0], t_span[1], self.SIMULATION_RESOLUTION)

        # Solve the ODE for the coast phase with the event to stop at the target azimuth
        solution = solve_ivp(
            self.ode_func,
            t_span,
            initial_state,
            method='DOP853',  # Choose 'DOP853' for a high-order explicit integrator or 'LSODA' for adaptive switching
            t_eval=t_eval,
            rtol=1e-9,         # Increase tolerance for higher accuracy
            atol=1e-12         # Set absolute tolerance to complement relative tolerance
        )
        # Append the new results to the existing state array
        self.solution_array_segments[-1] = np.hstack(
            (self.solution_array_segments[-1], solution.y))  # Append along columns

        self.time_array_segments[-1] = np.hstack(
            (self.time_array_segments[-1], solution.t))  # Add time steps


    def normalize_angle(self, angle):
        """Normalize an angle to the range [-π, π]."""
        return (angle + 2*np.pi) % (2 * np.pi)

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

    def generate_path(self, orbits_params, sequence_type="All", plotting = False, ax=None):
        self.initial_state_from_orbit_params(orbits_params[0])

        most_recent_state_vector = self.solution_array_segments[-1][:, -1]
        # if ax:
        #     print(f"Starting ta: {0}")
        #     print(f"Starting state: {most_recent_state_vector}")        
        #     pl.plot_state(ax, most_recent_state_vector, "green")

        # Try all transfer methods and pick the best one
        if len(orbits_params) > 2:
            self.ta_at_last_impulse = orbits_params[0]["initial_true_anomaly"]
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
                    self.generate_hohman_like_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    dvs[0] = self.dv

                    # Reset the state vector
                    self.solution_array_segments = prev_solution_array
                    self.time_array_segments = prev_time_array
                    self.num_segments = prev_num_segments
                    self.dv = prev_dv
                    
                    self.generate_circularised_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    dvs[1] = self.dv

                    self.solution_array_segments = prev_solution_array
                    self.time_array_segments = prev_time_array
                    self.num_segments = prev_num_segments
                    self.dv = prev_dv

                    self.generate_lambert_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    dvs[2] = self.dv

                    self.solution_array_segments = prev_solution_array
                    self.time_array_segments = prev_time_array
                    self.num_segments = prev_num_segments
                    self.dv = prev_dv

                    best_orbit = dvs.index(min(dvs))

                    if best_orbit == 0:
                        self.generate_hohman_like_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    elif best_orbit == 1:
                        self.generate_circularised_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    elif best_orbit == 2:
                        self.generate_lambert_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)

                if sequence_type == "hohmann-like":
                    print(f"\nHOHMANN TRANSFER {i}")

                    self.generate_hohman_like_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                    
                    # print("Hohmann transfer delta-V:")
                    # print(self.dv)

                elif sequence_type == "circularising":
                    self.generate_circularising_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
                elif sequence_type == "lambert":
                    self.generate_lambert_transfer(starting_orbit, target_orbit, plotting=plotting, ax=ax)
            self.simulate_coast()
        else:
            if sequence_type == "coast":
                orbit = o.Orbit()

                most_recent_state_vector = self.solution_array_segments[-1][:, -1]
                orbit.calculate_initial_parameters_from_state_vector(most_recent_state_vector)

                self.generate_coast(orbit)

    def ta_at_intersection_line(self, node_line: np.ndarray, rotation_matrix: np.ndarray) -> float:
        node_line_in_perifocal = np.dot(np.linalg.inv(rotation_matrix), node_line)
        return np.arctan2(node_line_in_perifocal[1], node_line_in_perifocal[0])

    def generate_hohman_like_transfer(self, starting_orbit, target_orbit, plotting=False, ax=None):        
        starting_orbit = starting_orbit
        # print(f"Starting argument of perigee: {starting_orbit.argument_of_perigee}")

        transfer_orbit = o.Orbit()
        target_orbit = target_orbit

        n_1 = starting_orbit.normal
        n_2 = target_orbit.normal
        angle_diff = la.angle_between_vectors(n_1, n_2)
        intersection_line = np.cross(n_1, n_2)

        # if plotting:
        #     print("Intersection Line")
        #     print(intersection_line*1000)
        #     pl.plot_unit_vector(ax, intersection_line, "red", label="intersction_line")

        ta_starting_orbit_at_node_line = self.ta_at_intersection_line(
            -intersection_line, 
            starting_orbit.perifocal_to_geocentric_rotation()
        )

        ta_target_orbit_at_node_line = self.ta_at_intersection_line(
            intersection_line, 
            target_orbit.perifocal_to_geocentric_rotation()
        )

        hohmann_perigee = starting_orbit.get_radius(ta_starting_orbit_at_node_line)
        hohmann_apogee = target_orbit.get_radius(ta_target_orbit_at_node_line)
        
        # if plotting:

        #     r = starting_orbit.get_radius_vector(ta_starting_orbit_at_node_line)
        #     p = starting_orbit.get_radius_vector(0)
            
        #     print(starting_orbit)
        #     print(f"True anomaly in starting orbit at departure point: {ta_starting_orbit_at_node_line}")
        #     print(f"Displacement at departure point: {r}")
        #     print(f"Perigee of starting orbit: {p}")
        #     pl.plot_current_position(ax, r, "purple", label=f"target_departure_point: {ta_starting_orbit_at_node_line}")
        #     pl.plot_current_position(ax, p, "pink", label=f"starting_perigee")

        # if plotting:
        #     r = target_orbit.get_radius_vector(ta_target_orbit_at_node_line)
        #     pl.plot_current_position(ax, r, "red", label=f"target_arrive_point: {ta_target_orbit_at_node_line}")

        # print("Starting_orbit")
        # print(f"  (perigee, apogee): {(starting_orbit.radius_of_perigee, starting_orbit.radius_of_apogee)}")
        # print("Transfer_orbit")
        # print(f"  (perigee, apogee): {(hohmann_perigee, hohmann_apogee)}")
        # print("Target_orbit")
        # print(f"  (perigee, apogee): {(target_orbit.radius_of_perigee, target_orbit.radius_of_apogee)}")

        if (hohmann_apogee < hohmann_perigee):
            ta_transfer = np.pi
            temp = hohmann_apogee
            hohmann_apogee = hohmann_perigee
            hohmann_perigee = temp
        else:
            ta_transfer = 0

        semi_major_axis = (hohmann_perigee + hohmann_apogee)/2

        # print("Semi-major-axis")
        # print(semi_major_axis)
        r = starting_orbit.get_radius_vector(ta_starting_orbit_at_node_line)
        radius = np.linalg.norm(r)

        # Velocity vector at the starting point
        v_starting = starting_orbit.get_velocity(ta_starting_orbit_at_node_line)

        # Calculate the specific angular momentum vector as the cross product of radius and velocity
        h_vector = np.cross(r, v_starting)

        # Compute the desired velocity magnitude from the vis viva equation
        v_mag = np.sqrt(self.MU * (2 / radius - 1 / semi_major_axis))

        # Determine the velocity direction using the cross product of h_vector and r
        v_direction = np.cross(h_vector, r)
        v_direction /= np.linalg.norm(v_direction)  # Normalize to unit vector

        # Scale to the correct magnitude
        v = v_mag * v_direction

        state_vector = np.concatenate((r, v))
        
        # if ax:
        #     pl.plot_state(ax, state_vector, "yellow")
        
        # Create transfer orbit
        transfer_orbit.calculate_initial_parameters_from_state_vector(state_vector)

        # print("Radius comparison")
        # print((starting_orbit.get_radius(ta_starting_orbit_at_node_line), target_orbit.get_radius(ta_target_orbit_at_node_line)))
        # print((hohmann_perigee, hohmann_apogee))
        # print((transfer_orbit.get_radius(0), transfer_orbit.get_radius(np.pi)))

        # print("Radius vector comparison")
        # print((starting_orbit.get_radius_vector(ta_starting_orbit_at_node_line), target_orbit.get_radius_vector(ta_target_orbit_at_node_line)))
        # print((hohmann_perigee, hohmann_apogee))
        # print((transfer_orbit.get_radius_vector(0), transfer_orbit.get_radius_vector(np.pi)))

        # if plotting:
        #     pl.plot_orbit(ax, transfer_orbit)
        #     p_transfer = transfer_orbit.get_radius_vector(ta_transfer)
        #     pl.plot_current_position(ax, p_transfer, "red", "hohmann_perigee", marker='x')

        #     r = transfer_orbit.get_radius_vector(0)
        #     pl.plot_current_position(ax, r, "green", label=f"hohmann ta=0")

        #     r = transfer_orbit.get_radius_vector(np.pi)
        #     pl.plot_current_position(ax, r, "blue", label=f"hohmannta=pi")

        coast_1_duration = starting_orbit.calculate_time_between_anomalies(
            self.ta_at_last_impulse, ta_starting_orbit_at_node_line)

        print(f"Coast start ta: {self.ta_at_last_impulse}")
        print(f"Coast end ta: {ta_starting_orbit_at_node_line}")
        print(f"Coast duration: {coast_1_duration}")
        print(f"Orbital Period of starting orbit: {starting_orbit.orbital_period}")
        
        # print("Ta stuff:")
        # print((self.ta_at_last_impulse, ta_starting_orbit_at_node_line))

        # Propagate the position
        self.simulate_coast(coast_1_duration)
        
        # if ax:
        #     last_state = self.solution_array_segments[-1][:, -1]
        #     pl.plot_state(ax, last_state, "orange")

        self.simulate_impulse(transfer_orbit.get_velocity(ta_transfer))
        
        last_state = self.solution_array_segments[-1][:, -1]
        
        # if ax:
            # pl.plot_state(ax, last_state, "orange")

        current_orbit = o.Orbit()
        current_orbit.calculate_initial_parameters_from_state_vector(last_state)
        coast_2_duration = current_orbit.orbital_period / 2

        self.simulate_coast(coast_2_duration)
        self.simulate_impulse(target_orbit.get_velocity(ta_target_orbit_at_node_line))

        last_state = self.solution_array_segments[-1][:, -1]

        current_orbit.calculate_initial_parameters_from_state_vector(last_state)

        self.ta_at_last_impulse = current_orbit.initial_true_anomaly

    def generate_coast(self, orbit):
        period = orbit.orbital_period

        self.simulate_coast(period)



def main() -> None:
    return

if __name__ == "__main__":
    main()