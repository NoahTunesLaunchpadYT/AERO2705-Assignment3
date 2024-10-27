# orbit.py
# Author: 530494928
# Date: 2024-08-23
# Description: Defines an Orbit class and provides methods for calculating orbital parameters.
# Includes methods for calculating semi-major axis, eccentricity, velocity, and simulating orbits
# with and without oblateness effects (J2 perturbation).

import numpy as np
from scipy.integrate import solve_ivp
from ground_station_simulations import linear_algebra as la
from ground_station_simulations import definitions as d

class Orbit:
    """
    The Orbit class represents an orbital path around Earth and provides methods to calculate 
    key orbital parameters and simulate satellite motion. It supports calculating orbital elements 
    such as semi-major axis, eccentricity, specific angular momentum, velocities at perigee and apogee, 
    and the orbital period based on initial conditions or TLE (Two-Line Element) data. 

    The class also includes functionality for numerically simulating orbits using differential equations 
    with or without considering Earth's oblateness (J2 effects). It allows for the calculation of key 
    orbital parameters such as radius, velocity, and position at any true anomaly, and can output 3D 
    visualizations of the orbit.

    Attributes:
        MU (float): Earth's gravitational constant (km^3/s^2).
        EQUATORIAL_RADIUS (float): Earth's equatorial radius (km).
        ANGULAR_TOLERANCE (float): Small tolerance for angular calculations (radians).
        J2 (float): Earth's second zonal harmonic (oblateness constant).
        SIMULATION_RESOLUTION (int): Resolution for numerical simulations (number of steps).

        semi_major_axis (float): Semi-major axis of the orbit (km).
        eccentricity (float): Orbital eccentricity (dimensionless).
        specific_angular_momentum (float): Specific angular momentum (km^2/s).
        specific_energy (float): Specific orbital energy (km^2/s^2).
        orbital_period (float): Orbital period (seconds).
        radius_of_perigee (float): Radius at perigee (km).
        radius_of_apogee (float): Radius at apogee (km).
        altitude_of_perigee (float): Altitude of perigee above Earth's surface (km).
        altitude_of_apogee (float): Altitude of apogee above Earth's surface (km).
        velocity_at_perigee (float): Velocity at perigee (km/s).
        velocity_at_apogee (float): Velocity at apogee (km/s).
        mean_motion (float): Mean motion (revolutions per day).
        inclination_angle (float): Inclination angle of the orbit (radians).
        raan (float): Right Ascension of the Ascending Node (radians).
        argument_of_perigee (float): Argument of perigee (radians).
        normal (np.ndarray): Normal vector to the orbital plane.
        initial_mean_anomaly (float): Initial mean anomaly (radians).
        initial_true_anomaly (float): Initial true anomaly (radians).
        initial_displacement (np.ndarray): Initial displacement (position vector in ECI frame).
        initial_velocity (np.ndarray): Initial velocity vector (in ECI frame).
        simulation_duration (float): Duration for orbital simulation (seconds).
        solution_t (np.ndarray): Time array from the orbit simulation.
        solution_y (np.ndarray): State vector array (position, velocity) from the simulation.
        current_rotation (float): Current rotation angle of the orbit (radians).

    This class provides methods to:
    - Parse orbital elements from TLE data or a dictionary of parameters.
    - Compute orbital constants such as semi-major axis, eccentricity, period, and specific energy.
    - Calculate positions and velocities at any point in the orbit.
    - Simulate orbital motion with or without considering Earth's oblateness (J2 effects).
    - Visualize orbits in 3D and plot ground tracks.
    """

    MU = d.MU  # Gravitational constant (km^3/s^2)
    EQUATORIAL_RADIUS = d.EQUATORIAL_RADIUS # Earth’s equatorial radius (km)

    def __init__(self) -> None:
        """
        Initialize the Orbit object with default values for orbital elements.
        """    
        # Orbit Constants
        self.semi_major_axis = None
        self.eccentricity = None
        self.specific_angular_momentum = None
        self.specific_energy = None
        self.orbital_period = None
        self.radius_of_perigee = None
        self.radius_of_apogee = None
        self.altitude_of_perigee = None
        self.altitude_of_apogee = None
        self.inclination_angle = None 
        self.raan = None 
        self.argument_of_perigee = None 
        self.normal = None

        self.velocity_at_perigee = None
        self.velocity_at_apogee = None
        self.mean_motion = None
        self.greenwhich_sidereal_time = None

        # Orbiter Parameters
        self.initial_mean_anomaly = 0 # Radians
        self.initial_true_anomaly = None # Radians
        self.initial_displacement = None
        self.initial_velocity = None

        # Rotations
        self.current_rotation = 0.0  # Starting with no rotation
  
    def calculate_parameters_from_dictionary(self, params: dict) -> None:
        """
        Parse a dictionary of orbital parameters and set the corresponding attributes.

        Args:
            params (dict): Dictionary containing parameters such as altitude of perigee/apogee, 
                           inclination angle, RAAN, and argument of perigee.
        """   
        self.altitude_of_perigee = params["altitude_of_perigee"]
        self.altitude_of_apogee = params["altitude_of_apogee"]
        self.inclination_angle = np.radians(params["inclination_angle"])
        self.raan = np.radians(params["raan"])
        self.argument_of_perigee = np.radians(params["argument_of_perigee"])
        self.initial_true_anomaly = np.radians(params["initial_true_anomaly"])
        
        # Calculating parameters
        self.calculate_radius_of_perigee_from_altitude()
        self.calculate_radius_of_apogee_from_altitude()
        self.calculate_semi_major_axis_from_radii()
        self.calculate_period_from_semi_major_axis()
        self.calculate_eccentricity()
        self.calculate_specific_angular_momentum()
        self.calculate_specific_energy()
        self.calculate_velocity_at_perigee()
        self.calculate_velocity_at_apogee()
        self.calculate_normal()
        self.calculate_initial_displacement()
        self.calculate_initial_velocity()

    def calculate_initial_parameters_from_state_vector(self, state_vector) -> None:
        # Obtain displacement and veloctiy
        self.initial_displacement = state_vector[:3]
        self.initial_velocity = state_vector[3:6]

        # Calculate parameters directly from the state vector
        self.calculate_specific_angular_momentum_from_state_vector()
        self.calculate_specific_energy_from_state_vector()

        # Use specific angular momentum to calculate eccentricity
        self.calculate_eccentricity_from_angular_momentum()

        # Calculate other orbital parameters using derived quantities
        self.calculate_semi_major_axis_from_energy()
        self.calculate_inclination_angle_from_state_vector()
        self.calculate_raan_from_state_vector()
        self.calculate_argument_of_perigee_from_state_vector()

        # Calculate the initial true anomaly using eccentricity, angular momentum, and radius
        self.calculate_true_anomaly_from_e_h_r()

        # Calculate the remaining parameters
        self.calculate_radius_of_perigee_from_eccentricity()
        self.calculate_radius_of_apogee_from_eccentricity()
        self.calculate_altitude_of_perigee()
        self.calculate_altitude_of_apogee()
        self.calculate_velocity_at_perigee()
        self.calculate_velocity_at_apogee()
        self.calculate_normal()


    def calculate_specific_angular_momentum_from_state_vector(self) -> None:
        # Specific angular momentum h = r × v
        r = self.initial_displacement
        v = self.initial_velocity
        self.specific_angular_momentum_vec = np.cross(r, v)
        self.specific_angular_momentum = np.linalg.norm(np.cross(r, v))
    
    def calculate_specific_energy_from_state_vector(self) -> None:
        # Specific orbital energy ε = v^2/2 - μ/r
        r_norm = np.linalg.norm(self.initial_displacement)
        v_norm = np.linalg.norm(self.initial_velocity)
        self.specific_energy = (v_norm**2) / 2 - self.MU / r_norm
    
    def calculate_eccentricity_from_angular_momentum(self) -> None:
        # Eccentricity vector e = (v × h)/μ - r̂
        r = self.initial_displacement
        v = self.initial_velocity
        h = self.specific_angular_momentum_vec
        e_vec = (np.cross(v, h) / self.MU) - (r / np.linalg.norm(r))
        self.eccentricity = np.linalg.norm(e_vec)
        self.eccentricity_vector = e_vec
    
    def calculate_semi_major_axis_from_energy(self) -> None:
        # Semi-major axis a = -μ / (2ε)
        self.semi_major_axis = -self.MU / (2 * self.specific_energy)
    
    def calculate_inclination_angle_from_state_vector(self) -> None:
        # Inclination i = arccos(h_z / |h|)
        h = self.specific_angular_momentum_vec
        h_norm = np.linalg.norm(h)
        self.inclination_angle = np.arccos(h[2] / h_norm)
    
    def calculate_raan_from_state_vector(self) -> None:
        # RAAN Ω = arccos(n_x / |n|), adjust for quadrant
        h = self.specific_angular_momentum_vec
        k = np.array([0, 0, 1])  # Z-axis
        n = np.cross(k, h)  # Node vector
        n_norm = np.linalg.norm(n)
        
        if n_norm != 0:
            self.raan = np.arccos(n[0] / n_norm)
            if n[1] < 0:  # Adjust for quadrant
                self.raan = 2 * np.pi - self.raan
        else:
            self.raan = 0  # For equatorial orbits
    
    def calculate_argument_of_perigee_from_state_vector(self) -> None:
        # Argument of perigee ω = arccos(n · e / (|n| * |e|)), adjust for quadrant
        k = np.array([0, 0, 1])
        h = self.specific_angular_momentum_vec
        n = np.cross(k, h)  # Node vector
        n_norm = np.linalg.norm(n)
        e = self.eccentricity_vector
        e_norm = np.linalg.norm(e)
        
        if n_norm != 0 and e_norm != 0:
            cos_omega = np.dot(n, e) / (n_norm * e_norm)
            self.argument_of_perigee = np.arccos(np.clip(cos_omega, -1, 1))
            if e[2] < 0:  # Adjust for quadrant
                self.argument_of_perigee = 2 * np.pi - self.argument_of_perigee
        else:
            self.argument_of_perigee = 0  # Circular or equatorial orbit
    
    def calculate_true_anomaly_from_e_h_r(self) -> None:
        # True anomaly θ = arccos(e · r / (|e| * |r|)), adjust for quadrant
        r = self.initial_displacement
        e = self.eccentricity_vector
        r_norm = np.linalg.norm(r)
        e_norm = np.linalg.norm(e)
        
        if e_norm != 0:
            cos_theta = np.dot(e, r) / (e_norm * r_norm)
            self.true_anomaly = np.arccos(np.clip(cos_theta, -1, 1))
            if np.dot(r, self.initial_velocity) < 0:  # Adjust for quadrant
                self.true_anomaly = 2 * np.pi - self.true_anomaly
        else:
            self.true_anomaly = 0  # Circular orbit
    
    def calculate_radius_of_perigee_from_eccentricity(self) -> None:
        # Radius of perigee r_p = a * (1 - e)
        self.radius_of_perigee = self.semi_major_axis * (1 - self.eccentricity)
    
    def calculate_radius_of_apogee_from_eccentricity(self) -> None:
        # Radius of apogee r_a = a * (1 + e)
        self.radius_of_apogee = self.semi_major_axis * (1 + self.eccentricity)
    
    def calculate_period_from_mean_motion(self) -> None:
        """
        Calculate the orbital period in seconds based on the mean motion.
        """
        # Convert mean motion from rev/day to radians/second
        mean_motion_rads = self.mean_motion * 2 * np.pi / 86400
        self.orbital_period = 2 * np.pi / mean_motion_rads

    def calculate_semi_major_axis_from_mean_motion(self) -> None:
        """
        Calculate the semi-major axis based on the mean motion using Kepler's Third Law.
        """
        # Mean motion in radians/second
        mean_motion_rads = self.mean_motion * 2 * np.pi / 86400
        self.semi_major_axis = (self.MU / (mean_motion_rads**2))**(1/3)

    def calculate_velocity_at_perigee(self) -> float:
        """
        Calculate the velocity at perigee using the vis-viva equation.
        """
        self.velocity_at_perigee = np.sqrt(self.MU * (2 / self.radius_of_perigee - 1 / self.semi_major_axis))

    def calculate_velocity_at_apogee(self) -> float:
        """
        Calculate the velocity at apogee using the vis-viva equation.
        """
        self.velocity_at_apogee = np.sqrt(self.MU * (2 / self.radius_of_apogee - 1 / self.semi_major_axis))

    def calculate_specific_angular_momentum(self) -> None:
        """
        Calculate the specific angular momentum (h) in km^2/s.
        """
        # Semi-latus rectum (p) = a(1 - e^2), where a is semi-major axis and e is eccentricity
        semi_latus_rectum = self.semi_major_axis * (1 - self.eccentricity**2)
        self.specific_angular_momentum = np.sqrt(self.MU * semi_latus_rectum)

    def calculate_specific_energy(self) -> None:
        """
        Calculate the specific orbital energy (ε) in km^2/s^2.
        """
        self.specific_energy = -self.MU / (2 * self.semi_major_axis)

    def calculate_altitude_of_perigee(self) -> None:
        """
        Calculate the altitude of perigee above Earth's surface in kilometers.
        """
        self.altitude_of_perigee = self.radius_of_perigee - self.EQUATORIAL_RADIUS

    def calculate_altitude_of_apogee(self) -> None:
        """
        Calculate the altitude of apogee above Earth's surface in kilometers.
        """
        self.altitude_of_apogee = self.radius_of_apogee - self.EQUATORIAL_RADIUS

    def calculate_radius_of_perigee_from_altitude(self) -> None:
        """
        Calculate the radius of perigee based on the altitude of perigee 
        and Earth's equatorial radius.
        """
        self.radius_of_perigee = self.altitude_of_perigee + self.EQUATORIAL_RADIUS

    def calculate_radius_of_apogee_from_altitude(self) -> None:
        """
        Calculate the radius of apogee based on the altitude of apogee 
        and Earth's equatorial radius.
        """     
        self.radius_of_apogee = self.altitude_of_apogee + self.EQUATORIAL_RADIUS

    def calculate_semi_major_axis_from_radii(self) -> None:
        """
        Calculate the semi-major axis based on the radii of perigee and apogee.
        """    
        self.semi_major_axis = (self.radius_of_apogee + self.radius_of_perigee) /2

    def calculate_period_from_semi_major_axis(self) -> None:
        """
        Calculate the orbital period based on the semi-major axis using Kepler's Third Law.
        """
        mu = self.MU
        a = self.semi_major_axis
        T = 2 * np.pi / np.sqrt(mu) * a**(3/2)
        self.orbital_period = T

    def calculate_eccentricity(self) -> None:
        """
        Calculate the eccentricity based on the radii of perigee and apogee.
        """    
        r_a = self.radius_of_apogee
        r_p = self.radius_of_perigee
        e = (r_a - r_p) / (r_a + r_p)
        self.eccentricity = e

    def calculate_normal(self):
        """
        Calculate the normal vector to the orbital plane.

        Returns:
            np.ndarray: A 3D normal vector [n_x, n_y, n_z] perpendicular to the orbital plane.
        """
        # Extract inclination and RAAN
        i = self.inclination_angle  # Inclination in radians
        Omega = self.raan  # RAAN in radians

        # Calculate the normal vector
        n_x = np.sin(Omega) * np.sin(i)
        n_y = -np.cos(Omega) * np.sin(i)
        n_z = np.cos(i)

        # Return the normal vector as a NumPy array
        self.normal = np.array([n_x, n_y, n_z])

    def rotate_velocity(self, target_theta: float):
        """
        Adjust the velocity vector to match a target rotation relative to its initial state.
        This method updates the velocity vector based on the difference between the current
        rotation and the target rotation.

        Args:
            target_theta (float): The target rotation angle in radians.
        """
        # Calculate the difference needed to reach the target rotation
        rotation_difference = target_theta - self.current_rotation

        # Ensure initial displacement exists
        if self.initial_displacement is None:
            raise ValueError("Initial displacement vector is not defined.")

        # Get the axis (normalized initial displacement vector)
        axis = self.initial_displacement / np.linalg.norm(self.initial_displacement)

        # Use the helper function to rotate the velocity vector
        self.initial_velocity = la.rotate_about(self.initial_velocity, axis, rotation_difference)

        # Update the current rotation to the new target rotation
        self.current_rotation = target_theta

        # Recalculate orbital elements based on the new velocity
        self.recalculate_orbital_elements()

    def recalculate_orbital_elements(self):
        """
        Recalculate orbital elements such as semi-major axis, eccentricity, etc.,
        based on the current state vectors (position and velocity).
        """
        r = np.linalg.norm(self.initial_displacement)
        v = np.linalg.norm(self.initial_velocity)
        mu = self.MU

        # Specific orbital energy (epsilon)
        epsilon = v**2 / 2 - mu / r

        # Semi-major axis (a)
        self.semi_major_axis = -mu / (2 * epsilon)

        # Specific angular momentum (h) vector
        h_vector = np.cross(self.initial_displacement, self.initial_velocity)
        self.specific_angular_momentum = np.linalg.norm(h_vector)

        # Eccentricity vector and magnitude (e)
        e_vector = (np.cross(self.initial_velocity, h_vector) / mu) - (self.initial_displacement / r)
        self.eccentricity = np.linalg.norm(e_vector)

        # Inclination (i)
        h_z = h_vector[2]
        self.inclination_angle = np.arccos(h_z / self.specific_angular_momentum)

        # Right ascension of ascending node (Omega)
        n_vector = np.cross([0, 0, 1], h_vector)
        n_mag = np.linalg.norm(n_vector)
        if n_mag != 0:
            self.raan = np.arctan2(n_vector[1], n_vector[0])
        else:
            self.raan = 0  # For equatorial orbits

        # Argument of perigee (omega)
        if n_mag != 0 and self.eccentricity > 0:
            e_n = np.dot(e_vector, n_vector) / (n_mag * self.eccentricity)
            e_x = e_vector[0]
            if e_n > 1:
                e_n = 1
            elif e_n < -1:
                e_n = -1
            self.argument_of_perigee = - np.arccos(e_n)
            if e_x < 0:
                self.argument_of_perigee = - 2 * np.pi - self.argument_of_perigee
        else:
            self.argument_of_perigee = 0  # Undefined for circular or equatorial orbits

        # Update the period and other related parameters as needed
        self.orbital_period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / mu)
    
    def get_radius(self, ta: float) -> float:
        """
        Uses specific angular momentum and true anomaly to find the radius 
        at a specific true anomaly.

        Args:
            ta (float): True anomaly in radians.

        Returns:
            float: The radius at the given true anomaly.
        """
        h = self.specific_angular_momentum  # Specific angular momentum (h)
        e = self.eccentricity  # Eccentricity (e)
        mu = self.MU  # Standard gravitational parameter (mu)
        
        # Calculate the radius using specific angular momentum
        radius = (h ** 2 / mu) / (1 + e * np.cos(ta))
        
        return radius
    
    def get_radius_vector(self, true_anomaly: float) -> np.ndarray:
        """
        Calculate the position vector at a given true anomaly in the ECI frame.

        Args:
            true_anomaly (float): The true anomaly in radians for which to compute the position vector.

        Returns:
            np.ndarray: The position vector at the given true anomaly in the ECI frame (3D).
        """
        # Get the radius at the specified true anomaly using the get_radius function
        radius = self.get_radius(true_anomaly)

        # Position vector in the perifocal frame
        r_pf = np.array([
            radius * np.cos(true_anomaly),  # X-component in the perifocal frame
            radius * np.sin(true_anomaly),  # Y-component in the perifocal frame
            0  # Z-component is 0 in the perifocal frame
        ])

        # Convert the position vector from the perifocal frame to the ECI frame
        rotation_matrix = self.perifocal_to_geocentric_rotation()
        r_eci = np.dot(rotation_matrix, r_pf)

        return r_eci

    
    def get_velocity(self, ta: float) -> np.ndarray:
        """
        Calculates the velocity vector in the ECI frame for a given true anomaly (ta).

        Args:
            ta (float): True anomaly in radians.

        Returns:
            np.ndarray: Velocity vector in the ECI frame (3D).
        """
        # Specific angular momentum (h), eccentricity (e), and standard gravitational parameter (mu)
        h = self.specific_angular_momentum
        e = self.eccentricity
        mu = self.MU

        # Calculate radial and tangential velocity components in the perifocal frame
        v_r = (mu / h) * e * np.sin(ta)  # Radial velocity component
        v_t = (mu / h) * (1 + e * np.cos(ta))  # Tangential velocity component

        # Velocity vector in the perifocal frame
        v_pf = np.array([
            v_r * np.cos(ta) - v_t * np.sin(ta),  # X component in perifocal frame
            v_r * np.sin(ta) + v_t * np.cos(ta),  # Y component in perifocal frame
            0  # Z component in perifocal frame (always 0 in perifocal frame)
        ])

        # Get the rotation matrix to convert from perifocal to ECI frame
        rotation_matrix = self.perifocal_to_geocentric_rotation()

        # Transform the velocity vector to the ECI frame
        v_eci = np.dot(rotation_matrix, v_pf)

        return v_eci

    def calculate_initial_true_anomaly(self) -> None:
        """ Calculate Eccentric Anomaly from Mean anomaly, using Newton's
        using Newton's method.
        M_e = E - esin(E)           (3.11)
        f(E) = E - esin(E) - M_e 
        Want to make f(E) = 0
        f'(E) = 1 - ecos(E)
        E_{i+1} = E_i - (E_i - esin(E_i) - M_e)/(1 - ecos(E))
        If (E_i - esin(E_i) - M_e)/(1 - ecos(E)) < tollerance, stop"""

        M_e = self.initial_mean_anomaly
        e = self.eccentricity

        # Initial estimate of E (Algorithm 3.1)
        if M_e < np.pi:
            E = M_e + e / 2
        else:
            E = M_e - e / 2

        # Newton's method iteration to solve for E (Eccentric Anomaly)
        while True:
            f_E = E - e * np.sin(E) - M_e
            f_prime_E = 1 - e * np.cos(E)

            # Update the estimate of E
            ratio = f_E / f_prime_E
            E = E - ratio

            # If the change is smaller than the tolerance, stop iterating
            if abs(ratio) < self.ANGULAR_TOLERANCE:
                break
        
        # Calculate tru anomaly (3.10b)

        self.initial_true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * 
                                                   np.sin(E / 2), 
                                                   np.sqrt(1 - e) * 
                                                   np.cos(E / 2))

    def calculate_time_between(self, ta1: float, ta2: float) -> float:
        """
        Calculate the time between two true anomalies (ta1 and ta2) using Kepler's law.

        Args:
            ta1 (float): Initial true anomaly (in radians).
            ta2 (float): Final true anomaly (in radians).

        Returns:
            float: Time between the two anomalies in seconds.
        """

        # Ensure semi-major axis and eccentricity are defined
        if self.semi_major_axis is None or self.eccentricity is None:
            raise ValueError("Semi-major axis or eccentricity is not defined.")

        # Mean motion (n) in radians per second
        mean_motion = np.sqrt(self.MU / (self.semi_major_axis**3))

        # Convert true anomalies to eccentric anomalies (E)
        E1 = 2 * np.arctan(np.sqrt((1 - self.eccentricity) / (1 + self.eccentricity)) * np.tan(ta1 / 2))
        E2 = 2 * np.arctan(np.sqrt((1 - self.eccentricity) / (1 + self.eccentricity)) * np.tan(ta2 / 2))

        # Calculate the mean anomalies (M1 and M2) from the eccentric anomalies (Kepler's equation)
        M1 = E1 - self.eccentricity * np.sin(E1)
        M2 = E2 - self.eccentricity * np.sin(E2)

        # Time between the two anomalies
        dt = (M2 - M1) / mean_motion

        # Handle time wrapping around the orbit if the result is negative
        if dt < 0:
            dt += self.orbital_period

        return dt

    def perifocal_to_geocentric_rotation(self) -> np.ndarray:
        """Create the rotation matrix from perifocal to geocentric equatorial frame using
        the orbital elements (RAAN, inclination, and argument of perigee)."""

        # Convert angles to radians for matrix computation
        i = self.inclination_angle  # Inclination
        raan = self.raan  # Right Ascension of Ascending Node
        arg_perigee = self.argument_of_perigee  # Argument of perigee

        # Compute the components of the rotation matrix
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        cos_arg_perigee = np.cos(arg_perigee)
        sin_arg_perigee = np.sin(arg_perigee)
        cos_i = np.cos(i)
        sin_i = np.sin(i)

        # Rotation matrix from perifocal to geocentric equatorial frame (4.44)
        rotation_matrix = np.array([
            [cos_raan * cos_arg_perigee - sin_raan * sin_arg_perigee * cos_i, -cos_raan * sin_arg_perigee - sin_raan * cos_arg_perigee * cos_i, sin_raan * sin_i],
            [sin_raan * cos_arg_perigee + cos_raan * sin_arg_perigee * cos_i, -sin_raan * sin_arg_perigee + cos_raan * cos_arg_perigee * cos_i, -cos_raan * sin_i],
            [sin_arg_perigee * sin_i, cos_arg_perigee * sin_i, cos_i]
        ])

        return rotation_matrix

    def calculate_initial_displacement(self) -> None:
        """Calculate the initial displacement (position vector) in the perifocal frame,
        and transform it to the geocentric equatorial frame."""

        # Compute the radius in the perifocal frame (2.62)
        r = self.semi_major_axis * (1 - self.eccentricity ** 2) / (1 + self.eccentricity * np.cos(self.initial_true_anomaly))

        # Perifocal frame position vector
        r_pf = np.array([
            r * np.cos(self.initial_true_anomaly),  # x-component
            r * np.sin(self.initial_true_anomaly),  # y-component
            0  # z-component (always 0 in perifocal frame)
        ])

        # Rotation matrix from perifocal to geocentric equatorial frame
        rotation_matrix = self.perifocal_to_geocentric_rotation()

        # Transform to geocentric equatorial frame
        self.initial_displacement = np.dot(rotation_matrix, r_pf)


    def calculate_initial_velocity(self) -> None:
        """Calculate the initial velocity vector in the perifocal frame,
        and transform it to the geocentric equatorial frame."""

        # Compute the velocity in the perifocal frame 
        h = self.specific_angular_momentum  # specific angular momentum

        r = np.linalg.norm(self.initial_displacement)

        # Perifocal frame velocity vector (v_pf) using your provided equations
        v_r = (self.MU / h) * self.eccentricity * np.sin(self.initial_true_anomaly)  # Radial velocity (2.39)
        v_perp = (self.MU / h) * (1 + self.eccentricity * np.cos(self.initial_true_anomaly))  # Tangential velocity (2.38)

        # Velocity vector in the perifocal frame (v_pf)
        v_pf = np.array([
            v_r * np.cos(self.initial_true_anomaly) - v_perp * np.sin(self.initial_true_anomaly),  # x-component
            v_r * np.sin(self.initial_true_anomaly) + v_perp * np.cos(self.initial_true_anomaly),  # y-component
            0  # z-component (always 0 in perifocal frame)
        ])

        # Rotation matrix from perifocal to geocentric equatorial frame
        rotation_matrix = self.perifocal_to_geocentric_rotation()

        # Transform to geocentric equatorial frame
        self.initial_velocity = np.dot(rotation_matrix, v_pf)

    def __repr__(self):
        """String representation of the orbital parameters."""
        return (
            f"Orbital Parameters:\n"
            f"  Inclination Angle: {self.inclination_angle} radians\n"
            f"  RAAN: {self.raan} radians\n"
            f"  Eccentricity: {self.eccentricity}\n"
            f"  Argument of Perigee: {self.argument_of_perigee} radians\n"
            f"  Mean Anomaly: {self.initial_mean_anomaly} rad\n"
            f"  Mean Motion: {self.mean_motion} rev/day\n"
            f"  Semi-Major Axis: {self.semi_major_axis} km\n"
            f"  Orbital Period: {self.orbital_period} seconds\n"
            f"  Radius of Perigee: {self.radius_of_perigee} km\n"
            f"  Radius of Apogee: {self.radius_of_apogee} km\n"
            f"  Specific Angular Momentum: {self.specific_angular_momentum} km^2/s\n"
            f"  Specific Energy: {self.specific_energy} km^2/s^2\n"
            f"  Altitude of Perigee: {self.altitude_of_perigee} km\n"
            f"  Altitude of Apogee: {self.altitude_of_apogee} km\n"
            f"  Velocity at Perigee: {self.velocity_at_perigee:.4f} km/s\n"
            f"  Velocity at Apogee: {self.velocity_at_apogee:.4f} km/s"
        )

def main() -> None:

    return 

if __name__ == "__main__":
    main()