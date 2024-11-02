import numpy as np
import matplotlib.pyplot as plt

# Constants for Earth
MU = 3.986e14  # Standard gravitational parameter for Earth (m^3/s^2)

class OrbitalMechanics:
    """
    A class to model basic orbital mechanics of a satellite.

    Attributes:
    -----------
    a : float
        Semi-major axis of the orbit (in meters).
    e : float
        Eccentricity of the orbit.
    i : float
        Inclination of the orbit (in radians).
    RAAN : float
        Right ascension of the ascending node (in radians).
    arg_perigee : float
        Argument of perigee (in radians).
    true_anomaly : float
        True anomaly at the epoch (in radians).

    Methods:
    --------
    orbital_velocity(r):
        Calculate the orbital velocity at a given radius.
    position_orbital_plane(theta):
        Calculate the satellite's position in the orbital plane.
    rotation_matrix():
        Return the full rotation matrix for converting from the orbital plane to the ECI frame.
    satellite_position_velocity():
        Calculate the satellite's position and velocity in ECI coordinates.
    """

    def __init__(self, a, e, i, RAAN, arg_perigee, true_anomaly):
        """
        Initializes the orbital parameters for the satellite.

        Parameters:
        -----------
        a : float
            Semi-major axis of the orbit (in meters).
        e : float
            Eccentricity of the orbit.
        i : float
            Inclination of the orbit (in radians).
        RAAN : float
            Right ascension of the ascending node (in radians).
        arg_perigee : float
            Argument of perigee (in radians).
        true_anomaly : float
            True anomaly at the epoch (in radians).
        """
        self.a = a
        self.e = e
        self.i = i
        self.RAAN = RAAN
        self.arg_perigee = arg_perigee
        self.true_anomaly = true_anomaly

    def orbital_velocity(self, r):
        """
        Calculate the orbital velocity at a given radius.

        Parameters:
        -----------
        r : float
            Radius at which the velocity is being calculated (in meters).

        Returns:
        --------
        float
            Orbital velocity (in meters per second).
        """
        return np.sqrt(MU * (2 / r - 1 / self.a))

    def position_orbital_plane(self, theta):
        """
        Calculate the satellite's position in the orbital plane.

        Parameters:
        -----------
        theta : float
            True anomaly of the satellite (in radians).

        Returns:
        --------
        tuple(np.ndarray, float)
            The position of the satellite in the orbital plane (x, y, 0) and the radius.
        """
        r = self.a * (1 - self.e ** 2) / (1 + self.e * np.cos(self.true_anomaly))
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)
        return np.array([x_orb, y_orb, 0]), r

    def rotation_matrix(self):
        """
        Return the full rotation matrix for converting from the orbital plane to the ECI frame.

        Returns:
        --------
        np.ndarray
            3x3 rotation matrix.
        """
        R_z_RAAN = np.array([
            [np.cos(self.RAAN), -np.sin(self.RAAN), 0],
            [np.sin(self.RAAN), np.cos(self.RAAN), 0],
            [0, 0, 1]
        ])
        R_x_i = np.array([
            [1, 0, 0],
            [0, np.cos(self.i), -np.sin(self.i)],
            [0, np.sin(self.i), np.cos(self.i)]
        ])
        R_z_arg_perigee = np.array([
            [np.cos(self.arg_perigee), -np.sin(self.arg_perigee), 0],
            [np.sin(self.arg_perigee), np.cos(self.arg_perigee), 0],
            [0, 0, 1]
        ])
        return R_z_RAAN @ R_x_i @ R_z_arg_perigee

    def satellite_position_velocity(self):
        """
        Calculate satellite position and velocity in the Earth-Centered Inertial (ECI) frame.

        Returns:
        --------
        tuple(np.ndarray, np.ndarray)
            Satellite position and velocity in the ECI frame.
        """
        position_orbital, r = self.position_orbital_plane(self.true_anomaly)
        v_orbital_mag = self.orbital_velocity(r)
        v_orbital = np.array([-np.sin(self.true_anomaly), self.e + np.cos(self.true_anomaly), 0])
        v_orbital = v_orbital * v_orbital_mag / np.linalg.norm(v_orbital)
        
        R = self.rotation_matrix()

        position_eci = R @ position_orbital
        velocity_eci = R @ v_orbital

        return position_eci, velocity_eci

class GasPlumeSimulator:
    """
    A class to simulate the dispersion of gas molecules in space.
    """

    def __init__(self, satellite_velocity: np.ndarray, N_molecules: int, dt: float, total_time: float,
                 P_tank: float, P_space: float, molar_mass: float, V_ullage: float, mass_gas: float) -> None:
        """
        Initialize the gas dispersion parameters and calculate initial values.
        """
        # Set heat capacity ratio for hydrazine (assumed value for simplicity)
        self.gamma = 1.085  
        
        # Calculate the number of moles of gas using mass and molar mass
        self.n_moles: float = self.calculate_moles_from_mass(mass_gas, molar_mass)
        
        # Calculate initial temperature inside the tank using the ideal gas law
        self.temp_in_tank: float = self.calculate_temperature_from_ullage(P_tank, V_ullage, self.n_moles)
        
        # Calculate the normalized opposite direction of satellite velocity for gas expansion
        self.v_gas_direction: np.ndarray = -satellite_velocity / np.linalg.norm(satellite_velocity)
        
        # Calculate initial speed of gas molecules based on choked flow theory
        self.gas_speed: float = self.calculate_choked_flow_speed(P_tank, P_space, self.temp_in_tank, molar_mass, self.gamma)
        
        # Set simulation parameters: number of molecules, time step, and total time for dispersion
        self.N_molecules: int = N_molecules
        self.dt: float = dt
        self.total_time: float = total_time

    def calculate_temperature_from_ullage(self, P: float, V_ullage: float, n_moles: float) -> float:
        """
        Calculate the temperature of the gas in the ullage using the ideal gas law.
        """
        # Universal gas constant in J/(mol·K)
        R: float = 8.314  
        
        # Calculate temperature in Kelvin using PV = nRT
        return (P * V_ullage) / (n_moles * R)

    def calculate_moles_from_mass(self, mass: float, molar_mass: float) -> float:
        """
        Calculate the number of moles of gas from its mass.
        """
        # Divide mass by molar mass to get moles (n = mass / molar_mass)
        return mass / molar_mass

    def calculate_choked_flow_speed(self, P1: float, P2: float, T: float, molar_mass: float, gamma: float) -> float:
        """
        Calculate gas speed based on pressure difference using the choked flow model.
        """
        # Universal gas constant in J/(mol K)
        R: float = 8.314  
        
        # Calculate the critical pressure ratio for choked flow conditions
        critical_pressure_ratio: float = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        
        # Calculate the pressure ratio between space and tank
        pressure_ratio: float = P2 / P1

        # If pressure ratio meets choked flow conditions, use choked flow speed equation
        if pressure_ratio <= critical_pressure_ratio:
            v_gas: float = np.sqrt((2 * gamma) / (gamma - 1) * (R * T / molar_mass) * (1 - pressure_ratio ** ((gamma - 1) / gamma)))
        else:
            # If not choked, use Bernoulli’s equation for pressure-driven expansion
            v_gas = np.sqrt(2 * (P1 - P2) / molar_mass)

        # Return calculated gas speed
        return v_gas

    def biased_gas_velocities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate initial velocities for gas molecules biased in the opposite direction to satellite velocity.
        """
        # Generate x-velocity components, biased towards gas flow direction and scaled by gas speed
        vx: np.ndarray = np.random.normal(loc=self.v_gas_direction[0], scale=0.25, size=self.N_molecules) * self.gas_speed
        
        # Generate y-velocity components with similar bias
        vy: np.ndarray = np.random.normal(loc=self.v_gas_direction[1], scale=0.25, size=self.N_molecules) * self.gas_speed
        
        # Generate z-velocity components, also biased in opposite direction to satellite
        vz: np.ndarray = np.random.normal(loc=self.v_gas_direction[2], scale=0.25, size=self.N_molecules) * self.gas_speed

        # Return the 3D velocity components
        return vx, vy, vz

    def simulate_gas_dispersion(self, satellite_position: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Simulate the time evolution of gas dispersion.
        """
        # Initialize velocities for all molecules with a biased direction
        vx, vy, vz = self.biased_gas_velocities()

        # Initialize x, y, and z positions at the satellite's starting position
        x: np.ndarray = np.ones(self.N_molecules) * satellite_position[0]
        y: np.ndarray = np.ones(self.N_molecules) * satellite_position[1]
        z: np.ndarray = np.ones(self.N_molecules) * satellite_position[2]

        # Determine the number of time steps in the simulation
        time_steps: int = int(self.total_time / self.dt)
        
        # Prepare an array to store positions of molecules over time
        positions: np.ndarray = np.zeros((time_steps, self.N_molecules, 3))
        
        # Track maximum distance traveled by any molecule from satellite
        max_distance: float = 0.0

        # Iterate over each time step
        for t in range(time_steps):
            # Update x, y, and z positions by adding velocity * time step
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            # Calculate distance of each molecule from the satellite's initial position
            distances: np.ndarray = np.sqrt((x - satellite_position[0])**2 + (y - satellite_position[1])**2 + (z - satellite_position[2])**2)
            
            # Update maximum distance if a greater distance is found
            max_distance = max(max_distance, np.max(distances))
            
            # Store the updated positions
            positions[t] = np.stack((x, y, z), axis=1)

        # Return the full position data and the maximum distance reached
        return positions, max_distance, self.gas_speed
    
    def calculate_plume_width(self, positions: np.ndarray) -> float:
        """
        Calculate the width of the widest part of the hydrazine plume.
        """
        # Direction vector for gas flow from satellite
        gas_direction = self.v_gas_direction
        
        # Extract final positions of all molecules (at the last time step)
        final_positions = positions[-1]
        
        # Project final positions onto plane perpendicular to the gas flow direction
        projected_positions = []
        for position in final_positions:
            # Subtract component in the direction of flow to get projection
            projection = position - np.dot(position, gas_direction) * gas_direction
            projected_positions.append(projection)
        projected_positions = np.array(projected_positions)

        # Initialize max distance variable
        max_distance = 0.0

        # Calculate pairwise distances between projected points to determine plume width
        for i in range(len(projected_positions)):
            for j in range(i+1, len(projected_positions)):
                # Calculate distance between two points
                distance = np.linalg.norm(projected_positions[i] - projected_positions[j])
                
                # Update max distance if current distance is greater
                if distance > max_distance:
                    max_distance = distance

        # Return maximum distance as the plume width
        return max_distance

    def plot_orbit_and_gas(self, positions: np.ndarray, satellite_position: np.ndarray,
                           orbit_positions_eci: np.ndarray, N_molecules: int) -> None:
        """
        Plot the gas dispersion and satellite orbit in 3D.
        """
        # Create a 3D plot for visualizing orbit and gas plume
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectories of each gas molecule in the dispersion plume
        for i in range(N_molecules):
            ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], alpha=0.5)

        # Plot satellite position and orbit
        # ax.plot(orbit_positions_eci[:, 0], orbit_positions_eci[:, 1], orbit_positions_eci[:, 2], 'b-', label='Satellite Orbit (ECI)')
        ax.scatter([satellite_position[0]], [satellite_position[1]], [satellite_position[2]], color='red', s=100, label='Satellite', marker='o')

        # Label axes for clarity
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        
        # Display legend and plot
        ax.legend()
        plt.show()

def main():
    """
    Main function to run the orbital mechanics and gas dispersion simulation.

    The function sets up the orbital parameters, initializes the satellite, and simulates
    the gas dispersion from the satellite's fuel tank in space.
    """
    # Define orbital parameters
    a = 30e3
    e = 0.01
    i = np.radians(30)
    RAAN = np.radians(40)
    arg_perigee = np.radians(45)
    true_anomaly = np.radians(123)

    # Create orbital mechanics object and calculate satellite position/velocity
    orb_mech = OrbitalMechanics(a, e, i, RAAN, arg_perigee, true_anomaly)
    satellite_position, satellite_velocity = orb_mech.satellite_position_velocity()

    # Example parameters for hydrazine
    N_molecules = 1000
    dt = 1e-2
    total_time = 0.25
    P_tank = 1e6  # Pressure inside the tank in Pascals (e.g., 1 MPa)
    P_space = 1e-3  # Pressure outside the tank in Pascals (near vacuum in space)
    molar_mass_hydrazine = 32.05 / 1000  # Molar mass of hydrazine in kg/mol
    V_ullage = 0.05  # Volume of the ullage space in m³
    mass_gas = 0.5  # kg

    # Gas dispersion simulation
    gas_disp = GasPlumeSimulator(satellite_velocity, N_molecules, dt, total_time, P_tank, P_space, molar_mass_hydrazine, V_ullage, mass_gas)
    positions, max_distance, choke_gas_speed = gas_disp.simulate_gas_dispersion(satellite_position)

    # Print the gas escape velocity at choke
    print(f"Gas velocity at choke: {choke_gas_speed: .2f}m/s")

    # Print the maximum distance reached by any gas molecule
    print(f"The maximum distance of dispersion: {max_distance:.2f}m.")

    # Calculate the width of the plume
    plume_width = gas_disp.calculate_plume_width(positions)
    print(f"The maximum width of plume is: {plume_width:.2f}m")
    
    # Orbit plotting
    theta_orbit = np.linspace(0, 2 * np.pi, 500)
    orbit_positions_eci = []
    for theta in theta_orbit:
        pos_orb, _ = orb_mech.position_orbital_plane(theta)
        R = orb_mech.rotation_matrix()

        pos_eci = R @ pos_orb

        orbit_positions_eci.append(pos_eci)

    orbit_positions_eci = np.array(orbit_positions_eci)

    gas_disp.plot_orbit_and_gas(positions, satellite_position, orbit_positions_eci, N_molecules=1000)

if __name__ == "__main__":
    main()
