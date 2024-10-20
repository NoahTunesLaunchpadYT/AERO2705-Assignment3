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

class GasDispersion:
    """
    A class to simulate the dispersion of gas molecules in space.

    Attributes:
    -----------
    satellite_velocity : np.ndarray
        Velocity vector of the satellite.
    N_molecules : int
        Number of gas molecules to simulate.
    dt : float
        Time step size for the simulation.
    total_time : float
        Total time over which the gas dispersion is simulated.
    P_tank : float
        Pressure inside the fuel tank (in Pascals).
    P_space : float
        Pressure outside the fuel tank (vacuum in space, in Pascals).
    molar_mass : float
        Molar mass of the gas (in kg/mol).
    V_ullage : float
        Volume of the ullage space in the tank (in cubic meters).
    mass_gas : float
        Mass of the gas (in kilograms).

    Methods:
    --------
    calculate_temperature_from_ullage(P, V_ullage, n_moles):
        Calculate the temperature of the gas in the ullage using the ideal gas law.
    calculate_moles_from_mass(mass, molar_mass):
        Calculate the number of moles of gas from its mass.
    calculate_choked_flow_speed(P1, P2, T, molar_mass, gamma):
        Calculate the speed of the gas using the choked flow model.
    biased_gas_velocities():
        Generate the biased initial velocities for gas molecules.
    simulate_gas_dispersion(satellite_position):
        Simulate the time evolution of gas dispersion.
    plot_orbit_and_gas(positions, satellite_position, orbit_positions_eci, N_molecules):
        Plot the gas dispersion and satellite orbit in 3D.
    """

    def __init__(self, satellite_velocity: np.ndarray, N_molecules: int, dt: float, total_time: float,
                 P_tank: float, P_space: float, molar_mass: float, V_ullage: float, mass_gas: float) -> None:
        """
        Initialize the gas dispersion parameters and calculate initial values.

        Parameters:
        -----------
        satellite_velocity : np.ndarray
            Velocity vector of the satellite.
        N_molecules : int
            Number of gas molecules to simulate.
        dt : float
            Time step size for the simulation.
        total_time : float
            Total time over which the gas dispersion is simulated.
        P_tank : float
            Pressure inside the fuel tank (in Pascals).
        P_space : float
            Pressure outside the fuel tank (vacuum in space, in Pascals).
        molar_mass : float
            Molar mass of the gas (in kg/mol).
        V_ullage : float
            Volume of the ullage space in the tank (in cubic meters).
        mass_gas : float
            Mass of the gas (in kilograms).
        """
        self.gamma = 1.085  # Heat capacity ratio
        self.n_moles: float = self.calculate_moles_from_mass(mass_gas, molar_mass)
        self.temp_in_tank: float = self.calculate_temperature_from_ullage(P_tank, V_ullage, self.n_moles)
        self.v_gas_direction: np.ndarray = -satellite_velocity / np.linalg.norm(satellite_velocity)
        self.gas_speed: float = self.calculate_choked_flow_speed(P_tank, P_space, self.temp_in_tank, molar_mass, self.gamma)
        self.N_molecules: int = N_molecules
        self.dt: float = dt
        self.total_time: float = total_time

    def calculate_temperature_from_ullage(self, P: float, V_ullage: float, n_moles: float) -> float:
        """
        Calculate the temperature of the gas in the ullage using the ideal gas law.

        Parameters:
        -----------
        P : float
            Pressure inside the tank (in Pascals).
        V_ullage : float
            Volume of the ullage space in cubic meters.
        n_moles : float
            Number of moles of gas in the tank.

        Returns:
        --------
        float
            Temperature of the gas in Kelvin.
        """
        R: float = 8.314  # Universal gas constant in J/(mol·K)
        return (P * V_ullage) / (n_moles * R)

    def calculate_moles_from_mass(self, mass: float, molar_mass: float) -> float:
        """
        Calculate the number of moles of gas from its mass.

        Parameters:
        -----------
        mass : float
            Mass of the gas (in kilograms).
        molar_mass : float
            Molar mass of the gas (in kg/mol).

        Returns:
        --------
        float
            Number of moles of gas.
        """
        return mass / molar_mass

    def calculate_choked_flow_speed(self, P1: float, P2: float, T: float, molar_mass: float, gamma: float) -> float:
        """
        Calculate gas speed based on pressure difference using the choked flow model.

        Parameters:
        -----------
        P1 : float
            Pressure inside the tank (in Pascals).
        P2 : float
            Pressure outside the tank (vacuum in Pascals).
        T : float
            Temperature of the gas (in Kelvin).
        molar_mass : float
            Molar mass of the gas (in kg/mol).
        gamma : float
            Heat capacity ratio (adiabatic index).

        Returns:
        --------
        float
            Gas speed (in meters per second).
        """
        R: float = 8.314  # Universal gas constant in J/(mol K)
        critical_pressure_ratio: float = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
        pressure_ratio: float = P2 / P1

        if pressure_ratio <= critical_pressure_ratio:
            v_gas: float = np.sqrt((2 * gamma) / (gamma - 1) * (R * T / molar_mass) * (1 - pressure_ratio ** ((gamma - 1) / gamma)))
        else:
            v_gas = np.sqrt(2 * (P1 - P2) / molar_mass)

        return v_gas

    def biased_gas_velocities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate initial velocities for gas molecules biased in the opposite direction to satellite velocity.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays for velocities in the x, y, and z directions for all gas molecules.
        """
        vx: np.ndarray = np.random.normal(loc=self.v_gas_direction[0], scale=0.25, size=self.N_molecules) * self.gas_speed
        vy: np.ndarray = np.random.normal(loc=self.v_gas_direction[1], scale=0.25, size=self.N_molecules) * self.gas_speed
        vz: np.ndarray = np.random.normal(loc=self.v_gas_direction[2], scale=0.25, size=self.N_molecules) * self.gas_speed

        return vx, vy, vz

    def simulate_gas_dispersion(self, satellite_position: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Simulate the time evolution of gas dispersion.

        Parameters:
        -----------
        satellite_position : np.ndarray
            The initial position of the satellite in the ECI frame (in meters).

        Returns:
        --------
        tuple[np.ndarray, float]
            Array of gas positions over time and the maximum distance reached by the gas.
        """
        vx, vy, vz = self.biased_gas_velocities()

        x: np.ndarray = np.ones(self.N_molecules) * satellite_position[0]
        y: np.ndarray = np.ones(self.N_molecules) * satellite_position[1]
        z: np.ndarray = np.ones(self.N_molecules) * satellite_position[2]

        time_steps: int = int(self.total_time / self.dt)
        positions: np.ndarray = np.zeros((time_steps, self.N_molecules, 3))
        max_distance: float = 0.0

        for t in range(time_steps):
            x += vx * self.dt
            y += vy * self.dt
            z += vz * self.dt

            distances: np.ndarray = np.sqrt((x - satellite_position[0])**2 + (y - satellite_position[1])**2 + (z - satellite_position[2])**2)
            max_distance = max(max_distance, np.max(distances))
            positions[t] = np.stack((x, y, z), axis=1)

        return positions, max_distance
    
    def calculate_plume_width(self, positions: np.ndarray) -> float:
        """
        Calculate the wideness of the widest part of the hydrazine plume.

        Parameters:
        -----------
        positions : np.ndarray
            Array of gas positions over time (x, y, z).

        Returns:
        --------
        float
            The maximum width of the hydrazine plume.
        """
        # Gas direction vector
        gas_direction = self.v_gas_direction
        
        # Get the final positions of the gas molecules (at the last time step)
        final_positions = positions[-1]
        
        # Calculate projections onto the plane perpendicular to the gas flow direction
        projected_positions = []
        for position in final_positions:
            projection = position - np.dot(position, gas_direction) * gas_direction
            projected_positions.append(projection)
        projected_positions = np.array(projected_positions)

        # Calculate distances between all pairs of projected points
        max_distance = 0.0
        for i in range(len(projected_positions)):
            for j in range(i+1, len(projected_positions)):
                distance = np.linalg.norm(projected_positions[i] - projected_positions[j])
                if distance > max_distance:
                    max_distance = distance

        return max_distance

    def plot_orbit_and_gas(self, positions: np.ndarray, satellite_position: np.ndarray,
                           orbit_positions_eci: np.ndarray, N_molecules: int) -> None:
        """
        Plot the gas dispersion and satellite orbit in 3D.

        Parameters:
        -----------
        positions : np.ndarray
            The array of gas positions over time (x, y, z).
        satellite_position : np.ndarray
            The satellite's position in the ECI frame (in meters).
        orbit_positions_eci : np.ndarray
            Array of satellite positions along its orbit in the ECI frame.
        N_molecules : int
            Number of gas molecules simulated.

        Returns:
        --------
        None
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(N_molecules):
            ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], alpha=0.5)

        ax.plot(orbit_positions_eci[:, 0], orbit_positions_eci[:, 1], orbit_positions_eci[:, 2], 'b-', label='Satellite Orbit (ECI)')
        ax.scatter([satellite_position[0]], [satellite_position[1]], [satellite_position[2]], color='red', s=100, label='Satellite', marker='o')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.legend()

        plt.show()

def test_gas_dispersion():
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
    total_time = 2
    P_tank = 1e6  # Pressure inside the tank in Pascals (e.g., 1 MPa)
    P_space = 1e-3  # Pressure outside the tank in Pascals (near vacuum in space)
    molar_mass_hydrazine = 32.05 / 1000  # Molar mass of hydrazine in kg/mol
    V_ullage = 0.05  # Volume of the ullage space in m³
    mass_gas = 0.5  # kg

    # Gas dispersion simulation
    gas_disp = GasDispersion(satellite_velocity, N_molecules, dt, total_time, P_tank, P_space, molar_mass_hydrazine, V_ullage, mass_gas)
    positions, max_distance = gas_disp.simulate_gas_dispersion(satellite_position)

    # Print the maximum distance reached by any gas molecule
    print(f"The maximum distance at which the gas dispersed is: {max_distance:.2f} meters.")

    # Calculate the width of the plume
    plume_width = gas_disp.calculate_plume_width(positions)
    print(f"The width of the hydrazine plume is: {plume_width:.2f} meters")
    
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

def main():
    test_gas_dispersion()

if __name__ == "__main__":
    main()
