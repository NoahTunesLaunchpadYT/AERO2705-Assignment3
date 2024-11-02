from gas_dispersion import gas_plume
from gas_dispersion import absorption_interference as absorption
from gas_dispersion import gas_dispersion
import numpy as np

def random_index(array_length):
    """
    Selects a random index for an array of the given length.

    Parameters:
    -----------
    array_length : int
        The length of the array from which to select a random index.

    Returns:
    --------
    int
        A randomly selected index for an array of the specified length.
    """
    # Generate a random integer from 0 to array_length - 1
    return np.random.randint(0, array_length)

def extract_position_velocity(state_array, time_index) -> np.ndarray:
    """
    Extracts the position and velocity vectors at a specified index from a state array.

    Parameters:
    -----------
    state_array : np.ndarray
        A 6xN array where the first three rows are position components (x, y, z)
        and the last three rows are velocity components (vx, vy, vz).
    time_index : int
        The time step index at which to extract the position and velocity vectors.

    Returns:
    --------
    tuple
        A tuple containing two 1x3 arrays:
        - position vector (x, y, z) at the specified index
        - velocity vector (vx, vy, vz) at the specified index
    """
    # Extract the 1x3 position vector from the first three rows at the given index
    position = state_array[:3, time_index]
    
    # Extract the 1x3 velocity vector from the last three rows at the given index
    velocity = state_array[3:, time_index]
    
    return position, velocity

def extract_position_components(state_array) -> np.ndarray:
    """
    Extracts only the positional components from a state array in N x 3 format.

    Parameters:
    -----------
    state_array : np.ndarray
        A 6xN array where the first three rows are position components (x, y, z)
        and the last three rows are velocity components (vx, vy, vz).

    Returns:
    --------
    np.ndarray
        An N x 3 array where each row is a time step and columns are the x, y, z components.
    """
    # Transpose the position rows (x, y, z) from 3xN to N x 3 format
    return state_array[:3, :].T

class Payload:
    def __init__(self, params, satellite):
        self.satellite = satellite
        
        # Parameters for hydrazine dispersal in space
        self.N_molecules = round(params["N_molecules"]) # Number of molecules to simulate in the gas plume (unitless)
        self.dt = params["dt"] # Time step for the simulation in seconds (0.01 s)
        self.total_time = params["total_time"] # Total simulation time in seconds
        self.P_tank = params["P_tank"] # Pressure inside the hydrazine tank in Pascals (1 MPa, typical tank pressure)
        self.P_space = params["P_space"] # Pressure in the surrounding space in Pascals (near vacuum)
        self.molar_mass_hydrazine = params["molar_mass_hydrazine"] # Molar mass of hydrazine in kg/mol (32.05 g/mol)
        self.V_ullage = params["V_ullage"] # Volume of the ullage space in the tank in cubic meters
        self.mass_gas = params["mass_gas"] # Mass of the hydrazine gas in the tank in kilograms
        
    def simulate_payload_on_path(self, time_arrays: list, solution_arrays: list):
        
        # loop over every other orbit (the target orbits) and simulate payload function there
        print((round(len(solution_arrays) + 1) / 2))

        for i in range(1, round((len(solution_arrays) + 1) / 2)):
            orbit_index = i*2

            orbit = solution_arrays[orbit_index]
            times = time_arrays[orbit_index]

            # Select random point in time of orbit
            time_step_index = random_index( len( times ) )

            satellite_position, satellite_velocity = extract_position_velocity( orbit, time_step_index )
            time = times[time_step_index]

            print(satellite_position, satellite_velocity)

            N_molecules = self.N_molecules  
            dt = self.dt  
            total_time = self.total_time 
            P_tank = self.P_tank  
            P_space = self.P_space  
            molar_mass_hydrazine = self.molar_mass_hydrazine 
            V_ullage = self.V_ullage  
            mass_gas = self.mass_gas  

            # Gas dispersion simulation
            gas_plume_sim = gas_plume.GasPlumeSimulator(satellite_velocity, N_molecules, dt, total_time, P_tank, P_space, molar_mass_hydrazine, V_ullage, mass_gas)
            positions, max_distance, gas_speed = gas_plume_sim.simulate_gas_dispersion(satellite_position)

            # Print the maximum distance reached by any gas molecule
            print(f"\nThe maximum distance at which the gas dispersed is: {max_distance:.2f} meters.")

            # Calculate the width of the plume (meters)
            plume_width = gas_plume_sim.calculate_plume_width(positions)
            print(f"\nThe width of the hydrazine plume is: {plume_width:.2f} meters")

            orbit_positions_eci = extract_position_components( orbit )
            gas_plume_sim.plot_orbit_and_gas(positions, satellite_position, orbit_positions_eci, N_molecules=1000)

            # Define absorption properties for gases
            gases = [
                absorption.GasAbsorptionSimulator(name="Hydrazine", peak_wavelength=9.7, peak_height=0.9),  # Hydrazine properties
                absorption.GasAbsorptionSimulator(name="Inteference A", peak_wavelength=12.0, peak_height=0.3),  # Interfering gas A properties
                absorption.GasAbsorptionSimulator(name="Inteference B", peak_wavelength=10.6, peak_height=0.2)  # Interfering gas B properties
            ]

            # List of standard deviations for absorption simulation (σ values, dimensionless)
            std_devs = [0.05, 0.25, 0.5]

            # Molar absorptivity for hydrazine (in L·mol^{-1}·cm^{-1}), an example value
            molar_absorptivity_hydrazine = 8.1e4

            # Create the transmittance simulation object
            transmitance_sim = absorption.TransmittanceSimulator(gases=gases, std_devs=std_devs)

            # Run the transmittance simulation to generate transmittance data for each standard deviation
            transmitance_sim.simulate_transmittance()

            # Calculate and print concentrations at 3 distances: 1/3, 2/3, and full plume width
            distances = [(plume_width / 1000) * fraction for fraction in (1/3, 2/3, 1)]  # Convert plume width to km
            for distance in distances:
                transmitance_sim.print_concentration_for_distance(molar_absorptivity_hydrazine, distance)

            # Plot the transmittance spectra for all standard deviations
            transmitance_sim.plot_transmittance()

            # For very small length of hydrazine medium such as at the orifice
            transmitance_sim.print_concentration_for_distance(molar_absorptivity_hydrazine, 1e-3)
            concentration = transmitance_sim.calculate_concentration(molar_absorptivity_hydrazine, 1e-3, 0.25)

            # Create and run the diffusion simulation with parameters
            gas_disp_sim = gas_dispersion.DiffusionSimulator(
                50,  # Grid size for diffusion simulation (n by n grid)
                200,  # Number of time steps for diffusion simulation
                1e5,  # Diffusion coefficient for hydrazine (example value in m²/s)
                concentration,  # Concentration at the source
                0.1 # Time step size (s)
            )
            gas_disp_sim.run_simulation()