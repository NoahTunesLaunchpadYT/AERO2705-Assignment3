import numpy as np
import matplotlib.pyplot as plt
import constants as CV

class PropulsionSystem:
    def __init__(self, 
                position, 
                velocity, 
                delta_v,
                satellite_mass, 
                tank_mass,
                tank_volume, 
                tank_pressure,
                engine_thermal_properties, 
                propellant_mass, 
                propellant_molar_mass,
                propellant_flow_rate,
                propellant_a_constant,
                propellant_b_constant,
                propellant_density,
                pressurant_moles,
                tank_temperature, 
                specific_impulse, 
                thrust, 
                burn_time,
                satellite_thickness,
                satellite_surface_area):
        """
        Initialize the propulsion system with relevant parameters.
        """
        self.position = position
        self.velocity = velocity
        self.delta_v = [delta_v]

        # Tank and propellant properties
        self.tank_mass = tank_mass
        self.tank_volume = tank_volume
        self.tank_pressure = tank_pressure
        self.tank_temperature = tank_temperature
        self.propellant_mass = propellant_mass
        self.propellant_molar_mass = propellant_molar_mass
        self.propellant_flow_rate = propellant_flow_rate
        self.propellant_vanderwalls_a = propellant_a_constant
        self.propellant_vanderwalls_b = propellant_b_constant
        self.propellant_density = propellant_density

        # Pressurant Properties
        self.pressurant_moles = pressurant_moles

        # Engine properties
        self.engine_thermal_properties = engine_thermal_properties
        self.specific_impulse = specific_impulse
        self.thrust = thrust
        self.burn_time = burn_time

        # Satellite Properties
        self.satellite_thickness = satellite_thickness
        self.satellite_surface_area = satellite_surface_area
        self.satellite_mass = satellite_mass
        # Derived quantities
        self.exhaust_velocity = self.specific_impulse * 9.81  # Effective exhaust velocity (m/s)
        self.total_mass = []
        self.total_mass.append(satellite_mass)

    def calculate_thrust(self, delta_v, burn_time):
        """
        Calculate the thrust required to achieve a specific delta-v over a given burn time.
        """
        # Establishing Parameters
        self.delta_v.append(delta_v)
        delta_v_metres = delta_v * 1e3  # Converting delta-v to m/s
        initial_mass = self.total_mass[-1]  # Mass of satellite

        # Calculating propellant consumed
        propellant_consumed = self.delta_m_over_burn(initial_mass, delta_v_metres)

        # Calculating from mass flow rate
        mass_flow_rate = propellant_consumed / burn_time
        thrust = mass_flow_rate * self.exhaust_velocity
        return thrust

    def delta_m_over_burn(self, initial_mass, delta_v):
        """
        Calculate the propellant user for a burn with a given delta-v
        """
        # Calculate the final mass
        final_mass = initial_mass / np.exp(delta_v / self.exhaust_velocity)

        # Append to array of satellite mass
        self.total_mass.append(final_mass)

        # Calculate the propellant_consumed
        propellant_consumed = initial_mass - final_mass

        self.propellant_mass = self.propellant_mass - propellant_consumed

        print("Propellant mass is: ",self.propellant_mass)

        return propellant_consumed
    

    def thermal_characteristics(self, grid_height=50, grid_width=20, iterations=500):
        """
        Simulates thermal characteristics of the propulsion system in a side-on view 
        of the engine chamber. Generates a heat map using the Laplace equation for 
        steady-state heat diffusion.
        Returns the average temperature in the combustion chamber region.
        """
        # Define material properties
        thermal_conductivity = self.engine_thermal_properties.get('thermal_conductivity', 0.5)  # W/(m·K)
        combustion_temp = self.engine_thermal_properties.get('combustion_temperature', 3500)  # K
        ambient_temp = self.engine_thermal_properties.get('ambient_temperature', 300)  # K

        # Create a grid to simulate the engine chamber from a side-on view
        grid = np.zeros((grid_height, grid_width))
        
        # Initial conditions: combustion chamber at the bottom center is hot
        chamber_width = grid_width // 3  # Define chamber width relative to the engine
        chamber_start = (grid_width - chamber_width) // 2  # Starting index for combustion chamber
        grid[-1, chamber_start:chamber_start + chamber_width] = combustion_temp

        # Simulate heat diffusion using the Laplace equation
        for _ in range(iterations):
            new_grid = grid.copy()
            for i in range(1, grid_height - 1):
                for j in range(1, grid_width - 1):
                    new_grid[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])

            # Enforce boundary conditions (ambient temperature)
            new_grid[0, :] = ambient_temp  # Top is ambient temp
            new_grid[:, 0] = ambient_temp  # Left side is ambient temp
            new_grid[:, -1] = ambient_temp  # Right side is ambient temp
            new_grid[-1, :chamber_start] = ambient_temp  # Bottom left side is ambient temp
            new_grid[-1, chamber_start + chamber_width:] = ambient_temp  # Bottom right side is ambient temp

            grid = new_grid

        # Create a heatmap of the engine chamber from a side-on view (optional visualization)
        plt.imshow(grid, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Temperature (K)')
        plt.title('Heat Diffusion in Engine Chamber (Side-On View)')
        plt.xlabel('Horizontal Position in Engine Chamber')
        plt.ylabel('Vertical Position in Engine Chamber')
        plt.show()

        # Calculate average temperature in the combustion chamber region
        chamber_region = grid[-1, chamber_start:chamber_start + chamber_width]
        chamber_temperature = np.mean(chamber_region)

        return chamber_temperature

    def calculate_efficiency(self, delta_v):
        """
        Calculate the efficiency of the burn using tank pressure, temperature, and delta-v.
        Incorporates the chamber temperature from thermal characteristics after a delta-v burn.
        """

        # Step 1: Use the thermal characteristics function to get the chamber temperature
        chamber_temperature = self.thermal_characteristics()  # Get steady-state chamber temperature

        # Step 2: Exhaust velocity calculation based on chamber temperature
        gamma = self.engine_thermal_properties.get('gamma', 1.4)  # Specific heat ratio
        gas_constant = 8.314 / self.propellant_molar_mass  # Specific gas constant (J/kg·K)
        
        v_exhaust = np.sqrt(2 * gamma / (gamma - 1) * gas_constant * chamber_temperature)
        self.exhaust_velocity = v_exhaust  # Update exhaust velocity based on new conditions

        # Step 3: Calculate propellant consumption for the given delta-v
        delta_v_metres = delta_v * 1e3  # Convert delta-v to m/s
        initial_mass = self.total_mass[-1]
        final_mass = initial_mass / np.exp(delta_v_metres / v_exhaust)
        propellant_consumed = initial_mass - final_mass

        # Step 4: Calculate Kinetic Energy of Exhaust
        kinetic_energy_exhaust = 0.5 * propellant_consumed * v_exhaust**2

        # Step 5: Calculate Chemical Energy Released
        specific_heat_combustion = self.engine_thermal_properties.get('specific_heat_combustion', 4200)  # J/kg·K
        delta_temp_combustion = self.engine_thermal_properties.get('delta_temp_combustion', 3000)  # K

        # Total chemical energy released
        chemical_energy_released = propellant_consumed * specific_heat_combustion * delta_temp_combustion

        # Step 6: Efficiency Calculation
        if chemical_energy_released > 0:
            efficiency = (kinetic_energy_exhaust / chemical_energy_released)/10
        else:
            efficiency = 0.0

        return efficiency
        
    def simulate_fourier_diffusion(self, time_step=1, total_time=100):
        """
        Simulate heat diffusion from the engine chamber to the rest of the satellite over time,
        updating the satellite's internal temperature and tank temperature accordingly.
        Also generates a plot to visualize the temperature changes over time.

        Parameters:
        - time_step: Time step for each iteration in seconds.
        - total_time: Total time to simulate in seconds.

        Returns:
        - Final satellite temperature after the simulation.
        """
        # Define material and physical properties for the heat transfer simulation
        thermal_conductivity_satellite = self.engine_thermal_properties.get('thermal_conductivity_satellite', 0.05)  # W/m·K
        specific_heat_satellite = self.engine_thermal_properties.get('specific_heat_satellite', 900)  # J/kg·K (aluminum)
        satellite_surface_area = self.satellite_surface_area
        satellite_thickness = self.satellite_thickness
        combustion_temp = self.engine_thermal_properties.get('combustion_temperature', 3500)  # K

        # Initial temperature of the satellite (assume tank and satellite start at the same temperature)
        satellite_temperature = self.tank_temperature

        # Lists to store time and temperature values for plotting
        time_values = []
        temperature_values = []

        # Time evolution of heat transfer from the chamber to the satellite
        for t in range(0, total_time, time_step):
            # Apply Fourier's Law to calculate heat flow from the combustion chamber to the satellite body
            heat_flux = (thermal_conductivity_satellite * satellite_surface_area * (combustion_temp - satellite_temperature)) / satellite_thickness

            # Update satellite temperature using the heat flux and specific heat of the satellite
            temperature_change = (heat_flux * time_step) / (self.satellite_mass * specific_heat_satellite)
            satellite_temperature += temperature_change
            
            # Update the tank temperature since it's in thermal equilibrium with the satellite
            self.tank_temperature = satellite_temperature

            # Store the time and temperature for plotting
            time_values.append(t)
            temperature_values.append(satellite_temperature)

        # Plot the temperature change over time
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, temperature_values, label="Satellite Temperature", color='r')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (K)')
        plt.title('Satellite Temperature Over Time Due to Heat Diffusion')
        plt.grid(True)
        plt.legend()
        plt.show()

        return satellite_temperature

    
    def update_tank_pressure(self):
        # Total tank volume (constant)
        total_tank_volume = self.tank_volume

        # Calculate the current volume of liquid propellant
        propellant_volume = self.propellant_mass / self.propellant_density
        
        # The remaining volume is filled with the pressurant gas
        gas_volume = total_tank_volume - propellant_volume

        # Ensure we don't get a negative or zero volume for the gas
        if gas_volume <= 0:
            raise ValueError("No more room for gas; tank is fully filled with liquid.")

        # Update the pressure of the gas using the Ideal Gas Law
        current_gas_moles = self.pressurant_moles

        self.current_gas_pressure = (current_gas_moles * CV.R * self.tank_temperature) / gas_volume
    
        return self.current_gas_pressure
    
    #----------------------------------------Ionic Thuster------------------------------------------------------

    def simulate_heat_distribution(self, grid_height=50, grid_width=20, iterations=1000):
        """
        Simulates the heat distribution inside a Hall thruster chamber using the Laplace equation.
        The chamber is modeled as a 2D grid, with heat generated primarily in the plasma discharge area.
        """
        # Initialize grid to ambient temperature
        grid = np.full((grid_height, grid_width), self.engine_thermal_properties.get('ambient_temperature', 300))

        # Heat sources:
        plasma_temp = self.engine_thermal_properties.get('plasma_temperature', 10000)  # Plasma discharge temperature
        coil_temp = self.engine_thermal_properties.get('coil_temperature', 400)  # Magnetic coil temperature

        # Define the regions of the grid where heat is applied:
        # Plasma discharge zone (center-bottom of the grid)
        plasma_zone_height = grid_height // 5
        plasma_zone_width = grid_width // 3
        plasma_start_x = (grid_width - plasma_zone_width) // 2
        plasma_end_x = plasma_start_x + plasma_zone_width
        plasma_start_y = grid_height - plasma_zone_height
        
        # Set initial temperature of plasma discharge zone
        grid[plasma_start_y:, plasma_start_x:plasma_end_x] = plasma_temp

        # Magnetic coils: assumed to be around the edges of the chamber
        coil_zone_width = grid_width // 5
        coil_zone_height = grid_height // 10

        # Left and right coil zones
        grid[-coil_zone_height:, :coil_zone_width] = coil_temp
        grid[-coil_zone_height:, -coil_zone_width:] = coil_temp

        # Apply the Laplace equation to simulate heat diffusion
        for _ in range(iterations):
            new_grid = grid.copy()
            for i in range(1, grid_height - 1):
                for j in range(1, grid_width - 1):
                    # Update the temperature at each point based on the average of its neighbors
                    new_grid[i, j] = 0.25 * (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1])
            
            # Boundary conditions: The edges are set to ambient temperature
            new_grid[0, :] = self.engine_thermal_properties.get('ambient_temperature', 300)  # Top boundary
            new_grid[:, 0] = self.engine_thermal_properties.get('ambient_temperature', 300)  # Left boundary
            new_grid[:, -1] = self.engine_thermal_properties.get('ambient_temperature', 300)  # Right boundary
            new_grid[-1, :] = self.engine_thermal_properties.get('ambient_temperature', 300)  # Bottom boundary, excluding plasma zone
            
            # Preserve the heat source temperatures in the plasma and coil zones
            new_grid[plasma_start_y:, plasma_start_x:plasma_end_x] = plasma_temp
            new_grid[-coil_zone_height:, :coil_zone_width] = coil_temp
            new_grid[-coil_zone_height:, -coil_zone_width:] = coil_temp
            
            # Update the grid for the next iteration
            grid = new_grid

        # Visualize the heat distribution
        plt.imshow(grid, cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Temperature (K)')
        plt.title('Heat Distribution in Hall Thruster Chamber')
        plt.xlabel('Horizontal Position in Thruster Chamber')
        plt.ylabel('Vertical Position in Thruster Chamber')
        plt.show()

        return grid


#---------------------Valdating Successful Operation---------------------------------------------------------------

def test_propulsion_subsystem():
    # Example use of the thermal characteristics
    thermal_properties = {
        'thermal_conductivity': 0.7,  # W/m·K
        'combustion_temperature': 3500,  # K
        'ambient_temperature': 300  # K
    }

    propulsion_system = PropulsionSystem(
        position=(10000, 4000, 600), 
        velocity=(3, 2, 1), 
        delta_v=4000,
        satellite_mass=1000, 
        tank_mass=500,
        tank_volume = 10000,
        tank_pressure=1e6, 
        engine_thermal_properties=thermal_properties, 
        propellant_mass=100000,
        propellant_molar_mass = 1e-5,
        propellant_flow_rate=5,
        propellant_a_constant=10,
        propellant_b_constant=20,
        propellant_density=100,
        pressurant_moles=20,
        tank_temperature=30, 
        specific_impulse=300, 
        thrust=15000, 
        burn_time=300,
        satellite_thickness=0.01,
        satellite_surface_area=5
    )

    # Example delta_v_array
    delta_v_array_1 = [3, 5.5, 10, 8.2, 3.2]
    delta_v_array_2 = [5, 1, 8, 2, 4, 2]
    delta_v_array = delta_v_array_1
    tank_pressure_array = []

    # Loop through delta_v_array to calculate thrust and update tank pressure
    for i in range(len(delta_v_array)):
        propulsion_system.calculate_thrust(delta_v_array[i], 3000)  # Assuming the second argument is mass or other constant
        propulsion_system.update_tank_pressure()  # Update the tank pressure based on the thrust
        tank_pressure_array.append(propulsion_system.current_gas_pressure)  # Store the current tank pressure

    # Plotting the results
    delta_v_array_accumulated = []
    for i in range(len(delta_v_array)):
        if i == 0:
            delta_v_array_accumulated.append(delta_v_array[i]) 
        else:
            delta_v_array_accumulated.append(delta_v_array[i] + delta_v_array_accumulated[i-1])
    print(delta_v_array_accumulated)
    plt.plot(delta_v_array_accumulated, tank_pressure_array, marker='o', linestyle='-', color='b', label='Tank Pressure')
    plt.xlabel('Delta V (km/s)')
    plt.ylabel('Tank Pressure (Pa)')
    plt.title('Delta V vs. Tank Pressure')
    plt.grid(True)
    plt.legend()
    plt.show()


    # Generate the thermal heat map for the side-on view of the engine chamber
    propulsion_system.thermal_characteristics(grid_height=50, grid_width=20, iterations=1000)

    # Calculate efficiency after a delta-v burn
    efficiency = propulsion_system.calculate_efficiency(delta_v=3)  # Delta-v in km/s
    print(f"Efficiency: {efficiency:.4f}")

    propulsion_system.simulate_heat_distribution(grid_height=50, grid_width=20, iterations=1000)

    propulsion_system.simulate_fourier_diffusion(total_time=2000)

def main():
    test_propulsion_subsystem()

if __name__ == "__main__":
    main()