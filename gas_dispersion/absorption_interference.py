import numpy as np
import matplotlib.pyplot as plt

class GasAbsorption:
    """
    A class to represent the absorption characteristics of gases.

    Attributes:
    -----------
    name : str
        The name of the gas.
    peak_wavelength : float
        The wavelength (in micrometers) where the gas absorbs maximally.
    peak_height : float
        The peak absorption value at the center wavelength.
    """

    def __init__(self, name, peak_wavelength, peak_height):
        """
        Constructs all the necessary attributes for the GasAbsorption object.
        """
        self.name = name
        self.peak_wavelength = peak_wavelength
        self.peak_height = peak_height

    def absorption_curve(self, wavelengths, std_dev):
        """
        Generate the absorption curve for the gas.
        """
        return self.peak_height * np.exp(-((wavelengths - self.peak_wavelength) ** 2) / (2 * std_dev ** 2))


class TransmittanceSimulation:
    """
    A class to simulate and plot the transmittance spectrum of gases, including noise.
    """

    def __init__(self, gases, std_devs, wavelength_range=(9, 12), num_points=500, noise_mean=0, noise_std_dev=0.02):
        """
        Constructs all the necessary attributes for the TransmittanceSimulation object.
        """
        self.gases = gases
        self.std_devs = std_devs
        self.wavelength_start, self.wavelength_end = wavelength_range
        self.num_points = num_points
        self.noise_mean = noise_mean
        self.noise_std_dev = noise_std_dev
        self.wavelengths = np.linspace(self.wavelength_start, self.wavelength_end, self.num_points)
        self.transmittance_data = {}

    def simulate_transmittance(self):
        """
        Simulate the transmittance spectrum for each standard deviation scenario.
        """
        for std_dev in self.std_devs:
            total_absorption = np.zeros_like(self.wavelengths)

            # Sum absorption contributions from all gases
            for gas in self.gases:
                total_absorption += gas.absorption_curve(self.wavelengths, std_dev)

            # Calculate transmittance using Beer-Lambert Law
            transmittance = np.exp(-total_absorption)

            # Add noise
            noise = np.random.normal(self.noise_mean, self.noise_std_dev, transmittance.shape)
            noisy_transmittance = np.clip(transmittance + noise, 0, 1)

            # Store transmittance for each std_dev
            self.transmittance_data[std_dev] = noisy_transmittance
        return self.transmittance_data

    def calculate_concentrations(self, molar_absorptivity, distance_km):
        """
        Calculate the concentration of hydrazine at the peak wavelength using Beer-Lambert Law for all standard deviations.
        
        Parameters:
        -----------
        molar_absorptivity : float
            The molar absorptivity (ε) in L·mol^{-1}·cm^{-1}.
        distance_km : float
            The path length in kilometers.

        Returns:
        --------
        dict
            A dictionary with standard deviations as keys and calculated concentrations (in mol/L) as values.
        """
        hydrazine_gas = next(gas for gas in self.gases if gas.name == "Hydrazine")
        peak_wavelength_index = np.argmin(np.abs(self.wavelengths - hydrazine_gas.peak_wavelength))
        
        concentrations = {}

        for std_dev in self.std_devs:
            # Find the transmittance at the peak wavelength for the current std_dev
            peak_transmittance = self.transmittance_data[std_dev][peak_wavelength_index]
            
            # Calculate absorbance from transmittance
            absorbance = -np.log10(peak_transmittance)
            
            # Convert distance from km to cm (Beer-Lambert uses cm)
            path_length_cm = distance_km * 1e5
            
            # Calculate concentration using Beer-Lambert Law: A = ε * c * l
            concentration = absorbance / (molar_absorptivity * path_length_cm)
            
            # Store concentration in dictionary
            concentrations[std_dev] = concentration

        print(concentrations)
        
        return concentrations

    def plot_transmittance(self):
        """
        Plot the simulated transmittance spectra for all standard deviations.
        """
        plt.figure(figsize=(10, 6))

        for std_dev in self.std_devs:
            plt.plot(self.wavelengths, self.transmittance_data[std_dev], label=f'σ = {std_dev}')

        # Mark the center wavelength of each gas with a vertical line
        for gas in self.gases:
            plt.axvline(x=gas.peak_wavelength, linestyle='--', label=f'{gas.name} Center ({gas.peak_wavelength} μm)', color='gray')
        
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Transmittance')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

def test_absorption_interference():
    # Gas absorption parameters, including Hydrazine and other interfering gases
    gases = [
        GasAbsorption(name="Hydrazine", peak_wavelength=9.7, peak_height=0.9),
        GasAbsorption(name="Inteference A", peak_wavelength=10.1, peak_height=0.3),
        GasAbsorption(name="Inteference B", peak_wavelength=10.6, peak_height=0.2)
    ]

    # List of standard deviations to simulate
    std_devs = [0.05, 0.25, 0.5]

    # Molar absorptivity (example value in L·mol^{-1}·cm^{-1})
    molar_absorptivity_hydrazine = 8.1e4

    # Create the simulation object
    simulation = TransmittanceSimulation(gases=gases, std_devs=std_devs)

    # Run the simulation
    simulation.simulate_transmittance()

    # Calculate the concentration of hydrazine at a distance of 2.7 km for all standard deviations
    concentrations_hydrazine = simulation.calculate_concentrations(molar_absorptivity_hydrazine, distance_km=2.7)

    # Print the concentrations for each standard deviation
    for std_dev, concentration in concentrations_hydrazine.items():
        print(f"Concentration of hydrazine for σ = {std_dev}: {concentration:.6f} mol/L")

    # Plot the results with center lines and multiple standard deviations
    simulation.plot_transmittance()

def main():
    test_absorption_interference()

if __name__ == "__main__":
    main()
    