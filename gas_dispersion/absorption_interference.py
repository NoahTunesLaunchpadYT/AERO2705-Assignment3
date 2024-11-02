import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class GasAbsorptionSimulator:
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
        Constructs all the necessary attributes for the GasAbsorptionSimulator object.
        """
        # Set the name of the gas
        self.name = name
        # Set the wavelength (in micrometers) at which absorption is at maximum
        self.peak_wavelength = peak_wavelength
        # Set the peak absorption value, representing maximum absorption strength
        self.peak_height = peak_height

    def absorption_curve(self, wavelengths, std_dev):
        """
        Generate the absorption curve for the gas.
        """
        # Calculate Gaussian absorption centered at peak_wavelength with std_dev
        return self.peak_height * np.exp(-((wavelengths - self.peak_wavelength) ** 2) / (2 * std_dev ** 2))

class TransmittanceSimulator:
    """
    A class to simulate and plot the transmittance spectrum of gases, including noise.
    """

    def __init__(self, gases, std_devs, wavelength_range=(9, 12), num_points=500, noise_mean=0, noise_std_dev=0.02):
        """
        Constructs all the necessary attributes for the TransmittanceSimulator object.
        """
        # List of GasAbsorptionSimulator objects representing different gases
        self.gases = gases
        # List of standard deviations for absorption curves
        self.std_devs = std_devs
        # Start and end values for wavelength range (in micrometers)
        self.wavelength_start, self.wavelength_end = wavelength_range
        # Number of wavelength points to simulate
        self.num_points = num_points
        # Mean of the Gaussian noise added to transmittance
        self.noise_mean = noise_mean
        # Standard deviation of the noise added to transmittance
        self.noise_std_dev = noise_std_dev
        # Generate array of wavelengths from start to end with num_points points
        self.wavelengths = np.linspace(self.wavelength_start, self.wavelength_end, self.num_points)
        # Dictionary to store simulated transmittance data for each std_dev
        self.transmittance_data = {}
        # Dictionary to store simulated smoothed/filtered transmitance data for each std_dev
        self.smoothed_transmittance_data = {}

    def simulate_transmittance(self):
        """
        Simulate the transmittance spectrum for each standard deviation scenario.
        """
        for std_dev in self.std_devs:
            # Initialize total absorption array with zeros (same size as wavelengths)
            total_absorption = np.zeros_like(self.wavelengths)

            # Sum absorption contributions from all gases for current std_dev
            for gas in self.gases:
                total_absorption += gas.absorption_curve(self.wavelengths, std_dev)

            # Calculate transmittance using Beer-Lambert Law (exp(-absorption))
            transmittance = np.exp(-total_absorption)

            # Generate noise and add to transmittance, clipping values between 0 and 1
            noise = np.random.normal(self.noise_mean, self.noise_std_dev, transmittance.shape)
            noisy_transmittance = np.clip(transmittance + noise, 0, 1)

            # Store noisy transmittance in dictionary with std_dev as key
            self.transmittance_data[std_dev] = noisy_transmittance
        # Return dictionary of transmittance data
        return self.transmittance_data

    def calculate_concentration(self, molar_absorptivity, distance_km, std_dev):
        """
        Calculate the concentration of hydrazine at the peak wavelength using Beer-Lambert Law for a given standard deviation and distance.

        Parameters:
        -----------
        molar_absorptivity : float
            The molar absorptivity (ε) in L·mol^{-1}·cm^{-1}.
        distance_km : float
            The path length in kilometers.
        std_dev : float
            The standard deviation to use in the calculation.

        Returns:
        --------
        float
            Calculated concentration (in pmol/L).
        """
        # Identify hydrazine gas from the list of gases
        hydrazine_gas = next(gas for gas in self.gases if gas.name == "Hydrazine")
        
        # Find the index of the wavelength closest to hydrazine's peak absorption
        peak_wavelength_index = np.argmin(np.abs(self.wavelengths - hydrazine_gas.peak_wavelength))
        
        # Extract transmittance at the peak wavelength for current std_dev
        peak_transmittance = self.transmittance_data[std_dev][peak_wavelength_index]
        
        # Calculate absorbance from transmittance (A = -log10(T))
        absorbance = -np.log10(peak_transmittance)
        
        # Convert distance from km to cm for Beer-Lambert formula
        path_length_cm = distance_km * 1e5
        
        # Calculate concentration using Beer-Lambert Law: A = ε * c * l
        concentration = absorbance / (molar_absorptivity * path_length_cm)
        
        # Return concentration in pmol/L (multiplied by 1e12 to convert mol/L to pmol/L)
        return concentration * 1e12
    
    def apply_savgol_filter(self, window_length=31, polyorder=3):
        """
        Apply Savitzky-Golay filtering to smooth each noisy transmittance spectrum.
        """
        for std_dev, noisy_transmittance in self.transmittance_data.items():
            smoothed_transmittance = savgol_filter(noisy_transmittance, window_length, polyorder)
            self.smoothed_transmittance_data[std_dev] = smoothed_transmittance

    def print_concentration_for_distance(self, molar_absorptivity, distance_km):
        """
        Prints the concentration of hydrazine at the peak wavelength for each standard deviation at a specific distance.

        Parameters:
        -----------
        molar_absorptivity : float
            The molar absorptivity (ε) in L·mol^{-1}·cm^{-1}.
        distance_km : float
            The path length in kilometers.
        """
        # Print header for distance
        print(f"\nAt medium length {distance_km:.4f} km--------------------------------------------------------------")
        for std_dev in self.std_devs:
            # Calculate concentration for each std_dev and print
            concentration = self.calculate_concentration(molar_absorptivity, distance_km, std_dev)
            print(f"Concentration of hydrazine for σ = {std_dev}: {concentration:.4f} pmol/L")

    def plot_transmittance(self):
        """
        Plot the simulated transmittance spectra for all standard deviations.
        """
        # Initialize plot for transmittance spectra
        plt.figure(figsize=(10, 6))

        # Plot transmittance data for each standard deviation
        for std_dev in self.std_devs:
            plt.plot(self.wavelengths, self.transmittance_data[std_dev], label=f'Noisy σ={std_dev}', linestyle='--', alpha=0.5)
            if std_dev in self.smoothed_transmittance_data:
                plt.plot(self.wavelengths, self.smoothed_transmittance_data[std_dev], label=f'Smoothed σ={std_dev}')

        # Mark center wavelength of each gas with a vertical dashed line
        for gas in self.gases:
            plt.axvline(x=gas.peak_wavelength, linestyle='--', label=f'{gas.name} Center ({gas.peak_wavelength} μm)', color='gray')
        
        # Label x and y axes for clarity
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Transmittance')
        
        # Set y-axis limits to display transmittance range
        plt.ylim(0, 1)
        
        # Add grid for better readability
        plt.grid(True)
        
        # Display legend and plot the figure
        plt.legend()
        plt.show()

def main():
    # Gas absorption parameters, including Hydrazine and other interfering gases
    gases = [
        GasAbsorptionSimulator(name="Hydrazine", peak_wavelength=9.7, peak_height=0.9),
        GasAbsorptionSimulator(name="Inteference A", peak_wavelength=11.95, peak_height=0.5),
        GasAbsorptionSimulator(name="Inteference B", peak_wavelength=10.6, peak_height=0.2)
    ]

    # List of standard deviations to simulate
    std_devs = [0.05, 0.25, 0.5]

    # Molar absorptivity (example value in L·mol^{-1}·cm^{-1})
    molar_absorptivity_hydrazine = 8.1e4

    # Create the simulation object
    simulation = TransmittanceSimulator(gases=gases, std_devs=std_devs)

    # Run the simulation
    simulation.simulate_transmittance()

    # Calculate and print concentrations at 3 different distances (1/3, 2/3, and full distance)
    distances = [0.65989 * fraction for fraction in (1/3, 2/3, 1)]
    for distance in distances:
        simulation.print_concentration_for_distance(molar_absorptivity_hydrazine, distance)

    # For any size orifice, concentration of very small length of hydrazine medium
    simulation.print_concentration_for_distance(molar_absorptivity_hydrazine, 1e-3)

    # Apply Savitzky-Golay filter to smooth each transmittance spectrum
    simulation.apply_savgol_filter(window_length=10, polyorder=2)

    # Plot the results with center lines and multiple standard deviations
    simulation.plot_transmittance()

if __name__ == "__main__":
    main()