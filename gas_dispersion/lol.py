from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt

class GasAbsorption:
    """
    A class to represent the absorption characteristics of gases.
    """

    def __init__(self, name, peak_wavelength, peak_height):
        self.name = name
        self.peak_wavelength = peak_wavelength
        self.peak_height = peak_height

    def absorption_curve(self, wavelengths, std_dev):
        return self.peak_height * np.exp(-((wavelengths - self.peak_wavelength) ** 2) / (2 * std_dev ** 2))


class TransmittanceSimulation:
    """
    A class to simulate and plot the transmittance spectrum of gases, including noise.
    """

    def __init__(self, gases, std_devs, wavelength_range=(9, 12), num_points=500, noise_mean=0, noise_std_dev=0.02):
        self.gases = gases
        self.std_devs = std_devs
        self.wavelength_start, self.wavelength_end = wavelength_range
        self.num_points = num_points
        self.noise_mean = noise_mean
        self.noise_std_dev = noise_std_dev
        self.wavelengths = np.linspace(self.wavelength_start, self.wavelength_end, self.num_points)
        self.transmittance_data = {}
        self.smoothed_transmittance_data = {}

    def simulate_transmittance(self):
        for std_dev in self.std_devs:
            total_absorption = np.zeros_like(self.wavelengths)

            for gas in self.gases:
                total_absorption += gas.absorption_curve(self.wavelengths, std_dev)

            transmittance = np.exp(-total_absorption)

            noise = np.random.normal(self.noise_mean, self.noise_std_dev, transmittance.shape)
            noisy_transmittance = np.clip(transmittance + noise, 0, 1)

            self.transmittance_data[std_dev] = noisy_transmittance
        return self.transmittance_data

    def apply_savgol_filter(self, window_length=31, polyorder=3):
        """
        Apply Savitzky-Golay filtering to smooth each noisy transmittance spectrum.
        """
        for std_dev, noisy_transmittance in self.transmittance_data.items():
            smoothed_transmittance = savgol_filter(noisy_transmittance, window_length, polyorder)
            self.smoothed_transmittance_data[std_dev] = smoothed_transmittance

    def plot_transmittance(self):
        """
        Plot the simulated transmittance spectra for all standard deviations, showing both noisy and smoothed versions.
        """
        plt.figure(figsize=(10, 6))

        for std_dev in self.std_devs:
            plt.plot(self.wavelengths, self.transmittance_data[std_dev], label=f'Noisy σ={std_dev}', linestyle='--', alpha=0.5)
            if std_dev in self.smoothed_transmittance_data:
                plt.plot(self.wavelengths, self.smoothed_transmittance_data[std_dev], label=f'Smoothed σ={std_dev}')

        # Mark the center wavelength of each gas with a vertical line
        for gas in self.gases:
            plt.axvline(x=gas.peak_wavelength, linestyle='--', label=f'{gas.name} Peak ({gas.peak_wavelength} μm)', color='gray')

        plt.title('Noisy and Smoothed Transmittance Spectrum with Multiple Gas Absorption Peaks')
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Transmittance')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()


# Gas absorption parameters, including Hydrazine and by-products Ammonia and Nitrous Oxide
gases = [
    GasAbsorption(name="Hydrazine", peak_wavelength=9.7, peak_height=0.9),      # Strong absorption at 9.7 μm
    GasAbsorption(name="Ammonia", peak_wavelength=10.2, peak_height=0.5),       # Moderate absorption at 10.2 μm
    GasAbsorption(name="Nitrous Oxide", peak_wavelength=10.5, peak_height=0.3), # Small absorption at 10.5 μm
]

# List of standard deviations to simulate
std_devs = [0.05, 0.25, 0.5]

# Create the simulation object
simulation = TransmittanceSimulation(gases=gases, std_devs=std_devs)

# Run the simulation
simulation.simulate_transmittance()

# Apply Savitzky-Golay filter to smooth each transmittance spectrum
simulation.apply_savgol_filter(window_length=31, polyorder=3)

# Plot the results with both noisy and smoothed transmittance
simulation.plot_transmittance()