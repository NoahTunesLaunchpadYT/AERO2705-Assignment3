import numpy as np
import matplotlib.pyplot as plt

class Power:
    def __init__(self, duration, panel_area, sunlight_intensity, efficiency, eclipse_periods, battery_capacity,
                 initial_charge, base_power, peak_power_times, peak_duration):
        self.duration = duration
        self.panel_area = panel_area
        self.sunlight_intensity = sunlight_intensity
        self.efficiency = efficiency
        self.eclipse_periods = eclipse_periods
        self.battery_capacity = battery_capacity
        self.initial_charge = initial_charge
        self.base_power = base_power
        self.peak_power_times = peak_power_times
        self.peak_duration = peak_duration

    def simulate_power_consumption(self):
        time_points = np.arange(0, self.duration, 1)
        power_consumption = np.full(len(time_points), self.base_power)
        for peak_time in self.peak_power_times:
            peak_indices = (time_points >= peak_time) & (time_points < peak_time + self.peak_duration)
            power_consumption[peak_indices] = 10196.8
        return power_consumption, time_points

    def simulate_battery_charge(self, power_generated, power_consumed):
        battery_charge = [self.initial_charge]
        for gen, cons in zip(power_generated, power_consumed):
            new_charge = battery_charge[-1] + (gen - cons) / 3600  # Convert W to W·h
            new_charge = min(max(new_charge, 0), self.battery_capacity)
            battery_charge.append(new_charge)
        return battery_charge

    def calculate_battery_soc(self, battery_charge):
        return [(charge / self.battery_capacity) * 100 for charge in battery_charge]

    def simulate_solar_power(self):
        power_generated = []
        time_points = np.arange(0, self.duration, 1)
        for t in time_points:
            in_eclipse = any(start <= t < end for start, end in self.eclipse_periods)
            power = 0 if in_eclipse else self.sunlight_intensity * self.panel_area * self.efficiency
            power_generated.append(power)
        return np.array(power_generated), time_points

    def run_simulation(self):
        power_generated, time_points = self.simulate_solar_power()
        power_consumed, _ = self.simulate_power_consumption()
        battery_charge = self.simulate_battery_charge(power_generated, power_consumed)
        soc = self.calculate_battery_soc(battery_charge)

        # Plotting all results in one figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Power Consumption Plot
        axs[0].plot(time_points, power_consumed, color="blue", label="Power Consumption (W)")
        axs[0].set_ylabel("Power (W)")
        axs[0].set_title("Power Consumption during Station Keeping and Proximity Maneuvers")
        axs[0].grid(True)
        axs[0].legend()

        # Solar Power Generation Plot
        axs[1].plot(time_points, power_generated, color="orange", label="Solar Power Generation (W)")
        axs[1].set_title("Solar Power Generation Throughout Orbit")
        axs[1].set_ylabel("Power (W)")
        axs[1].grid(True)
        axs[1].legend()

        # Battery Charge Plot
        axs[2].plot(battery_charge, color="green", label="Battery Charge (W·h)")
        axs[2].set_ylabel("Battery Charge (W·h)")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()

#----------------------------Example Usage---------------------------------------------------

def test_power_subsystem():
    # Define simulation parameters
    duration = 10000  # 10,000 seconds
    panel_area = 30  # Area of solar panels in m²
    sunlight_intensity = 1361  # W/m² (solar constant)
    efficiency = 0.312  # Solar panel efficiency
    eclipse_periods = [(3000, 4000), (7000, 8000)]  # Eclipse periods
    battery_capacity = 5400  # Battery capacity in W·h
    initial_charge = 5400  # Initial battery charge in W·h
    base_power = 300  # Baseline power consumption in W
    peak_power_times = [2000, 6000]  # Peak consumption start times
    peak_duration = 500  # Duration of peak power periods

    # Create an instance of Power and run the simulation
    power_system = Power(duration, panel_area, sunlight_intensity, efficiency, eclipse_periods,
                        battery_capacity, initial_charge, base_power, peak_power_times, peak_duration)
    power_system.run_simulation()


    return None