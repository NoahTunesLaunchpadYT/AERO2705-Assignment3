import numpy as np
import matplotlib.pyplot as plt

class ThermalControl:
    def __init__(self, Kp, Ki, Kd, set_point_min, set_point_max, desired_internal_temp, thermal_resistance, thermal_capacity):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point_min = set_point_min
        self.set_point_max = set_point_max
        self.desired_internal_temp = desired_internal_temp
        self.thermal_resistance = thermal_resistance
        self.thermal_capacity = thermal_capacity
        self.integral = 0
        self.previous_error = 0

    def satellite_external_temperature(self, true_anomaly, eclipse_start, eclipse_end, T_sun=102, T_shadow=-150, tau=20):
        if eclipse_start <= true_anomaly <= eclipse_end:
            return (T_shadow + (T_sun - T_shadow) * np.exp(-(true_anomaly - eclipse_start) / tau)) / 2
        else:
            return (T_sun - (T_sun - T_shadow) * np.exp(-(true_anomaly - eclipse_end) / tau)) / 2

    def heating_power_consumption(self, internal_temp, passive_internal_temp, delta_t=1):
        error = self.desired_internal_temp - passive_internal_temp
        proportional = self.Kp * error
        self.integral += error * delta_t
        self.integral = max(min(self.integral, 10), -10)  # Limit integral windup
        integral_term = self.Ki * self.integral
        derivative = (error - self.previous_error) / delta_t
        derivative_term = self.Kd * derivative
        self.previous_error = error
        power = proportional + integral_term + derivative_term
        return max(0, min(power, 50))  # Limit power between 0 and 50 W

    def simulate_internal_temperature(self, eclipse_start=110, eclipse_end=250, T_sun=102, T_shadow=-150, tau=20, delta_t=1):
        anomalies = np.linspace(0, 360, 360)
        external_temps = [self.satellite_external_temperature(ta, eclipse_start, eclipse_end, T_sun, T_shadow, tau) for ta in anomalies]
        passive_internal_temp = self.desired_internal_temp
        controlled_internal_temp = self.desired_internal_temp
        passive_internal_temps = []
        controlled_internal_temps = []
        power_consumptions = []

        for external_temp in external_temps:
            heat_transfer = (external_temp - passive_internal_temp) / self.thermal_resistance
            passive_internal_temp += (heat_transfer * delta_t) / self.thermal_capacity
            passive_internal_temps.append(passive_internal_temp)

            power = self.heating_power_consumption(controlled_internal_temp, passive_internal_temp, delta_t)
            power_consumptions.append(power)

            temp_difference = passive_internal_temp - controlled_internal_temp
            adjustment = min(abs(temp_difference), power * delta_t / self.thermal_capacity)

            if temp_difference > 0:
                controlled_internal_temp += adjustment
            elif temp_difference < 0:
                controlled_internal_temp -= adjustment

            controlled_internal_temp = max(self.set_point_min, min(controlled_internal_temp, self.set_point_max))
            controlled_internal_temps.append(controlled_internal_temp)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(anomalies, external_temps, 'r-', label="External Temperature (°C)")
        ax1.plot(anomalies, passive_internal_temps, 'b-', label="Passive Internal Temperature (°C)")
        ax1.plot(anomalies, controlled_internal_temps, 'g-', label="Controlled Internal Temperature (°C)")
        ax1.set_xlabel("True Anomaly (degrees)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_ylim(-100, 150)
        ax1.axvspan(eclipse_start, eclipse_end, color="gray", alpha=0.3, label="Eclipse Period")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(anomalies, power_consumptions, 'y--', label="Heating/Cooling Power Consumption (W)")
        ax2.set_ylabel("Power Consumption (W)", color='y')
        ax2.set_ylim(0, 120)
        ax2.legend(loc="upper right")

        plt.title("Satellite External and Internal Temperature with Passive and Controlled Internal Temperature")
        plt.grid(True)
        plt.show()

#---------------------------------Example Usage--------------------------------------------------

def test_thermal_subsystem():
        
    # PID Controller Parameters for Internal Temperature
    Kp = 0.1  # Proportional gain
    Ki = 0.01 # Integral gain
    Kd = 0.05 # Derivative gain

    # Temperature range for internal stability
    set_point_min = -20  # Minimum acceptable internal temperature (°C)
    set_point_max = 40   # Maximum acceptable internal temperature (°C)
    desired_internal_temp = (set_point_min + set_point_max) / 2  # Target midpoint temperature

    # Thermal properties
    thermal_resistance = 0.05  # Thermal resistance between external and internal (°C/W)
    thermal_capacity = 500  # Thermal capacity of the internal system (J/°C)

    # Creating an instance of the ThermalControl class
    thermal_control_system = ThermalControl(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        set_point_min=set_point_min,
        set_point_max=set_point_max,
        desired_internal_temp=desired_internal_temp,
        thermal_resistance=thermal_resistance,
        thermal_capacity=thermal_capacity
    )

    # Example usage of simulate_internal_temperature
    thermal_control_system.simulate_internal_temperature()

    return None
