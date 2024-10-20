import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import constants as const
import Communications.simulate as sim
import Communications.plane_transformation as pt

class GroundStation:
    def __init__(self, name, latitude, longitude, altitude, antenna_gain, power_transmitted, uplink_frequency):
        self.name = name
        self.latitude = latitude  # Latitude of the ground station
        self.longitude = longitude  # Longitude of the ground station
        self.altitude = altitude  # Altitude in km
        self.antenna_gain = antenna_gain  # Antenna gain in dB
        self.power_transmitted = power_transmitted  # Power transmitted by ground station in dB
        self.uplink_frequency = uplink_frequency  # Uplink frequency in Hz

    def eirp(self):
        return self.power_transmitted + self.antenna_gain
    
    def get_position(self):
        earth_radius = const.R_E  # Earth radius in meters
        x = earth_radius * math.cos(math.radians(self.latitude)) * math.cos(math.radians(self.longitude))
        y = earth_radius * math.cos(math.radians(self.latitude)) * math.sin(math.radians(self.longitude))
        z = self.altitude
        return x, y, z

class GroundStationNetwork:
    def __init__(self, ground_stations):
        self.ground_stations = ground_stations

    def select_nearest_station(self, satellite_position):
        # Find the nearest ground station based on the satellite's current position
        min_distance = float('inf')
        selected_station = None
        for station in self.ground_stations:
            station_pos = station.get_position()
            distance = np.linalg.norm(np.array(satellite_position) - np.array(station_pos))
            if distance < min_distance:
                min_distance = distance
                selected_station = station
        return selected_station

class Satellite:
    def __init__(self, name, altitude, downlink_frequency, antenna_gain, power_transmitted, a, e):
        self.name = name
        self.altitude = altitude  # Altitude of the satellite in meters
        self.downlink_frequency = downlink_frequency  # Downlink frequency in Hz
        self.antenna_gain = antenna_gain  # Antenna gain in dB
        self.power_transmitted = power_transmitted  # Power transmitted by satellite in dB
        self.a = a  # Semi-major axis of the satellite orbit in meters
        self.e = e  # Eccentricity of the satellite orbit
    
    def eirp(self):
        return self.power_transmitted + self.antenna_gain

    def get_position(self, time, positions):
        x = positions[time][0]
        y = positions[time][1]
        z = positions[time][2]
        return x, y, z

    def get_velocity(self, time, velocities):
        vx = velocities[time][0]
        vy = velocities[time][1]
        vz = velocities[time][2]

        return [vx, vy, vz]
    
    def simulate_orbit(self, duration, eccentricity, true_anomaly, specific_angular_momentum, RAAN, inclination, argument_of_perigee):

        perifocal_eci_matrix = pt.perifocal_to_ECI_matrix(RAAN, inclination, argument_of_perigee)

        time, positions, velocities = sim.simulate_launch(eccentricity=eccentricity, 
                                                          true_anomaly=true_anomaly, 
                                                          specific_angular_momentum=specific_angular_momentum, 
                                                          perifocal_eci_matrix=perifocal_eci_matrix, 
                                                          period=duration, 
                                                          J2=False, 
                                                          max_step=1)
        return time, positions, velocities



class LinkBudget:
    def __init__(self, satellite, ground_station, noise_temp, bandwidth):
        self.satellite = satellite
        self.ground_station = ground_station
        self.noise_temp = noise_temp
        self.bandwidth = bandwidth

    def distance_between(self, sat_pos, gs_pos):
        return math.sqrt((sat_pos[0] - gs_pos[0])**2 + (sat_pos[1] - gs_pos[1])**2 + (sat_pos[2] - gs_pos[2])**2)

    def relative_velocity(self, sat_vel, sat_pos, gs_pos):
        # Relative velocity along the line-of-sight direction (radial velocity)
        rel_pos = [sat_pos[i] - gs_pos[i] for i in range(3)]
        rel_vel = sum(sat_vel[i] * rel_pos[i] for i in range(3)) / self.distance_between(sat_pos, gs_pos)
        return rel_vel

    def doppler_shift(self, frequency, relative_velocity):
        c = 3e8 / 1000  # Speed of light in km/s
        return frequency * (relative_velocity / c)

    def free_space_loss(self, distance, frequency):
        c = 3e8 / 1000  # Speed of light (km/s)
        return 20 * math.log10(distance) + 20 * math.log10(frequency) - 147.55
    
    def noise_power(self):
        k = 1.38e-23
        noise_power = 10 * math.log10(k * self.noise_temp * self.bandwidth)
        return noise_power

    def snr(self, recieved_power, noise_power):
        print(f"Recieved Power: {recieved_power:.2f} dB")
        print(f"Noise Power: {noise_power:.2f} dB")
        print(f"SNR: {recieved_power - noise_power:.2f} dB")
        return recieved_power - noise_power
    
    def propogation_delay(self, distance):
        c = 3e8
        return distance / c


    def received_power(self, distance, relative_velocity):
        # Apply Doppler effect to uplink and downlink frequencies
        uplink_doppler_shift = self.doppler_shift(self.ground_station.uplink_frequency, relative_velocity)
        downlink_doppler_shift = self.doppler_shift(self.satellite.downlink_frequency, -relative_velocity)

        # Adjust frequencies for Doppler shift
        uplink_frequency_shifted = self.ground_station.uplink_frequency + uplink_doppler_shift
        downlink_frequency_shifted = self.satellite.downlink_frequency + downlink_doppler_shift

        # Uplink (Ground to Satellite)
        uplink_fspl = self.free_space_loss(distance, uplink_frequency_shifted)
        uplink_received = (self.ground_station.eirp() +
                           self.satellite.antenna_gain - 
                           uplink_fspl)

        # Downlink (Satellite to Ground)
        downlink_fspl = self.free_space_loss(distance, downlink_frequency_shifted)
        downlink_received = (self.satellite.eirp() +
                             self.ground_station.antenna_gain - 
                             downlink_fspl)

        return uplink_received, downlink_received, uplink_fspl, downlink_fspl




class Communications:
    def __init__(self, satellite, ksat_network, ssc_network, noise_temp, bandwidth):
        self.satellite = satellite
        self.ksat_network = ksat_network
        self.ssc_network = ssc_network
        self.noise_temp = noise_temp
        self.bandwidth = bandwidth

    def select_best_station(self, satellite_position):
        ksat_station = self.ksat_network.select_nearest_station(satellite_position)
        ssc_station = self.ssc_network.select_nearest_station(satellite_position)

        # Compare the distances and return the closest station
        ksat_distance = np.linalg.norm(np.array(satellite_position) 
                                       - np.array(ksat_station.get_position()))
        ssc_distance = np.linalg.norm(np.array(satellite_position) 
                                      - np.array(ssc_station.get_position()))

        if ksat_distance < ssc_distance:
            return ksat_station
        else:
            return ssc_station

    def simulate_communication(self, duration):
        satellite = self.satellite
        noise_temp = self.noise_temp
        bandwidth = self.bandwidth
        

        # PARKING ORBIT
        inclination = np.radians(33.4)
        RAAN = np.radians(149.58)
        argument_of_perigee = 0
        e = satellite.e
        a = satellite.a
        true_anomaly = 0
        specific_angular_momentum = get_specific_angular_momentum(a, e)
        time_steps, positions, velocities = satellite.simulate_orbit(duration, e, true_anomaly, specific_angular_momentum, RAAN, inclination, argument_of_perigee)

        # 3D Plot: Satellite Orbit in space
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the orbit path
        ax.plot(positions[0], positions[1], positions[2], label='Satellite Orbit', color='b')

        # Set labels and title
        ax.set_xlabel('X Position (km)')
        ax.set_ylabel('Y Position (kn)')
        ax.set_zlabel('Z Position (kn)')
        ax.set_title('Satellite Orbit in 3D')

        # Show the plot
        plt.legend()
        plt.show()

        
        distances = []

        sat_eirp_vals = []
        gs_eirp_vals = []
        sat_eirp = satellite.eirp()
        sat_eirp_vals.append(sat_eirp)

        uplink_powers = []
        downlink_powers = []

        uplink_fspls = []
        downlink_fspls = []

        uplink_snr_vals = []
        downlink_snr_vals = []

        prop_delays = []


        for idx, t in enumerate(time_steps):
            sat_pos = satellite.get_position(idx, positions.T)

            ground_station = self.select_best_station(sat_pos)

            link_budget = LinkBudget(satellite, ground_station, noise_temp, bandwidth)

            sat_vel = satellite.get_velocity(idx, velocities.T)

            gs_pos = ground_station.get_position()
            distance = link_budget.distance_between(sat_pos, gs_pos)
            rel_velocity = link_budget.relative_velocity(sat_vel, sat_pos, gs_pos)
            uplink_power, downlink_power, uplink_fspl, downlink_fspl = link_budget.received_power(distance, rel_velocity)
            gs_eirp = ground_station.eirp()
            gs_eirp_vals.append(gs_eirp)
        

            distances.append(distance)
            uplink_powers.append(uplink_power)
            downlink_powers.append(downlink_power)
            uplink_fspls.append(uplink_fspl)
            downlink_fspls.append(downlink_fspl)

            noise_power = link_budget.noise_power()
            uplink_snr = link_budget.snr(uplink_power, noise_power)
            downlink_snr = link_budget.snr(downlink_power, noise_power)
            prop_delay = link_budget.propogation_delay(distance)

            uplink_snr_vals.append(uplink_snr)
            downlink_snr_vals.append(downlink_snr)
            prop_delays.append(prop_delay)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, distances)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title('Distance Between Satellite and Ground Station Over Time')
        plt.show()


        plt.figure(figsize=(10, 6))
        plt.bar(['Satellite EIRP', 'Ground Station EIRP'], [sat_eirp, gs_eirp], color=['orange', 'blue'])
        plt.ylabel('EIRP (dB)')
        plt.title('Satellite and Ground Station EIRP')
        plt.show()

        # UPLINk AND DOWNLINK POWER
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(time_steps, uplink_powers, label='Uplink Power (dB)')
        plt.plot(time_steps, downlink_powers, label='Downlink Power (dB)', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Received Power (dB)')
        plt.title('Uplink and Downlink Power Over Time (With Doppler Effect)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time_steps, uplink_fspls, label='Uplink FSPL (dB)', color='orange')
        plt.plot(time_steps, downlink_fspls, label='Downlink FSPL (dB)', linestyle='--', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('FSPL (dB)')
        plt.title('Uplink and Downlink Free Space Path Loss Over Time')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time_steps, uplink_snr_vals, label='Uplink SNR (dB)')
        plt.plot(time_steps, downlink_snr_vals, label='Downlink SNR (dB)', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('SNR (dB)')
        plt.title('Uplink and Downlink Signal-to-Noise Ratio Over Time')
        plt.legend()

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, prop_delays, label='Propagation Delay (s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Propagation Delay (s)')
        plt.title('Propagation Delay Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()


    



def get_specific_angular_momentum(semimajor_axis: 
                                  float, eccentricity: float
                                  ) -> float:
    
    specific_angular_momentum = np.sqrt( semimajor_axis * const.mu 
                                        * ( 1- ( eccentricity ** 2 ) ) )
    
    return specific_angular_momentum


def get_orbital_period(semimajor_axis: float) -> float:
    """
    Calculate the orbital period of the satellite.

    Parameters:
        semimajor_axis (float): The semi-major axis of the orbit.

    Returns:
        float: The orbital period of the satellite in seconds.
    """
    mu = const.mu
    orbital_period  = ( ( 2 * math.pi ) / math.sqrt(mu) * semimajor_axis**(3 / 2) )
    return orbital_period






# Create Satellite and GroundStation objects
alt_park = 400
r_park = const.R_E + alt_park 
a_park = r_park / 2
e_park = 0
a1 = 26553
e1 = 0.737

satellite = Satellite(name="CommSat", altitude=30000, downlink_frequency=27.5e9, 
                      antenna_gain=40, power_transmitted=10, a=a_park, e=e_park)  # Satellite power in dB

# Create GroundStation object for Bathurst
ground_station = GroundStation(name="Bathurst Launch Site", 
                               latitude=-33.4197, longitude=149.5806, 
                               altitude=0.670, antenna_gain=40, power_transmitted=40, uplink_frequency=2.18e9)


T = get_orbital_period(satellite.a)
# Simulate the communication

# KSAT Ground Stations
ksat_ground_stations = [
    GroundStation(name="Svalbard", latitude=78.2298, longitude=15.4078, altitude=0.0, antenna_gain=40, power_transmitted=50, uplink_frequency=2.2e9),
    GroundStation(name="Troll", latitude=-72.0114, longitude=2.5350, altitude=1.29, antenna_gain=40, power_transmitted=50, uplink_frequency=2.2e9),
    GroundStation(name="Tromsø", latitude=69.6492, longitude=18.9560, altitude=0.1, antenna_gain=42, power_transmitted=55, uplink_frequency=2.2e9),
    GroundStation(name="Inuvik", latitude=68.3600, longitude=-133.7200, altitude=0.1, antenna_gain=40, power_transmitted=50, uplink_frequency=2.2e9),
    GroundStation(name="Singapore", latitude=1.3521, longitude=103.8198, altitude=0.02, antenna_gain=35, power_transmitted=48, uplink_frequency=2.2e9),
    GroundStation(name="Mauritius", latitude=-20.3484, longitude=57.5522, altitude=0.1, antenna_gain=38, power_transmitted=50, uplink_frequency=2.2e9)
]


# SSC Ground Stations
ssc_ground_stations = [
    GroundStation(name="Esrange", latitude=67.8831, longitude=21.0506, altitude=0.3, antenna_gain=38, power_transmitted=50, uplink_frequency=2.18e9),
    GroundStation(name="Santiago", latitude=-33.4489, longitude=-70.6693, altitude=0.52, antenna_gain=35, power_transmitted=50, uplink_frequency=2.2e9),
    GroundStation(name="Clewiston", latitude=26.7545, longitude=-80.9347, altitude=0.01, antenna_gain=38, power_transmitted=48, uplink_frequency=2.18e9),
    GroundStation(name="Punta Arenas", latitude=-53.1638, longitude=-70.9171, altitude=0.1, antenna_gain=36, power_transmitted=50, uplink_frequency=2.18e9),
    GroundStation(name="South Point Hawaii", latitude=19.0820, longitude=-155.6221, altitude=0.02, antenna_gain=39, power_transmitted=50, uplink_frequency=2.2e9),
    GroundStation(name="Western Australia", latitude=-31.9505, longitude=115.8605, altitude=0.2, antenna_gain=38, power_transmitted=50, uplink_frequency=2.2e9)
]


# Create Networks
ksat_network = GroundStationNetwork(ksat_ground_stations)
ssc_network = GroundStationNetwork(ssc_ground_stations)

def run_communication_test():
    comms = Communications(satellite, ksat_network, ssc_network, noise_temp=290, bandwidth=10e6)
    comms.simulate_communication(duration=T)

def main():
    comms = Communications(satellite, ksat_network, ssc_network, noise_temp=290, bandwidth=10e6)
    comms.simulate_communication(duration=T)

if __name__ == "__main__":
    main()