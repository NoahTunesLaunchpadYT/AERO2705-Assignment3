import math
import numpy as np
import constants as const
import random
import ground_station_simulations.transfer_search_algorithms as ta

class GroundStation:
    def __init__(self, params, AOCS):
        print("- GroundStation initialised")

        self.params = params
        self.target_orbits = []
        self.AOCS = AOCS
        self.starting_orbit = self.get_starting_orbit()
        self.solution_ts = [[]]
        self.solution_ys = [[[]]]

    def send_solution(self):
        self.target_orbits = self.generate_three_random_orbits()
        
        orbit_list = [self.generate_starting_orbit()] + self.target_orbits
        
        path = ta.get_best_solution(orbit_list, "hohmann-like", True)
        other_path = ta.get_best_solution(orbit_list, "hohmann-like-with-phasing", True)

        self.solution_ts = path.time_array_segments
        self.solution_ys = path.solution_array_segments

        return self.solution_ts, self.solution_ys

    def generate_three_orbits(self):

        target_orbit_A = {
            "altitude_of_perigee": 3000,
            "altitude_of_apogee": 3400,
            "inclination_angle": 45,
            "raan": -90,
            "argument_of_perigee": -60,
            "initial_true_anomaly": 0
        }

        target_orbit_B = {
            "altitude_of_perigee": 6000,
            "altitude_of_apogee": 7000,
            "inclination_angle": 500,
            "raan": 15,
            "argument_of_perigee": -10,
            "initial_true_anomaly": 0
        }

        target_orbit_C = {
            "altitude_of_perigee": 2000,
            "altitude_of_apogee": 20000,
            "inclination_angle": -20,
            "raan": 2,
            "argument_of_perigee": 14,
            "initial_true_anomaly": 40
        }

        random_orbits = [target_orbit_A, target_orbit_B, target_orbit_C]

        return random_orbits

    def generate_three_random_orbits(self):
        orbit_types = {
            "LEO": [6378 + 160, 6378 + 2000],      # LEO: 160 km to 2000 km alt
            "MEO": [6378 + 2000, 6378 + 35786],    # MEO: 2000 to 35786 km alt
            "GEO": [6378 + 35786, 6378 + 35786],   # GEO: 35786 km alt
            "HEO": [6378 + 1000, 6378 + 40270]     # HEO: 1000 to 40270 km alt (Maximum altitude before a satellite escapes Earth's gravity)
        }
        
        orbit_choices = ["LEO", "MEO", "GEO", "HEO"]
        weights = [(3790/4550), (139/4550), (565/4550), (56/4550)]  # 83.3% LEO, 3.05% MEO, 12.42% GEO, 1.23% HEO
        random_orbits = []
        
        print("\nPlease input an orbit type of: LEO, MEO, GEO, HEO. Empty or invalid inputs will generate random parameters.")
        
        for i in range(3):
            input_type = input(f"What type of orbit will orbit {i+1} be? ").upper().strip()
    
            if input_type not in orbit_types:
                input_type = random.choices(orbit_choices, weights)[0]
            lower_range, upper_range = orbit_types[input_type]
    
            if input_type == "HEO":
                perigee = random.uniform(6378 + 1000, 6378 + 2000)  # +1000km LEO range for perigee
                apogee = random.uniform(6378 + 35786, 6378 + 40270)  # GEO or higher for apogee
            else:
                perigee = random.uniform(lower_range, upper_range)
                apogee = random.uniform(lower_range, upper_range)
            
            if perigee > apogee:
                perigee, apogee = apogee, perigee
            
            RAAN = random.uniform(0, 90)
            inclination = random.uniform(0, 90)
            arg_of_perigee = random.uniform(0, 90)
            
            generated_orbit = {
                "altitude_of_perigee": perigee - 6378,
                "altitude_of_apogee": apogee - 6378,
                "inclination_angle": inclination,
                "raan": RAAN,
                "argument_of_perigee": arg_of_perigee,
                "initial_true_anomaly": 0
            }
            
            random_orbits.append(generated_orbit)

        for i in range(0, 3):
            print(f"Orbit {i + 1} - Type: {input_type}")

        return random_orbits
    
    def get_starting_orbit(self):
        
        starting_orbit = self.AOCS.starting_orbit

        starting_orbit = {
            "altitude_of_perigee": starting_orbit.alt_p,
            "altitude_of_apogee": starting_orbit.alt_a,
            "inclination_angle": starting_orbit.i_deg,
            "raan": starting_orbit.RAAN_deg,
            "argument_of_perigee": starting_orbit.arg_p_deg,
            "initial_true_anomaly": 0
        }

        return starting_orbit
    
    def generate_starting_orbit(self):
        
        starting_params = {
            "altitude_of_perigee": 500,
            "altitude_of_apogee": 505,
            "inclination_angle": 30,
            "raan": 0,
            "argument_of_perigee": 0,
            "initial_true_anomaly": 0
        }

        return starting_params

class Communications:
    def __init__(self, params, AOCS):
        print("- Communications initialised")
        satellite_position = AOCS.solution_ys[0][0][0:3]

        self.params = params
        self.AOCS = AOCS
        self.gs = self.select_best_station(satellite_position)

    def receive_solution(self):
        return self.gs.send_solution()        

    def select_best_station(self, satellite_position):
        x, y, z = satellite_position

        gs = GroundStation(self.params, self.AOCS)
        
        return gs
    
