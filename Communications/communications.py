import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import constants as const
import random
import ground_station_simulations.maneuver_simulations as ms
import ground_station_simulations.orbit_plots as op

class GroundStation:
    def __init__(self, params, AOCS):
        self.params = params
        self.target_orbits = []
        self.AOCS = AOCS
        self.starting_orbit = self.get_starting_orbit()
        self.solution_ts = [[]]
        self.solution_ys = [[[]]]

    def send_solution(self):
        self.target_orbits = self.generate_three_random_orbits()
        
        self.generate_solution_arrays()

        return self.solution_ts, self.solution_ys

    def generate_three_random_orbits(self):
        orbit_types = {
            "LEO": [const.R_E + 160, const.R_E + 2000],      # LEO: 160 km to 2000 km alt
            "MEO": [const.R_E + 2000, const.R_E + 35786],    # MEO: 2000 to 25786 km alt
            "GEO": [const.R_E + 35786, const.R_E + 35786],   # GEO: 35786 km alt
            "HEO": [const.R_E + 1000, const.R_E + 40000]     # HEO: 1000 to 40000 km alt
        }
        
        default_range = [6378 + 160, 6378 + 40270] 
        # quick google search says 40,270 km is the max alt before escaping earths gravity but
        # someone please fact check all these numbers ty <3
        random_orbits = []
        
        print("Please input an orbit type of: LEO, MEO, GEO, HEO. Empty or invalid inputs will generate random parameters.")
        
        for i in range(3):
            input_type = input(f"What type of orbit will orbit {i+1} be? ").upper().strip()

            if input_type in orbit_types:
                lower_range, upper_range = orbit_types[input_type]
            else:
                lower_range, upper_range = default_range
            perigee = random.uniform(lower_range, upper_range)
            apogee = random.uniform(lower_range, upper_range)

            if perigee > apogee:
                perigee, apogee = apogee, perigee
            RAAN = random.uniform(10, 50)
            inclination = random.uniform(10, 50)
            arg_of_perigee = random.uniform(10, 50)
            generated_orbit = {
                "altitude_of_perigee": perigee - const.R_E,
                "altitude_of_apogee": apogee - const.R_E,
                "inclination_angle": inclination,
                "raan": RAAN,
                "argument_of_perigee": arg_of_perigee
            }

            random_orbits.append(generated_orbit)

        return random_orbits
    
    def get_starting_orbit(self):
        starting_orbit = self.AOCS.starting_orbit

        starting_orbit = {
            "altitude_of_perigee": starting_orbit.alt_p,
            "altitude_of_apogee": starting_orbit.alt_a,
            "inclination_angle": starting_orbit.i_deg,
            "raan": starting_orbit.RAAN_deg,
            "argument_of_perigee": starting_orbit.arg_p_deg
        }

        return starting_orbit
    
    def generate_solution_arrays(self):
        transfer_dvs = {}

        satellite_array = [self.starting_orbit, 
                           self.target_orbits[0], 
                           self.target_orbits[1], 
                           self.target_orbits[2]]
    
        # Find dv for each transfer manoeuvre
        for i in satellite_array:
            for j in satellite_array:
                if i == j:
                    continue
                else: 
                    dv = ms.investigate_maneuver(i, j, False)
                    transfer_dvs[f'{i} + {j}'] = dv

        # Generate all permutations of the four orbits (indices 1 to 4)
        manoeuvre_combos = self.permute_orbits([0, 1, 2, 3])
        print("\nPERMUTATIONS")
        print(manoeuvre_combos)
        print("\n")
            
        total_delta_vs = []
        
        for combo in manoeuvre_combos:
            total_delta_v = sum(
                transfer_dvs[f'{satellite_array[combo[i]]} + {satellite_array[combo[i+1]]}']
                for i in range(len(combo) - 1)
            )
            total_delta_vs.append(total_delta_v)
        
        # Find the best manoeuvre (minimum delta_v)
        best_manoeuvre_index = total_delta_vs.index(min(total_delta_vs))
        best_combo = manoeuvre_combos[best_manoeuvre_index]

        # Reorder the satellite array to match the best combo
        reordered_satellite_array = [satellite_array[i] for i in best_combo]
        
        print("Best Manoeuvre:", min(total_delta_vs))
        print("Optimal Order of Indices:", best_combo)

        self.solution_ts, self.solution_ys = ms.simulate_manoeuvres(reordered_satellite_array)

    def permute_orbits(self, arr):
        if len(arr) == 1:
            return [arr]
        
        parking_orbit = arr[0]
        remaining = arr[1:]
        
        result = []
        for i in range(len(remaining)):
            current = remaining[i]
            rest = remaining[:i] + remaining[i+1:]
            for p in self.permute_orbits([current] + rest):
                result.append([parking_orbit] + p)
        
        return result

    def plot_all_data(self, AOCS):
        self.plot_path(AOCS)

    def plot_path(self, AOCS):
        ax = op.new_figure()
        op.plot_eci_orbit_segments(ax, AOCS.solution_ys, color_offset=0, label_ends=False)
        op.show(ax)

class Communications:
    def __init__(self, params, AOCS):
        satellite_position = AOCS.solution_ys[0][0][0:3]

        self.params = params
        self.AOCS = AOCS
        self.gs = self.select_best_station(satellite_position)

    def receive_solution(self):
        print("Requesting solution to flight")
        return self.gs.send_solution()        

    def select_best_station(self, satellite_position):
        x, y, z = satellite_position
        print(f"Satellite position in km: X = {x:.2f} km, Y = {y:.2f} km, Z = {z:.2f} km")

        gs = GroundStation(self.params, self.AOCS)
        
        return gs
    
