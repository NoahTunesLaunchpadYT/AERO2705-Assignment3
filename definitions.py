# definitions.py
# Author: 530494928
# Date: 2024-09-23
# Description: Contains all orbital parameters and constants used in mainA1.py

import numpy as np

EQUATORIAL_RADIUS = 6378
MU = 398600                 # km^3/s^2

# Investigation 1

FLOCK_TLE = (
    "1 70360C 24149BM 24235.92391204 .00016145 00000+0 79964-3 0 2359 \n"
    "2 70360  97.4446 311.4156 0006108  97.4854  52.0254 15.17819342 15"
)

# Investigation 2

starting_params = {
    "altitude_of_perigee": 500,
    "altitude_of_apogee": 500,
    "inclination_angle": 30,
    "raan": 90,
    "argument_of_perigee": 0
}

target_params = {
    "altitude_of_perigee": 700,
    "altitude_of_apogee": 3400,
    "inclination_angle": 45,
    "raan": 30,
    "argument_of_perigee": 10
}

Isp_values = np.array([200, 300, 400])

# Investigation 3
starting_mass_array = [1, 10, 100, 1000]
G0 = 9.80665e-3 # km/s^2

thrusters_array = [
    {"name": "BHT_100", "thrust": 7e-6, "isp": 1000, "mass": 1.16},
    {"name": "BHT_200", "thrust": 13e-6, "isp": 1390, "mass": 0.98},  
    {"name": "BHT_350", "thrust": 17e-6, "isp": 1244, "mass": 1.7},  
    {"name": "BHT_600", "thrust": 39e-6, "isp": 1300, "mass": 2.6}, 
    {"name": "BHT_1500", "thrust": 101e-6, "isp": 1710, "mass": 6.3}, 
    {"name": "BHT_6000", "thrust": 201e-6, "isp": 1900, "mass": 120},
    {"name": "BHT_20K", "thrust": 1005e-6, "isp": 2515, "mass": 400} 
]

# Investigation 4

chase_parameters = [
    [1000, 2.7, 1100],
    [1000, 3.5, 1100],
    [1000, 4.1, 800],
    [1000, 4.6, 1000]
]