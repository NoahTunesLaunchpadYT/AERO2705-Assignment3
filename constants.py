'''
SID: 530616209
'''

import numpy as np

R_E = 6378                              # Earth radius (km)
mu = 398600                             # Earth gravitational parameter (km^3/s^2) (G*m1*m2)
PERIGEE_ANGLE = 0                       # angle from perigee to perigee
APOGEE_ANGLE = np.pi                    # angle from perigee to apogee
J2 = 0.00108263
OMEGA_E = (2*np.pi+2*np.pi/365.26)/(24*3600)
g = 9.80665
KM_TO_M = 1e3
M_TO_KM = 1e-3
k = 1.38e-23
R  = 8.314              #   J/(mol*k)   Universal Gas Constant  


ASSUMED_MASS = 1000
CHOSEN_THRUST = 0.039 
CHOSEN_ISP = 1400
CHOSEN_MASS = 2565


g_EARTH = 9.81
R_EARTH = 6378 
R_pol_EARTH = 6357
mu_EARTH = 398600
f_EARTH = 0.003353
J2_EARTH = 1.08263 * (10 ** (-3))
