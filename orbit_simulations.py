# orbit_simulations.py
# Author: 530494928
# Date: 2024-08-23
# Description: This module provides simulations to investigate different orbital scenarios.
# The file includes functions to run orbital simulations with or without oblateness effects and 
# to analyze orbital parameters based on TLE data.

import orbit as o
import definitions

def investigate_orbit_simulations() -> None:
    """
    Investigates various orbital scenarios by simulating orbits and calculating orbital parameters
    using Two-Line Element (TLE) data. Simulates orbits with and without considering Earth's oblateness effects.

    The simulation includes:
        1. Parsing TLE data for orbital parameters.
        2. Simulating the orbital dynamics both with and without the J2 oblateness effect.
        3. Plotting the resulting orbit paths and ground tracks.

    Returns: None
    """
    # Create an instance of the OrbitalParameters class
    orbit = o.Orbit()

    # Parse the TLE data
    orbit.parse_tle(definitions.FLOCK_TLE)
    orbit.calculate_orbital_constants_from_tle()

    # Display the extracted information
    print(orbit)

    # Simulating
    orbit.calculate_initial_state()
    print("Simulating orbit about a spherical Earth")
    orbit.simulate_orbit(oblateness_effects = False)
    print("Simulating orbit about an oblate, non rotating Earth")
    orbit.simulate_orbit(oblateness_effects = True, duration = 7 * 24 * 60 * 60, stationary_ground = True)
    print("Simulating orbit about an oblate Earth")
    orbit.simulate_orbit(oblateness_effects = True, duration = 7 * 24 * 60 * 60)

    return 

def main() -> None:
    investigate_orbit_simulations()
    return

if __name__ == "__main__":
    main()

