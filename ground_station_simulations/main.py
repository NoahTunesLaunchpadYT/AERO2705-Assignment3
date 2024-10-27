import transfer_algorithms as al
import plotting as pl

starting_params = {
    "altitude_of_perigee": 500,
    "altitude_of_apogee": 500,
    "inclination_angle": 30,
    "raan": 90,
    "argument_of_perigee": 0,
    "initial_true_anomaly": 0
}

target_orbit_A = {
    "altitude_of_perigee": 700,
    "altitude_of_apogee": 3400,
    "inclination_angle": 45,
    "raan": 30,
    "argument_of_perigee": 10,
    "initial_true_anomaly": 0
}

target_orbit_B = {
    "altitude_of_perigee": 5000,
    "altitude_of_apogee": 6000,
    "inclination_angle": 145,
    "raan": 15,
    "argument_of_perigee": 10,
    "initial_true_anomaly": 0
}

target_orbit_C = {
    "altitude_of_perigee": 2000,
    "altitude_of_apogee": 2400,
    "inclination_angle": 20,
    "raan": 0,
    "argument_of_perigee": 0,
    "initial_true_anomaly": 0
}

def main() -> None:
    orbits_params = [starting_params, target_orbit_A, target_orbit_B, target_orbit_C]
    
    al.get_best_solution(orbits_params, sequence_type="hohmann-like", plotting=True)

    return

if __name__ == "__main__":
    main()
