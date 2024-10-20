import random
import numpy as np

def main():
    orbit_types = {
        "LEO": [6378 + 160, 6378 + 2000],      # LEO: 160 km to 2000 km alt
        "MEO": [6378 + 2000, 6378 + 35786],    # MEO: 2000 to 25786 km alt
        "GEO": [6378 + 35786, 6378 + 35786],   # GEO: 35786 km alt
        "HEO": [6378 + 1000, 6378 + 40000]     # HEO: 1000 to 40000 km alt
    }
    
    default_range = [0, 6378 + 40270] 
    # quick google search says 40,270 km is the max alt before escaping earths gravity but
    # someone please fact check all these numbers ty <3
    random_orbits = []
    
    print("Please input an orbit type of: LEO, MEO, GEO, HEO. Empty or invalid inputs will generate random parameters.")
    
    for i in range(3):
        input_type = input(f"What type of orbit will orbit {i+1} be (LEO, MEO, GEO or HEO)?: ").upper().strip()

        if input_type in orbit_types:
            lower_range, upper_range = orbit_types[input_type]
        else:
            lower_range, upper_range = default_range

        perigee = random.uniform(lower_range, upper_range)
        apogee = random.uniform(lower_range, upper_range)

        if perigee > apogee:
            perigee, apogee = apogee, perigee

        RAAN = random.uniform(0, 90)
        inclination = random.uniform(0, 90)
        arg_of_perigee = random.uniform(0, 90)

        generated_orbit = {
            "perigee": perigee,
            "apogee": apogee,
            "RAAN": RAAN,
            "inclination": inclination,
            "argument of perigee": arg_of_perigee
        }

        random_orbits.append(generated_orbit)
    
    for i in range(3):
        print(f"\n---ORBIT {i+1}---")
        print(f"Perigee:                {random_orbits[i]['perigee']}")
        print(f"Apogee:                 {random_orbits[i]['apogee']}")
        print(f"RAAN:                   {random_orbits[i]['RAAN']}")
        print(f"Inclination:            {random_orbits[i]['inclination']}")
        print(f"Argument of Perigee:    {random_orbits[i]['argument of perigee']}")

       
if __name__ == "__main__":
    main()
