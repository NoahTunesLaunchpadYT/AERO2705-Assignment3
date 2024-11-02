import numpy as np
from Satellite import satellite as s
from Communications import communications as c
from Propulsion import propulsion as prop
from Power import power as pwr
from Thermal import thermal as thr
import telemetry as t
# import propulsion as p
import gas_dispersion.absorption_interference as ai
import gas_dispersion.gas_dispersion as gd


def main():
    print("\n" + "="*47)
    print("      🚀 Starting ICARUS Simulation 🚀")
    print("="*47 + "\n")
    satellite = s.Satellite("parameters.txt")
    satellite.launch()

    # Satellite Testings
    print("\n\n" + "="*47)
    print("      Running communications test")
    print("="*47 + "\n")
    t.run_communication_test()

    print("\n\n" + "="*47)
    print("      Running propulsion test")
    print("="*47 + "\n")
    prop.test_propulsion_subsystem()

    print("\n\n" + "="*47)
    print("      Running power test")
    print("="*47 + "\n")
    pwr.test_power_subsystem()

    print("\n\n" + "="*47)
    print("      Running thermal test")
    print("="*47 + "\n")
    thr.test_thermal_subsystem()

if __name__ == "__main__":
    main()