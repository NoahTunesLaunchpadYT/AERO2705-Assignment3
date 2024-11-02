import numpy as np
from Satellite import satellite as s
from Communications import communications as c
import telemetry as t
import propulsion as p
import gas_dispersion.absorption_interference as ai
import gas_dispersion.gas_dispersion as gd


def main():
    print("\n" + "="*47)
    print("      🚀 Starting ICARUS Simulation 🚀")
    print("="*47 + "\n")
    satellite = s.Satellite("parameters.txt")
    satellite.launch()

    # Satellite Testings
    # t.run_communication_test()
    # p.test_propulsion_subsystem()

    # Science testing
    # ai.test_absorption_interference()
    # gd.test_gas_dispersion()

if __name__ == "__main__":
    main()