from Communications import communications as comms
from Communications import orbits as o
from gas_dispersion import payload as pld
import constants as c
import numpy as np

class AOCS:
    def __init__(self, params):
        print("AOCS initialised")
        self.initial_alt_p = params['alt_p']
        self.initial_alt_a = params['alt_a']
        self.initial_inclination = params['i_deg']
        self.initial_RAAN = params['RAAN_deg']
        self.initial_arg_p = params['arg_p_deg']
        
        self.starting_orbit = o.Orbit(self.initial_alt_p + c.R_E,
                                 self.initial_alt_a + c.R_E,
                                 self.initial_inclination,
                                 self.initial_RAAN,
                                 self.initial_arg_p,
                                 )
        self.starting_orbit.compute_orbital_parameters()

        t = self.starting_orbit.propagated_times[0]
        y = self.starting_orbit.propagated_orbit[0]

        self.solution_ts = [[t]]
        self.solution_ys = [[y]]

        self.communications = comms.Communications(params, self)
        self.payload = pld.Payload(params, self)


    def run(self):
        self.solution_ts, self.solution_ys = self.communications.receive_solution()
        
        self.concatenated_state_array = np.hstack(self.solution_ys)
        self.concatenated_time_array = np.hstack(self.solution_ts)

        self.payload.simulate_payload_on_path(self.solution_ts, self.solution_ys)
        
        gs = self.communications.select_best_station(self.solution_ys[-1][0:3, -1])