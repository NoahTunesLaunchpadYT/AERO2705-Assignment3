import os
import AOCS.AOCS as a

class Satellite:
    def __init__(self, file_name: str = "parameters.txt"):
        # Construct the relative path to parameters.txt in the parent directory
        file_path = os.path.join(os.path.dirname(__file__), '..', file_name)

        # Read parameters from the file
        params = self.read_orbital_parameters(file_path)

        # Initialize AOCS with parameters
        self.AOCS = a.AOCS(params)

    def launch(self):
        print("Running")
        self.AOCS.run()

    def get_payload(self):
        return self.payload

    def get_comms(self):
        return self.communications

    def read_orbital_parameters(self, file_path):
        parameters = {}
        with open(file_path, 'r') as file:
            for line in file:
                split_line = line.split(':')
                parameter_name, value = split_line[1].split('=')
                parameters[parameter_name.strip()] = float(value.strip())
        
        return parameters
