import AOCS.AOCS as a

class Satellite:
    def __init__(self, file_path: str):
        params = self.read_orbital_parameters(file_path)

        self.AOCS = a.AOCS(params)

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
        file.close()

        return parameters
    