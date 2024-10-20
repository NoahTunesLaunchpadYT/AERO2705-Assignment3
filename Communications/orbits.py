import numpy as np
import constants as const
import scipy.integrate as spi
import matplotlib.pyplot as plt
# add antenna gain to satellite class and add to constructor

def period(n: float)-> float:
    """Calculates the period of a Keplerian orbit.

    Args:
        n (float): mean motion

    Returns:
        float: the orbital period
    """    
    # convert mean motion from rev/day to rad/s
    n = (n*2*np.pi)/(60*60*24) 
    T = 2*np.pi / n
    return T

def specific_angular_momentum(a: float, e: float) -> float:
    """Calculates the specific angular momentum of the orbit

    Args:
        a (float): semi-major axis (km)
        e (float): eccentricity of the orbit

    Returns:
        float: the specific angular momentum (km^2/s)
    """    
    h = np.sqrt(a*const.mu*(1-e**2))
    return h

def perigee_radius(h: float, e: float) -> float:
    """ Calculates the perigee radius

    Args:
        h (float): specific angular momentum
        e (float): eccentricity

    Returns:
        float: perigee radius
    """    
    r_p = (h**2/const.mu)*(1/(1+e))
    return r_p

def apogee_radius(h: float, e: float) -> float:
    """ Calculates the apogee radius

    Args:
        h (float): specific angular momentum
        e (float): eccentricity

    Returns:
        float: apogee radius
    """  
    r_a = (h**2/const.mu)*(1/(1-e))
    return r_a

def specific_energy(a: float) -> float:
    """Calculates the specific energy of an orbit

    Args:
        a (float): semimajor axis (km)

    Returns:
        float: specific energy of the orbit (km^2/s^2)
    """    
    # epsilon = -(1/2)*(const.mu**2/h**2)*(1-e**2)
    epsilon = -const.mu/(2*a)
    return epsilon


def orbital_period(a: float) -> float:
    """Calculates the period of a Keplerian orbit.

    Args:
        a (float): semimajor axis (km)

    Returns:
        float: The orbital period in seconds
    """
    T = ((2*np.pi)/np.sqrt(const.mu))*(a**(3/2))
    return T

def semimajor_from_period(T: float) -> float:
    """ Calculate the semi-major axis from the orbital period

    Args:
        T (float): Orbital period in seconds

    Returns:
        float: The semi-major axis in kilometers
    """
    a = ((T*np.sqrt(const.mu))/(2*np.pi))**(2/3)
    return a

def apogee_from_a(a: float, r_p: float) -> float:
    """ Calculate the apogee of an orbit based on semi-major axis and perigee distance.

    Args:
        a (float): Semi-major axis of the orbit.
        r_p (float): Perigee distance from the center of the Earth.

    Returns:
        float: The apogee distance from the center of the Earth.
    """
    r_a = 2*a - r_p
    return r_a

def semimajor_axis(r_p: float, r_a: float) -> float:
    """ Calculate the semi-major axis from perigee and apogee distances.

    Args:
        r_p (float): Perigee distance from the center of the Earth.
        r_a (float): Apogee distance from the center of the Earth.

    Returns:
        float: The semi-major axis in kilometers.
    """
    a = (r_p+r_a)/2
    return a

def eccentricity(r_p: float, r_a: float) -> float:
    """Calculates the eccentricity of the orbit.

    Args:
        r_p (float): perigee radius (km)
        r_a (float): apogee radius (km)

    Returns:
        float: eccentricity of the orbit
    """
    e = (r_a-r_p)/(r_a+r_p)
    return e

def specific_angular_momentum_from_perigee(r_p: float, e: float) -> float:
    """Calculates the specific angular momentum of the orbit

    Args:
        r_p (float): perigee radius (km)
        e (float): eccentricity of the orbit

    Returns:
        float: the specific angular momentum (km^2/s)
    """    
    h = np.sqrt((r_p*const.mu)*(1+e))
    return h

def periapsis_velocity(h: float, r_p: float) -> float:
    """Calculates the velocity at periapsis (perigee)

    Args:
        h (float): specific angular momentum (km^2/s)
        r_p (float): perigee radius (km)

    Returns:
        float: velocity at periapsis (km/s)
    """    
    if h == 0:
        return 0
    v_p = h/r_p
    return v_p

def apoapsis_velocity(h: float, r_a: float) -> float:
    """Calculates the velocity at apoapsis (apogee)

    Args:
        h (float): specific angular momentum (km^2/s)
        r_a (float): apogee radius (km)

    Returns:
        float: velocity at apoapsis (km/s)
    """    
    if h == 0:
        return 0
    v_a = h/r_a
    return v_a

def orbit_dynamics(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """ Calculate the dynamics of an orbit

    Args:
        t (float): Time
        y (np.ndarray): State vector containing position and velocity 
        mu (float): Gravitational parameter 

    Returns:
        np.ndarray: The time derivatives of position and velocity 
    """
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    r_dot = v
    v_dot = -(const.mu/r_norm**3)*r
    
    return np.concatenate((r_dot, v_dot))

def plot_orbit(positions: np.ndarray, title: str):
    """ Plot the 3D orbit of a satellite.

    Args:
        positions (np.ndarray): A 2D NumPy array of positions
        title (str): The title for the plot.

    Returns:
        None
    """
    x = positions[0]
    y = positions[1]
    z = positions[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Orbit trajectory')
    ax.set_xlabel('X position (km)')
    ax.set_ylabel('Y position (km)')
    ax.set_zlabel('Z position (km)')
    ax.set_title(title)
    ax.scatter(0, 0, 0, color='darkseagreen', label='Earth', s=50)
    ax.set_box_aspect([1,1,1]) 
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.set_zlim([min(z), max(z)])
    plt.legend()
    plt.show()
    return


def rot_1(angle: float) -> np.ndarray:
    """Returns the rotation matrix 1

    Args:
        angle (float): The angle of rotation in radians.

    Returns:
        np.ndarray: The rotation matrix.
    """
    R_1 = np.array([[1, 0, 0],
                    [0, np.cos(angle), np.sin(angle)],
                    [0, -np.sin(angle), np.cos(angle)]])
    return R_1

def rot_2(angle: float) -> np.ndarray:
    """Returns the rotation matrix about axis 2, the y-axis.

    Args:
        angle (float): The angle of rotation in radians.

    Returns:
        np.ndarray: The rotation matrix.
    """
    R_2 = np.array([[np.cos(angle), 0, -np.sin(angle)],
                    [0, 1, 0],
                    [np.sin(angle), 0, np.cos(angle)]])
    return R_2
    
def rot_3(angle: float) -> np.ndarray:
    """Returns the rotation matrix about axis 3, the z-axis.

    Args:
        angle (float): The angle of rotation in radians.

    Returns:
        np.ndarray: The rotation matrix.
    """
    R_3 = np.array([[np.cos(angle), np.sin(angle), 0], 
                    [-np.sin(angle), np.cos(angle), 0], 
                    [0, 0, 1]])
    return R_3

def euler_angle_sequence(arg_perigee: float, inclination: float, RAAN: float) -> np.ndarray:
    """ Calculate the rotation matrix 
    
    Args:
        arg_perigee (float): The argument of perigee in radians
        inclination (float): The inclination in radians
        RAAN (float): The right ascension of the ascending node in radians

    Returns:
        np.ndarray: The rotation matrix 
    """
    Q = rot_3(arg_perigee) @ rot_1(inclination) @ rot_3(RAAN)
    return Q


def perifocal_position_vector(h: float, e: float, theta: float) -> np.ndarray:
    """ Calculate the perifocal position vector of an orbiting body.
    
    Args:
        h (float): Specific angular momentum of the orbit
        e (float): Eccentricity of the orbit
        theta (float): True anomaly 

    Returns:
        np.ndarray: The perifocal position vector
    """
    vector = np.array([np.cos(theta), np.sin(theta), 0])
    scalar = (h**2/const.mu)*(1/(1 + e * np.cos(theta)))
    r = scalar*vector
    return r

def perifocal_velocity_vector(h: float, e: float, theta: float) -> np.ndarray:
    """ Calculate the perifocal velocity vector of an orbiting body.

    Args:
        h (float): Specific angular momentum of the orbit
        e (float): Eccentricity of the orbit
        theta (float): True anomaly 

    Returns:
        np.ndarray: The perifocal velocity vector 
    """
    vector = np.array([-np.sin(theta), e + np.cos(theta), 0])
    scalar = const.mu/h
    v = scalar*vector
    return v

class Orbit:
    def __init__(self, r_p, r_a, i_deg, RAAN_deg, arg_p_deg, start_time = 0, end_time = 0, dv_at_pm = 0):
        self.r_p = r_p
        self.r_a = r_a
        self.alt_p = r_p + const.R_E
        self.alt_a = r_a + const.R_E
        self.i_deg = i_deg
        self.RAAN_deg = RAAN_deg
        self.arg_p_deg = arg_p_deg
        self.i = np.deg2rad(i_deg)
        self.RAAN = np.deg2rad(RAAN_deg)
        self.arg_p = np.deg2rad(arg_p_deg)
        self.e = 0
        self.a = 0
        self.h = 0
        self.T = 0
        self.v_p = 0
        self.v_a = 0
        self.start_time = start_time
        self.end_time = end_time
        self.dv_at_pm = dv_at_pm
        self.compute_orbital_parameters()
        self.propagated_orbit, self.propagated_times = self.propagate_orbit()
        return

    def compute_orbital_parameters(self):
        self.e = eccentricity(self.r_p, self.r_a)
        self.a = semimajor_axis(self.r_p, self.r_a)
        self.h = specific_angular_momentum_from_perigee(self.r_p, self.e)
        self.T = orbital_period(self.a)
        self.v_p = periapsis_velocity(self.h, self.r_p)
        self.v_a = apoapsis_velocity(self.h, self.r_a)
        return

    def set_start_time(self, start_time):
        self.start_time = start_time
        return
    
    def set_end_time(self, end_time):
        self.end_time = end_time
        return
    
    def set_dv_at_pm(self, dv_at_pm):
        self.dv_at_pm = dv_at_pm
        return

    def propagate_orbit(self):
        r_perifocal = perifocal_position_vector(self.h, self.e, 0)
        v_perifocal = perifocal_velocity_vector(self.h, self.e, 0)
        QXx = euler_angle_sequence(self.arg_p, self.i, self.RAAN)
        QxX = np.transpose(QXx)
        r_inertial = QxX @ r_perifocal
        v_inertial = QxX @ v_perifocal
        # Combine into initial state vector for integrator
        y_0 = np.concatenate((r_inertial, v_inertial))
        
        # Perform integration in the inertial frame
        full_orbit_time = [0, self.T]
        solution = spi.solve_ivp(
            orbit_dynamics, 
            full_orbit_time, 
            y_0, 
            max_step=1, 
            args=(const.mu, )
        )
        propogated_orbit = solution.y
        propogated_times = solution.t
        plot_orbit(propogated_orbit, "orbit")
        return propogated_orbit, propogated_times
    