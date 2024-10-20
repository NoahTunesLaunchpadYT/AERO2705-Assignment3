import numpy as np
import math
import constants as const
import datetime as datetime

def perifocal_to_ECI_matrix(RAAN: float, inclination: float, argument_of_perigee: float) -> np.ndarray:
    """
    Calculate the transformation matrix from perifocal to the ECI frame.

    Parameters:
        RAAN (float): Right Ascension of the Ascending Node in radians.
        inclination (float): Inclination of the orbit in radians.
        argument_of_perigee (float): Argument of Perigee in radians.

    Returns:
        np.ndarray: The transformation matrix from ECI to the perifocal frame.
    """
    # Calculate the transformation matrix from ECI to the perifocal frame
    Q_xX = np.array([
                    [(-math.sin(RAAN) * math.cos(inclination) * math.sin(argument_of_perigee)) + (math.cos(RAAN) * math.cos(argument_of_perigee)),
                    (-math.sin(RAAN) * math.cos(inclination) * math.cos(argument_of_perigee)) - (math.cos(RAAN) * math.sin(argument_of_perigee)),
                    math.sin(RAAN) * math.sin(inclination)
                    ],
                    [(math.cos(RAAN) * math.cos(inclination) * math.sin(argument_of_perigee)) + (math.sin(RAAN) * math.cos(argument_of_perigee)),
                    (math.cos(RAAN) * math.cos(inclination) * math.cos(argument_of_perigee)) - (math.sin(RAAN) * math.sin(argument_of_perigee)),
                    -math.cos(RAAN) * math.sin(inclination)
                    ],
                    [math.sin(inclination) * math.sin(argument_of_perigee),
                    math.sin(inclination) * math.cos(argument_of_perigee),
                    math.cos(inclination)
                    ]
                ]
                )
    return Q_xX


def get_perifocal_vectors(eccentricity: float,
                          true_anomaly: float,
                          specific_angular_momentum: float, 
                          ) -> np.ndarray:
   
    """
    Calculate the position and velocity vectors in the perifocal frame.

    Parameters:
        eccentricity (float): Eccentricity of the orbit.
        true_anomaly (float): True Anomaly in radians.
        specific_angular_momentum (float): Specific Angular Momentum of the orbit.
    
    Returns:
        np.ndarray: Position and velocity vectors in the perifocal frame.
    """
    # Calculate the radial distance
    r_mag = ((specific_angular_momentum ** 2)/ const.mu_EARTH ) * (1 / (1 + eccentricity * np.cos(true_anomaly)))
    

    # Calculate the position vector in the perifocal frame
    r_p = r_mag * np.cos(true_anomaly)
    r_q = r_mag * np.sin(true_anomaly)

    r_perifocal = np.array([r_p, r_q, 0])

    v_p = (const.mu_EARTH / specific_angular_momentum) * (-math.sin(true_anomaly))
    v_q = (const.mu_EARTH / specific_angular_momentum) * (eccentricity + math.cos(true_anomaly))

    # v_p = (r_dot * np.cos(true_anomaly)) - (r_theta_dot * np.sin(true_anomaly))
    # v_q = (r_dot * np.sin(true_anomaly)) + (r_theta_dot * np.cos(true_anomaly))

    v_perifocal = np.array([v_p, v_q, 0])


    return r_perifocal, v_perifocal



def perifocal_to_ECI(Q_xX: np.ndarray, r_perifocal: np.ndarray, v_perifocal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform the position and velocity vectors from the perifocal frame to the ECI frame.

    Parameters:
        Q_xX (np.ndarray): Transformation matrix from perifocal to ECI frame.
        r_perifocal (np.ndarray): Position vector in the perifocal frame.
        v_perifocal (np.ndarray): Velocity vector in the perifocal frame.

    Returns:
        tuple[np.ndarray, np.ndarray]: Position and velocity vectors in the ECI frame.
    """
    # Transform the position vector to the ECI frame
    r_ECI = Q_xX @ r_perifocal

    # Transform the velocity vector to the ECI frame
    v_ECI = Q_xX @ v_perifocal

    return r_ECI, v_ECI


def propagate_orbit(Q_xX, r_perifocal, v_perifocal, e, h, period=1, num_steps=1000):
    """
    Propagates the orbit over one period using true anomaly steps.
    
    Args:
    Q_xX (np.array): Transformation matrix from perifocal to ECI frame.
    r0 (np.array): Initial position vector in perifocal frame (in km).
    v0 (np.array): Initial velocity vector in perifocal frame (in km/s).
    T (float): Orbital period in seconds.
    num_steps (int): Number of steps to calculate over one period.
    
    Returns:
    np.array: Array of ECI positions over the orbit (n x 3).
    """
    
    # Generate an array of true anomalies from 0 to 2π
    theta = np.linspace(0, period* 2* np.pi, num_steps)
    
    # Initialize an array to store the ECI positions
    positions = np.zeros((num_steps, 3))
    velocities = np.zeros((num_steps, 3))
    
    for i, true_anomaly in enumerate(theta):
        # Get the perifocal position and velocity for the current true anomaly
        r_perifocal, v_perifocal = get_perifocal_vectors(e, true_anomaly, h)

        r_ECI, v_ECI = perifocal_to_ECI(Q_xX, r_perifocal, v_perifocal)
        
        # Store the ECI position
        positions[i] = r_ECI
        velocities[i] = v_ECI

    return positions, velocities


def gregorian_to_julian(day: int,
                        month: int,
                        year: int,
                        hour = 0,
                        minutes = 0,
                        seconds = 0) -> float:
    """
    Convert a Gregorian date to Julian date.
    
    Parameters:
        year (int): Year.
        month (int): Month (1 - 12)
        day (int): Day (1-31)
        hour (int): Hour (0 - 23), (default 0)
        minutes (int): Minutes (0-59), (default 0)
        seconds (int): Seconds (0-59), (default 0).
    """

    # Account for Jan and Feb
    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = A // 4
    C = 2 - A + B
    E = int(365.25 * (year + 4716))
    F = int(30.6001 * (month + 1))

    JD = C + day + E + F - 1524.5

    return JD

def julian_centruty(julian_date: float) -> float:
    """
    Calculate the Julian Centuries from the Julian Date.
    
    Parameters:
        julian_date (float): Julian Date.
    
    Returns:
        float: Julian Centuries.
    """
    T = (julian_date - 2451545.0) / 36525.0
    return T

def julian_to_GMST(julian_date: float) -> float:
    """
    Calculate the Greenwich Mean Sidereal Time (GMST) from the Julian Centuries.
    
    Parameters:
        julian_century (float): Julian Centuries.
    
    Returns:
        float: Greenwich Mean Sidereal Time in radians.
    """
    JD = julian_date

    # Calculate Julian Century
    T = julian_centruty(julian_date)

    # Calculate the Greenwich Mean Sidereal Time
    GMST = (280.46061837 + 360.98564736629 * (JD - 2451545.0) +
            0.000387933 * T**2 - (T**3 / 38710000.0))

    # Reduce GMST to between 0-360 deg
    GMST = GMST % 360

    # Convert to radians
    theta_GMST = np.radians(GMST)

    return theta_GMST

def GMST(date_time: datetime) -> float:
    """
    Calculate the Greenwich Sidereal Time (GST) from the Julian Date.
    
    Parameters:
        julian_date (float): Julian Date.
    
    Returns:
        float: Greenwich Sidereal Time in radians.
    """

    year = date_time.year
    month = date_time.month
    day = date_time.day
    hour = date_time.hour
    minute = date_time.minute
    second = date_time.second


    julian_date = gregorian_to_julian(day, month, year, hour, minute, second)

    # Calculate the Julian Centuries
    T = julian_centruty(julian_date)

    # Calculate the Greenwich Sidereal Time
    theta_GST = 67310.54841 + (876600 * 3600 + 8640184.812866) * T + 0.093104 * T ** 2 - 6.2e-6 * T ** 3

    # Convert to radians
    theta_GST = np.radians(theta_GST % 86400 * 360 / 86400)

    return theta_GST


def eci_to_ecef_matrix(theta) -> np.ndarray:
    """
    Calculate the transformation matrix from the ECI to the ECEF frame.

    Returns:
        np.ndarray: The transformation matrix from ECI to the ECEF frame.
    """
    # Calculate the transformation matrix from ECI to the ECEF frame
    Q_ECI_ECEF = np.array([
                    [np.cos(theta), np.sin(theta),  0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0,              0,             1]
                ]
                )
    return Q_ECI_ECEF

def eci_to_ecef(Q_ECI_ECEF: np.ndarray, r_ECI: np.ndarray) -> np.ndarray:
    """
    Transform the position vector from the ECI to the ECEF frame.

    Parameters:
        Q_ECI_ECEF (np.ndarray): Transformation matrix from ECI to ECEF frame.
        r_ECI (np.ndarray): Position vector in the ECI frame.

    Returns:
        np.ndarray: Position vector in the ECEF frame.
    """
    # Transform the position vector to the ECEF frame
    r_ECEF = Q_ECI_ECEF @ r_ECI

    return r_ECEF



def ecef_to_longitude(r_ECEF: np.ndarray) -> tuple[float, float]:
    """
    Calculate the longitude from the ECEF position vector.

    Parameters:
        r_ECEF (np.ndarray): Position vector in the ECEF frame.

    Returns:
        tuple[float, float]: Longitude in radians.
    """
    x_ECEF = r_ECEF[0]
    y_ECEF = r_ECEF[1]

    # Calculate the longitude
    longitude = np.arctan2(y_ECEF, x_ECEF)

    return longitude

def ecef_to_latitude(r_ECEF: np.ndarray, semimajor_axis: float) -> float:
    """
    Calculate the latitude from the ECEF position vector iteratively.

    Parameters:
        r_ECEF (np.ndarray): Position vector in the ECEF frame.
        semimajor_axis (float): Earth's semimajor axis (in km).

    Returns:
        float: Latitude in radians.
    """
    x_ECEF = r_ECEF[0]
    y_ECEF = r_ECEF[1]
    z_ECEF = r_ECEF[2]

    # Constants
    a = semimajor_axis  # Semimajor axis of Earth (around 6378.137 km)
    f = const.f_EARTH   # Flattening factor
    e2 = 2 * f - f ** 2 # First eccentricity squared
    b = a * (1 - f)     # Semiminor axis

    # Distance from the z-axis
    p = math.sqrt(x_ECEF ** 2 + y_ECEF ** 2)

    # Initial latitude estimate (geocentric latitude)
    latitude = np.arctan2(z_ECEF, p * (1 - e2))
    

    # Iterative refinement of the latitude
    delta_lat = 1e-12  # Convergence threshold
    diff = 1  # Difference between iterations
    iteration_count = 0  # To limit iterations

    while diff > delta_lat and iteration_count < 100:
        sin_lat = np.sin(latitude)
        N = a / math.sqrt(1 - e2 * sin_lat ** 2)  # Radius of curvature in the prime vertical
        new_latitude = np.arctan2(z_ECEF + e2 * N * sin_lat, p)
        diff = np.abs(new_latitude - latitude)
        latitude = new_latitude
        iteration_count += 1

    # If the iteration did not converge, raise a warning
    if iteration_count == 100:
        print("Warning: Latitude iteration did not converge.")


    return latitude


def j2_RAAN_dot(a, e, i):
    """
    Calculate the rate of change of the RAAN (Ω) due to the J2 perturbation.
    
    Parameters:
    a (float): Semi-major axis in km.
    e (float): Eccentricity of the orbit.
    i (float): Inclination in radians.
    
    Returns:
    float: Rate of change of RAAN in radians per second.
    """
    J2 = const.J2_EARTH  # Earth's J2 coefficient
    R_earth = const.R_EARTH  # Earth's radius in km
    mu = const.mu_EARTH  # Earth's gravitational parameter in km³/s²
    
    n = np.sqrt(mu / a**3) 

    raan_dot = (-3/2) * J2 * (R_earth**2 / (a**2 * (1 - e**2)**2)) * n * np.cos(i)

    return raan_dot

def j2_omega_dot(a, e, i):
    """
    Calculate the rate of change of the argument of perigee (ω) due to the J2 perturbation.
    
    Parameters:
    a (float): Semi-major axis in km.
    e (float): Eccentricity of the orbit.
    i (float): Inclination in radians.
    
    Returns:
    float: Rate of change of the argument of perigee in radians per second.
    """
    J2 = const.J2_EARTH  # Earth's J2 coefficient
    R_earth = const.R_EARTH  # Earth's radius in km
    n = np.sqrt(const.mu_EARTH / a**3)  # Mean motion

    omega_dot = (3/4) * J2 * (R_earth**2 / (a**2 * (1 - e**2)**2)) * n * (5 * np.cos(i)**2 - 1)



    return omega_dot


def propagate_orbit_with_J2(e, h, a, i, RAAN, w, n, num_days=7, dt=100):
    """
    Propagates the orbit over a given number of days, including J2 perturbation effects.

    Parameters:
    e (float): Eccentricity of the orbit.
    h (float): Specific angular momentum.
    a (float): Semi-major axis of the orbit in km.
    i (float): Inclination of the orbit in radians.
    RAAN (float): Initial RAAN in radians.
    w (float): Initial argument of perigee in radians.
    n (float): Mean motion in rad/s.
    num_days (int): Number of days to propagate the orbit.
    dt (int): Time step in seconds.

    Returns:
    np.array: Array of ECI positions over the orbit (n x 3).
    """
    num_steps = int(num_days * 24 * 60 * 60 / dt)

    positions = np.zeros((num_steps, 3))

    # Generate an array of true anomalies from 0 to 2π
    theta = np.linspace(0, 2*np.pi, num_steps)

    RAAN_ = RAAN
    w_ = w
    
    for i, true_anomaly in enumerate(theta):
        
        # Calculate the rates of change for RAAN and argument of perigee due to J2
        RAAN_dot = j2_RAAN_dot(a, e, i, n)
        omega_dot = j2_omega_dot(a, e, i, n)
        
        # Update RAAN and argument of perigee over time
        RAAN_ += RAAN_dot * dt
        w_ += omega_dot * dt

        RAAN_ = RAAN_ % (2 * np.pi)
        w_ = w_ % (2 * np.pi)
        
        # Update the transformation matrix based on the new RAAN and argument of perigee
        Q_xX = perifocal_to_ECI_matrix(RAAN_, i, w_)
        
        # Get the perifocal position and velocity based on the current true anomaly
        r_perifocal, v_perifocal = get_perifocal_vectors(e, true_anomaly, h)

        # Convert perifocal to ECI using the updated transformation matrix
        r_ECI, _ = perifocal_to_ECI(Q_xX, r_perifocal, v_perifocal)
        
        # Store the ECI positions
        positions[i] = r_ECI
        print(r_ECI)


    return positions



def get_fuel_consumed(I_sp: float, delta_v: float, m_initial=1) -> float:
    """
    Calculates the fuel consumpiton,

    Parameters:
        I_sp (float): Specific impulse of the thruster in seconds.
        delta_v (float): Change in velocity in m/s.
        m_i (float): Initial mass of the spacecraft in kg (default 1 kg).
    
    Returns:
        float: Fuel consumed in kg.
    """

    delta_m = (1 - math.exp(-delta_v / (I_sp * const.g_EARTH))) * m_initial

    return delta_m 

    


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Returns a rotation matrix that rotates a vector by `angle` radians around `axis`.
    The axis must be a normalized vector.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    # Components of the axis vector
    x, y, z = axis

    # Construct the rotation matrix using the axis-angle formula
    rotation_matrix = np.array([
        [cos_angle + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_angle, x * z * one_minus_cos + y * sin_angle],
        [y * x * one_minus_cos + z * sin_angle, cos_angle + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_angle],
        [z * x * one_minus_cos - y * sin_angle, z * y * one_minus_cos + x * sin_angle, cos_angle + z * z * one_minus_cos]
    ])

    return rotation_matrix