import constants as const
import numpy as np
import scipy.integrate as spi
import plane_transformation as pt

def launch_dynamics(
        t: float,
        state_vector: np.ndarray,
        J2: bool
        ):
    """
    Defines the system dynamics for a satellite during launch, including the time derivatives of position and velocity.

    Parameters:
    -----------
    t : float
        The current time in seconds (required by the solver, though not directly used in the dynamics).
    state_vector : np.ndarray
        A 1D array containing the current state vector, where the first two elements are the position vector [x, y] in meters, 
        and the last two elements are the velocity vector [vx, vy] in meters per second.

    Returns:
    --------
    np.ndarray
        A 1D array containing the time derivatives of the position and velocity vectors, which corresponds to the velocity vector [vx, vy] and 
        the acceleration vector [ax, ay], respectively.
    """
    # Extract State Vector
    r = state_vector[0:3]
    v = state_vector[3:6]

    # System Dynamics
    r_mag = np.linalg.norm(r)
    mu = const.mu_EARTH

    r_dot = v
    p = np.zeros(3)
    if J2:
        p_mag = (3/2) * ((const.J2_EARTH * const.mu_EARTH * const.R_EARTH ** 2 )
                            / r_mag ** 4)
        p[0] = p_mag * (r[0] / r_mag) * (5 * (r[2]**2 / r_mag**2) - 1)
        p[1] = p_mag * (r[1] / r_mag) * (5 * (r[2]**2 / r_mag**2) - 1)
        p[2] = p_mag * (r[2] / r_mag) * (5 * (r[2]**2 / r_mag**2) - 3)




    r_dot = v
    v_dot = -(mu / (r_mag**3)) * r + p

    return np.concatenate((r_dot, v_dot))


def simulate_launch(eccentricity: float,
                    true_anomaly: float,
                    specific_angular_momentum: float,
                    perifocal_eci_matrix: np.ndarray,
                    period: int,
                    J2=False,
                    max_step=1
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    mu = const.mu_EARTH

    # Calculate the position vector in the perifocal frame
    r_perifocal_init, v_perifocal_init = pt.get_perifocal_vectors(eccentricity, true_anomaly, specific_angular_momentum)

    r_ECI_init, v_ECI_init = pt.perifocal_to_ECI(perifocal_eci_matrix, r_perifocal_init, v_perifocal_init)

    # Initial State Vec
    initial_vector = np.concatenate((r_ECI_init, v_ECI_init))

    # Sim time span
    t_span = [0, period]
    
    # Solve the differential equations
    solution = spi.solve_ivp(
        launch_dynamics, 
        t_span, 
        initial_vector, 
        args=(J2,),
        max_step=max_step)
    
    time = solution.t
    position = solution.y[0:3]
    velocity = solution.y[3:6]

    return time, position, velocity