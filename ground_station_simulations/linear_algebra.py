import numpy as np

def angle_between_vectors(v1, v2):
    """
    Calculate the angle (in radians) between two vectors using the dot product.
    
    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.
        
    Returns:
        float: Angle in radians between the vectors.
    """
    # Normalize the vectors (or get their magnitudes)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Ensure the vectors are not zero-length
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("One of the vectors has zero length.")
    
    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Clip the cosine value to avoid numerical issues with acos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(cos_theta)
    
    return angle

def rotate_about(vector: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate a vector around a given axis by a specified angle using Rodrigues' rotation formula.

    Args:
        vector (np.ndarray): The vector to be rotated.
        axis (np.ndarray): The axis to rotate around (should be normalized).
        theta (float): The angle of rotation in radians.

    Returns:
        np.ndarray: The rotated vector.
    """
    # Ensure the axis is normalized
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula components
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)

    # Compute the rotation matrix using Rodrigues' formula
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    # Rotate the vector
    return R @ vector