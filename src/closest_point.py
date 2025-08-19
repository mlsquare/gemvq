
import numpy as np

"""
Closest Point Algorithm for Z_n Lattice.
"""

def custom_round(x):
    """
    Custom rounding function that handles edge cases for lattice quantization.
    
    This function implements a rounding scheme that ensures consistent behavior
    at boundary points (0.5) for lattice quantization algorithms.
    
    Parameters:
    -----------
    x : float or numpy.ndarray
        The value(s) to be rounded. Can be a scalar or array.
        
    Returns:
    --------
    float or numpy.ndarray
        The rounded value(s) using the custom rounding scheme.
        
    Notes:
    ------
    For positive values: rounds down if fractional part is exactly 0.5, 
    otherwise rounds to nearest integer.
    For negative values: rounds up if fractional part is exactly 0.5,
    otherwise rounds to nearest integer.
    """
    if isinstance(x, np.ndarray):
        return np.array([custom_round(val) for val in x])
    else:
        if x > 0:
            return np.floor(x + 0.5) - 1 if (x - np.floor(x)) == 0.5 else np.floor(x + 0.5)
        else:
            return np.ceil(x - 0.5) + 1 if (np.ceil(x) - x) == 0.5 else np.ceil(x - 0.5)


"""
Closest Point Algorithm for the D_n Lattice.
The lattice D_n consists of points where the sum of components is even.
"""

def g_x(x):
    """
    Compute g(x) by rounding the vector x to the nearest integers,
    but flip the rounding for the coordinate farthest from an integer.
    
    This is a helper function for the D_n lattice closest point algorithm.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input vector to be processed.
        
    Returns:
    --------
    numpy.ndarray
        Modified rounded vector with one coordinate flipped to ensure
        the sum of components has the desired parity.
        
    Notes:
    ------
    This function is used in the D_n lattice algorithm to find an alternative
    rounding when the standard rounding doesn't satisfy the D_n lattice constraints.
    """
    f_x = custom_round(x)
    delta = np.abs(x - f_x)
    k = np.argmax(delta)
    g_x_ = f_x.copy()

    if x[k] >= 0:
        g_x_[k] = f_x[k] + 1 if f_x[k] < x[k] else f_x[k] - 1
    else:
        g_x_[k] = f_x[k] + 1 if f_x[k] <= x[k] else f_x[k] - 1

    return g_x_


def closest_point_Dn(x):
    """
    Find the closest point in the D_n lattice for a given vector x.
    
    The D_n lattice consists of all integer points where the sum of coordinates
    is even. This algorithm finds the closest such point to the input vector x.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input vector of dimension n.
        
    Returns:
    --------
    numpy.ndarray
        The closest point in the D_n lattice to x.
        
    Notes:
    ------
    The algorithm works by:
    1. Rounding x to the nearest integer vector
    2. If the sum is even, this is the closest point
    3. If the sum is odd, flip the rounding for the coordinate farthest
       from an integer to get a valid D_n lattice point
       
    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding 
    algorithms for lattice quantizers and codes. IEEE Transactions on 
    Information Theory, 28(2), 227-232.
    """
    f_x = custom_round(x)
    g_x_res = g_x(x)
    return f_x if np.sum(f_x) % 2 == 0 else g_x_res


"""
Closest Point Algorithm for the E_8 Lattice.
The lattice E_8 is constructed from D_8 and a coset.
"""

def closest_point_E8(x):
    """
    Find the closest point in the E_8 lattice for a given vector x.
    
    The E_8 lattice is an 8-dimensional lattice that can be constructed
    from the D_8 lattice and a coset. This algorithm finds the closest
    E_8 lattice point to the input vector.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input vector of dimension 8.
        
    Returns:
    --------
    numpy.ndarray
        The closest point in the E_8 lattice to x.
        
    Notes:
    ------
    The E_8 lattice is constructed as the union of D_8 and D_8 + (0.5)^8.
    The algorithm:
    1. Finds the closest point in D_8 to x
    2. Finds the closest point in D_8 + (0.5)^8 to x
    3. Returns the closer of the two points
        
    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding 
    algorithms for lattice quantizers and codes. IEEE Transactions on 
    Information Theory, 28(2), 227-232.
    """
    y_0 = custom_round(x) if np.sum(custom_round(x)) % 2 == 0 else g_x(x)

    f_x_shifted = custom_round(x - 0.5)
    g_x_shifted = g_x(x - 0.5)

    y_1 = f_x_shifted + 0.5 if np.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5

    if np.linalg.norm(x - y_0) < np.linalg.norm(x - y_1):
        return y_0
    else:
        return y_1


def upscale(u):
    """
    Upscale a 2D vector to 3D space for A_2 lattice processing.
    
    This function transforms a 2D vector into 3D space using a specific
    transformation matrix that preserves the A_2 lattice structure.
    
    Parameters:
    -----------
    u : numpy.ndarray
        Input 2D vector.
        
    Returns:
    --------
    numpy.ndarray
        3D vector obtained by applying the upscaling transformation.
        
    Notes:
    ------
    The transformation matrix M is:
    [[1, 0, -1],
     [1/√3, -2/√3, 1/√3]]
    """
    M = np.array([[1, 0, -1], [1/np.sqrt(3), -2/np.sqrt(3), 1/np.sqrt(3)]])
    return np.dot(u, M)


def downscale(x):
    """
    Downscale a 3D vector back to 2D space for A_2 lattice processing.
    
    This function transforms a 3D vector back to 2D space using the
    transpose of the upscaling transformation matrix.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input 3D vector.
        
    Returns:
    --------
    numpy.ndarray
        2D vector obtained by applying the downscaling transformation.
        
    Notes:
    ------
    This is the inverse operation of upscale(), using the transpose
    of the transformation matrix multiplied by 0.5.
    """
    M_t = np.array([[1, 0, -1], [1 / np.sqrt(3), -2 / np.sqrt(3), 1 / np.sqrt(3)]]).T
    return 0.5 * np.dot(x, M_t)


"""
Closest Point Algorithm for the A_2 Lattice.
"""

def closest_point_A2(u):
    """
    Find the closest point in the A_2 lattice for a given vector u.
    
    The A_2 lattice is a 2D hexagonal lattice. This algorithm finds
    the closest A_2 lattice point to the input vector using a 3D
    embedding approach.
    
    Parameters:
    -----------
    u : numpy.ndarray
        Input 2D vector.
        
    Returns:
    --------
    numpy.ndarray
        The closest point in the A_2 lattice to u.
        
    Notes:
    ------
    The algorithm works by:
    1. Upscaling the 2D vector to 3D space
    2. Finding the closest point in the 3D lattice with sum constraint
    3. Downscaling the result back to 2D space
        
    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding 
    algorithms for lattice quantizers and codes. IEEE Transactions on 
    Information Theory, 28(2), 227-232.
    """
    x = upscale(u)
    s = np.sum(x)
    x_p = x - (s / len(x)) * np.array([1, 1, 1])
    f_x_p = custom_round(x_p)
    delta = int(np.sum(f_x_p))

    distances = x - f_x_p
    sorted_indices = np.argsort(distances)

    if delta == 0:
        return downscale(f_x_p)
    elif delta > 0:
        for i in range(delta):
            f_x_p[sorted_indices[i]] -= 1
    elif delta < 0:
        for i in range(-delta):
            f_x_p[sorted_indices[-i-1]] += 1

    return downscale(f_x_p)
