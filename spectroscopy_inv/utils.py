import numpy as np


def absorbance_to_coeffs(A):
    c1 = (A[1] + A[2])/2
    c2 = (A[2] + A[0])/2
    c3 = (A[0] + A[1])/2
    return np.array([c1, c2, c3])

def coeffs_to_absorbance(c):
    A1 = c[1] + c[2] - c[0]
    A2 = c[2] + c[0] - c[1]
    A3 = c[0] + c[1] - c[2]
    return np.array([A1, A2, A3])

def spherical_to_cartesian(r, theta, phi):
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

