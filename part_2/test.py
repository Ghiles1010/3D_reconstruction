from scipy.spatial.transform import Rotation
from numpy.linalg import norm
import numpy as np

def get_tangant(vec, axis):

	rotation_degrees = 90
	rotation_radians = np.radians(rotation_degrees)
	rotation_axis = np.zeros(3)
	rotation_axis[axis] = 1

	rotation_vector = rotation_radians * rotation_axis
	rotation = Rotation.from_rotvec(rotation_vector)
	rotated_vec = rotation.apply(vec)

	return rotated_vec

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

v = np.array([0.54,0,0])
e = np.array([1,0,0])
print(angle_between(v,e))