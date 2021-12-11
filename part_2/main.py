import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import calculate as clc
import operations as op
import cv2
from copy import copy



def next_z_move_x(z_origin, normal, l=0.5):
	
	v = op.get_tangant(normal, 1)
	v[1] = 0
	theta = op.angle_between(v, np.array([1, 0, 0]))
	z = np.tan(theta) * l

	z = -z if v[2] < 0 else z

	z = z_origin + z
	return z


def create_3d_model():
	pixel_coords, mask, normals = clc.get_normal()
	crop_dims = clc.get_mask_info(mask)

	mask = clc.crop(mask, crop_dims)
	normals = clc.crop(normals, crop_dims)

	h, w, _ = normals.shape
	step = 3

	points, colors = [], []

	
	for i in range(0, h, step):
		z_origin = 0
		for j in range(0, w, step):
			if mask[i,j] != 0 :
				z = next_z_move_x(z_origin, normals[i,j], step)
				new_point = np.array([i, j, z])
				points.append(new_point)
				z_origin = z
				colors.append(clc.get_grey_pixel(normals[i,j]))


	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	points = np.array(points, np.float16)

	X, Y, Z = points[:,0], points[:, 1], points[:, 2]

	ax.scatter(X, Y, Z, c=colors)

	plt.show()



def main():
	create_3d_model()


if __name__ == '__main__':
	main()
	