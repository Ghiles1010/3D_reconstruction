import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import calculate as clc
import operations as op
import cv2
from copy import copy
import sys


def next_z_move_x(z_origin, normal, l=0.5):
	
	v = op.get_tangant(normal, 1)
	v[1] = 0
	theta = op.angle_between(v, np.array([1, 0, 0]))
	z = np.tan(theta) * l

	z = -z if v[2] < 0 else z

	z = z_origin + z
	return z

def add_point(i, j, z, colors, points, normals):
	new_point = np.array([i, j, z])
	colors.append(clc.get_grey_pixel(normals[i,j]))
	points.append(new_point)


def create_3d_model(test):
	pixel_coords, mask, normals = clc.get_normal(test)
	crop_dims = clc.get_mask_info(mask)

	mask = clc.crop(mask, crop_dims)
	normals = clc.crop(normals, crop_dims)

	h, w, _ = normals.shape
	step = 3

	points, colors = [], []

	for i in range(0, h, step):
		first_in_row = True
		for j in range(0, w, step):
			if mask[i,j] != 0 :

				if first_in_row :
					first_in_row, z = False, 0
					add_point(i, j, z, colors, points, normals)

				else:
					z = next_z_move_x(z_origin, normals[i,j-step], step)
					add_point(i, j, z, colors, points, normals)

				z_origin = z


	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	points = np.array(points, np.float16)

	X, Y, Z = points[:,0], points[:, 1], points[:, 2]

	ax.scatter(X, Y, Z, c=colors)

	plt.show()



def main():

	test = True if len(sys.argv) == 2 and sys.argv[1] == 't' else False

	create_3d_model(test)


if __name__ == '__main__':

	
	main()
	