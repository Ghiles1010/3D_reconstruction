import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import calculate as clc
import operations as op
import cv2
from copy import copy



def next_z_move_x(origin, normal, l=0.5):
	
	v = op.get_tangant(normal, 1)
	v[1] = 0
	theta = op.angle_between(v, np.array([1, 0, 0]))
	z = np.tan(theta) * l

	z = -z if v[2] < 0 else z

	z = origin[2] + z
	return z


def create_3d_model():
	pixel_coords, mask, normals = clc.get_normal()
	crop_dims = clc.get_mask_info(mask)

	mask = clc.crop(mask, crop_dims)
	normals = clc.crop(normals, crop_dims)

	h, w, _ = normals.shape
	step = 3

	cv2.imshow("", mask)
	cv2.imshow("a", normals)

	points, colors = [], []
	for i in range(0, h, step):
		origin = np.array([i,0,0])

		for j in range(0, w, step):
			#if (mask[i,j]).all() != 0 :
			origin[1]=j
			z = next_z_move_x(origin, normals[i,j], step)
			new_point = np.array([i, j, 0])
			points.append(new_point)
			origin = np.array(new_point)
			colors.append(clc.get_grey_pixel(normals[i,j]))




	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	points = np.array(points).astype(np.uint8)

	X, Y, Z = points[:,0], points[:, 1], points[:, 2]

	ax.scatter(X, Y, Z, c=colors)
	#ax.plot_trisurf(X, Y, Z)


	plt.show()



def main():
	create_3d_model()


if __name__ == '__main__':
	main()
	