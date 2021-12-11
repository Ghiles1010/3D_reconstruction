from utils import printProgressBar

import os
from numpy import genfromtxt
import numpy as np
import cv2

LIGHT_SOURCE = "light_directions.txt"
LIGHT_INTENSITY = "light_intensities.txt"
MASK = "mask.png"
FILE_NAMES = "filenames.txt"
PATH = "dataset/"

TREATED_IMAGES_DIR = "treated_images/"
TREATED_IMAGES_FILE = "treated_images/new_images.npy"
PIXELS_COORDS_PATH = "treated_images/pixels_coords.txt"


def load_light_source():
	return np.genfromtxt(get_path(LIGHT_SOURCE), delimiter=' ')

def load_light_intensity():
	return np.genfromtxt(get_path(LIGHT_INTENSITY), delimiter=' ')

def load_mask():
	return cv2.imread(get_path(MASK), cv2.IMREAD_UNCHANGED)

def get_concerened_pixels(mask):
	return list(zip(*np.where(mask!=0)))

def get_mask_info(mask):
	x, y = np.where(mask!=0)
	return x.min(), x.max(), y.min(), y.max()

def get_grey_pixel(pixel):
	coefficients = [0.3, 0.59, 0.11]
	return np.dot(pixel, coefficients)


def crop(image, dims):
	x_min, x_max, y_min, y_max = dims
	return image[x_min : x_max, y_min : y_max]

def RGB2BGR(pixel):
	return pixel[::-1]

def get_path(file):
	global PATH
	return os.path.join(PATH, file)

def menu():
	global PATH
	PATH = input("Entrez le path du dossier \"dataset\" : ")


def load_images():

	menu()

	if os.path.isfile(TREATED_IMAGES_FILE) :

		images = np.load(TREATED_IMAGES_FILE)
		pixels_coords = []
		with open(PIXELS_COORDS_PATH) as file:
			pixels_coords = file.read().rstrip().split("\n")

		for idx, pix in enumerate(pixels_coords):
			i, j = pix.split(',')
			pixels_coords[idx] = (int(i),int(j))

		return np.array(images), pixels_coords

	else :

		images = []
		with open(get_path(FILE_NAMES)) as file:
			file_names = file.read().rstrip().split("\n")

		for file in file_names:
			img = cv2.imread(get_path(file), cv2.IMREAD_UNCHANGED)
			img = img / 65535.0
			images.append(img)


	intensities = load_light_intensity()

	images = np.array(images)

	_, h, w, _ = images.shape
	new_images = []
	pixels_coords =  get_concerened_pixels(load_mask())
	l = len(intensities)

	print("Normalisation des images :")

	for idx, intens in enumerate(intensities):
		new_image = np.zeros((h,w))
		for i, j in pixels_coords:
			images[idx, i, j] = np.divide(images[idx, i, j], RGB2BGR(intens))
			new_image[i,j] = get_grey_pixel(images[idx, i, j])

		new_images.append(new_image)
		printProgressBar(idx+1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

	
	os.makedirs(os.path.dirname(TREATED_IMAGES_DIR), exist_ok=True)

	new_images = np.array(new_images)

	np.save(TREATED_IMAGES_FILE, new_images)

	with open(PIXELS_COORDS_PATH, 'w') as file:
		for i, j in pixels_coords:
			file.write("{i},{j}\n".format(i=i, j=j))


	return np.array(new_images), pixels_coords





def calculate_normal(images, pixels_coords):

	np.seterr(divide='ignore', invalid='ignore')

	S_i = np.linalg.pinv(load_light_source())

	_, h, w = images.shape
	image = np.zeros((h,w,3))

	for i, j in pixels_coords:
		E = images[:,i,j]
		n = np.dot(S_i, E)
		n = n / np.linalg.norm(n)
		image[i,j] = n 

	return image





def get_normal():
	images, pixels_coords = load_images()
	normals = calculate_normal(images, pixels_coords)
	mask = load_mask()
	return pixels_coords, mask, normals





def main():

	images, pixels_coords = load_images()
	normals = calculate_normal(images, pixels_coords)
	
	cv2.imshow("", normals)
	cv2.waitKey(0)


if __name__ == "__main__":

	_,mask,_ = get_normal()

	dims = get_mask_info(mask)
	mask = crop(mask, dims)

	cv2.imshow("", mask)
	cv2.waitKey(0)
	#main()