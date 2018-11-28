import cv2
import numpy as np
import os
from BM3D_Step1 import BM3D_step1
from BM3D_Step2 import BM3D_step2
from BM3D_step1_demo import BM3D_1st_step
from BM3D_step2_demo import BM3D_2nd_step
import skimage


def main():

	filename = 'lena.jpg'
	noise_mode = 's&p'

	origin_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	noise_image = skimage.util.random_noise(origin_image/256.0, mode=noise_mode)#var=0.005

	if os.path.exists(noise_mode+'.jpg'):
		noise_image = cv2.imread(noise_mode+'.jpg', cv2.IMREAD_GRAYSCALE)
	else:
		cv2.imwrite(noise_mode+'.jpg', 256.0*noise_image)
		noise_image = cv2.imread(noise_mode+'.jpg', cv2.IMREAD_GRAYSCALE)

	print("Noise Mode:", noise_mode, "Source Image: ", noise_image.shape)
	# ======================================================================
	# start BM3D denoise
	# ======================================================================
	print("Block Matching 3D Step 1:")
	# BM3D_step1(noise_image)
	BM3D_1st_step(noise_image)

	print("Block Matching 3D Step 2:")
	image = cv2.imread("BM3D_step1_demo.jpg", cv2.IMREAD_GRAYSCALE)
	# BM3D_2nd_step(image, noise_image)
	BM3D_2nd_step(image, noise_image)

if __name__ == '__main__':
	main()