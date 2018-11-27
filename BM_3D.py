import cv2
import numpy as np
import PSNR
from BM3D_Step1 import BM3D_step1
from BM3D_Step2 import BM3D_step2
import skimage


def main():
	'''
	filename = 'lena.jpg'
	origin_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	noise_image = skimage.util.random_noise(origin_image/256.0, mode='gaussian', seed=42)
	cv2.imwrite("noise_image.jpg", 256.0*noise_image)
	'''

	noise_image = cv2.imread("noise_image.jpg", cv2.IMREAD_GRAYSCALE)
	print("Source Image: ", noise_image.shape)
	'''
	print("Block Matching 3D Step 1:")
	BM3D_step1(noise_image)
	'''

	print("Block Matching 3D Step 2:")
	image = cv2.imread("BM3D_step1.jpg", cv2.IMREAD_GRAYSCALE)
	BM3D_step2(image, noise_image)

if __name__ == '__main__':
	main()