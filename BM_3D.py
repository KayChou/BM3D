import cv2
import numpy as np
import PSNR
from BM3D_Step1 import BM3D_step1
from BM3D_Step2 import BM3D_step2


def main():
	filename = 'BM3D_step1.jpg'
	origin_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	print("Source Image: ", origin_image.shape)

	print("BM3D Step 1:")
	BM3D_step1(origin_image)
	
	print("BM3D Step 2:")
	image = cv2.imread("BM3D_step1.jpg", cv2.IMREAD_GRAYSCALE)
	BM3D_step2(image, origin_image)

if __name__ == '__main__':
	main()