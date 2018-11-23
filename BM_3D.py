import cv2
import numpy as np
import PSNR
from BM3D_Step1 import BM3D_step1


def main():
	filename = 'BM3D_step1.jpg'
	image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	print("Source Image: ", image.shape)
	BM3D_step1(image)

if __name__ == '__main__':
	main()