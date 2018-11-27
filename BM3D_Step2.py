import cv2
import numpy as np
from BM3D_Step1 import coord_boundary, get_search_window

block_size = 8
search_window_size = 39
search_step = 3
match_threshold = 20
max_match_block = 2
beta_Kaiser = 2.0
sigma = 25


# ==============================================================
# Calculate the other image block and its similarity
# and join the group if the similarity reaches the threshold
# ==============================================================
def block_matching(image, noise_image, block_coord):
	x, y = block_coord
	# DCT transform for image processed in step1
	img_block = image[x:x+block_size, y:y+block_size]
	img_block_dct = cv2.dct(img_block.astype(np.float64))

	# DCT transform for raw image with noise
	noise_img_block = noise_image[x:x+block_size, y:y+block_size]
	noise_img_block_dct = cv2.dct(noise_img_block.astype(np.float64))

	blk_num = int((search_window_size - block_size)/search_step)
	window_x, window_y = get_search_window(image, block_coord)

	# init variables to be returned
	similar_blks_dct = np.zeros([blk_num**2, block_size, block_size])
	similar_blks_pos = np.zeros([blk_num**2, 2], dtype=int)
	blk_distances = np.zeros(blk_num**2)

	match_index = 0
	for i in range(blk_num):
		x = window_x + i*search_step
		y = window_y
		for j in range(blk_num):
			y = window_y + j*search_step
			temp_blk = image[x:x+block_size, y:y+block_size]
			temp_blk_dct = cv2.dct(temp_blk.astype(np.float64))

			distance = np.linalg.norm(img_block_dct - temp_blk_dct)
			#print(distance)

			# Threshold filtering
			if distance < match_threshold:
				similar_blks_dct[match_index, :, :] = temp_blk_dct
				similar_blks_pos[match_index, :] = x, y
				blk_distances[match_index] = distance
				match_index = match_index + 1

	# Because OpenCV cannot do odd-numbered DCT transforms
	# so need to limit the value of w
	if match_index > max_match_block:
		match_index = max_match_block

	if match_index%2 == 1:
		match_index = match_index - 1
	if match_index == 0:
		match_index = 2

	# blk_distances = blk_distances[0:match_index]
	sort = blk_distances.argsort()

	# init variables to be returned
	final_blks_dct = np.zeros([match_index, block_size, block_size])
	final_noise_blks = np.zeros([match_index, block_size, block_size])
	final_blks_pos = np.zeros([match_index, 2], dtype=int)

	final_blks_dct[0, :, :] = img_block_dct
	final_noise_blks[0, :, :] = noise_img_block_dct
	final_blks_pos[0, :] = block_coord 

	for i in range(1, match_index):
		final_blks_dct[i, :, :] = similar_blks_dct[sort[i-1], :, :]

		x, y = similar_blks_pos[sort[i-1], :]
		noise_blk = noise_image[x:x+block_size, y:y+block_size]
		final_noise_blks[i, :, :] = cv2.dct(noise_blk.astype(np.float64))

		final_blks_pos[i, :] = similar_blks_pos[sort[i-1], :]

	return final_blks_dct, final_noise_blks, final_blks_pos, match_index+1


# =======================================================-======
# 3D Wiener filtering
# ==============================================================
def wiener_filter(similar_blks_dct, noise_blk_dct):
	shape = similar_blks_dct.shape
	wiener_weight = np.zeros((shape[1], shape[2]), dtype=float)

	for i in range(shape[1]):
		for j in range(shape[2]):
			correspond_pixel = similar_blks_dct[:, i, j]
			pixel_dct = np.matrix(cv2.dct(correspond_pixel))
			norm2 = np.float(pixel_dct.T * pixel_dct)
			# print(norm2)
			# norm2 = np.linalg.norm(pixel_dct, 2)
			weight = norm2 / (norm2 + sigma**2)
			if weight != 0:
				wiener_weight[i, j] = 1/(weight**2 + sigma**2)

			# process origin image(noise_blk_dct)
			temp = noise_blk_dct[:, i, j]
			temp = weight*cv2.dct(temp)
			similar_blks_dct[:, i, j] = cv2.idct(temp)[0]

	return similar_blks_dct, wiener_weight


def Aggregation_wiener(similar_blks_dct, wiener_weight, blks_pos, image_base, weight_base):
	shape = similar_blks_dct.shape

	for i in range(shape[0]):
		x, y = blks_pos[i, :]
		blk_idct = wiener_weight*cv2.idct(similar_blks_dct[i, :, :])
		image_base[x:x+shape[1], y:y+shape[2]] += blk_idct
		weight_base[x:x+shape[1], y:y+shape[2]] += wiener_weight
		# weight_base[np.where(weight_base==0)] = 1


# =======================================================-======
# The first stage of the BM3D algorithm
# ==============================================================
def BM3D_step2(image, noise_image):
	width, height = image.shape

	image_base = np.zeros(image.shape)
	weight_base = np.zeros(image.shape)
	k = np.matrix(np.kaiser(block_size, beta_Kaiser))
	kaiser = np.array(k.T*k)

	for i in range(0, width, search_step):
		print("Processing line ", i)
		for j in range(0, height, search_step):
			i, j = coord_boundary(i, j, width, height)
			blks_dct, blks_n_dct, blks_pos, idx = block_matching(image, noise_image, [i, j])
			blks_dct, wiener_weight = wiener_filter(blks_dct, blks_n_dct)
			Aggregation_wiener(blks_dct, wiener_weight, blks_pos, image_base, weight_base)

	image_base[:, :] = image_base[:, :] / weight_base[:, :]
	cv2.imwrite("BM3D_step2.jpg", image_base)
			# print(blks_dct.shape, blks_n_dct.shape)
			# blks_dct, nonzero_num = filter_3D(blks_dct)
			# Aggregation(blks_dct, blks_pos, nonzero_num, image_base, weight_base, kaiser)
	# image_base[:, :] = image_base[:, :]/weight_base[:, :]
	# cv2.imwrite("BM3D_step1.jpg", image_base)