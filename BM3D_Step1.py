import cv2
import numpy as np

block_size = 8
search_window_size = 50
search_step = 3
match_threshold = 100
max_match_block = 8
filter_3D_threshold = 70
beta_Kaiser = 2.0

# =======================================================-======
# Returns the top left and bottom left coordinates
# ==============================================================
def get_search_window(image, block_coord):
	x, y = block_coord
	m, n = image.shape

	blk_size = block_size
	window_size = search_window_size

	left_up_x = x + blk_size/2 - window_size/2
	left_up_y = y + blk_size/2 - window_size/2
	right_down_x = left_up_x + window_size
	right_down_y = left_up_y + window_size

	# Determine whether it is out of bounds
	if left_up_x < 0: left_up_x = 0
	elif right_down_x > m: left_up_x = m - window_size
	if left_up_y < 0: left_up_y = 0
	elif right_down_y > n: left_up_y = n - window_size
	return int(left_up_x), int(left_up_y)


# ============================================================
# Calculate the other image block and its similarity
# and join the group if the similarity reaches the threshold
# ============================================================
def block_matching(image, block_coord):
	x, y = block_coord
	img_block = image[x:x+block_size, y:y+block_size]
	img_block_dct = cv2.dct(img_block.astype(np.float64))

	blk_num = int((search_window_size - block_size)/search_step)
	window_x, window_y = get_search_window(image, block_coord)

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

			# Threshold filtering
			if distance < match_threshold:
				similar_blks_dct[match_index, :, :] = temp_blk_dct
				similar_blks_pos[match_index, :] = x, y
				blk_distances[match_index] = distance
				match_index = match_index + 1

	# Because OpenCV cannot do odd-numbered DCT transforms
	# so need to limit the value of w
	if match_index >= max_match_block:
		match_index = max_match_block

	if match_index%2 == 1:
		match_index = match_index - 1
	if match_index == 0:
		match_index = 2

	blk_distances = blk_distances[0:match_index]
	sort = blk_distances.argsort()

	final_blks_dct = np.zeros([match_index, block_size, block_size])
	final_blks_pos = np.zeros([match_index, 2], dtype=int)
	final_blks_dct[0, :, :] = img_block_dct
	final_blks_pos[0, :] = block_coord 

	for i in range(1, match_index):
		final_blks_dct[i, :, :] = similar_blks_dct[sort[i-1], :, :]
		final_blks_pos[i, :] = similar_blks_pos[sort[i-1], :]
	return final_blks_dct, final_blks_pos, match_index+1


# =======================================================-======
# Returns the top left and bottom left coordinates
# ==============================================================
def filter_3D(similar_blks_dct):
	nonzero_num = 0
	shape = similar_blks_dct.shape
	for i in range(shape[1]):
		for j in range(shape[2]):
			correspond_pixel = similar_blks_dct[:, i, j]
			pixel_dct = cv2.dct(correspond_pixel)
			coord = np.where(np.abs(pixel_dct)<filter_3D_threshold)
			pixel_dct[coord] = 0
			nonzero_num += pixel_dct.nonzero()[0].size
			similar_blks_dct[:, i, j] = cv2.idct(pixel_dct)[0]
	return similar_blks_dct, nonzero_num


# ==============================================================
# pixel fusion——Weighted average
# ==============================================================
def Aggregation(blks_dct, blks_pos, nonzero_num, image_base, weight_base, kaiser):
	shape = np.array(blks_dct.shape)
	if nonzero_num < 1: nonzero_num = 1
	weight = kaiser/nonzero_num
	image_blk = np.zeros(shape)
	for i in range(shape[0]):
		x, y = blks_pos[i, :]
		blk_idct = weight*cv2.idct(blks_dct[i, :, :])
		image_base[x:x+shape[1], y:y+shape[2]] += blk_idct
		weight_base[x:x+shape[1], y:y+shape[2]] += weight


# ===============================================================
# Ensure that the coordinates do not exceed the range
# ===============================================================
def coord_boundary(i, j, width, height):
	if i + block_size < width:
		coord_x = i
	else:
		coord_x = width - block_size
	if j + block_size < height:
		coord_y = j
	else:
		coord_y = height - block_size
	coord = np.array([coord_x, coord_y], dtype=int)
	return coord


# =======================================================-======
# The first stage of the BM3D algorithm
# ==============================================================
def BM3D_step1(image):
	width, height = image.shape

	image_base = np.zeros(image.shape)
	weight_base = np.zeros(image.shape)
	k = np.matrix(np.kaiser(block_size, beta_Kaiser))
	kaiser = np.array(k.T*k)

	for i in range(0, width, search_step):
		print("Processing line ", i)
		for j in range(0, height, search_step):
			i, j = coord_boundary(i, j, width, height)
			blks_dct, blks_pos, idx = block_matching(image, [i, j])
			blks_dct, nonzero_num = filter_3D(blks_dct)
			Aggregation(blks_dct, blks_pos, nonzero_num, image_base, weight_base, kaiser)
	image_base[:, :] = image_base[:, :]/weight_base[:, :]
	cv2.imwrite("BM3D_step1.jpg", image_base)