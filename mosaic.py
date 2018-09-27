import sys
import cv2
import logging
import os
import numpy as np
from matplotlib import pyplot as plt

#每个图像块的宽度
BLOCK_WIDTH = 72
#每个图像块的高度
BLOCK_HEIGHT = 48
#色彩的级数
COLOR_SAMPLE_RATE = 32
#默认最大值
MAX_VALUE = 1e9
#颜色相似度的权值
COLOR_WEIGHT = 0.8
#纹理相似度的权值
TEXTURE_WEIGHT = 0.2
#相似度裕量
SIMILARITY_MARGIN = 0.001
#创建做区域直方图均衡的算子
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#得到LBP编码表，采用LBP等价模式：对跳变不超过2的编码单独计数，超过2的归为一类，记为"default"
def get_lbp_codebook():
	lbp_codes = []
	for i in range(2**8):
		code = bin(i)[2:]
		if len(code) < 8:
			code = "0" * (8-len(code)) + code
		jump = 0
		state = "-1"
		for k in range(len(code)):
			if code[k] != state and state != "-1":
				jump += 1
			state = code[k]
		if jump <= 2:
			lbp_codes.append(code)
	
	global lbp_codes_dict
	lbp_codes_dict = dict()
	for i in range(len(lbp_codes)):
		lbp_codes_dict[lbp_codes[i]] = i
	lbp_codes_dict["default"] = len(lbp_codes)
	
	global lbp_codes_length
	lbp_codes_length = len(lbp_codes_dict)


#定义Block类，用作记录图像块的特征
class Block:
	def __init__(self, _width, _height):
		self.width = _width
		self.height = _height
		self.pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
		self.color_hists = []
		self.texture_hists = []
		self.index = -1

	#预处理，计算直方图
	def get_hists(self):
		#layer1用来计算纹理特征
		downsampled_image_layer1 = pryDown_color_image(self.pixels)
		self.texture_hists = calc_texture_hist(downsampled_image_layer1)
		#layer2用来计算颜色特征
		downsampled_image_layer2 = pryDown_color_image(downsampled_image_layer1)
		self.color_hists = calc_color_hist(downsampled_image_layer2)

	def get_index(self, _index):
		self.index = _index


#定义Candidate子类，用作记录候选图的信息
class Candidate(Block):
	def __init__(self, _url, _width, _height):
		super(Candidate, self).__init__(_width, _height)
		self.url = _url
		self.coordinate_lists = []
		image = cv2.imread(self.url)
		resized_image = cv2.resize(image, (self.width, self.height), interpolation = cv2.INTER_CUBIC)
		self.pixels = equalized_color_image(resized_image)
		super(Candidate, self).get_hists()


#定义SubImage子类，用作记录子图的信息
class SubImage(Block):
	def __init__(self, _image, _x, _y, _width, _height):
		super(SubImage, self).__init__(_width, _height)
		self.x = _x
		self.y = _y
		self.block_index = -1
		self.pixels = _image[_y:(_y + _height), _x:(_x + _width), :]
		super(SubImage, self).get_hists()		 


#改变色彩通道顺序
def change_color_channels(image):
	return image[:, :, (2, 1, 0)]


#对彩色图像做直方图均衡：分离通道各自做直方图均衡
def equalized_color_image(image):
	res = np.zeros(image.shape, dtype=np.uint8)
	for k in range(image.shape[-1]):
		res[:, :, k] = clahe.apply(image[:, :, k])
	return res


#对彩色图像做降采样：分离通道各自做降采样
def pryDown_color_image(image):
	if len(image.shape) == 3:
		shape = (int((image.shape[0] + 1) / 2), int((image.shape[1] + 1) / 2), image.shape[2])
	else:
		return cv2.pyrDown(image)
	res = np.zeros(shape, dtype=np.uint8)
	for k in range(image.shape[-1]):
		res[:, :, k] = cv2.pyrDown(image[:, :, k].astype(np.uint8))
	return res


#计算颜色直方图，返回各个通道的直方图
def calc_color_hist(image):
	height, width, channel = image.shape
	hists = []
	for k in range(channel):
		hist, bins = np.histogram(image[:,:,k].flatten(), COLOR_SAMPLE_RATE, [0, 256])
		hists.append(hist)
	return hists


#计算单个通道的LBP特征，按编码表返回纹理直方图
def calc_lbp(mat):
	height, width = mat.shape
	hist = [0] * lbp_codes_length
	for i in range(height-2):
		for j in range(width-2):
			jump = 0
			state = -1
			code = ""
			for k in [-1, 0, 1]:
				for l in [-1, 0, 1]:
					if k == 0 and l == 0:
						continue
					cur = 1 if mat[i+1,j+1] > mat[i+1+k,j+1+l] else 0
					if state != cur and state != -1:
						jump += 1
					state = cur
					code += str(cur)
			if jump <= 2:
				hist[lbp_codes_dict[code]] += 1
			else:
				hist[lbp_codes_dict["default"]] += 1
	return hist


#计算纹理直方图，返回各个通道的直方图
def calc_texture_hist(image):
	height, width, channel = image.shape
	hists = []
	for k in range(channel):
		hist = calc_lbp(image[:,:,k])
		hists.append(hist)
	return hists


#计算两个直方图的相似度
def calc_hist_similarity(hists1, hists2):
	if len(hists1) != len(hists2):
		return MAX_VALUE
	num_channels = len(hists1)
	similarity = 0.0
	for k in range(num_channels):
		if len(hists1[k]) != len(hists2[k]):
			return MAX_VALUE
		num_bins = len(hists1[k])
		score = 0.0
		for i in range(num_bins):
			n1 = hists1[k][i]
			n2 = hists2[k][i]
			if n1 != 0 or n2 != 0:
				score += (n1 * 1.0 / n2) if n1 <= n2 else (n2 * 1.0 / n1)
		score /= num_bins
		similarity += score
	similarity /= num_channels
	return similarity


#计算两个Block的相似度
def calc_similarity(block1, block2):
	similarity = 0
	similarity += calc_hist_similarity(block1.color_hists, block2.color_hists) * COLOR_WEIGHT
	similarity += calc_hist_similarity(block1.texture_hists, block2.texture_hists) * TEXTURE_WEIGHT
	return similarity


def main():
	#加载LBP编码表
	get_lbp_codebook()
	
	#获取候选图片集合
	candidate_path = "./data_all/"
	candidate_files = os.listdir(candidate_path)
	candidates = []
	for file in candidate_files:
		if os.path.splitext(file)[1] == ".jpg" or os.path.splitext(file)[1] == ".png":
			candidate = Candidate(candidate_path + file, BLOCK_WIDTH, BLOCK_HEIGHT)
			candidates.append(candidate)
	
	#确定目标图的尺寸
	target_height = 480
	target_width = 720
	if target_height % BLOCK_HEIGHT != 0 or target_width % BLOCK_WIDTH != 0:
		logging.warn("Size of the target image is not acceptable, " + \
			"target_width:%d, target_height:%d, block_width:%d, block_height:%d" \
			% (target_width, target_height, BLOCK_WIDTH, BLOCK_HEIGHT))
		return -1
	
	#加载目标图原图，并做预处理
	target_image_path = "ptest.png"
	image = cv2.imread(target_image_path)
	hist_equalized_image = equalized_color_image(image)
	target_image = resized_image = cv2.resize(hist_equalized_image, (target_width, target_height), interpolation = cv2.INTER_CUBIC)
	
	#获取子图集合
	num_in_vertical = int(target_height / BLOCK_HEIGHT)
	num_in_horizontal = int(target_width / BLOCK_WIDTH)
	sub_images = []
	for i in range(num_in_vertical):
		for j in range(num_in_horizontal):
			sub_image = SubImage(target_image, j*BLOCK_WIDTH, i*BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT)
			sub_images.append(sub_image)

	#计算子图与候选图的相似度
	similarity = np.zeros((len(sub_images), len(candidates)), dtype=float)
	for i in range(len(sub_images)):
		sub_images[i].get_index(i)
		for j in range(len(candidates)):
			candidates[j].get_index(j)
			similarity[i, j] = calc_similarity(sub_images[i], candidates[j])

	#计算与每个子图的相似度最高（similarity最小）的候选图
	for i in range(len(sub_images)):
		min_similarity = MAX_VALUE
		min_index = -1
		coordinate = (sub_images[i].x, sub_images[i].y)
		for j in range(len(candidates)):
			if abs(similarity[i, j] - min_similarity) < SIMILARITY_MARGIN:
				if len(candidates[j].coordinate_lists) < len(candidates[min_index].coordinate_lists):
					min_index = j
				min_similarity = min(similarity[i, j], min_similarity)
			elif similarity[i, j] < min_similarity:
				min_index = j
				min_similarity = similarity[i, j]
		candidates[min_index].coordinate_lists.append(coordinate)
		sub_images[i].block_index = min_index

	#拼接
	mosaic_image = np.zeros(target_image.shape, dtype=np.uint8)
	for i in range(len(sub_images)):
		x = sub_images[i].x
		y = sub_images[i].y
		bi = sub_images[i].block_index
		bw = sub_images[i].width
		bh = sub_images[i].height
		mosaic_image[y:y+bh, x:x+bw, :] = candidates[bi].pixels
	plt.imshow(mosaic_image)
	plt.show()
	
if __name__ == '__main__':
	main()