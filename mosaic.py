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
	def __init__(self，url, width, height):
		self._width = width
		self._height = height
		self._pixels = np.zeros((self.height, self.width, 3), dtype=int)
		self._color_hists = []
		self._texture_hists = []
		self._index = -1

	#预处理，计算直方图
	def preprocess(self):
		#layer1用来计算纹理特征
		downsampled_image_layer1 = cv2.pyrDown(self._pixels)
		self._texture_hists = calc_texture_hist(downsampled_image_layer1)
		#layer2用来计算颜色特征
		downsampled_image_layer2 = cv2.pyrDown(downsampled_image_layer1)
		self._color_hists = calc_color_hist(downsampled_image_layer2)

	def get_index(self, index):
		self._index = index


#定义Candidate子类，用作记录候选图的信息
class Candidate(Block):
	def __init__(self, url, width, height):
		super(Candidate, self).__init__(width, height)
		self._url = url
		self._coordinate_lists = []
		image = cv2.imread(self._url)
		resized_image = cv2.resize(image, (self._width, self._height), interpolation = cv2.INTER_CUBIC)
		self._pixels = hist_equalized_image = clahe.apply(resized_image)
		super(Candidate, self).preprocess()


#定义SubImage子类，用作记录子图的信息
class SubImage(Block):
	def __init__(self, image, x, y, width, height):
		super(SubImage, self).__init__(width, height)
		self._x = x
		self._y = y
		self._block_index = -1
		self._pixels = image[x:x+width,y:y+height,:]
		super(SubImage, self).preprocess()		 


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
	similarity += calc_hist_similarity(block1._color_hists, block2._color_hists) * COLOR_WEIGHT
	similarity += calc_hist_similarity(block1._texture_hists, block2._texture_hists) * TEXTURE_WEIGHT
	return similarity


def main():
	#加载LBP编码表
	get_lbp_codebook()
	
	#获取候选图片集合
	candidate_path = "./"
	candidate_files = os.listdir(candidate_path)
	candidates = []
	for file in candidate_files:
		candidate = Candidate(file, BLOCK_WIDTH, BLOCK_HEIGHT)
		candidates.append(candidate)
	
	#确定目标图的尺寸
	target_height = 2400
	target_width = 3600
	if target_height % BLOCK_HEIGHT != 0 or target_width % BLOCK_WIDTH != 0:
		logging.warn("Size of the target image is not acceptable, " + \
			"target_width:%d, target_height:%d, block_width:%d, block_height:%d" \
			% (target_width, target_height, BLOCK_WIDTH, BLOCK_HEIGHT))
		return -1
	
	#加载目标图原图，并做预处理
	target_image_path = "ptest.png"
	image = cv2.imread(target_image_path)
	hist_equalized_image = clahe.apply(image)
	resized_image = cv2.resize(hist_equalized_image, (width, height), interpolation = cv2.INTER_CUBIC)
	
	#获取子图集合
	num_in_vertical = height / BLOCK_HEIGHT
	num_in_horizontal = width / BLOCK_WIDTH
	sub_images = []
	for i in range(num_in_vertical):
		for j in range(num_in_horizontal):
			sub_image = SubImage(image, j*BLOCK_WIDTH, i*BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT)
			sub_images.append(sub_image)

	#计算子图与候选图的相似度
	similarity = np.zeros((len(sub_images), len(candidates)), dtype=float)
	for i in range(len(sub_images)):
		sub_images[i].get_index(i)
		for j in range(len(candidates)):
			candidates[i].get_index(j)
			similarity[i,j] = calc_similarity(sub_images[i], candidates[j])

	#TODO: choose

	
if __name__ == '__main__':
	main()