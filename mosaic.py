import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
BLOCK_WIDTH = 72
BLOCK_HEIGHT = 48
STRIDE = 8
COLOR_SAMPLE_RATE = 32
MAX_VALUE = 1e9

#定义Block类用作记录对全图的替换
class Block:
	def __init__(self):
		self._width = BLOCK_WIDTH
		self._height = BLOCK_HEIGHT
		self._x = -1
		self._y = -1
		self._url = None
		self._pixels = np.zeros((self.height, self.width, 3), dtype=int)
		self._hists = []

#创建做区域直方图均衡的算子
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

#读取图像并做预处理
def preprocess(url):
	image = cv2.imread(url)
	resized_image = cv2.resize(image, (BLOCK_WIDTH, BLOCK_HEIGHT), interpolation = cv2.INTER_CUBIC)
	hist_equalized_image = clahe.apply(resized_image)
	#layer1用来计算纹理特征
	downsampled_image_layer1 = cv2.pyrDown(hist_equalized_image)
	#layer2用来计算颜色特征
	downsampled_image_layer2 = cv2.pyrDown(downsampled_image_layer1)
	return downsampled_image_layer2

#计算颜色直方图，返回各个通道的直方图
def calc_color_hist(block):
	height, width, channel = block.shape
	hists = []
	for k in range(channel):
		hist, bins = np.histogram(img[:,:,k].flatten(), COLOR_SAMPLE_RATE, [0, 256])
		hists.append(hist)
	return hists

#得到LBP编码表，采用LBP等价模式：对跳变不超过2的编码单独计数，超过2的归为一类，记为"default"
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
lbp_codes_dict = {}
for i in range(len(lbp_codes)):
	lbp_codes_dict[lbp_codes[i]] = i
lbp_codes_dict["default"] = len(lbp_codes)
lbp_codes_length = len(lbp_codes_dict)

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
def calc_texture_hist(block):
	height, width, channel = block.shape
	hists = []
	for k in range(channel):
		hist = calc_lbp(block[:,:,k])
		hists.append(hist)
	return hists

#计算两个直方图的相似度
def calc_similarity(hists1, hists2):
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

def main():
	#TODO
	
if __name__ == '__main__':
	filename = "ptest.png"
	img = cv2.imread(filename)
	print img.shape
	height = 2400
	width = 3600