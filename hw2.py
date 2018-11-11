import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import floor, pi, exp
from PyQt5 import QtGui, QtCore, QtWidgets

############
# Gokay Gas
# 150150107
# 18.10.2018
############

class Window(QtWidgets.QMainWindow):

	def __init__(self):
		super(Window, self).__init__()
		self.setWindowTitle("Filtering & Geometric Transforms")
		self.setWindowState(QtCore.Qt.WindowMaximized)
    
		self.Img = None
		self.outputImg = None
		self.isInputOpen = False

		mainMenu = self.menuBar()

		fileMenu = mainMenu.addMenu('&File')
		filtersMenu = mainMenu.addMenu('&Filters')
		transformsMenu = mainMenu.addMenu('&Geometric Transforms')

		# file menu actions

		openAction = QtWidgets.QAction("Open", self)
		openAction.triggered.connect(self.open_image)

		saveAction = QtWidgets.QAction("Save", self)
		saveAction.triggered.connect(self.save_image)

		exitAction = QtWidgets.QAction("Exit", self)
		exitAction.triggered.connect(QtCore.QCoreApplication.instance().quit)

		fileMenu.addAction(openAction)
		fileMenu.addAction(saveAction)
		fileMenu.addAction(exitAction)

		# filter menu actions

		averageFiltersMenu = filtersMenu.addMenu('&Average Filters')
		gaussianFiltersMenu = filtersMenu.addMenu('&Gaussian Filters')
		medianFiltersMenu = filtersMenu.addMenu('&Median Filters')

		threeByThreeAction = QtWidgets.QAction("3x3", self)
		threeByThreeAction.triggered.connect(lambda: self.average_filtering(3))
		fiveByFiveAction = QtWidgets.QAction("5x5", self)
		fiveByFiveAction.triggered.connect(lambda: self.average_filtering(5))
		sevenBySevenAction = QtWidgets.QAction("7x7", self)
		sevenBySevenAction.triggered.connect(lambda: self.average_filtering(7))
		nineByNineAction = QtWidgets.QAction("9x9", self)
		nineByNineAction.triggered.connect(lambda: self.average_filtering(9))
		elevenByElevenAction = QtWidgets.QAction("11x11", self)
		elevenByElevenAction.triggered.connect(lambda: self.average_filtering(11))
		thirteenByThirteenAction = QtWidgets.QAction("13x13", self)
		thirteenByThirteenAction.triggered.connect(lambda: self.average_filtering(13))
		fifteenByFifteenAction = QtWidgets.QAction("15x15", self)
		fifteenByFifteenAction.triggered.connect(lambda: self.average_filtering(15))

		gaussianThreeByThreeAction = QtWidgets.QAction("3x3", self)
		gaussianThreeByThreeAction.triggered.connect(lambda: self.gaussian_filtering(3))
		gaussianFiveByFiveAction = QtWidgets.QAction("5x5", self)
		gaussianFiveByFiveAction.triggered.connect(lambda: self.gaussian_filtering(5))
		gaussianSevenBySevenAction = QtWidgets.QAction("7x7", self)
		gaussianSevenBySevenAction.triggered.connect(lambda: self.gaussian_filtering(7))
		gaussianNineByNineAction = QtWidgets.QAction("9x9", self)
		gaussianNineByNineAction.triggered.connect(lambda: self.gaussian_filtering(9))
		gaussianElevenByElevenAction = QtWidgets.QAction("11x11", self)
		gaussianElevenByElevenAction.triggered.connect(lambda: self.gaussian_filtering(11))
		gaussianThirteenByThirteenAction = QtWidgets.QAction("13x13", self)
		gaussianThirteenByThirteenAction.triggered.connect(lambda: self.gaussian_filtering(13))
		gaussianFifteenByFifteenAction = QtWidgets.QAction("15x15", self)
		gaussianFifteenByFifteenAction.triggered.connect(lambda: self.gaussian_filtering(15))

		medianThreeByThreeAction = QtWidgets.QAction("3x3", self)
		medianThreeByThreeAction.triggered.connect(lambda: self.median_filtering(3))
		medianFiveByFiveAction = QtWidgets.QAction("5x5", self)
		medianFiveByFiveAction.triggered.connect(lambda: self.median_filtering(5))
		medianSevenBySevenAction = QtWidgets.QAction("7x7", self)
		medianSevenBySevenAction.triggered.connect(lambda: self.median_filtering(7))
		medianNineByNineAction = QtWidgets.QAction("9x9", self)
		medianNineByNineAction.triggered.connect(lambda: self.median_filtering(9))
		medianElevenByElevenAction = QtWidgets.QAction("11x11", self)
		medianElevenByElevenAction.triggered.connect(lambda: self.median_filtering(11))
		medianThirteenByThirteenAction = QtWidgets.QAction("13x13", self)
		medianThirteenByThirteenAction.triggered.connect(lambda: self.median_filtering(13))
		medianFifteenByFifteenAction = QtWidgets.QAction("15x15", self)
		medianFifteenByFifteenAction.triggered.connect(lambda: self.median_filtering(15))

		averageFiltersMenu.addAction(threeByThreeAction)
		averageFiltersMenu.addAction(fiveByFiveAction)
		averageFiltersMenu.addAction(sevenBySevenAction)
		averageFiltersMenu.addAction(nineByNineAction)
		averageFiltersMenu.addAction(elevenByElevenAction)
		averageFiltersMenu.addAction(thirteenByThirteenAction)
		averageFiltersMenu.addAction(fifteenByFifteenAction)

		gaussianFiltersMenu.addAction(gaussianThreeByThreeAction)
		gaussianFiltersMenu.addAction(gaussianFiveByFiveAction)
		gaussianFiltersMenu.addAction(gaussianSevenBySevenAction)
		gaussianFiltersMenu.addAction(gaussianNineByNineAction)
		gaussianFiltersMenu.addAction(gaussianElevenByElevenAction)
		gaussianFiltersMenu.addAction(gaussianThirteenByThirteenAction)
		gaussianFiltersMenu.addAction(gaussianFifteenByFifteenAction)

		medianFiltersMenu.addAction(medianThreeByThreeAction)
		medianFiltersMenu.addAction(medianFiveByFiveAction)
		medianFiltersMenu.addAction(medianSevenBySevenAction)
		medianFiltersMenu.addAction(medianNineByNineAction)
		medianFiltersMenu.addAction(medianElevenByElevenAction)
		medianFiltersMenu.addAction(medianThirteenByThirteenAction)
		medianFiltersMenu.addAction(medianFifteenByFifteenAction)

		# transform menu actions

		rotateTransformsMenu = transformsMenu.addMenu('&Rotate')
		scaleTransformsMenu = transformsMenu.addMenu('&Scale')
		translateTransformsMenu = transformsMenu.addMenu('&Translate')

		rotateRightAction = QtWidgets.QAction("Rotate 10 Degree Right", self)
		rotateRightAction.triggered.connect(lambda: self.rotation_transform(10))
		rotateLeftAction = QtWidgets.QAction("Rotate 10 Degree Left", self)
		rotateLeftAction.triggered.connect(lambda: self.rotation_transform(-10))

		scaleDoubleAction = QtWidgets.QAction("2x", self)
		scaleDoubleAction.triggered.connect(lambda: self.scale_transform(2))
		scaleHalfAction = QtWidgets.QAction("1/2x", self)
		scaleHalfAction.triggered.connect(lambda: self.scale_transform(0.5))

		translateRightAction = QtWidgets.QAction("Right", self)
		translateRightAction.triggered.connect(lambda: self.translate_transform(1))
		translateLeftAction = QtWidgets.QAction("Left", self)
		translateLeftAction.triggered.connect(lambda: self.translate_transform(-1))

		rotateTransformsMenu.addAction(rotateRightAction)
		rotateTransformsMenu.addAction(rotateLeftAction)

		scaleTransformsMenu.addAction(scaleDoubleAction)
		scaleTransformsMenu.addAction(scaleHalfAction)

		translateTransformsMenu.addAction(translateRightAction)
		translateTransformsMenu.addAction(translateLeftAction)

		# central widget for the opened image.
		self.centralwidget = QtWidgets.QWidget(self)
		self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
		self.horizontalLayout.setContentsMargins(100, 10, 100, 10)
		self.horizontalLayout.setSpacing(100)
		self.setCentralWidget(self.centralwidget)

		self.show()


	def open_image(self):
		# Image
		self.Img = cv2.imread("input.png")

		R, C, B = self.Img.shape
		qImg = QtGui.QImage(self.Img.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		
		#pix = QtGui.QPixmap('color1.png')
		self.label = QtWidgets.QLabel(self.centralwidget)
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)
		self.label.setAlignment(QtCore.Qt.AlignCenter)
		self.label.setStyleSheet("border:0px")
		
		self.horizontalLayout.addWidget(self.label)

	def save_image(self):
		cv2.imwrite("./output-image.png", self.outputImg)

	def average_filtering(self, size):

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1], 3], dtype=np.uint8)

		# kernel full of 1's
		kernel = np.zeros([size, size, 3], dtype=np.uint8)
		kernel[:,:,:] = 1

		# expand image for convolution
		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2), 3], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)),:] = self.Img

		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j,:] = np.sum(np.sum((kernel*expandedImage[i:i+size, j:j+size,:]),0),0) // (size*size)

		# show the outputted image
		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def gaussian_filtering(self, size):
		standardDeviation = 0.2 * size

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1], 3], dtype=np.uint8)

		# prepare the gaussian kernel
		kernel = np.zeros([size, size, 3], dtype='int64')
		for i in range(size):
			for j in range(size):
				kernel[i,j,:] = round((1 / (2 * pi * standardDeviation)) * exp(-((((i - (size // 2))*(i - (size // 2))) + ((j - (size // 2))*(j - (size // 2)))) / (2 * standardDeviation * standardDeviation))) * 100)

		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2), 3], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)),:] = self.Img

		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j,:] = np.sum(np.sum((kernel*expandedImage[i:i+size, j:j+size,:]),0),0) // np.sum(np.sum(kernel, 0), 0)[0]

		# show the outputted image
		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def median_filtering(self, size):

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1], 3], dtype=np.uint8)

		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2), 3], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)),:] = self.Img

		# get the median of the box of pixels
		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j,0] = np.median(expandedImage[i:i+size, j:j+size,0])
				self.outputImg[i,j,1] = np.median(expandedImage[i:i+size, j:j+size,1])
				self.outputImg[i,j,2] = np.median(expandedImage[i:i+size, j:j+size,2])

		# show the outputted image
		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)


	def rotation_transform(self, size):
		raise NotImplementedError

	def scale_transform(self, size):
		multiplier = 1 / size

		# output image with the new size
		self.outputImg = np.zeros([int(self.Img.shape[0] // multiplier), int(self.Img.shape[1] // multiplier), 3], dtype=np.uint8)

		# interpolate the backward mapped coordinates
		for i in range(self.outputImg.shape[0]):
			for j in range(self.outputImg.shape[1]):
				self.outputImg[i,j,:] = np.round(self.bicubic_interpolation(i * multiplier, j * multiplier))

		# show the outputted image
		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def translate_transform(self, size):
		if size == 1:
			self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1] + 20, 3], dtype=np.uint8)
			for i in range(self.outputImg.shape[0]):
				for j in range(self.outputImg.shape[1]):
					if j - 20 >= 0:
						self.outputImg[i,j,:] = self.Img[i, j - 20,:]
		elif size == -1:
			self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1] + 20, 3], dtype=np.uint8)
			for i in reversed(range(self.outputImg.shape[0])):
				for j in reversed(range(self.outputImg.shape[1])):
					if j <= self.Img.shape[1] - 1:
						self.outputImg[i,j,:] = self.Img[i, j,:]

		# show the outputted image
		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def bicubic_interpolation(self, x, y):
		
		if(floor(x) == x and floor(y) == y):
			return self.Img[int(x),int(y),:]

		if floor(x) == self.Img.shape[0] - 1:
			x = self.Img.shape[0] - 2

		if floor(y) == self.Img.shape[1] - 1:
			y = self.Img.shape[1] - 2

		p = np.zeros([4,4,3], dtype='int64')

		if(x >= 1 and x < self.Img.shape[0] - 2 and y >= 1 and y < self.Img.shape[1] - 2):
			p[:,:,:] = self.Img[floor(x) - 1: floor(x) + 3, floor(y) - 1: floor(y) + 3,:]
		else:
			if x < 1:
				p[0,1:3,:] = self.Img[floor(x), floor(y):floor(y) + 2, :]
			elif x >= self.Img.shape[0] - 2:
				p[3,1:3,:] = self.Img[floor(x) + 1, floor(y):floor(y) + 2, :]
			else:
				p[0,1:3,:] = self.Img[floor(x) - 1, floor(y):floor(y) + 2, :]
				p[3,1:3,:] = self.Img[floor(x) + 2, floor(y):floor(y) + 2, :]

			if y < 1:
				p[1:3,0,:] = self.Img[floor(x):floor(x) + 2, floor(y), :]
			elif y >= self.Img.shape[1] - 2:
				p[1:3,3,:] = self.Img[floor(x):floor(x) + 2, floor(y) + 1, :]
			else:
				p[1:3,0,:] = self.Img[floor(x):floor(x) + 2, floor(y) - 1, :]
				p[1:3,3,:] = self.Img[floor(x):floor(x) + 2, floor(y) + 2, :]

			p[1:3,1:3,:] = self.Img[floor(x):floor(x) + 2, floor(y):floor(y) + 2, :]

			p[0,0,:] = p[0,1,:] # northeast point gets the value from right
			p[3,0,:] = p[3,1,:] # southeast point gets the value from right
			p[0,3,:] = p[0,2,:] # northwest point gets the value from left
			p[3,3,:] = p[3,2,:] # southwest point gets the value from left
	
		# inspired from https://www.paulinternet.nl/?page=bicubic /// translated from java version to python
		
		a00 = p[1,1,:]
		a01 = -.5*p[1,0,:] + .5*p[1,2,:]
		a02 = p[1,0,:] - 2.5*p[1,1,:] + 2*p[1,2,:] - .5*p[1,3,:]
		a03 = -.5*p[1,0,:] + 1.5*p[1,1,:] - 1.5*p[1,2,:] + .5*p[1,3,:]
		a10 = -.5*p[0,1,:] + .5*p[2,1,:]
		a11 = .25*p[0,0,:] - .25*p[0,2,:] - .25*p[2,0,:] + .25*p[2,2,:]
		a12 = -.5*p[0,0,:] + 1.25*p[0,1,:] - p[0,2,:] + .25*p[0,3,:] + .5*p[2,0,:] - 1.25*p[2,1,:] + p[2,2,:] - .25*p[2,3,:]
		a13 = .25*p[0,0,:] - .75*p[0,1,:] + .75*p[0,2,:] - .25*p[0,3,:] - .25*p[2,0,:] + .75*p[2,1,:] - .75*p[2,2,:] + .25*p[2,3,:]
		a20 = p[0,1,:] - 2.5*p[1,1,:] + 2*p[2,1,:] - .5*p[3,1,:]
		a21 = -.5*p[0,0,:] + .5*p[0,2,:] + 1.25*p[1,0,:] - 1.25*p[1,2,:] - p[2,0,:] + p[2,2,:] + .25*p[3,0,:] - .25*p[3,2,:]
		a22 = p[0,0,:] - 2.5*p[0,1,:] + 2*p[0,2,:] - .5*p[0,3,:] - 2.5*p[1,0,:] + 6.25*p[1,1,:] - 5*p[1,2,:] + 1.25*p[1,3,:] + 2*p[2,0,:] - 5*p[2,1,:] + 4*p[2,2,:] - p[2,3,:] - .5*p[3,0,:] + 1.25*p[3,1,:] - p[3,2,:] + .25*p[3,3,:]
		a23 = -.5*p[0,0,:] + 1.5*p[0,1,:] - 1.5*p[0,2,:] + .5*p[0,3,:] + 1.25*p[1,0,:] - 3.75*p[1,1,:] + 3.75*p[1,2,:] - 1.25*p[1,3,:] - p[2,0,:] + 3*p[2,1,:] - 3*p[2,2,:] + p[2,3,:] + .25*p[3,0,:] - .75*p[3,1,:] + .75*p[3,2,:] - .25*p[3,3,:]
		a30 = -.5*p[0,1,:] + 1.5*p[1,1,:] - 1.5*p[2,1,:] + .5*p[3,1,:]
		a31 = .25*p[0,0,:] - .25*p[0,2,:] - .75*p[1,0,:] + .75*p[1,2,:] + .75*p[2,0,:] - .75*p[2,2,:] - .25*p[3,0,:] + .25*p[3,2,:]
		a32 = -.5*p[0,0,:] + 1.25*p[0,1,:] - p[0,2,:] + .25*p[0,3,:] + 1.5*p[1,0,:] - 3.75*p[1,1,:] + 3*p[1,2,:] - .75*p[1,3,:] - 1.5*p[2,0,:] + 3.75*p[2,1,:] - 3*p[2,2,:] + .75*p[2,3,:] + .5*p[3,0,:] - 1.25*p[3,1,:] + p[3,2,:] - .25*p[3,3,:]
		a33 = .25*p[0,0,:] - .75*p[0,1,:] + .75*p[0,2,:] - .25*p[0,3,:] - .75*p[1,0,:] + 2.25*p[1,1,:] - 2.25*p[1,2,:] + .75*p[1,3,:] + .75*p[2,0,:] - 2.25*p[2,1,:] + 2.25*p[2,2,:] - .75*p[2,3,:] - .25*p[3,0,:] + .75*p[3,1,:] - .75*p[3,2,:] + .25*p[3,3,:]

		x = x - floor(x)
		y = y - floor(y)

		x2 = x * x
		x3 = x2 * x
		y2 = y * y
		y3 = y2 * y

		return (a00 + a01 * y + a02 * y2 + a03 * y3) + (a10 + a11 * y + a12 * y2 + a13 * y3) * x + (a20 + a21 * y + a22 * y2 + a23 * y3) * x2 + (a30 + a31 * y + a32 * y2 + a33 * y3) * x3

def main():
	app = QtWidgets.QApplication(sys.argv)
	GUI = Window()
	sys.exit(app.exec_())

main()
