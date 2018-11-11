import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import floor, pi, exp
from PyQt5 import QtGui, QtCore, QtWidgets
from math import floor

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
		self.Img = cv2.imread("color1.png")

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
		raise NotImplementedError

	def average_filtering(self, size):

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1], 3], dtype=np.uint8)

		kernel = np.zeros([size, size, 3], dtype=np.uint8)
		kernel[:,:,:] = 1

		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2), 3], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)),:] = self.Img

		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j,:] = np.sum(np.sum((kernel*expandedImage[i:i+size, j:j+size,:]),0),0) // (size*size)

		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def gaussian_filtering(self, size):
		standardDeviation = 0.2 * size

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1], 3], dtype=np.uint8)

		kernel = np.zeros([size, size, 3], dtype='int64')
		for i in range(size):
			for j in range(size):
				kernel[i,j,:] = round((1 / (2 * pi * standardDeviation)) * exp(-((((i - (size // 2))*(i - (size // 2))) + ((j - (size // 2))*(j - (size // 2)))) / (2 * standardDeviation * standardDeviation))) * 100)

		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2), 3], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)),:] = self.Img

		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j,:] = np.sum(np.sum((kernel*expandedImage[i:i+size, j:j+size,:]),0),0) // np.sum(np.sum(kernel, 0), 0)[0]

		R, C, B = self.outputImg.shape
		qImg = QtGui.QImage(self.outputImg.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def median_filtering(self, size):
		raise NotImplementedError

	def rotation_transform(self, size):
		raise NotImplementedError

	def scale_transform(self, size):
		raise NotImplementedError

	def translate_transform(self, size):
		raise NotImplementedError


def main():
	app = QtWidgets.QApplication(sys.argv)
	GUI = Window()
	sys.exit(app.exec_())

main()
