# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir

def loadDataSet():
	dataMat=[]
	labelMat=[]
	# 打开txt文件
	fr = open('testSet.txt')
	# readData中保存了所有的数据行
	readData = fr.readlines()
	# 利用正则，每行数据以 \t 划分开的
	# 返回的结果是list
	for line in readData:
		lineArr = line.strip().split("\t")
		dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat,labelMat

def sigmoid(intX):
	# sigmoid 函数
	return 1.0/(1+exp(-intX))

def gradAscent(dataMatIn,classLabels):
	# 梯度上升法
	DataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m,n = shape(DataMatrix)
	# alpha 学习率
	alpha = 0.001 
	maxCycles = 500
	weight = ones((n,1))
	# 下面计算都是按照矩阵方式
	# 1.计算sigmoid函数，也就是根据当前的weight，判断类别
	# 2.计算误差
	# 3.更新权值 对w_i,只是计算了自己的误差，误差大，更新权值w_i相对越大
	# 4.迭代到达最大次数结束 
	for k in range(maxCycles):
		h= sigmoid(DataMatrix*weight)
		error = (labelMat - h)
		weight = weight  + alpha * DataMatrix.transpose()*error
	return weight

def plotBestFit(weights):
	import matplotlib.pyplot as plt
	dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = [];ycord1=[]
	xcord2 = [];ycord2=[]
	for i in range(n):
		if(int(labelMat[i])==1):
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
	ax.scatter(xcord2,ycord2,s=30,c='green')
	x = arange(-3.0,3.0,0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1');plt.ylabel('X2')
	plt.show()

def runTest():
	dataArr,labelMat = loadDataSet()
	weights = gradAscent(dataArr,labelMat)
	plotBestFit(weights.getA())

def stocGradAscent0(dataMatrix,classLabels):
	# 随机梯度上升法
	m,n=shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	# 选取前m个样本更新权值
	# 每次对所有的weights 更新的权值一样 
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h
		weights = weights + alpha*error * dataMatrix[i]
	return weights

def runTest1():
	dataArr,labelMat = loadDataSet()
	weights = stocGradAscent0(array(dataArr),labelMat)
	plotBestFit(weights)

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
	# 改进的随机梯度上升法
	m,n=shape(dataMatrix)
	weights = ones(n)
	# 随机的选取样本，计算误差，更新权值
	# 这里的alpha 也根据迭代次数更新大小，
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1.0+i+j)+0.01
			randIndex = int(random.uniform(0,len(dataIndex)))
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h 
			weights = weights + alpha* error* dataMatrix[randIndex]
			del(dataIndex[randIndex])
	return weights
	
def runTest2():
	dataArr,labelMat = loadDataSet()
	weights = stocGradAscent1(array(dataArr),labelMat)
	plotBestFit(weights)
	
def classifyVector(intX,weights):
	prob = sigmoid(sum(intX*weights))
	if prob>0.5: return 1.0 
	else: return 0.0 

def colicTest():
	frTrain = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []
	trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = [] 
		for i in range(21):
			lineArr.append(float(currLine[i]))
		trainingSet.append(lineArr)
		trainingLabels.append(float(currLine[i]))
	trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
	errorCount = 0 
	numTestVec = 0.0 
	for line in frTest.readlines():
		numTestVec += 1.0 
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append(float(currLine[i]))
		if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
			errorCount +=1 
	errorRate = float(errorCount)/numTestVec
	print "the error rate of this test is:%f"%errorRate
	return errorRate

def mulitTest():
	numTests = 10
	errorSum = 0.0 
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is:%f"%(numTests,errorSum/float(numTests))