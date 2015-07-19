# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:04:59 2015

@author: 
"""
from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


	
def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line=line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append((listFromLine[-1]))
		index+=1 
	return returnMat,classLabelVector

# def labelStrToNum(labelSet,mylabel):
	# label= mylabel
	# numLabel = []
	# m = 1000
	# for index in range(0,m):
		# if (labelSet[index] == label[0]):
			# numLabel.append(0)
        # elif (labelSet[index] == label[1]):
			# numLabel.append(1)
        # elif (labelSet[index] == label[2]):
			# numLabel.append(2)
	# return index,numLabel

def datingCalssTest():
	hoRatio = 0.1
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
	numLabel = []
	labelSet=datingLabels
	m=1000
	label=["didntLike","smallDoses","largeDoses"]
	for index in range(0,m):
		if labelSet[index] == label[0]:
			numLabel.append(0)
		elif labelSet[index] == label[1]:
			numLabel.append(1)
		elif labelSet[index] == label[2]:
			numLabel.append(2)
	datingLabels = numLabel
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classfierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print "predict class:%d,real class:%d"%(classfierResult,datingLabels[i])
		if(classfierResult != datingLabels[i]): errorCount+=1.0
	print "the total error rate is :%f" %(errorCount/float(numTestVecs))
	
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet / tile(ranges,(m,1))
	return normDataSet, ranges, minVals
	
def classify0(intX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(intX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.iteritems(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(linStr[j])
	return returnVect
	
def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('digits/trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('digits/trainingDigits/%s'%fileNameStr)
	testFileList = listdir('digits/testDigits')
	errorCount = 0.0 
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('digits/testDigits/%s'%fileNameStr)
		classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
		if(classifierResult!=classNumStr):
			errorCount+=1.0
			print "the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumStr)
	print "\nthe total number of error is:%d"%errorCount
	print "\nthe total error rate is:%f"%(errorCount/float(mTest))
	