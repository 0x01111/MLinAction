# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 21:04:04 2015

@author: TheSkull
"""

import kNN

datingDataMat,datingLabels=kNN.file2matrix('datingTestSet.txt')

reload(kNN)

datingDataMat,datingLabels=kNN.file2matrix('datingTestSet.txt')

normMat,ranges,minVals = kNN.autoNorm(datingDataMat)

kNN.datingCalssTest()


label=["didntLike","smallDoses","largeDoses"]
index,numLabel = kNN.labelStrToNum(datingLabels,label)
"""
didntLike

smallDoses

largeDoses

"""

numLabel = []
labelSet=datingLabels
m=1000
for index in range(0,m):
	if labelSet[index] == label[0]:
		numLabel.append(0)
	elif labelSet[index] == label[1]:
		numLabel.append(1)
	elif labelSet[index] == label[2]:
		numLabel.append(2)

datingLabels = numLabel
