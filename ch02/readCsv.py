from numpy import *
import operator
from os import listdir


def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,8))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line=line.strip()
		listFromLine = line.split(' ')
		returnMat[index,:] = listFromLine[0:8]
		
		index+=1 
	return returnMat