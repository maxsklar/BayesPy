#!/usr/bin/python
#
# Finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2012 Max Sklar

# A sample of a file to pipe into this python script is given by test.csv

# ex
# cat test.csv | ./finaDirichletPrior.py --sampleRate 1

# Paper describing the basic formula:
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf

# Each columns is a different category, and it is assumed that the counts are pulled out of
# a different distribution for each row.
# The distribution for each row is pulled from a Dirichlet distribution; this script finds that
# dirichlet which maximizes the probability of the output.

# Parameter: the first param is the sample rate.  This is to avoid using the full data set when we
# have huge amounts of data.

import sys
import csv
import math
import random
import time
from optparse import OptionParser

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
(options, args) = parser.parse_args()
K = int(options.K)

#Find the log probability that we see a certain set of data
# give our prior.
def dirichLogProb(priorList, uMatrix, vVector):
	total = 0.0
	for k in range(0, K):
		for i in range(0, len(uMatrix[k])):
			total += uMatrix[k][i]*math.log(priorList[k] + i)
	
	sumPrior = sum(priorList)
	for i in range(0, len(vVector)):
		total -= vVector[i] * math.log(sumPrior + i)
			
	return total

#Gives the derivative with respect to the log of prior.  This will be used to adjust the loss
def priorGradient(priorList, uMatrix, vVector):
	
	termToSubtract = 0
	for i in range(0, len(vVector)):
		termToSubtract += float(vVector[i]) / (sum(priorList) + i)
	
	retVal = [0]*K
	for j in range(0, K):
		for i in range(0, len(uMatrix[j])):
			retVal[j] += float(uMatrix[j][i]) / (priorList[j] + i)
	
	for j in range(0, K):
		retVal[j] -= termToSubtract
		
	return retVal

#The hessian is actually the sum of two matrices: a diagonal matrix and a constant-value matrix.
#We'll write two functions to get both
def priorHessianConst(priorList, vVector):
	total = 0
	for i in range(0, len(vVector)):
		total += float(vVector[i]) / (sum(priorList) + i)**2
	return total

def priorHessianDiag(priorList, uMatrix):
	retVal = [0]*K
	for k in range(0, K):
		for i in range(0, len(uMatrix[k])):
			retVal[k] -= uMatrix[k][i] / (priorList[k] + i)**2
	return retVal

	
# Compute the next value to try here
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
def getPredictedStep(hConst, hDiag, gradient):
	numSum = 0.0
	for i in range(0, K):
		numSum += gradient[i] / hDiag[i]
	
	denSum = 0.0
	for i in range(0, K): denSum += 1.0 / hDiag[i]
	
	b = numSum / ((1.0/hConst) + denSum)

	retVal = [0]*K
	for i in range(0, K): retVal[i] = (b - gradient[i]) / hDiag[i]
	return retVal

# Uses the diagonal hessian on the log-alpha values	
def getPredictedStepAlt(hConst, hDiag, gradient, alphas):
	retVal = [0]*K
	for i in range(0, K): 
		term = alphas[i] * (hDiag[i] + hConst)
		retVal[i] = -1* gradient[i] / (gradient[i] + term)
	return retVal

#The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors, uMatrix, vVector):
	return -1*dirichLogProb(trialPriors, uMatrix, vVector)
	
def predictStepUsingHessian(gradient, priors):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)		
	return getPredictedStep(totalHConst, totalHDiag, gradient)
	
def predictStepUsingDiagHessian(gradient, priors):
	totalHConst = priorHessianConst(priors, vVector)
	totalHDiag = priorHessianDiag(priors, uMatrix)
	return getPredictedStepAlt(totalHConst, totalHDiag, gradient, priors)
	

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, uMatrix, vVector):
	n = len(trialPriors)
	for i in range(0, n): 
		if trialPriors[i] <= 0: 
			return float("inf")
		
	return getTotalLoss(trialPriors, uMatrix, vVector)
	
def sqVectorSize(v):
	s = 0
	for i in range(0, len(v)): s += v[i] ** 2
	return s

#####
# Load Data
#####

csv.field_size_limit(1000000000)
reader = csv.reader(sys.stdin, delimiter='\t')
print "Loading data"
priors = [0.]*K

# Special data vector
uMatrix = []
for i in range(0, K): uMatrix.append([])
vVector = []

i = 0
for row in reader:
	i += 1

	if (random.random() < float(options.sampleRate)):
		data = map(int, row)
		if (len(data) != K):
			print "Error: there are " + str(K) + " categories, but line has " + str(len(data)) + " counts."
			print "line " + str(i) + ": " + str(data)
		
		sumData = sum(data)
		weightForMean = 1.0 / (1.0 + sumData)
		for i in range(0, K): 
			priors[i] += data[i] * weightForMean
			uVector = uMatrix[i]
			for j in range(0, data[i]):
				if (len(uVector) == j): uVector.append(0)
				uVector[j] += 1
			
		for j in range(0, sumData):
			if (len(vVector) == j): vVector.append(0)
			vVector[j] += 1

	if (i % 1000000) == 0: print "Loading Data", i

print "all data loaded into memory"

initPriorWeight = 1
priorSum = sum(priors)
for i in range(0, K): priors[i] /= initPriorWeight * priorSum

# Let the learning begin!!			
learnRate = 100
momentum = [0]*K

#Only step in a positive direction, get the current best loss.
currentLoss = getTotalLoss(priors, uMatrix, vVector)

gradientToleranceSq = 2 ** -30

mixer = 1
count = 0
accepted2 = False
while(count < 10000):
	
	count += 1
	print  count, "Loss: ", currentLoss, ", Priors: ", priors
	
	#Get the data for taking steps
	gradient = priorGradient(priors, uMatrix, vVector)
	if (sqVectorSize(gradient) < gradientToleranceSq):
		print "Converged with small gradient"
		break
	
	trialStep = predictStepUsingHessian(gradient, priors)
	
	#First, try the second order method
	trialPriors = [0]*len(priors)
	for i in range(0, len(priors)): trialPriors[i] = priors[i] + trialStep[i]
	
	#TODO: Check for taking such a small step that the loss change doesn't register (essentially converged)
	#  Fix by ending
	
	loss = testTrialPriors(trialPriors, uMatrix, vVector)
	if loss < currentLoss:
		currentLoss = loss
		priors = trialPriors
		continue
	else:
		trialStep = predictStepUsingDiagHessian(gradient, priors)
		trialPriors = [0]*len(priors)
		for i in range(0, len(priors)): trialPriors[i] = priors[i] * math.exp(trialStep[i])
		loss = testTrialPriors(trialPriors, uMatrix, vVector)
		if loss < currentLoss:
			currentLoss = loss
			priors = trialPriors
			continue
		else:
			print "Converged with no loss improvement"
			break
		
		
print "Final priors: ", priors
print "Final average loss:", getTotalLoss(priors, uMatrix, vVector)

endTime = time.time()
totalTime = endTime - startTime
print "Total Time: " + str(totalTime)
	
	