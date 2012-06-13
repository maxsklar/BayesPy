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
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
(options, args) = parser.parse_args()

# think of this as log((x + k -1)! / (x+1)!)
# or: log(x) + log(x+1) + log(x+2) + ... + log(x+k-1)
def partialLogSums(x, k): return sum(map(lambda i: math.log(x + i), range(0, k)))

# This is the derivative of partial Log Sums
# 1/x + 1/(x+1) + 1/(x+2) + ... + 1/(x+k-1)	
def partialHarmonic(x, k): return sum(map(lambda i: 1.0 / (x + i), range(0, k)))

# This is the second derivative of partial Log Sums
# -1/x^2 + -1/(x+1)^2 + -1/(x+2)^2 + ... + -1/(x+k-1)^2
def partialHarmonicPrime(x, k): return sum(map(lambda i: -1.0 / (x + i)**2, range(0, k)))

#Find the log probability that we see a certain set of data
# give our prior.
def dirichLogProb(priorList, dataList):
	totalPrior = sum(priorList)
	totalData = sum(dataList)
	
	total = 0
	total -= partialLogSums(totalPrior, totalData)
	total += partialLogSums(1, totalData)
	for i in range(0, len(dataList)):
		total += partialLogSums(priorList[i], dataList[i])
		total -= partialLogSums(1, dataList[i])
	
	return total

#Gives the derivative with respect to the log of prior.  This will be used to adjust the loss
def priorGradient(priorList, dataList):
	totalPrior = sum(priorList)
	totalData = sum(dataList)
	
	n = len(priorList)
	retVal = [0]*n
	
	for i in range(0, n):
		retVal[i] -= partialHarmonic(totalPrior, totalData) 
		retVal[i] += partialHarmonic(priorList[i], dataList[i])
		
	return retVal

#The hessian is actually the sum of two matrices: a diagonal matrix and a constant-value matrix.
#We'll write two functions to get both
def priorHessianConst(priorList, dataList):
	totalPrior = sum(priorList)
	totalData = sum(dataList)
	return -1* partialHarmonicPrime(totalPrior, totalData)

def priorHessianDiag(priorList, dataList):
	n = len(priorList)
	retVal = [0]*n
	for i in range(0, n):
		retVal[i] += partialHarmonicPrime(priorList[i], dataList[i])
	return retVal
	
# Compute the next value to try here
# http://research.microsoft.com/en-us/um/people/minka/papers/dirichlet/minka-dirichlet.pdf (eq 18)
def getPredictedStep(hConst, hDiag, gradient):
	n = len(gradient)
	
	numSum = 0.0
	for i in range(0, n):
		numSum += gradient[i] / hDiag[i]
	
	denSum = 0.0
	for i in range(0, n): denSum += 1.0 / hDiag[i]
	
	b = numSum / ((1.0/hConst) + denSum)

	retVal = [0]*n
	for i in range(0, n): retVal[i] = (b - gradient[i]) / hDiag[i]
	return retVal

#The priors and data are global, so we don't need to pass them in
def getTotalLoss(trialPriors):
	totalLoss = 0	
	for data in allData:
		totalLoss += -1*dirichLogProb(trialPriors, data)
	return totalLoss

def getTotalGradient():
	n = len(priors)
	totalGradient = [0]*n
		
	for data in allData:
		gradient = priorGradient(priors, data)
		for i in range(0, n): totalGradient[i] += gradient[i]
	
	return totalGradient
	
def predictStepUsingHessian(sample, gradient, priors):
	n = len(priors)
	totalHConst = 0
	totalHDiag = [0]*n
	
	for data in sample:
		hConst = priorHessianConst(priors, data)
		hDiag = priorHessianDiag(priors, data)
		
		totalHConst += hConst
		for i in range(0, n): totalHDiag[i] += hDiag[i]
			
	return getPredictedStep(totalHConst, totalHDiag, gradient)

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors):
	n = len(trialPriors)
	for i in range(0, n): 
		if trialPriors[i] <= 0: 
			return False, float("inf")
		
	return getTotalLoss(trialPriors)

#####
# Load Data
#####

csv.field_size_limit(1000000000)
reader = csv.reader(sys.stdin, delimiter='\t')
print "Loading data"
allData = []
i = 0

numCategories = 0
for row in reader:
	i += 1
	numCategories = max(numCategories, len(row))

	if (random.random() < options.sampleRate):
		data = map(int, row)
		allData.append(data)

	if (i % 1000000) == 0: print "Loading Data", i

print "all data loaded into memory"

priors = [1.0 / numCategories]*numCategories

# Let the learning begin!!			
learnRate = 100
momentum = [0]*numCategories

#Only step in a positive direction, get the current best loss.
currentLoss = getTotalLoss(priors)

learnRateChange = 1.5
momentumDecay = .9

mixer = 1
count = 0
accepted2 = False
while(count < 10000):
	
	count += 1
	end = "USE HESSIAN"
	if (not accepted2): end = "Learn: " + str(learnRate) + ", Momentum: " + str(momentum)
	print  count, "Loss: ", (currentLoss / len(allData)), ", Priors: ", priors, "," + end
	
	#Get the data for taking steps
	gradient = getTotalGradient()
	trialStep = predictStepUsingHessian(allData, gradient, priors)
	
	#First, try the second order method
	trialPriors = [0]*len(priors)
	for i in range(0, len(priors)): trialPriors[i] = priors[i] + trialStep[i]
	
	loss = testTrialPriors(trialPriors)
	
	accepted2 = loss < currentLoss
	if loss < currentLoss:
		currentLoss = loss
		priors = trialPriors
		continue
		
	#It didn't work, so we're going to fall back to gradient methods:
	
	trialPriors = [0]*len(priors)
	for i in range(0, len(priors)): trialPriors[i] = priors[i] + learnRate*gradient[i] + momentum[i]
	
	loss = testTrialPriors(trialPriors)
	accepted = loss < currentLoss
	
	if (accepted):
		currentLoss = loss
		priors = trialPriors
		learnRate *= learnRateChange
		for i in range(0, len(priors)): momentum[i] = momentumDecay*momentum[i] + learnRate*gradient[i] + momentum[i]
		continue
	
	#Lower the learning rate until it's accepted
	for i in range(0, len(priors)): momentum[i] = 0
	
	while(not accepted):
		learnRate /= learnRateChange
		trialPriors = [0]*len(priors)
		for i in range(0, len(priors)): trialPriors[i] = priors[i] + learnRate*gradient[i]
		loss = testTrialPriors(trialPriors)
		if (loss < currentLoss): break
		if (learnRate < 2**(-40)): break
		
	if learnRate > 2**(-40): 
		currentLoss = loss
		priors = trialPriors
		for i in range(0, len(priors)): momentum[i] = learnRate*gradient[i]
	else:
		print "Converged"
		break
		
		
print "Final priors: ", priors
print "Final average loss:", getTotalLoss(priors) / len(allData)
	
	