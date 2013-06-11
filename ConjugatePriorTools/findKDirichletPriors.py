#!/usr/bin/python
#
# Finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2012 Max Sklar

# A sample of a file to pipe into this python script is given by test.csv

# Before using this file, see the header for findDirichletPrior
# This file differs from findDirichletPriors in that now we're estimating the data with a mixture of K dirichlet priors

# ex
# cat test.csv | ./finaKDirichletPriors.py --sampleRate 1 -k 5

import sys
import csv
import math
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-k', '--k', dest='k', default='1', help='Find a mixture of this many dirichlet priors')
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

# Define an energy-function to measure a dirichlet prior against data
def dirichletEnergy(priorList, dataList):
	totalPrior = sum(priorList)
	totalData = sum(dataList)
	
	total = 0
	total += partialLogSums(totalPrior, totalData)
	for i in range(0, len(dataList)):
		total -= partialLogSums(priorList[i], dataList[i])
	
	return total

# Gives the derivative with respect to the energy
def dirichletEnergyGradient(priorList, dataList):
	totalPrior = sum(priorList)
	totalData = sum(dataList)
	
	n = len(priorList)
	retVal = [0]*n
	
	for i in range(0, n):
		retVal[i] += partialHarmonic(totalPrior, totalData) 
		retVal[i] -= partialHarmonic(priorList[i], dataList[i])
		
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

def getTotalEnergy(priors, dataPoints):
	currentEnergy = 0
	for sample in dataPoints: 
		currentEnergy += dirichletEnergy(priors, sample)
	return currentEnergy

def getTotalGradient(priors, dataPoints):
	n = len(priors)
	totalGradient = [0]*n
		
	for data in dataPoints:
		gradient = dirichletEnergyGradient(priors, data)
		for i in range(0, n): totalGradient[i] += gradient[i]
	
	return totalGradient
	
def predictStepUsingHessian(dataPoints, gradient, priors):
	n = len(priors)
	totalHConst = 0
	totalHDiag = [0]*n
	
	for weight, data in dataPoints:
		hConst = priorHessianConst(priors, data)
		hDiag = priorHessianDiag(priors, data)
		
		totalHConst += weight*hConst
		for i in range(0, n): totalHDiag[i] += weight*hDiag[i]
			
	return getPredictedStep(totalHConst, totalHDiag, gradient)

# Returns whether it's a good step, and the loss	
def testTrialPriors(trialPriors, dataPoints):
	n = len(trialPriors)
	for i in range(0, n): 
		if trialPriors[i] <= 0: 
			return False, float("inf")
		
	return getTotalLoss(trialPriors, dataPoints)

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
		#data = (1, map(int, row))
		data = map(int, row)
		allData.append(data)

	if (i % 1000000) == 0: print "Loading Data", i


print "all data loaded into memory"

def findNewPrior(prior, dataSample, learnRate):
	trialPriors = [0]*numCategories
	gradient = getTotalGradient(prior, dataSample)
	for i in range(0, len(prior)): trialPriors[i] = prior[i] - learnRate*gradient[i]
	return trialPriors

def adjustDirichletDistribution(prior, responsibility, dataPoints):
	learnRate = 0.005
	
	dataSample = []
	for i in range(0, 50): dataSample.append(pullRandomDataPoint(responsibility, dataPoints))

	# Calculate the currentEnergy
	currentEnergy = getTotalEnergy(prior, dataPoints)
	
	# Calculate the gradient
	newPrior = findNewPrior(prior, dataSample, learnRate)
	
	# See if it worked:
	newEnergy = getTotalEnergy(newPrior, dataPoints)
	energyDiff = newEnergy - currentEnergy
	
	if (energyDiff < 0): return newPrior
	return prior

def pullRandomDataPoint(responsibility, dataPoints):
	choice = random.uniform(0, sum(responsibility))
	
	responsibilityCDF = 0
	for i in range(0, len(dataPoints)):
		responsibilityCDF += responsibility[i]
		if choice <= responsibilityCDF: return dataPoints[i]
	
	print "BUG IN PULL RANDOM DATA POINT"
	exit()
	

# Returns K data lists		
def assignResponsibilitiesToData(dataPoints, priorLists):
	dataResponsibilityLists = []
	k = int(options.k)
	for i in range(0, k): dataResponsibilityLists.append([])
	for data in dataPoints:
		resp = map(lambda x: math.exp(-1 * dirichletEnergy(x, data)), priorLists)
		totalWeight = sum(resp)
		for i in range(0, k):
			dataResponsibilityLists[i].append(resp[i]/totalWeight)
	return dataResponsibilityLists		
			
def initPriors():
	priorLists = []
	for i in range(0, int(options.k)): 
		priorList = []
		for j in range(0, numCategories): priorList.append(random.gammavariate(1, 1))
		priorLists.append(priorList)
	return priorLists
	
def adjustKDirichlets(priorLists, dataResponsibilityLists, dataPoints):
	priors = []
	for i in range(0, int(options.k)):
		newPrior = adjustDirichletDistribution(priorLists[i], dataResponsibilityLists[i], dataPoints)
		priors.append(newPrior)
	return priors
	
def learnKDirichByEM(dataPoints, numIterations, n):
	priorLists = initPriors()
	for i in range(0, numIterations):
		print "E-M ITERATION", i
		dataResponsibilityLists = assignResponsibilitiesToData(dataPoints, priorLists)
		priorLists = adjustKDirichlets(priorLists, dataResponsibilityLists, dataPoints)
		print "Prior Lists: ", priorLists
	
	return priorLists, dataResponsibilityLists

pLists, rLists = learnKDirichByEM(allData, 300, numCategories)
print "priors:", pLists
print "Relative weights:", map(sum, rLists)

#learned = learnDirichletDistribution(allData, 1000)		
#print "Final priors: ", learned["priors"]
#print "Final average loss:", learned["avgLoss"]
	
	