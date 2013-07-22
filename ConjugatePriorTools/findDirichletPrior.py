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
import dirichletPriorTools as DPT
from optparse import OptionParser

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
parser.add_option('-V', '--verbose', dest='V', default="True", help='Whether the print out the debug information in the calculation')
(options, args) = parser.parse_args()
K = int(options.K)
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
for i in range(0, K): priors[i] /= priorSum

verbose = options.V == "True"
priors = DPT.findDirichletPriors(uMatrix, vVector, priors, verbose)	
print "Final priors: ", priors
print "Final average loss:", DPT.getTotalLoss(priors, uMatrix, vVector)

endTime = time.time()
totalTime = endTime - startTime
print "Total Time: " + str(totalTime)
	
	