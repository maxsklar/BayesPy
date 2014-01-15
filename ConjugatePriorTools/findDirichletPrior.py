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
import dirichletMultinomialEstimation as DME
import samplingTools as Sample
from optparse import OptionParser
import logging

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
parser.add_option('-M', '--maxCountPerRow', dest='M', type=int, default=sys.maxint, help='The maximum number of the count per row.  Setting this lower increases the running time')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")

(options, args) = parser.parse_args()
K = int(options.K)

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)
#####
# Load Data
#####

csv.field_size_limit(1000000000)
reader = csv.reader(sys.stdin, delimiter='\t')
logging.debug("Loading data")
priors = [0.]*K

# Special data vector
uMatrix = []
for i in range(0, K): uMatrix.append([])
vVector = []

idx = 0
for row in reader:
	idx += 1

	if (random.random() < float(options.sampleRate)):
		data = map(int, row)
		if (len(data) != K):
			logging.error("There are %s categories, but line has %s counts." % (K, len(data)))
			logging.error("line %s: %s" % (i, data))
		
		
		while sum(data) > options.M: data[Sample.drawCategory(data)] -= 1
		
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

	if (idx % 1000000) == 0: logging.debug("Loading Data: %s rows done" % idx)

dataLoadTime = time.time()
logging.debug("loaded %s records into memory" % idx)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

for row in uMatrix:
	if len(row) == 0:
		raise Exception("You can't have any columns with all 0s")

initPriorWeight = 1
priorSum = sum(priors)
for i in range(0, K): priors[i] /= priorSum

priors = DME.findDirichletPriors(uMatrix, vVector, priors)	
print "Final priors: ", priors
logging.debug("Final average loss: %s" % DME.getTotalLoss(priors, uMatrix, vVector))

totalTime = time.time() - dataLoadTime
logging.debug("Time to calculate: %s" % totalTime)
