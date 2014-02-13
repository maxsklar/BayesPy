#!/usr/bin/python
#
# Finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2012 Max Sklar

import sys
import csv
import logging
import math
import random
import time
import dirichletEstimation as DE
from optparse import OptionParser

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")

(options, args) = parser.parse_args()
K = int(options.K)
#####
# Load Data
#####

csv.field_size_limit(1000000000)
reader = csv.reader(sys.stdin, delimiter='\t')
print "Loading data"
priors = [1.0/K]*K

# Special data vector
ss = [0]*K

i = 0
for row in reader:
	i += 1

	if (random.random() < float(options.sampleRate)):
		data = map(float, row)
		if (len(data) != K):
			print "Error: there are " + str(K) + " categories, but line has " + str(len(data)) + " counts."
			print "line " + str(i) + ": " + str(data)
		
		for k in range(0, K): ss[k] += math.log(data[k])

	if (i % 1000000) == 0: print "Loading Data", i

for k in range(0, K): ss[k] /= i

dataLoadTime = time.time()
logging.debug("all data loaded into memory")
logging.debug("time to load memory: ", dataLoadTime - startTime)

priors = DE.findDirichletPriors(ss, priors)	
print "Final priors: ", priors
logging.debug("Final average loss:", DE.getTotalLoss(priors, ss))
logging.debug("best loss: ", DE.getTotalLoss([1,2], ss))

totalTime = time.time() - dataLoadTime
logging.debug("Time to calculate: " + str(totalTime))
	
	