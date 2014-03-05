#!/usr/bin/python
#
# Finding the optimal dirichlet mixture from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2014 Max Sklar

# ex
# cat test.csv | ./finaDirichletPrior.py --sampleRate 1

import sys
import csv
import math
import random
import time
import dirichletMixtureEstimation as DMX
import samplingTools as Sample
from optparse import OptionParser
import logging

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option('-K', '--numCategories', dest='K', default='2', help='The number of (tab separated) categories that are being counted')
parser.add_option('-C', '--numComponents', dest='C', default='2', help='The number of components desired for the model')
parser.add_option('-M', '--maxCountPerRow', dest='M', type=int, default=sys.maxint, help='The maximum number of the count per row.  Setting this lower increases the running time')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option('-H', '--hyperPrior', dest='H', default="0,0,1", help='The hyperprior of the Dirichlet (paper coming soon!) comma separated K+1 values (Beta then W)')
parser.add_option('-D', '--mixtureDirichlet', dest='D', default="", help='a dirichlet-prior for the C-dimensional mixture')
parser.add_option('-i', '--iterations', dest='iterations', default='50', help='How many EM iterations to do')


(options, args) = parser.parse_args()

K = int(options.K)

iterations = int(options.iterations)
#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

logging.debug("K = " + str(K))

W = 0
Beta = [0]*K
Hstr = options.H.split(",")
hasHyperprior = False
if (len(Hstr) == K + 1):
	for i in range(0, K): Beta[i] = float(Hstr[i])
	W = float(Hstr[K])
	hasHyperprior = True
else:
	Beta = None
	W = None

logging.debug("Beta = " + str(Beta))
logging.debug("W = " + str(W))
	
#####
# Load Data
#####

csv.field_size_limit(1000000000)
reader = csv.reader(sys.stdin, delimiter='\t')
logging.debug("Loading data")
dataObj = []

idx = 0
for row in reader:
  idx += 1
  
  if (random.random() < float(options.sampleRate)):
    data = map(int, row)
    if (len(data) != K):
      logging.error("There are %s categories, but line has %s counts." % (K, len(data)))
      logging.error("line %s: %s" % (i, data))
    
    while sum(data) > options.M: data[Sample.drawCategory(data)] -= 1
    dataObj.append(data)
  
  if (idx % 1000000) == 0: logging.debug("Loading Data: %s rows done" % idx)

dataLoadTime = time.time()
logging.debug("loaded %s records into memory" % idx)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

# TODO(max): enforce this
#for row in dataObj:
#	if len(row) == 0 and not hasHyperprior:
#		# TODO(max): write up a paper describing the hyperprior and link it.
#		raise Exception("You can't have any columns with all 0s, unless you provide a hyperprior (-H)")

# Mixture hyperparams (the mixture itself has a dirichlet prior)
D = map(float, options.D.split(","))
C = len(D)

hyperparams = DMX.DirichletMixtureModelHyperparams(C, K, Beta, W, D)
model = DMX.initMixtureModel(dataObj, hyperparams)
logging.debug("Model Initialized")
model.logToDebug()
for i in range(0, iterations):
  model = DMX.updateMixtureModel(dataObj, model, hyperparams)
  logging.debug("Finished Iteration " + str(i))
  model.logToDebug()

logging.debug("Finished iterating, ready to print model")
model.outputToFile(sys.stdout)

totalTime = time.time() - dataLoadTime
logging.debug("Time to calculate: %s" % totalTime)
