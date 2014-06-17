#!/usr/bin/python
#
# THIS MODEL IS EXPERIMENTAL
#
# Finding the optimal dirichlet prior from counts
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2014 Max Sklar

# A sample of a file to pipe into this python script is given by logisticRegressionTest.csv

# ex
# cat logisticRegressionTest.csv | ./findLogisticModel.py

# Each row contains first the label, then a series of feature:value pairs separated by tabs
# If the feature isn't given at all, it is assumed to be 0
# If the feature is give with no value, the value is assumed to be 1

import sys
import csv
import math
import random
import multiLogisticRegression as MLR
import time
from optparse import OptionParser
import logging
from random import shuffle

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("--L1", '--lassoReg', action="store", dest="L1", default='-1', help="L1 Lasso regularizer param")
parser.add_option("--L2", '--ridgeReg', action="store", dest="L2", default='0', help="L2 Ridge regularizer param")
parser.add_option("--M", '--maxFeatures', action="store", dest="M", default='0', help="Only consider the M most popular features.")
parser.add_option('-i', '--iterations', dest='iterations', default='50', help='How many iterations to do')
parser.add_option('-K', '--K', dest='K', default='2', help='Number of classes')

(options, args) = parser.parse_args()

iterations = int(options.iterations)

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

K = int(options.K)
dataPointAccum = MLR.DataPointAccumulator(K)
for line in sys.stdin:
  row = line.replace("\n", "").split("\t")
  
  if (random.random() < float(options.sampleRate)):
    label = int(row[0])
   
    features = {}
    for i in range(1, len(row)):
      featureStr = row[i]
      if (featureStr == "__CONST__"): continue
      featureCutPointA = featureStr.rfind(":")
      featureCutPointB = featureCutPointA + 1
      feature = featureStr[:featureCutPointA]
      count = int(float(featureStr[featureCutPointB:]))
      features[feature] = count
    
    dataPointAccum.appendRow(features, label)
    if (dataPointAccum.size % 1000000 == 0): logging.debug("Loading Data: %s rows done" % dataPointAccum.size )

dataLoadTime = time.time()
logging.debug("k=" + str(K))
logging.debug("loaded %s records into memory" % dataPointAccum.size)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

L1 = float(options.L1)
L2 = float(options.L2)

dataPointAccum.finalize(int(options.M))

logging.debug("finalized data")

params = MLR.batchCompute(dataPointAccum, L1, L2, 0.001, iterations, int(options.M))

logging.debug("Printing final weights: ")
print "__CONST__\t" + "\t".join(map(str, dataPointAccum.__CONST__))
for feature in params:
  weights = params[feature]
  print str(feature) + "\t" + "\t".join(map(str, weights))

totalTime = time.time() - dataLoadTime
