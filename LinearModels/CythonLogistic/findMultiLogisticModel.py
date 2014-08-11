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
parser.add_option("-F", '--featureListFile', action="store", dest="featureListFile", help="A file listing allowed features in the order you want to compute them. The file multiLogisticRegressionBuildFeatureList.py generates this automatically.")
parser.add_option('-i', '--iterations', dest='iterations', default='50', help='How many iterations to do')
parser.add_option('-K', '--K', dest='K', default='2', help='Number of classes')
parser.add_option('-N', '--N', dest='N', default='2', help='Number of datapoints')

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

K = int(options.K)
N = int(options.N)
dataPointAccum = MLR.DataPointAccumulator(K, N)

logging.debug("Loading feature list")
for line in open(options.featureListFile):
  dataPointAccum.addFeature(line.replace("\n", ""))
dataPointAccum.finalizeFeatures()

logging.debug("Loading data")

for line in sys.stdin:
  (label, features) = MLR.lineToLabelAndFeatures(line)
  dataPointAccum.appendRow(features, label)
  if (dataPointAccum.dataPointIx % 1000000 == 0): logging.debug("Loading Data: %s rows done" % dataPointAccum.dataPointIx )

dataLoadTime = time.time()
logging.debug("K = " + str(K))
logging.debug("N = " + str(N))
logging.debug("loaded %s records into memory" % dataPointAccum.dataPointIx)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

L1 = float(options.L1)
L2 = float(options.L2)

dataPointAccum.finalize()

logging.debug("finalized data")

params = MLR.batchCompute(dataPointAccum, L1, L2, 0.001, iterations)

logging.debug("Printing final weights: ")
print "__CONST__\t" + "\t".join(map(str, dataPointAccum.__CONST__))
for featureIx in params:
  feature = dataPointAccum.featureForwardLookup[featureIx]
  weights = params[featureIx]
  print str(feature) + "\t" + "\t".join(map(str, weights))

totalTime = time.time() - dataLoadTime
logging.debug("Total time: " + str(totalTime))
