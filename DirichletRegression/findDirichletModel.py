#!/usr/bin/python
#
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2022 Max Sklar

import sys
import csv
import math
import random
import dirichletRegression as DR
import time
from optparse import OptionParser
import logging
from random import shuffle

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("--L1", '--lassoReg', action="store", dest="L1", default='0', help="L1 Lasso regularizer param")
parser.add_option("--L2", '--ridgeReg', action="store", dest="L2", default='0', help="L2 Ridge regularizer param")
parser.add_option("-F", '--featureListFile', action="store", dest="featureListFile", help="A file listing allowed features in the order you want to compute them. The file multiLogisticRegressionBuildFeatureList.py generates this automatically.")
parser.add_option('-i', '--iterations', dest='iterations', default='20', help='How many iterations to do')
parser.add_option('-K', '--K', dest='K', default='2', help='Number of classes')
parser.add_option("-H", '--holdoutPct', dest='H', default='0.0', help="Probability of an item being placed in the holdout set")

(options, args) = parser.parse_args()

iterations = int(options.iterations)
holdoutPct = float(options.H)

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
dataPointAccum = DR.DataPointAccumulator(K, holdoutPct)

logging.debug("Loading feature list")
for line in open(options.featureListFile):
  dataPointAccum.addFeature(line.replace("\n", ""))
dataPointAccum.finalizeFeatures()
logging.debug(" -found " + str(dataPointAccum.numFeatures) + " features")

logging.debug("Loading data")

for line in sys.stdin:
  (label, features) = DR.lineToLabelAndFeatures(line, K)
  dataPointAccum.appendRow(features, label)
  if (dataPointAccum.N % 1000000 == 0): logging.debug("Loading Data: %s rows done" % dataPointAccum.N )

dataLoadTime = time.time()
logging.debug("K = " + str(K))
logging.debug("loaded %s records into training set" % dataPointAccum.N)
logging.debug("loaded %s records into holdout set" % dataPointAccum.N_holdout)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

L1 = float(options.L1)
L2 = float(options.L2)

dataPointAccum.finalize()

logging.debug("finalized data")

baseline, params = DR.batchCompute(dataPointAccum, L1, L2, 0.001, iterations)

logging.debug("Printing final weights: ")

print "__BASELINE__\t" + "\t".join(map(str, baseline))

for featureIx in params:
  feature = dataPointAccum.featureForwardLookup[featureIx]
  weights = params.get(featureIx, [0.0]*K)
  print str(feature) + "\t" + "\t".join(map(str, weights))

totalTime = time.time() - dataLoadTime
