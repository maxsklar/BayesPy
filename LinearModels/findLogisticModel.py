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
import logisticRegression as LR
import time
from optparse import OptionParser
import logging
from random import shuffle

startTime = time.time()
parser = OptionParser()
parser.add_option('-s', '--sampleRate', dest='sampleRate', default='1', help='Randomly sample this fraction of rows')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("--L1", '--lassoReg', action="store", dest="L1", default='-1', help="L1 Lasso regularizer param")
parser.add_option("--L2", '--ridgeReg', action="store", dest="L2", default='-1', help="L1 Lasso regularizer param")
parser.add_option("-T", '--hyperparamTuningSetSize', action="store", dest="hyperparamTuningSetSize", default='1000', help="Size of set on which to train hyperparams")
parser.add_option("-H", '--tuningHoldoutPercent', action="store", dest="tuningHoldoutPercent", default='0.5', help="Sample rate for holdout of hyperparameter trainer")
parser.add_option('-i', '--iterations', dest='iterations', default='50', help='How many iterations to do')

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

data = []
labels = []
data_labels = []
idx = 0
for row in reader:
  idx += 1

  if (random.random() < float(options.sampleRate)):
    label = (not (int(row[0]) == 0))

    features = {"__CONST__": 1}
    for i in range(1, len(row)):
      splitRow = row[i].split(":")
      feature = splitRow[0]
      if (len(splitRow) > 1): features[feature] = int(splitRow[1])
      else: features[feature] = 1

    labels.append(label)
    data.append(features)
    data_labels.append([features, label])
  if (idx % 1000000) == 0: logging.debug("Loading Data: %s rows done" % idx)

dataLoadTime = time.time()
logging.debug("loaded %s records into memory" % idx)
logging.debug("time to load memory: %s " % (dataLoadTime - startTime))

L1 = float(options.L1)
L2 = float(options.L2)
if (L1 >= 0):
  logging.debug("Using given L1 regularizer: " + str(L1))
else:
  logging.debug("Finding optimal regularizer")
  shuffle(data_labels)
  tuningSetSize = int(options.hyperparamTuningSetSize)

  trainingSet = []
  trainingLabels = []
  holdoutSet = []
  holdoutLabels = []

  for tuningData, label in data_labels[:tuningSetSize]:
    if (random.random() < float(options.tuningHoldoutPercent)):
      holdoutSet.append(tuningData)
      holdoutLabels.append(label)
    else:
      trainingSet.append(tuningData)
      trainingLabels.append(label)
  (L1, L2) = LR.findOptimalRegulizers(trainingSet, trainingLabels, holdoutSet, holdoutLabels, 0.002, 500)
  logging.debug("optimal regularizer: " + str(L1) + ", " + str(L2))

params = LR.batchCompute(data, labels, L1, L2, 0.001, iterations)	

logging.debug("Printing final weights: ")
for feature in params:
  weight = params[feature]
  print str(feature) + "\t" + str(weight)


totalTime = time.time() - dataLoadTime
