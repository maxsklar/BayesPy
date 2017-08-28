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
import time
from optparse import OptionParser
import logging
from random import shuffle

startTime = time.time()
parser = OptionParser()
parser.add_option('-m', '--model', dest='model', help='File containing the logistic model')
parser.add_option('-k', '--k', dest='k', help='File containing the logistic model')
parser.add_option('-t', '--testSet', dest='testSet', help='File containing the test set')
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")

(options, args) = parser.parse_args()

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
modelReader = csv.reader(open(options.model, 'r'), delimiter='\t')
logging.debug("Loading model")

featureWeights = {}
numFeatures = 0
K = int(options.k)
for row in modelReader:
  scores = [0.0]*K
  for k in range(0, K): 
    if (len(row) > k + 1):
      scores[k] = float(row[k + 1])
  featureWeights[row[0]] = scores
  numFeatures += 1
  
logging.debug("k=" + str(K))
logging.debug("loaded %s records into memory" % numFeatures)

logging.debug("Reading Test data")

for line in open(options.testSet, 'r'):
  row = line.replace("\n", "").split("\t")
  rowName = int(row[0])
  features = {"__CONST__": 1}
  for i in range(1, len(row)):
    featureStr = row[i]
    featureCutPointA = featureStr.rfind(":")
    featureCutPointB = featureCutPointA + 1
    feature = featureStr[:featureCutPointA]
    count = int(float(featureStr[featureCutPointB:]))
    features[feature] = count
  
  # Add up the features scores
  scores = [0.0] * K
  for feature in features:
    count = features[feature]
    weights = featureWeights.get(feature, [0.0] * K)
    for k in range(0, K): scores[k] += (count * weights[k])

  scoresExp = map(math.exp, scores)
  scoresSum = sum(scoresExp)
  probabilities = map(lambda x: x / scoresSum, scoresExp)

  print str(rowName) + "\t"  + "\t".join(map(lambda x: str(x), probabilities))


calcTime = time.time()
logging.debug("time to calculate: %s " % (calcTime - startTime))
