#!/usr/bin/python
#
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

totalLoss = 0.0
numDataPoints = 0
numCorrect = 0
numWithinOne = 0
numWithinTwo = 0
totalDistance = 0.0

confusionCountMatrix = []
probabilisticConfusion = []
for k in range(0, K):
  confusionCountMatrix.append([0] * K)
  probabilisticConfusion.append([0] * K)

for line in open(options.testSet, 'r'):
  row = line.replace("\n", "").split("\t")
  label = int(row[0])
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
  
  labelWeight = scores[label]

  exponentiateScores = map(math.exp, scores)
  sumOfExpScores = sum(exponentiateScores)
  
  totalLoss += math.log(sumOfExpScores)
  totalLoss -= labelWeight

  probabilities = map(lambda x: x / sumOfExpScores, exponentiateScores)

  # Which Scores is highest
  highestFeature = 0
  for k in range(0, K):
    probabilisticConfusion[label][k] += probabilities[k]
    if (scores[k] > scores[highestFeature]):
      highestFeature = k

  if (highestFeature == label): numCorrect += 1

  distance = abs(highestFeature - label)

  if (distance <= 1): numWithinOne += 1
  if (distance <= 2): numWithinTwo += 1
  totalDistance += distance

  confusionCountMatrix[label][highestFeature] += 1

  numDataPoints += 1

print "Total Loss: " + str(totalLoss) 
print "Num datapoints: " + str(numDataPoints)
print "Num correct: " + str(numCorrect) + " Prob: " + str(float(numCorrect)/ numDataPoints)
print "Num within one: " + str(numWithinOne) + " Prob: " + str(float(numWithinOne)/ numDataPoints)
print "Num within two: " + str(numWithinTwo) + " Prob: " + str(float(numWithinTwo)/ numDataPoints)

print "Average Loss: " + str(totalLoss / numDataPoints)

print "Confusion Matrix on Highest Probability"
print "\texamples",
for k in range(0, K): print "\tPredict " + str(k),
print ""
for i in range(0, len(confusionCountMatrix)):
  row = confusionCountMatrix[i]
  s = sum(row)
  if (s == 0): s = 1
  r = map(lambda x: "{0:.0f}%".format(float(x) * 100 / s), row)

  print "Label " + str(i) + "\t" + str(s) + "\t" + "\t".join(map(str, r))

print "Probabilistic Confusion Matrix"
print "\texamples",
for k in range(0, K): print "\tPredict " + str(k),
print ""
for i in range(0, len(probabilisticConfusion)):
  row = probabilisticConfusion[i]
  s = sum(row)
  if (s == 0): s = 1
  r = map(lambda x: "{0:.0f}%".format(float(x) * 100 / s), row)

  print "Label " + str(i) + "\t" + str(s) + "\t" + "\t".join(map(str, r))

calcTime = time.time()
logging.debug("time to calculate loss: %s " % (calcTime - startTime))
