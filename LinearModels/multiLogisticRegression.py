#!/usr/bin/python
#
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import logging
import random
import scipy.special as mathExtra
import numpy.random as R
import logisticRegression as LRUtil

class DataPointAccumulator:
  size = 0
  
  def __init__(self, K):
    self.K = K
    self.dataPoints = []
    self.labels = []
    self.labelCounts = [0]*K
    self.featureToDataPointIxs = {}
    
    # Finalized vars
    self.sortedFeatures = []
    
  def appendRow(self, dataPoint, label):
    self.dataPoints.append(dataPoint)
    self.labels.append(label)
    
    for feature in dataPoint:
      if (feature not in self.featureToDataPointIxs):
        self.featureToDataPointIxs[feature] = []
      self.featureToDataPointIxs[feature].append(self.size)
    
    self.size += 1
    self.labelCounts[label] += 1
    
  def finalize(self, maxFeatures):
    self.sortedFeatures = sorted(self.featureToDataPointIxs, key=(lambda x: -len(self.featureToDataPointIxs.get(x))))
    self.sortedFeatures = self.sortedFeatures[:maxFeatures]
    
    self.__CONST__ = map(lambda x: math.log(float(x) / size), self.labelCounts)
    logging.debug("CONST: " + str(self.__CONST__))
    
    newMap = {}
    for feature in self.sortedFeatures:
      newMap[feature] = self.featureToDataPointIxs[feature]
    self.featureToDataPointIxs = newMap

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# K: the number of different categories, or possible labels
# labels: same length as dataPoints. Each label is a number from 0 to K-1
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
# convergence: a small number to detect convergence
def batchCompute(dataPointAccumulator, L1, L2, convergence, maxIters, allowLogging = True):
  scores = []
  for i in range(0, dataPointAccumulator.size): scores.append(dataPointAccumulator.__CONST__)
  params = {}
  for i in range(0, maxIters):
    (maxDist, maxDistF, maxDistD) = batchStep(dataPointAccumulator, L1, L2, params, scores, allowLogging)
    if (allowLogging):
      logging.debug("Iteration " + str(i) + ", Dist: " + str(maxDist) + " on " + maxDistF + ":" + str(maxDistD) + " now " + str(params.get(maxDistF, [0.0, 0.0, 0.0])) + ", Features: " + str(len(params)))
    if (maxDist < convergence):
      if (allowLogging): logging.debug("Converge criteria met.")
      return params

  if (allowLogging): logging.debug("Convergence did not occur in " + str(maxIters) + " iterations")
  return params

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# labels: same length as dataPoints. Each label is a boolean
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
#
# Returns: (newParams, distance, avgLoss)
# distance: maximum distance between new and old params
def batchStep(dataPointAccumulator, L1, L2, params, scores, allowLogging = True):
  numDatapoints = dataPointAccumulator.size
  totalLoss = 0.0
  
  maxDistance = 0.0
  featureWithMaxDistance = ''
  dimWithMaxDistance = 0
  K = dataPointAccumulator.K
  
  for feature in dataPointAccumulator.sortedFeatures:
    #print "**", feature
    featureDeriv = [0.0]*K
    
    # This really should be a 2D hession, but for now use the diagonal hessian
    diagHessian = [0.0]*K
    
    dataPointIxs = dataPointAccumulator.featureToDataPointIxs[feature]
    for dataPointIx in dataPointIxs:
      count = dataPointAccumulator.dataPoints[dataPointIx][feature]
      label = dataPointAccumulator.labels[dataPointIx]
      currentEnergies = scores[dataPointIx]
      currentExpEnergies = map(math.exp, currentEnergies)
      currentExpEnergiesSum = sum(currentExpEnergies)
      prob = currentExpEnergies[label] / currentExpEnergiesSum
      
      for k in range(0, K):
        probIx = (currentExpEnergies[k] / currentExpEnergiesSum)
        featureDeriv[k] += count * probIx
        if (k == label): featureDeriv[k] -= count
        
        # Diagonal part of the hessian
        diagHessian[k] += (count ** 2) * (currentExpEnergiesSum ** -1) * currentExpEnergies[k] + (count ** 2) * (currentExpEnergiesSum ** -2) * currentExpEnergies[k] * currentExpEnergies[k]
    
    for i in range(0, K):
      featureDeriv[i] /= numDatapoints
      diagHessian[i] /= numDatapoints
        
    # Add L2 regularization
    currentValues = params.get(feature, [0.0]*K)
    for i in range(0, K):
      featureDeriv[i] += L2*currentValues[i]
      diagHessian[i] += L2
      
    # Add L1 regularization (tricky!)
    for i in range(0, K):
      if (currentValues[i] > 0 or (currentValues[i] == 0 and featureDeriv[i] < -L1)):
        featureDeriv[i] += L1
      elif (currentValues[i] < 0 or (currentValues[i] == 0 and featureDeriv[i] > L1)):
        featureDeriv[i] -= L1
      else: # Snap-to-zero
        featureDeriv[i] = 0
    
    diffs = [0.0] * K
    for i in range(0, K):
      diffs[i] += featureDeriv[i] / diagHessian[i]
    
    # Check if any diffs cause the values to cross 0. If they do, snap to zero!
    snap = 1.0
    zeroOut = -1
    for i in range(0, K):
      if (currentValues[i] > 0):
        if (snap * diffs[i] > currentValues[i]):
          snap = currentValues[i] / diffs[i]
          zeroOut = i
      elif (currentValues[i] < 0):
        if (snap*diffs[i] < currentValues[i]):
          snap = currentValues[i] / diffs[i]
          zeroOut = i
    
    newValues = [0.0] * K
    for i in range(0, K):
      if (zeroOut != i):
        newValues[i] = currentValues[i] - diffs[i]
    
    for i in range(0, K):
      distance = abs(newValues[i] - currentValues[i])
      if (distance > maxDistance):
        maxDistance = distance
        featureWithMaxDistance = feature
        dimWithMaxDistance = i
          
    #Update Feature Weight
    if (all(v == 0.0 for v in newValues)): 
      if feature in params: del params[feature]
    else: params[feature] = newValues
    
    # Update Scores Vector
    for dataPointIx in dataPointAccumulator.featureToDataPointIxs[feature]:
      dataPoint = dataPointAccumulator.dataPoints[dataPointIx]
      count = dataPoint[feature]
      for i in range(0, K):
        scores[dataPointIx][i] += count * (newValues[i] - currentValues[i])
          
  return (maxDistance, featureWithMaxDistance, dimWithMaxDistance)

def energy(dataPoint, params, K):
  total = [0.0] * K
  for feature in dataPoint:
    param = params.get(feature, [0.0] * K)
    for i in range(0, K): total[i] += dataPoint[feature] * param[i]
  return total

def computeLossForDataset(dataPointAccumulator, params):
  K = dataPointAccumulator.K
  totalLoss = 0
  totalDataPoints = 0
  for i in range(0, dataPointAccumulator.size):
    dataPoint = dataPointAccumulator.dataPoints[i]
    label = dataPointAccumulator.labels[i]
    totalLoss += computeLossForDatapoint(dataPoint, label, params, K)
    totalDataPoints += 1
  return totalLoss / totalDataPoints

def computeLossForDatapoint(dataPoint, label, params, K):
  E = energy(dataPoint, params, K)
  expEnergies = map(math.exp, E)
  return math.log(sum(expEnergies)) - E[label]