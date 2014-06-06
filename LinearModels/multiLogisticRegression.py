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

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# K: the number of different categories, or possible labels
# labels: same length as dataPoints. Each label is a number from 0 to K-1
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
# convergence: a small number to detect convergence
def batchCompute(dataPoints, K, labels, L1, L2, convergence, maxIters, allowLogging = True):
  # Some pre-processing
  scores = []
  for i in range(0, len(dataPoints)): scores.append([0.0] * K)
  sortedFeatures = sorted(featuresToDataPointIxs, key=(lambda x: -len(featuresToDataPointIxs.get(x))))
  
  featuresToDataPointIxs = LRUtil.createFeaturesToDatapointIxMap(dataPoints)

  params = {}
  for i in range(0, maxIters):
    (maxDist, maxDistF, maxDistD) = batchStep(dataPoints, K, labels, L1, L2, params, scores, featuresToDataPointIxs, sortedFeatures, allowLogging)
    if (allowLogging):
      dataLoss = computeLossForDataset(dataPoints, labels, params, K)
      logging.debug("Iteration " + str(i) + ", Loss: " + dataLoss + ", Dist: " + str(maxDist) + " on " + maxDistF + ":" + str(maxDistD) + " now " + str(params.get(maxDistF, [0.0, 0.0, 0.0])) + ", Features: " + str(len(params)))
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
def batchStep(dataPoints, K, labels, L1, L2, params, scores, featuresToDataPointIxs, sortedFeatures, allowLogging = True):
  numDatapoints = len(dataPoints)
  totalLoss = 0.0
  
  maxDistance = 0.0
  featureWithMaxDistance = ''
  dimWithMaxDistance = 0
  
  for feature in sortedFeatures:
    featureDeriv = [0.0]*K
    
    # This really should be a 2D hession, but for now use the diagonal hessian
    featureDeriv2 = [0.0]*K
    
    for dataPointIx in featuresToDataPointIxs[feature]:
      count = dataPoints[dataPointIx][feature]
      label = labels[dataPointIx]
      currentEnergies = scores[dataPointIx]
      currentExpEnergies = map(math.exp, currentEnergies)
      
      prob = currentExpEnergies[label] / sum(currentExpEnergies)
      
      for i in range(0, K):
        probIx = (currentExpEnergies[i] / sum(currentExpEnergies))
        featureDeriv[i] += probIx
        if (i == label): featureDeriv[i] -= 1
        featureDeriv2[i] += probIx * (1 - probIx)
    
    for i in range(0, K):
      featureDeriv[i] /= numDatapoints
      featureDeriv2[i] /= numDatapoints
        
    # Add L2 regularization
    currentValues = params.get(feature, 0)
    for i in range(0, K):
      featureDeriv[i] += L2*currentValues[i]
      featureDeriv2[i] += L2
      
    # Add L1 regularization (tricky!)
    for i in range(0, K):
      if (currentValues[i] > 0 or (currentValues[i] == 0 and featureDeriv[i] < -L1)):
        featureDeriv += L1
      elif: (currentValues[i] < 0 or (currentValues[i] == 0 and featureDeriv[i] > L1)):
        featureDeriv -= L1
      else: # Snap-to-zero
        featureDeriv[i] = 0
    
    diffs = [0.0] * K
    for i in range(0, K): diffs[i] = featureDeriv[i] / featureDeriv2[i]
    
    # Check if any diffs cause the values to cross 0. If they do, snap to zero!
    snap = 1.0
    zeroOut = -1
    for i in range(0, K):
      if (currentValues[i] > 0):
        if (snap * diffs[i] > currentValues[i]):
          snap = currentValues[i] / diffs[i]
          zeroOut = i
      elif: (currentValues[i] < 0):
        if (snap*diffs[i] < currentValues[i]):
          snap = currentValues[i] / diffs[i]
          zeroOut = i
    
    newValues = [0.0] * i
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
    if (all(v == 0.0 for v in newValues)): del params[feature]
    else: params[feature] = newValues
    if (newValue != 0): params[feature] = newValue
    else:
      if (feature in params): del params[feature]
    
    # Update Scores Vector
    for dataPointIx in featuresToDataPointIxs[feature]:
      dataPoint = dataPoints[dataPointIx]
      count = dataPoint[feature]
      for i in range(0, K):
        scores[dataPointIx][i] += count * (newValues[i] - currentValues[i])
          
  return (maxDistance, featureWithMaxDistance, dimWithMaxDistance)

def energy(dataPoint, params):
  total = [0.0] * K
  for feature in dataPoint:
    param = params.get(feature, [0.0] * K)
    for i in range(0, K): total[i] += dataPoint[feature] * param[i]
  return total

# Parameter Tuning
# Needs logLevel to turn logging off for each run
# return (L1, L2) regularizers
def findOptimalRegulizers(trainingSet, trainingLabels, testSet, testLabels, conv, maxIter):
  logL1 = 0
  logL2 = 0
  currentLoss = float("inf")
  numRejects = 0
  while numRejects < 10:
    changeL1 = (R.normal() > 0)
    newLogL1 = logL1
    newLogL2 = logL2
    if (changeL1): newLogL1 = logL1 + R.normal()
    else: newLogL2 = logL2 + R.normal()
    
    L1 = math.exp(newLogL1)
    L2 = math.exp(newLogL2)
    
    params = batchCompute(trainingSet, trainingLabels, L1, L2, conv, maxIter, False)
    avgLoss = computeLossForDataset(testSet, testLabels, params)
    
    accept = avgLoss < currentLoss
    logging.debug("New " + ("L1" if changeL1 else "L2") + ": L1 = " + str(L1) + ", L2 = " + str(L2) + ", loss: " + str(avgLoss) + ", " + ("ACCEPT" if accept else "REJECT"))
    if (accept):
      currentLoss = avgLoss
      logL1 = newLogL1
      logL2 = newLogL2
      numRejects = 0
    else:
      numRejects += 1
  return (math.exp(logL1), math.exp(logL2))

def computeLossForDataset(dataPoints, labels, params):
  totalLoss = 0
  totalDataPoints = 0
  for dataPoint, label in zip(dataPoints, labels):
    totalLoss += computeLossForDatapoint(dataPoint, label, params)
    totalDataPoints += 1
  return totalLoss / totalDataPoints

def computeLossForDatapoint(dataPoint, label, params):
  E = energy(dataPoint, params)
  expEnergies = map(math.exp, E)
  return math.log(sum(expEnergies)) - E[label]
  
def computeTrainingLossForDataset(dataPoints, labels, L1, params):
  datasetLoss = computeLossForDataset(dataPoints, labels, params)
  l1Loss = 0
  for feature in params: l1Loss += abs(params[feature])
  l1Loss *= L1
  return datasetLoss + l1Loss