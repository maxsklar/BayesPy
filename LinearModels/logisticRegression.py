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

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# labels: same length as dataPoints. Each label is a boolean
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
# convergence: a small number to detect convergence
def batchCompute(dataPoints, labels, L1, L2, convergence, maxIters, allowLogging = True):
  params = {}
  for i in range(0, maxIters):
    (params, maxDist, maxDistF) = batchStep(dataPoints, labels, L1, L2, params, allowLogging)
    if (allowLogging): logging.debug("Iteration " + str(i) + ", Dist: " + str(maxDist) + " on " + maxDistF)
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
def batchStep(dataPoints, labels, L1, L2, params, allowLogging = True):
  featureDerivs = {}
  feature2ndDerivs = {}
  totalDatapointLoss = 0.0
  numDatapoints = 0

  for dataPoint, label in zip(dataPoints, labels):
    currentEnergy = energy(dataPoint, params)
    expEnergy = math.exp(currentEnergy)
    for feature in dataPoint:
      count = dataPoint[feature]
      
      d1 = derivativeForFeature(label, count, expEnergy)
      featureDerivs[feature] = featureDerivs.get(feature, 0) + d1
      
      d2 = secondDerivativeForFeature(count, expEnergy)
      feature2ndDerivs[feature] = feature2ndDerivs.get(feature, 0) + d2
    totalDatapointLoss += math.log(expEnergy + 1)
    if (label): totalDatapointLoss -= currentEnergy
    numDatapoints += 1
  
  totalLoss = totalDatapointLoss / numDatapoints
  if (allowLogging): logging.debug("Current Loss: " + str(totalLoss))
  
  featuresToLeaveAtZero = set()
  
  maxDistance = 0
  featureWithMaxDistance = ""
  newParams = {}
  currentRLoss = 0
  for feature in featureDerivs:
    currentValue = params.get(feature, 0)
    deriv = (featureDerivs[feature] / numDatapoints) + 2*L2*currentValue
    currentRLoss += L1*abs(currentValue) + L2*(currentValue**2)
    deriv2 = (feature2ndDerivs.get(feature, 0) / numDatapoints) + 2*L2
    
    oldMaxDistance = maxDistance
    
    if (currentValue > 0):
      diff = (deriv + L1) / deriv2
      maxDistance = max(abs(diff), maxDistance)
      newValue = currentValue - diff
      if (sameSign(newValue, currentValue)): newParams[feature] = newValue
    if (currentValue < 0):
      diff = (deriv - L1) / deriv2
      maxDistance = max(abs(diff), maxDistance)
      newValue = currentValue - diff
      if (sameSign(newValue, currentValue)): newParams[feature] = newValue
    if (currentValue == 0):
      if (deriv > L1):
        diff = (deriv - L1) / deriv2
        maxDistance = max(abs(diff), maxDistance)
        newParams[feature] = currentValue - diff
      if (deriv < -L1):
        diff = (deriv + L1) / deriv2
        maxDistance = max(abs(diff), maxDistance)
        newParams[feature] = currentValue - diff
    if (maxDistance > oldMaxDistance): featureWithMaxDistance = feature
  
  if (allowLogging):
    logging.debug("Current R Loss: " + str(currentRLoss) + ", Total = " + str(totalLoss + currentRLoss))

  return (newParams, maxDistance, featureWithMaxDistance)

def sameSign(a, b): return (a > 0) == (b > 0)

def derivativeForFeature(label, count, expEnergy):
  term1 = -1 * labelToInt(label) * count
  term2 = count * expEnergy / (expEnergy + 1)
  return term1 + term2

def secondDerivativeForFeature(count, expEnergy):
  if (abs(math.log(expEnergy)) > 50): print expEnergy
  return (count ** 2) * expEnergy / ((1 + expEnergy) ** 2)

def energy(dataPoint, params):
  total = 0.0
  for feature in dataPoint:
    param = params.get(feature, 0)
    total += dataPoint[feature] * param
  return total

def labelToInt(label):
  if (label): return 1
  return 0

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
  expEnergy = math.exp(E)
  if (label): return math.log(expEnergy + 1) - E
  return math.log(expEnergy + 1)
  
def computeTrainingLossForDataset(dataPoints, labels, L1, params):
  datasetLoss = computeLossForDataset(dataPoints, labels, params)
  l1Loss = 0
  for feature in params: l1Loss += abs(params[feature])
  l1Loss *= L1
  return datasetLoss + l1Loss