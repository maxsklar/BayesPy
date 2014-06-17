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
  # Some pre-processing
  scores = [0.0] * len(dataPoints)
  featuresToDataPointIxs = createFeaturesToDatapointIxMap(dataPoints)
  sortedFeatures = sorted(featuresToDataPointIxs, key=(lambda x: -len(featuresToDataPointIxs.get(x))))
  
  i = 0
  for feature in sortedFeatures:
    numAppearances = len(featuresToDataPointIxs[feature])
    i += 1
  
  params = {}
  for i in range(0, maxIters):
    (maxDist, maxDistF, dataLoss, paramLoss) = batchStep(dataPoints, labels, L1, L2, params, scores, featuresToDataPointIxs, sortedFeatures, allowLogging)
    if (allowLogging): logging.debug("Iteration " + str(i) + ", Loss: " + str(dataLoss) + " + " + str(paramLoss) + ", Dist: " + str(maxDist) + " on " + maxDistF + " now " + str(params.get(maxDistF, 0)) + ", Features: " + str(len(params)))
    if (maxDist < convergence):
      if (allowLogging): logging.debug("Converge criteria met.")
      return params

  if (allowLogging): logging.debug("Convergence did not occur in " + str(maxIters) + " iterations")
  return params

def createFeaturesToDatapointIxMap(dataPoints):
  featuresToDataPointIxs = {}
  for i in range(0, len(dataPoints)):
    dataPoint = dataPoints[i]
    for feature in dataPoint:
      if feature not in featuresToDataPointIxs: featuresToDataPointIxs[feature] = []
      current = featuresToDataPointIxs.get(feature, [])
      current.append(i)
  return featuresToDataPointIxs

def createFeaturesToNumDatapointsMap(dataPoints):
  featuresToNumDatapointsMap = {}
  for i in range(0, len(dataPoints)):
    dataPoint = dataPoints[i]
    for feature in dataPoint:
      if feature not in featuresToNumDatapointsMap: featuresToNumDatapointsMap[feature] = 0
      featuresToDataPointIxs[feature] += 1
  return featuresToNumDatapointsMap

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# labels: same length as dataPoints. Each label is a boolean
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
#
# Returns: (newParams, distance, avgLoss)
# distance: maximum distance between new and old params
def batchStep(dataPoints, labels, L1, L2, params, scores, featuresToDataPointIxs, sortedFeatures, allowLogging = True):
  numDatapoints = len(dataPoints)
  paramLoss = 0.0
  
  maxDistance = 0.0
  featureWithMaxDistance = ''
  
  for feature in sortedFeatures:
    featureDeriv = 0.0
    featureDeriv2 = 0.0
    
    for dataPointIx in featuresToDataPointIxs.get(feature, []):
      count = dataPoints[dataPointIx][feature]
      label = labels[dataPointIx]
      currentEnergy = scores[dataPointIx]
      expEnergy = math.exp(currentEnergy)
      featureDeriv += derivativeForFeature(label, count, currentEnergy)
      featureDeriv2 += secondDerivativeForFeature(count, currentEnergy)
    
    featureDeriv /= numDatapoints
    featureDeriv2 /= numDatapoints
        
    # Add L2 regularization
    currentValue = params.get(feature, 0)
    featureDeriv += L2*currentValue
    featureDeriv2 += L2
    
    # Add L1 regularization (tricky!)
    if (currentValue > 0 or (currentValue == 0 and featureDeriv < -L1)):
      featureDeriv += L1
    else:
      if (currentValue < 0 or (currentValue == 0 and featureDeriv > L1)):
        featureDeriv -= L1
      else: # Snap-to-zero
        featureDeriv = 0
    
    
    # Calculate new value
    diff = float(featureDeriv) / featureDeriv2
    newValue = currentValue - diff
    if ((currentValue != 0) and (not sameSign(newValue, currentValue))): 
      newValue = 0
    diff = newValue - currentValue
    
    #Update Feature Weight
    if (newValue != 0): params[feature] = newValue
    else:
      if (feature in params): del params[feature]
    
    # update maxDistance
    if (abs(diff) > maxDistance):
      maxDistance = abs(diff)
      featureWithMaxDistance = feature
      
    # Update Scores Vector
    for dataPointIx in featuresToDataPointIxs[feature]:
      dataPoint = dataPoints[dataPointIx]
      count = dataPoint[feature]
      scores[dataPointIx] += count * diff
    
    paramLoss += L1*abs(newValue) + 0.5 * L2 * (newValue ** 2)
  
  totalDataLoss = 0
  for i in range(0, numDatapoints):
    totalDataLoss += math.log(math.exp(scores[i]) + 1)
    if (labels[i]): totalDataLoss -= scores[i]
  avgDataLoss = totalDataLoss / numDatapoints
  
  return (maxDistance, featureWithMaxDistance, avgDataLoss, paramLoss)

def sameSign(a, b): return (a > 0) == (b > 0)

def lossForFeature(label, count, expEnergy):
  return -1 * label * math.log(expEnergy) + math.log(expEnergy + 1)

def derivativeForFeature(label, count, energy):
  term1 = -1 * labelToInt(label) * count
  
  if (energy >= 0.0):
    expEnergy = math.exp(energy)
    term2 = count * expEnergy / (expEnergy + 1)
    return term1 + term2
  else:
    expNEnergy = math.exp(-energy)
    term2 = count / (expNEnergy + 1)
    return term1 + term2

def secondDerivativeForFeature(count, energy):
  if (energy >= 0.0):
    expEnergy = math.exp(energy)
    return (count ** 2) * expEnergy / ((1 + expEnergy) ** 2)
  else:
    expNEnergy = math.exp(-energy)
    return (count ** 2) * expNEnergy / ((1 + expNEnergy) ** 2)

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
    loss = computeLossForDatapoint(dataPoint, label, params)
    totalLoss += loss
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