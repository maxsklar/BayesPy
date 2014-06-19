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
import numpy as np
import logisticRegression as LRUtil
import scipy.sparse as sparse

class FeatureListComputer:
  def __init__(self, K):
    self.K = K
    self.featureCounts = {}
  
  def appendRow(self, dataPoint):
    for feature in dataPoint: 
      if (feature not in self.featureCounts): self.featureCounts[feature] = 0
      self.featureCounts[feature] += 1
  
  def finalizeAndPrint(self, maxFeatures):
    sortedFeatures = sorted(self.featureCounts, key=(lambda x: -self.featureCounts[x]))
    for f in sortedFeatures[:maxFeatures]: print(f)

class DataPointAccumulator:
  # K: num labels
  # N: num data points
  def __init__(self, K, N):
    self.K = K
    self.N = N
    
    self.dataPointIx = 0

    self.featureReverseLookup = {}
    self.featureForwardLookup = []
    self.numFeatures = 0
    self.labels = []
    self.labelCounts = [0] * K
  
  def addFeature(self, feature):
    self.featureReverseLookup[feature] = self.numFeatures
    self.featureForwardLookup.append(feature)
    self.numFeatures += 1
  
  def finalizeFeatures(self):
    self.featureMatrix = []
    for f in range(0, self.numFeatures): self.featureMatrix.append({})
  
  def appendRow(self, dataPoint, label):
    self.labels.append(label)
    self.labelCounts[label] += 1
    for feature in dataPoint:
      if (feature not in self.featureReverseLookup): continue
      featureIx = self.featureReverseLookup[feature]
      count = dataPoint[feature]
      self.featureMatrix[featureIx][self.dataPointIx] = count
    
    self.dataPointIx += 1
    
  def finalize(self):
    self.__CONST__ = map(lambda x: math.log(float(x) / self.N), self.labelCounts)
    logging.debug("CONST: " + str(self.__CONST__))

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# K: the number of different categories, or possible labels
# labels: same length as dataPoints. Each label is a number from 0 to K-1
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
# convergence: a small number to detect convergence
def batchCompute(dataPointAccumulator, L1, L2, convergence, maxIters, allowLogging = True):
  scores = np.zeros((dataPointAccumulator.N, dataPointAccumulator.K))
  for i in range(0, dataPointAccumulator.N):
    for k in range(0, dataPointAccumulator.K):
      scores[i][k] = dataPointAccumulator.__CONST__[k]
      
  logging.debug("Built the score matrix")
  params = {}
  for i in range(0, maxIters):
    (maxDist, maxDistF, maxDistD) = batchStep(dataPointAccumulator, L1, L2, params, scores, allowLogging)
    if (allowLogging):
      maxDistIx = dataPointAccumulator.featureReverseLookup[maxDistF]
      logging.debug("Iteration " + str(i) + ", Dist: " + str(maxDist) + " on " + maxDistF + ":" + str(maxDistD) + " now " + str(params.get(maxDistIx, [0.0, 0.0, 0.0])) + ", Features: " + str(len(params)))
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
  numDatapoints = dataPointAccumulator.N
  totalLoss = 0.0
  
  maxDistance = 0.0
  featureWithMaxDistance = ''
  dimWithMaxDistance = 0
  K = dataPointAccumulator.K
  
  featureIx = 0
  
  featureDeriv = np.zeros(3)
  # This really should be a 2D hession, but for now use the diagonal hessian
  diagHessian = np.zeros(3)
  
  for featureIx in range(0, dataPointAccumulator.numFeatures):
    for k in range(0, K):
      featureDeriv[k] = 0.0
      diagHessian[k] = 0.0
    
    for dataPointIx in dataPointAccumulator.featureMatrix[featureIx]:
      count = dataPointAccumulator.featureMatrix[featureIx][dataPointIx]
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
    currentValues = params.get(featureIx, [0.0]*K)
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
        featureWithMaxDistance = dataPointAccumulator.featureForwardLookup[featureIx]
        dimWithMaxDistance = i
          
    #Update Feature Weight
    if (all(v == 0.0 for v in newValues)): 
      if featureIx in params: del params[featureIx]
    else: params[featureIx] = newValues
    
    # Update Scores Vector
    for dataPointIx in dataPointAccumulator.featureMatrix[featureIx]:
      count = dataPointAccumulator.featureMatrix[featureIx][dataPointIx]
      for i in range(0, K):
        scores[dataPointIx][i] += count * (newValues[i] - currentValues[i])
    
    featureIx += 1
          
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
  for i in range(0, dataPointAccumulator.N):
    dataPoint = dataPointAccumulator.dataPoints[i]
    label = dataPointAccumulator.labels[i]
    totalLoss += computeLossForDatapoint(dataPoint, label, params, K)
    totalDataPoints += 1
  return totalLoss / totalDataPoints

def computeLossForDatapoint(dataPoint, label, params, K):
  E = energy(dataPoint, params, K)
  expEnergies = map(math.exp, E)
  return math.log(sum(expEnergies)) - E[label]

def lineToLabelAndFeatures(line):
  row = line.replace("\n", "").split("\t")
  label = int(row[0])
  features = {}
  
  for i in range(1, len(row)):
    featureStr = row[i]
    featureCutPointA = featureStr.rfind(":")
    featureCutPointB = featureCutPointA + 1
    feature = featureStr[:featureCutPointA]
    if (feature == "__CONST__"): continue
    count = int(float(featureStr[featureCutPointB:]))
    features[feature] = count
  
  return label, features