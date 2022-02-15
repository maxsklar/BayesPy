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
import numpy as np
import scipy.sparse as sparse

def digamma(x): return float(mathExtra.psi(x))
def trigamma(x): return float(mathExtra.polygamma(1, x))

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
  
  def finalizeAndPrintToFile(self, maxFeatures, fileObj):
    sortedFeatures = sorted(self.featureCounts, key=(lambda x: -self.featureCounts[x]))
    for f in sortedFeatures[:maxFeatures]: fileObj.write(f + "\n")
    fileObj.close()

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
    self.labelCounts = np.zeros(K, dtype=np.int32)
  
  def addFeature(self, feature):
    self.featureReverseLookup[feature] = self.numFeatures
    self.featureForwardLookup.append(feature)
    self.numFeatures += 1
  
  def finalizeFeatures(self):
    self.featureMatrix = []
    for f in range(0, self.numFeatures): self.featureMatrix.append({})
  
  def appendRow(self, dataPoint, label):
    self.labels.append(label)
    
    for k in range(0, self.K):
      self.labelCounts[k] += label[k]

    for feature in dataPoint:
      if (feature not in self.featureReverseLookup): continue
      featureIx = self.featureReverseLookup[feature]
      count = dataPoint[feature]
      self.featureMatrix[featureIx][self.dataPointIx] = count
    
    self.dataPointIx += 1
    
  def finalize(self):
    # Took out constant feature
    #self.__CONST__ = map(lambda x: math.log((0.1 + float(x)) / (self.N + 0.3)), self.labelCounts)
    #logging.debug("CONST: " + str(self.__CONST__))
    logging.debug("Note: Constant feature must be included in original dataset")

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
      scores[i][k] = 0.0 #dataPointAccumulator.__CONST__[k]
      
  logging.debug("Built the score matrix")
  params = {}
  for i in range(0, maxIters):
    (maxDist, maxDistF, maxDistD) = batchStep(dataPointAccumulator, L1, L2, params, scores, allowLogging)
    
    if (allowLogging):
      maxDistIx = dataPointAccumulator.featureReverseLookup.get(maxDistF, 0.0)
      logging.debug("Iteration " + str(i) + ", Dist: " + str(maxDist) + " on " + str(maxDistF) + ":" + str(maxDistD) + " now " + str(params.get(maxDistIx, [0.0, 0.0, 0.0])) + ", Features: " + str(len(params)))
    
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
  numFeatures = dataPointAccumulator.numFeatures
  totalLoss = 0.0
  
  maxDistance = 0.0
  featureWithMaxDistance = 0
  dimWithMaxDistance = 0
  K = dataPointAccumulator.K
  
  featureIx = 0
  
  featureDeriv = np.zeros(K)

  # This really should be a 2D hession, but for now use the diagonal hessian
  diagHessian = np.zeros(K)
  
  for featureIx in range(0, numFeatures):
    for k in range(0, K):
      featureDeriv[k] = 0.0
      diagHessian[k] = 0.0
    
    for dataPointIx in dataPointAccumulator.featureMatrix[featureIx]:
      if (dataPointIx >= numDatapoints): continue # we are limiting ourselves to these data
      
      count = dataPointAccumulator.featureMatrix[featureIx][dataPointIx]
      if (count == 0): continue # no change to derivatives

      label = dataPointAccumulator.labels[dataPointIx]
      labelSum = sum(label)
      currentEnergies = scores[dataPointIx]

      alpha = map(math.exp, currentEnergies)
      alphaSum = sum(alpha)


      #probs = map(lambda x: x / currentExpEnergiesSum, currentExpEnergies)
      #prob = probs[label]
      
      for k in range(0, K):
        # TODO: There's a trick to compute these without the special functions!
        D = 0.0
        D += digamma(alpha[k])
        D += digamma(alphaSum + labelSum)
        D -= digamma(alphaSum)
        D -= digamma(alpha[k] + label[k])

        featureDeriv[k] += count * alpha[k] * D

        D2 = 0.0
        D2 += trigamma(alpha[k])
        D2 += trigamma(alphaSum + labelSum)
        D2 -= trigamma(alphaSum)
        D2 -= trigamma(alpha[k] + label[k])
        
        # Diagonal part of the hessian
        diagHessian[k] += (count ** 2) * alpha[k] * (D + alpha[k] * D2)
        

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
      if (featureDeriv[i] == 0 and diagHessian[i] == 0): continue
      if (diagHessian[i] == 0):
        logging.debug("PROBLEM")
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
    for dataPointIx in range(0, numDatapoints):
      count = dataPointAccumulator.featureMatrix[featureIx].get(dataPointIx, 0)
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

# Note this calculation is specific to dirichletLogisticRegression
def computeLossForDatapoint(dataPoint, label, params, K):
  E = energy(dataPoint, params, K)
  alpha = map(math.exp, E)
  sum_term = sum(math.lgamma(alpha[k]) - math.lgamma(alpha[k] + label[k]) for k in range(K))
  return sum_term + math.lgamma(sum(alpha) + sum(label)) - math.lgamma(sum(alpha))

def lineToLabelAndFeatures(line, numCategories):
  row = line.replace("\n", "").split("\t")
  label = map(int, row[0:numCategories])
  features = {}
  
  for i in range(numCategories, len(row)):
    featureStr = row[i]
    featureCutPointA = featureStr.rfind(":")

    if (featureCutPointA == -1):
      features[featureStr] = 1
    else:
      featureCutPointB = featureCutPointA + 1
      feature = featureStr[:featureCutPointA]
      if (feature == "__CONST__"): continue
      count = int(float(featureStr[featureCutPointB:]))
      features[feature] = count
  
  return label, features