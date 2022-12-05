#!/usr/bin/python
#
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

import math
import logging
import random
import scipy.special as mathExtra
import numpy as np
import scipy.sparse as sparse
import sys
import dirichletMultinomialEstimation as DME

def fs(x): return "%.4f" % x
def digamma(x): return float(mathExtra.psi(x))
def trigamma(x): return float(mathExtra.polygamma(1, x))

# shortcut to calculating digamma(x + N) - digamma(x)
def digamma2(x, N):
  S = 0
  for n in range(0, N): S += 1.0 / (x + n)
  return S

# shortcut to calculating trigamma(x + N) - trigamma(x)
def trigamma2(x, N):
  S = 0
  for n in range(0, N): S += (x + n) ** (-2)
  return -1 * S

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
  def __init__(self, K, holdoutPct):
    self.K = K
    self.holdoutPct = holdoutPct

    self.N = 0

    self.featureReverseLookup = {}
    self.featureForwardLookup = []
    self.numFeatures = 0
    self.labels = []
    self.datapoints = []
    self.labelCounts = np.zeros(K, dtype=np.int32)

    self.labels_holdout = []
    self.datapoints_holdout = []
    self.N_holdout = 0
  
  def addFeature(self, feature):
    self.featureReverseLookup[feature] = self.numFeatures
    self.featureForwardLookup.append(feature)
    self.numFeatures += 1
  
  def finalizeFeatures(self):
    self.featureMatrix = []
    for f in range(0, self.numFeatures): self.featureMatrix.append({})
  
  def appendRow(self, dataPoint, label):
    # First determine if this is a training data or holdout data
    if (random.random() < self.holdoutPct):
      # Holdout data
      self.labels_holdout.append(label)

      datapoint_feature_map = {}

      for feature in dataPoint:
        if (feature not in self.featureReverseLookup): continue
        featureIx = self.featureReverseLookup[feature]
        count = dataPoint[feature]
        datapoint_feature_map[featureIx] = count

      self.datapoints_holdout.append(datapoint_feature_map)
      self.N_holdout += 1
    else:
      self.labels.append(label)

      for k in range(0, self.K):
        self.labelCounts[k] += label[k]

      datapoint_feature_map = {}

      for feature in dataPoint:
        if (feature not in self.featureReverseLookup): continue
        featureIx = self.featureReverseLookup[feature]
        count = dataPoint[feature]
        self.featureMatrix[featureIx][self.N] = count
        datapoint_feature_map[featureIx] = count

      self.datapoints.append(datapoint_feature_map)
      self.N += 1
    
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
  K = dataPointAccumulator.K
  params = {}

  baseline = [0.0] * K
  training_loss = computeAverageLossForTrainingSet(dataPointAccumulator, baseline, params)
  holdout_loss = computeAverageLossForHoldoutSet(dataPointAccumulator, baseline, params)
  logging.debug("Init Loss without Baseline: " + fs(holdout_loss) + ", " + fs(training_loss))

  dataObj = DME.CompressedRowData(K)
  for label in dataPointAccumulator.labels: dataObj.appendRow(label, 1)
  dirichlet_priors = DME.findDirichletPriors(dataObj, [1.0] * K, 50)

  baseline = map(math.log, dirichlet_priors)

  logging.debug("Baseline: " + str(baseline))
  
  training_loss = computeAverageLossForTrainingSet(dataPointAccumulator, baseline, params)
  holdout_loss = computeAverageLossForHoldoutSet(dataPointAccumulator, baseline, params)
  logging.debug("Init Loss with Baseline: " + fs(holdout_loss) + ", " + fs(training_loss))

  feature_prob_of_use = [1.0] * dataPointAccumulator.numFeatures

  for i in range(0, maxIters):
    (maxDist, maxDistF, maxDistD, fSkiped) = batchStep(dataPointAccumulator, L1, L2, baseline, params, feature_prob_of_use, i, allowLogging)
    
    pct_training_loss = 0.1
    if (i == maxIters - 1): pct_training_loss = 0.1

    if (allowLogging):
      training_loss = computeAverageLossForTrainingSet(dataPointAccumulator, baseline, params, pct_training_loss)
      holdout_loss = computeAverageLossForHoldoutSet(dataPointAccumulator, baseline, params)
      maxDistIx = dataPointAccumulator.featureReverseLookup.get(maxDistF, 0.0)
      logging.debug("Iter " + str(i) + " Loss: " + fs(holdout_loss) + ", " + fs(training_loss) + ", Dist: " + str(maxDist) + " on " + str(maxDistF) + ":" + str(maxDistD) + " now " + ", ".join(map(fs, params.get(maxDistIx, [0.0] * K))) + ", Features: " + str(len(params)) + ", skipped " + str(fSkiped))
    
    if (maxDist < convergence):
      if (allowLogging): logging.debug("Converge criteria met.")
      return params

  if (allowLogging): logging.debug("Convergence did not occur in " + str(maxIters) + " iterations")
  return baseline, params

# dataPoints: a list of data points. Each data point is a map from a feature name (a string) to a number
# if a feature doesn't exist in the map, it is assumed to be zero
# labels: same length as dataPoints. Each label is a boolean
# L1: a positive float representing the L1 regression term
# params: the parameters for each feature (a map). The parameters are doubles, and assumed to be zero 
#  if they are not in the map
#
# Returns: (newParams, distance, avgLoss)
# distance: maximum distance between new and old params
def batchStep(dataPointAccumulator, L1, L2, baseline, params, feature_prob_of_use, iterationNumber, allowLogging = True):
  numDatapoints = dataPointAccumulator.N
  numFeatures = dataPointAccumulator.numFeatures
  #numFeatures = 100
  
  maxDistance = 0.0
  featureWithMaxDistance = 0
  dimWithMaxDistance = 0
  K = dataPointAccumulator.K
  
  featureIx = 0
  
  max_samples = 200
  learningRate = 0.2
  
  featureDeriv = np.zeros(K)
  
  # This really should be a 2D hession, but for now use the diagonal hessian
  diagHessian = np.zeros(K)
  
  features_skipped = 0
  for featureIx in range(0, numFeatures):
    if (random.random() > feature_prob_of_use[featureIx]):
      features_skipped += 1
      continue
    
    datapoints = dataPointAccumulator.featureMatrix[featureIx]
    num_datapoints_with_feature = len(datapoints)
    #logging.debug("feature " + str(featureIx) + " = " + str(dataPointAccumulator.featureForwardLookup[featureIx]) + " with " + str(num_datapoints_with_feature) + " datapoints")
    #if (featureIx % 1000 == 0):
    #  logging.debug("feature " + str(featureIx) + " = " + str(dataPointAccumulator.featureForwardLookup[featureIx]) + " with " + str(num_datapoints_with_feature) + " datapoints, Current Weights: " + str(params.get(featureIx, [0.0]*K)))

    if (num_datapoints_with_feature == 0): continue

    num_samples = min(num_datapoints_with_feature, max_samples)

    sampledDatapoints = random.sample(datapoints, num_samples)
      
    for k in range(0, K):
      featureDeriv[k] = 0.0
      diagHessian[k] = 0.0

    for dataPointIx in sampledDatapoints:
      if (dataPointIx >= numDatapoints): continue # we are limiting ourselves to these data
      
      count = dataPointAccumulator.featureMatrix[featureIx].get(dataPointIx, 0)
      if (count == 0):
        #logging.debug("Count is Zero")
        continue # no change to derivatives

      label = dataPointAccumulator.labels[dataPointIx]
      labelSum = sum(label)

      currentEnergies = energy(dataPointAccumulator.datapoints[dataPointIx], baseline, params, K)

      alpha = map(math.exp, currentEnergies)
      alphaSum = sum(alpha)

      #probs = map(lambda x: x / currentExpEnergiesSum, currentExpEnergies)
      #prob = probs[label]
      
      for k in range(0, K):
        # digamma2 is a shortcut to calculating digamma(x + N) - digamma(x)
        D = digamma2(alphaSum, labelSum) - digamma2(alpha[k], label[k])

        featureDeriv[k] += count * alpha[k] * D

        D2 = trigamma2(alphaSum, labelSum) - trigamma2(alpha[k], label[k])

        # Diagonal part of the hessian
        diagHessian[k] += (count ** 2) * alpha[k] * (D + alpha[k] * D2)
    

    # Correct for the number of samples
    for k in range(0, K):
      featureDeriv[k] *= float(num_datapoints_with_feature) / num_samples
      diagHessian[k] *= float(num_datapoints_with_feature) / num_samples

    # Add L2 regularization
    currentValues = params.get(featureIx, [0.0]*K)
    for k in range(0, K):
      featureDeriv[k] += L2*currentValues[k]
      diagHessian[k] += L2

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
      diffs[i] += learningRate * featureDeriv[i] / diagHessian[i]  

    if sum(map(abs, diffs)) == 0:
      feature_prob_of_use[featureIx] /= 2
    else:
      feature_prob_of_use[featureIx] = 1.0

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

    if (iterationNumber < 3):
      logging.debug("feature " + str(featureIx) + " = " + str(dataPointAccumulator.featureForwardLookup[featureIx]) + " with " + str(num_datapoints_with_feature) + " datapoints, Diffs = " + str(diffs) + " -- newValues: " + str(newValues))
    
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

  return (maxDistance, featureWithMaxDistance, dimWithMaxDistance, features_skipped)

def energy(dataPoint, baseline, params, K):
  total = [0.0] * K
  for k in range(0, K): total[k] = baseline[k]

  for feature in dataPoint:
    param = params.get(feature, [0.0] * K)
    for i in range(0, K): total[i] += dataPoint[feature] * param[i]
  return total

def computeAverageLossForTrainingSet(dataPointAccumulator, baseline, params, pct=1):
  K = dataPointAccumulator.K
  totalLoss = 0
  totalDataPoints = 0
  for i in range(0, dataPointAccumulator.N):
    if (random.random() > pct): continue
    dataPoint = dataPointAccumulator.datapoints[i]
    label = dataPointAccumulator.labels[i]
    totalLoss += computeLossForDatapoint(dataPoint, label, baseline, params, K)
    totalDataPoints += 1
  return totalLoss / totalDataPoints

def computeAverageLossForHoldoutSet(dataPointAccumulator, baseline, params):
  K = dataPointAccumulator.K
  totalLoss = 0
  totalDataPoints = 0
  for i in range(0, dataPointAccumulator.N_holdout):
    dataPoint = dataPointAccumulator.datapoints_holdout[i]
    label = dataPointAccumulator.labels_holdout[i]
    totalLoss += computeLossForDatapoint(dataPoint, label, baseline, params, K)
    totalDataPoints += 1

  if (totalDataPoints == 0): return float("nan")
  return totalLoss / totalDataPoints

# Note this calculation is specific to dirichletLogisticRegression
def computeLossForDatapoint(dataPoint, label, baseline, params, K):
  E = energy(dataPoint, baseline, params, K)
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