#!/usr/bin/python
#
# A library for finding a dirichlet mixture mixture from count data
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import random
import logging
import csv
import samplingTools as ST

class MultinomialMixtureModel:
  # C: number of components
  # K: number of categories
  # multinomials: a list of C multinomial distributions, each with K parameters.
  # mixture: a C-dimensional multinomial describing the mixture probabilities.
  def __init__(self, C, K, multinomials, mixture):
      self.C = C
      self.K = K
      
      if (len(multinomials) != C):
        logging.error("dirichlets param must have C=" + str(C) + " components")
      
      for d in multinomials:
        if (len(d) != K): logging.error("multinomials distributions must have K=" + str(K) + " parameters")
        
      if (len(mixture) != C): logging.error("mixture probabilities must have C=" + str(C) + " components")
      
      self.multinomials = multinomials
      self.mixture = mixture
  
  def outputToFile(self, out):
    out.write("\t".join(map(str, self.mixture)))
    out.write("\n")
    for d in self.multinomials: 
      out.write("\t".join(map(str, d)))
      out.write("\n")
    out.close
    
  def logToDebug(self):
    logging.debug("\t".join(map(str, self.mixture)))
    for d in self.multinomials: 
      logging.debug("\t".join(map(str, d)))
  
  def sampleRow(self, amount):
    category = ST.drawCategory(self.mixture)
    multinomial = self.multinomials[category]
    retVal = [0]*self.K
    for i in range(0, amount):
      k = ST.drawCategory(multinomial)
      retVal[k] += 1
    return retVal
  
  def logstate(self):
    logging.debug(str(self.mixture) + "*" + str(self.multinomials))

def importFile(filename):
  infile = file(filename, 'r')
  reader = csv.reader(infile, delimiter='\t')
  mixture = map(float, reader.next())
  multinomials = []
  for row in reader: multinomials.append(map(float, row))
  K = 2
  if (len(multinomials) > 0): K = len(multinomials[0])
  return MultinomialMixtureModel(len(mixture), K, multinomials, mixture)

class MultinomialMixtureModelHyperparams:
  # C: number of components
  # K: number of categories
  # mixtureDirich: a dirichlet distribution representing the prior over the mixture parameters
  # componentDirich: a dirichlet distribution representing the prior over the components
  def __init__(self, C, K, mixtureDirich, componentDirich):
    self.C = C
    self.K = K
    
    if (len(mixtureDirich) != C): logging.error("mixture dirichlet must have C=" + str(C) + " components")
    if (len(componentDirich) != K): logging.error("component dirichlet must have K=" + str(K) + " components")
    self.mixtureDirich = mixtureDirich
    self.componentDirich = componentDirich

def logProbsToProbabilityDistribution(logProbs):
  highest = max(logProbs)
  logProbs = map(lambda x: x - highest, logProbs)
  unnormalized = map(lambda x: math.exp(x), logProbs)
  S = sum(unnormalized)
  return map(lambda x: float(x) / S, unnormalized)

# Input counts: A K-dimensional count vector for a given row
# we want to know which component of the mixture model best describes the counts
# output: a C-dimensional probability distribution over the possible components
def getComponentProbabilitiesForCounts(counts, model):
  # Find the energy for each multinomial
  logProbs = [0.] * model.C
  
  for c in range(0, model.C):
    logProbs[c] = getLogProbabilityForComponent(counts, model, c)
  
  return logProbsToProbabilityDistribution(logProbs)

def getLogProbabilityForComponent(counts, model, c):
  logProb = 0.
  for k in range(0, model.K):
    logProb += math.log(model.multinomials[c][k]) * counts[k]
  logProb += math.log(model.mixture[c])
  return logProb

# given a count vector C, and a multinomial M, the energy is
def findMultinomialFromCountsAndWeights(counts, weights, hyperparams):
  N = len(weights)
  if (len(counts) != N): logging.error("Weights vector must match counts in length")
  totalCounts = [0.0] * hyperparams.componentDirich
  for k in range(0, K):
    for n in range(0, N): totalCounts[k] += weights[n] * counts[n][k]
    totalCounts[k] += hyperparams[k]
  
  S = sum(totalCounts)
  for k in range(0, K): totalCounts[k] /= S
  return totalCounts

# Input data: N rows, K columns of count data
# params: MultinomialMixtureModel (current model)
# hyperparams: MultinomialMixtureModelHyperparams
# output: a new MultinomialMixtureModel based on the E-M algorithm
def updateMixtureModel(data, params, hyperParams):
  # Initialize parameter data structs
  C = params.C
  K = params.K
  mixtureCounts = [0.]*C
  multinomialCounts = []
  
  for c in range(0, C): 
    mixtureCounts[c] += hyperParams.mixtureDirich[c]
    multinomialCounts.append([0.]*K)
    for k in range(0, K):
      multinomialCounts[c][k] += hyperParams.componentDirich[k]

  # Loop through the data and update param data structs
  for row in data:
    cProbs = getComponentProbabilitiesForCounts(row, params)
    for c in range(0, C): 
      mixtureCounts[c] += cProbs[c]
      for k in range(0, K): multinomialCounts[c][k] += row[k] * cProbs[c]
  
  # Update the mixture probabilities
  sumMixtureCounts = sum(mixtureCounts)
  for c in range(0, C): mixtureCounts[c] /= sumMixtureCounts
  
  # Update the multinomial probabilities
  for c in range(0, C):
    multinomialSum = sum(multinomialCounts[c])
    for k in range(0, K): multinomialCounts[c][k] /= multinomialSum
  
  return MultinomialMixtureModel(C, K, multinomialCounts, mixtureCounts)

# Come up with an initial model
def initMixtureModel(data, hyperParams):
  # Initialize parameter data structs
  C = hyperParams.C
  K = hyperParams.K
  
  mixture = [1.0 / C] * C
  
  multinomials = []
  for c in range(0, C):
    multinomials.append([0.0]*K)
    denominator = sum(data[c]) + sum(hyperParams.componentDirich)
    for k in range(0, K): 
      numerator = float(data[c][k]) + hyperParams.componentDirich[k]
      multinomials[c][k] = numerator / denominator

  
  return MultinomialMixtureModel(C, K, multinomials, mixture)

def computeDirichletMixture(data, hyperParams, iterations):
  model = initMixtureModel(data, hyperParams)
  for i in range(0, iterations):
    newmodel = updateMixtureModel(data, model, hyperParams)
    mixDiff = diffModels(model, newmodel)
    logging.info("Iter: " + str(i) + ", mixDiff: " + str(mixDiff))
    model = newmodel
    logging.debug(", model = " + str(model.mixture) + " * " + str(model.multinomials))
    
  return model

def diffModels(oldParams, newParams):
  mixtureDiff = 0
  for c in range(0, oldParams.C):
    mixtureDiff += (oldParams.mixture[c] - newParams.mixture[c]) ** 2
  return mixtureDiff



# Find the datapoint that fits the model the worst
def worstFit(data, model):
  n = 0
  worstLogProb = 0
  worstN = -1
  worstC = -1

  for row in data:
    bestComponent = -1
    bestComponentProb = 0

    componentProbs = getComponentProbabilitiesForCounts(row, model)
    for c in range(0, model.C):
      if (componentProbs[c] > bestComponentProb):
        bestComponent = c
        bestComponentProb = componentProbs[c]

    logProb = getLogProbabilityForComponent(row, model, bestComponent)

    if (logProb < worstLogProb):
      worstLogProb = logProb
      worstN = n
      worstC = bestComponent

    n += 1

  return worstLogProb, worstN, worstC




