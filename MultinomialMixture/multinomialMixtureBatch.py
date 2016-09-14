#!/usr/bin/python
#
# A library for finding a dirichlet mixture mixture from count data
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2015 Max Sklar

import math
import random
import logging
import csv
import samplingTools as ST
import multinomialMixtureEstimation as MME
import random

# Ensures a predictable result
random.seed(0)

def makeMinibatch(data, batchSize):
  N = len(data)
  minibatch = []
  for i in range(0, batchSize):
    minibatch.append(data[random.randint(0, N-1)])
  return minibatch

# Input data: N rows, K columns of count data
# params: MultinomialMixtureModel (current model)
# hyperparams: MultinomialMixtureModelHyperparams
# output: a new MultinomialMixtureModel based on the E-M algorithm
def updateMixtureModel(data, params, hyperParams, batchSize, learnRate):
  minibatch = makeMinibatch(data, batchSize)

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
  for row in minibatch:
    cProbs = MME.getComponentProbabilitiesForCounts(row, params)
    for c in range(0, C): 
      mixtureCounts[c] += cProbs[c]
      for k in range(0, K): multinomialCounts[c][k] += row[k] * cProbs[c]
  
  # Update the mixture probabilities
  sumMixtureCounts = sum(mixtureCounts)
  for c in range(0, C): 
    mixtureCounts[c] /= sumMixtureCounts
    mixtureCounts[c] = learnRate * mixtureCounts[c] + (1 - learnRate) * params.mixture[c]
  
  # Update the multinomial probabilities
  for c in range(0, C):
    multinomialSum = sum(multinomialCounts[c])
    for k in range(0, K): 
      multinomialCounts[c][k] /= multinomialSum
      multinomialCounts[c][k] = learnRate * multinomialCounts[c][k] + (1 - learnRate) * params.multinomials[c][k]
  
  return MME.MultinomialMixtureModel(C, K, multinomialCounts, mixtureCounts)

# Come up with an initial model
# TODO: don't base this on the data
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
  
  return MME.MultinomialMixtureModel(C, K, multinomials, mixture)

def computeDirichletMixture(data, hyperParams, iterations, batchSize, learnRate):
  model = initMixtureModel(data, hyperParams)
  for i in range(0, iterations):
    newmodel = updateMixtureModel(data, model, hyperParams, batchSize, learnRate)
    mixDiff = diffModels(model, newmodel)
    
    testMinibatch = makeMinibatch(data, batchSize)
    
    logging.info("Iter: " + str(i) + ", mixDiff: " + str(mixDiff))
    model = newmodel
    logging.debug(", model = " + str(model.mixture) + " * " + str(model.multinomials))
    
  return model

def diffModels(oldParams, newParams):
  mixtureDiff = 0
  for c in range(0, oldParams.C):
    mixtureDiff += (oldParams.mixture[c] - newParams.mixture[c]) ** 2
  return mixtureDiff

