#!/usr/bin/python
#
# A library for finding a dirichlet mixture mixture from count data
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

################################################
############ UNFINISHED
#################################################

# Copyright 2013 Max Sklar

import math
import random
import logging
import csv
import dirichletMultinomialEstimation as DME
import samplingTools as ST

class DirichletMixtureModel:
  # C: number of components
  # K: number of categories
  # dirichlets: a list of C dirichlet distributions, each with K parameters
  # mixture: a C-dimensional multinomial describing the mixture probabilities of the C dirichlets.
  def __init__(self, C, K, dirichlets, mixture):
      self.C = C
      self.K = K
      
      if (len(dirichlets) != C):
        logging.error("dirichlets param must have C=" + str(C) + " components")
      
      for d in dirichlets:
        if (len(d) != K): logging.error("dirichlet distributions must have K=" + str(K) + " parameters")
        
      if (len(mixture) != C): logging.error("mixture probabilities must have C=" + str(C) + " components")
      
      self.dirichlets = dirichlets
      self.mixture = mixture
  
  def outputToFile(self, filename):
    out = file(filename, 'w')
    out.write("\t".join(map(str, self.mixture)))
    out.write("\n")
    for d in self.dirichlets: 
      out.write("\t".join(map(str, d)))
      out.write("\n")
    out.close
  
  def sampleRow(self, amount):
    category = ST.drawCategory(self.mixture)
    dirichlet = self.dirichlets[category]
    multinomial = ST.drawFromDirichlet(dirichlet)
    retVal = [0]*self.K
    for i in range(0, amount):
      k = ST.drawCategory(multinomial)
      retVal[k] += 1
    return retVal

def importDirichletMixtureFile(filename):
  infile = file(filename, 'r')
  reader = csv.reader(infile, delimiter='\t')
  mixture = map(float, reader.next())
  dirichlets = []
  for row in reader: dirichlets.append(map(float, row))
  K = 2
  if (len(dirichlets) > 0): K = len(dirichlets[0])
  return DirichletMixtureModel(len(mixture), K, dirichlets, mixture)

class DirichletMixtureModelHyperparams:
  # C: number of components
  # K: number of categories
  # beta, W: the hyperdirichlet over the dirichlet distribution
  # mixtureDirich: a dirichlet distribution representing the prior over the mixture parameters
  def __init__(self, C, K, beta, W, mixtureDirich):
    self.C = C
    self.K = K
    
    if (len(beta) != K): logging.error("hyperdirichlet beta have K=" + str(K) + " parameters")
    self.beta = beta
    self.W = W
    
    if (len(mixtureDirich) != C): logging.error("mixture dirichlet must have C=" + str(C) + " components")
    self.mixtureDirich = mixtureDirich

# log(base) + log(base+1) + log(base+2) + ... + log(base+n-1)
def sumOfLogs(base, n):
  S = 0
  for i in range(0, n): S += math.log(base + i)
  return S
  
def logProbsToProbabilityDistribution(logProbs):
  highest = max(logProbs)
  logProbs = map(lambda x: x - highest, logProbs)
  unnormalized = map(lambda x: math.exp(x), logProbs)
  S = sum(unnormalized)
  return map(lambda x: float(x) / S, unnormalized)

# Input counts: A K-dimensional count vector for a given row
# we want to know which component of the mixture model best describes the counts
# output: a C-dimensional probability distribution over the possible components
def getComponentProbabilitiesForCounts(counts, dmm):
  logProbs = [0]*C
  
  for c in range(0, dmm.C):
    for k in range(0, dmm.K): logProbs[c] += sumOfLogs(dmm.dirichlets[c][k], x[k])
    logProbs[c] -= sumOfLogs(sum(dmm.dirichlets[c]), sum(x))
    logProbs[c] += math.log(dmm.mixture(c))
  
  logProbsToProbabilityDistribution(logProbs)

# Input data: N rows, K columns of count data
# params: DirichletMixtureModel (current model)
# hyperparams: DirichletMixtureModelHyperparams
# output: a new DirichletMixtureModel based on the E-M algorithm
def updateMixtureModel(data, params, hyperParams):
  # Initialize parameter data structs
  C = params.C
  K = params.K
  componentCompressedData = []
  for c in range(0, C): componentCompressedData.append(DME.CompressedRowData(K))
  mixtureCounts = [0.]*C

  # Loop through the data and update param data structs
  for row in data:
    cProbs = getComponentProbabilitiesForCounts(row, params)
    for c in range(0, C):
      cProb = cProbs[c]
      componentCompressedData[c].appendRow(row, cProb)
      mixtureCounts[c] += cProb

  # Compute information for new model
  dirichlets = []
  for c in range(0, C):
    D = DME.findDirichletPriors(componentCompressedData[c], [1.]*K, 50, hyperParams.beta, hyperParams.W)
    dirichlets.append(D)
  
  mixtureD = map(lambda c: mixtureCounts[c] + hyperParams.mixtureDirich[c], range(0, C))
  S = sum(mixtureD)
  mixture = map(lambda x: float(x) / S, mixtureD)

  return DirichletMixtureModel(C, K, dirichlets, mixture)

# Come up with an initial model
def initMixtureModel(data, hyperParams):
  # Initialize parameter data structs
  C = params.C
  K = params.K
  componentCompressedData = []
  for c in range(0, C): componentCompressedData.append(DME.CompressedRowData(K))
  mixtureCounts = [0.]*C

  # Loop through the data and update param data structs
  for n in range(0, len(data)):
    c = n % C
    componentCompressedData[c].appendRow(row, 1)
    mixtureCounts[c] += 1

  # Compute information for new model
  dirichlets = []
  for c in range(0, C):
    D = DME.findDirichletPriors(componentCompressedData[c], [1.]*K, 50, hyperParams.beta, hyperParams.W)
    dirichlets.append(D)

  mixtureD = map(lambda c: mixtureCounts[c] + hyperParams.mixtureDirich[c], range(0, C))
  S = sum(mixtureD)
  mixture = map(lambda x: float(x) / S, mixtureD)

  return DirichletMixtureModel(C, K, dirichlets, mixture)

def computeDirichletMixture(data, hyperParams, iterations):
  model = initMixtureModel(data, hyperParams)
  for i in range(0, iterations):
    logging.debug("Iter: " + str(i) + ", model = " + str(model))
    model = updateMixtureModel(data, model, hyperParams)

  return model