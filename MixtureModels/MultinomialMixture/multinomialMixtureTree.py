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

# In this model, we have a tree of multinomial mixtures (which is much faster to compute than a flat mixture)
class MultinomialMixtureTree:
  # mixture: MultinomialMixtureModel, is the top level mixture model
  def __init__(self, multinomialMixture):
    self.K = multinomialMixture.K

    # This is the top level multinomial mixture.
    self.multinomialMixture = multinomialMixture

    # Each component can be further branched into a tree, but it starts out with an empty tree (represented by None)
    # if it's not branched.
    self.mixtureNodes = [None] * multinomialMixture.C

  def outputToFileDontClose(self, out):
    self.outputToFileWithTreeMarkings(out, [])
    
  def outputToFileWithTreeMarkings(self, out, treeLocations):
    # output the location of this mixture in the tree
    out.write("\t".join(map(str, treeLocations)))
    out.write("\n")

    #output the mixture
    self.multinomialMixture.outputToFileDontClose(out)

    #output the submixtures
    for c in range(0, self.multinomialMixture.C):
      if (self.mixtureNodes[c]):
        self.mixtureNodes[c].outputToFileWithTreeMarkings(treeLocations + [c])
      else:
        out.write("") # Empty mixture

  def outputToFile(self, out):
    self.outputToFileDontClose(out)
    out.close

  def sampleRow(self, amount):
    c = ST.drawCategory(self.multinomialMixture.mixture)
    if (self.mixtureNodes[c]): return self.mixtureNodes[c].sampleRow(amount)

    multinomial = self.multinomialMixture.multinomials[category]
    retVal = [0]*self.K
    for i in range(0, amount):
      k = ST.drawCategory(multinomial)
      retVal[k] += 1
    return retVal

def importFile(filename):
  infile = file(filename, 'r')
  reader = csv.reader(infile, delimiter='\t')

  treeModel = readSingleTreeAndChildrenFromInfile(reader)
  return treeModel

def readSingleTreeAndChildrenFromInfile(reader):
  model = readSingleTreeFromInfile(reader)
  if (model == None): return None
  treeModel = new MultinomialMixtureTree(model)

  for c in range(0, treeModel.multinomialMixture.C):
    child = readSingleTreeAndChildrenFromInfile(reader)
    treeModel.mixtureNodes[c] = child

  return treeModel

def readSingleTreeFromInfile(reader):
  mixture = map(float, reader.next())

  if (len(mixture) == 0): return None # We've reached a terminal node

  multinomials = []
  for i in range(0, len(mixture)):
    row = reader.next()
    multinomials.append(map(float, row))

  K = 2
  if (len(multinomials) > 0): K = len(multinomials[0])

  return MultinomialMixtureModel(len(mixture), K, multinomials, mixture)

# The standard way to build a mixture tree 
# - set a fixed number of branches per node
# - set a fixed height
def buildSimpleMixtureTree(data, K, iterations, height, branchesPerNode = 2):
  if (height == 0): return None

  # hyperparameters are fixed here:
  hyperP = MME.MultinomialMixtureModelHyperparams(branchesPerNode, K, [1.0 / C]*C, [1.0 / K]*K)

  mixtureModel = MME.computeDirichletMixture(dataset, hyperP, iterations)
  
  smallerDatasets = []
  for c in range(0, hyperP.C): smallerDatasets.append([])

  for counts in data:
    c = MME.assignComponentToCounts(counts, mixtureModel)
    smallerDatasets[c].append(counts)

  treeModel = new MultnomialMixtureTree(mixtureModel)

  for c in range(0, hyperP.C):
    smallerDataset = smallerDatasets[c]
    child = buildSimpleMixtureTree(smallerDataset, K, iterations, height - 1, branchesPerNode)
    treeModel.mixutreNodes[c] = child

  return treeModel


