#!/usr/bin/python
#
# Sampling library from a variety of different distributions
#
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import math
import random

# Returns a discrete distribution, unnormalized so it may not sum to 1
def drawFromDirichlet(alphas):
  K = len(alphas)
  multinomial = [0]*K
  runningTotal = 0
  for i in range(0, K): multinomial[i] = random.gammavariate(alphas[i], 1)
  return multinomial

# Draws a category from an unnormalized distribution
def drawCategory(distribution):
  K = len(distribution)
  total = sum(distribution)
  
  r = sum(distribution) * random.random()
  runningTotal = 0
  for k in range(0, K):
    runningTotal += distribution[k]
    if (r < runningTotal): return k
  
  return K

def sampleFromMultinomial(multinomial, M):
  buckets = [0]*len(multinomial)

  for m in range(0, M):
    category = drawCategory(multinomial)
    buckets[category] += 1
    
  return buckets

# Generates the U-Matrix for N rows, and M data points per row.
# The v-vector is just going to be [N]*K
def generateRandomDataset(M, N, alphas):
  K = len(alphas)
  U = []
  for i in range(0, K): U.append([0] * M)

  for i in range(0, N):
	  multinomial = drawFromDirichlet(alphas)
	  buckets = sampleFromMultinomial(multinomial, M)
	  
	  for k in range(0, K):
	    for count in range(0, buckets[k]):
	      U[k][count] += 1
  return U