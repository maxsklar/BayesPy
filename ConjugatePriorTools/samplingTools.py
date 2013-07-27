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

# Returns a discrete distribution
def drawFromDirichlet(alphas):
  K = len(alphas)
  multinomial = [0]*K
  runningTotal = 0
  for i in range(0, K): multinomial[i] = random.gammavariate(alphas[i], 1)
  S = sum(multinomial)
  return map(lambda i: i/S, multinomial)

# Draws a category from an unnormalized distribution
def drawCategory(distribution):
  K = len(distribution)
  
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

def generateRandomDirichlets(N, alphas):
	D = []
	for n in range(0, N):
		D.append(drawFromDirichlet(alphas))
	return D
	
def generateRandomDirichletsSS(N, alphas):
	K = len(alphas)
	ss = [0]*K
	for n in range(0, N):
		distr = drawFromDirichlet(alphas)
		for k in range(0, K): ss[k] += distr[k]
	
	for k in range(0, K): ss[k] /= N
	return ss