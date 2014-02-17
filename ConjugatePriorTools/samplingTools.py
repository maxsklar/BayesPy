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
from collections import defaultdict


"""
http://en.wikipedia.org/wiki/Chinese_restaurant_process
"""
def chinese_restaurant_process(number_of_customers_param, alpha_param):
    open_table = 0
    alpha = float(alpha_param)
    number_of_customers = long(number_of_customers_param) - 1
    assert number_of_customers >= 0
    table_assignments = defaultdict(int)

    #assign first person to table 1
    table_assignments[open_table] += 1

    for customer_number in xrange(number_of_customers):
        new_table_prob = alpha / (customer_number + alpha)
        use_new_table = random.random() < new_table_prob

        if use_new_table:
            open_table += 1
            table_assignments[open_table] += 1
        else:
            distribution = []
            for table_index, number_of_people_sitting_at_table in table_assignments.iteritems():
                probability_of_table = number_of_people_sitting_at_table/float(customer_number+alpha)
                distribution.append(probability_of_table)
            
            idx = drawCategory(distribution)
            assert(idx in table_assignments)
            table_assignments[idx] += 1

    return table_assignments


# Returns a discrete distribution
def drawFromDirichlet(alphas):
  K = len(alphas)
  multinomial = [0]*K
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
  
  return K-1

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
		for k in range(0, K): ss[k] += math.log(distr[k])
	
	for k in range(0, K): ss[k] /= N
	return ss