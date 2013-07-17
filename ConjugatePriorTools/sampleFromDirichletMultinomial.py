#!/usr/bin/python
#
# First Argument: number of multinomials to draw
# Second Argument: number of points to draw from each mutlinomial
# The rest of the arguments: the alpha parameters for the dirichlet 
#
#

import sys
import csv
import math
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-N', '--numMulinomials', dest='N', type="int", default=100, help='Number of distinct multinomials to generate')
parser.add_option('-M', '--numSamplesPerRow', dest='M', type="int", default=100, help='The number of samples for each multinomial')
parser.add_option('-A', '--alpha', dest='A', default='1,1', help='Comma-separated dirichlet parameters')
parser.add_option('-O', '--outputType', dest='O', default='countMatrix', help='The type of output: countMatrix (default), or UMatrix')
(options, args) = parser.parse_args()


alphas = map(float, options.A.split(","))
K = len(alphas)
outputType = "countMatrix"
if (options.O == "UMatrix"): outputType = options.O

def drawMultinomial():
  multinomial = [0]*K
  runningTotal = 0
  for i in range(0, K):
	  if (alphas[i] != 0): runningTotal += random.gammavariate(alphas[i], 1)
	  multinomial[i] = runningTotal
  return multinomial, runningTotal
  
def sampleFromMultinomial(multinomial, total):
  buckets = [0]*K
  
  for j in range(0, options.M):
	  r = total * random.random()
	
	  for i in range(0, K):
		  if (r <= multinomial[i]):
			  buckets[i] += 1
			  break
  return buckets

if (outputType == "UMatrix"):
  # Init U-Matrix
  U = []
  for i in range(0, K): U.append([0] * options.M)

  for i in range(0, options.N):
	  multinomial, total = drawMultinomial()
	  buckets = sampleFromMultinomial(multinomial, total)
	  
	  for k in range(0, K):
	    for count in range(0, buckets[k]):
	      U[k][count] += 1
	      
  for i in range(0, K):
    print "\t".join(map(str, U[i]))
else:
  for i in range(0, options.N):
	  multinomial, total = drawMultinomial()
	  buckets = sampleFromMultinomial(multinomial, total)
	  print "\t".join(map(str, buckets))