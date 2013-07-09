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

numMultinomials = int(sys.argv[1])
numPoints = int(sys.argv[2])
K = len(sys.argv) - 3

alphas = [0]*K
for i in range(0, K):
	alphas[i] = float(sys.argv[3 + i])

for i in range(0, numMultinomials):
	# Draw multinomial
	multinomial = [0]*K
	runningTotal = 0
	for i in range(0, K):
		runningTotal += random.gammavariate(alphas[i], 1)
		multinomial[i] = runningTotal
	
	buckets = [0]*K
	
	for j in range(0, numPoints):
		r = runningTotal * random.random()
		
		for i in range(0, K):
			if (r <= multinomial[i]):
				buckets[i] += 1
				break
				
	print "\t".join(map(str, buckets))