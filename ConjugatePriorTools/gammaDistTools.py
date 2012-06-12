#!/usr/bin/python

import sys
import math

# Probability of seeing k items after observing for time t given
# that the rate is taken from gamma distribution with params alpha and beta
def logNegativeBin(k, t, alpha, beta):
	p = beta / (beta + t)
	q = t / (beta + t)
	return partialLogSums(alpha, k) - partialLogSums(1, k) + alpha * math.log(p) + k * math.log(q)

# If you are doing gradient ascent on the priors
def getPriorGradient(k, t, alpha, beta):
	dalpha = partialHarmonic(alpha, k) + math.log(beta) - math.log(beta + t)
	dbeta = (alpha / beta) - ((alpha + k )/ (beta + t))
	return [dalpha, dbeta]

# think of this as log((x + k -1)! / (x+1)!)
# or: log(x) + log(x+1) + log(x+2) + ... + log(x+k-1)
def partialLogSums(x, k): return sum(map(lambda i: math.log(x + i), range(0, k)))

# This is the derivative of partial Log Sums
# 1/x + 1/(x+1) + 1/(x+2) + ... + 1/(x+k-1)	
def partialHarmonic(x, k): return sum(map(lambda i: 1.0 / (x + i), range(0, k)))
