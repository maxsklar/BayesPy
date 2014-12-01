#!/usr/bin/python
#
# Evaluating the performance of the dirichlet MLE calculation
# By: Max Sklar
# @maxsklar
# https://github.com/maxsklar

# Copyright 2013 Max Sklar

import sys
import csv
import math
import time
import samplingTools as Sample
import dirichletMultinomialEstimation as DME
import dirichletEstimation as DE
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-a', '--alphas', dest='alphas', default='1,1', help='Comma-separated alpha values of the dirichlet to be tested')
(options, args) = parser.parse_args()


alphasList = [[1, 2], [0.2, 0.05], [0.3, 0.4, 0.5]]

#alphas = map(float, options.alphas.split(","))

def getError(priors, MLE):
  total = 0
  for k in range(0, len(priors)):
    diff = math.log(priors[k]) - math.log(MLE[k])
    total += diff ** 2
    
  return math.sqrt(total)

for alphas in alphasList:
  for N in [10, 100, 1000, 10000, 100000, 1000000, 10000000]:
    print
    print "****************************************"
    print "alphas = ", alphas
    print
    K = len(alphas)
  
    for M in [5]:
      errors = []

      for i in range(0, 1000):
        uMatrix = Sample.generateRandomDataset(M, N, alphas)
        vVector = [N]*M
        init = [1.0 / K]*K
        MLEPriors = DME.findDirichletPriors(uMatrix, vVector, init, False)
        errors.append(getError(alphas, MLEPriors))

      errors.sort()

      print "\t".join(map(str, [N, M, errors[300], errors[500], errors[700], errors[900]]))

    # Test the M = infinity case
    errors = []

    for i in range(0, 1000):
      ss = Sample.generateRandomDirichletsSS(N, alphas)
      init = [1.0 / K]*K
      MLEPriors = DE.findDirichletPriors(ss, init, False)
      error = getError(alphas, MLEPriors)
      errors.append(error)

    errors.sort()

    print "\t".join(map(str, [N, "Inf", errors[300], errors[500], errors[700], errors[900]]))
