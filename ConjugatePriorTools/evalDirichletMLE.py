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
import dirichletPriorTools as DPT
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-a', '--alphas', dest='alphas', default='1,1', help='Comma-separated alpha values of the dirichlet to be tested')
(options, args) = parser.parse_args()


alphas = map(float, options.alphas.split(","))
K = len(alphas)
N = 1000

def getError(priors, MLE):
  total = 0
  for k in range(0, len(priors)):
    diff = math.log(priors[k]) - math.log(MLE[k])
    total += diff ** 2
    
  return math.sqrt(total)

for M in range(2, 20):
  errors = []

  for i in range(0, 1000):
    uMatrix = Sample.generateRandomDataset(M, N, alphas)
    vVector = [N]*M
    init = [1.0 / K]*K
    MLEPriors = DPT.findDirichletPriors(uMatrix, vVector, init, False)
    errors.append(getError(alphas, MLEPriors))

  errors.sort()

  print "\t".join(map(str, [M, errors[300], errors[500], errors[700], errors[900]]))
