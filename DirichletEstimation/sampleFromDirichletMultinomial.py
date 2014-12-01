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
import samplingTools as Sample
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

if (outputType == "UMatrix"):
  data = Sample.generateRandomDataset(options.M, options.N, alphas)
  print "\t".join(map(str, data))
else:
  for i in range(0, options.N):
	  multinomial = Sample.drawFromDirichlet(alphas)
	  buckets = Sample.sampleFromMultinomial(multinomial, options.M)
	  print "\t".join(map(str, buckets))