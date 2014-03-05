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
import dirichletMixtureEstimation as DMX
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-N', '--numMulinomials', dest='N', type="int", default=100, help='Number of distinct multinomials to generate')
parser.add_option('-M', '--numSamplesPerRow', dest='M', type="int", default=100, help='The number of samples for each multinomial')
parser.add_option('-F', '--filename', dest='filename', type="string", help='filename specifying the dirichlet mixture model')

(options, args) = parser.parse_args()

model = DMX.importDirichletMixtureFile(options.filename)

for n in range(0, options.N):
  row = model.sampleRow(options.M)
  print "\t".join(map(str, row))