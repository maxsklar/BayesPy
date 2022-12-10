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
import logging
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-m', '--mixtureAlphas', dest='MA', default='1,1', help="Comma-separated dirichlet parameters for the singly generated mixture")
parser.add_option('-A', '--alpha', dest='A', default='1,1,1', help='Comma-separated dirichlet parameters for the individual components to be generated')
(options, args) = parser.parse_args()

mixtureAlphas = list(map(float, options.MA.split(",")))
alphas = list(map(float, options.A.split(",")))

K = len(alphas)

mixture = Sample.drawFromDirichlet(mixtureAlphas)

print(",".join(str(m) for m in mixture))

for i in range(len(mixture)):
  component = Sample.drawFromDirichlet(alphas)
  print(",".join(str(c) for c in component))