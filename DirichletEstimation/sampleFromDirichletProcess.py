#!/usr/bin/python
#
# Sample from a chinese restaurant process (infinite category dirichlet distribution)
# First Argument: number of rows to draw
# Second Argument: number of points to draw from each mutlinomial
# The rest of the arguments: the alpha parameter for the chinese restaurant process 
#
#

import sys
import csv
import math
import random
import samplingTools as Sample
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-N', '--numRows', dest='N', type="int", default=100, help='Number of distinct rows')
parser.add_option('-M', '--numSamplesPerRow', dest='M', type="int", default=100, help='The number of samples for each row')
parser.add_option('-A', '--alpha', dest='A', default='1', help='Chinese restaurant process parameter')
(options, args) = parser.parse_args()

alpha = float(options.A)

for i in range(0, options.N):
  result = Sample.chinese_restaurant_process(options.M, alpha)
  print("\t".join(list(map(str, result))))