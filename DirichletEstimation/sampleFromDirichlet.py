#!/usr/bin/python

import sys
import csv
import math
import random
import samplingTools as Sample
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-N', '--numMulinomials', dest='N', type="int", default=100, help='Number of distinct multinomials to generate')
parser.add_option('-A', '--alpha', dest='A', default='1,1', help='Comma-separated dirichlet parameters')
parser.add_option('-O', '--outputType', dest='O', default='multnomials', help='The type of output: multnomials (default), or ss (sufficient statistic) which is the average of the sum of the logs for each category')
(options, args) = parser.parse_args()


alphas = list(map(float, options.A.split(",")))
K = len(alphas)

if (options.O == "ss"):
  for i in range(0, options.N):
    multinomial = Sample.drawFromDirichlet(alphas)
    print("\t".join(map(str, multinomial)))
else:
  for i in range(0, options.N):
    multinomial = Sample.drawFromDirichlet(alphas)
    print("\t".join(map(str, multinomial)))