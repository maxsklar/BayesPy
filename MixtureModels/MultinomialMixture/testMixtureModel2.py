#!/usr/bin/python

import multinomialMixtureEstimation as MME
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-C", '--numComponents', action="store", dest="C", default="1", help="the number of components in the mixture model")
parser.add_option("-I", '--numIterations', action="store", dest="I", default="50", help="the number of iterations to run the mixture model")

(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

C = int(options.C)
iterations = int(options.I)

print "init dataset"
dataset = []
for row in sys.stdin:
  splitrow = row.split("\t")
  dataset.append(map(int, splitrow))
print "finished dataset"

hyperP = MME.MultinomialMixtureModelHyperparams(C, 168, [1]*C, [1]*168)

finalModel = MME.computeDirichletMixture(dataset, hyperP, iterations)

logging.debug("Final Model:")

print "component\t",
for i in range(0, C): print str(i) + "\t",
print ""
print "prior\t" + "\t".join(map(str, finalModel.mixture))

for k in range(0, 168):
  print str(k) + "\t",
  for i in range(0, C):
    print str(finalModel.multinomials[i][k]) + "\t",
  print ""

(worseLogProb, worstN, worstC) = MME.worstFit(dataset, finalModel)
print "worstLogProb", worseLogProb
print "worst N", worstN
print "worst C", worstC