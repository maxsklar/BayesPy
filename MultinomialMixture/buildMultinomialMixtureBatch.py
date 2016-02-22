#!/usr/bin/python

import multinomialMixtureEstimation as MME
import multinomialMixtureBatch as MMB
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-C", '--numComponents', action="store", dest="C", default="1", help="the number of components in the mixture model")
parser.add_option("-I", '--numIterations', action="store", dest="I", default="50", help="the number of iterations to run the mixture model")
parser.add_option("-O", '--outputModelFile', action="store", dest="outputModel", default="", help="store the model on this file")
parser.add_option("-B", '--batchSize', action="store", dest="B", default="", help="size of the minibatch")
parser.add_option("-R", '--learnRate', action="store", dest="R", default="", help="learn rate (between 0 and 1)")
parser.add_option("-K", '--numCategories', action="store", dest="K", default="", help="the number of categories")


(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

C = int(options.C)
K = int(options.K)
iterations = int(options.I)

print "init dataset"
dataset = []
N = 0
for row in sys.stdin:
  if (N % 100000 == 0): print "processed " + str(N) + " rows."
  splitrow = row.split("\t")
  dataset.append(map(int, splitrow))
  N += 1
print "finished dataset"

hyperP = MME.MultinomialMixtureModelHyperparams(C, K, [1]*C, [1]*K)

finalModel = MMB.computeDirichletMixture(dataset, hyperP, iterations, int(options.B), float(options.R))

logging.debug("Final Model:")
outputModel = sys.stdin
if (options.outputModel): outputModel = open(options.outputModel, 'w')
finalModel.outputToFile(outputModel)

#finalModel.outputToTSV(sys.stdout)

(worst, worstN, worstC) = MME.worstFit(dataset, finalModel)
print "worst", worst
print "worst N", worstN
print "worst C", worstC