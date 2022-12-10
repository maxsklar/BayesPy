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

logging.debug("init dataset")
dataset = []
N = 0
for row in sys.stdin:
  if (N % 100000 == 0): logging.debug("processed " + str(N) + " rows.")
  splitrow = row.split("\t")
  dataset.append([int(x) for x in splitrow])
  N += 1
logging.debug("finished dataset")

hyperP = MME.MultinomialMixtureModelHyperparams(C, K, [1]*C, [1]*K)

finalModel = MME.computeDirichletMixture(dataset, hyperP, iterations)

logging.debug("Output final model")
finalModel.outputToFile(sys.stdout)

(worst, worstN, worstC) = MME.worstFit(dataset, finalModel)
logging.debug("worst: " + str(worst))
logging.debug("worst N: " + str(worstN))
logging.debug("worst: " + str(dataset[worstN]))
logging.debug("worst C: " + str(worstC))