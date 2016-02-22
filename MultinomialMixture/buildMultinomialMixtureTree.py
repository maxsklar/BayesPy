#!/usr/bin/python

import multinomialMixtureEstimation as MME
import multinomialMixtureTree as MMT
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-C", '--numComponents', action="store", dest="C", default="1", help="the number of components in the mixture model")
parser.add_option("-I", '--numIterations', action="store", dest="I", default="50", help="the number of iterations to run the mixture model")
parser.add_option("-O", '--outputModelFile', action="store", dest="outputModel", default="", help="store the model on this file")
parser.add_option("-M", '--maxKL', action="store", dest="M", default="0.5", help="the maximum acceptable KL divergence")
parser.add_option("-K", '--numCategories', action="store", dest="K", default="", help="the number of categories in each multinomial")

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

finalModel = MMT.buildMixtureTreeMaxKL(dataset, int(options.K), iterations, float(options.M), C)

logging.debug("Final Model:")
outputModel = sys.stdout
if (options.outputModel): outputModel = open(options.outputModel, 'w')
finalModel.outputToFile(outputModel)
