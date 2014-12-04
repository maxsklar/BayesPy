#!/usr/bin/python

import multinomialMixtureEstimation as MME
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")

(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

model = MME.importFile("sampleModel.txt")

dataset = []
for row in sys.stdin:
  splitrow = row.split("\t")
  dataset.append(map(int, splitrow))

hyperP = MME.MultinomialMixtureModelHyperparams(5, 168, [1]*5, [1]*168)

finalModel = MME.computeDirichletMixture(dataset, hyperP, 500)

print "Final Model:"
print finalModel.mixture
print "****************************"
for multinomial in finalModel.multinomials:
  row = "\t".join(map(str, multinomial))
  print row