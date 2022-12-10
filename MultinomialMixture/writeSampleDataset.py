#!/usr/bin/python

# Make sure to pipe in the model file:
# Ex.
# > cat sampleModel.txt | python writeSampleDataset.py

import multinomialMixtureEstimation as MME
import sys
import logging

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-C", '--numComponents', action="store", dest="C", default="1", help="the number of components in the mixture model")
parser.add_option("-N", '--numRows', action="store", dest="N", default="50", help="the number of rows to produce")
parser.add_option("-M", '--numSamplesPerRow', action="store", dest="M", default="10", help="The number of samples to produce per row, or the sum of each row")

(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

model = MME.importFileFromHandle(sys.stdin)

logging.debug("Imported Mixture Model Parameters...")
logging.debug(",".join(str(m) for m in model.mixture))

for multinomial in model.multinomials:
  logging.debug(multinomial)

for i in range(0, int(options.N)):
  sample = model.sampleRow(int(options.M))
  print("\t".join(map(str, sample)))