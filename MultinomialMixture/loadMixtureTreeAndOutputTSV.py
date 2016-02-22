#!/usr/bin/python

import multinomialMixtureEstimation as MME
import multinomialMixtureTree as MMT
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-m", '--modelFile', action="store", dest="modelFile", default="", help="the stored model file")

(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

model = MMT.importFile(options.modelFile)

model.outputToTSV(sys.stdout)