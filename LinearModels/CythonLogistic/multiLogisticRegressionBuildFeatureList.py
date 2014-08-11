#!/usr/bin/python
#
# EXPERIMENTAL

import sys
import csv
import math
import random
import multiLogisticRegression as MLR
import time
from optparse import OptionParser
import logging
from random import shuffle

startTime = time.time()
parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-M", '--maxFeatures', action="store", dest="M", default='0', help="Only consider the M most popular features.")
parser.add_option('-K', '--K', dest='K', default='2', help='Number of classes')

(options, args) = parser.parse_args()
K = int(options.K)

featureListComputer = MLR.FeatureListComputer(K)

for line in sys.stdin:
  (label, features) = MLR.lineToLabelAndFeatures(line)
  featureListComputer.appendRow(features)

featureListComputer.finalizeAndPrint(int(options.M))