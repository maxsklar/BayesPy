#!/usr/bin/python

import multinomialMixtureEstimation as MME
import logging
import sys
from optparse import OptionParser
import string

parser = OptionParser()
parser.add_option("-L", '--loglevel', action="store", dest="loglevel", default='DEBUG', help="don't print status messages to stdout")
parser.add_option("-C", '--numComponents', action="store", dest="C", default="1", help="the number of components in the mixture model")
parser.add_option("-m", '--modelFile', action="store", dest="modelFile", default="", help="the stored model file")

(options, args) = parser.parse_args()

#Set the log level
log_level = options.loglevel
numeric_level = getattr(logging, log_level, None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

C = int(options.C)

model = MME.importFile(options.modelFile)

dataset = []
for row in sys.stdin:
  splitrow = row.split("\t")
  dataset.append(map(int, splitrow))

#for n in range(0, len(dataset)):
#  counts = dataset[n]
#  print str(n) + "\t" + str(MME.assignComponentToCounts(counts, model))

# print file for google docs
#print "component\t",
#for i in range(0, C): print str(i) + "\t",
#print ""
#print "prior\t" + "\t".join(map(str, finalModel.mixture))
#
#for k in range(0, 168):
#  print str(k) + "\t",
#  for i in range(0, C):
#    print str(finalModel.multinomials[i][k]) + "\t",
#  print ""

rowInfoList = []
N = 0
for row in dataset:
  c = MME.assignComponentToCounts(row, model)
  klDiv = MME.klTest(row, model.multinomials[c])
  rowInfoList.append([N, c, klDiv, sum(row)])
  N += 1

print "row\tmodel\tklDivergence\tNumber of Data Points"
for row in rowInfoList:
  print "\t".join(map(str, row))