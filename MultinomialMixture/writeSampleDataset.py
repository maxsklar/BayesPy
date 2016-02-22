#!/usr/bin/python

import multinomialMixtureEstimation as MME

model = MME.importFile("multinomialMixtureExample.txt")

for i in range(0, 500):
  sample = model.sampleRow(8)
  print("\t".join(map(str, sample)))