#!/usr/bin/python

import multinomialMixtureEstimation as MME
import logging
logging.basicConfig(level=logging.DEBUG)

model = MME.importFile("sampleModel.txt")

dataset = []
for i in range(0, 500): dataset.append(model.sampleRow(8))

hyperP = MME.MultinomialMixtureModelHyperparams(2, 3, [1, 1], [1, 1, 1])

finalModel = MME.computeDirichletMixture(dataset, hyperP, 10)

print "Final Model:"
print finalModel.mixture
print finalModel.multinomials