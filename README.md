BayesPy
=======

Bayesian Inference Tools in Python

Our goal is, given the discrete outcomes of events, estimate the distribution of categories.  Using gradient descent we can estimate the parameters of a dirchlet prior from past data that can be combined as a conjugate prior with the multinomial distribution to better estimate the likelihood of seeing an event of a given type in the future.

Conjugate Prior Tools:  The main file is ./findDirichletPrior - you pipe in your counts (given in test.csv as an example) and the maximum-likelihood dirichlet comes out.

Some things to try on your terminal:
cat test.csv | ./findDirichletPrior.py
-- This will find the priors for a test file

./flipCoins .7 1.2 | ./findDirichletPrior.py 
-- This will generate a data set on the fly using dirichlet parameters .7 1.2 (feel free to change those)
-- findDirichletPrior should come up with a good estimate of those numbers using only the coin flips

cat oneDoublesided.csv | ./findDirichletPrior.py
-- This is a sample of a case where findDirichletPrior won't give you a great result.  This is because every
-- coin in the input is fair except two coins: one is double sided heads, and the other tails.
-- Dirichlet distributions cannot handle this trimodal data very well, but it'll end up giving a compromise solution


#Using the priors
You can test the strength of your prior using the H parameter. Higher values for Beta will give lower probabilities.

python findDirichletPrior.py -H1,4,5 < /dev/null



gammaDistTools is not used.  These functions will be used for a future gamma distribution estimations.



Multinomial mixture model
=========================

DO NOT CONFUSE WITH LATENT DIRICHLET ALLOCATION! This is a much simpler model

Here is a command that will test a multinomial mixture model:
python writeSampleModel.py -A 0.3,0.3,0.3 -m 2,2 | python writeSampleDataset.py -N 10000 -M 500 | python3 inferMultinomialMixture.py -K 3 -C 2  

writeSampleModel: will output a model (formatted in a particular way) to stdout. This is a random mulitnomial mixture model, which is pulled from 2 dirichlet distributions (each component selected from the A param dirichlet, and the mixture itself from the m param dirichlet)

writeSampleDataset: will take the model as stdin, and then produce a dataset of a certain size given that model

inferMultinomialMixture (with K being the number of categories and C the number of mixed components): this will try to figure out the model based on the dataset.
