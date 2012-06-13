BayesPy
=======

Bayesian Inference Tools in Python

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