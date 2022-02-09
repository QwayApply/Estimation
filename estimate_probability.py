import scipy as sp
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


# input expected ranking and cut
expected_ranking = 19
expected_cut = 11

  

rv = sp.stats.norm(loc=expected_ranking, scale=5)
prob = rv.cdf(expected_cut)
print(prob)
