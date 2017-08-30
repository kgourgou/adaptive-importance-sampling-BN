"""

Simple network to try importance sampling on.

Note that all nodes are binary in this case.

"""

from pprint import pprint
from scipy import mean, var
from samplers import likelihood_weight
from adaptive_sampler import adaptive_sampler

from bayes_net import BNNoisyORLeaky

import matplotlib.pyplot as pl

import seaborn as sns

from misc import weight_average
import scipy

A = "A"
B = "B"
C = "C"
D = "D"
E = "E"
leak = "leak_node"

NODES = {A, B, C, D, E}

# The format is child : parents
GRAPH = {A: {None}, B: {A}, C: {A}, E: {B, C}, D: {A}}

prior = {A: [0.9, 0.1]}

lambdas = {
    B: {
        A: 0.2,
        leak: 0.3
    },
    C: {
        A: 0.8,
        leak: 0.5
    },
    E: {
        B: 0.8,
        C: 0.4,
        leak: 0.2
    },
    D: {
        A: 0.7,
        leak: 0.9
    }
}

net = BNNoisyORLeaky(GRAPH, lambdas, prior)
samples = net.msample(num_of_samples=1000)


def spread(w):
    """
    The closer to zero, the better.
    """
    result = max(w) / sum(w)
    return result


def f(x):
    return 1


sampler = adaptive_sampler(net, rep="Noisy-OR")
sampler.set_evidence({C: 1})
samples, weights, _ = sampler.ais_bn(
    num_of_samples=10000, update_proposal_every=100)


weights = scipy.exp(weights)
print("Estimate of P(B=1|C=1) = {}".format(
    sum([sample[B] * weights[i]
         for i, sample in enumerate(samples)]) / sum(weights)))
print("variance of weights = {}".format(var(weights)))
print("mean of weights = {}".format(mean(weights)))
print("diameter = {}".format(max(weights)-min(weights)))

# pl.clf()
# pl.hist(weights, bins=20, label="with noisy-OR adaptive proposal")

# In order to reset evidence
sampler.set_evidence({C: 1})
samples, weights, _ = sampler.ais_bn(
    num_of_samples=10000, update_proposal_every=1000)

weights = scipy.exp(weights)

print("Estimate of P(B=1|C=1) = {}".format(
    sum([sample[B] * weights[i]
         for i, sample in enumerate(samples)]) / sum(weights)))
print("variance of weights = {}".format(var(weights)))
print("mean of weights = {}".format(mean(weights)))
# pl.hist(weights, bins=20, label="likelihood weighting", alpha=0.8, color="red")

print("diameter = {}".format(max(weights)-min(weights)))
# pl.legend(fontsize=15, loc=0)
# pl.title("Histogram of the weights", fontsize=15)

# pl.show()
