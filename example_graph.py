"""

Simple network to try importance sampling on.

Note that all nodes are binary in this case.

"""

from pprint import pprint
from scipy import mean, var
from samplers import likelihood_weight
from adaptive_sampler import adaptive_sampler

from bayes_net import BayesNet, BNNoisyORLeaky

import matplotlib.pyplot as pl

A = "A"
B = "B"
C = "C"
D = "D"
E = "E"

NODES = {A, B, C, D, E}

# The format is child : parents
GRAPH = {A: {None}, B: {A}, C: {A}, E: {B, C}, D: {A}}

# Format is P(A = a|state of parent nodes) = prob
CPT = {
    # Prior for A
    A: [0.3, 0.7],
    B: {
        'A0': [0.8, 0.2],
        'A1': [1-1e-4, 1e-4]
    },
    C: {
        'A0': [0.5, 0.5],
        'A1': [1 - 1e-6, 1e-6]
    },
    E: {
        'B0C0': [0.1, 0.9],
        'B0C1': [1e-7, 1 - 1e-7],
        'B1C0': [1 - 1e-10, 1e-10],
        'B1C1': [1e-5, 1 - 1e-5]
    },
    D: {
        'A0': [0.4, 0.6],
        'A1': [0.1, 0.9]
    }
}

net = BayesNet(graph=GRAPH, cpt=CPT)
samples = net.msample()

print(mean([s[B] for s in samples]))

def spread(w):
    """
    The closer to zero, the better.
    """
    result = max(w) / sum(w)
    return result


def f(x):
    return 1


sampler = adaptive_sampler(net)
sampler.set_evidence({B: 1})
samples, weightsg, _ = sampler.ais_bn(
    num_of_samples=10000, update_proposal_every=100)

est = sum([sample[C] * weightsg[i]
           for i, sample in enumerate(samples)]) / sum(weightsg)
print("Estimate of P(C=1|B=1) = {}".format(est))
print("variance of weightsg = {}".format(var(weightsg)))

pl.clf()
pl.hist(weightsg, bins=30, label="with adaptive proposal")

# In order to reset evidence
sampler.set_evidence({B: 1})
samples, weights, _ = sampler.ais_bn(
    num_of_samples=10000, update_proposal_every=10000)

print("Estimate of P(C=1|B=1) = {}".format(
    sum([sample[C] * weights[i]
         for i, sample in enumerate(samples)]) / sum(weights)))
print("variance of weights = {}".format(var(weights)))
pl.hist(weights, bins=30, label="likelihood weighting")

pl.legend(fontsize=15, loc=0)
pl.title("Histogram of the weights", fontsize=15)

pl.show()
