"""

Simple network to try importance sampling on.

Note that all nodes are binary in this case.

"""

from scipy import mean, var
from samplers import likelihood_weight
from adaptive_sampler import adaptive_sampler

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
    A: [0.2, 0.8],
    B: {
        'A0': [0.3, 0.7],
        'A1': [0.6, 0.4]
    },
    C: {
        'A0': [0.8, 0.2],
        'A1': [0.9, 0.1]
    },
    E: {
        'B0C0': [0.1, 0.9],
        'B0C1': [1e-7, 1 - 1e-7],
        'B1C0': [0.4, 0.6],
        'B1C1': [0.1, 0.999]
    },
    D: {
        'A0': [0.4, 0.6],
        'A1': [0.1, 0.9]
    }
}

f = lambda x: 1
sampler = adaptive_sampler(graph=GRAPH, cpt=CPT, importance_weight_fun=f)
sampler.set_evidence({C:1})
samples, weights, _ = sampler.ais_bn(num_of_samples=100)

print("Estimate of P(B=1|C=1) = {}".format(
    sum([sample[B] * weights[i]
         for i, sample in enumerate(samples)]) / sum(weights)))
print("variance of weights = {}".format(var(weights)))

# This won't work anymore
# samples, weights = likelihood_weight(GRAPH, CPT, {C: 1}, num_of_samples=1000)
# norm_weights = weights / sum(weights)

# print('var = {}'.format(var(norm_weights)))
# c_values = [1 - i[E] for i in samples]
# print("prob = {}".format(sum(c_values * norm_weights)))
