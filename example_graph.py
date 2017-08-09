"""

Simple network to try importance sampling on.

"""

from scipy import mean

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
    A: [0.5, 0.5],
    B: {
        tuple([0]): [0.3, 0.7],
        tuple([1]): [0.8, 0.2]
    },
    C: {
        tuple([0]): [0.5, 0.5],
        tuple([1]): [0.1, 0.9]
    },
    E: {
        tuple([0, 0]): [0.1, 0.9],
        tuple([0, 1]): [0.7, 0.3],
        tuple([1, 0]): [0.4, 0.6],
        tuple([1, 1]): [0, 1.0]
    },
    D: {
        tuple([0]): [0.4, 0.6],
        tuple([1]): [0.1, 0.9]
    }
}

from samplers import likelihood_weight
samples, weights = likelihood_weight(GRAPH, CPT, {D:1}, num_of_samples=1000)
print(mean(weights))
