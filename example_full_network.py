"""

Simple network to try importance sampling on.

Note that all nodes are binary in this case.

"""

from pprint import pprint
from scipy import mean, var, exp
from samplers import likelihood_weight
from adaptive_sampler import adaptive_sampler
from bayes_net import BNNoisyORLeaky
from time import clock
import matplotlib.pyplot as pl
import seaborn as sns
from misc import weight_average, parse_node_file

filename = "data/approximate_network_all_mixed_any_age.json"
GRAPH, prior, lambdas = parse_node_file(filename)

net = BNNoisyORLeaky(GRAPH, lambdas, prior)
# samples = net.msample(num_of_samples=1000)

sampler = adaptive_sampler(net, rep="Noisy-OR")
sampler.set_evidence({
    "d0542984-7a52-4da5-b31f-c6847c2efe0c": 1,
    "450535cc-04c5-4fed-8c4b-92189438e58a": 0,
    "f838d3d9-78f7-4323-bea5-ee5d5dd2cd95": 1,
    "32de23a9-0925-499c-8475-e3601d89588c": 0,
    "d09fcdbb-9411-46d8-affd-fa5fce2fab00": 1
})

tic = clock()
samples, weights, _ = sampler.ais_bn(
    num_of_samples=10000, update_proposal_every=1000, skip_n=3)
toc = clock()

print("elapsed time = {:1.3f} min.".format((toc - tic)/60.0))

wa = exp(weights)
print("variance of weights = {}".format(var(wa)))
print("mean of weights = {}".format(mean(wa)))
print("diameter = {}".format(max(wa) - min(wa)))
print("spread = {}".format(max(wa)/sum(wa)))

# est = weight_average(samples, weights)
# f = lambda x: x["4758798f-fc15-4fab-9d76-4f66397c0b09"]
# print(est.eval(f))
# print(0.10486845959871564)

# pl.clf()
# pl.hist(wa, bins=20, label="with noisy-OR adaptive proposal")

# sampler.set_evidence({
#     "d0542984-7a52-4da5-b31f-c6847c2efe0c": 1,
#     "d09fcdbb-9411-46d8-affd-fa5fce2fab00": 1
# })

# tic = clock()
# samples, weights, _ = sampler.ais_bn(
#     num_of_samples=10000, update_proposal_every=10000)
# toc = clock()

# print("elapsed time = {}".format(toc - tic))

# w = exp(weights)
# print("variance of weights = {}".format(var(w)))
# print("mean of weights = {}".format(mean(w)))
# print("diameter = {}".format(max(w) - min(w)))

# pl.hist(w, bins=20, label="with likelihood weighting")

# pl.legend(fontsize=15, loc=0)
# pl.title("Histogram of the weights", fontsize=15)

# pl.show()
