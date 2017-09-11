import json
from misc import weight_average, parse_node_file, char_fun, card_to_evidence, init_cpt_table
from adaptive_sampler import adaptive_sampler

from bayes_net import BNNoisyORLeaky, BNHybrid

from joblib import Parallel, delayed
import scipy as sc

with open('data/approximate_network_all_mixed_any_age.json', 'r') as f:
    network_JSON = json.load(f)

with open('data/allcards.json', 'r') as f:
    cards = json.load(f)

with open('data/approved_cards.json', 'r') as f:
    approved_card_ids = set(json.load(f))

approved_cards = list(filter(lambda c: c['id'] in approved_card_ids, cards))
assert len(approved_card_ids) == len(approved_cards)

dataset = {}
for c in approved_cards:
    evidence = card_to_evidence(c, network_JSON)
    dataset[c['id']] = evidence

filename = "data/approximate_network_all_mixed_any_age.json"
GRAPH, prior, lambdas = parse_node_file(filename)

net = BNNoisyORLeaky(GRAPH, lambdas, prior)


def single_run(evidence, i, nsample=1e6):
    """
    Carry out a single run with adaptive importance
    sampling.

    - evidence: dict, to be used as evidence.
    - i: int, index of the evidence
    - nodes:list, the names of all nodes
    - nsample:int, number of samples to use
    """

    print("i={}".format(i))

    # Initialize proposal distribution

    nodes = {key: "noisy" for key in net.nodes}

    # parents of evidence nodes
    cpts = {}
    for e in evidence:
        if net.is_root_node(e):
            continue

        for key in net.graph[e]:
            if net.is_root_node(key):
                continue
            else:
                if len(net.graph[key]) < 6:
                    nodes[key] = "cpt"
                    cpts[key] = init_cpt_table(key, net.graph[key],
                                               net.joint_prob)

    # initialize priors
    for node in prior:
        cpts[node] = prior[node]

    prop = BNHybrid(GRAPH, nodes, net.lambdas, prior, cpts)

    for node in net.prior_dict:
        prop.noisy_net.prior_dict[node] = net.prior_dict[node].copy()

    sampler = adaptive_sampler(net, rep="Hybrid", proposal=prop)

    nodes = net.nodes
    sampler.set_evidence(evidence)
    samples, weights, _ = sampler.ais_bn(
        num_of_samples=int(nsample), update_proposal_every=30000, skip_n=3)

    print("Done with sampling.")

    w = sc.exp(weights)
    var_value = sc.var(w)
    print("variance of weights = {}".format(var_value))
    print("mean = {} ".format(sc.mean(w)))
    print("spread = {}".format(max(w) / sc.sum(w)))

    # initialize estimator class
    est = weight_average(samples, weights)

    def f(sample):
        return char_fun(sample, evidence)

    p_evi, _ = est.eval(f)

    if abs(p_evi) < 1e-10:
        raise ValueError("Bad value of p(evidence)={}".format(p_evi))

    # compute marginals here for anything that isn't evidence.
    # probs = {}
    # for key in nodes:
    #     if key not in evidence:
    #         # Compute probability P(key=True|Evidence)
    #         probs[key] = estimate_joint(key, evidence, est)
    #         probs[key] = probs[key] / p_evi

    # with open("samples_and_weights/marginals_{}.json".format(i), "w") as f:
    #     json.dump(probs, f)

    print("i={} is done.".format(i))


def estimate_joint(node, evidence, est):
    """
    Estimate P(node=True|evidence) from a set
    of weights and samples.

    Arguments:
    - node: str, name of node
    - evidence: dict, contains the evidence nodes with
    truth values.
    - est: weight_estimator object, used to estimate
    weighted averages.
    """
    temp = evidence.copy()
    temp[node] = True

    def f(sample):
        return char_fun(sample, temp)

    p, _ = est.eval(f)
    return p


# Parallel(n_jobs=5)(delayed(single_run)(
#     dataset[i], i, sampler, nodes, nsample=1e5) for i in dataset)

for i in dataset:
    print(i)
    single_run(dataset[i], i, nsample=1e5)
    break

# for k, v in dataset.items():
#     ires = engine.analyse(v)
#     with open('marginals_{}.p'.format(k), 'wb') as f:
#         pickle.dump(ires, f)
