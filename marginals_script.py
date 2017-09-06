import json
from misc import weight_average, parse_node_file, char_fun
from adaptive_sampler import adaptive_sampler
from bayes_net import BNNoisyORLeaky

from joblib import Parallel, delayed

from os.path import exists

import scipy as sc


def card_to_evidence(card, pgm_network):
    items = card['symptoms'] + card['risk_factors']
    items = sorted(items, key=lambda x: x['weight'])

    evidence = {}
    # Gather evidence for symptom and risk factor nodes
    for item in items:
        # Check if node exists in the PGM
        concept_id = item['concept'].get('id', "NoneBeta")

        if concept_id in pgm_network:
            variable = pgm_network[concept_id]

            if variable['type'].lower() == "disease":
                # Skipping diseases for now.
                continue

            # Map severity to True/False
            if 'presence' in item:
                if item['presence'] == "UNSURE":
                    continue
                state = 'True' if item['presence'] == "PRESENT" else 'False'
                assert (item['presence'] == "PRESENT" or
                        item['presence'] == "NOT_PRESENT")
            elif 'severity' in item:
                if item['severity'] == "UNSURE":
                    continue
                if item['severity'] == 'NOT_PRESENT':
                    state = 'False'
                else:
                    state = 'True'

            evidence[concept_id] = state

    return evidence


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
sampler = adaptive_sampler(net, rep="Noisy-OR")

nodes = net.nodes


def single_run(evidence, i, sampler, nodes, nsample=1e6):
    """
    Carry out a single run with adaptive importance
    sampling.

    - evidence: dict, to be used as evidence.
    - i: int, index of the evidence
    - nodes:list, the names of all nodes
    - nsample:int, number of samples to use
    """

    print("i={}".format(i))

    sampler.set_evidence(evidence)
    samples, weights, _ = sampler.ais_bn(
        num_of_samples=int(nsample), update_proposal_every=30000, skip_n=3)

    print("Done with sampling.")

    w = sc.exp(weights)
    var_value = sc.var(w)
    print("variance of weights = {}".format(var_value))
    print("mean = {} ".format(sc.mean(w)))
    print("spread = {}".format(max(w)/sc.sum(w)))

    # initialize estimator class
    est = weight_average(samples, weights)

    def f(sample):
        return char_fun(sample, evidence)

    p_evi, _ = est.eval(f)

    if abs(p_evi) < 1e-10:
        raise ValueError("Bad value of p(evidence)={}".format(p_evi))

    # compute marginals here for anything that isn't evidence.
    probs = {}
    for key in nodes:
        if key not in evidence:
            # Compute probability P(key=True|Evidence)
            probs[key] = estimate_joint(key, evidence, est)
            probs[key] = probs[key] / p_evi

    with open("samples_and_weights/marginals_{}.json".format(i), "w") as f:
        json.dump(probs, f)

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
    single_run(dataset[i], i, sampler, nodes, nsample=1e4)
    break

# for k, v in dataset.items():
#     ires = engine.analyse(v)
#     with open('marginals_{}.p'.format(k), 'wb') as f:
#         pickle.dump(ires, f)
