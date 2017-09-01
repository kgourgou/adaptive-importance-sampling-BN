import json
import pickle

from misc import weight_average, parse_node_file
from adaptive_sampler import adaptive_sampler
from bayes_net import BNNoisyORLeaky
import csv

from joblib import Parallel, delayed

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


def single_run(data, i, sampler, nsample=1e6):
    """
    Carry out a single run with adaptive importance
    sampling. 
    """
    print("i={}".format(i))
    sampler.set_evidence(data)
    samples, weights, _ = sampler.ais_bn(
        num_of_samples=int(nsample), update_proposal_every=10000, skip_n=3)
    # store samples
    with open("samples_and_weights/sw_{}.csv".format(i), 'w') as f:
        writer = csv.writer(f, delimiter=",")
        for i, _ in enumerate(samples):
            writer.writerow([samples[i], weights[i]])


Parallel(n_jobs=6)(delayed(single_run)(dataset[i], i, sampler, nsample=1e6)
                   for i in dataset)


# for k, v in dataset.items():
#     ires = engine.analyse(v)
#     with open('marginals_{}.p'.format(k), 'wb') as f:
#         pickle.dump(ires, f)
