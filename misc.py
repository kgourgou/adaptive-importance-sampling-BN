"""
misc. functions
"""

from scipy import mean, var
import json

import scipy as sc
from itertools import product


def dict_to_string(some_dict):
    """
    Takes the {"A":some_value, "B":some_other_value}

    to

    "A:some_value;B:some_other_value;".

    This can then be used as a key for a CPT table.
    """
    result = ""
    sorted_keys = sorted(some_dict, key=lambda key: key)
    new_dict = {key: some_dict[key] for key in sorted_keys}

    for key in new_dict:
        result += str(key) + ":" + str(new_dict[key]) + ";"

    return result


def string_to_dict(s):
    """
    inverse of dict_to_string
    """
    split_string = s.split(";")
    # remove last element
    split_string = split_string[:-1]

    new_dict = {}
    for key in split_string:
        temp = key.split(":")
        new_dict[temp[0]] = bool(temp[1])

    return new_dict


def init_cpt_table(child, parents, joint_prob=None):
    """
    Initializes a cpt table for a child with
    two states and n parents. Requires dict_to_string
    function.

    >>> init_cpt_table("A", ["B","C"])
    >>> {"B0C0":[0.5,0.5],...}
    """
    np = len(parents)
    states = product([0, 1], repeat=np)

    cpt = {}

    if joint_prob is None:
        def joint_prob(x):
            return[0.5, 0.5]

    for state in states:
        temp = dict(zip(parents, state))
        key = dict_to_string(temp)
        temp[child] = True
        p = joint_prob(temp)

        if p < 0.1:
            p = 0.2
        else:
            p = 0.8

        cpt[key] = [1-p, p]

    return cpt


def char_fun(A, b):
    """

    Returns True if dictionary b is a subset
    of dictionary A and False otherwise.
    """

    result = b.items() <= A.items()

    return result


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


class weight_average(object):
    """
    Class to calculate weighted averages.
    Defaults to normal averages if no
    weights are provided.
    """

    def __init__(self, values, weights=None):
        """
        values: iterable of 1xN dimension, the values on
        which to evaluate the quantity of interest.
        weights: iterable of 1xN dimension or None, the corresponding
        weight for each value.
        """
        self.values = values

        if weights is not None:
            self.weights = sc.exp(weights)
            self.weights = self.weights / sum(self.weights)

    def eval(self, f=lambda x: 1):
        """"
        evaluate function on the values
        and take weighted average.
        """

        if self.weights is None:
            vec = [f(v) for v in self.values]
            result = mean(vec)
            variance = var(vec)
        else:
            v = self.values
            num_non_zeros = sum([int(f(v) > 0) for v in self.values])
            if num_non_zeros < 5:
                print("Warning: small number of samples: {}".format(
                    num_non_zeros))
                result = 0
                variance = -1
                return result, variance

            vec = [f(v[i]) * w for i, w in enumerate(self.weights)]
            result = sum(vec)
            variance = var(vec)

        return result, variance


def parse_node_file(filename):
    """
    Parses a node file and creates the following variables:

    graph = {child:{None}, child:{Parent1, Parent2}, ...}
    prior = {node: [prior values]}
    lambdas = {parent:{child1:lambda1, child2:lambda2, leak_node:lambda0}}

    Those can then be used with the samplers, e.g., adaptive, annealed, etc.
    """

    with open(filename) as inputfile:
        data = json.load(inputfile)

        graph = {}
        prior = {}
        lambdas = {}

        for key in data:
            # root node
            d = data[key]
            if len(d["parents"]) == 0:
                graph[key] = {None}
                prior[key] = d["cpt"]
            else:
                graph[key] = {p for p in d["parents"]}
                t = graph[key]
                c = d["cpt"]
                lambdas[key] = {node: c[i] for i, node in enumerate(t)}
                lambdas[key]["leak_node"] = c[len(t)]

    return graph, prior, lambdas


if __name__ == '__main__':
    some_dict = {
        "B-lorem-ipsum-seven-seven-seven-five": 0,
        "1Acdf": 1,
        "ublaham": 1
    }
    result = dict_to_string(some_dict)
    print(result)
    result = string_to_dict(result)
    print(result)

    child = "A"
    parents = ["B", "C"]
    print(init_cpt_table(child, parents))
