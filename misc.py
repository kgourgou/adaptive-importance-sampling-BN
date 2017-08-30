"""
misc. functions
"""

from scipy import mean, var
import json

import scipy as sc


def dict_to_string(some_dict):
    result = ""
    sorted_keys = sorted(some_dict, key=lambda key: key)
    new_dict = {key: some_dict[key] for key in sorted_keys}

    for key in new_dict:
        result += str(key) + str(new_dict[key])

    return result


def string_to_dict(s):
    """
    Partial-inverse of "dict_to_string", will only work
    if the state is described by a single digit.
    """
    sp = [a for a in s]
    n = len(sp)

    if n % 2 != 0:
        raise ValueError("string has an odd number of characters.")

    new_dict = {sp[i]: float(sp[i + 1]) for i in range(0, n - 1, 2)}
    return new_dict


def char_fun(A, b):
    """

    Returns True if dictionary b is a subset
    of dictionary A and False.
    """

    result = b.items() <= A.items()

    return result


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
    some_dict = {"B": 0, "A": 1}
    print(dict_to_string(some_dict))

    filename = "data/approximate_network_all_mixed_any_age.json"
    graph, prior, lambdas = parse_node_file(filename)
