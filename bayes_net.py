"""
Classes to represent Bayesian networks.
"""

from toposort import toposort_flatten as flatten
from misc import dict_to_string

from scipy import rand, prod, clip


class BayesNet(object):
    """
    Object to hold and evaluate probabilities
    for BayesNets described by cpts.
    """

    def __init__(self, graph, cpt, clipper=False):
        """
        graph: dictionary of form {child:{parent1, parent2},}.
        Expects {None} for the parents of root nodes.


        cpt: dictionary, holds conditional probabilities for the
        nodes of the network. In general, the form is expected to be:

        cpt = {child: prior_probs, child:{parent+value: probs,
        parent+value: probs}}

        Example:
        cpt = {"A":[0.2,0.8], B:{"A0":..., "A1":...}}


        The values next to the node name correspond to the values
        of the parent node.
        """

        self.nodes = flatten(graph)
        self.nodes = self.nodes[1::]
        self.graph = graph
        self.cpt = cpt
        self.clipper = clipper

    def joint_prob(self, node_values):
        """
        Calculate the joint probability of an instantiation
        of the graph.

        Input:
        node_values: dictionary, assumed to be of form {node1:value1,
        node2:value2, ...}

        """

        result = 1.0
        for node in node_values:
            if self.is_root_node(node):
                # root node
                result *= self.prior(node, node_values[node])
            else:
                result *= self.cond_prob(node, node_values[node], node_values)

        if self.clipper:
            result = clip(result, 1e-10, 1-1e-10)

        return result

    def cond_prob(self, child, state, all_vals):
        """
        Evaluates the conditional probability
        P(child = state  | all_vals)
        by looking up the values from the Icpt table.
        """
        parents = {key: int(all_vals[key]) for key in self.graph[child]}
        key = dict_to_string(parents)
        result = self.cpt[child][key][state]
        return result

    def prior(self, node, value):
        """
        Returns the prior of a root node.
        """

        result = None
        if self.is_root_node(node):
            result = self.cpt[node][value]

        return result

    def sample(self, set_nodes={}):
        """
        Generate single sample from BayesNet with
        binary variables.
        """

        # sample all but the already set nodes
        nodes = (n for n in self.nodes if n not in set_nodes)

        sample = set_nodes.copy()

        for node in nodes:
            if self.is_root_node(node):
                p = self.prior(node, True)
            else:
                p = self.cond_prob(node, True, sample)

            sample[node] = int(rand() < p)
        return sample

    def msample(self, num_of_samples=100):
        """
        Generate multiple samples.
        """

        samples = [None] * num_of_samples
        for i in range(num_of_samples):
            samples[i] = self.sample()
        return samples

    def is_root_node(self, node):
        result = self.graph[node] == {None}
        return result


class BNNoisyORLeaky(BayesNet):
    """

    """

    def __init__(self, graph, lambdas, prior, clipper=False):
        """
        graph: dictionary of form {child:{parent1, parent2},}.
        Expects {None} for the parents of root nodes.


        lambdas: dictionary, contains the lambdas from each node.
        Format assumed to be {node:{leak_node:value, parent_1:value}}


        The values next to the node name correspond to the values
        of the parent node.
        """

        self.nodes = flatten(graph)
        self.nodes = self.nodes[1::]
        self.graph = graph
        self.lambdas = lambdas
        self.prior_dict = prior
        self.clipper = clipper

    def prior(self, node, node_value):
        if node_value is True:
            result = self.prior_dict[node][1]
        else:
            result = self.prior_dict[node][0]

        return result

    def cond_prob(self, child, value, all_vals):
        """

        Evaluates the conditional probability

        P(child=value|all_vals)

        for a noisy-or model.

        Arguments:
        child: name of child nodes
        value: boole, value of child node
        parents: dict, {Parent1:boole1, Parent2:boole2, ...}

        """
        rel_lambdas = self.lambdas[child]

        result = 1
        n = len(rel_lambdas)
        for key in rel_lambdas:
            if key != "leak_node":
                if all_vals[key] == int(True):
                    result *= rel_lambdas[key]

        # # multiply by leak node
        result *= rel_lambdas["leak_node"]

        if value == int(True):
            result = 1 - result

        return result
