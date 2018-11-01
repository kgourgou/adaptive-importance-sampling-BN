"""
Classes to represent Bayesian networks.
"""
from toposort import toposort_flatten as flatten


"""
NOTE We can use this to represent a bayesian network!
"""

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
            if key != "leak_node": # TODO need to make sure this is parsed correctly by the cpt
                if all_vals[key] == int(True):
                    result *= rel_lambdas[key]

        # # multiply by leak node
        result *= rel_lambdas["leak_node"]

        if value == int(True):
            result = 1 - result

        return result
