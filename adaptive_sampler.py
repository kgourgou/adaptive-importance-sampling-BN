"""

Class for importance sampler.

"""
from toposort import toposort_flatten as flatten
from misc import dict_to_string

import scipy as sc


class adaptive_sampler(object):
    def __init__(self,
                 graph,
                 cpt,
                 importance_weight_fun,
                 update_proposal_every=5):
        """
        Pass initializing data to importance sampler.

        Note: This should be indepedent of the evidence possibly.
        """
        self.eta_weight = importance_weight_fun
        self.update_prop = update_proposal_every
        self.graph = graph

        self.nodes = flatten(graph)

        # Remove None's
        self.nodes = self.nodes[1::]

        self.cpt = cpt
        self.graph = graph

        self.num_of_variables = len(self.nodes)

    def set_evidence(self, evidence):
        """
        Sets the evidence for the run
        and initializes the ICPT tables.

        """
        self.evidence = evidence

        # we do not have to sample evidence nodes
        self.nodes_minus_e = [
            node for node in self.graph if node not in evidence
        ]

        self._set_icpt()

    def _set_icpt(self):
        """
        Initialize the icp table for the
        proposal distribution.
        """

        self.icpt = self.cpt.copy()
        icpt = self.icpt

        # setting icpt for parents of evidence nodes
        # to uniform distribution (according to original paper on AIS-BN)
        for e in self.evidence:
            for parent in self.graph[e]:
                if parent is not None:
                    if isinstance(icpt[parent], list):
                        # prior node
                        n = len(icpt[parent])
                        icpt[parent] = [1.0/n]*n
                    else:
                        for p in icpt[parent]:
                            n = len(icpt[parent][p])
                            icpt[parent][p] = [1.0/n]*n

    def ais_bn(self, num_of_samples=100):
        """
        Generates samples and weights through
        adaptive importance sampling.
        """

        samples = [None]*num_of_samples
        weights = sc.zeros([num_of_samples])

        prop_weight = 1
        sum_prop_weight = 0

        for i in range(num_of_samples):

            if i % self.update_prop == 0:
                # update proposal
                self._update_proposal(i)

            samples[i] = self.proposal_sample()
            weights[i] = self._weight(samples[i], scalar=prop_weight)
            sum_prop_weight += prop_weight

        return samples, weights, sum_prop_weight

    def proposal_sample(self):
        """
        Generates a sample from the current proposal distribution.
        """

        sample = self.evidence.copy()
        for node in self.nodes_minus_e:
            icpt = self.icpt[node]

            if self.graph[node] == {None}:
                p = sc.cumsum(icpt)
            else:
                # get relevant prob. dist
                parents = {key: sample[key] for key in self.graph[node]}
                key = dict_to_string(parents)
                p = sc.cumsum(icpt[key])

            u = sc.rand()
            # sample state for node
            # states are assumed to be enumerated 0, ..., n
            for i, prob in enumerate(p):
                if u < prob:
                    sample[node] = i
                    break

        return sample

    def _weight(self, sample, scalar=1):
        """
        Computes the importance sampling weight,

        P(sample, evidence)/Q(sample)*scalar

        where P is the original joint, Q is the proposal.
        """

        P = self._eval_joint(sample, self.cpt)
        Q = self._eval_joint(sample, self.icpt)

        if abs(P-Q) < 1e-10:
            ratio = 1.0
        else:
            ratio = P/Q

        return scalar * ratio

    def _eval_joint(self, states, cpt):
        """
        states: dict, {"A":0, "B":1}
        cpt: dict of dicts with the conditional probability
        distributions. See example.

        This function evaluates the joint over the
        states in the "states" dictionary.
        """

        nodes = [key for key in self.nodes if key in states]

        evi = self.evidence.copy()

        result = 1.0

        for node in nodes:
            node_cpt = cpt[node]
            state = states[node]

            if self.graph[node] == {None}:
                result *= node_cpt[state]
            else:
                parents = {key: states[key] for key in self.graph[node]}
                key = dict_to_string(parents)
                result *= node_cpt[key][state]

        return result

    def _update_proposal(self, index):
        """
        Updates current proposal given the new data.
        """

        pass
