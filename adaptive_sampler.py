"""

Class for defining adaptive importance samplers for Bayesian networks.

Note that this currently only works for nodes with two states, although
some work has been done to generalize it to networks with more dimension.

This is a (loose) implementation of AIS-BN by Cheng, Druzdzel.
See: https://arxiv.org/abs/1106.0253

"""

import scipy as sc
from copy import deepcopy

from update_proposals import update_proposal_cpt, update_proposal_lambdas


class adaptive_sampler(object):
    def __init__(self, net, rep="CPT"):
        """
        Pass initializing data to importance sampler.
        """
        self.graph = net.graph
        self.nodes = net.nodes
        self.num_of_variables = len(self.nodes)

        self.net = net

        self.rep = rep
        self.proposal = deepcopy(net)

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

        # parents of the evidence nodes
        self.evidence_parents = []
        for e in self.evidence:
            for node in self.graph[e]:
                self.evidence_parents.append(node)

        if self.rep == "CPT":
            self._set_icpt()
        elif self.rep == "Noisy-OR":
            self._set_causality_strength()
        else:
            raise ValueError("Unknown option.")

    def _set_icpt(self):
        """
        Initialize the icp table for the
        proposal distribution.
        """
        # setting icpt for parents of evidence nodes
        # to uniform distribution (according to original paper on AIS-BN)
        for e in self.evidence:
            for parent in self.graph[e]:
                if parent is not None:
                    if isinstance(self.proposal.cpt[parent], list):
                        # prior node
                        n = len(self.proposal.cpt[parent])
                        self.proposal.cpt[parent] = [1.0 / n] * n
                    else:
                        for p in self.proposal.cpt[parent]:
                            n = len(self.proposal.cpt[parent][p])
                            self.proposal.cpt[parent][p] = [1.0 / n] * n

    def _set_causality_strength(self):
        """
        Initialize lambdas for the parents
        of evidence nodes.
        """

        for e in self.evidence:
            for parent in self.graph[e]:
                if self.proposal.is_root_node(parent):
                    n = len(self.proposal.prior_dict[parent])
                    self.proposal.prior_dict[parent] = [1.0/n]*n
                else:
                    # init lambdas
                    for p in self.proposal.lambdas[parent]:
                        if p == "leak_node":
                            self.proposal.lambdas[parent][p] = 0.5
                        else:
                            self.proposal.lambdas[parent][p] = 1.0

    def ais_bn(self,
               num_of_samples=100,
               prop_weight_fun=lambda x: float(x > 1),
               update_proposal_every=100):
        """
        Generates samples and weights through
        adaptive importance sampling.

        num_of_samples: int, number of samples to generate.
        prop_weight_fun: function that depends on the current
        number of samples generated. Will scale the current
        importance sampling weight.
        """

        samples = [None] * num_of_samples
        weights = sc.zeros([num_of_samples])

        self.update_prop = update_proposal_every

        sum_prop_weight = 0
        prop_update_num = 0
        last_update = 0

        # parameter for the learning of the proposal
        self.kmax = int(num_of_samples / self.update_prop)

        for i in range(num_of_samples):

            if i % self.update_prop == 0 and i > 0:
                # update proposal with the latest samples
                learn_samples = samples[last_update:i]
                if self.rep == "CPT":
                    self.proposal = update_proposal_cpt(
                        self.proposal, learn_samples, weights[last_update:i],
                        prop_update_num, self.graph, self.evidence_parents,
                        self.eta_rate)

                elif self.rep == "Noisy-OR":
                    self.proposal = update_proposal_lambdas(
                        self.proposal, learn_samples, weights[last_update:i],
                        prop_update_num, self.graph, self.evidence_parents,
                        self.eta_rate)

                prop_update_num += 1
                last_update = i

            prop_weight = prop_weight_fun(i)

            samples[i] = self.proposal_sample()
            weights[i] = self._weight(samples[i], scalar=prop_weight)
            sum_prop_weight += prop_weight

        return samples, weights, sum_prop_weight

    def proposal_sample(self):
        """
        Generates a sample from the current proposal distribution.
        """
        sample = self.proposal.sample(set_nodes=self.evidence)

        return sample

    def _weight(self, sample, scalar=1):
        """
        Computes the importance sampling weight,

        P(sample, evidence)/Q(sample)*scalar

        where P is the original joint_prob, Q is the proposal.
        """

        P = self.net.joint_prob(sample)
        Q = self.proposal.joint_prob(sample)
        ratio = sc.log(P) - sc.log(Q)
        result = ratio*scalar

        return result

    def eta_rate(self, k, a=0.3, b=0.14):
        """
        Parametric learning rate.
        """
        return a * (a / b)**(k / self.kmax)
