"""

Class for defining adaptive importance samplers for Bayesian networks.

Note that this currently only works for nodes with two states, although
some work has been done to generalize it to networks with more dimension.

This is a (loose) implementation of AIS-BN by Cheng, Druzdzel.
See: https://arxiv.org/abs/1106.0253

"""

from toposort import toposort_flatten as flatten
from misc import dict_to_string, weight_average, string_to_dict, char_fun

import scipy as sc


class adaptive_sampler(object):
    def __init__(self, graph, cpt):
        """
        Pass initializing data to importance sampler.

        Note: This should be indepedent of the evidence possibly.
        """
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

        # parents of the evidence nodes
        self.evidence_parents = []
        for e in self.evidence:
            for node in self.graph[e]:
                self.evidence_parents.append(node)

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
                        self.icpt[parent] = [1.0 / n] * n
                    else:
                        for p in icpt[parent]:
                            n = len(icpt[parent][p])
                            self.icpt[parent][p] = [1.0 / n] * n

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
                self._update_proposal(learn_samples, weights[last_update:i],
                                      prop_update_num)
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

        sample = self.evidence.copy()
        for node in self.nodes_minus_e:
            icpt = self.icpt[node]

            if self.graph[node] == {None}:
                p = sc.cumsum(icpt)
            else:
                # get relevant prob. dist
                try:
                    parents = {key: sample[key] for key in self.graph[node]}
                except KeyError:
                    # TODO bug to fix with sampling
                    print("This shouldn't happen!")
                    raise

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

        if abs(P - Q) < 1e-13:
            ratio = 1.0
        else:
            ratio = P / Q

        result = scalar*ratio

        return result

    def _eval_joint(self, sample, cpt):
        """
        sample: dict, {"A":0, "B":1}
        cpt: dict of dicts with the conditional probability
        distributions. See example.

        This function evaluates the joint over the
        sample in the "sample" dictionary.
        """

        nodes = [key for key in self.nodes if key in sample]

        result = 1.0

        for node in nodes:
            node_cpt = cpt[node]
            state = sample[node]

            if self.graph[node] == {None}:
                result *= node_cpt[state]
            else:
                parents = {key: sample[key] for key in self.graph[node]}
                key = dict_to_string(parents)
                result *= node_cpt[key][state]

        return result

    def _update_proposal(self, samples, weights, index):
        """
        Updates current proposal given the new data.

        Arguments
        =========
        samples: the current samples to use.
        index: the current index (used to pick the weight)

        """

        # Initialize weighted estimator
        wei_est = weight_average(samples, weights)

        # estimate CPT table from samples
        for node in self.evidence_parents:
            if node is None:
                continue
            elif self.graph[node] == {None}:

                def f(sample):
                    res1 = char_fun(sample, {node: 0})
                    return res1

                p, _ = wei_est.eval(f)

                self.icpt[node][0] += self.eta_rate(index) * (
                    p - self.icpt[node][0])
                self.icpt[node][1] += self.eta_rate(index) * (
                    1 - p - self.icpt[node][1])
            else:
                lcpt = self.icpt[node]
                for key in lcpt:

                    parent_dict = string_to_dict(key)

                    def f(sample):
                        res1 = char_fun(sample, {node: 0})
                        res2 = char_fun(sample, parent_dict)
                        return res1 * res2

                    def g(sample):
                        res2 = char_fun(sample, parent_dict)
                        return res2

                    p, _ = wei_est.eval(f)
                    q, _ = wei_est.eval(g)
                    ratio = p / q

                    self.icpt[node][key][0] += self.eta_rate(index) * (
                        ratio - self.icpt[node][key][0])
                    self.icpt[node][key][1] += self.eta_rate(index) * (
                        1 - ratio - self.icpt[node][key][1])

    def eta_rate(self, k, a=0.4, b=0.14):
        """
        Parametric learning rate.
        """

        return a * (a / b)**(k / self.kmax)
